from collections import OrderedDict
from typing import Literal, Callable, Optional
from functools import partial

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from lizrd.core import misc
from lizrd.core.misc import default, Aggregate
from lizrd.core.initialization import get_init_weight, ValidInitType
from lizrd.core.misc import Linear, LoggingLayer


def decode_bias_string(bias):
    assert bias in ["both", "first", "second", "none"]
    if bias == "both":
        bias_first = bias_second = True
    elif bias == "first":
        bias_first = True
        bias_second = False
    elif bias == "second":
        bias_first = False
        bias_second = True
    else:
        bias_first = bias_second = False
    return bias_first, bias_second


class SwiGLUFeedForward(LoggingLayer):
    def __init__(
        self,
        dmodel,
        dff,
        init_type: ValidInitType,
        init_scale: float,
    ):
        super().__init__()
        self.w1_gate = Linear(
            dmodel, dff * 2, init_type=init_type, init_scale=init_scale, bias=False
        )
        self.w2 = Linear(
            dff, dmodel, init_type=init_type, init_scale=init_scale, bias=False
        )

    def forward(self, x):
        pre_activation, gate = torch.chunk(self.w1_gate(x), 2, dim=-1)
        activation = nn.functional.silu(pre_activation)
        return self.w2(activation * gate)


def FeedForward(
    dmodel,
    dff,
    init_type: ValidInitType,
    init_scale: float,
    bias: Literal["both", "first", "second", "none"] = "both",
):
    bias_first, bias_second = decode_bias_string(bias)

    return nn.Sequential(
        OrderedDict(
            [
                (
                    "logging_ff_pre_relu",
                    Linear(
                        dmodel,
                        dff,
                        bias=bias_first,
                        init_type=init_type,
                        init_scale=init_scale,
                    ),
                ),
                ("relu", nn.ReLU()),
                (
                    "logging_ff_post_relu",
                    Linear(
                        dff,
                        dmodel,
                        bias=bias_second,
                        init_type=init_type,
                        init_scale=init_scale,
                    ),
                ),
            ]
        )
    )


class NgptFeedForward(LoggingLayer):
    """
    nGPT-compatible version of the FeedForward layer using SwiGLU activation.
    """

    def __init__(self, dmodel: int, dff: int, args: argparse.Namespace):
        super().__init__()
        self.dmodel = dmodel
        self.doutput = dmodel
        self.args = args

        self.w1_gate = Linear(
            dmodel, 
            dff * 2,
            bias=False, 
            init_type=args.init_type, 
            init_scale=args.init_scale
        )
        self.w2 = Linear(
            dff, 
            dmodel, 
            bias=False, 
            init_type=args.init_type, 
            init_scale=args.init_scale
        )

        # Initialize s_uv as a learnable parameter
        s_uv_init_value = self.args.s_u_init  # Should be 1.0
        s_uv_init_scaling = 1.0  # Paper uses 1.0
        
        self.s_uv = nn.Parameter(
            torch.full((2 * dff,), s_uv_init_scaling, dtype=torch.float32)
        )
        self.s_uv_init_value = s_uv_init_value
        self.s_uv_init_scaling = s_uv_init_scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.update_cache_for_logging("input_norm", x.norm(p=2, dim=-1).mean())
        uv = self.w1_gate(x)
        
        # 1. Calculate the effective learnable scaler
        s_uv_effective = self.s_uv * (self.s_uv_init_value / self.s_uv_init_scaling)
        self.update_cache_for_logging("s_uv_mean", s_uv_effective.mean())
        
        # 2. Apply the learnable scaler first
        uv_learnable_scaled = s_uv_effective * uv

        # 3. Apply the fixed sqrt(dmodel) scaling separately
        # This aligns with the paper's appendix and reference code's intent
        uv_final_scaled = uv_learnable_scaled * math.sqrt(self.dmodel)
        
        # 4. Now split into u and v
        u, v = torch.chunk(uv_final_scaled, 2, dim=-1)

        self.update_cache_for_logging("v_mean", v.mean())
        self.update_cache_for_logging("v_std", v.std())
        
        silu_activated = F.silu(v)
        activation = u * silu_activated
        
        self.update_cache_for_logging("v_vector_norm", v.norm(p=2, dim=-1).mean())
        self.update_cache_for_logging("silu_activation_mean", silu_activated.mean())
        
        output = self.w2(activation)
        self.update_cache_for_logging("output_norm", output.norm(p=2, dim=-1).mean())
        return output

    def log_heavy(self):
        logs = {}
        for key, value in self.logging_cache.items():
            logs[key] = value
        return logs


class EveryOtherLayer:
    def __init__(
        self, layer1_fn: Callable[[], nn.Module], layer2_fn: Callable[[], nn.Module]
    ):
        """
        This class is used to alternate between two layers.
        It is useful for Mixture of Experts,
        where every other layer is a regular linear layer.
        """
        self.layer1_fn = layer1_fn
        self.layer2_fn = layer2_fn
        self.counter = 0

    def __call__(self):
        if self.counter % 2 == 0:
            layer = self.layer1_fn()
        else:
            layer = self.layer2_fn()
        self.counter += 1
        return layer


class Residual(LoggingLayer):
    def __init__(self, layer):
        super(Residual, self).__init__()
        self.layer = layer

    def forward(self, x):
        out = self.layer(x)
        self.update_cache_for_logging("update", out)
        self.update_cache_for_logging("residual_stream", x)
        return out + x

    def log_heavy(self):
        updates = self.logging_cache["update"]
        residual_stream = self.logging_cache["residual_stream"]

        update_norms = torch.norm(updates, dim=-1)
        residual_norms = torch.norm(residual_stream, dim=-1)

        update_norms_mean = torch.mean(update_norms)
        update_norms_std = torch.std(update_norms)
        residual_norms_mean = torch.mean(residual_norms)
        residual_norms_std = torch.std(residual_norms)

        update_to_residual_ratio = update_norms / residual_norms
        update_to_residual_ratio_mean = torch.mean(update_to_residual_ratio)
        update_to_residual_ratio_std = torch.std(update_to_residual_ratio)

        return {
            "update_norms/mean": update_norms_mean,
            "update_norms/std": update_norms_std,
            "residual_norms/mean": residual_norms_mean,
            "residual_norms/std": residual_norms_std,
            "update_to_residual_ratio/mean": update_to_residual_ratio_mean,
            "update_to_residual_ratio/std": update_to_residual_ratio_std,
        }


class Parallel(nn.Module):
    def __init__(self, *layers):
        super(Parallel, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        return sum(layer(x) for layer in self.layers)


class SplitLastAxis(nn.Module):
    def __init__(self, a, b):
        super(SplitLastAxis, self).__init__()
        self.a = a
        self.b = b

    def forward(self, x):
        a, b = self.a, self.b
        assert x.shape[-1] == a * b
        result = x.view(x.shape[:-1] + (a, b))
        assert result.shape[-2:] == (a, b)
        # print("wtf", x.shape, result.shape)
        return result


class MergeLastAxis(nn.Module):
    def forward(self, x):
        result = x.reshape(x.shape[:-2] + (-1,))
        # print('wtf', x.shape, result.shape)
        return result


class Transpose(nn.Module):
    def forward(self, x):
        # return einops.rearrange(x, '... a b -> ... b a')
        return torch.transpose(x, -1, -2)


def LowRank(dinput, doutput, dlowrank):
    return nn.Sequential(
        Linear(dinput, dlowrank, bias=False),
        Linear(dlowrank, doutput),
    )


def attention_mechanism(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dhead: int,
    causal: bool,
    flash: bool,
    use_ngpt_scaling: bool = False,
):
    if flash:
        scale = math.sqrt(dhead) if use_ngpt_scaling else None
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=False
        ):
            output = F.scaled_dot_product_attention(
                query=query.contiguous(),
                key=key.contiguous(),
                value=value.contiguous(),
                attn_mask=None,
                is_causal=causal,
                scale=scale
            )
    else:
        # implementation without flash assumes other dim order
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        a = torch.einsum("... l h d, ... L h d -> ... h l L", query, key)
        if use_ngpt_scaling:
            a = a * math.sqrt(dhead)
        else:
            a = a * (1 / dhead**0.5)
        if causal:
            a.masked_fill_(
                torch.tril(torch.ones_like(a)) == 0, float("-inf")
            )  # mask out future tokens
        a = torch.softmax(a, dim=-1)
        output = torch.einsum("... h l L, ... L h d -> ... l h d", a, value)
        output = output.transpose(1, 2)

    return output


class AttentionMechanism(nn.Module):
    def __init__(self, use_flash_attention: bool, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_flash_attention = use_flash_attention

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dhead: int,
        causal: bool,
        use_ngpt_scaling: bool = False,
        *args,
        **kwargs,
    ):
        return attention_mechanism(
            query=query,
            key=key,
            value=value,
            dhead=dhead,
            causal=causal,
            flash=self.use_flash_attention,
            use_ngpt_scaling=use_ngpt_scaling
        )


class Attention(LoggingLayer):
    def __init__(
        self,
        dmodel,
        heads,
        causal,
        init_type: str,
        init_scale: float,
        dhead=None,
        flash=False,
    ):
        super(Attention, self).__init__()
        if dhead is None:
            assert dmodel % heads == 0
            dhead = dmodel // heads

        self.heads = heads
        self.dhead = dhead
        self.causal = causal
        self.flash = flash

        self.input_projection = Linear(
            dmodel,
            3 * heads * dhead,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.output_projection = Linear(
            heads * dhead,
            dmodel,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.attention_mechanism = AttentionMechanism(use_flash_attention=flash)

    def forward(self, x):
        projected = self.input_projection(x)

        batch, seq_len = x.shape[:-1]
        projected = projected.view(
            batch, seq_len, self.heads, 3 * self.dhead
        ).transpose(1, 2)
        q, k, v = torch.chunk(projected, chunks=3, dim=-1)

        attention_output = self.attention_mechanism(
            query=q, key=k, value=v, dhead=self.dhead, causal=self.causal
        )

        output = self.output_projection(attention_output.transpose(1, 2).flatten(-2))

        return output


class NgptAttention(Attention):
    """
    nGPT-compatible version of the Attention layer.
    """

    def __init__(
        self,
        dmodel,
        heads,
        causal,
        init_type,
        init_scale,
        args,
        dhead=None,
        flash=False,
    ):
        super().__init__(dmodel, heads, causal, init_type, init_scale, dhead, flash)
        self.args = args
        self.dmodel = dmodel

        # Initialize the learnable scaling factor for Q and K
        s_qk_init_value = self.args.s_qk_init
        s_qk_init_scaling = (
            self.args.s_qk_scale
            if self.args.s_qk_scale is not None
            else 1 / math.sqrt(self.dmodel)
        )

        self.s_qk = nn.Parameter(
            torch.full((self.heads, self.dhead), s_qk_init_scaling, dtype=torch.float32)
        )
        self.s_qk_init_value = s_qk_init_value
        self.s_qk_init_scaling = s_qk_init_scaling

    def forward(self, x):
        self.update_cache_for_logging("weight_norm_input_proj", self.input_projection.weight.norm())
        projected = self.input_projection(x)
        batch, seq_len = x.shape[:-1]
        projected = projected.view(
            batch, seq_len, self.heads, 3 * self.dhead
        ).transpose(1, 2)
        q, k, v = torch.chunk(projected, chunks=3, dim=-1)

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        s_qk_effective = self.s_qk * (self.s_qk_init_value / self.s_qk_init_scaling)
        self.update_cache_for_logging("s_qk_mean", s_qk_effective.mean())
        # Scale Q and K by the learnable scaling factor
        q = q * s_qk_effective.unsqueeze(0).unsqueeze(2)  # Add batch and seq_len dims for broadcasting
        k = k * s_qk_effective.unsqueeze(0).unsqueeze(2)

        # The nGPT paper states the scaling factor should be sqrt(d_head).
        # We perform the pre-scale inside of the `scaled_dot_product_attention` call.
        attention_output = self.attention_mechanism(
            query=q, key=k, value=v, dhead=self.dhead, causal=self.causal, use_ngpt_scaling=True
        )

        output = self.output_projection(attention_output.transpose(1, 2).flatten(-2))
        return output

    def log_heavy(self):
        logs = {}
        if "s_qk_mean" in self.logging_cache:
            logs["s_qk_mean"] = self.logging_cache["s_qk_mean"]
        if "weight_norm_input_proj" in self.logging_cache:
            logs["weight_norm_input_proj"] = self.logging_cache["weight_norm_input_proj"]
        return logs


class RoPE(nn.Module):
    # features are paired x_i, x_{i + d_head/2}
    def __init__(self, dhead, length):
        super().__init__()
        self.dhead = dhead
        self.length = length
        angle_exponents = torch.arange(0, dhead, 2) / dhead
        angles = torch.pow(1 / 10000, angle_exponents).reshape(1, -1)
        angle_per_token = angles * torch.arange(0, length).reshape(-1, 1)
        self.register_buffer("sin", torch.sin(angle_per_token).repeat(1, 2))
        self.register_buffer("cos", torch.cos(angle_per_token).repeat(1, 2))

    def forward(self, x):
        [y1, y2] = torch.chunk(x, chunks=2, dim=-1)
        x_rotated = torch.cat([-y2, y1], dim=-1)
        return x * self.cos + x_rotated * self.sin


class AttentionRoPE(LoggingLayer):
    def __init__(
        self,
        dmodel,
        heads,
        causal,
        length,
        init_type: str,
        init_scale: float,
        dhead=None,
        flash=False,
    ):
        super(AttentionRoPE, self).__init__()
        if dhead is None:
            assert dmodel % heads == 0
            dhead = dmodel // heads

        self.heads = heads
        self.dhead = dhead
        self.causal = causal
        self.flash = flash

        self.input_projection = Linear(
            dmodel,
            3 * heads * dhead,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.output_projection = Linear(
            heads * dhead,
            dmodel,
            bias=False,
            init_type=init_type,
            init_scale=init_scale,
        )
        self.rope = RoPE(dhead, length=length)
        self.attention_mechanism = AttentionMechanism(use_flash_attention=flash)

    def forward(self, x):
        projected = self.input_projection(x)

        batch, seq_len = x.shape[:-1]
        projected = projected.view(
            batch, seq_len, self.heads, 3 * self.dhead
        ).transpose(1, 2)
        q, k, v = torch.chunk(projected, chunks=3, dim=-1)
        q = self.rope(q)
        k = self.rope(k)

        target_dtype = v.dtype
        q = q.to(target_dtype)
        k = k.to(target_dtype)
        attention_output = self.attention_mechanism(
            query=q, key=k, value=v, dhead=self.dhead, causal=self.causal
        )

        output = self.output_projection(attention_output.transpose(1, 2).flatten(-2))

        return output

class NgptAttentionRoPE(AttentionRoPE):
    """
    nGPT-compatible version of the Attention layer that also includes RoPE.
    """
    def __init__(self, dmodel, heads, causal, length, init_type, init_scale, args, dhead=None, flash=False):
        super().__init__(dmodel, heads, causal, length, init_type, init_scale, dhead, flash)
        self.args = args
        self.dmodel = dmodel

        # Initialize the learnable scaling factor for Q and K from NgptAttention
        s_qk_init_value = self.args.s_qk_init
        s_qk_init_scaling = (
            self.args.s_qk_scale
            if self.args.s_qk_scale is not None
            else 1 / math.sqrt(self.dmodel)
        )

        self.s_qk = nn.Parameter(
            torch.full((self.heads, self.dhead), s_qk_init_scaling, dtype=torch.float32)
        )
        self.s_qk_init_value = s_qk_init_value
        self.s_qk_init_scaling = s_qk_init_scaling

    def forward(self, x):
        projected = self.input_projection(x)
        batch, seq_len = x.shape[:-1]
        projected = projected.view(
            batch, seq_len, self.heads, 3 * self.dhead
        ).transpose(1, 2)
        q, k, v = torch.chunk(projected, chunks=3, dim=-1)

        # Apply RoPE first, as in AttentionRoPE
        q = self.rope(q)
        k = self.rope(k)

        # Now, apply nGPT normalization and scaling
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        s_qk_effective = self.s_qk * (self.s_qk_init_value / self.s_qk_init_scaling)
        self.update_cache_for_logging("s_qk_mean", s_qk_effective.mean())
        q = q * s_qk_effective.unsqueeze(0).unsqueeze(2)
        k = k * s_qk_effective.unsqueeze(0).unsqueeze(2)

        target_dtype = v.dtype
        q = q.to(target_dtype)
        k = k.to(target_dtype)
        attention_output = self.attention_mechanism(
            query=q, key=k, value=v, dhead=self.dhead, causal=self.causal, use_ngpt_scaling=True
        )

        output = self.output_projection(attention_output.transpose(1, 2).flatten(-2))
        return output

    def log_heavy(self):
        logs = {}
        if "s_qk_mean" in self.logging_cache:
            logs["s_qk_mean"] = self.logging_cache["s_qk_mean"]
        return logs


class RMSNorm(nn.Module):
    def __init__(self, dmodel, eps=1e-5):
        super().__init__()
        self.eps = eps

        self.g = nn.Parameter(torch.ones(dmodel))
        self.b = nn.Parameter(torch.zeros(dmodel))

    def forward(self, x):
        norm = torch.mean(x**2, dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.g + self.b


class ReZero(nn.Module):
    def __init__(self, fn, init=0.0):
        super().__init__()
        self.rezero_g = nn.Parameter(torch.tensor(init))
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.rezero_g


def RezeroBlock(dmodel, layer, name):
    return Residual(ReZero(layer))


def PostNormBlock(dmodel, layer, name, norm_class=nn.LayerNorm):
    return nn.Sequential(
        OrderedDict(
            [
                (f"{name}", Residual(layer)),
                ("post_norm", norm_class(dmodel)),
            ]
        )
    )


def ParallelPreNormBlock(dmodel, layer, name, norm_class=nn.LayerNorm):
    assert isinstance(layer, Parallel)
    layer.layers = nn.ModuleList(
        *[
            torch.nn.Sequential(
                OrderedDict(
                    [
                        ("pre_norm", norm_class(dmodel)),
                        (f"{type(module)}", module),
                    ]
                )
            )
            for module in layer.layers
        ]
    )
    return Residual(layer)


def PreNormBlock(dmodel, layer, name, norm_class=nn.LayerNorm):
    return Residual(
        nn.Sequential(
            OrderedDict(
                [
                    ("pre_norm", norm_class(dmodel)),
                    (f"{name}", layer),
                ]
            )
        )
    )


class TransformerBlock(nn.Module):
    def __init__(self, dmodel, layers, residual_fn):
        super(TransformerBlock, self).__init__()

        residual_fn = default(residual_fn, partial(PreNormBlock, dmodel=dmodel))
        residual_layers = [
            (f"residual_{name}", residual_fn(layer=layer, name=name))
            for name, layer in layers
        ]
        self.block = nn.Sequential(OrderedDict(residual_layers))

    def forward(self, x):
        return self.block(x)


class NgptBlock(LoggingLayer):
    def __init__(
        self, dmodel: int, attention_layer: nn.Module, ff_layer: nn.Module, args
    ):
        super().__init__()
        self.dmodel = dmodel
        self.attention = attention_layer
        self.feedforward = ff_layer
        self.args = args

        # Eigen learning rates
        alpha_a_init_scaling = 1 / math.sqrt(self.dmodel)
        self.alpha_a = nn.Parameter(
            torch.full((dmodel,), alpha_a_init_scaling, dtype=torch.float32)
        )
        self.alpha_a_init_value = self.args.alpha_a
        self.alpha_a_init_scaling = alpha_a_init_scaling

        alpha_m_init_scaling = 1 / math.sqrt(self.dmodel)
        self.alpha_m = nn.Parameter(
            torch.full((dmodel,), alpha_m_init_scaling, dtype=torch.float32)
        )
        self.alpha_m_init_value = self.args.alpha_m
        self.alpha_m_init_scaling = alpha_m_init_scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.update_cache_for_logging("input_norm", x.norm(p=2, dim=-1).mean())

        # Attention block with nGPT residual update
        attn_output = self.attention(x)
        attn_output_norm = F.normalize(attn_output, p=2, dim=-1)
        alpha_a_effective = torch.abs(
            self.alpha_a * (self.alpha_a_init_value / self.alpha_a_init_scaling)
        )

        self.update_cache_for_logging("alpha_a_mean", alpha_a_effective.mean())

        x = F.normalize(x + alpha_a_effective * (attn_output_norm - x), p=2, dim=-1)

        # FeedForward/MoE block with nGPT residual update
        ff_output = self.feedforward(x)
        ff_output_norm = F.normalize(ff_output, p=2, dim=-1)
        alpha_m_effective = torch.abs(
            self.alpha_m * (self.alpha_m_init_value / self.alpha_m_init_scaling)
        )

        self.update_cache_for_logging("alpha_m_mean", alpha_m_effective.mean())
        x = F.normalize(x + alpha_m_effective * (ff_output_norm - x), p=2, dim=-1)
        self.update_cache_for_logging("output_norm", x.norm(p=2, dim=-1).mean())
        return x

    def log_heavy(self):
        """
        This method is called by the trainer to collect the metrics to be logged.
        """
        # The values were already averaged in the forward pass, so we just retrieve them.
        logs = {
            "input_norm": self.logging_cache["input_norm"],
            "alpha_a_mean": self.logging_cache["alpha_a_mean"],
            "alpha_m_mean": self.logging_cache["alpha_m_mean"],
            "output_norm": self.logging_cache["output_norm"],
        }
        
        # It's good practice to also include logs from child LoggingLayers
        # if they have their own log_heavy methods.
        if hasattr(self.attention, "log_heavy"):
            logs.update(self.attention.log_heavy())
        if hasattr(self.feedforward, "log_heavy"):
            logs.update(self.feedforward.log_heavy())
            
        return logs


class TransformerTower(nn.Module):
    def __init__(
        self,
        n_blocks,
        dmodel,
        layer_or_block_definition,
        device: torch.device = None,
        model_fragmentation: Optional[list[int]] = None,
        residual_fn: Optional[Callable] = None,
    ):
        super().__init__()
        if type(layer_or_block_definition) is dict:
            misc.check_layer_funs(*layer_or_block_definition.values())
        elif type(layer_or_block_definition) is list:
            for layer in layer_or_block_definition:
                misc.check_layer_funs(*layer)
            assert len(layer_or_block_definition) == n_blocks

        self.blocks = []
        self.model_fragmentation = (
            [] if model_fragmentation is None else model_fragmentation
        )
        self.device = device

        for i_block in range(n_blocks):
            _, current_device = self.get_current_device(i_block)

            # We use `residual_fn` as a signal to determine which block type to build.
            # If it's None, we know it's an nGPT run.
            if residual_fn is None:
                block = layer_or_block_definition()
            else:
                # This is the original logic for standard blocks
                layer_dict = layer_or_block_definition
                if isinstance(layer_or_block_definition, list):
                    layer_dict = layer_or_block_definition[i_block]
                
                layers_info = [
                    (name, layer_fun()) for name, layer_fun in layer_dict.items()
                ]

                for name, layer in layers_info:
                    layer.layer_type = name
                    layer.block_number = i_block
                
                block = TransformerBlock(
                    dmodel,
                    layers_info,
                    residual_fn,
                )

            if current_device != torch.device("cpu"):
                block = block.to(current_device)

            name_and_block = (f"block_{i_block}", block)
            self.blocks.append(name_and_block)
        
        self.blocks = nn.Sequential(OrderedDict(self.blocks))

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            should_transfer, current_device = self.get_current_device(i)
            if should_transfer:
                x = x.to(current_device)
            x = block(x)
        return x

    def get_current_device(self, block_num):
        if self.model_fragmentation is None or self.device == torch.device("cpu"):
            return False, self.device

        for i, split_num in enumerate(self.model_fragmentation):
            if split_num > block_num:
                return block_num in self.model_fragmentation, torch.device(f"cuda:{i}")

        return block_num in self.model_fragmentation, torch.device(
            f"cuda:{len(self.model_fragmentation)}"
        )


def TokenEmbedding(
    vocab_size,
    embedding_dim,
    init_type: ValidInitType,
    init_scale: float,
):
    weight = get_init_weight(
        shape=(vocab_size, embedding_dim),
        fan_in=1,  # fan_in=1 is also default in pytorch
        init_type=init_type,
        scale=init_scale,
    )
    return nn.Embedding(vocab_size, embedding_dim, _weight=weight)


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        max_length,
        embedding_dim,
        init_type: ValidInitType,
        init_scale: float,
    ):
        super(PositionalEmbedding, self).__init__()
        self.layer = nn.Embedding(max_length, embedding_dim)
        default_weight = self.layer.weight.data
        self.layer.weight.data = get_init_weight(
            shape=default_weight.shape,
            fan_in=1,
            init_type=init_type,
            scale=init_scale,
            dtype=default_weight.dtype,
        )
        # TODO(jaszczur): add initialization as positional encoding

    def forward(self, x):
        positions = torch.arange(0, x.shape[-1], device=x.device)
        positions = positions * torch.ones_like(x)
        embeddings = self.layer(positions)
        return embeddings


class EmbeddingLayer(Aggregate):
    def __init__(self, *layers):
        super(EmbeddingLayer, self).__init__((lambda x, y: x + y), *layers)


class PredictionHead(Linear):
    def __init__(self, embedding_dim, output_size, init_type, init_scale, use_ngpt=False, args=None):
        super(PredictionHead, self).__init__(
            embedding_dim, output_size, init_type=init_type, init_scale=init_scale
        )
        self.use_ngpt = use_ngpt
        if self.use_ngpt:
            print(f"Using nGPT scaling in PredictionHead with output size {output_size} and embedding dim {embedding_dim}")
            # For nGPT, create the learnable scaling factor for logits ('sz')
            base_scale = 1 / math.sqrt(embedding_dim)
            sz_init_value = args.s_z_init
            # Use provided scale or default to base_scale
            sz_init_scaling = (
                args.s_z_scale if args.s_z_scale is not None else base_scale
            )

            self.sz = nn.Parameter(
                torch.full((output_size,), sz_init_scaling, dtype=torch.float32)
            )
            # Store init values to calculate the effective scaler later
            self.sz_init_value = sz_init_value
            self.sz_init_scaling = sz_init_scaling
    def forward(self, x):
        # Call the parent Linear's forward method
        logits = super().forward(x)
        if self.use_ngpt:
            # Apply the learnable scaling factor for logits
            sz_effective = self.sz * (self.sz_init_value / self.sz_init_scaling)
            logits = logits * sz_effective
        return logits


class LLM(nn.Module):
    def __init__(
        self,
        embedding_layer,
        encoder_tower,
        head,
        args
    ):
        super(LLM, self).__init__()
        self.embedding_layer = embedding_layer
        self.encoder = encoder_tower
        self.head = head
        self.use_ngpt = args.use_ngpt

    def forward(self, *args, **kwargs):
        x = self.embedding_layer(*args, **kwargs)
        x = self.encoder(x)
        x = self.head(x)
        return x
