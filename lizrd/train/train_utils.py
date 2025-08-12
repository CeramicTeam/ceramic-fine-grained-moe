from typing import Callable, Optional, Union, Type
import argparse
import math
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.nn.functional as F

from lizrd.core import llm
from lizrd.core.distributed import wrap_in_fsdp, wrap_in_ddp
from lizrd.train.checkpointing import make_checkpoint_wrapper_function
from lizrd.train.load_and_save_model import load_model_weights
from research.conditional.moe_layers.expert_choice import ExpertChoiceFF
from research.conditional.moe_layers.moe_gating import ExpertGating
from research.conditional.utils.model_utils import (
    get_ff_layer,
    get_attention_layer,
)

def normalize_parameters(model: torch.nn.Module, already_in_fsdp_context: bool = False):
    """
    Recursively iterates through all modules and normalizes the weights
    of specific nGPT layers. This is robust to FSDP and other wrappers.
    
    Args:
        model: The model to normalize
        already_in_fsdp_context: If True, assumes we're already in an FSDP.summon_full_params context
    """
    is_fsdp = isinstance(model, FSDP) or any(isinstance(m, FSDP) for m in model.modules())
    
    if is_fsdp and not already_in_fsdp_context:
        # For FSDP models, we need to use summon_full_params context
        with FSDP.summon_full_params(model, writeback=True):
            _normalize_parameters_impl(model)
    else:
        # For non-FSDP models or when already in FSDP context, normalize directly
        _normalize_parameters_impl(model)


def normalize_parameters_for_fsdp_context(model: torch.nn.Module):
    """
    Wrapper for normalize_parameters that's safe to call from within FSDP context.
    This is meant to be used as the after_step_callback.
    """
    normalize_parameters(model, already_in_fsdp_context=True)


def _normalize_parameters_impl(model: torch.nn.Module):
    """
    All weight matrices are normalized row-wise (dim=1) for maximum stability. (except ExpertGating)
    """
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, llm.NgptAttentionRoPE):
                # Normalize both input and output projections by row
                module.input_projection.weight.data.copy_(
                    F.normalize(module.input_projection.weight.data, p=2, dim=1)
                )
                module.output_projection.weight.data.copy_(
                    F.normalize(module.output_projection.weight.data, p=2, dim=1)
                )
            
            elif isinstance(module, llm.NgptFeedForward):
                # Normalize both FFN matrices by row
                module.w1_gate.weight.data.copy_(
                    F.normalize(module.w1_gate.weight.data, p=2, dim=1)
                )
                module.w2.weight.data.copy_(
                    F.normalize(module.w2.weight.data, p=2, dim=1)
                )

            elif isinstance(module, ExpertGating):
                # The gating matrix has shape (d_model, n_experts).
                # To normalize the expert prototype vectors, we normalize the columns.
                # This is the one exception to the row-wise rule.
                if hasattr(module, 'gate') and module.gate is not None:
                    module.gate.data.copy_(
                        F.normalize(module.gate.data, p=2, dim=0)
                    )
        
        # This logic for embeddings and the prediction head is already correct (row-wise).
        if hasattr(model, 'embedding_layer'):
            if hasattr(model.embedding_layer, 'layers'):
                for layer in model.embedding_layer.layers:
                    if isinstance(layer, torch.nn.Embedding):
                        layer.weight.data.copy_(F.normalize(layer.weight.data, p=2, dim=1))
        
        if hasattr(model, 'head') and isinstance(model.head, llm.PredictionHead):
            model.head.weight.data.copy_(F.normalize(model.head.weight.data, p=2, dim=1))

def log_normalization_verification(model: torch.nn.Module, step_or_stage: str):
    """
    A comprehensive verification function that logs the norms of all key nGPT
    weight matrices to ensure they are correctly normalized.
    This function should be called within an FSDP.summon_full_params context if the model is sharded.
    """
    print(f"\n--- Running Normalization Verification at: {step_or_stage} ---")
    with torch.no_grad():
        actual_model = model._fsdp_wrapped_module if isinstance(model, FSDP) else model
        for i, block in enumerate(actual_model.encoder.blocks):
            actual_block = block._fsdp_wrapped_module if isinstance(block, FSDP) else block
            
            attn_layer = actual_block.attention
            if isinstance(attn_layer, llm.NgptAttentionRoPE):
                weight_in = attn_layer.input_projection.weight
                norm_in = weight_in.norm().item()
                expected_norm_in = math.sqrt(weight_in.shape[0])
                print(f"Block {i} | Attention In Norm : {norm_in:.4f} (Expected: ~{expected_norm_in:.4f})")
                
                weight_out = attn_layer.output_projection.weight
                norm_out = weight_out.norm().item()
                expected_norm_out = math.sqrt(weight_out.shape[1])
                print(f"Block {i} | Attention Out Norm: {norm_out:.4f} (Expected: ~{expected_norm_out:.4f})")

            ff_layer = actual_block.feedforward
            if isinstance(ff_layer, llm.NgptFeedForward):
                weight_w2 = ff_layer.w2.weight
                norm_w2 = weight_w2.norm().item()
                expected_norm_w2 = math.sqrt(weight_w2.shape[1])
                print(f"Block {i} | Dense FF Out Norm  : {norm_w2:.4f} (Expected: ~{expected_norm_w2:.4f})")
            
            elif isinstance(ff_layer, ExpertChoiceFF):
                gating_module = ff_layer.gating
                if isinstance(gating_module, FSDP):
                    actual_gating_module = gating_module._fsdp_wrapped_module
                else:
                    actual_gating_module = gating_module
                
                if isinstance(actual_gating_module, ExpertGating) and hasattr(actual_gating_module, 'gate') and actual_gating_module.gate is not None:
                    weight_gate = actual_gating_module.gate
                    norm_gate = weight_gate.norm().item()
                    expected_norm_gate = math.sqrt(weight_gate.shape[1])
                    print(f"Block {i} | MoE Gating Norm    : {norm_gate:.4f} (Expected: ~{expected_norm_gate:.4f})")

        if hasattr(actual_model, 'embedding_layer'):
            emb_layer_wrapper = actual_model.embedding_layer
            actual_emb_layer = emb_layer_wrapper._fsdp_wrapped_module if isinstance(emb_layer_wrapper, FSDP) else emb_layer_wrapper
            if hasattr(actual_emb_layer, 'layers'):
                token_emb = actual_emb_layer.layers[0]
                if isinstance(token_emb, torch.nn.Embedding):
                    norm = token_emb.weight.norm().item()
                    expected = math.sqrt(token_emb.weight.shape[0])
                    print(f"Embeddings | Token Emb Norm     : {norm:.4f} (Expected: ~{expected:.4f})")

        if hasattr(actual_model, 'head'):
            head_wrapper = actual_model.head
            actual_head = head_wrapper._fsdp_wrapped_module if isinstance(head_wrapper, FSDP) else head_wrapper
            if isinstance(actual_head, llm.PredictionHead):
                norm = actual_head.weight.norm().item()
                expected = math.sqrt(actual_head.weight.shape[0])
                print(f"Head       | Prediction Head Norm : {norm:.4f} (Expected: ~{expected:.4f})")
    print("--- End of Verification ---")

def get_model(
    args: argparse.Namespace,
    max_length: int,
    vocab_size: int,
    block_modules: dict[str, Callable[[], torch.nn.Module]],
    dm: int,
    n_blocks: int,
    device: torch.device,
    init_type,
    init_scale,
    ddp_enabled: bool,
    fsdp_enabled: bool,
    fsdp_param_precision: torch.dtype,
    fsdp_mixed_precision_ignore_classes: list[Type[torch.nn.Module]],
    fsdp_offload_params: bool,
    fsdp_min_num_params: int,
    fsdp_modules_to_wrap: Union[tuple[Type[torch.nn.Module]], None],
    activation_checkpointing_modules: Union[tuple[Type[torch.nn.Module]], None],
    is_logging_process: bool,
    local_rank=None,
    model_fragmentation: Optional[list[int]] = None,
    residual_fn: Callable[[], torch.nn.Module] = None,
    include_positional_embedding: bool = True,
    checkpoint: dict[str, torch.Tensor] = None,
):
    if args.use_ngpt:
        block_modules = {
            "attention": get_attention_layer(args),
            "feedforward": get_ff_layer(args),
        }

        block_creator_fun = lambda: llm.NgptBlock(
            dmodel=dm,
            attention_layer=block_modules["attention"](),
            ff_layer=block_modules["feedforward"](),
            args=args,
        )
        block_modules_override = block_creator_fun 
        residual_fn_override = None
    else:
        block_modules_override = block_modules
        residual_fn_override = residual_fn  

    if model_fragmentation is None or device == torch.device("cpu"):
        first_gpu = device
        last_gpu = device
    else:
        first_gpu = torch.device("cuda:0")
        last_gpu = torch.device(f"cuda:{len(model_fragmentation)}")

    embedding_components = [
        llm.TokenEmbedding(vocab_size, dm, init_type=init_type, init_scale=init_scale)
    ]

    if include_positional_embedding:
        embedding_components.append(
            llm.PositionalEmbedding(
                max_length, dm, init_type=init_type, init_scale=init_scale
            )
        )

    embedding_layer = llm.EmbeddingLayer(*embedding_components).to(first_gpu)

    # Python officially preserves dict order since 3.7, so we pass the layer dict
    encoder_tower = llm.TransformerTower(
        n_blocks,
        dm,
        layer_or_block_definition=block_modules_override,
        device=device,
        model_fragmentation=model_fragmentation,
        residual_fn=residual_fn_override,
    )

    head = llm.PredictionHead(
        dm,
        vocab_size,
        init_type=init_type,
        init_scale=init_scale,
        use_ngpt=args.use_ngpt,
        args=args
    ).to(last_gpu)

    model = llm.LLM(
        embedding_layer,
        encoder_tower,
        head,
        args=args
    )

    # Load checkpoint weights if provided (before FSDP wrapping)
    if checkpoint is not None:
        load_model_weights(model, checkpoint)

    # For nGPT, normalize BEFORE FSDP wrapping
    if args.use_ngpt:
        print("nGPT model created. Performing initial normalization before FSDP wrapping.")
        normalize_parameters(model)
        if is_logging_process:
            # We don't need the FSDP context here because the model is not yet wrapped
            log_normalization_verification(model, "Initial (pre-FSDP)")

    # Wrap model in DDP or FSDP if needed
    if ddp_enabled:
        model = wrap_in_ddp(module=model, local_rank=local_rank)
    elif fsdp_enabled:
        model = wrap_in_fsdp(
            module=model,
            local_rank=local_rank,
            param_precision=fsdp_param_precision,
            cast_inputs=True,
            mixed_precision_ignored_classes=fsdp_mixed_precision_ignore_classes,
            offload_params=fsdp_offload_params,
            print_model=True,
            min_num_params=fsdp_min_num_params,
            modules_to_wrap=fsdp_modules_to_wrap,
            is_logging_process=is_logging_process,
        )

    # Apply activation checkpointing if needed
    if activation_checkpointing_modules is not None:
        check_fn = lambda x: isinstance(x, activation_checkpointing_modules)
        apply_activation_checkpointing(
            model,
            check_fn=check_fn,
            checkpoint_wrapper_fn=make_checkpoint_wrapper_function(),
        )

    return model
