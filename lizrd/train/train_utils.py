from typing import Callable, Optional, Union, Type
import argparse
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)

from lizrd.core import llm
from lizrd.core.distributed import wrap_in_fsdp, wrap_in_ddp
from lizrd.train.checkpointing import make_checkpoint_wrapper_function
from lizrd.train.load_and_save_model import load_model_weights


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
        # 1. Define nGPT-specific layer creation functions
        attention_layer_fun = lambda: llm.NgptAttention(dm, args.n_att_heads, args)

        # Expert layer for MoE will now be the normalized version
        expert_layer_fun = lambda: llm.NgptFeedForward(dm, args.dff, args)

        # Determine the feed-forward layer based on ff_mode
        if args.ff_mode == "expert_choice":
            ff_layer_fun = lambda: ExpertChoiceFF(
                dm, args.dff, args.expansion_rate, args.granularity, expert_layer_fun
            )
        else:
            ff_layer_fun = expert_layer_fun

        # 2. Define the block creation function for the TransformerTower
        # This function will be called by TransformerTower to create each block.
        # It doesn't need block_modules because nGPTBlock is self-contained.
        block_creator_fun = lambda: llm.NgptBlock(
            dmodel=dm,
            attention_layer=attention_layer_fun(),
            ff_layer=ff_layer_fun(),
            args=args,
        )

        # 3. Override residual_fn because nGPT handles residuals internally
        residual_fn_override = None

    else:
        # If not using nGPT, use the block_modules and residual_fn passed from cc_train.py
        block_creator_fun = lambda: llm.TransformerBlock(
            dm, block_modules, residual_fn=residual_fn
        )
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
        block_creator_fun,
        device,
        model_fragmentation=model_fragmentation,
        residual_fn=residual_fn_override,
    )

    head = llm.PredictionHead(
        dm, vocab_size, init_type=init_type, init_scale=init_scale
    ).to(last_gpu)

    model = llm.LLM(
        embedding_layer,
        encoder_tower,
        head,
        dmodel=dm,
        vocab_size=vocab_size,
        use_ngpt=args.use_ngpt,
        args=args,
    )

    if checkpoint is not None:
        load_model_weights(model, checkpoint)

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

    if activation_checkpointing_modules is not None:
        check_fn = lambda x: isinstance(x, activation_checkpointing_modules)
        apply_activation_checkpointing(
            model,
            check_fn=check_fn,
            checkpoint_wrapper_fn=make_checkpoint_wrapper_function(),
        )

    return model
