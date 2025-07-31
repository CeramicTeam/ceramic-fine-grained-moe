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
from research.conditional.utils.model_utils import (
    get_ff_layer,
    get_attention_layer,
)


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
        print("Using nGPT model configuration...")
        block_modules = {
            "attention": get_attention_layer(args),
            "feedforward": get_ff_layer(args),
        }
        Define the nGPT block creator
        block_creator_fun = lambda: llm.NgptBlock(
            dmodel=dm,
            attention_layer=block_modules["attention"](),
            ff_layer=block_modules["feedforward"](),
            args=args,
        )
        # We pass the creator function itself to the tower.
        block_modules_override = block_creator_fun 
        # Residual function is not used when we create the block directly.
        residual_fn_override = None

    else:
        # If not using nGPT, use the block_modules and residual_fn passed from cc_train.py
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
