#!/bin/bash

# ===================================================================================
# --- Single-Node, Multi-GPU Distributed Training Launcher ---
# ===================================================================================

# --- Configuration ---
GPUS_PER_NODE=8
MASTER_PORT=29500 # A random high port
MASTER_ADDR="localhost" # Always localhost for single-node

# --- Environment Setup ---
cd /home/ubuntu/ceramic-fine-grained-moe
# source venv/bin/activate # Uncomment if you fix your venv

export PYTHONPATH="/home/ubuntu/ceramic-fine-grained-moe:$PYTHONPATH"
export HF_DATASETS_CACHE=/ephemeral/huggingface_cache
export HF_HOME=/ephemeral/huggingface_cache
export TRANSFORMERS_CACHE=/ephemeral/huggingface_cache
# export HF_DATASETS_OFFLINE=1
export HF_DATASETS_DISABLE_MULTIPROCESSING=1

# Network interface (still good practice for multi-GPU communication)
INTERFACE=$(ip route get 8.8.8.8 | grep -oP 'dev \K[^ ]+' | head -1)
export NCCL_SOCKET_IFNAME=$INTERFACE
export GLOO_SOCKET_IFNAME=$INTERFACE
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

echo "================================================"
echo "--- Starting Single-Node Distributed Launch ---"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Interface: $INTERFACE"
echo "GPUs: $GPUS_PER_NODE"
echo "================================================"

# Use the legacy launcher for single-node, multi-GPU training
python3 -m torch.distributed.launch \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    research/conditional/train/cc_train.py \
        --use_ngpt \
        --alpha_m 0.1 \
        --alpha_a 0.05 \
        --model_type gpt \
        --n_blocks 12 \
        --dmodel 768 \
        --dff 3072 \
        --effective_dff_x 4 \
        --n_att_heads 12 \
        --no_positional_embedding \
        --scheduler cosine \
        --learning_rate 5e-4 \
        --init_type truncated_normal \
        --init_scale 0.0361 \
        --dataset_type c4 \
        --batch_size 2048 \
        --cutoff 256 \
        --logger_types wandb \
        --name MoE_nGPT_MoE_16B_Tokens_lr5e-4_interleave_gate_scale \
        --n_gpus 8 \
        --n_nodes 1 \
        --n_steps 30500 \
        --lr_warmup_steps 0 \
        --final_lr_step 30500 \
        --final_lr_fraction 0.1 \
        --grad_clip 0.5 \
        --weight_decay 0.0 \
        --ff_mode expert_choice \
        --softmax_over experts \
        --group_granular_moe_by_batch \
        --use_torch_bmm \
        --granular_moe_one_hot_impl \
        --expansion_rate 16 \
        --granularity 4 \
        --every_other_layer \
        --standard_ff_first \
        --fsdp_enabled \
        --mixed_precision \
        --mixed_precision_dtype bfloat16 \
        --flash_attention \
        --fsdp_modules_to_wrap 'NgptBlock,EmbeddingLayer,PredictionHead' \
        --activation_checkpointing_modules 'NgptBlock,EmbeddingLayer,PredictionHead' \
        --wandb_entity ceramicai \
        --wandb_project fine-grained-moe \
        --tags moe_ngpt reproduction 85M 16B_tokens E16 G4 single_node \
        --save_weights_path /ephemeral/moeNgptCkpts/ngpt_moe \
        --save_weights_interval 5000 \
        --num_workers 2 \
        --gradient_accumulation_steps 1 \
        --logging_interval_loss 1000 \
        --logging_interval_heavy 1000 \
        --eval_interval 1000 \
        --n_eval_batches 200 \
        --decoding_interval 0 \
        --project_name fine-grained-moe