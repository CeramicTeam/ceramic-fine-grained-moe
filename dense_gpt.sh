#!/bin/bash
# ===================================================================================
# --- GPT Baseline | Single-Node, 8-GPU Distributed Training ---
# --- Architecture: RoPE, Interleaved MoE, LayerNorm in Experts ---
# ===================================================================================

# --- Configuration ---
GPUS_PER_NODE=8
MASTER_PORT=29500
MASTER_ADDR="localhost"

# --- Environment Setup ---
# Set to the root directory of your project
cd /home/ubuntu/ceramic-fine-grained-moe

export PYTHONPATH="/home/ubuntu/ceramic-fine-grained-moe:$PYTHONPATH"
export HF_DATASETS_CACHE=/ephemeral/huggingface_cache
export HF_HOME=/ephemeral/huggingface_cache
export TRANSFORMERS_CACHE=/ephemeral/huggingface_cache
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_DISABLE_MULTIPROCESSING=1

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

# --- Training ---
# Total tokens: 16B. Global Batch Size: 256*2048=524288. Steps: 16e9/524288 ≈ 30517
python3 -m torch.distributed.launch \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    research/conditional/train/cc_train.py \
        --model_type gpt \
        --n_blocks 12 \
        --dmodel 768 \
        --dff 3072 \
        --effective_dff_x 4 \
        --n_att_heads 12 \
        --attention_mode rope \
        --no_positional_embedding \
        --scheduler cosine \
        --learning_rate 5e-4 \
        --init_type truncated_normal \
        --init_scale 0.02 \
        --dataset_type c4 \
        --batch_size 2048 \
        --cutoff 256 \
        --logger_types wandb \
        --name Dense_GPT_85M_16B_Tokens \
        --n_gpus 8 \
        --n_nodes 1 \
        --n_steps 30500 \
        --lr_warmup_steps 305 \
        --final_lr_step 30500 \
        --final_lr_fraction 0.1 \
        --grad_clip 0.5 \
        --weight_decay 0.1 \
        --ff_mode vanilla \
        --fsdp_enabled \
        --mixed_precision \
        --mixed_precision_dtype bfloat16 \
        --flash_attention \
        --fsdp_modules_to_wrap 'TransformerBlock,EmbeddingLayer,PredictionHead' \
        --activation_checkpointing_modules 'TransformerBlock,EmbeddingLayer,PredictionHead' \
        --wandb_entity ceramicai \
        --wandb_project fine-grained-moe \
        --tags moe_baseline reproduction 85M 16B_tokens E16 G4 \
        --save_weights_path /ephemeral/moeNgptCkpts/dense_gpt \
        --save_weights_interval 5000 \
        --num_workers 2 \
        --gradient_accumulation_steps 1 \
        --logging_interval_loss 1000 \
        --logging_interval_heavy 1000 \
        --eval_interval 1000 \
        --n_eval_batches 200 \
        --decoding_interval 0 \
        --project_name fine-grained-moe