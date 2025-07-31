#!/bin/bash

# ===================================================================================
# --- Legacy Distributed Training Launcher using torch.distributed.launch ---
# ===================================================================================

# --- Configuration ---
GPUS_PER_NODE=8
MASTER_PORT=29500

# --- Argument Parsing ---
MASTER_ADDR=""
NODE_RANK=-1
NNODES=-1

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --master_addr) MASTER_ADDR="$2"; shift ;;
        --node_rank) NODE_RANK="$2"; shift ;;
        --nnodes) NNODES="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# --- Validate Arguments ---
if [[ -z "$MASTER_ADDR" ]] || [[ "$NODE_RANK" -eq -1 ]] || [[ "$NNODES" -eq -1 ]]; then
    echo "ERROR: Missing required arguments."
    echo "Usage: $0 --master_addr <MASTER_IP> --node_rank <RANK> --nnodes <TOTAL_NODES>"
    exit 1
fi

# --- Environment Setup ---
cd /home/ubuntu/ceramic-fine-grained-moe
# source venv/bin/activate

export PYTHONPATH="/home/ubuntu/ceramic-fine-grained-moe:$PYTHONPATH"
export HF_DATASETS_CACHE=/ephemeral/huggingface_cache
export HF_HOME=/ephemeral/huggingface_cache
export TRANSFORMERS_CACHE=/ephemeral/huggingface_cache
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_DISABLE_MULTIPROCESSING=1

# Network interface
INTERFACE=$(ip route get 8.8.8.8 | grep -oP 'dev \K[^ ]+' | head -1)
export NCCL_SOCKET_IFNAME=$INTERFACE
export GLOO_SOCKET_IFNAME=$INTERFACE
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

echo "================================================"
echo "--- Starting Legacy Distributed Launch ---"
echo "Node Rank: $NODE_RANK of $NNODES"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Interface: $INTERFACE"
echo "================================================"

# Use the legacy launcher which is more stable
python3 -m torch.distributed.launch \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    research/conditional/train/cc_train.py \
        --use_ngpt \
        --model_type gpt \
        --n_blocks 12 \
        --dmodel 768 \
        --dff 3072 \
        --effective_dff_x 4 \
        --n_att_heads 12 \
        --scheduler cosine \
        --learning_rate 0.001 \
        --init_type truncated_normal \
        --init_scale 0.1 \
        --dataset_type c4 \
        --batch_size 2048 \
        --cutoff 256 \
        --logger_types wandb \
        --name moe_ngpt_85M_33B_E16_G4_repro \
        --n_gpus 16 \
        --n_nodes 2 \
        --n_steps 63000 \
        --lr_warmup_steps 0 \
        --final_lr_step 63000 \
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
        --fsdp_enabled \
        --mixed_precision \
        --mixed_precision_dtype bfloat16 \
        --flash_attention \
        --fsdp_modules_to_wrap 'NgptBlock,EmbeddingLayer,PredictionHead' \
        --activation_checkpointing_modules 'NgptBlock,EmbeddingLayer,PredictionHead' \
        --wandb_entity ceramicai \
        --wandb_project fine-grained-moe \
        --tags moe_ngpt reproduction 85M 33B_tokens E16 G4 \
        --save_weights_path /ephemeral/moeNgptCkpts \
        --save_weights_interval 5000 \
        --num_workers 2 \
        --gradient_accumulation_steps 1 \
        --logging_interval_loss 1000 \
        --logging_interval_heavy 1000 \
        --eval_interval 1000 \
        --n_eval_batches 200 \
        --decoding_interval 0 \
        --project_name fine-grained-moe