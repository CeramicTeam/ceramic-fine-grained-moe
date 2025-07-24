#!/bin/bash

# This script can be run in two modes:
# 1. Master Mode: Run on the master node to launch training on all nodes.
#    Usage: bash launch_distributed.sh --master_addr <MASTER_IP>
#
# 2. Worker Mode: This mode is triggered via SSH by the master. You don't run this manually.
WORKER_NODES=("worker-ip-1" "worker-ip-2" "worker-ip-3")
SSH_USER="lucas"
GPUS_PER_NODE=8
MASTER_ADDR=""
MASTER_PORT=29500
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

if [[ "$NODE_RANK" -eq -1 ]]; then
    echo "--- MASTER MODE: Launching distributed training ---"
    if [[ -z "$MASTER_ADDR" ]]; then
        echo "Error: Master IP address must be provided using --master_addr"
        exit 1
    fi
    NNODES=$(( ${#WORKER_NODES[@]} + 1 ))
    NGPUS=$(( NNODES * GPUS_PER_NODE ))
    echo "Detected $NNODES total nodes and $NGPUS total GPUs."
    WORKER_RANK=1
    for WORKER_IP in "${WORKER_NODES[@]}"; do
        echo "Launching on worker node $WORKER_RANK ($WORKER_IP)..."
        ssh -n "$SSH_USER@$WORKER_IP" "bash $(realpath $0) --master_addr $MASTER_ADDR --node_rank $WORKER_RANK --nnodes $NNODES" &
        ((WORKER_RANK++))
    done
    NODE_RANK=0    
    echo "Launching on master node 0 ($MASTER_ADDR)..."
else
    echo "--- WORKER MODE (Rank $NODE_RANK): Starting training process ---"
    NGPUS=$(( NNODES * GPUS_PER_NODE ))
fi

cd /home/lucas/ceramic-fine-grained-moe
source venv/bin/activate
export PYTHONPATH="/home/lucas/ceramic-fine-grained-moe:$PYTHONPATH"
export HF_DATASETS_CACHE=/ephemeral/datasets/c4
export HF_HOME=/ephemeral/datasets/c4
export TRANSFORMERS_CACHE=/ephemeral/datasets/c4

echo "Starting torchrun on node $NODE_RANK of $NNODES (Total GPUs: $NGPUS)..."
echo "Master Address: $MASTER_ADDR:$MASTER_PORT"

torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    research/conditional/train/cc_train.py \
        --use_ngpt \
        --model_type gpt \
        --n_blocks 12 \
        --dmodel 768 \
        --dff 3072 \
        --effective_dff_x 4 \
        --n_att_heads 12 \
        --scheduler cosine \
        --learning_rate 0.0002 \
        --init_type truncated_normal \
        --init_scale 0.1 \
        --dataset_type c4 \
        --batch_size 2048 \
        --cutoff 256 \
        --logger_types wandb \
        --name moe_85M_33B_E16_G4_repro \
        --n_gpus "$NGPUS" \
        --n_nodes "$NNODES" \
        --n_steps 63000 \
        --lr_warmup_steps 0 \
        --final_lr_step 63000 \
        --final_lr_fraction 0.1 \
        --grad_clip 0.5 \
        --weight_decay 0.0 \
        --ff_mode expert_choice \
        --softmax_over experts \
        --layer_norm_in_expert_choice \
        --group_granular_moe_by_batch \
        --use_torch_bmm \
        --granular_moe_one_hot_impl \
        --expansion_rate 16 \
        --granularity 4 \
        --fsdp_enabled \
        --mixed_precision \
        --mixed_precision_dtype bfloat16 \
        --flash_attention \
        --fsdp_modules_to_wrap 'TransformerBlock,EmbeddingLayer,PredictionHead' \
        --activation_checkpointing_modules 'TransformerBlock,EmbeddingLayer,PredictionHead' \
        --wandb_entity ceramicai \
        --wandb_project fine-grained-moe \
        --tags moe_baseline reproduction 85M 33B_tokens E16 G4 \
        --save_weights_path model_checkpoints \
        --save_weights_interval 5000 \
        --num_workers 4 \
        --gradient_accumulation_steps 1 \
        --logging_interval_loss 1000 \
        --logging_interval_heavy 5000 \
        --eval_interval 1000 \
        --n_eval_batches 200 \
        --decoding_interval 0 \
        --project_name fine-grained-moe
