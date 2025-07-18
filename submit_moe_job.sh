#!/bin/bash
#SBATCH --job-name=moe_85M_33B_E16_G4_repro
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Load necessary modules
module load cuda/11.7

# Create logs directory
mkdir -p logs

# Set environment variables
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=$((29500 + RANDOM % 1000))
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

# Use shared file system for HuggingFace cache
export HF_DATASETS_CACHE=/scratch/shared/datasets/c4
export HF_HOME=/scratch/shared/datasets/c4
export TRANSFORMERS_CACHE=/scratch/shared/datasets/c4

echo "Using shared dataset cache at: $HF_DATASETS_CACHE"

# Run with explicit Python environment
srun bash -c "
    cd /home/lucas/ceramic-fine-grained-moe && \
    source venv/bin/activate && \
    export PYTHONPATH=/home/lucas/ceramic-fine-grained-moe:\$PYTHONPATH && \
    export HF_DATASETS_CACHE=/scratch/shared/datasets/c4 && \
    export HF_HOME=/scratch/shared/datasets/c4 && \
    export TRANSFORMERS_CACHE=/scratch/shared/datasets/c4 && \
    export RANK=\$SLURM_PROCID && \
    export LOCAL_RANK=\$SLURM_LOCALID && \
    python3 research/conditional/train/cc_train.py \
        --model_type gpt \
        --n_blocks 12 \
        --dmodel 768 \
        --dff 3072 \
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
        --n_gpus 32 \
        --n_nodes 4 \
        --n_steps 63000 \
        --lr_warmup_steps 630 \
        --final_lr_step 63000 \
        --final_lr_fraction 0.1 \
        --grad_clip 0.5 \
        --weight_decay 0.1 \
        --ff_mode expert_choice \
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
        --eval_interval 5000 \
        --n_eval_batches 200 \
        --decoding_interval 0 \
        --project_name fine-grained-moe
"