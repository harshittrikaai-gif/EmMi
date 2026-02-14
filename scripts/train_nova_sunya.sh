#!/bin/bash
#SBATCH --job-name=nova_sunya_1.2t
#SBATCH --nodes=4096              # 4096 nodes * 8 GPUs = 32,768 H100s
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --partition=h100_cluster
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# -----------------------------------------------------------------------------
# Emmit Nova Sunya 1.2T - Training/Fine-tuning Launch Script
# -----------------------------------------------------------------------------

# Environment setup
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * 8))

# Optimization flags
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0

# Default args (can be overridden by cluster env or CLI)
CONFIG=${CONFIG:-configs/nova_sunya_1.2t.yaml}
JOB_NAME=${JOB_NAME:-nova_sunya_1.2t}
PRETRAINED_PATH=${PRETRAINED_PATH:-""}

# Launch training with 3D parallelism
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    scripts/04_train_model.py \
    --config $CONFIG \
    --output_dir outputs/$JOB_NAME \
    --deepspeed configs/ds_config_zero3.json \
    --pretrained_path "$PRETRAINED_PATH" \
    --report_to wandb
