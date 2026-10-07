#!/bin/bash
#SBATCH -A amowli_lab
#SBATCH -p free-gpu32
#SBATCH --qos low
#SBATCH --job-name=train-nucleus2-fm
#SBATCH -o out.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=12:59:00

export WANDB_API_KEY=$(cat config/wandb_api_key.txt)

source ~/envs/sciml/bin/activate
module load cuda/13.0.1
export PYTHONPATH=/data/homezvol3/srachaba/Projects/NUCLEUS/src:${PYTHONPATH:-}

PYTHONNOUSERSITE=1 python scripts/train.py \
    model_cfg=flow_matching/nucleus2_moe_fm_small \
    data_cfg=poolboiling64_gravity \
    data_dir=/share/crsp/lab/amowli/share/BubbleML_2/ \
    pydataset=in_mem_forecast \
    normalizer_cfg=standard \
    batch_size=8 \
    accumulate_grad_batches=2 \
    optim_cfg.params.lr=2e-3 \
    optim_cfg.params.weight_decay=1e-3 \
    scheduler_cfg=cosine_warmup \
    max_steps=50000 \
    scheduler_cfg.params.warmup=2000 \
    log_dir=~/nucleus2_fm_logs
