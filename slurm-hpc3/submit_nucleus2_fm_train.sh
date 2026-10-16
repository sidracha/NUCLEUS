#!/bin/bash
#SBATCH -A amowli_lab
#SBATCH -p free-gpu32
#SBATCH --job-name=train-nucleus2-fm
#SBATCH -o out_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:RTX6000:1
#SBATCH --time=2-23:59:00


set -e

export WANDB_API_KEY=$(cat config/wandb_api_key.txt)

module load python/3.10.2
source ~/envs/sciml/bin/activate
module load cuda/13.0.1
export PYTHONPATH=/data/homezvol3/srachaba/Projects/NUCLEUS/src:${PYTHONPATH:-}
export PYTHONNOUSERSITE=1

python -c "import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0)); x=torch.zeros(1, device='cuda'); print(x)"

PYTHONNOUSERSITE=1 python scripts/train.py \
  model_cfg=flow_matching/nucleus2_moe_fm_concat_experiment \
  model_cfg.ar_rollout_schedule=[1] \
  model_cfg.ar_rollout_milestones=[] \
  data_cfg=poolboiling128 \
  data_dir=/share/crsp/lab/amowli/share/BubbleML_2_downsampled128_bicubic \
  start_time=200 \
  pydataset=forecast \
  normalizer_cfg=standard \
  optim_cfg.params.lr=1e-3 \
  optim_cfg.params.weight_decay=0 \
  scheduler_cfg=cosine_warmup \
  scheduler_cfg.params.warmup=10000 \
  max_steps=75000 \
  val_check_interval=2000 \
  check_val_every_n_epoch=null \
  log_dir=/share/crsp/lab/amowli/srachaba/nucleus2_fm_logs
