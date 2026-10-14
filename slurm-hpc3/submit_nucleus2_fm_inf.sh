#!/bin/bash
#SBATCH -A amowli_lab
#SBATCH -p free-gpu32
#SBATCH --job-name=inf-nucleus2-fm
#SBATCH -o out_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00


set -e

export WANDB_API_KEY=$(cat config/wandb_api_key.txt)

module load python/3.10.2
source ~/envs/sciml/bin/activate
module load cuda/13.0.1
export PYTHONPATH=/data/homezvol3/srachaba/Projects/NUCLEUS/src:${PYTHONPATH:-}
export PYTHONNOUSERSITE=1

python -c "import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0)); x=torch.zeros(1, device='cuda'); print(x)"

PYTHONNOUSERSITE=1 python scripts/inf_fm.py \
  checkpoint_path=$HOME/nucleus2_fm_logs/serene-firefly-69/checkpoints/last.ckpt \
  model_cfg=flow_matching/nucleus2_moe_fm_concat_medium \
  data_dir=/share/crsp/lab/amowli/share/BubbleML_2_downsampled128_bicubic \
  data_cfg=poolboiling128 \
  normalizer_cfg=standard \
  history_time_window=4 \
  future_time_window=4 \
  start_time=300 \
  trajectory_steps=200 \
  seed=0 \
  plot=false
