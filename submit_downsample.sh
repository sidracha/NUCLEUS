#!/bin/bash
#SBATCH -A amowli_lab
#SBATCH -p free-gpu32
#SBATCH --job-name=downsample128
#SBATCH -o out_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=2-23:59:00

set -e

module load python/3.10.2
source ~/envs/sciml/bin/activate
module load cuda/13.0.1
export PYTHONPATH=/data/homezvol3/srachaba/Projects/NUCLEUS/src:${PYTHONPATH:-}
export PYTHONNOUSERSITE=1

SRC=/share/crsp/lab/amowli/share/BubbleML_2
DST=/share/crsp/lab/amowli/share/BubbleML_2_downsampled128_bicubic
CFG=/data/homezvol3/srachaba/Projects/NUCLEUS/config/data_cfg/poolboiling.yaml

python3 - <<'EOF'
import re, shutil, sys
from pathlib import Path
import torch
sys.path.insert(0, "/data/homezvol3/srachaba/Projects/NUCLEUS/scripts")
from downsample_dataset import downsample_hdf5

src_root = Path("/share/crsp/lab/amowli/share/BubbleML_2")
dst_root = Path("/share/crsp/lab/amowli/share/BubbleML_2_downsampled128_bicubic")
cfg_path = Path("/data/homezvol3/srachaba/Projects/NUCLEUS/config/data_cfg/poolboiling64.yaml")

# Extract every path that looks like a .hdf5 reference (strip ${data_dir} prefix)
text = cfg_path.read_text()
rel_paths = sorted(set(re.findall(r"\$\{data_dir\}(/[^\s]+\.hdf5)", text)))
print(f"Found {len(rel_paths)} unique paths in config")

for rel in rel_paths:
    src = src_root / rel.lstrip("/")
    dst = dst_root / rel.lstrip("/")
    if not src.exists():
        print(f"  MISSING: {src}")
        continue
    print(f"  {src.name} -> {dst}", flush=True)
    downsample_hdf5(src, dst, size=128, device="cuda", chunk_t=256)
    # copy companion JSON
    json_src = src.with_suffix(".json")
    if json_src.exists():
        json_dst = dst.with_suffix(".json")
        json_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(json_src, json_dst)

print("Done.")
EOF
