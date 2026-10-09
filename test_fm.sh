source ~/envs/sciml/bin/activate
export PYTHONPATH=/home/sidracha/Projects/NUCLEUS/src:${PYTHONPATH:-}

PYTHONNOUSERSITE=1 python scripts/inf_fm.py \
	checkpoint_path=~/nucleus2_fm_logs/nucleus2_moe_fm_poolboiling64_op250_2026-10-07/checkpoints/last.ckpt \
	data_dir=~/BubbleML_2_downsample64 \
	trajectory_steps=40
