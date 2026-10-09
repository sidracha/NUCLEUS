export WANDB_API_KEY=$(cat config/wandb_api_key.txt)

source ~/envs/sciml/bin/activate
export PYTHONPATH=/home/sidracha/Projects/NUCLEUS/src:${PYTHONPATH:-}

PYTHONNOUSERSITE=1 python scripts/train.py \
    model_cfg=flow_matching/nucleus2_moe_fm_small \
    data_cfg=poolboiling64_op250 \
    data_dir=~/BubbleML_2_downsample64 \
    pydataset=in_mem_forecast \
    normalizer_cfg=standard \
    batch_size=4 \
    accumulate_grad_batches=2 \
    optim_cfg.params.lr=2e-3 \
    optim_cfg.params.weight_decay=1e-3 \
    scheduler_cfg=cosine_warmup \
    max_steps=50000 \
    scheduler_cfg.params.warmup=2000 \
    log_dir=~/nucleus2_fm_logs
