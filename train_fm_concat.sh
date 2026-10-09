source ~/envs/sciml/bin/activate
export WANDB_API_KEY=$(cat config/wandb_api_key.txt)
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}

PYTHONNOUSERSITE=1 python scripts/train.py \
    model_cfg=flow_matching/nucleus2_moe_fm_concat_small \
    model_cfg.flow_time_power=2 \
    model_cfg.context_noise=true \
    model_cfg.residual=false \
    model_cfg.prediction_target=velocity \
    data_cfg=poolboiling64_op250 \
    data_dir=$HOME/BubbleML_2_downsample64 \
    'data_cfg.train_paths=[${data_dir}/PoolBoiling-Subcooled-OP250-2D/Twall_85.hdf5]' \
    'data_cfg.val_paths=[${data_dir}/PoolBoiling-Subcooled-OP250-2D/Twall_85.hdf5]' \
    pydataset=in_mem_forecast \
    normalizer_cfg=op250_64_nondim \
    batch_size=4 \
    accumulate_grad_batches=2 \
    optim_cfg.params.lr=2e-3 \
    optim_cfg.params.weight_decay=0 \
    scheduler_cfg=cosine_warmup \
    scheduler_cfg.params.warmup=200 \
    max_steps=25000 \
    log_dir=$HOME/nucleus2_fm_logs
