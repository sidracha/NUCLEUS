source ~/envs/sciml/bin/activate
export WANDB_API_KEY=$(cat config/wandb_api_key.txt)
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}

# the setup that stayed stable in the 4-window sanity check (full-field velocity, white context
# noise + SDF corruptions, uniform t), trained on the full OP-250 split:
# train Twall 85/89/101, validate Twall 81 (test 93/97 held out), every frame from 0
# conditioning: x, y, seed SDF and heater mask, encoded separately and added to the tokens
PYTHONNOUSERSITE=1 python scripts/train.py \
    model_cfg=flow_matching/nucleus2_moe_fm_concat_small \
    model_cfg.prediction_target=velocity \
    model_cfg.residual=false \
    model_cfg.flow_time_power=1 \
    model_cfg.context_noise=true \
    model_cfg.sdf_corruptions=true \
    model_cfg.params.num_condition_channels=4 \
    +model_cfg.params.condition_injection=additive \
    +model_cfg.params.condition_maps=geometry \
    data_cfg=poolboiling64_op250_split \
    data_dir=$HOME/BubbleML_2_downsample64 \
    start_time=0 \
    pydataset=in_mem_forecast \
    normalizer_cfg=op250_64_nondim \
    batch_size=4 \
    accumulate_grad_batches=2 \
    optim_cfg.params.lr=2e-3 \
    optim_cfg.params.weight_decay=0 \
    scheduler_cfg=cosine_warmup \
    scheduler_cfg.params.warmup=1000 \
    max_steps=30000 \
    check_val_every_n_epoch=5 \
    log_dir=$HOME/nucleus2_fm_logs
