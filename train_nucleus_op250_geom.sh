source ~/envs/sciml/bin/activate
export WANDB_API_KEY=$(cat config/wandb_api_key.txt)
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}

# treasured-bird-24's NUCLEUS settings on the same full OP-250 split and step budget as
# train_fm_op250.sh: train Twall 85/89/101, validate Twall 81, every frame from 0.
# torch.compile is disabled because NATTEN 0.17's fused kernel can't be traced (and the
# flow-matching model never runs compiled either, so speed comparisons stay fair).
# conditioning: x, y, seed SDF and heater mask, encoded separately and added to the tokens
TORCH_COMPILE_DISABLE=1 PYTHONNOUSERSITE=1 python scripts/train.py \
    model_cfg=nucleus2/nucleus2_experiment \
    model_cfg.params.embed_dim=256 \
    model_cfg.params.moe_intermediate_dim=1024 \
    model_cfg.params.processor_blocks=6 \
    model_cfg.params.num_heads=4 \
    model_cfg.params.num_experts=4 \
    model_cfg.params.topk=2 \
    +model_cfg.params.num_condition_channels=4 \
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
    max_steps=25000 \
    check_val_every_n_epoch=5 \
    log_dir=$HOME/nucleus2_fm_logs
