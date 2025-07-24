# python hrl/main_hot.py env=lorel_sawyer_obs method=traj_option dt.n_layer=1 dt.n_head=4 option_selector.option_transformer.n_layer=1 option_selector.option_transformer.n_head=4 option_selector.commitment_weight=0.1 option_selector.option_transformer.hidden_size=128 batch_size=256 seed=1 warmup_steps=5000 resume=True checkpoint_path=${checkpoint_pretrain_ckpt} diffuser.loadpath=${diffuser_pretrain_ckpt}

export CUDA_VISIBLE_DEVICES="4,5,6,7"

python -m torch.distributed.run \
    --nproc_per_node=4 \
    skilldiffuser/hrl/main_hot.py \
    env=lorel_sawyer_obs \
    method=traj_option \
    wandb=True \
    render=False \
    dt.n_layer=1 dt.n_head=4 \
    option_selector.option_transformer.n_layer=1 \
    option_selector.option_transformer.n_head=4 \
    option_selector.commitment_weight=0.1 \
    option_selector.option_transformer.hidden_size=128 \
    batch_size=320 seed=1 warmup_steps=5000 resume=True \
    max_iters=100 \
    checkpoint_path=/root/codespace/SkillDiffuserTune/outputs/2025-07-23/09-58-57/checkpoints/LorlEnv-v0-40108-traj_option-2025-07-23-09:58:57/model_40.ckpt \
    diffuser.loadpath=/root/codespace/SkillDiffuserTune/outputs/2025-07-23/09-58-57/buckets/diffuser_log/checkpoint/state_200000.pt