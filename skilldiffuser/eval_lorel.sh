# BASE_PATH="/home/lyh/Data/lorel/skdif/checkpoint_path"
# ITER=100
# CKPT="${BASE_PATH}/model_${ITER}.ckpt"

#method=option_dt
# python hrl/main_hot.py env=lorel_sawyer_obs method=traj_option trainer.device='cuda' dt.n_layer=1 dt.n_head=4 option_selector.option_transformer.n_layer=1 option_selector.option_transformer.n_head=4 option_selector.commitment_weight=0.1 option_selector.option_transformer.hidden_size=128 batch_size=256 seed=1 warmup_steps=5000 eval=True render=True checkpoint_path=${CKPT}
python skilldiffuser/hrl/main_hot.py env=lorel_sawyer_obs method=traj_option \
    dt.n_layer=1 dt.n_head=4 \
    option_selector.option_transformer.n_layer=1 \
    option_selector.option_transformer.n_head=4 \
    option_selector.commitment_weight=0.1 \
    option_selector.option_transformer.hidden_size=128 \
    batch_size=256 seed=1 warmup_steps=5000 resume=True \
    eval=True \
    checkpoint_path=/root/codespace/SkillDiffuserTune/outputs/2025-07-07/11-24-34/checkpoints/LorlEnv-v0-40108-traj_option-2025-07-07-11:24:34/model_50.ckpt \
    diffuser.loadpath=/root/codespace/SkillDiffuserTune/outputs/2025-07-07/11-24-34/buckets/diffuser_log/checkpoint/state_360000.pt
