BASE_PATH="/home/lyh/Data/lorel/skdif/checkpoint_path"
ITER=100
CKPT="${BASE_PATH}/model_${ITER}.ckpt"

#method=option_dt
# python hrl/main_hot.py env=lorel_sawyer_obs method=traj_option trainer.device='cuda' dt.n_layer=1 dt.n_head=4 option_selector.option_transformer.n_layer=1 option_selector.option_transformer.n_head=4 option_selector.commitment_weight=0.1 option_selector.option_transformer.hidden_size=128 batch_size=256 seed=1 warmup_steps=5000 eval=True render=True checkpoint_path=${CKPT}
python hrl/main_hot.py env=lorel_sawyer_obs method=traj_option \
    dt.n_layer=1 dt.n_head=4 \
    option_selector.option_transformer.n_layer=1 \
    option_selector.option_transformer.n_head=4 \
    option_selector.commitment_weight=0.1 \
    option_selector.option_transformer.hidden_size=128 \
    batch_size=256 seed=1 warmup_steps=5000 resume=True \
    checkpoint_path=/home/lyh/Data/lorel/skdif/checkpoint_path/perf40_model_300.ckpt \
    diffuser.loadpath=/home/lyh/Data/lorel/skdif/diffuser_loadpath/perf40_state_480000.pt
