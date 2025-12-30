export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES="2,3,4,5,6,7"

python -m torch.distributed.run \
    --nproc_per_node=6 \
    skilldiffuser/hrl/main_hot.py \
    env=lorel_sawyer_obs \
    method=traj_option \
    wandb=True \
    render=False \
    eval=True \
    dt.n_layer=1 dt.n_head=4 \
    option_selector.option_transformer.n_layer=1 \
    option_selector.option_transformer.n_head=4 \
    option_selector.commitment_weight=0.1 \
    option_selector.option_transformer.hidden_size=128 \
    batch_size=280 seed=1 warmup_steps=5000 resume=True \
    checkpoint_path=/home/lyh/data_folder/skdif_tune_outputs/2025-08-22/13-05-33/checkpoints/LorlEnv-v0-40108-traj_option-2025-08-22-13:05:33/model_60.ckpt \
    diffuser.loadpath=/home/lyh/data_folder/skdif_tune_outputs/2025-08-22/13-05-33/buckets/diffuser_log/checkpoint/state_490000.pt
