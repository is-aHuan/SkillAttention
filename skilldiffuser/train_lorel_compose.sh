# python hrl/main_hot.py env=lorel_sawyer_obs method=traj_option dt.n_layer=1 dt.n_head=4 option_selector.option_transformer.n_layer=1 option_selector.option_transformer.n_head=4 option_selector.commitment_weight=0.1 option_selector.option_transformer.hidden_size=128 batch_size=256 seed=1 warmup_steps=5000 resume=True checkpoint_path=${checkpoint_pretrain_ckpt} diffuser.loadpath=${diffuser_pretrain_ckpt}
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES="0,1,2,3"

ulimit -n 65535
python -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=29600 \
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
    batch_size=400 seed=1 warmup_steps=5000 resume=False \
    load_options=False \
    freeze_loaded_options=True \
    checkpoint_path=/home/lyh/data_folder/dataspace/lorel/skdif/checkpoint_path/perf40_model_300.ckpt \
    diffuser.loadpath=/home/lyh/data_folder/dataspace/lorel/skdif/diffuser_loadpath/perf40_state_480000.pt