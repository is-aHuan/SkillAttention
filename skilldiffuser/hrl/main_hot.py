import ddp_patch
import sys

try:
    import lorl_env
except:
    print('Lorl env not installed')

import os
os.environ["TZ"] = "Asia/Shanghai"
# time.tzset()
import gym
import numpy as np
import torch
import torch.distributed as dist

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import logging
import wandb
import datetime
import time
import ast
import hydra
import random

from omegaconf import DictConfig, OmegaConf
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader
from expert_dataset import ExpertDataset
from hrl_model import HRLModel
from trainer import Trainer
from difftrainer import DiffTrainer
import diffuser.utils as diffutils
from omegaconf import open_dict
from inferencer.robobrain_planner import RoboBrainPlanner

def create_global_logger(log_file=None, log_level=logging.INFO):
    """创建全局 logger，只有 rank 0 输出"""
    is_ddp = dist.is_available() and dist.is_initialized()
    global_rank = dist.get_rank() if is_ddp else 0
    logger = logging.getLogger('main')  # 使用固定名称
    
    if logger.handlers:
        logger.handlers.clear()
        
    logger.setLevel(log_level if global_rank == 0 else logging.ERROR)
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')

    console = logging.StreamHandler()
    console.setLevel(log_level if global_rank == 0 else logging.ERROR)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if global_rank == 0 else logging.ERROR)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger

def init_logger():
    global logger
    logger = create_global_logger()

def build_train_dataset_and_loader(args, device, batch_size):
    """
    使用 mmap 版 ExpertDataset：
      - rank0 负责一次性写出 memmap 文件
      - 其他 rank 只附着，不再广播大对象
    """
    # 识别 DDP 环境/进程信息
    local_rank = int(os.environ.get("LOCAL_RANK", -1)) if "cuda" in device else -1
    is_ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_ddp else 0

    # 取数据集参数
    train_dataset_args = dict(args.train_dataset)  # hydra 的 DictConfig 转 dict
    # 统一一个 memmap 输出目录（可在 config 里设置 train_dataset.memmap_dir）
    memmap_dir = train_dataset_args.pop(
        "memmap_dir",
        os.path.join(getattr(args, "hydra_base_dir", "."), "mmap_train")
    )

    # 组装 ExpertDataset 的参数（与你原来的 env 开关兼容）
    common_kwargs = dict(
        expert_location=train_dataset_args["expert_location"],
        num_trajectories=train_dataset_args.get("num_trajectories", 1),
        subsample_frequency=train_dataset_args.get("subsample_frequency", 1),
        seed=getattr(args, "seed", 0),
        full_traj=train_dataset_args.get("full_traj", True),
        normalize_states=train_dataset_args.get("normalize_states", True),
        no_lang=train_dataset_args.get("no_lang", False),
        memmap_dir=memmap_dir,
        # writer / force_rebuild 下面分别设
    )
    if 'BabyAI' in args.env.name:
        common_kwargs["use_direction"] = getattr(args.env, "use_direction", False)
    elif 'Lorl' in args.env.name:
        common_kwargs["use_state"] = getattr(args.env, "use_state", False)
    # Hopper 无附加开关

    # ---------- 不再广播大对象，rank0 写文件，其他 rank 附着 ----------
    if is_ddp:
        if rank == 0:
            print("Rank 0: building memmap files...")
            train_dataset = ExpertDataset(**common_kwargs, writer=True,
                                          force_rebuild=train_dataset_args.get("force_rebuild", False))
        dist.barrier()  # 确保写完
        if rank != 0:
            print(f"Rank {rank}: attaching to memmap at {memmap_dir} ...")
            train_dataset = ExpertDataset(**common_kwargs, writer=False)
        dist.barrier()  # 可选再同步一次
    else:
        print("Single process: building memmap files...")
        train_dataset = ExpertDataset(**common_kwargs, writer=True,
                                      force_rebuild=train_dataset_args.get("force_rebuild", False))

    # ---------- Sampler / DataLoader ----------
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if is_ddp else None
    shuffle = False if train_sampler is not None else True
    num_workers = 8 if is_ddp else 32

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler
    )
    return train_dataset, train_loader, train_sampler


def evaluate(cfg):
    # load saved arguments
    device = cfg.trainer.device 
    checkpoint = torch.load(cfg.checkpoint_path, map_location='cpu')
    args = checkpoint['config']
    max_length = checkpoint['train_dataset_max_length']
    args.eval = cfg.eval
    args.render = cfg.render
    args.checkpoint_path = cfg.checkpoint_path
    args.diffuser.loadpath = cfg.diffuser.loadpath
    args.hydra_base_dir = cfg.hydra_base_dir

    args.trainer.device = device
    args.method = args.model.name
    exp_name = f'{args.project_name}-{args.train_dataset.num_trajectories}-{args.method}'
    args.savepath = f'{args.hydra_base_dir}/{args.savedir}/{exp_name}-{datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}'

    # with open_dict(args.train_dataset):
    #     args.train_dataset.expert_location = '/home/lyh/dataspace/lorel/may_08_sawyer_50k/prep_data.pkl'
    #     # args.diffuser.loadpath = '/root/lorel/skdif_eval/state_400000.pt',
    #     args.train_dataset.num_trajectories = 1000

    # # Set num train_trajs to something small
    # args.train_dataset.num_trajectories = 100
    print(OmegaConf.to_yaml(args))

    args.method = args.model.name

    exp_name = f'Tune-eval-{args.project_name}-{args.train_dataset.num_trajectories}-{args.method}'
    args.savepath = f'{args.hydra_base_dir}/{args.savedir}/{exp_name}-{datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}'

    if args.wandb:
        wandb.init(
            name=exp_name,
            group=args.method,
            project=f'debug_eval_hrl_{args.env.name}',
            config=dict(args),
            settings=wandb.Settings(init_timeout=480)
            # entity='language-rl'
        )

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    num_eval_episodes = args.trainer.num_eval_episodes
    print('=' * 50)
    print(f'Starting evaluation: {args.env.name}')
    print(f'{args.trainer.num_eval_episodes} trajectories')
    print('=' * 50)

    state_dim = args.env.state_dim
    action_dim = args.env.action_dim
    if isinstance(state_dim, str):
        state_dim = ast.literal_eval(state_dim)

    if isinstance(state_dim, tuple):
        assert not args.trainer.state_il, "Cannot do state imitation learning with an image input"

    if not args.env.eval_offline:
        env = gym.make(args.env.name)
        env_name = args.env.name
        env.seed(args.seed)

    if 'BabyAI' in args.env.name:
        state_dim += 4*args.env.use_direction

    # args.train_dataset = {'expert_location': '/home/liangzhixuan.p/dataset/lorel/may_08_sawyer_50k/prep_data3.pkl', 
    #                       'num_trajectories': 1000, 
    #                       'normalize_states': False, 
    #                       'no_lang': False, 'seed': 1, 
    #                       'aug': True, 
    #                       'path': '/home/lyh/dataspace/lorel/may_08_sawyer_50k/prep_data.pkl'}
    


    train_dataset_args = dict(args.train_dataset)
    batch_size = args.batch_size
    if 'BabyAI' in args.env.name:
        train_dataset = ExpertDataset(**train_dataset_args, use_direction=args.env.use_direction)
    elif 'Lorl' in args.env.name:
        train_dataset = ExpertDataset(**train_dataset_args, use_state=args.env.use_state)
    elif 'Hopper' in args.env.name:
        train_dataset = ExpertDataset(**train_dataset_args)
    else:
        raise NotImplementedError


    local_rank = None
    if 'cuda' in device and os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', -1))

    train_sampler = None
    shuffle = True

    if local_rank is not None and local_rank != -1: # 如果在DDP环境中
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        shuffle = False # 当提供sampler时，shuffle必须为False

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=32,
                              shuffle=shuffle, pin_memory=True, drop_last=True, sampler=train_sampler)

    print('=' * 50)
    print(f'Starting new experiment: {args.env.name} {args.train_dataset.num_trajectories}')
    print(f'{len(train_dataset)} trajectories, {train_dataset.total_timesteps} timesteps found')
    print('=' * 50)

    if args.method == 'traj_option':
        args.option_selector.option_transformer.max_length = int(max_length)
        args.option_selector.option_transformer.max_ep_len = args.env.eval_episode_factor * \
            int(max_length)

    option_selector_args = dict(args.option_selector)
    option_selector_args['state_dim'] = state_dim
    option_selector_args['option_dim'] = args.option_dim
    option_selector_args['codebook_dim'] = args.codebook_dim
    state_reconstructor_args = dict(args.state_reconstructor)
    lang_reconstructor_args = dict(args.lang_reconstructor)
    # decision_transformer_args = {'state_dim': state_dim,
    #                              'action_dim': action_dim,
    #                              'option_dim': args.option_dim,
    #                              'discrete': args.env.discrete,
    #                              'hidden_size': args.dt.hidden_size,
    #                              'use_language': args.method == 'vanilla',
    #                              'use_options': args.method != 'vanilla',
    #                              'option_il': args.dt.option_il,
    #                              'max_length': max_length if args.method != 'traj_option' else args.model.K,
    #                              'max_ep_len': args.env.eval_episode_factor*max_length,
    #                              'action_tanh': False,
    #                              'n_layer': args.dt.n_layer,
    #                              'n_head': args.dt.n_head,
    #                              'n_inner': 4*args.dt.hidden_size,
    #                              'activation_function': args.dt.activation_function,
    #                              'n_positions': args.dt.n_positions,
    #                              'n_ctx': args.dt.n_positions,
    #                              'resid_pdrop': args.dt.dropout,
    #                              'attn_pdrop': args.dt.dropout,
    #                              'eval': args.eval,
    #                              }

    decision_args = {'state_dim': state_dim,
                     'action_dim': action_dim,
                     'option_dim': args.option_dim,
                     'discrete': args.env.discrete,
                     'hidden_size': args.dt.hidden_size,
                     'use_language': args.method == 'vanilla',
                     'use_options': args.method != 'vanilla',
                     'predict_q': args.use_iq,
                     'max_length': train_dataset.max_length if 'option' not in args.method else args.model.K,  # used to be K
                     'max_ep_len': args.env.eval_episode_factor * train_dataset.max_length,
                     'eval': args.eval,
                     }

    # tmp_path = os.path.split(args.hydra_base_dir)
    # diffuser_save_path = os.path.join("outputs", os.path.split(tmp_path[0])[1], tmp_path[1])
    diffuser_save_path = args.hydra_base_dir
    if args.diffuser.diffusion == 'models.GaussianInvDynDiffusion':
        model_config = diffutils.Config(
            args.diffuser.model,
            # savepath=os.path.join(diffuser_save_path, 'model_config.pkl'),
            horizon=args.diffuser.horizon,
            # transition_dim=observation_dim,
            # cond_dim=observation_dim,
            transition_dim=args.dt.hidden_size,  # TODO
            cond_dim=args.dt.hidden_size,  # TODO
            dim_mults=eval(args.diffuser.dim_mults),
            returns_condition=args.diffuser.returns_condition,
            dim=args.diffuser.dim,
            condition_dropout=args.diffuser.condition_dropout,
            calc_energy=args.diffuser.calc_energy,
            device=device,
        )

        diffusion_config = diffutils.Config(
            args.diffuser.diffusion,
            # savepath=os.path.join(diffuser_save_path, 'diffusion_config.pkl'),
            horizon=args.diffuser.horizon,
            # observation_dim=observation_dim,
            # action_dim=action_dim,
            observation_dim=args.dt.hidden_size,  # TODO
            action_dim=args.env.action_dim,  # TODO
            n_timesteps=args.diffuser.n_diffusion_steps,
            loss_type=args.diffuser.loss_type,
            clip_denoised=args.diffuser.clip_denoised,
            predict_epsilon=args.diffuser.predict_epsilon,
            hidden_dim=args.diffuser.inv_hidden_dim,  # TODO
            ar_inv=args.diffuser.ar_inv,
            train_only_inv=args.diffuser.train_only_inv,
            ## loss weighting
            action_weight=args.diffuser.action_weight,
            loss_weights=args.diffuser.loss_weights,
            loss_discount=args.diffuser.loss_discount,
            returns_condition=args.diffuser.returns_condition,
            condition_guidance_w=args.diffuser.condition_guidance_w,
            device=device,
        )
        
    elif args.diffuser.diffusion == 'models.ActionDiffusion':
        model_config = diffutils.Config(
            args.diffuser.model,
            # savepath=os.path.join(diffuser_save_path, 'model_config.pkl'),
            horizon=args.diffuser.horizon,
            # transition_dim=observation_dim + action_dim,
            # cond_dim=observation_dim,
            transition_dim=args.dt.hidden_size,  # TODO (action, obs)
            cond_dim=args.dt.hidden_size,  # TODO
            dim_mults=eval(args.diffuser.dim_mults),
            returns_condition=args.diffuser.returns_condition,
            dim=args.diffuser.dim,
            condition_dropout=args.diffuser.condition_dropout,
            calc_energy=args.diffuser.calc_energy,
            device=device,
        )

        diffusion_config = diffutils.Config(
            args.diffuser.diffusion,
            # savepath=os.path.join(diffuser_save_path, 'diffusion_config.pkl'),
            horizon=args.diffuser.horizon,
            hidden_size=args.dt.hidden_size,  # TODO
            # observation_dim=observation_dim,
            # action_dim=action_dim,
            # observation_dim=args.dt.hidden_size,  # TODO
            action_dim=args.env.action_dim,  # TODO
            n_timesteps=args.diffuser.n_diffusion_steps,
            loss_type=args.diffuser.loss_type,
            clip_denoised=args.diffuser.clip_denoised,
            predict_epsilon=args.diffuser.predict_epsilon,
            ## loss weighting
            # action_weight=args.diffuser.action_weight,
            # loss_weights=args.diffuser.loss_weights,
            loss_discount=args.diffuser.loss_discount,
            returns_condition=args.diffuser.returns_condition,
            condition_guidance_w=args.diffuser.condition_guidance_w,
            device=device,
        )

    else:
        model_config = diffutils.Config(
            args.diffuser.model,
            # savepath=os.path.join(diffuser_save_path, 'model_config.pkl'),
            horizon=args.diffuser.horizon,
            # transition_dim=observation_dim + action_dim,
            # cond_dim=observation_dim,
            transition_dim=2 * args.dt.hidden_size,  # TODO (action, obs)
            cond_dim=args.dt.hidden_size,  # TODO
            dim_mults=eval(args.diffuser.dim_mults),
            returns_condition=args.diffuser.returns_condition,
            dim=2 * args.diffuser.dim,
            condition_dropout=args.diffuser.condition_dropout,
            calc_energy=args.diffuser.calc_energy,
            device=device,
        )

        diffusion_config = diffutils.Config(
            args.diffuser.diffusion,
            # savepath=os.path.join(diffuser_save_path, 'diffusion_config.pkl'),
            horizon=args.diffuser.horizon,
            # observation_dim=observation_dim,
            # action_dim=action_dim,
            observation_dim=args.dt.hidden_size,  # TODO
            action_dim=args.env.action_dim,  # TODO
            n_timesteps=args.diffuser.n_diffusion_steps,
            loss_type=args.diffuser.loss_type,
            clip_denoised=args.diffuser.clip_denoised,
            predict_epsilon=args.diffuser.predict_epsilon,
            ## loss weighting
            action_weight=args.diffuser.action_weight,
            loss_weights=args.diffuser.loss_weights,
            loss_discount=args.diffuser.loss_discount,
            returns_condition=args.diffuser.returns_condition,
            condition_guidance_w=args.diffuser.condition_guidance_w,
            device=device,
        )

    diff_trainer_args = dict(
        # savepath=os.path.join(diffuser_save_path, 'trainer_config.pkl'),
        train_batch_size=args.diffuser.batch_size,  # TODO
        train_lr=args.diffuser.learning_rate,
        gradient_accumulate_every=args.diffuser.gradient_accumulate_every,
        ema_decay=args.diffuser.ema_decay,
        sample_freq=args.diffuser.sample_freq,
        save_freq=args.diffuser.save_freq,
        log_freq=args.diffuser.log_freq,
        label_freq=int(args.diffuser.n_train_steps // args.diffuser.n_saves),
        save_parallel=args.diffuser.save_parallel,
        bucket=args.diffuser.bucket,
        n_reference=args.diffuser.n_reference,
        train_device=device,
        save_checkpoints=args.diffuser.save_checkpoints,
        update_ema_every=10,
        parallel=args.parallel,
    )
                        
    hrl_model_args = dict(args.model)

    iq_args = cfg.iq

    unet = model_config()
    diffusion_model = diffusion_config(unet)
    diff_trainer = DiffTrainer(args, diffusion_model, diff_trainer_args)

    model = HRLModel(args, option_selector_args, state_reconstructor_args,
                     lang_reconstructor_args, decision_args, iq_args, diff_trainer, device, **hrl_model_args)
    
    model.to(device)
    model.load_state_dict(checkpoint['model'], strict=False)

    # start_iter = checkpoint['iter_num'] + 1
    assert train_dataset.max_length == checkpoint[
        'train_dataset_max_length'], f"Expected max length of dataset to be {train_dataset.max_length} but got {checkpoint['train_dataset_max_length']}"

    if ".pt" in args.diffuser.loadpath:
        model.diff_trainer.load(args.diffuser.loadpath)

    del checkpoint
    torch.cuda.empty_cache()

    print(model)
    if local_rank is not None: # 使用 DDP
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                          find_unused_parameters=True)
    elif args.parallel and torch.cuda.device_count() > 1: # 使用 DataParallel 
        model = torch.nn.DataParallel(model)

    trainer_args = dict(args.trainer)

    # planner = RoboBrainPlanner()
    trainer = Trainer(
        args,
        model=model,
        # planner=planner,
        tokenizer=tokenizer,
        optimizer=None,
        train_loader=train_loader,
        env=env,
        env_name=env_name,
        val_loader=None,
        scheduler=None,
        skip_words=args.env.skip_words,
        **trainer_args
    )

    # Restore trainer from checkpoint
    # trainer.load(args.checkpoint_path)
    # trainer.model.diff_trainer.load(args.diffuser.loadpath)

    for iter_num in range(10):
        eval_time_start = time.time()
        render_path = f'{args.hydra_base_dir}/renders'
        eval_metrics = trainer.evaluate(iter_num=iter_num+1, render=args.render, max_ep_len=500, render_path=render_path, lorl_compose=False, lorl_single=False)

        logs = dict()
        for k,v in eval_metrics.items():
            logs[f'evaluation/{k}'] = v
        logs['evaluation/eval_time'] = time.time() - eval_time_start
        wandb.log(logs, step=iter_num+1)
        
        print('=' * 80)
        for k,v in logs.items():
            print(f'{k}: {v}')


def train(args):
    device = args.trainer.device
    args.method = args.model.name
    exp_name = f'{args.project_name}-{args.train_dataset.num_trajectories}-{args.method}'
    args.savepath = f'{args.hydra_base_dir}/{args.savedir}/{exp_name}-{datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}'


    is_ddp = dist.is_available() and dist.is_initialized()
    global_rank = dist.get_rank() if is_ddp else 0
    world_size = dist.get_world_size() if is_ddp else 1
    num_gpus = torch.cuda.device_count()
    local_rank = int(os.environ.get("LOCAL_RANK", -1)) if is_ddp else 0

    print("rank: %d, world_size: %d" % (global_rank, world_size))
    print("local_rank: %d, num_gpus: %d" % (local_rank, num_gpus))

    wandb_logger = logging.getLogger("wandb")
    wandb_logger.setLevel(logging.INFO)
    if args.wandb:
        # os.environ["WANDB_MODE"] = "offline"
        wandb.init(
            name=exp_name,
            group=args.method,
            project=f'Tune-train-{args.env.name}',  # diff_hot_pluo_diff_loss_{args.env.name} # TODO frozen  diff_hot_A100_LN
            config=dict(args),
            # entity='language-rl'
        )

    if not os.path.isdir(args.savepath):
        os.makedirs(args.savepath, exist_ok=True)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    #K = args['K']
    batch_size = args.batch_size

    # ==== 替换开始：用 mmap 版 ExpertDataset 构建数据集和 DataLoader ====
    train_dataset, train_loader, train_sampler = build_train_dataset_and_loader(
        args=args,
        device=args.trainer.device,
        batch_size=batch_size
    )
    # ==== 替换结束 ====

    print('=' * 50)
    total_timesteps = int(train_dataset.trajectories["lengths"].sum())  # memmap/内存模式都可用
    print(f'Starting new experiment: {args.env.name} {args.train_dataset.num_trajectories}')
    print(f'{len(train_dataset)} trajectories, {total_timesteps} timesteps found')
    print('=' * 50)

    state_dim = args.env.state_dim
    action_dim = args.env.action_dim
    if isinstance(state_dim, str):
        state_dim = ast.literal_eval(state_dim)

    if isinstance(state_dim, tuple):
        assert not args.trainer.state_il, "Cannot do state imitation learning with an image input"

    if not args.env.eval_offline:
        if args.env.eval_env:
            env_name = args.env.eval_env
        else:
            env_name = args.env.name
        # print(f'-->Testing on {env_name}')
        print(f'-->Testing on {env_name}')

        env = gym.make(env_name)
        env_name = env_name
        val_loader = None
    else:
        val_dataset_args = dict(args.val_dataset)
        if 'BabyAI' in args.env.name:
            val_dataset = ExpertDataset(**val_dataset_args, use_direction=args.env.use_direction)
        elif 'lorel' in args.env.name:
            val_dataset = ExpertDataset(**val_dataset_args, use_state=args.env.use_state)
        else:
            raise NotImplementedError
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=32,
                                shuffle=True, pin_memory=True, drop_last=True)

    if 'BabyAI' in args.env.name:
        state_dim += 4*args.env.use_direction

    # print(f'--> Train episode max length: {train_dataset.max_length}')
    print(f'--> Train episode max length: {train_dataset.max_length}')
    if args.method == 'traj_option':
        args.option_selector.option_transformer.max_length = int(train_dataset.max_length)
        args.option_selector.option_transformer.max_ep_len = args.env.eval_episode_factor * \
            int(train_dataset.max_length)

    if args.model.horizon == 'max':
        args.model.horizon = int(train_dataset.max_length)
    if args.model.K == 'max':
        args.model.K = int(train_dataset.max_length)

    option_selector_args = dict(args.option_selector)
    option_selector_args['state_dim'] = state_dim
    option_selector_args['option_dim'] = args.option_dim
    option_selector_args['codebook_dim'] = args.codebook_dim
    state_reconstructor_args = dict(args.state_reconstructor)
    lang_reconstructor_args = dict(args.lang_reconstructor)
    decision_args = {'state_dim': state_dim,
                     'action_dim': action_dim,
                     'option_dim': args.option_dim,
                     'discrete': args.env.discrete,
                     'hidden_size': args.dt.hidden_size,
                     'use_language': args.method == 'vanilla',
                     'use_options': args.method != 'vanilla',
                     'predict_q': args.use_iq,
                     'max_length': train_dataset.max_length if 'option' not in args.method else args.model.K,  # used to be K
                     'max_ep_len': args.env.eval_episode_factor * train_dataset.max_length,
                     'eval': args.eval,
                     }

    # decision_transformer_args = {
    #                              'option_il': args.dt.option_il,
    #                              'n_layer': args.dt.n_layer,
    #                              'n_head': args.dt.n_head,
    #                              'activation_function': args.dt.activation_function,
    #                              'n_positions': args.dt.n_positions,
    #                              'n_ctx': args.dt.n_positions,
    #                              'resid_pdrop': args.dt.dropout,
    #                              'attn_pdrop': args.dt.dropout,
    #                              'no_states': args.dt.no_states,
    #                              'no_actions': args.dt.no_actions,
    #                              }

    # tmp_path = os.path.split(args.hydra_base_dir)
    # diffuser_save_path = os.path.join("outputs", os.path.split(tmp_path[0])[1], tmp_path[1])

    diffuser_save_path = args.hydra_base_dir
    if args.diffuser.diffusion == 'models.GaussianInvDynDiffusion':
        model_config = diffutils.Config(
            args.diffuser.model,
            savepath=os.path.join(diffuser_save_path, 'model_config.pkl'),
            horizon=args.diffuser.horizon,
            # transition_dim=observation_dim,
            # cond_dim=observation_dim,
            transition_dim=args.dt.hidden_size,  # TODO
            cond_dim=args.dt.hidden_size,  # TODO
            dim_mults=eval(args.diffuser.dim_mults),
            returns_condition=args.diffuser.returns_condition,
            dim=args.diffuser.dim,
            condition_dropout=args.diffuser.condition_dropout,
            calc_energy=args.diffuser.calc_energy,
            device=device,
        )

        diffusion_config = diffutils.Config(
            args.diffuser.diffusion,
            savepath=os.path.join(diffuser_save_path, 'diffusion_config.pkl'),
            horizon=args.diffuser.horizon,
            # observation_dim=observation_dim,
            # action_dim=action_dim,
            observation_dim=args.dt.hidden_size,  # TODO
            action_dim=args.env.action_dim,  # TODO
            n_timesteps=args.diffuser.n_diffusion_steps,
            loss_type=args.diffuser.loss_type,
            clip_denoised=args.diffuser.clip_denoised,
            predict_epsilon=args.diffuser.predict_epsilon,
            hidden_dim=args.diffuser.inv_hidden_dim,  # TODO
            ar_inv=args.diffuser.ar_inv,
            train_only_inv=args.diffuser.train_only_inv,
            ## loss weighting
            action_weight=args.diffuser.action_weight,
            loss_weights=args.diffuser.loss_weights,
            loss_discount=args.diffuser.loss_discount,
            returns_condition=args.diffuser.returns_condition,
            condition_guidance_w=args.diffuser.condition_guidance_w,
            device=device,
        )
        
    elif args.diffuser.diffusion == 'models.ActionDiffusion':
        model_config = diffutils.Config(
            args.diffuser.model,
            savepath=os.path.join(diffuser_save_path, 'model_config.pkl'),
            horizon=args.diffuser.horizon,
            # transition_dim=observation_dim + action_dim,
            # cond_dim=observation_dim,
            transition_dim=args.dt.hidden_size,  # TODO (action, obs)
            cond_dim=args.dt.hidden_size,  # TODO
            dim_mults=eval(args.diffuser.dim_mults),
            returns_condition=args.diffuser.returns_condition,
            dim=args.diffuser.dim,
            condition_dropout=args.diffuser.condition_dropout,
            calc_energy=args.diffuser.calc_energy,
            device=device,
        )

        diffusion_config = diffutils.Config(
            args.diffuser.diffusion,
            savepath=os.path.join(diffuser_save_path, 'diffusion_config.pkl'),
            horizon=args.diffuser.horizon,
            hidden_size=args.dt.hidden_size,  # TODO
            # observation_dim=observation_dim,
            # action_dim=action_dim,
            # observation_dim=args.dt.hidden_size,  # TODO
            action_dim=args.env.action_dim,  # TODO
            n_timesteps=args.diffuser.n_diffusion_steps,
            loss_type=args.diffuser.loss_type,
            clip_denoised=args.diffuser.clip_denoised,
            predict_epsilon=args.diffuser.predict_epsilon,
            ## loss weighting
            # action_weight=args.diffuser.action_weight,
            # loss_weights=args.diffuser.loss_weights,
            loss_discount=args.diffuser.loss_discount,
            returns_condition=args.diffuser.returns_condition,
            condition_guidance_w=args.diffuser.condition_guidance_w,
            device=device,
        )

    else:
        model_config = diffutils.Config(
            args.diffuser.model,
            savepath=os.path.join(diffuser_save_path, 'model_config.pkl'),
            horizon=args.diffuser.horizon,
            # transition_dim=observation_dim + action_dim,
            # cond_dim=observation_dim,
            transition_dim=2 * args.dt.hidden_size,  # TODO (action, obs)
            cond_dim=args.dt.hidden_size,  # TODO
            dim_mults=eval(args.diffuser.dim_mults),
            returns_condition=args.diffuser.returns_condition,
            dim=2 * args.diffuser.dim,
            condition_dropout=args.diffuser.condition_dropout,
            calc_energy=args.diffuser.calc_energy,
            device=device,
        )

        diffusion_config = diffutils.Config(
            args.diffuser.diffusion,
            savepath=os.path.join(diffuser_save_path, 'diffusion_config.pkl'),
            horizon=args.diffuser.horizon,
            # observation_dim=observation_dim,
            # action_dim=action_dim,
            observation_dim=args.dt.hidden_size,  # TODO
            action_dim=args.env.action_dim,  # TODO
            n_timesteps=args.diffuser.n_diffusion_steps,
            loss_type=args.diffuser.loss_type,
            clip_denoised=args.diffuser.clip_denoised,
            predict_epsilon=args.diffuser.predict_epsilon,
            ## loss weighting
            action_weight=args.diffuser.action_weight,
            loss_weights=args.diffuser.loss_weights,
            loss_discount=args.diffuser.loss_discount,
            returns_condition=args.diffuser.returns_condition,
            condition_guidance_w=args.diffuser.condition_guidance_w,
            device=device,
        )

    diff_trainer_args = dict(
        savepath=os.path.join(diffuser_save_path, 'trainer_config.pkl'),
        train_batch_size=args.diffuser.batch_size,  # TODO
        train_lr=args.diffuser.learning_rate,
        gradient_accumulate_every=args.diffuser.gradient_accumulate_every,
        ema_decay=args.diffuser.ema_decay,
        sample_freq=args.diffuser.sample_freq,
        save_freq=args.diffuser.save_freq,
        log_freq=args.diffuser.log_freq,
        label_freq=int(args.diffuser.n_train_steps // args.diffuser.n_saves),
        save_parallel=args.diffuser.save_parallel,
        bucket=args.diffuser.bucket,
        n_reference=args.diffuser.n_reference,
        train_device=device,
        save_checkpoints=args.diffuser.save_checkpoints,
        update_ema_every=10,
        parallel=args.parallel,
    )

    hrl_model_args = dict(args.model)
    iq_args = args.iq

    unet = model_config()
    diffusion_model = diffusion_config(unet)
    diff_trainer = DiffTrainer(args, diffusion_model, diff_trainer_args)

    model = HRLModel(args, option_selector_args, state_reconstructor_args,
                     lang_reconstructor_args, decision_args, iq_args, diff_trainer, device, **hrl_model_args)

    # # load saved arguments
    # checkpoint = torch.load(cfg.checkpoint_path)
    # args = checkpoint['config']
    # max_length = checkpoint['train_dataset_max_length']
    # args.eval = cfg.eval
    # args.render = cfg.render
    # args.checkpoint_path = cfg.checkpoint_path
    # device = cfg.trainer.device
    #
    # # Set num train_trajs to something small
    # args.train_dataset.num_trajectories = 1000

    start_iter = 1
    if args.resume:
        args.warmup_steps = 0
        #checkpoint = trainer.load(args.checkpoint_path)
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        model = model.to(device)
        # checkpoint['model']['diffuser.embed_ln.weight'] = checkpoint['model']['diffuser.embed_ln.weight']
        # checkpoint['model']['diffuser.embed_ln.bias'] = checkpoint['model']['diffuser.embed_ln.bias']
        model.load_state_dict(checkpoint['model'], strict=False)
        # start_iter = checkpoint['iter_num'] + 1
        assert train_dataset.max_length == checkpoint[
            'train_dataset_max_length'], f"Expected max length of dataset to be {train_dataset.max_length} but got {checkpoint['train_dataset_max_length']}"

        if ".pt" in args.diffuser.loadpath:
            model.diff_trainer.load(args.diffuser.loadpath)

        del checkpoint
        torch.cuda.empty_cache()

    else:
        model = model.to(device) # 先将模型移到指定设备
        
    # if ".pt" in args.diffuser.loadpath:
    #     model.diff_trainer.load(args.diffuser.loadpath)

    if args.load_options:
        checkpoint = torch.load(args.checkpoint_path)
        checkpoint = checkpoint['model']
        # pdb.set_trace()
        state_dict = {k:v for k,v in checkpoint.items() if k.startswith('option_selector.Z')}
        loaded = model.load_state_dict(state_dict, strict=False)
        # assert loaded.unexpected_keys == []   ## simple check
        if args.freeze_loaded_options:
            for name, param in model.named_parameters():
                if name.startswith('option_selector.Z'):
                    param.requires_grad = False
            assert not model.option_selector.Z.project_out.bias.requires_grad   ## simple check
    
    model.count_parameters()

    # if args.load_frozen:
    #     checkpoint = torch.load(args.checkpoint_path)
    #     checkpoint = checkpoint['model']
    #     state_dict = {k:v for k,v in checkpoint.items() if not k.startswith('decision_transformer.')}
    #     loaded = model.load_state_dict(state_dict, strict=False)
    #     assert loaded.unexpected_keys == []   ## simple check
    #     for name, param in model.named_parameters():
    #         if not name.startswith('diffuser.'):
    #             param.requires_grad = False

    # ===== 包装模型 =====
    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if local_rank >= 0 else None,
            output_device=local_rank if local_rank >= 0 else None,
            find_unused_parameters=True
        )
    elif args.parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Setting up the optimizer
    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    # setting different learning rates for the LM part, OS part and other parts
    lm_params = {'params': [v for k, v in params if k.startswith('lm.') or k.startswith('module.lm.')], 'lr': args.lm_learning_rate}
    os_params = {'params': [v for k, v in params if k.startswith('option_selector.') or k.startswith('module.option_selector.')], 'lr': args.os_learning_rate}
    other_params = {'params': [v for k, v in params if not k.startswith(
        'lm.') and not k.startswith('option_selector.')
        and not k.startswith('module.lm.') and not k.startswith('module.option_selector.')]}
    # for the option selector need separate lr?
    optimizer = torch.optim.AdamW(
        [other_params, lm_params, os_params],
        lr=args.learning_rate, weight_decay=args.weight_decay,)

    def adjust_lr(steps):
        if steps < args.warmup_steps:
            return min((steps + 1) / args.warmup_steps, 1)
        num_decays = (steps + 1) // args.decay_steps
        return args.lr_decay ** (num_decays)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, adjust_lr)
    # optimizer, scheduler = None, None

    # planner = RoboBrainPlanner()
    trainer_args = dict(args.trainer)
    trainer = Trainer(
        args=args,
        model=model,
        # planner=planner,
        tokenizer=tokenizer,
        optimizer=optimizer,
        train_loader=train_loader,
        env=env,
        env_name=env_name,
        val_loader=val_loader,
        scheduler=scheduler,
        eval_episode_factor=2,
        skip_words=args.env.skip_words,
        **trainer_args
    )

    # Training loop
    for iter_num in range(start_iter, start_iter + args.max_iters):
        outputs = trainer.train_iteration(
            iter_num=iter_num, print_logs=True, eval_render=args.render)

        if args.wandb and iter_num % args.log_interval == 0:
            wandb.log(outputs, step=iter_num)

        if iter_num % args.save_interval == 0:
            if local_rank == 0:
                trainer.save(iter_num, f'{args.savepath}/model_{iter_num}.ckpt', args)

        # break

def get_args(cfg: DictConfig):
    if torch.cuda.is_available():
        local_rank_env = os.environ.get('LOCAL_RANK')
        if local_rank_env is not None:
            # DDP 环境：每个进程使用其指定的 GPU
            local_rank = int(local_rank_env)
            cfg.trainer.device = f"cuda:{local_rank}"
            print(f"Multi-GPU (DDP): Process using device cuda:{local_rank}")
        else:
            # 单 GPU 或 DataParallel (使用 cuda:0 作为主设备)
            cfg.trainer.device = "cuda:0"
            if torch.cuda.device_count() > 1:
                print(f"Multi-GPU (DataParallel): Using cuda:0 as primary. Found {torch.cuda.device_count()} GPUs.")
            else:
                print("Single GPU: Using cuda:0.")
    else:
        cfg.trainer.device = "cpu"
        print("CPU: No CUDA GPUs available.")
    # cfg.hydra_base_dir = os.getcwd()
    # tmp_path = os.path.split(cfg.hydra_base_dir)
    return cfg


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):

    cfg.hydra_base_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    args = get_args(cfg)

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if 'cuda' in args.trainer.device and os.environ.get('LOCAL_RANK') is not None:
        torch.distributed.init_process_group(backend='nccl')
    
    # init_logger()

    device = torch.device(args.trainer.device)
    if device.type == 'cuda' and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    elif device.type == 'cuda' and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # logger.info("--> Running in ", os.getcwd())

    if args.eval:
        evaluate(cfg)
        return

    # train
    print(OmegaConf.to_yaml(cfg))
    train(args)


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"]="4"
    main()
