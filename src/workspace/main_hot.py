if __name__ == "__main__":
    import sys
    import os
    import pathlib

    current_file = pathlib.Path(__file__)
    
    # ROOT_DIR = .../SkillAttention
    ROOT_DIR = str(current_file.parent.parent.parent)
    
    # SRC_DIR = .../SkillAttention/src
    SRC_DIR = str(current_file.parent.parent.parent / "src")

    sys.path.append(ROOT_DIR)
    sys.path.append(SRC_DIR)
    os.chdir(ROOT_DIR)

import os
os.environ["TZ"] = "Asia/Shanghai"

# import gym  # can be reused later if connecting to env
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import logging
import wandb
import hydra
import random
import builtins  # ✅ for monkey-patching print

from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer

from src.policy.diffusion_unet_hybrid_image_policy import SkillAttentionUnetModel
# from src.policy.diffusion_unet_with_lang import SkillAttentionUnetLangModel
from src.policy.diffusion_unet_lorel_policy import SkillAttentionUnetLorelModel
# from src.diffusion_transformer_hybrid_image_policy import SkillAttentionTransformerModel
from src.policy.diffusion_transformer_with_lang import SkillAttentionTransformerLangModel
# from src.policy.diffusion_transformer_lorel_policy import SkillAttentionTransformerLorelModel
from src.policy.diffusion_transformer_lorel_with_lang import SkillAttentionTransformerLorelLangModel

from torch.utils.data import DataLoader
from dataset.expert_dataset import ExpertDataset
from dataset.expert_dataset_realman import RealmanExpertDataset


# ----------------- DDP / helper functions -----------------
def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0

def is_main_process() -> bool:
    return get_rank() == 0

def get_visible_gpus():
    """Estimate number of visible GPUs from CUDA_VISIBLE_DEVICES or torch.cuda.device_count"""
    env = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if env is not None and env.strip() != "":
        # e.g. "0,1" -> 2
        return len([x for x in env.split(",") if x.strip() != ""])
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0

# ----------------- Only allow print on main process (also affects prints in other modules) -----------------
def setup_rank0_print():
    """
    Override global print:
      - non-distributed / single-process: unchanged
      - torchrun scenario: only the process with LOCAL_RANK=0 actually prints; other processes' prints are ignored
    """
    orig_print = builtins.print

    def rank0_print(*args, **kwargs):
        local_rank_env = os.environ.get("LOCAL_RANK", None)
        # If not using torchrun (no LOCAL_RANK) or local_rank==0: print normally
        if local_rank_env is None or int(local_rank_env) == 0:
            orig_print(*args, **kwargs)

    builtins.print = rank0_print

# ----------------- logging & helpers -----------------
def create_global_logger(log_file=None, log_level=logging.INFO):
    """
    Create a global logger:
      - main process (rank0): output to console + (optional) file
      - other ranks: do not add handlers, logs are effectively suppressed
    """
    if is_distributed() and not is_main_process():
        os.environ.setdefault("WANDB_INIT_DISABLED", "true")

    logger = logging.getLogger("main")

    # Clear old handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()

    if is_main_process():
        logger.setLevel(log_level)
        formatter = logging.Formatter("%(asctime)s  %(levelname)5s  %(message)s")

        console = logging.StreamHandler()
        console.setLevel(log_level)
        console.setFormatter(formatter)
        logger.addHandler(console)

        if log_file is not None:
            file_handler = logging.FileHandler(filename=log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    else:
        logger.setLevel(logging.CRITICAL + 1)

    logger.propagate = False
    return logger


def wandb_sample_image(train_loader, memmap_dir, max_images=6):
    
    batch = next(iter(train_loader))
    _, obs, _, _, _, _ = batch

    head = obs['head_image']   # [B, K, 3, H, W]
    wrist = obs['wrist_image'] # [B, K, 3, H, W]

    B, K, C, H, W = head.shape
    t0 = 0  # which frame to visualize, can change

    # load scale and offset for inverse normalization
    head_offset = np.load(os.path.join(memmap_dir, "head_offset.npy")).astype(np.float32)
    head_scale = np.load(os.path.join(memmap_dir, "head_scale.npy")).astype(np.float32)

    wrist_offset = np.load(os.path.join(memmap_dir, "wrist_offset.npy")).astype(np.float32)
    wrist_scale = np.load(os.path.join(memmap_dir, "wrist_scale.npy")).astype(np.float32)

    # reshape for broadcasting (C,1,1)
    head_offset = head_offset.reshape(C, 1, 1)
    head_scale = head_scale.reshape(C, 1, 1)
    wrist_offset = wrist_offset.reshape(C, 1, 1)
    wrist_scale = wrist_scale.reshape(C, 1, 1)

    images_to_log = []
    n = min(max_images, B)

    for i in range(n):
        # take i-th sample and t0 frame, shape 3xHxW, move to cpu + numpy
        head_chw  = head[i, t0].cpu().numpy().astype(np.float32)   # [3, H, W]
        wrist_chw = wrist[i, t0].cpu().numpy().astype(np.float32)  # [3, H, W]

        # inverse normalization: (x - offset) / scale
        head_chw  = (head_chw - head_offset) / head_scale
        wrist_chw = (wrist_chw - wrist_offset) / wrist_scale

        # clip pixels to [0, 1]
        head_chw  = np.clip(head_chw, 0.0, 1.0)
        wrist_chw = np.clip(wrist_chw, 0.0, 1.0)

        # CHW -> HWC
        head_hwc  = np.transpose(head_chw,  (1, 2, 0))  # [H, W, 3]
        wrist_hwc = np.transpose(wrist_chw, (1, 2, 0))  # [H, W, 3]

        # concatenate along width: shape [H, W_head + W_wrist, 3]
        cat = np.concatenate([head_hwc, wrist_hwc], axis=1)
        images_to_log.append(
            wandb.Image(cat, caption=f"sample {i}, t={t0}")
        )

    if images_to_log and wandb.run is not None:
        wandb.log({"camera_views": images_to_log}, step=0)

def wandb_sample_lorel_image(train_loader, memmap_dir, max_images=6):
    batch = next(iter(train_loader))
    _, obs, _, _, _, _ = batch

    im = obs['images']   # [B, K, 3, H, W]
    B, K, C, H, W = im.shape
    t0 = 0  # which frame to visualize

    image_scale = np.load(os.path.join(memmap_dir, "image_scale.npy")).astype(np.float32)
    image_offset = np.load(os.path.join(memmap_dir, "image_offset.npy")).astype(np.float32)

    image_scale = image_scale.reshape(C, 1, 1)
    image_offset = image_offset.reshape(C, 1, 1)

    images_to_log = []
    n = min(max_images, B)

    for i in range(n):
        im_chw = im[i, t0].cpu().numpy().astype(np.float32)   # [3, H, W]

        im_chw = (im_chw - image_offset) / image_scale
        im_chw  = np.clip(im_chw, 0.0, 1.0)

        # CHW -> HWC
        im_hwc  = np.transpose(im_chw,  (1, 2, 0))  # [H, W, 3]
        images_to_log.append(
            wandb.Image(im_hwc, caption=f"sample {i}, t={t0}")
        )

    if images_to_log and wandb.run is not None:
        wandb.log({"camera_views": images_to_log}, step=0)


def _disable_file_logging_for_non_rank0():
    if not is_distributed() or is_main_process():
        return

    # Remove FileHandler for non-main processes (compatible with other loggers)
    for name in ["", "root", "hydra", "main"]:
        lg = logging.getLogger(name)
        for h in list(lg.handlers):
            if isinstance(h, logging.FileHandler):
                try:
                    lg.removeHandler(h)
                    h.close()
                except Exception:
                    pass

def get_args(cfg: DictConfig):
    """Set trainer.device based on CUDA_VISIBLE_DEVICES and DDP environment"""
    num_visible_gpus = get_visible_gpus()

    if num_visible_gpus > 0:
        local_rank_env = os.environ.get("LOCAL_RANK")

        if num_visible_gpus > 1 and local_rank_env is not None:
            # DDP: each process uses its own GPU
            local_rank = int(local_rank_env)
            cfg.trainer.device = f"cuda:{local_rank}"
            # This print will be controlled by setup_rank0_print later; only rank0 will actually print
            print(f"DDP: num_visible_gpus={num_visible_gpus}, process using device cuda:{local_rank}")
        else:
            # single-process training
            cfg.trainer.device = "cuda:0"
            if num_visible_gpus > 1:
                print(f"Single process with {num_visible_gpus} visible GPUs: using cuda:0 (no DDP).")
            else:
                print("Single GPU: Using cuda:0.")
    else:
        cfg.trainer.device = "cpu"
        print("CPU: No CUDA GPUs available.")

    return cfg

# ----------------- RealMan-specific dataset + dataloader -----------------
def build_realman_dataset_and_loader(args, batch_size):
    logger = logging.getLogger("main")
    device = args.trainer.device
    is_ddp = is_distributed()

    trainset_cfg = args.task
    horizon = trainset_cfg.horizon
    memmap_dir = trainset_cfg.memmap_dir

    common_kwargs = dict(
        expert_locations    = trainset_cfg.expert_location,
        subsample_frequency= getattr(trainset_cfg, "subsample_frequency", 1),
        seed               = args.seed,
        normalize_obs      = getattr(trainset_cfg, "normalize_obs", True),
        no_lang            = getattr(trainset_cfg, "no_lang", False),
        memmap_dir         = memmap_dir,
        task_name          = trainset_cfg.task_lang,
        chunk_horizon      = horizon,
        image_shape        = args.model.obs_encoder.resize_shape,
    )

    if is_ddp:
        if is_main_process():
            logger.info(f"[Rank 0] Building/attaching RealMan memmap at {memmap_dir} ...")
            train_dataset = RealmanExpertDataset(
                **common_kwargs,
                writer=True,
                force_rebuild=trainset_cfg.force_rebuild,
            )
        dist.barrier()
        if not is_main_process():
            train_dataset = RealmanExpertDataset(
                **common_kwargs,
                writer=False,
                force_rebuild=False,
            )
        dist.barrier()
    else:
        logger.info("[Single process] Building/attaching RealMan memmap ...")
        train_dataset = RealmanExpertDataset(
            **common_kwargs,
            writer=True,
            force_rebuild=trainset_cfg.force_rebuild,
        )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if is_ddp else None
    shuffle = train_sampler is None

    num_workers = 32
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=("cuda" in device),
        drop_last=True,
    )
    return train_dataset, train_loader, train_sampler

def build_lorel_datatset_and_dataloader(args, batch_size):
    logger = logging.getLogger("main")
    device = args.trainer.device
    is_ddp = is_distributed()

    trainset_cfg = args.task
    horizon = trainset_cfg.horizon
    memmap_dir = trainset_cfg.memmap_dir

    common_kwargs = dict(
        expert_location    = trainset_cfg.expert_location,
        memmap_dir         = memmap_dir,
        num_trajectories   = trainset_cfg.num_trajectories,
        subsample_frequency= getattr(trainset_cfg, "subsample_frequency", 1),
        seed               = args.seed,
        normalize_obs      = getattr(trainset_cfg, "normalize_obs", True),
        no_lang            = getattr(trainset_cfg, "no_lang", False),
        task_name          = trainset_cfg.task_lang,
        chunk_horizon      = horizon,
    )

    if is_ddp:
        if is_main_process():
            logger.info(f"[Rank 0] Building/attaching RealMan memmap at {memmap_dir} ...")
            train_dataset = ExpertDataset(
                **common_kwargs,
                writer=True,
                force_rebuild=trainset_cfg.force_rebuild,
            )
        dist.barrier()
        if not is_main_process():
            train_dataset = ExpertDataset(
                **common_kwargs,
                writer=False,
                force_rebuild=False,
            )
        dist.barrier()
    else:
        logger.info("[Single process] Building/attaching RealMan memmap ...")
        train_dataset = ExpertDataset(
            **common_kwargs,
            writer=True,
            force_rebuild=trainset_cfg.force_rebuild,
        )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if is_ddp else None
    shuffle = train_sampler is None

    num_workers = 32
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=("cuda" in device),
        drop_last=True,
    )
    return train_dataset, train_loader, train_sampler

# ----------------- Training -----------------
def train(args: DictConfig):
    logger = logging.getLogger("main")

    device = args.trainer.device
    is_ddp = is_distributed()
    global_rank = get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", -1)) if is_ddp else -1

    logger.info(f"rank: {global_rank}, world_size: {dist.get_world_size() if is_ddp else 1}")
    logger.info(f"local_rank: {local_rank}, num_gpus: {torch.cuda.device_count()}")

    args.trainer.ckt_save_path = os.path.join(
        args.hydra_base_dir,
        "checkpoints",           
    )
    if global_rank == 0 and not os.path.isdir(args.trainer.ckt_save_path):
        os.makedirs(args.trainer.ckt_save_path, exist_ok=True)

    if 'lorel' in args.task.name:
        # ==== LoReL dataset & DataLoader ====
        train_dataset, train_loader, train_sampler = build_lorel_datatset_and_dataloader(
            args=args,
            batch_size=args.trainer.batch_size
        )
    else:
        # ==== RealMan dataset & DataLoader ====
        train_dataset, train_loader, train_sampler = build_realman_dataset_and_loader(
            args=args,
            batch_size=args.trainer.batch_size
        )

    total_timesteps = int(train_dataset.trajectories["lengths"].sum())
    logger.info("=" * 50)
    logger.info(f"Starting experiment: {args.env.name}")
    logger.info(f"{len(train_dataset)} chunks (length == horizon={args.model.horizon}), "
                f"{total_timesteps} timesteps in total")
    logger.info("=" * 50)

    # ==== Build SkillAttentionModel ====
    memmap_dir = train_dataset.memmap_dir
    if 'lorel' in args.task.name:
        if args.diffuser.backbone == "transformer":
            base_model = SkillAttentionTransformerLorelLangModel(
                policy_args=args.model,
                diff_args=args.diffuser,
                device=device,
                memmap_dir=memmap_dir,
            ).to(device)
        elif args.diffuser.backbone == "conditional_unet":
            base_model = SkillAttentionUnetLorelModel(
                policy_args=args.model,
                diff_args=args.diffuser,
                device=device,
                memmap_dir=memmap_dir,
            ).to(device)
        else:
            raise ValueError(f"Unknown diffuser.backbone: {args.diffuser.backbone}")
        
    else:
        if args.diffuser.backbone == "transformer":
            base_model = SkillAttentionTransformerLangModel(
                policy_args=args.model,
                diff_args=args.diffuser,
                device=device,
                memmap_dir=memmap_dir,
            ).to(device)
        elif args.diffuser.backbone == "conditional_unet":
            base_model = SkillAttentionUnetModel(
                policy_args=args.model,
                diff_args=args.diffuser,
                device=device,
                memmap_dir=memmap_dir,
            ).to(device)
        else:
            raise ValueError(f"Unknown diffuser.backbone: {args.diffuser.backbone}")

    # ===== DDP / DataParallel wrapping =====
    model = base_model
    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if local_rank >= 0 else None,
            output_device=local_rank if local_rank >= 0 else None,
            find_unused_parameters=True,
        )
    elif torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # ===== optimizer / scheduler =====
    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    lm_params = {"params": [v for k, v in params if k.startswith("lm.") or k.startswith("module.lm.")],
                 "lr": args.trainer.lm_learning_rate}
    os_params = {"params": [v for k, v in params
                            if k.startswith("option_selector.") or k.startswith("module.option_selector.")],
                 "lr": args.trainer.os_learning_rate}
    other_params = {
        "params": [
            v for k, v in params
            if not k.startswith("lm.")
            and not k.startswith("option_selector.")
            and not k.startswith("module.lm.")
            and not k.startswith("module.option_selector.")
        ]
    }

    optimizer = torch.optim.AdamW(
        [other_params, lm_params, os_params],
        lr=args.trainer.learning_rate,
        weight_decay=args.trainer.weight_decay,
        betas=tuple(args.trainer.betas)
    )

    def adjust_lr(steps):
        if steps < args.trainer.warmup_steps:
            return min((steps + 1) / args.trainer.warmup_steps, 1.0)
        num_decays = (steps + 1) // args.trainer.decay_steps
        return args.trainer.lr_decay ** num_decays

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, adjust_lr)

    trainer_args = dict(args.trainer)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        scheduler=scheduler,
        **trainer_args,
    )

    # ===== resume logic =====
    start_iter = 1
    if args.resume:
        args.trainer.warmup_steps = 0
        # Use Trainer.load to restore: model + optimizer + scheduler + diffusion_step
        meta = trainer.load(args.trainer.checkpoint_path)
        start_iter = meta.get("iter_num", 0) + 1

        # ===== Check total step budget: main model & diffusion part =====
        if start_iter > args.trainer.max_iters:
            logger.info(f"[Main] checkpoint iter {start_iter-1} 已经 >= max_iters={args.trainer.max_iters}，直接退出。")
            return

    # ===== Compose experiment name using task_name =====
    task_tag = getattr(args.model, "task_name", "all")   # e.g. "pick_up_the_green_cube" / "all"
    horizon   = args.model.horizon
    exp_name = f"{task_tag}-H{horizon}"  # RealMan experiment name: task + horizon

    # ----------------- wandb: record run_id using a txt file -----------------
    if args.wandb and global_rank == 0:
        robot_name = "Lorel_Sawyer" if 'lorel' in args.task.name else "RealMan" 
        wandb_project = f"{robot_name}-{args.project_name}"
        wandb_dir = os.path.join(args.hydra_base_dir, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        run_id_file = os.path.join(wandb_dir, "run_id.txt")

        run_id = None
        if args.resume and os.path.exists(run_id_file):
            with open(run_id_file, "r") as f:
                run_id = f.read().strip() or None

        if run_id is None:
            run_id = wandb.util.generate_id()
            with open(run_id_file, "w") as f:
                f.write(run_id)

        wandb.init(
            project=wandb_project,
            name=exp_name,
            id=run_id,
            resume="allow",
            group=args.model.name,
            config=dict(args),
            dir=wandb_dir,          # Lock wandb's directory to hydra_base_dir/wandb
        )

    if args.wandb and global_rank == 0:
        if 'lorel' in args.task.name:      
            wandb_sample_lorel_image(train_loader, memmap_dir=memmap_dir, max_images=6,)  
        else:
            wandb_sample_image(train_loader, memmap_dir=memmap_dir, max_images=6,)

    # ===== Training loop =====
    for iter_num in range(start_iter, args.trainer.max_iters + 1):
        outputs = trainer.train_iteration(
            iter_num=iter_num,
            print_logs=True,
        )

        if global_rank == 0:
            last_checkpoint_path = os.path.join(args.trainer.ckt_save_path, f"model_{iter_num - 1}.ckpt")
            if os.path.exists(last_checkpoint_path) and ((iter_num - 1) % args.trainer.save_interval != 0):
                os.remove(last_checkpoint_path)

            ckpt_path = os.path.join(args.trainer.ckt_save_path, f"model_{iter_num}.ckpt")
            trainer.save(iter_num, ckpt_path, args)

        if args.wandb and global_rank == 0 and iter_num % args.trainer.log_interval == 0:
            wandb.log(outputs, step=iter_num)


def resolve_run_dir(resume, checkpoint_path, outputs_root, task_lang, date, time):
    """Resolve Hydra output directory; reuse checkpoint parent directory when resuming."""
    resume_flag = str(resume).lower() in ("1", "true", "yes")
    if resume_flag and checkpoint_path:
        return os.path.abspath(os.path.join(os.path.dirname(checkpoint_path), ".."))
    return os.path.join(outputs_root, task_lang, date, time)


OmegaConf.register_new_resolver("resolve_run_dir", resolve_run_dir, replace=True)


@hydra.main(version_base=None, config_path="../conf", config_name="train_lorel_sawyer")
def main(cfg: DictConfig):
    setup_rank0_print()

    if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    try:
        cfg = get_args(cfg)
        cfg.hydra_base_dir = HydraConfig.get().runtime.output_dir
        os.makedirs(cfg.hydra_base_dir, exist_ok=True)

        log_file = os.path.join(cfg.hydra_base_dir, "train.log")
        logger = create_global_logger(log_file=log_file)
        _disable_file_logging_for_non_rank0()

        if is_main_process():
            logger.info(f"Hydra output dir: {cfg.hydra_base_dir}")
            logger.info(OmegaConf.to_yaml(cfg))

        train(cfg)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
