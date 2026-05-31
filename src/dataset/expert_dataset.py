from typing import Any, Dict, IO, List, Tuple, Union, Optional
import logging
import re
import numpy as np
import pandas as pd
import pickle
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import json
from PIL import Image
import gc

def load_trajectories(expert_location: str,
                      num_trajectories: int = 10,
                      seed: int = 0,
                      **kwargs) -> Dict[str, Any]:
    """Load expert trajectories
    Args:
        expert_location:          Location of saved expert trajectories.
        num_trajectories:         Number of expert trajectories to sample (randomized).
        deterministic:            If true, random behavior is switched off.
    Returns:
        Dict containing keys {"states", "lengths"} and optionally {"actions", "rewards", "language", "dones"} with values
        containing corresponding expert data attributes.
    """
    if os.path.isfile(expert_location) or os.path.isdir(expert_location):
        # Load data from single file.
        if 'babyai' in expert_location:
            trajs = load_babyai_data(expert_location, num_trajectories, seed, **kwargs)
            # BabyAI does the random shuffling and taking subset for us
            return trajs
        elif 'lorel' in expert_location or 'lorel' in str(kwargs.get('task_name', '')).lower() or 'lorel' in str(kwargs.get('name', '')).lower():
            trajs = load_lorel_data(expert_location, **kwargs)
        elif 'calvin' in expert_location:
            trajs = load_calvin_data(expert_location, num_trajectories, seed, **kwargs)
        else:
            with open(expert_location, 'rb') as f:
                trajs = read_file(expert_location, f)

            # Add empty strings if no language instructions in the trajectories
            if "language" not in trajs:
                trajs["language"] = ["" for i in range(len(trajs["states"]))]

        rng = np.random.RandomState(seed)
        # Sample random `num_trajectories` experts.
        perm = np.arange(len(trajs["actions"]))
        perm = rng.permutation(perm)

        idx = perm[:num_trajectories]
        for k, v in trajs.items():
            # if not torch.is_tensor(v):
            #     v = np.array(v)  # convert to numpy array
            trajs[k] = [v[i] for i in idx]
    else:
        raise ValueError(f"{expert_location} is not a valid path")
    return trajs


def read_file(path: str, file_handle: IO[Any]) -> Dict[str, Any]:
    """Read file from the input path. Assumes the file stores dictionary data.
    Args:
        path:               Local or S3 file path.
        file_handle:        File handle for file.
    Returns:
        The dictionary representation of the file.
    """
    if path.endswith("pt"):
        data = torch.load(file_handle)
    elif path.endswith("pkl"):
        data = pickle.load(file_handle)
    elif path.endswith("npy"):
        data = np.load(file_handle, allow_pickle=True)
        if data.ndim == 0:
            data = data.item()
    else:
        raise NotImplementedError
    return data


def load_babyai_data(expert_location, num_trajs, seed, **kwargs):
    from babyai.utils.demos import load_demos, transform_demos
    demos = transform_demos(load_demos(expert_location), num_trajs, seed)
    trajs = {"states": [], "actions": [], "rewards": [], "lengths": [], "language": [], "dones": []}
    for demo in tqdm(demos):
        trajs["language"].append(demo[0][0]["mission"])
        trajs["lengths"].append(len(demo))
        traj_states, traj_actions, traj_rewards, traj_dones = [], [], [], []
        for step in demo:
            obs, action, done = step
            if kwargs['use_direction']:
                direction = np.zeros(4)
                direction[obs["direction"]] = 1.
                state = np.concatenate([obs["image"].reshape(-1), direction])
            else:
                state = obs["image"].reshape(-1)
            traj_states.append(state)
            traj_actions.append(action)
            traj_rewards.append(None)
            traj_dones.append(done)
        trajs["states"].append(np.array(traj_states))
        trajs["actions"].append(np.array(traj_actions))
        trajs["rewards"].append(np.array(traj_rewards))
        trajs["dones"].append(np.array(traj_dones))

    assert len(trajs["states"]) == len(trajs["actions"]) == len(trajs["rewards"]) == len(
        trajs["lengths"]) == len(trajs["language"]) == len(trajs["dones"])
    return trajs


def load_lorel_data(expert_location, **kwargs):
    trajs = {"proprio": [], "images":[], "actions": [], "rewards": [], "lengths": [], "language": [], "dones": []}
    data = pickle.load(open(expert_location, 'rb'))
    num_trajs, traj_len = data['actions'].shape[0], data['actions'].shape[1]
    if "state" in data.keys():
        trajs["proprio"] = data["state"] 

    trajs["images"] = np.moveaxis(data["ims"], 4, 2)  # making images C,H,W
    trajs["actions"] = data["actions"]
    trajs["rewards"] = np.array([[None for _ in range(traj_len)] for _ in range(num_trajs)])
    trajs["lengths"] = np.array([traj_len]*num_trajs)
    trajs["dones"] = np.array([[0]*(traj_len-1) + [1] for _ in range(num_trajs)])

    if 'sawyer' in expert_location:
        trajs["language"] = data['langs'].reshape(-1)
    elif 'franka' in expert_location:
        trajs["images"] = np.concatenate([trajs["images"], trajs["images"]], axis=0)
        trajs["actions"] = np.concatenate([trajs["actions"], trajs["actions"]], axis=0)
        trajs["rewards"] = np.concatenate([trajs["rewards"], trajs["rewards"]], axis=0)
        trajs["lengths"] = np.concatenate([trajs["lengths"], trajs["lengths"]], axis=0)
        trajs["dones"] = np.concatenate([trajs["dones"], trajs["dones"]], axis=0)
        trajs["language"] = data['langs'].T.reshape(-1)
    else:
        raise NotImplementedError

    assert len(
        trajs["proprio"]) == len(
        trajs["images"]) == len(
        trajs["actions"]) == len(
        trajs["rewards"]) == len(
        trajs["lengths"]) == len(
        trajs["language"]) == len(
        trajs["dones"])
    return trajs


def load_calvin_data(expert_location, num_trajs, seed, **kwargs):
    trajs = {"states": [], "actions": [], "rewards": [], "lengths": [], "language": [], "dones": []}
    lang_anns = np.load(f'{expert_location}/lang_annotations/auto_lang_ann.npy', allow_pickle=True).item()
    np.random.seed(seed)
    chosen_traj_inds = np.random.choice(len(lang_anns['info']['indx']), size=num_trajs, replace=False)
    for i in chosen_traj_inds:
        start_idx, end_idx = lang_anns['info']['indx'][i]
        states, actions = [], []
        length = end_idx - start_idx + 1
        rewards = np.array([None] * length)
        language = lang_anns['language']['ann'][i]
        dones = np.array([0] * (length-1) + [1])
        for idx in range(start_idx, end_idx+1):
            info = np.load(f'{expert_location}/episode_{str(idx).zfill(7)}.npz')
            states.append(np.moveaxis(info['rgb_static'], 2,0))
            actions.append(info['actions'])
        assert len(states) == len(actions) == length
        states, actions = np.array(states), np.array(actions)
        trajs['states'].append(states)
        trajs['actions'].append(actions)
        trajs['rewards'].append(rewards)
        trajs['lengths'].append(length)
        trajs['language'].append(language)
        trajs['dones'].append(dones)
    assert len(
        trajs["states"]) == len(
        trajs["actions"]) == len(
        trajs["rewards"]) == len(
        trajs["lengths"]) == len(
        trajs["language"]) == len(
        trajs["dones"])
    return trajs


class ExpertDataset(Dataset):
    """
    Keys:
        language: [num_experts]
        proprio:  [num_experts, max_length, proprio_dim]     (flatten -> [total_T, proprio_dim])
        images:   [num_experts, max_length, C, H, W]         (flatten -> [total_T, C, H, W])
        actions:  [num_experts, max_length, action_dim]
        rewards:  [num_experts, max_length]
        lengths:  [num_experts]
        dones:    [num_experts, max_length]
    """

    def __init__(self,
                 expert_location: str,
                 memmap_dir: str,
                 num_trajectories: int = 1,
                 subsample_frequency: int = 1,
                 seed: int = 0,
                 full_traj: bool = True,
                 normalize_obs: bool = True,  
                 normalize_actions: bool =True,
                 no_lang: bool = False,
                 writer: bool = False,
                 force_rebuild: bool = False,
                 chunk_horizon: Optional[int] = 8,  
                 **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.full_traj = full_traj
        self.normalize_obs = normalize_obs
        self.normalize_actions = normalize_actions
        self.no_lang = no_lang
        self.chunk_horizon = chunk_horizon
        self.memmap_dir = memmap_dir

        if 'aug' in self.kwargs and self.kwargs['aug'] and transforms is not None:
            # From the LORL paper
            self.aug = torch.nn.Sequential(
                transforms.ColorJitter(brightness=0.02, contrast=0.02, saturation=0.02, hue=0.02),
                transforms.RandomAffine(20, translate=(0.1, 0.1), scale=(0.9, 1.1))
            )
        else:
            self.aug = None

        # Memmap file paths
        os.makedirs(self.memmap_dir, exist_ok=True)
        self.meta_path = os.path.join(self.memmap_dir, "meta.json")
        self.lengths_path = os.path.join(self.memmap_dir, "lengths.npy")
        self.lang_path = os.path.join(self.memmap_dir, "language.json")
        
        self.proprio_scale_path = os.path.join(self.memmap_dir, "proprio_scale.npy")
        self.proprio_offset_path = os.path.join(self.memmap_dir, "proprio_offset.npy")

        self.image_scale_path = os.path.join(self.memmap_dir, "image_scale.npy")
        self.image_offset_path = os.path.join(self.memmap_dir, "image_offset.npy")

        self.action_scale_path = os.path.join(self.memmap_dir, "action_scale.npy")
        self.action_offset_path = os.path.join(self.memmap_dir, "action_offset.npy")

        self.proprio_path = os.path.join(self.memmap_dir, "proprio.mmap")
        self.images_path  = os.path.join(self.memmap_dir, "images.mmap")   
        self.actions_path = os.path.join(self.memmap_dir, "actions.mmap")
        self.rewards_path = os.path.join(self.memmap_dir, "rewards.mmap")
        self.dones_path   = os.path.join(self.memmap_dir, "dones.mmap")

        # Rank0 build memmap (added chunk processing)
        if writer and (force_rebuild or not self._memmap_exists()):
            logger = logging.getLogger("main")
            logger.info(f"Memmap cache is missing or force_rebuild=True, rebuilding at {self.memmap_dir}")
            logger.info(f"Loading trajectories from {expert_location} ...")
            all_traj = self._load_and_subsample(
                expert_location, num_trajectories, subsample_frequency, seed, **kwargs
            )
            logger.info(f"Loaded {len(all_traj['actions'])} trajectories.")
            logger.info("Inferring normalization dimensions ...")
            self._infer_normalize_dim(expert_location, all_traj, **kwargs)
            logger.info("Computing normalization statistics before chunking ...")
            self._compute_stats(all_traj)
            # Added: chunk trajectory splitting. Statistics can be computed before
            # chunking because chunking only splits/reorders the same timesteps.
            if chunk_horizon is not None:
                logger.info(f"Chunking trajectories with horizon={chunk_horizon} ...")
                all_traj = self._chunk_trajectories(all_traj, chunk_horizon)
                logger.info(f"Chunked into {len(all_traj['actions'])} chunks.")
            logger.info("Writing memmap files ...")
            self._build_memmaps(all_traj)
            logger.info("Memmap rebuild finished.")
            del all_traj

        self._attach_memmaps()

        # Precompute (traj_idx, offset) indices for not-full-traj
        if not self.full_traj:
            self.get_idx = []
            for t_idx, L in enumerate(self.trajectories["lengths"]):
                for i in range(int(L)):
                    self.get_idx.append((t_idx, i))

        self.max_length = int(np.max(self.trajectories["lengths"]))
        self.length = int(np.sum(self.trajectories["lengths"])) if not self.full_traj else len(self.trajectories["lengths"])

    # ---------- Added: Chunk trajectory splitting ----------
    def _chunk_trajectories(self, trajs: Dict[str, List[np.ndarray]], horizon: int) -> Dict[str, List[np.ndarray]]:
        """
        Split each trajectory into multiple sub-trajectories of length horizon, keeping the last segment even if it's shorter than horizon.
        """
        new_traj = {
            "proprio": [], "images": [], "actions": [], "rewards": [],
            "dones": [], "lengths": [], "language": []
        }

        for i in tqdm(range(len(trajs["proprio"])), desc="Chunking trajectories"):
            proprio_i = trajs["proprio"][i]
            image_i   = trajs["images"][i]
            action_i = trajs["actions"][i]
            reward_i = trajs["rewards"][i]
            done_i = trajs["dones"][i]
            L = trajs["lengths"][i]
            lang_i = trajs["language"][i]

            start = 0
            while start < L:
                end = min(start + horizon, L)
                new_traj["proprio"].append(proprio_i[start:end])
                new_traj["images"].append(image_i[start:end])
                new_traj["actions"].append(action_i[start:end])
                new_traj["rewards"].append(reward_i[start:end])
                new_traj["dones"].append(done_i[start:end])
                new_traj["lengths"].append(end - start)
                new_traj["language"].append(lang_i)
                start = end

        return new_traj

    # ---------- Replaced: Compute Scale/Offset (Streaming) ----------
    def _compute_stats(self, all_traj: Dict[str, Any], range_eps: float = 1e-4):
        """
        compute scale/offset so that x_norm = x * scale + offset maps to approx [-1, 1]
        Uses streaming computation to avoid large intermediate arrays.
        """
        logger = logging.getLogger("main")
        
        # Initialize min/max trackers
        p_min, p_max = None, None
        im_min, im_max = None, None
        a_min, a_max = None, None
        
        # Fast path for dense LoReL arrays before chunking:
        # proprio/actions are (N,T,D), images are (N,T,C,H,W). This avoids
        # iterating over 120k small chunks and avoids many tiny astype/min calls.
        if isinstance(all_traj["proprio"], np.ndarray) and isinstance(all_traj["images"], np.ndarray):
            logger.info("Computing statistics from dense arrays ...")
            p = all_traj["proprio"].astype(np.float32, copy=False)
            im = all_traj["images"].astype(np.float32, copy=False)
            a = all_traj["actions"].astype(np.float32, copy=False)

            p_min = p.min(axis=(0, 1))
            p_max = p.max(axis=(0, 1))
            im_min = im.min(axis=(0, 1, 3, 4))
            im_max = im.max(axis=(0, 1, 3, 4))
            a_min = a.min(axis=(0, 1))
            a_max = a.max(axis=(0, 1))
        else:
            # Generic streaming path for list-of-trajectories data.
            logger.info("Computing statistics (streaming)...")
            for i in tqdm(range(len(all_traj["proprio"])), desc="Computing stats"):
                # Proprio
                p = all_traj["proprio"][i].astype(np.float32, copy=False)
                if p_min is None:
                    p_min = p.min(axis=0)
                    p_max = p.max(axis=0)
                else:
                    p_min = np.minimum(p_min, p.min(axis=0))
                    p_max = np.maximum(p_max, p.max(axis=0))
                
                # Images (per-channel min/max)
                im = all_traj["images"][i].astype(np.float32, copy=False)  # (T, C, H, W)
                im_i_min = im.min(axis=(0, 2, 3))  # (C,)
                im_i_max = im.max(axis=(0, 2, 3))  # (C,)
                if im_min is None:
                    im_min = im_i_min
                    im_max = im_i_max
                else:
                    im_min = np.minimum(im_min, im_i_min)
                    im_max = np.maximum(im_max, im_i_max)
                
                # Actions
                a = all_traj["actions"][i].astype(np.float32, copy=False)
                if a_min is None:
                    a_min = a.min(axis=0)
                    a_max = a.max(axis=0)
                else:
                    a_min = np.minimum(a_min, a.min(axis=0))
                    a_max = np.maximum(a_max, a.max(axis=0))

        # ----- proprio -----
        p_range = p_max - p_min
        p_safe = np.where(p_range < range_eps, 1.0, p_range)
        proprio_scale  = (2.0 / p_safe).astype(np.float32)
        proprio_offset = (-(p_max + p_min) / p_safe).astype(np.float32)
        small = (p_range < range_eps)
        proprio_scale  = np.where(small, 0.0, proprio_scale).astype(np.float32)
        proprio_offset = np.where(small, 0.0, proprio_offset).astype(np.float32)

        # ----- images -----
        im_range = im_max - im_min
        im_safe = np.where(im_range < range_eps, 1.0, im_range)
        image_scale  = (2.0 / im_safe).astype(np.float32)
        image_offset = (-(im_max + im_min) / im_safe).astype(np.float32)
        small_im = (im_range < range_eps)
        image_scale  = np.where(small_im, 0.0, image_scale).astype(np.float32)
        image_offset = np.where(small_im, 0.0, image_offset).astype(np.float32)

        # ----- actions -----
        a_range = a_max - a_min
        a_safe = np.where(a_range < range_eps, 1.0, a_range)
        action_scale  = (2.0 / a_safe).astype(np.float32)
        action_offset = (-(a_max + a_min) / a_safe).astype(np.float32)
        small_a = (a_range < range_eps)
        action_scale  = np.where(small_a, 0.0, action_scale).astype(np.float32)
        action_offset = np.where(small_a, 0.0, action_offset).astype(np.float32)

        np.save(self.proprio_scale_path, proprio_scale)
        np.save(self.proprio_offset_path, proprio_offset)
        np.save(self.image_scale_path, image_scale)
        np.save(self.image_offset_path, image_offset)
        np.save(self.action_scale_path, action_scale)
        np.save(self.action_offset_path, action_offset)

    # ---------- Prepare/Load Data ----------
    def _load_and_subsample(self, expert_location: str, num_traj: int, ss: int, seed: int, **kwargs) -> Dict[str, Any]:
        """Load from disk and subsample, returning dict(list/np.array)."""
        all_traj = load_trajectories(expert_location, num_traj, seed, **kwargs)

        # subsample
        if ss > 1:
            for k, v in all_traj.items():
                if k == "lengths":
                    all_traj[k] = np.array(v) // ss
                elif k == "language":
                    continue
                else:
                    all_traj[k] = [traj[0::ss] for traj in v]
        return all_traj

    def _infer_normalize_dim(self, expert_location: str, all_traj: Dict[str, Any], **kwargs):
        """ self.normalize_state_dim """
        # Check if this is LoReL data (by expert_location or task_name/name)
        is_lorel = ('lorel' in expert_location or
                    'lorel' in str(kwargs.get('task_name', '')).lower() or
                    'lorel' in str(kwargs.get('name', '')).lower())
        
        if is_lorel:
            # LoReL data uses proprio as state
            self.normalize_proprio_dim = int(np.array(all_traj["proprio"][0]).shape[-1])
        elif 'babyai' in expert_location and kwargs.get("use_direction", False):
            self.normalize_state_dim = all_traj["states"][0].shape[-1] - 4
        elif 'calvin' in expert_location:
            self.normalize_state_dim = all_traj["states"][0].shape[1:]
        else:
            self.normalize_state_dim = np.array(all_traj["states"][0]).shape[1:]

    def _memmap_exists(self) -> bool:
        paths = [
        self.meta_path, self.lengths_path, self.lang_path,
        self.proprio_path, self.images_path,
        self.proprio_scale_path, self.proprio_offset_path,
        self.image_scale_path, self.image_offset_path,
        self.actions_path, self.action_scale_path, self.action_offset_path,
        self.rewards_path, self.dones_path
        ]
        return all(os.path.exists(p) for p in paths)

    def _build_memmaps(self, all_traj: Dict[str, Any]):
        """rank: Write in-memory list data to disk memmap and save metadata and statistics.
        Uses chunked writing to avoid large intermediate arrays and reduce memory pressure.
        """
        lengths = np.array(all_traj["lengths"], dtype=np.int64)
        total_T = int(np.sum(lengths))

        logger = logging.getLogger("main")
        
        # Infer shapes from first element (no full concatenate yet)
        proprio_tail = tuple(all_traj["proprio"][0].shape[1:])   # (Dp,)
        image_tail   = tuple(all_traj["images"][0].shape[1:])    # (C,H,W)
        action_tail  = tuple(all_traj["actions"][0].shape[1:])   # (Da,)
        
        # Write files sequentially with normal buffered IO. This avoids slow mmap
        # dirty-page writeback during cache construction; later _attach_memmaps()
        # still reads these files through np.memmap with the same format.
        image_item_bytes = int(np.prod(image_tail)) * np.dtype(np.float32).itemsize
        target_batch_bytes = int(self.kwargs.get("memmap_write_batch_mb", 2048)) * 1024 * 1024
        batch_T = max(1, target_batch_bytes // max(1, image_item_bytes))
        logger.info(
            f"Writing raw memmap backing files with sequential IO: target_batch={target_batch_bytes / 1024 / 1024:.0f}MB, "
            f"batch_T≈{batch_T} timesteps"
        )

        n = len(all_traj["proprio"])
        pbar = tqdm(total=total_T, desc="Writing memmap backing files", unit="step")
        start_i = 0
        with open(self.proprio_path, "wb") as f_proprio, \
             open(self.images_path, "wb") as f_images, \
             open(self.actions_path, "wb") as f_actions, \
             open(self.rewards_path, "wb") as f_rewards, \
             open(self.dones_path, "wb") as f_dones:
            while start_i < n:
                end_i = start_i
                cur_T = 0
                while end_i < n and (cur_T == 0 or cur_T + int(lengths[end_i]) <= batch_T):
                    cur_T += int(lengths[end_i])
                    end_i += 1

                np.ascontiguousarray(
                    np.concatenate(all_traj["proprio"][start_i:end_i], axis=0), dtype=np.float32
                ).tofile(f_proprio)
                np.ascontiguousarray(
                    np.concatenate(all_traj["images"][start_i:end_i], axis=0), dtype=np.float32
                ).tofile(f_images)
                np.ascontiguousarray(
                    np.concatenate(all_traj["actions"][start_i:end_i], axis=0), dtype=np.float32
                ).tofile(f_actions)
                np.ascontiguousarray(
                    np.concatenate(all_traj["rewards"][start_i:end_i], axis=0), dtype=np.float32
                ).tofile(f_rewards)
                np.ascontiguousarray(
                    np.concatenate(all_traj["dones"][start_i:end_i], axis=0), dtype=np.uint8
                ).tofile(f_dones)

                pbar.update(cur_T)
                start_i = end_i
                gc.collect()

        pbar.close()

        np.save(self.lengths_path, lengths)
        with open(self.lang_path, "w", encoding="utf-8") as f:
            json.dump(all_traj.get("language", [""] * len(lengths)), f, ensure_ascii=False)

        meta = dict(
            total_timesteps=total_T,
            num_trajectories=int(len(lengths)),
            proprio_tail=proprio_tail,
            image_tail=image_tail,
            action_tail=action_tail,
            dtype_states="float32",
            dtype_actions="float32",
            dtype_rewards="float32",
            dtype_dones="uint8",
        )
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def _attach_memmaps(self):
        """All ranks: attach to existing memmap files and load necessary metadata."""
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.meta = meta

        lengths = np.load(self.lengths_path).astype(np.int64)
        self.trajectories = {}
        self.trajectories["lengths"] = lengths
        self.trajectory_indices = np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)])
        total_T = int(meta["total_timesteps"])

        proprio_tail = tuple(meta["proprio_tail"])
        image_tail   = tuple(meta["image_tail"])
        action_tail  = tuple(meta["action_tail"])

        self.trajectories["proprio"] = np.memmap(
            self.proprio_path, mode="r", dtype=np.float32, shape=(total_T,) + proprio_tail
        )
        self.trajectories["images"] = np.memmap(
            self.images_path, mode="r", dtype=np.float32, shape=(total_T,) + image_tail
        )
        self.trajectories["actions"] = np.memmap(
            self.actions_path, mode="r", dtype=np.float32, shape=(total_T,) + action_tail
        )
        self.trajectories["rewards"] = np.memmap(
            self.rewards_path, mode="r", dtype=np.float32, shape=(total_T,)
        )
        self.trajectories["dones"] = np.memmap(
            self.dones_path, mode="r", dtype=np.uint8, shape=(total_T,)
        )

        with open(self.lang_path, "r", encoding="utf-8") as f:
            self.trajectories["language"] = json.load(f)

        self.proprio_scale  = np.load(self.proprio_scale_path).astype(np.float32)
        self.proprio_offset = np.load(self.proprio_offset_path).astype(np.float32)
        self.image_scale    = np.load(self.image_scale_path).astype(np.float32)
        self.image_offset   = np.load(self.image_offset_path).astype(np.float32)
        self.action_scale   = np.load(self.action_scale_path).astype(np.float32)
        self.action_offset  = np.load(self.action_offset_path).astype(np.float32)

    def _np_slice_to_torch(self, x):
        """
        Convert a zero-copy numpy/memmap slice to torch.Tensor, and clone() to avoid in-place normalization writing back to the underlying data.
        If it is already a torch.Tensor, return a clone() directly.
        """
        if isinstance(x, torch.Tensor):
            return x.clone().float()
        xt = torch.from_numpy(np.array(x, copy=True)).float()
        return xt.clone()

    # ---------- Tools ----------
    @staticmethod
    def _pad_torch(x: torch.Tensor, max_len: int) -> torch.Tensor:
        """Pad zeros to max_len along dim=0."""
        if x.size(0) == max_len:
            return x
        out_shape = (max_len,) + tuple(x.shape[1:])
        out = torch.zeros(out_shape, dtype=x.dtype)
        out[:x.size(0)] = x
        return out

    # ---------- Dataset interface ----------
    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        if not self.full_traj:
            traj_idx, offset = self.get_idx[i]
            g = int(self.trajectory_indices[traj_idx] + offset)
            g_next = g + 1

            state = self._np_slice_to_torch(self.trajectories["states"][g])
            next_state = self._np_slice_to_torch(self.trajectories["states"][g_next])

            if self.normalize_states:
                d = self.normalize_state_dim
                if isinstance(d, (tuple, list)):  # CHW Normalization
                    state = (state - self.state_mean) / self.state_std
                    next_state = (next_state - self.state_mean) / self.state_std
                else:
                    state[..., :d] = (state[..., :d] - self.state_mean) / self.state_std
                    next_state[..., :d] = (next_state[..., :d] - self.state_mean) / self.state_std

            action = self._np_slice_to_torch(self.trajectories["actions"][g])
            reward = self._np_slice_to_torch(self.trajectories["rewards"][g])
            done = self._np_slice_to_torch(self.trajectories["dones"][g].astype(np.float32))
            return (state, next_state, action, reward, done)
        else:
            start = int(self.trajectory_indices[i])
            end   = int(self.trajectory_indices[i + 1])

            proprio = self._np_slice_to_torch(self.trajectories["proprio"][start:end])   # (T,Dp)
            images  = self._np_slice_to_torch(self.trajectories["images"][start:end])    # (T,C,H,W)
            actions = self._np_slice_to_torch(self.trajectories["actions"][start:end])   # (T,Da)
            dones   = self._np_slice_to_torch(self.trajectories["dones"][start:end].astype(np.float32))

            T = proprio.size(0)
            if self.normalize_obs:
                ps = torch.as_tensor(self.proprio_scale, dtype=proprio.dtype, device=proprio.device)
                po = torch.as_tensor(self.proprio_offset, dtype=proprio.dtype, device=proprio.device)
                proprio = proprio * ps[None, :] + po[None, :]

                ims = torch.as_tensor(self.image_scale, dtype=images.dtype, device=images.device)
                imo = torch.as_tensor(self.image_offset, dtype=images.dtype, device=images.device)
                images = images * ims[None, :, None, None] + imo[None, :, None, None]

            if self.normalize_actions:
                ascale = torch.as_tensor(self.action_scale, dtype=actions.dtype, device=actions.device)
                aoffset = torch.as_tensor(self.action_offset, dtype=actions.dtype, device=actions.device)
                actions = actions * ascale[None, :] + aoffset[None, :]

            timesteps = torch.arange(0, T, dtype=torch.long)
            attention_mask = torch.ones(T, dtype=torch.float32)
            language = '' if self.no_lang else self.trajectories["language"][i]

            L = self.chunk_horizon
            proprio = self._pad_torch(proprio, L)
            images  = self._pad_torch(images, L)
            actions = self._pad_torch(actions, L)
            dones   = self._pad_torch(dones, L)
            timesteps = self._pad_torch(timesteps, L)
            attention_mask = self._pad_torch(attention_mask, L)

            obs = {"proprio": proprio, "images": images}
            return (language, obs, actions, timesteps, dones, attention_mask)


if __name__ == '__main__':
    exper_path = "/home/lyh/data_folder/dataspace/lorel/may_08_sawyer_50k/prep_data.pkl"
    memmap_dir = "/home/lyh/data_folder/dataspace/lorel/may_08_sawyer_50k/memmap_cache_lorel"

    # First run with writer=True to build memmap
    ds = ExpertDataset(
        expert_location=exper_path,
        memmap_dir=memmap_dir,
        num_trajectories=40108,
        subsample_frequency=1,
        seed=0,
        full_traj=True,
        normalize_obs=True,
        no_lang=False,
        writer=True,
        force_rebuild=False,
        chunk_horizon=8,  
    )

    print("len(dataset) =", len(ds))
    lang, obs, act, t, d, mask = ds[0]
    print("proprio:", obs["proprio"].shape)
    print("images:", obs["images"].shape)
    print("actions:", act.shape)
    print("timesteps:", t.shape)
    print("mask:", mask.shape)
