from typing import Any, Dict, IO, List, Tuple, Union, Optional
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
        elif 'lorel' in expert_location:
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

        # Memmap 路径（新增 scale/offset 保存路径）
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
        self.images_path  = os.path.join(self.memmap_dir, "images.mmap")   # 修正 .mmap
        self.actions_path = os.path.join(self.memmap_dir, "actions.mmap")
        self.rewards_path = os.path.join(self.memmap_dir, "rewards.mmap")
        self.dones_path   = os.path.join(self.memmap_dir, "dones.mmap")

        # Rank0 构建 memmap（新增 chunk 处理）
        if writer and (force_rebuild or not self._memmap_exists()):
            all_traj = self._load_and_subsample(
                expert_location, num_trajectories, subsample_frequency, seed, **kwargs
            )
            # 新增：chunk轨迹切分
            if chunk_horizon is not None:
                all_traj = self._chunk_trajectories(all_traj, chunk_horizon)
            self._infer_normalize_dim(expert_location, all_traj, **kwargs)
            self._compute_stats(all_traj)
            self._build_memmaps(all_traj)
            del all_traj

        self._attach_memmaps()

        # 预计算 not-full-traj 的 (traj_idx, offset) 索引
        if not self.full_traj:
            self.get_idx = []
            for t_idx, L in enumerate(self.trajectories["lengths"]):
                for i in range(int(L)):
                    self.get_idx.append((t_idx, i))

        self.max_length = int(np.max(self.trajectories["lengths"]))
        self.length = int(np.sum(self.trajectories["lengths"])) if not self.full_traj else len(self.trajectories["lengths"])

    # ---------- 新增：Chunk轨迹切分 ----------
    def _chunk_trajectories(self, trajs: Dict[str, List[np.ndarray]], horizon: int) -> Dict[str, List[np.ndarray]]:
        """
        把每条轨迹按horizon切分为多个子轨迹，最后一段长度不足horizon也保留
        """
        new_traj = {
            "proprio": [], "images": [], "actions": [], "rewards": [],
            "dones": [], "lengths": [], "language": []
        }

        for i in range(len(trajs["proprio"])):
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

    # ---------- 替换：计算Scale/Offset（替代原mean/std） ----------
    def _compute_stats(self, all_traj: Dict[str, Any], range_eps: float = 1e-4):
        """
        compute scale/offset so that x_norm = x * scale + offset maps to approx [-1, 1]
        """
        all_proprio = np.concatenate(all_traj["proprio"], axis=0).astype(np.float32)  # (sum_T, Dp)
        all_images  = np.concatenate(all_traj["images"], axis=0).astype(np.float32)   # (sum_T, C, H, W)
        all_actions = np.concatenate(all_traj["actions"], axis=0).astype(np.float32)  # (sum_T, Da)

        # ----- proprio -----
        p_min = all_proprio.min(axis=0)
        p_max = all_proprio.max(axis=0)
        p_range = p_max - p_min
        p_safe = np.where(p_range < range_eps, 1.0, p_range)

        proprio_scale  = (2.0 / p_safe).astype(np.float32)
        proprio_offset = (-(p_max + p_min) / p_safe).astype(np.float32)

        small = (p_range < range_eps)
        proprio_scale  = np.where(small, 0.0, proprio_scale).astype(np.float32)
        proprio_offset = np.where(small, 0.0, proprio_offset).astype(np.float32)

        # ----- images: per-channel -----
        # (T,C,H,W) -> keep C
        im_min = all_images.min(axis=(0, 2, 3))
        im_max = all_images.max(axis=(0, 2, 3))
        im_range = im_max - im_min
        im_safe = np.where(im_range < range_eps, 1.0, im_range)

        image_scale  = (2.0 / im_safe).astype(np.float32)             # (C,)
        image_offset = (-(im_max + im_min) / im_safe).astype(np.float32)

        small_im = (im_range < range_eps)
        image_scale  = np.where(small_im, 0.0, image_scale).astype(np.float32)
        image_offset = np.where(small_im, 0.0, image_offset).astype(np.float32)

        # ----- actions -----
        a_min = all_actions.min(axis=0)
        a_max = all_actions.max(axis=0)
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

    # ---------- 准备/加载 数据 ----------
    def _load_and_subsample(self, expert_location: str, num_traj: int, ss: int, seed: int, **kwargs) -> Dict[str, Any]:
        """读盘并做降采样，返回 dict(list/np.array)。"""
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
        if 'babyai' in expert_location and kwargs.get("use_direction", False):
            self.normalize_state_dim = all_traj["states"][0].shape[-1] - 4
        elif 'lorel' in expert_location:
            self.normalize_proprio_dim = int(np.array(all_traj["proprio"][0]).shape[-1])
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
        """rank: 把内存中的列表数据写到磁盘 memmap 并保存元信息与统计量。"""
        lengths = np.array(all_traj["lengths"], dtype=np.int64)
        total_T = int(np.sum(lengths))

        proprio_np = np.concatenate(all_traj["proprio"], axis=0).astype(np.float32)
        images_np  = np.concatenate(all_traj["images"], axis=0).astype(np.float32)
        actions_np = np.concatenate(all_traj["actions"], axis=0).astype(np.float32)
        actions_np =np.concatenate(all_traj["actions"], axis=0).astype(np.float32)
        rewards_np = np.concatenate(all_traj["rewards"], axis=0).astype(np.float32)
        dones_np = np.concatenate(all_traj["dones"], axis=0).astype(np.uint8)

        proprio_tail = tuple(proprio_np.shape[1:])   # (Dp,)
        image_tail   = tuple(images_np.shape[1:])    # (C,H,W)
        action_tail  = tuple(actions_np.shape[1:])   # (Da,)

        self._write_memmap(self.proprio_path, proprio_np)
        self._write_memmap(self.images_path, images_np)
        self._write_memmap(self.actions_path, actions_np)
        self._write_memmap(self.rewards_path, rewards_np)
        self._write_memmap(self.dones_path, dones_np)

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
        """所有 rank：附着到已存在的 memmap 文件，并加载必要元信息。"""
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

    def _write_memmap(self, path: str, arr: np.ndarray):
        mm = np.memmap(path, mode="w+", dtype=arr.dtype, shape=arr.shape)
        mm[:] = arr
        del mm  # flush

    def _np_slice_to_torch(self, x):
        """
        将 numpy / memmap 切片零拷贝转 torch.Tensor，并 clone() 以避免原地归一化写回底层。
        如果本身已经是 torch.Tensor 则直接返回 clone()。
        """
        if isinstance(x, torch.Tensor):
            return x.clone().float()
        xt = torch.from_numpy(np.array(x, copy=True)).float()
        return xt.clone()

    # ---------- 公共工具 ----------
    @staticmethod
    def _pad_torch(x: torch.Tensor, max_len: int) -> torch.Tensor:
        """在 dim=0 方向补零到 max_len。"""
        if x.size(0) == max_len:
            return x
        out_shape = (max_len,) + tuple(x.shape[1:])
        out = torch.zeros(out_shape, dtype=x.dtype)
        out[:x.size(0)] = x
        return out

    # ---------- Dataset 接口 ----------
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
                if isinstance(d, (tuple, list)):  # CHW 归一化
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

    # 第一次跑用 writer=True 建 memmap
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
