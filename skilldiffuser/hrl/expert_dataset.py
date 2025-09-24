from typing import Any, Dict, IO, List, Tuple

import numpy as np
import pandas as pd
import pickle
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from utils import pad, calculate_state_means_stds
from tqdm import tqdm

import json
from typing import Dict, Any, Tuple, Optional


# class ExpertDataset(Dataset):
#     """Dataset for expert trajectories.
#     Assumes expert dataset is a dict with keys {states, actions, rewards, lengths} with values
#     of given shapes.
#         shapes:
#             expert["language"] = [num_experts]
#             expert["states"] =  [num_experts, max_length, state_dim]
#             expert["actions"] =  [num_experts, max_length, action_dim]
#             expert["rewards"] =  [num_experts, max_length]
#             expert["lengths"] =  [num_experts]
#             expert["dones"] =  [num_experts, max_length]
#     """

#     def __init__(self,
#                  expert_location: str,
#                  num_trajectories: int = 1,
#                  subsample_frequency: int = 1,
#                  seed: int = 0,
#                  full_traj: bool = True,
#                  normalize_states: bool = True,
#                  no_lang=False,
#                  **kwargs):
#         """Subsamples an expert dataset from saved expert trajectories.
#         Args:
#             expert_location:          Location of saved expert trajectories.
#             num_trajectories:         Number of expert trajectories to sample (randomized).
#             subsample_frequency:      Subsamples each trajectory at specified frequency of steps.
#             seed:                     Seed for sampling trajectories.
#             full_traj:                If True, each item will be a full trajectory and not just a (s,s',a,r,d) tuple
#         """
#         all_trajectories = load_trajectories(expert_location, num_trajectories, seed, **kwargs)
#         self.kwargs = kwargs
#         self.trajectories = {}
#         self.full_traj = full_traj
#         self.normalize_states = normalize_states
#         self.no_lang = no_lang
        
#         if 'aug' in self.kwargs and self.kwargs['aug']:
#             ## From the LORL paper
#             self.aug = torch.nn.Sequential(
#                         transforms.ColorJitter(brightness=0.02, contrast=0.02, saturation=0.02, hue=0.02),
#                         transforms.RandomAffine(20, translate=(0.1, 0.1), scale=(0.9, 1.1)))
      

#         # skip the direction part for normalization
#         if 'babyai' in expert_location and kwargs["use_direction"]:
#             self.normalize_state_dim = all_trajectories["states"][0].shape[-1] - 4
#         elif 'lorel' in expert_location:
#             self.normalize_state_dim = all_trajectories["states"][0].shape[1:]
#         elif 'calvin' in expert_location:
#             self.normalize_state_dim = all_trajectories["states"][0].shape[1:]
#         else:
#             self.normalize_state_dim = np.array(all_trajectories["states"][0]).shape[1:]

#         if normalize_states:
#             self.state_mean, self.state_std = calculate_state_means_stds(
#                 all_trajectories["states"], self.normalize_state_dim)
#         else:
#             self.state_mean, self.state_std = np.zeros(self.normalize_state_dim), np.ones(self.normalize_state_dim)

#         # Randomize start index of each trajectory for subsampling
#         # start_idx = torch.randint(0, subsample_frequency, size=(num_trajectories,)).long()

#         # # Subsample expert trajectories with every `subsample_frequency` step.
#         # for k, v in all_trajectories.items():
#         #     data = v

#         #     if k != "lengths":
#         #         samples = []
#         #         for i in range(num_trajectories):
#         #             samples.append(data[i][0::subsample_frequency])
#         #         self.trajectories[k] = samples
#         #     else:
#         #         # Adjust the length of trajectory after subsampling
#         #         self.trajectories[k] = np.array(data) // subsample_frequency

#         # if not full_traj:
#         #     self.length = self.trajectories["lengths"].sum().item()
#         # else:
#         #     self.length = len(self.trajectories["lengths"])
#         #     self.total_timesteps = self.trajectories["lengths"].sum().item()

#         # --- 修改开始: 重新实现 subsample_frequency 接口 ---
#         if subsample_frequency > 1:
#             # 对 all_trajectories 中的数据进行降采样
#             for k, v in all_trajectories.items():
#                 if k == "lengths":
#                     # 调整降采样后轨迹的长度
#                     all_trajectories[k] = np.array(v) // subsample_frequency
#                 elif k == "language":
#                     # 语言指令是针对整个轨迹的，不需要降采样
#                     continue
#                 else:
#                     # 对每个轨迹的 steps 进行降采样
#                     # v 是一个轨迹列表, e.g., [traj1, traj2, ...]
#                     all_trajectories[k] = [traj[0::subsample_frequency] for traj in v]

#         # --- 修改开始: 整合数据以优化共享内存 ---
#         # 1. 将所有轨迹数据拼接成大的 Numpy 数组
#         #    并记录每条轨迹的起始索引
#         self.trajectory_indices = [0]
#         current_pos = 0
#         for length in all_trajectories["lengths"]:
#             current_pos += length
#             self.trajectory_indices.append(current_pos)

#         for k, v in all_trajectories.items():
#             if k == 'language' or k == 'rewards':
#                 self.trajectories[k] = v
#             # --- 修改开始: 确保 'lengths' 是一个 NumPy 数组 ---
#             elif k == 'lengths':
#                 self.trajectories[k] = np.array(v)
#             else:
#                 # 将轨迹列表 [ (len1, dim), (len2, dim), ... ] 拼接成一个大数组 (total_len, dim)
#                 self.trajectories[k] = torch.from_numpy(np.concatenate(v, axis=0)).float()

#         if not full_traj:
#             self.length = self.trajectories["lengths"].sum().item()
#         else:
#             self.length = len(self.trajectories["lengths"])
#             self.total_timesteps = self.trajectories["lengths"].sum().item()


#         del all_trajectories  # Not needed anymore

#         self.max_length = max(self.trajectories["lengths"])

#         # --- 修改开始: 将 Numpy 数组转换为 Torch 张量 ---
#         self.state_mean = torch.from_numpy(self.state_mean).float()
#         self.state_std = torch.from_numpy(self.state_std).float()

#         # for k, v in self.trajectories.items():
#         #     if k == 'language' or k == 'lengths':
#         #         continue
#         #     # 将列表中的每个 numpy array 转换为 tensor
#         #     self.trajectories[k] = [torch.from_numpy(arr).float() if arr.dtype != 'object' else arr for arr in v]
#         # # --- 修改结束 ---

#         # Convert flattened index i to trajectory indx and offset within trajectory
#         if not self.full_traj:
#             traj_idx = 0
#             i = 0
#             self.get_idx = []

#             for _j in range(self.length):
#                 while self.trajectories["lengths"][traj_idx].item() <= i:
#                     i -= self.trajectories["lengths"][traj_idx].item()
#                     traj_idx += 1

#                 self.get_idx.append((traj_idx, i))
#                 i += 1

#     def share_memory(self):
#         """
#         将数据集中的所有 torch.Tensor 移动到共享内存。
#         """
#         self.state_mean.share_memory_()
#         self.state_std.share_memory_()
#         for k, v in self.trajectories.items():
#             if isinstance(v, torch.Tensor):
#                 v.share_memory_()


#     def __len__(self) -> int:
#         """Return the length of the dataset."""
#         return self.length

#     def __getitem__(self, i):
#         if not self.full_traj:
#             traj_idx, i = self.get_idx[i]

#             state = self.trajectories["states"][traj_idx][i]
#             next_state = self.trajectories["states"][traj_idx][i+1]

#             # Rescale states and next_states to [0, 1] if are images
#             # if isinstance(states, np.ndarray) and states.ndim == 3:
#             #     states = np.array(states) / 255.0
#             # if isinstance(states, np.ndarray) and next_states.ndim == 3:
#             #     next_states = np.array(next_states) / 255.0
#             if self.normalize_states:
#                 state[:self.normalize_state_dim] = (
#                     state[:self.normalize_state_dim] - self.state_mean) / self.state_std
#                 next_state[:self.normalize_state_dim] = (
#                     next_state[:self.normalize_state_dim] - self.state_mean) / self.state_std

#             return (state,
#                     next_state,
#                     self.trajectories["actions"][traj_idx][i],
#                     self.trajectories["rewards"][traj_idx][i],
#                     self.trajectories["dones"][traj_idx][i])
#         else:
#             # --- 修改开始: 从大张量中切片获取轨迹 ---
#             start_idx = self.trajectory_indices[i]
#             end_idx = self.trajectory_indices[i+1]
#             states = self.trajectories["states"][start_idx:end_idx]

#             states = np.array(states)
#             if 'aug' in self.kwargs and self.kwargs['aug']:
#                 states = self.aug(torch.from_numpy(states))

#             # Rescale states and next_states to [0, 1] if are images
#             # if isinstance(states, np.ndarray) and states.ndim == 3:
#             #     states = np.array(states) / 255.0
#             if self.normalize_states:
#                 states[:, :self.normalize_state_dim] = (
#                     states[:, :self.normalize_state_dim] - self.state_mean) / self.state_std

#             # timesteps = np.arange(0, states.shape[0])
#             # attention_mask = np.ones(states.shape[0])

#             timesteps = torch.arange(0, states.shape[0])
#             attention_mask = torch.ones(states.shape[0])

#             language = '' if self.no_lang else self.trajectories["language"][i]

#             # return (language,
#             #         pad(states, self.max_length, axis=0),
#             #         pad(self.trajectories["actions"][i], self.max_length, axis=0),
#             #         pad(timesteps, self.max_length, axis=0),
#             #         pad(self.trajectories["dones"][i], self.max_length, axis=0),
#             #         pad(attention_mask, self.max_length, axis=0)
#             #         )

#             actions = self.trajectories["actions"][start_idx:end_idx]
#             dones = self.trajectories["dones"][start_idx:end_idx]
            
#             return (language,
#                     pad(states.numpy(), self.max_length, axis=0),
#                     pad(actions.numpy(), self.max_length, axis=0),
#                     pad(timesteps.numpy(), self.max_length, axis=0),
#                     pad(dones.numpy(), self.max_length, axis=0),
#                     pad(attention_mask.numpy(), self.max_length, axis=0)
#                     )

class ExpertDataset(Dataset):
    """
    Dataset for expert trajectories with optional disk-backed memmap.
    Keys:
        language: [num_experts] (list[str] or 1D array)
        states:   [num_experts, max_length, state_dim]  (after flatten -> [total_T, state_dim or ...])
        actions:  [num_experts, max_length, action_dim] (after flatten -> [total_T, action_dim])
        rewards:  [num_experts, max_length]             (after flatten -> [total_T])
        lengths:  [num_experts]
        dones:    [num_experts, max_length]             (after flatten -> [total_T])
    """

    def __init__(self,
                 expert_location: str,
                 num_trajectories: int = 1,
                 subsample_frequency: int = 1,
                 seed: int = 0,
                 full_traj: bool = True,
                 normalize_states: bool = True,
                 no_lang: bool = False,
                 # 新增：磁盘内存映射
                 memmap_dir: Optional[str] = None,
                 writer: bool = False,
                 force_rebuild: bool = False,
                 **kwargs):
        """
        Args:
            expert_location: 数据源位置（传给 load_trajectories）
            memmap_dir: 若提供，使用该目录做 mmap（推荐在 DDP 下使用）
            writer: 仅 rank 0 设为 True，负责构建 memmap 文件；其他 rank False。
            force_rebuild: 即使已存在 memmap 也强制重建（谨慎使用）
        """
        super().__init__()
        self.kwargs = kwargs
        self.full_traj = full_traj
        self.normalize_states = normalize_states
        self.no_lang = no_lang
        self.memmap_dir = memmap_dir
        self.using_memmap = memmap_dir is not None

        if 'aug' in self.kwargs and self.kwargs['aug'] and transforms is not None:
            # From the LORL paper
            self.aug = torch.nn.Sequential(
                transforms.ColorJitter(brightness=0.02, contrast=0.02, saturation=0.02, hue=0.02),
                transforms.RandomAffine(20, translate=(0.1, 0.1), scale=(0.9, 1.1))
            )
        else:
            self.aug = None

        # =========================
        # 路径与元信息（mmap 模式）
        # =========================
        if self.using_memmap:
            os.makedirs(self.memmap_dir, exist_ok=True)
            self.meta_path = os.path.join(self.memmap_dir, "meta.json")
            self.lengths_path = os.path.join(self.memmap_dir, "lengths.npy")
            self.state_mean_path = os.path.join(self.memmap_dir, "state_mean.npy")
            self.state_std_path = os.path.join(self.memmap_dir, "state_std.npy")
            self.lang_path = os.path.join(self.memmap_dir, "language.json")

            self.states_path = os.path.join(self.memmap_dir, "states.mmap")
            self.actions_path = os.path.join(self.memmap_dir, "actions.mmap")
            self.rewards_path = os.path.join(self.memmap_dir, "rewards.mmap")
            self.dones_path = os.path.join(self.memmap_dir, "dones.mmap")

            if writer and (force_rebuild or not self._memmap_exists()):
                # rank 0：构建 memmap
                all_traj = self._load_and_subsample(
                    expert_location, num_trajectories, subsample_frequency, seed, **kwargs
                )
                self._infer_normalize_state_dim(expert_location, all_traj, **kwargs)
                self._compute_stats(all_traj)  # self.state_mean / self.state_std (numpy)
                self._build_memmaps(all_traj)  # 写入 .mmap / .npy / .json
                del all_traj  # 释放内存

            # DDP 中建议 barrier 放在外部控制流里
            self._attach_memmaps()  # 所有 rank 附着
        else:
            # 非 mmap 模式：保持你原来的“内存拼接”实现（不推荐大数据用）
            all_traj = self._load_and_subsample(
                expert_location, num_trajectories, subsample_frequency, seed, **kwargs
            )
            self._infer_normalize_state_dim(expert_location, all_traj, **kwargs)
            if normalize_states:
                # 复用你原先的统计函数
                self.state_mean, self.state_std = calculate_state_means_stds(
                    all_traj["states"], self.normalize_state_dim
                )
            else:
                self.state_mean, self.state_std = np.zeros(self.normalize_state_dim), np.ones(self.normalize_state_dim)
            self._flatten_in_memory(all_traj)

        # 统一转换成 torch.Tensor（注意：mmap 下不要把底层数组改成 Tensor；只在 __getitem__ 返回时转 tensor）
        self.state_mean = torch.from_numpy(np.array(self.state_mean)).float()
        self.state_std = torch.from_numpy(np.array(self.state_std)).float()

        # 预计算 not-full-traj 的 (traj_idx, offset) 索引
        if not self.full_traj:
            self.get_idx = []
            for t_idx, L in enumerate(self.trajectories["lengths"]):
                for i in range(int(L)):
                    self.get_idx.append((t_idx, i))

        # 最大长度
        self.max_length = int(np.max(self.trajectories["lengths"]))
        # 数据集长度
        self.length = int(np.sum(self.trajectories["lengths"])) if not self.full_traj else len(self.trajectories["lengths"])

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

    def _memmap_exists(self) -> bool:
        paths = [self.meta_path, self.lengths_path, self.state_mean_path, self.state_std_path,
                 self.lang_path, self.states_path, self.actions_path, self.rewards_path, self.dones_path]
        return all(os.path.exists(p) for p in paths)

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

    def _infer_normalize_state_dim(self, expert_location: str, all_traj: Dict[str, Any], **kwargs):
        """保持你原来的规则，得到 self.normalize_state_dim。"""
        if 'babyai' in expert_location and kwargs.get("use_direction", False):
            self.normalize_state_dim = all_traj["states"][0].shape[-1] - 4
        elif 'lorel' in expert_location:
            self.normalize_state_dim = all_traj["states"][0].shape[1:]
        elif 'calvin' in expert_location:
            self.normalize_state_dim = all_traj["states"][0].shape[1:]
        else:
            self.normalize_state_dim = np.array(all_traj["states"][0]).shape[1:]

    def _compute_stats(self, all_traj: Dict[str, Any]):
        if self.normalize_states:
            self.state_mean, self.state_std = calculate_state_means_stds(
                all_traj["states"], self.normalize_state_dim
            )
        else:
            self.state_mean, self.state_std = np.zeros(self.normalize_state_dim), np.ones(self.normalize_state_dim)

    def _flatten_lists_to_array(self, seq_list, dtype):
        """把 [arr(T1,...), arr(T2,...), ...] 拼成一个 (sum_T, ...) 的 np.array。"""
        return np.concatenate(seq_list, axis=0).astype(dtype, copy=False)

    def _build_memmaps(self, all_traj: Dict[str, Any]):
        """rank 0：把内存中的列表数据写到磁盘 memmap，并保存元信息与统计量。"""
        lengths = np.array(all_traj["lengths"], dtype=np.int64)
        total_T = int(np.sum(lengths))

        # 扁平化（states/actions/rewards/dones）
        states_np = self._flatten_lists_to_array(all_traj["states"], np.float32)
        actions_np = self._flatten_lists_to_array(all_traj["actions"], np.float32)
        rewards_np = self._flatten_lists_to_array(all_traj["rewards"], np.float32)
        # dones 用 uint8 节省空间；使用时再转 float/bool
        dones_np = self._flatten_lists_to_array(all_traj["dones"], np.uint8)

        # 形状信息（每步的尾部形状）
        state_tail = tuple(states_np.shape[1:])
        action_tail = tuple(actions_np.shape[1:])

        # 写 memmap
        self._write_memmap(self.states_path, states_np)
        self._write_memmap(self.actions_path, actions_np)
        self._write_memmap(self.rewards_path, rewards_np)
        self._write_memmap(self.dones_path, dones_np)

        # 写统计与长度
        np.save(self.lengths_path, lengths)
        np.save(self.state_mean_path, np.array(self.state_mean, dtype=np.float32))
        np.save(self.state_std_path, np.array(self.state_std, dtype=np.float32))

        # 写语言（每条轨迹一个字符串）
        with open(self.lang_path, "w", encoding="utf-8") as f:
            json.dump(all_traj.get("language", [""] * len(lengths)), f, ensure_ascii=False)

        # 写 meta
        meta = dict(
            total_timesteps=total_T,
            num_trajectories=int(len(lengths)),
            state_tail=state_tail,
            action_tail=action_tail,
            dtype_states="float32",
            dtype_actions="float32",
            dtype_rewards="float32",
            dtype_dones="uint8",
            normalize_state_dim=self._shape_to_list(self.normalize_state_dim),
        )
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def _write_memmap(self, path: str, arr: np.ndarray):
        mm = np.memmap(path, mode="w+", dtype=arr.dtype, shape=arr.shape)
        mm[:] = arr
        del mm  # flush

    def _attach_memmaps(self):
        """所有 rank：附着到已存在的 memmap 文件，并加载必要元信息。"""
        # 1) 读 meta
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.meta = meta

        # 2) 读 normalize_state_dim（非 writer 路径需要恢复这个属性）
        nsd = meta.get("normalize_state_dim", None)
        if nsd is None:
            raise RuntimeError("meta.json missing 'normalize_state_dim'")
        # 存的时候是 list 或 int，这里转成合适的类型
        if isinstance(nsd, list):
            self.normalize_state_dim = tuple(nsd) if len(nsd) > 1 else int(nsd[0])
        else:
            self.normalize_state_dim = int(nsd)

        # 3) 读长度并构造全局起止索引
        lengths = np.load(self.lengths_path).astype(np.int64)
        self.trajectories: Dict[str, Any] = {}
        self.trajectories["lengths"] = lengths
        self.trajectory_indices = np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)])

        # 4) 附着各大数组（只读）
        total_T = int(meta["total_timesteps"])
        state_tail = tuple(meta["state_tail"])
        action_tail = tuple(meta["action_tail"])

        self.trajectories["states"] = np.memmap(
            self.states_path, mode="r", dtype=np.float32, shape=(total_T,) + state_tail
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

        # 5) 读语言（每条轨迹一个字符串）
        with open(self.lang_path, "r", encoding="utf-8") as f:
            self.trajectories["language"] = json.load(f)

        # 6) 读回统计量（关键：非 writer 路径需要加载它们）
        self.state_mean = np.load(self.state_mean_path).astype(np.float32)
        self.state_std  = np.load(self.state_std_path).astype(np.float32)


    def _flatten_in_memory(self, all_traj: Dict[str, Any]):
        """保持你原先的“内存拼接到大 Tensor”的实现（大数据下不推荐）。"""
        self.trajectories = {}
        self.trajectories["lengths"] = np.array(all_traj["lengths"], dtype=np.int64)
        self.trajectory_indices = np.concatenate([[0], np.cumsum(self.trajectories["lengths"], dtype=np.int64)])

        # 直接存 torch.Tensor（与你原代码一致）
        self.trajectories["states"] = torch.from_numpy(
            self._flatten_lists_to_array(all_traj["states"], np.float32)
        )
        self.trajectories["actions"] = torch.from_numpy(
            self._flatten_lists_to_array(all_traj["actions"], np.float32)
        )
        self.trajectories["rewards"] = torch.from_numpy(
            self._flatten_lists_to_array(all_traj["rewards"], np.float32)
        )
        self.trajectories["dones"] = torch.from_numpy(
            self._flatten_lists_to_array(all_traj["dones"], np.uint8)
        )
        self.trajectories["language"] = all_traj.get("language", [""] * len(self.trajectories["lengths"]))

    @staticmethod
    def _shape_to_list(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return list(x)
        return int(x)

    # ---------- Dataset 接口 ----------

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        # full_traj=False：i 是“步级索引”
        # full_traj=True：i 是“轨迹索引”
        if not self.full_traj:
            traj_idx, offset = self.get_idx[i]
            g = int(self.trajectory_indices[traj_idx] + offset)  # 全局步索引
            g_next = g + 1

            # 从后备存储中取出 numpy 视图，再转 torch，并 clone() 防止改写底层 memmap
            state = self._np_slice_to_torch(self.trajectories["states"][g])
            next_state = self._np_slice_to_torch(self.trajectories["states"][g_next])

            if self.normalize_states:
                d = self.normalize_state_dim
                state[..., :d] = (state[..., :d] - self.state_mean) / self.state_std
                next_state[..., :d] = (next_state[..., :d] - self.state_mean) / self.state_std

            action = self._np_slice_to_torch(self.trajectories["actions"][g])
            reward = self._np_slice_to_torch(self.trajectories["rewards"][g])
            done = self._np_slice_to_torch(self.trajectories["dones"][g].astype(np.float32))

            return (state, next_state, action, reward, done)

        else:
            # 轨迹级样本
            start = int(self.trajectory_indices[i])
            end = int(self.trajectory_indices[i+1])

            states = self._np_slice_to_torch(self.trajectories["states"][start:end])
            actions = self._np_slice_to_torch(self.trajectories["actions"][start:end])
            dones = self._np_slice_to_torch(self.trajectories["dones"][start:end].astype(np.float32))

            if self.aug is not None:
                # 你的 aug 期望 [T, ...] → 这里简单示意，如需图像增强请按实际通道顺序调整
                try:
                    states = self.aug(states)
                except Exception:
                    pass

            if self.normalize_states:
                d = self.normalize_state_dim
                states[..., :d] = (states[..., :d] - self.state_mean) / self.state_std

            T = states.size(0)
            timesteps = torch.arange(0, T, dtype=torch.long)
            attention_mask = torch.ones(T, dtype=torch.float32)
            language = '' if self.no_lang else self.trajectories["language"][i]

            # pad 到同一长度
            states = self._pad_torch(states, self.max_length)
            actions = self._pad_torch(actions, self.max_length)
            dones = self._pad_torch(dones, self.max_length)
            timesteps = self._pad_torch(timesteps, self.max_length)
            attention_mask = self._pad_torch(attention_mask, self.max_length)

            return (language, states, actions, timesteps, dones, attention_mask)

    def _np_slice_to_torch(self, x):
        """
        将 numpy / memmap 切片零拷贝转 torch.Tensor，并 clone() 以避免原地归一化写回底层。
        如果本身已经是 torch.Tensor 则直接返回 clone()。
        """
        if isinstance(x, torch.Tensor):
            return x.clone().float()
        xt = torch.from_numpy(np.array(x, copy=True)).float()
        return xt.clone()

    # ==========（可选）保持 share_memory_ 接口以兼容旧调用 ==========
    def share_memory(self):
        """
        对于 mmap 模式，这个函数无需做任何事（数据已在磁盘映射）。
        为了兼容旧代码，这里只在“内存模式”下把已有 Tensor 标记为共享。
        """
        if not self.using_memmap:
            self.state_mean.share_memory_()
            self.state_std.share_memory_()
            for k, v in self.trajectories.items():
                if isinstance(v, torch.Tensor):
                    v.share_memory_()

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
        perm = np.arange(len(trajs["states"]))
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
    trajs = {"states": [], "actions": [], "rewards": [], "lengths": [], "language": [], "dones": []}
    data = pickle.load(open(expert_location, 'rb'))
    num_trajs, traj_len = data['actions'].shape[0], data['actions'].shape[1]
    if "state" in data.keys() and kwargs['use_state']:
        trajs["states"] = data["state"]
    else:
        trajs["states"] = np.moveaxis(data["ims"], 4, 2)  # making images C,H,W
    trajs["actions"] = data["actions"]
    trajs["rewards"] = np.array([[None for _ in range(traj_len)] for _ in range(num_trajs)])
    trajs["lengths"] = np.array([traj_len]*num_trajs)
    trajs["dones"] = np.array([[0]*(traj_len-1) + [1] for _ in range(num_trajs)])

    if 'sawyer' in expert_location:
        trajs["language"] = data['langs'].reshape(-1)
    elif 'franka' in expert_location:
        trajs["states"] = np.concatenate([trajs["states"], trajs["states"]], axis=0)
        trajs["actions"] = np.concatenate([trajs["actions"], trajs["actions"]], axis=0)
        trajs["rewards"] = np.concatenate([trajs["rewards"], trajs["rewards"]], axis=0)
        trajs["lengths"] = np.concatenate([trajs["lengths"], trajs["lengths"]], axis=0)
        trajs["dones"] = np.concatenate([trajs["dones"], trajs["dones"]], axis=0)
        trajs["language"] = data['langs'].T.reshape(-1)
    else:
        raise NotImplementedError

    assert len(
        trajs["states"]) == len(
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


if __name__ == '__main__':
#     d = ExpertDataset('/atlas/u/divgarg/datasets/babyai/demos_iclr19_v2/BabyAI-GoToObj-v0_valid.pkl', 5, no_lang=True, normalize_states=False, use_direction=False)
#     print(d[0])
#     #print("**************")
#     #print(d[1])
#     del d

#     d = ExpertDataset('/atlas/u/divgarg/datasets/lorel/may_08_sawyer_50k/prep_data.pkl', 5, use_state=True)
#     print(d[0])
#     print("**************")
#     # print(d[1])
#     del d

#     d = ExpertDataset('/atlas/u/divgarg/datasets/lorel/may_08_sawyer_50k/prep_data.pkl', 5, use_state=False)
#     print(d[0])
#     print("**************")
#     print(d[1])
#     del d

#     d = ExpertDataset('/atlas/u/divgarg/datasets/lorel/may_06_franka_3k/prep_data.pkl', 5, use_state=False)
#     print(d[0])
#     print("**************")
#     # print(d[1])
#     del d

    d = ExpertDataset('/atlas/u/divgarg/datasets/calvin/dataset/task_D_D/training/', 5, normalize_states=False)
    print(d[0])
    print('****************')
    print(d[1])
    print('****************')
    del d
     
