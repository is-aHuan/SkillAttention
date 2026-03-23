import os
import re
import json
import pickle
from typing import Any, Dict, IO, List, Union, Optional
import cv2

import numpy as np
import torch
from torch.utils.data import Dataset

from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ------------------------------------------------------------
# tools
# ------------------------------------------------------------
def slugify_task_name(name: str, max_len: int = 80) -> str:
    s = name.strip().lower()
    s = re.sub(r'[^a-z0-9]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s[:max_len] if max_len > 0 else s


def image_to_chw_float(img: Any, shape: list[int, int]) -> Optional[np.ndarray]:
    """
    Normalize possible image representations into:
        - float32
        - [0,1]
        - shape (3, H, W)
    """

    a = None  # final RGB uint8 HWC
    target_h, target_w = shape  # Resize Image

    # 1) path
    if isinstance(img, str) and os.path.exists(img):
        bgr = cv2.imread(img, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        # Original is BGR, convert to RGB here
        a = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 2) bytes
    elif isinstance(img, (bytes, bytearray)):
        buf = np.frombuffer(img, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        a = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 3) numpy array
    elif isinstance(img, np.ndarray):
        x = img
        # grayscale -> 3 channels
        if x.ndim == 2:
            x = np.stack([x, x, x], axis=-1)
        elif x.ndim == 3:
            # if CHW -> convert to HWC
            if x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
                x = np.transpose(x, (1, 2, 0))
        else:
            return None

        # Assume x is currently HWC, channels are RGB (if your original data is BGR, you can change this to COLOR_BGR2RGB)
        if x.dtype != np.uint8:
            if x.max() <= 1.0:
                x = (x * 255.0).clip(0, 255).astype(np.uint8)
            else:
                x = x.clip(0, 255).astype(np.uint8)
        a = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    else:
        return None

    if a is None or a.ndim != 3 or a.shape[-1] != 3:
        return None

    # resize shape
    if a.shape[0] != target_h or a.shape[1] != target_w:
        a = cv2.resize(a, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # uint8 -> float32 [0,1.0]
    a_float = a.astype(np.float32) / 255.0  # (H,W,3)

    # HWC -> CHW
    chw = np.transpose(a_float, (2, 0, 1))  # (3,H,W)
    return np.ascontiguousarray(chw, dtype=np.float32)

def euler_xyz_to_matrix(euler):
    """
    euler: [..., 3], XYZ, units: radians
    Returns:  [..., 3, 3] rotation matrix
    """
    roll = euler[..., 0]
    pitch = euler[..., 1]
    yaw = euler[..., 2]

    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # R = Rz * Ry * Rx
    R = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float32)

    R[..., 0, 0] = cy * cp
    R[..., 0, 1] = cy * sp * sr - sy * cr
    R[..., 0, 2] = cy * sp * cr + sy * sr

    R[..., 1, 0] = sy * cp
    R[..., 1, 1] = sy * sp * sr + cy * cr
    R[..., 1, 2] = sy * sp * cr - cy * sr

    R[..., 2, 0] = -sp
    R[..., 2, 1] = cp * sr
    R[..., 2, 2] = cp * cr

    return R

def matrix_to_rot6d(R):
    """
    R: [..., 3, 3]
    rot6d: [..., 6], flatten the first two columns
    """
    rot6d = R[..., :, :2]   # [..., 3, 2]
    rot6d = rot6d.reshape(R.shape[:-2] + (6,))
    return rot6d.astype(np.float32)

def euler_to_rot6d(euler):
    R = euler_xyz_to_matrix(euler)
    return matrix_to_rot6d(R)

def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """
    rot6d: (..., 6), corresponding 6D rotation representation:
        first column: R[..., 0]
        second column: R[..., 1]
    Returns:
        R: (..., 3, 3)  rotation matrix
    """
    rot6d = rot6d.reshape(*rot6d.shape[:-1], 3, 2)   # (..., 3, 2)
    a1 = rot6d[..., 0]                               # (..., 3)
    a2 = rot6d[..., 1]                               # (..., 3)

    # b1 = normalize(a1)
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)

    # b2 = normalize(a2 - (b1·a2) b1)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)    # (...,1)
    a2_ortho = a2 - dot * b1
    b2 = a2_ortho / (np.linalg.norm(a2_ortho, axis=-1, keepdims=True) + 1e-8)

    # b3 = b1 × b2
    b3 = np.cross(b1, b2, axis=-1)                   # (...,3)

    R = np.stack([b1, b2, b3], axis=-1)              # (..., 3, 3)
    return R.astype(np.float32)

def matrix_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """
    R: (..., 3, 3)
    Returns:
        euler: (..., 3) = [roll, pitch, yaw], XYZ order, units: radians
    """
    # pitch = -asin(R[2,0])
    pitch = -np.arcsin(R[..., 2, 0])
    # roll  = atan2(R[2,1], R[2,2])
    roll  = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    # yaw   = atan2(R[1,0], R[0,0])
    yaw   = np.arctan2(R[..., 1, 0], R[..., 0, 0])

    euler = np.stack([roll, pitch, yaw], axis=-1)
    return euler.astype(np.float32)

def euler_xyz_to_quat(euler):
    """
    euler: [..., 3], XYZ , units: radians
    returns:  [..., 4], quaternion (w, x, y, z) 
    """
    roll = euler[..., 0]
    pitch = euler[..., 1]
    yaw = euler[..., 2]

    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    quat = np.stack([qw, qx, qy, qz], axis=-1)
    return quat.astype(np.float32)

def read_file(path: str, file_handle: IO[Any]) -> Dict[str, Any]:
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

def _process_one_episode(
    ep: Dict[str, Any],
    target_shape: List[int],
) -> Optional[Dict[str, np.ndarray]]:
    steps = ep.get("steps", [])
    if not steps:
        return None

    lang = (ep.get("metadata", {}) or {}).get("language_instruction", "")
    head_buf, wrist_buf = [], []
    proprio_buf = []
    dones_buf = []

    for s_idx, step in enumerate(steps):
        obs = step.get("observation", {})
        imgs = obs.get("images", {})

        head_raw = imgs.get("image_side")
        wrist_raw = imgs.get("image_wrist")

        head_img = image_to_chw_float(head_raw, target_shape)
        wrist_img = image_to_chw_float(wrist_raw, target_shape)

        ee_pose = obs.get("ee_pose", None)
        qpos    = obs.get("qpos", None)

        if ee_pose is None or qpos is None:
            continue

        ee_pose = np.asarray(ee_pose, dtype=np.float32).reshape(-1)
        qpos    = np.asarray(qpos, dtype=np.float32).reshape(-1)
        grip_pos = qpos[6:7]

        proprio = np.concatenate([ee_pose, grip_pos], axis=-1)

        if head_img is None or wrist_img is None or proprio.size == 0:
            continue

        head_buf.append(head_img)
        wrist_buf.append(wrist_img)
        proprio_buf.append(proprio)

        is_last = bool(step.get("is_last", False)) or (s_idx == len(steps) - 1)
        dones_buf.append(1 if is_last else 0)

    if len(head_buf) == 0:
        return None

    head_arr = np.stack(head_buf, axis=0).astype(np.float32)
    wrist_arr = np.stack(wrist_buf, axis=0).astype(np.float32)
    proprio_arr = np.stack(proprio_buf, axis=0).astype(np.float32)
    dones_arr = np.array(dones_buf, dtype=np.uint8)

    T = head_arr.shape[0]
    if T < 2:
        return None

    # Construct actions & observations
    pos   = proprio_arr[:, 0:3]
    euler = proprio_arr[:, 3:6]
    grip  = proprio_arr[:, 6:7]
    rot6d = euler_to_rot6d(euler)

    # action: next pose
    act_pos   = pos[1:]
    act_rot6d = rot6d[1:]
    act_grip  = grip[1:]
    actions_arr = np.concatenate([act_pos, act_rot6d, act_grip], axis=-1)
    dones_arr   = dones_arr[1:]

    # obs: current
    head_arr    = head_arr[:-1]
    wrist_arr   = wrist_arr[:-1]
    proprio_arr = proprio_arr[:-1]

    # proprio: euler -> quat
    euler_xyz = proprio_arr[:, 3:6]
    quat_arr = euler_xyz_to_quat(euler_xyz)
    proprio_arr = np.concatenate(
        [proprio_arr[:, :3], quat_arr, proprio_arr[:, 6:7]],
        axis=-1
    )

    T_new = head_arr.shape[0]

    return dict(
        head=head_arr,
        wrist=wrist_arr,
        proprio=proprio_arr,
        actions=actions_arr,
        dones=dones_arr,
        rewards=np.zeros((T_new,), dtype=np.float32),
        length=T_new,
        language=lang,
    )

def _process_one_file(args):
    """
    Process a single .pkl file within one worker/process:
      - read all episodes
      - apply task filtering
      - call _process_one_episode for each episode
    Returns:
      local_trajs: same structure as returned by load_realman_data
      found: total number of episodes read from this file
      loaded: number of valid episodes retained/loaded
    """
    f_path, target_shape, task_key = args

    local_trajs: Dict[str, List] = {
        "head_images": [],
        "wrist_images": [],
        "proprio": [],
        "actions": [],
        "rewards": [],
        "lengths": [],
        "language": [],
        "dones": [],
    }

    found = 0
    loaded = 0

    if not os.path.exists(f_path):
        print(f"[Worker] File not found: {f_path}, skip")
        return local_trajs, found, loaded

    print(f"[Worker] Processing file: {f_path}")

    with open(f_path, "rb") as f:
        while True:
            try:
                ep = pickle.load(f)
            except EOFError:
                break

            found += 1

            # task filter
            if task_key is not None:
                meta = ep.get("metadata", {}) or {}
                instr = str(meta.get("language_instruction", "")).lower()
                if task_key not in instr:
                    continue

            out = _process_one_episode(ep, target_shape)
            if out is None:
                continue

            local_trajs["head_images"].append(out["head"])
            local_trajs["wrist_images"].append(out["wrist"])
            local_trajs["proprio"].append(out["proprio"])
            local_trajs["actions"].append(out["actions"])
            local_trajs["rewards"].append(out["rewards"])
            local_trajs["dones"].append(out["dones"])
            local_trajs["lengths"].append(out["length"])
            local_trajs["language"].append(out["language"])

            loaded += 1

    return local_trajs, found, loaded

# ------------------------------------------------------------
# load realman pkl -> get head / wrist / proprio
# ------------------------------------------------------------
def load_realman_data(
    expert_locations: Union[str, List[str]],
    target_shape: List[int],
    task_name: Optional[str] = "all",
    num_workers: int = 0,
    **kwargs
) -> Dict[str, Any]:
    """
    Read multiple episodes from a realman .pkl file written with append_pickle.
    Each episode has the form:
    {
      "metadata": { "language_instruction": str, ... },
      "steps": [
        {
          "observation": {
            "images": {
              "image_side": ...,
              "image_wrist": ...
            },
            "ee_pose": np.array([...])  # or something similar
          },
          "action": [...],
          "is_last": bool,
          ...
        },
        ...
      ]
    }

    Returns:
        head_images: [num_traj] list of (T_i, 3, H, W)
        wrist_images: [num_traj] list of (T_i, 3, H, W)
        proprio: [num_traj] list of (T_i, Dp)
        actions: [num_traj] list of (T_i, Da)
        rewards: [num_traj] list of (T_i,)
        lengths: [num_traj] list of ints
        language: [num_traj] list of str
        dones: [num_traj] list of (T_i,)
    """
    # 1) file_paths
    if isinstance(expert_locations, str):
        file_paths = [expert_locations]
    else:
        file_paths = expert_locations

    if not file_paths:
        raise ValueError("expert_locations 为空")

    trajs: Dict[str, List] = {
        "head_images": [],
        "wrist_images": [],
        "proprio": [],
        "actions": [],
        "rewards": [],
        "lengths": [],
        "language": [],
        "dones": []
    }

    task_name = task_name or "all"
    task_key = None if task_name == "all" else task_name.lower().replace("_", " ")

    total_found = 0
    total_loaded = 0

    args_list = [(p, target_shape, task_key) for p in file_paths]

    # num_workers 
    if num_workers is None or num_workers <= 0:
        num_workers = min(cpu_count(), 8)

    if num_workers == 1 or len(args_list) == 1:
        # Single process sequential, convenient for debugging
        for args in args_list:
            local_trajs, found, loaded = _process_one_file(args)
            total_found += found
            total_loaded += loaded
            for k in trajs.keys():
                trajs[k].extend(local_trajs[k])
    else:
        print(f"[Realman] Using {num_workers} workers over {len(args_list)} files")
        with Pool(processes=num_workers) as pool:
            for local_trajs, found, loaded in tqdm(
                pool.imap_unordered(_process_one_file, args_list, chunksize=1),
                total=len(args_list),
                desc="Processing files"
            ):
                total_found += found
                total_loaded += loaded
                for k in trajs.keys():
                    trajs[k].extend(local_trajs[k])

    if total_loaded == 0:
        raise ValueError(f"No valid episodes found in: {file_paths}")

    print(f"Total episodes found:  {total_found}")
    print(f"Total episodes loaded: {total_loaded}")
    return trajs


# ------------------------------------------------------------
# Scheme: Multiple memmaps + multi-view normalization Dataset
# ------------------------------------------------------------
class RealmanExpertDataset(Dataset):
    def __init__(self,
                 expert_locations: Union[str, List[str]],
                 subsample_frequency: int = 1,
                 seed: int = 0,
                 full_traj: bool = True,
                 normalize_act: bool = True,
                 normalize_obs: bool = False,
                 no_lang: bool = False,
                 memmap_dir: Optional[str] = None,
                 writer: bool = False,
                 force_rebuild: bool = False,
                 task_name: Optional[str] = "all",
                 chunk_horizon: Optional[int] = None,
                 image_shape: List[int] = [240, 320],
                 num_workers: int = 24,
                 **kwargs):
        super().__init__()
        assert memmap_dir is not None, "It is recommended to always specify memmap_dir for realman to reduce memory pressure"

        self.kwargs = kwargs
        self.full_traj = full_traj
        self.normalize_act = normalize_act
        self.normalize_obs = normalize_obs
        self.no_lang = no_lang
        self.task_name = task_name
        self.img_shape = list(image_shape)
        self.num_workers = num_workers

        # ===== memmap directory naming =====
        #   - task_name = "pick up the green cube" → _realman_pick_up_the_green_cube
        #   - task_name = all                      → _realman_all
        if task_name == "all":
            safe_task = "all"
        else:
            safe_task = slugify_task_name(task_name)
        self.memmap_dir = f"{memmap_dir}_realman_{safe_task}"
        os.makedirs(self.memmap_dir, exist_ok=True)

        self.meta_path = os.path.join(self.memmap_dir, "meta.json")
        self.lengths_path = os.path.join(self.memmap_dir, "lengths.npy")
        self.lang_path = os.path.join(self.memmap_dir, "language.json")

        self.head_path = os.path.join(self.memmap_dir, "head_images.mmap")
        self.wrist_path = os.path.join(self.memmap_dir, "wrist_images.mmap")
        self.proprio_path = os.path.join(self.memmap_dir, "proprio.mmap")
        self.actions_path = os.path.join(self.memmap_dir, "actions.mmap")
        self.rewards_path = os.path.join(self.memmap_dir, "rewards.mmap")
        self.dones_path = os.path.join(self.memmap_dir, "dones.mmap")

        self.head_scale_path = os.path.join(self.memmap_dir, "head_scale.npy")
        self.head_offset_path = os.path.join(self.memmap_dir, "head_offset.npy")
        self.wrist_scale_path = os.path.join(self.memmap_dir, "wrist_scale.npy")
        self.wrist_offset_path = os.path.join(self.memmap_dir, "wrist_offset.npy")
        self.proprio_scale_path = os.path.join(self.memmap_dir, "proprio_scale.npy")
        self.proprio_offset_path = os.path.join(self.memmap_dir, "proprio_offset.npy")
        self.action_scale_path = os.path.join(self.memmap_dir, "action_scale.npy")
        self.action_offset_path = os.path.join(self.memmap_dir, "action_offset.npy")

        # ===== rank0 write / other ranks attach =====
        if writer and (force_rebuild or not self._memmap_exists()):
            # 1) Use multiprocessing to parse all required episodes into numpy arrays at once (high memory, but fast)
            trajs = load_realman_data(
                expert_locations=expert_locations,
                target_shape=self.img_shape,
                task_name=self.task_name,
                num_workers=self.num_workers,      # Use multiprocessing to speed up
            )

            # 2) Optional: chunk trajectories by horizon
            if chunk_horizon is not None:
                print("[Dataset] Chunking trajectories...")
                trajs = self._chunk_trajectories(trajs, chunk_horizon)

            # 3) Concatenate, calculate scale/offset, and write memmap
            print("[Dataset] Building memmaps and computing stats...")
            self._build_memmaps_and_stats(trajs)

        # All ranks: attach memmap
        self._attach_memmaps()

        # Prepare trajectory indices & max_length
        self.max_length = int(np.max(self.trajectories["lengths"]))
        self.horizon = chunk_horizon

        if self.full_traj:
            self.length = len(self.trajectories["lengths"])
        else:
            raise NotImplementedError("当前只支持 full_traj=True")

    # --------------------------------------------------------
    # Tools
    # --------------------------------------------------------
    def _memmap_exists(self) -> bool:
        paths = [
            self.meta_path, self.lengths_path, self.lang_path,
            self.head_path, self.wrist_path, self.proprio_path,
            self.actions_path, self.rewards_path, self.dones_path
        ]
        return all(os.path.exists(p) for p in paths)

    def _load_and_subsample(self,
                            expert_locations: Union[str, List[str]],
                            ss: int,
                            seed: int):
        """
        Load RealMan data and filter by the current task_name:
          - task_name != "all": load_realman_data(..., task_name=self.task_name) → single task
          - task_name == "all": load_realman_data(..., task_name="all")           → all tasks
        Then only do subsample, no longer randomly cut the number of trajectories.
        """
        trajs = load_realman_data(expert_locations, target_shape=self.img_shape, task_name=self.task_name)

        # max_traj = getattr(self, "max_trajectories", None)
        # if max_traj is not None:
        #     for k, v in trajs.items():
        #         trajs[k] = v[:max_traj]

        # ---------- subsample ----------
        if ss > 1:
            for k, v in trajs.items():
                if k in ("lengths", "language"):
                    continue
                new_list = []
                for arr in v:
                    new_list.append(arr[0::ss])
                trajs[k] = new_list

            # lengths must be updated too
            trajs["lengths"] = [len(x) for x in trajs["head_images"]]

        return trajs

    def _build_memmaps_and_stats(self, trajs: Dict[str, List[np.ndarray]], range_eps: float = 1e-4):
        lengths = np.array(trajs["lengths"], dtype=np.int64)
        total_T = int(lengths.sum())

        # concat
        head_np   = np.concatenate(trajs["head_images"], axis=0).astype(np.float32)
        wrist_np  = np.concatenate(trajs["wrist_images"], axis=0).astype(np.float32)
        proprio_np= np.concatenate(trajs["proprio"],      axis=0).astype(np.float32)
        actions_np= np.concatenate(trajs["actions"],      axis=0).astype(np.float32)
        dones_np  = np.concatenate(trajs["dones"],        axis=0).astype(np.uint8)

        # ====== 1) Calculate scale / offset ======
        head_min = head_np.min(axis=(0, 2, 3))
        head_max = head_np.max(axis=(0, 2, 3))
        head_range = head_max - head_min
        head_scale = np.where(head_range < range_eps, 1.0, 2.0 / (head_max - head_min))
        head_offset = np.where(head_range < range_eps, -head_min, -(head_max + head_min) / (head_max - head_min))

        wrist_min = wrist_np.min(axis=(0, 2, 3))
        wrist_max = wrist_np.max(axis=(0, 2, 3))
        wrist_range = wrist_max - wrist_min
        wrist_scale = np.where(wrist_range < range_eps, 1.0, 2.0 / (wrist_max - wrist_min))
        wrist_offset = np.where(wrist_range < range_eps, -wrist_min, -(wrist_max + wrist_min) / (wrist_max - wrist_min))

        proprio_min = proprio_np.min(axis=0)
        proprio_max = proprio_np.max(axis=0)
        proprio_range = proprio_max - proprio_min
        proprio_scale = np.where(proprio_range < range_eps, 1.0, 2.0 / (proprio_max - proprio_min))
        proprio_offset = np.where(proprio_range < range_eps, -proprio_min, -(proprio_max + proprio_min) / (proprio_max - proprio_min))

        action_min = actions_np.min(axis=0)
        action_max = actions_np.max(axis=0)
        action_range = action_max - action_min
        action_scale = np.where(action_range < range_eps, 1.0, 2.0 / (action_max - action_min))
        action_offset = np.where(action_range < range_eps, -action_min, -(action_max + action_min) / (action_max - action_min))

        # save scale / offset
        np.save(self.head_scale_path,   head_scale.astype(np.float32))
        np.save(self.head_offset_path,  head_offset.astype(np.float32))
        np.save(self.wrist_scale_path,  wrist_scale.astype(np.float32))
        np.save(self.wrist_offset_path, wrist_offset.astype(np.float32))
        np.save(self.proprio_scale_path, proprio_scale.astype(np.float32))
        np.save(self.proprio_offset_path, proprio_offset.astype(np.float32))
        np.save(self.action_scale_path,  action_scale.astype(np.float32))
        np.save(self.action_offset_path, action_offset.astype(np.float32))

        # ====== 2) Write memmap ======
        self._write_memmap(self.head_path,   head_np)
        self._write_memmap(self.wrist_path,  wrist_np)
        self._write_memmap(self.proprio_path, proprio_np)
        self._write_memmap(self.actions_path, actions_np)

        # rewards all 0: create directly, no need to concatenate list
        rewards_np = np.zeros((total_T,), dtype=np.float32)
        self._write_memmap(self.rewards_path, rewards_np)
        self._write_memmap(self.dones_path,   dones_np)

        np.save(self.lengths_path, lengths)
        with open(self.lang_path, "w", encoding="utf-8") as f:
            json.dump(trajs.get("language", [""] * len(lengths)), f, ensure_ascii=False)

        meta = dict(
            total_timesteps=int(total_T),
            num_trajectories=int(len(lengths)),
            head_shape=head_np.shape[1:],
            wrist_shape=wrist_np.shape[1:],
            proprio_shape=proprio_np.shape[1:],
            action_shape=actions_np.shape[1:],
            dtype_head="float32",
            dtype_wrist="float32",
            dtype_proprio="float32",
            dtype_actions="float32",
            dtype_rewards="float32",
            dtype_dones="uint8",
        )
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def _attach_memmaps(self):
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.meta = meta

        lengths = np.load(self.lengths_path).astype(np.int64)
        self.trajectories: Dict[str, Any] = {}
        self.trajectories["lengths"] = lengths
        self.trajectory_indices = np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)])
        total_T = int(meta["total_timesteps"])

        head_tail = tuple(meta["head_shape"])
        wrist_tail = tuple(meta["wrist_shape"])
        proprio_tail = tuple(meta["proprio_shape"])
        action_tail = tuple(meta["action_shape"])

        self.trajectories["head_images"] = np.memmap(
            self.head_path, mode="r", dtype=np.float32, shape=(total_T,) + head_tail
        )
        self.trajectories["wrist_images"] = np.memmap(
            self.wrist_path, mode="r", dtype=np.float32, shape=(total_T,) + wrist_tail
        )
        self.trajectories["proprio"] = np.memmap(
            self.proprio_path, mode="r", dtype=np.float32, shape=(total_T,) + proprio_tail
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

        # load scale and offset
        self.head_scale = np.load(self.head_scale_path).astype(np.float32)
        self.head_offset = np.load(self.head_offset_path).astype(np.float32)
        self.wrist_scale = np.load(self.wrist_scale_path).astype(np.float32)
        self.wrist_offset = np.load(self.wrist_offset_path).astype(np.float32)
        self.proprio_scale = np.load(self.proprio_scale_path).astype(np.float32)
        self.proprio_offset = np.load(self.proprio_offset_path).astype(np.float32)
        self.action_scale = np.load(self.action_scale_path).astype(np.float32)
        self.action_offset = np.load(self.action_offset_path).astype(np.float32)

    def _write_memmap(self, path: str, arr: np.ndarray):
        mm = np.memmap(path, mode="w+", dtype=arr.dtype, shape=arr.shape)
        mm[:] = arr
        del mm  # flush


    @staticmethod
    def _pad_torch(x: torch.Tensor, max_len: int) -> torch.Tensor:
        if x.size(0) == max_len:
            return x
        out_shape = (max_len,) + tuple(x.shape[1:])
        out = torch.zeros(out_shape, dtype=x.dtype)
        out[:x.size(0)] = x
        return out

    def _np_slice_to_torch(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.clone().float()
        return torch.from_numpy(np.array(x, copy=True)).float()

    def _chunk_trajectories(self, trajs: Dict[str, List[np.ndarray]], horizon: int) -> Dict[str, List[np.ndarray]]:
        """
        Split each original trajectory into multiple sub-trajectories by horizon, keeping the last segment even if its length < horizon.
        """
        new_traj: Dict[str, List] = {
            "head_images": [],
            "wrist_images": [],
            "proprio": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "lengths": [],
            "language": []
        }

        for i in range(len(trajs["head_images"])):
            head_i = trajs["head_images"][i]
            wrist_i = trajs["wrist_images"][i]
            proprio_i = trajs["proprio"][i]
            actions_i = trajs["actions"][i]
            rewards_i = trajs["rewards"][i]
            dones_i = trajs["dones"][i]
            L = trajs["lengths"][i]
            lang_i = trajs["language"][i]

            start = 0
            while start < L:
                end = min(start + horizon, L)
                new_traj["head_images"].append(head_i[start:end])
                new_traj["wrist_images"].append(wrist_i[start:end])
                new_traj["proprio"].append(proprio_i[start:end])
                new_traj["actions"].append(actions_i[start:end])
                new_traj["rewards"].append(rewards_i[start:end])
                new_traj["dones"].append(dones_i[start:end])
                new_traj["lengths"].append(end - start)
                new_traj["language"].append(lang_i)
                start = end

        return new_traj

    # --------------------------------------------------------
    # Dataset api
    # --------------------------------------------------------
    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        assert self.full_traj, "Currently only full_traj=True mode is implemented"

        start = int(self.trajectory_indices[idx])
        end = int(self.trajectory_indices[idx + 1])

        head = self._np_slice_to_torch(self.trajectories["head_images"][start:end])    # (T,3,H,W)
        wrist = self._np_slice_to_torch(self.trajectories["wrist_images"][start:end])  # (T,3,H,W)
        proprio = self._np_slice_to_torch(self.trajectories["proprio"][start:end])     # (T,Dp)
        actions = self._np_slice_to_torch(self.trajectories["actions"][start:end])     # (T,Da)
        dones = self._np_slice_to_torch(self.trajectories["dones"][start:end]).float() # (T,)

        T = head.size(0)

        # Normalization
        if self.normalize_obs:
            # Use scale and offset 
            head = (head * self.head_scale[None, :, None, None]) + self.head_offset[None, :, None, None]
            wrist = (wrist * self.wrist_scale[None, :, None, None]) + self.wrist_offset[None, :, None, None]
            proprio = (proprio * self.proprio_scale[None, :]) + self.proprio_offset[None, :]

        if self.normalize_act:
            actions = (actions * self.action_scale[None, :]) + self.action_offset[None, :]

        timesteps = torch.arange(0, T, dtype=torch.long)
        attention_mask = torch.ones(T, dtype=torch.float32)
        language = '' if self.no_lang else self.trajectories["language"][idx]

        # pad to horizon
        L = self.horizon
        head = self._pad_torch(head, L)
        wrist = self._pad_torch(wrist, L)
        proprio = self._pad_torch(proprio, L)
        actions = self._pad_torch(actions, L)
        dones = self._pad_torch(dones, L)
        timesteps = self._pad_torch(timesteps, L)
        attention_mask = self._pad_torch(attention_mask, L)

        obs = {
            'head_image': head,      # (T_max, 3, H, W)
            'wrist_image': wrist,    # (T_max, 3, H, W)
            'proprio': proprio       # (T_max, Dp)
        }

        return language, obs, actions, timesteps, dones, attention_mask


# ------------------------------------------------------------
# Test
# ------------------------------------------------------------
if __name__ == "__main__":
    pkl_path = "/home/lyh/data_folder/dataspace/lorel/may_08_sawyer_50k/realman.pkl"
    memmap_dir = "/home/lyh/data_folder/dataspace/lorel/may_08_sawyer_50k/memmap_cache"

    # First run with writer=True to build memmap
    ds = RealmanExpertDataset(
        expert_location=pkl_path,
        subsample_frequency=1,
        seed=0,
        full_traj=True,
        normalize_obs=True,
        no_lang=False,
        memmap_dir=memmap_dir,
        writer=True,
        force_rebuild=False,
        task_name="pick up the green cube",
        chunk_horizon=50,  # Split into sub-trajectories by horizon
    )

    print("len(dataset) =", len(ds))
    lang, obs, act, t, d, mask = ds[0]
    print("language:", lang)
    print("head_image:", obs["head_image"].shape)
    print("wrist_image:", obs["wrist_image"].shape)
    print("proprio:", obs["proprio"].shape)
    print("actions:", act.shape)
    print("timesteps:", t.shape)
    print("mask:", mask.shape)
