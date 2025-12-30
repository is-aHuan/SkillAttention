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
# 小工具
# ------------------------------------------------------------
def slugify_task_name(name: str, max_len: int = 80) -> str:
    s = name.strip().lower()
    s = re.sub(r'[^a-z0-9]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s[:max_len] if max_len > 0 else s


def image_to_chw_float(img: Any, shape: list[int, int]) -> Optional[np.ndarray]:
    """
    将各种可能的图像表示统一转成:
        - float32
        - [0,1]
        - 形状 (3, H, W)
    """

    a = None  # 最终 RGB uint8 HWC
    target_h, target_w = shape  # Reszie Imahe

    # 1) path
    if isinstance(img, str) and os.path.exists(img):
        bgr = cv2.imread(img, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        # 原始是 BGR，这里转成 RGB
        a = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 2) bytes
    elif isinstance(img, (bytes, bytearray)):
        buf = np.frombuffer(img, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        a = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 3) numpy 数组
    elif isinstance(img, np.ndarray):
        x = img
        # 灰度 -> 3 通道
        if x.ndim == 2:
            x = np.stack([x, x, x], axis=-1)
        elif x.ndim == 3:
            # 如果是 CHW -> 转 HWC
            if x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
                x = np.transpose(x, (1, 2, 0))
        else:
            return None

        # 假设 x 目前是 HWC，通道就是 RGB（如果你原始数据是 BGR，这里可以改成 COLOR_BGR2RGB）
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

    # resize 到 resize shape
    if a.shape[0] != target_h or a.shape[1] != target_w:
        a = cv2.resize(a, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # uint8 -> float32 [0,1.0]
    a_float = a.astype(np.float32) / 255.0  # (H,W,3)

    # HWC -> CHW
    chw = np.transpose(a_float, (2, 0, 1))  # (3,H,W)
    return np.ascontiguousarray(chw, dtype=np.float32)

def euler_xyz_to_matrix(euler):
    """
    euler: [..., 3], XYZ 顺序, 单位: 弧度
    返回:  [..., 3, 3] 旋转矩阵
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

    # R = Rz * Ry * Rx （根据你的实际约定，注意这里只是示例）
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
    rot6d: [..., 6]，取前两列 flatten
    """
    rot6d = R[..., :, :2]   # [..., 3, 2]
    rot6d = rot6d.reshape(R.shape[:-2] + (6,))
    return rot6d.astype(np.float32)

def euler_to_rot6d(euler):
    R = euler_xyz_to_matrix(euler)
    return matrix_to_rot6d(R)

def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """
    rot6d: (..., 6), 对应 Zhou 的 6D rotation 表示:
        第一列: R[..., 0]
        第二列: R[..., 1]
    返回:
        R: (..., 3, 3)  正交旋转矩阵
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

    # 拼成 R 的三列
    R = np.stack([b1, b2, b3], axis=-1)              # (..., 3, 3)
    return R.astype(np.float32)

def matrix_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """
    R: (..., 3, 3)
    返回:
        euler: (..., 3) = [roll, pitch, yaw]，XYZ 顺序
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
    euler: [..., 3], 假设是 XYZ 顺序, 单位: 弧度
    返回:  [..., 4], 四元数 (w, x, y, z) 或你习惯的顺序
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
    """
    单条 episode 的处理逻辑：
    - 和你原来 for ep in current_file_episodes 里的代码保持一致
    - 只是打包成一个函数，方便多进程并行
    """
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

    # 构造动作 & obs（保持你原来的逻辑不变）
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
    单个进程里处理一个 pkl 文件：
      - 读取所有 episodes
      - 任务过滤
      - 调 _process_one_episode
    返回:
      local_trajs: 和 load_realman_data 返回结构一样
      found: 当前文件里总共读到多少 ep
      loaded: 真正留下多少有效 ep
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

            # 任务过滤
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
# 读取 realman pkl -> 分离 head / wrist / proprio
# ------------------------------------------------------------
def load_realman_data(
    expert_locations: Union[str, List[str]],
    target_shape: List[int],
    task_name: Optional[str] = "all",
    num_workers: int = 0,
    **kwargs
) -> Dict[str, Any]:
    """
    从以 append_pickle 写入的 realman .pkl 中读取多条 episode。
    每条 episode 形如:
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

    返回:
        head_images: [num_traj] list of (T_i, 3, H, W)
        wrist_images: [num_traj] list of (T_i, 3, H, W)
        proprio: [num_traj] list of (T_i, Dp)
        actions: [num_traj] list of (T_i, Da)
        rewards: [num_traj] list of (T_i,)
        lengths: [num_traj]
        language: [num_traj] list of str
        dones: [num_traj] list of (T_i,)
    """
    # 1) 统一 file_paths
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

    # workers 数量
    if num_workers is None or num_workers <= 0:
        num_workers = min(cpu_count(), 8)

    if num_workers == 1 or len(args_list) == 1:
        # 单进程顺序，方便 debug
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
# 方案 B: 多 memmap + 多视角归一化 Dataset
# ------------------------------------------------------------
class RealmanExpertDataset(Dataset):
    """
    方案 B (更新版):
      - 不再按 num_trajectories 截断，而是：
          * task_name != all: 只用该任务的所有轨迹
          * task_name == all: 用所有任务的所有轨迹
      - mean/std 只用“当前选择的数据子集”来计算
    """

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
        assert memmap_dir is not None, "建议 realman 一定指定 memmap_dir，减小内存压力"

        self.kwargs = kwargs
        self.full_traj = full_traj
        self.normalize_act = normalize_act
        self.normalize_obs = normalize_obs
        self.no_lang = no_lang
        self.task_name = task_name
        self.img_shape = list(image_shape)
        self.num_workers = num_workers

        # ===== memmap 目录命名 =====
        #   - task_name = "pick up the green cube" → _realman_pick_up_the_green_cube
        #   - task_name = all                      → _realman_all
        if task_name == "all":
            safe_task = "all"
        else:
            safe_task = slugify_task_name(task_name)
        self.memmap_dir = f"{memmap_dir}_realman_{safe_task}"
        os.makedirs(self.memmap_dir, exist_ok=True)

        # 各种路径（不变）
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

        # ===== rank0 写 / 其他 rank 附着 =====
        if writer and (force_rebuild or not self._memmap_exists()):
            # 1) 多进程一次性把所有需要的 episode 解析成 numpy（高内存、但很快）
            trajs = load_realman_data(
                expert_locations=expert_locations,
                target_shape=self.img_shape,
                task_name=self.task_name,
                num_workers=self.num_workers,      # ✅ 用多进程加速
            )

            # 2) 可选：按 horizon 切成小段轨迹
            if chunk_horizon is not None:
                print("[Dataset] Chunking trajectories...")
                trajs = self._chunk_trajectories(trajs, chunk_horizon)

            # 3) 一次 concat，顺手算 scale/offset + 写 memmap
            print("[Dataset] Building memmaps and computing stats...")
            self._build_memmaps_and_stats(trajs)

        # 所有 rank：attach memmap
        self._attach_memmaps()

        # 准备 trajectory 索引 & max_length
        self.max_length = int(np.max(self.trajectories["lengths"]))
        self.horizon = chunk_horizon

        if self.full_traj:
            self.length = len(self.trajectories["lengths"])
        else:
            raise NotImplementedError("当前只支持 full_traj=True")

    # --------------------------------------------------------
    # 内部工具
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
        加载 RealMan 数据，并按当前 task_name 过滤：
          - task_name != "all": load_realman_data(..., task_name=self.task_name) → 单任务
          - task_name == "all": load_realman_data(..., task_name="all")           → 所有任务
        然后只做 subsample，不再随机裁掉轨迹数量。
        """
        # 这里假设 load_realman_data 本身支持 task_name="all" 表示 "all tasks"
        trajs = load_realman_data(expert_locations, target_shape=self.img_shape, task_name=self.task_name)

        # 如果你以后又想支持“最多只取前 N 条”，可以单独定义一个 max_trajectories 参数：
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

            # lengths 也要更新
            trajs["lengths"] = [len(x) for x in trajs["head_images"]]

        return trajs

    def _build_memmaps_and_stats(self, trajs: Dict[str, List[np.ndarray]], range_eps: float = 1e-4):
        lengths = np.array(trajs["lengths"], dtype=np.int64)
        total_T = int(lengths.sum())

        # 一次 concat
        head_np   = np.concatenate(trajs["head_images"], axis=0).astype(np.float32)
        wrist_np  = np.concatenate(trajs["wrist_images"], axis=0).astype(np.float32)
        proprio_np= np.concatenate(trajs["proprio"],      axis=0).astype(np.float32)
        actions_np= np.concatenate(trajs["actions"],      axis=0).astype(np.float32)
        dones_np  = np.concatenate(trajs["dones"],        axis=0).astype(np.uint8)

        # ====== 1) 计算 scale / offset（和原来完全一样） ======
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

        # 保存 scale / offset
        np.save(self.head_scale_path,   head_scale.astype(np.float32))
        np.save(self.head_offset_path,  head_offset.astype(np.float32))
        np.save(self.wrist_scale_path,  wrist_scale.astype(np.float32))
        np.save(self.wrist_offset_path, wrist_offset.astype(np.float32))
        np.save(self.proprio_scale_path, proprio_scale.astype(np.float32))
        np.save(self.proprio_offset_path, proprio_offset.astype(np.float32))
        np.save(self.action_scale_path,  action_scale.astype(np.float32))
        np.save(self.action_offset_path, action_offset.astype(np.float32))

        # ====== 2) 写 memmap ======
        self._write_memmap(self.head_path,   head_np)
        self._write_memmap(self.wrist_path,  wrist_np)
        self._write_memmap(self.proprio_path, proprio_np)
        self._write_memmap(self.actions_path, actions_np)

        # rewards 全 0：直接创建，不用拼 list
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

        # 读取 scale 和 offset
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
        把每条原始轨迹按 horizon 切成多个子轨迹，最后一段长度 < horizon 也保留。
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
    # Dataset 接口
    # --------------------------------------------------------
    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        assert self.full_traj, "当前只实现 full_traj=True 模式"

        start = int(self.trajectory_indices[idx])
        end = int(self.trajectory_indices[idx + 1])

        head = self._np_slice_to_torch(self.trajectories["head_images"][start:end])    # (T,3,H,W)
        wrist = self._np_slice_to_torch(self.trajectories["wrist_images"][start:end])  # (T,3,H,W)
        proprio = self._np_slice_to_torch(self.trajectories["proprio"][start:end])     # (T,Dp)
        actions = self._np_slice_to_torch(self.trajectories["actions"][start:end])     # (T,Da)
        dones = self._np_slice_to_torch(self.trajectories["dones"][start:end]).float() # (T,)

        T = head.size(0)

        # 归一化
        if self.normalize_obs:
            # 使用 scale 和 offset 进行归一化
            head = (head * self.head_scale[None, :, None, None]) + self.head_offset[None, :, None, None]
            wrist = (wrist * self.wrist_scale[None, :, None, None]) + self.wrist_offset[None, :, None, None]
            proprio = (proprio * self.proprio_scale[None, :]) + self.proprio_offset[None, :]

        if self.normalize_act:
            actions = (actions * self.action_scale[None, :]) + self.action_offset[None, :]

        timesteps = torch.arange(0, T, dtype=torch.long)
        attention_mask = torch.ones(T, dtype=torch.float32)
        language = '' if self.no_lang else self.trajectories["language"][idx]

        # pad 到 horizon
        L = self.horizon
        head = self._pad_torch(head, L)
        wrist = self._pad_torch(wrist, L)
        proprio = self._pad_torch(proprio, L)
        actions = self._pad_torch(actions, L)
        dones = self._pad_torch(dones, L)
        timesteps = self._pad_torch(timesteps, L)
        attention_mask = self._pad_torch(attention_mask, L)

        # obs dict 更接近 Diffusion Policy 的接口
        obs = {
            'head_image': head,      # (T_max, 3, H, W)
            'wrist_image': wrist,    # (T_max, 3, H, W)
            'proprio': proprio       # (T_max, Dp)
        }

        return language, obs, actions, timesteps, dones, attention_mask


# ------------------------------------------------------------
# 简单测试
# ------------------------------------------------------------
if __name__ == "__main__":
    # 你可以在本地测试一下：
    # 注意请改成你自己的 pkl 路径和 task 名字
    pkl_path = "/home/lyh/data_folder/dataspace/lorel/may_08_sawyer_50k/realman.pkl"
    memmap_dir = "/home/lyh/data_folder/dataspace/lorel/may_08_sawyer_50k/memmap_cache"

    # 第一次跑用 writer=True 建 memmap
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
        chunk_horizon=50,  # 可选：按 horizon 切成子轨迹
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
