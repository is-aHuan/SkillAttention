import argparse
import json
import os
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import asyncio

from omegaconf import DictConfig
from policy.diffusion_unet_hybrid_image_policy import SkillAttentionUnetModel  
# from policy.diffusion_unet_hybrid_image_policy import SkillAttentionUnetModel
# from policy.diffusion_transformer_hybrid_image_policy import SkillAttentionTransformerModel
from policy.diffusion_transformer_with_lang import SkillAttentionTransformerLangModel

# ----------------------- FastAPI app & globals -----------------------
app = FastAPI()

model: Optional[SkillAttentionUnetModel] = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 每个 websocket 维护一个 session
sessions: Dict[WebSocket, "InferenceSession"] = {}


# ----------------------- 一些小工具函数 -----------------------
def _to_chw_float(x: Any) -> torch.Tensor:
    """
    将用户传来的 head / wrist 图像转成:
        - numpy float32
        - [0,1]
        - 形状 (3, H, W)
    支持 HWC / CHW，通道 3 或 1（灰度会三通道复制）。
    """
    arr = np.array(x, dtype=np.float32)

    if arr.ndim != 3:
        raise ValueError(f"image ndim must be 3, got shape {arr.shape}")

    # HWC -> CHW
    if arr.shape[-1] in (1, 3):
        arr = np.transpose(arr, (2, 0, 1))
    elif arr.shape[0] in (1, 3):
        # 已经是 CHW
        pass
    else:
        raise ValueError(f"cannot infer channel dimension from shape {arr.shape}")

    # [0,255] -> [0,1]
    if arr.max() > 1.0:
        arr = arr / 255.0

    return torch.from_numpy(arr.astype(np.float32))

# ----------------------- InferenceSession -----------------------
class InferenceSession:
    """
    每个 WebSocket 维护一个 Session：
      - 累积多帧 head / wrist / proprio
      - 记录语言指令
      - 在需要时调用 model.get_action
    """
    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device

        # 仅保存最近两帧观测数据
        self.head_frames: List[torch.Tensor] = []     # 每帧 [3,H,W]
        self.wrist_frames: List[torch.Tensor] = []    # 每帧 [3,H,W]
        self.proprio_frames: List[torch.Tensor] = []  # 每帧 [Dp]
        self.lang: Optional[str] = None

    def append_observation(
        self,
        head_img: Any,
        wrist_img: Any,
        proprio: Any,
        lang: Optional[str] = None,
    ):
        # 1) 图像 -> CHW float tensor
        head_chw = _to_chw_float(head_img).to(self.device)
        wrist_chw = _to_chw_float(wrist_img).to(self.device)

        # 2) proprio -> 1D float tensor
        if proprio is None:
            raise ValueError("proprio (机器人状态向量) 不能为空")
        proprio_arr = np.array(proprio, dtype=np.float32)
        if proprio_arr.ndim != 1:
            raise ValueError(f"proprio must be 1D, got shape {proprio_arr.shape}")
        if proprio_arr.shape[0] != self.model.proprio_dim:
            raise ValueError(
                f"proprio dim mismatch: got {proprio_arr.shape[0]}, "
                f"expected {self.model.proprio_dim}"
            )
        proprio_t = torch.from_numpy(proprio_arr).to(self.device)

        # 3) 追加到历史，仅保留最近两帧
        self.head_frames.append(head_chw)
        self.wrist_frames.append(wrist_chw)
        self.proprio_frames.append(proprio_t)

        # 保证历史帧数最多为 self.model.n_obs_steps 
        if len(self.head_frames) > self.model.n_obs_steps:
            self.head_frames.pop(0)
        if len(self.wrist_frames) > self.model.n_obs_steps:
            self.wrist_frames.pop(0)
        if len(self.proprio_frames) > self.model.n_obs_steps:
            self.proprio_frames.pop(0)

        # 4) 初次设置语言
        if lang is not None and self.lang is None:
            self.lang = lang

    @torch.no_grad()
    def plan_actions(self) -> List[List[float]]:
        """
        构造 obs / timesteps / attention_mask 调用 SkillAttentionModel.get_action。
        返回 [n_action_steps, action_dim] 的 Python list。
        """
        if len(self.head_frames) == 0:
            raise RuntimeError("no observation yet")

        # 使用最近 n_obs_steps 帧
        K = self.model.n_obs_steps 

        head = torch.stack(self.head_frames[-K:], dim=0)    # [K,3,H,W]
        wrist = torch.stack(self.wrist_frames[-K:], dim=0)  # [K,3,H,W]
        proprio = torch.stack(self.proprio_frames[-K:], dim=0)  # [K,Dp]

        # 扩 batch 维：B=1
        head = head.unsqueeze(0)      # [1,K,3,H,W]
        wrist = wrist.unsqueeze(0)    # [1,K,3,H,W]
        proprio = proprio.unsqueeze(0)  # [1,K,Dp]K

        # timesteps 就简单用 [0..1]，和训练中 step 编码一致即可
        timesteps = torch.arange(K, device=self.device, dtype=torch.long).unsqueeze(0)  # [1,K]
        attention_mask = torch.ones(1, K, dtype=torch.long, device=self.device)         # [1,2]

        # 调用模型
        actions = self.model.get_action(
            langs=[self.lang or ""],
            obs={
                "head_image": head,
                "wrist_image": wrist,
                "proprio": proprio,
            },
            timesteps=timesteps,
            attention_mask=attention_mask,
        )  # [n_action_steps, action_dim]

        return actions.tolist()


# ----------------------- WebSocket handler -----------------------
@app.websocket("/ws/infer")
async def ws_infer(ws: WebSocket):
    await ws.accept()
    sess: Optional[InferenceSession] = None

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            # 兼容两种输入形式：
            # 1) 直接 head / wrist
            # 2) 只有 state（6 通道） -> 自动切成 head / wrist
            head = msg.get("head")
            wrist = msg.get("wrist")

            if head is None or wrist is None:
                await ws.send_text(json.dumps({
                    "ok": False,
                    "error": "request must contain 'head'/'wrist' 或者 'state' (6通道)"
                }))
                continue

            proprio = msg.get("proprio")
            lang = msg.get("lang")
            request_action = bool(msg.get("request_action", True))

            global model, device
            # 第一次消息时创建 session
            if sess is None:
                if model is None:
                    await ws.send_text(json.dumps({"ok": False, "error": "model not loaded"}))
                    continue
                sess = InferenceSession(model, device)
                sessions[ws] = sess

            # 追加观测
            try:
                sess.append_observation(head, wrist, proprio, lang)
            except Exception as e:
                await ws.send_text(json.dumps({"ok": False, "error": f"append_observation: {e}"}))
                continue

            # 是否需要返回动作
            if request_action:
                try:
                    actions = await asyncio.to_thread(sess.plan_actions)
                    await ws.send_text(json.dumps({"ok": True, "action": actions}))
                    sess.wrist_frames = []
                    sess.head_frames = []
                    sess.proprio_frames = []
                except Exception as e:
                    await ws.send_text(json.dumps({"ok": False, "error": f"plan_actions: {e}"}))
            else:
                await ws.send_text(json.dumps({"ok": True, "ack": True}))

    except WebSocketDisconnect:
        sessions.pop(ws, None)
    except Exception as e:
        await ws.send_text(json.dumps({"ok": False, "error": str(e)}))
        sessions.pop(ws, None)


# ----------------------- 构建 & 加载模型 -----------------------
def build_model_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
    memmap_dir: Optional[str] = None,
):
    """
    从 Trainer.save 存的 ckpt 构建并加载 SkillAttentionModel，
    如有需要再用 DiffTrainer.load 加载单独的 diffusion .pt。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "config" not in ckpt or "model" not in ckpt:
        raise RuntimeError("checkpoint 缺少 'config' 或 'model' 字段")

    cfg: DictConfig = ckpt["config"]
    policy_args = cfg.model
    diff_args = cfg.diffuser

    # memmap_dir：优先 CLI 传入，其次 config.task.memmap_dir
    if memmap_dir is None:
        if hasattr(cfg, "task") and hasattr(cfg.task, "memmap_dir"):
            memmap_dir = cfg.task.memmap_dir
        else:
            raise RuntimeError("请通过 --memmap_dir 指定包含 *_scale.npy / *_offset.npy 的目录")

    if diff_args.backbone == "transformer":
        model = SkillAttentionTransformerLangModel(
            policy_args=policy_args,
            diff_args=diff_args,
            device=device,
            memmap_dir=memmap_dir,
        ).to(device)
    else:
        model = SkillAttentionUnetModel(
            policy_args=policy_args,
            diff_args=diff_args,
            device=device,
            memmap_dir=memmap_dir,
        ).to(device)

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print("[build_model] missing keys:", missing)
    print("[build_model] unexpected keys:", unexpected)

    model.eval()
    return model


def init_runtime(
    ckpt_path: str,
    memmap_dir: Optional[str],
    device_str: Optional[str],
):
    global model, device
    if device_str is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"[server] using device: {device}")
    model = build_model_from_checkpoint(
        ckpt_path=ckpt_path,
        device=device,
        memmap_dir=memmap_dir,
    )
    print("[server] model loaded and ready for inference")


# ----------------------- CLI & main -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Trainer.save 生成的 .ckpt 文件路径")
    parser.add_argument("--memmap_dir", default=None,
                        help="包含 head_* / wrist_* / proprio_* / action_* .npy 的目录；"
                             "如果不提供，则尝试从 ckpt['config'].task.memmap_dir 读取")
    parser.add_argument("--device", default=None,
                        help="例如 'cuda:0' 或 'cpu'，默认自动选择")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    init_runtime(
        ckpt_path=args.checkpoint,
        memmap_dir=args.memmap_dir,
        device_str=args.device,
    )

    uvicorn.run(app, host=args.host, port=args.port,
                log_level="info", access_log=False)


if __name__ == "__main__":
    main()
