import argparse
import json
import os
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import asyncio
from PIL import Image

from transformers import DistilBertTokenizer
from difftrainer import DiffTrainer
import diffuser.utils as diffutils
from hrl_model import HRLModel 

app = FastAPI()

# 全局运行期对象（由 init_runtime 注入）
model = None
tokenizer = None
normalizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StateNormalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        # mean/std: (C) 或 (C,H,W)
        self.mean = mean.clone().float()
        self.std = std.clone().float().clamp_min(1e-6)

    def _as_broadcast(self, x: torch.Tensor):
        # 输入 x: [N,C,H,W] 或 [C,H,W]
        m, s = self.mean, self.std
        if m.ndim == 1:
            m = m.view(1, -1, 1, 1)
            s = s.view(1, -1, 1, 1)
        elif m.ndim == 3:
            m = m.view(1, *m.shape)
            s = s.view(1, *s.shape)
        return m.to(x.device, x.dtype), s.to(x.device, x.dtype)

    def channels(self) -> int:
        if self.mean.ndim == 1:
            return int(self.mean.numel())
        elif self.mean.ndim == 3:
            return int(self.mean.shape[0])
        else:
            raise ValueError("Unsupported mean shape for normalizer")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [C,H,W] 或 [1,C,H,W] 或 [N,C,H,W]
        squeezed = False
        if x.ndim == 3:
            x = x.unsqueeze(0)
            squeezed = True
        m, s = self._as_broadcast(x)
        y = (x - m) / s
        return y.squeeze(0) if squeezed else y

def load_normalizer(stats_dir: str, dev: torch.device) -> StateNormalizer:
    mean_path = os.path.join(stats_dir, "state_mean.npy")
    std_path = os.path.join(stats_dir, "state_std.npy")

    mean = torch.from_numpy(np.load(mean_path)).float().to(dev)
    std = torch.from_numpy(np.load(std_path)).float().to(dev)
    return StateNormalizer(mean, std)

@torch.no_grad()
def get_action(
    model, states, actions, options, timesteps, word_embeddings, options_list, weighted_options_list,
    cur_state, option, t, horizon, K, method, state_dim, act_dim, option_dim, device, ema_diffusion_model, **kwargs
):
    if method == 'vanilla':
        action = model.get_action(
            states.to(dtype=torch.float32),
            actions.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            word_embeddings=word_embeddings.to(dtype=torch.float32)
        )
        return action, None, states, actions, timesteps, None
    else:
        if t % (horizon - 1) == 0:
            if method == 'option':
                option, option_index, soft_idx = model.option_selector.get_option(
                    word_embeddings.mean(1, keepdim=True), states[-1].reshape(1, 1, -1), **kwargs)
            else:
                option, option_index, soft_idx = model.option_selector.get_option(
                    word_embeddings, states.to(dtype=torch.float32), timesteps.to(dtype=torch.long), **kwargs)

            options_list.append(option_index.cpu().item())
            weighted_options_list.append(soft_idx.cpu())

        options[-1] = option

        action = model.get_action(
            states.to(dtype=torch.float32),
            actions.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            options=options.to(dtype=torch.float32),
            ema_diffusion_model=ema_diffusion_model,
        )
        return action, option, states, actions, timesteps, options


class InferenceSession:
    def __init__(self, model, tokenizer, device, max_ep_len=500, horizon=8, K=8):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.max_ep_len = max_ep_len
        self.horizon = getattr(model, "horizon", horizon)
        self.method = getattr(model, "method", "vanilla")

        self.state_dim = getattr(model, "state_dim", None)
        self.act_dim = getattr(model, "act_dim", None)
        self.option_dim = getattr(getattr(model, "option_selector", None), "option_dim", None)

        self.states = None
        self.actions = None
        self.options = None
        self.timesteps = None
        self.option = None
        self.action_hist = None

        self.cls_embeddings = None
        self.word_embeddings = None

        self.options_list: List[int] = []
        self.weighted_options_list: List[torch.Tensor] = []

    def save_state(self, arr: np.ndarray, prefix: str = "recv") -> None:
        try:
            os.makedirs("debug_images", exist_ok=True)
            x = np.array(arr)

            def to_hwc_uint8(xc: np.ndarray) -> np.ndarray:
                if xc.dtype != np.uint8:
                    if xc.max() <= 1.0:
                        xc = (xc * 255.0).clip(0, 255)
                    else:
                        xc = xc.clip(0, 255)
                    xc = xc.astype(np.uint8)
                return xc
            
            # 转成 HWC 并拆分
            if x.ndim == 3 and x.shape[0] in (3, 6):  # CHW
                if x.shape[0] == 6:
                    head = np.transpose(x[:3], (1, 2, 0))
                    wrist = np.transpose(x[3:], (1, 2, 0))
                    head = to_hwc_uint8(head)
                    wrist = to_hwc_uint8(wrist)
                    Image.fromarray(head).save(f"debug_images/{prefix}_head.png")
                    Image.fromarray(wrist).save(f"debug_images/{prefix}_wrist.png")
                else:
                    img = np.transpose(x[:3], (1, 2, 0))
                    img = to_hwc_uint8(img)
                    Image.fromarray(img).save(f"debug_images/{prefix}.png")
            elif x.ndim == 3 and x.shape[-1] in (3, 6):  # HWC
                if x.shape[-1] == 6:
                    head = x[..., :3]
                    wrist = x[..., 3:6]
                    head = to_hwc_uint8(head)
                    wrist = to_hwc_uint8(wrist)
                    Image.fromarray(head).save(f"debug_images/{prefix}_head.png")
                    Image.fromarray(wrist).save(f"debug_images/{prefix}_wrist.png")
                else:
                    img = to_hwc_uint8(x[..., :3])
                    Image.fromarray(img).save(f"debug_images/{prefix}.png")
            else:
                print(f"[debug_image] No surpport shape: {x.shape}")
        except Exception as e:
            print(f"[debug_image] save failure: {e}")

    def _embed_lang(self, lang: str):
        lm_input = self.tokenizer(text=[lang], add_special_tokens=True,
                                  return_tensors='pt', padding=True).to(device=self.device)
        with torch.no_grad():
            lm_embeddings = self.model.lm(
                lm_input['input_ids'], lm_input['attention_mask']).last_hidden_state
            self.cls_embeddings = lm_embeddings[:, 0, :]
            self.word_embeddings = lm_embeddings[:, 1:, :]

    def _state_from_msg(self, state_msg: Any) -> torch.Tensor:
        arr = np.array(state_msg, dtype=np.float32)
        if arr.max() > 1.0: 
            arr = arr / 255.0

        global normalizer
        x = torch.from_numpy(arr).to(self.device, dtype=torch.float32)  # [C,H,W]
        x = normalizer(x)  # [C,H,W] -> folloe (state - mean)/std
        return x.unsqueeze(0)  # [1,C,H,W]

    def update_with_observation(self, state_msg, lang: Optional[str]):
        cur_state = self._state_from_msg(state_msg)
        self.save_state(np.array(state_msg, dtype=np.float32), prefix="recv")

        if lang is not None and self.word_embeddings is None:
            self._embed_lang(lang)

        if self.states is None:
            self.states = cur_state
            self.actions = torch.zeros((0, self.act_dim), device=self.device, dtype=torch.float32)
            self.timesteps = torch.tensor(0, device=self.device, dtype=torch.long).reshape(1, 1)
            if self.method != 'vanilla':
                self.options = torch.zeros((0, self.option_dim), device=self.device, dtype=torch.float32)
        else:
            self.states = torch.cat([self.states, cur_state], dim=0)
            self.timesteps = torch.cat(
                [self.timesteps, torch.ones((1, 1), device=self.device, dtype=torch.long) * (self.timesteps.shape[1])],
                dim=1
            )
            self.actions = torch.cat([self.actions, torch.zeros((1, self.act_dim), device=self.device)], dim=0)
            if self.method != 'vanilla':
                self.options = torch.cat([self.options, torch.zeros((1, self.option_dim), device=self.device)], dim=0)

    @torch.no_grad()
    def plan_actions(self, **kwargs):
        ema_diffusion_model = self.model.diff_trainer.ema_model
        if self.actions is not None:
            self.actions = torch.cat(
                [self.actions,
                 torch.zeros((1, self.act_dim), device=self.device, dtype=torch.float32)],
                dim=0
            )
            if self.method != 'vanilla':
                self.options = torch.cat(
                    [self.options,
                     torch.zeros((1, self.option_dim), device=self.device, dtype=torch.float32)],
                    dim=0
                )
        else:
            raise RuntimeError("actions is not initialized")

        t = self.timesteps.shape[1] - 1
        horizon = self.horizon
        K = kwargs.get("K", horizon)

        action_hist, self.option, self.states, self.actions, self.timesteps, self.options = get_action(
            self.model, self.states, self.actions, self.options, self.timesteps,
            self.word_embeddings, self.options_list, self.weighted_options_list,
            self.states[-1:], self.option, t, horizon, K, self.method,
            self.state_dim, self.act_dim, self.option_dim, self.device, ema_diffusion_model, **kwargs
        )

        if getattr(self.model, "diffuser", None) is not None and getattr(self.model.diffuser, "discrete", False):
            return [int(a.item()) for a in action_hist.view(-1)]
        else:
            return action_hist.detach().cpu().numpy().tolist()


sessions: Dict[WebSocket, InferenceSession] = {}


@app.websocket("/ws/infer")
async def ws_infer(ws: WebSocket):
    await ws.accept()
    sess: Optional[InferenceSession] = None
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            state = msg.get("state")
            lang = msg.get("lang")
            request_action = bool(msg.get("request_action", True))

            global model, tokenizer, device
            if sess is None:
                if model is None or tokenizer is None:
                    await ws.send_text(json.dumps({"ok": False, "error": "model not loaded"}))
                    continue
                sess = InferenceSession(model, tokenizer, device)
                sess.state_dim = state and tuple(np.array(state).shape)
                sess.act_dim = getattr(model, "act_dim", 7)
                if sess.method != "vanilla":
                    sess.option_dim = getattr(getattr(model, "option_selector", None), "option_dim", 8)
                sessions[ws] = sess

            sess.update_with_observation(state, lang)

            if request_action:
                actions = await asyncio.to_thread(sess.plan_actions)
                await ws.send_text(json.dumps({"ok": True, "action": actions}))
            else:
                await ws.send_text(json.dumps({"ok": True, "ack": True}))

    except WebSocketDisconnect:
        sessions.pop(ws, None)
    except Exception as e:
        await ws.send_text(json.dumps({"ok": False, "error": str(e)}))
        sessions.pop(ws, None)


# ---------------- 运行期初始化与启动 ----------------
def build_model_from_checkpoint(ckpt_path: str, pt_path: Optional[str], dev: torch.device):
    """
    复用你训练脚本里的实例化逻辑：用 checkpoint['config'] 还原 args，构建 HRLModel 并加载权重。
    """
    # 1) 读取 checkpoint（要求训练时保存了 config 和 model state_dict）
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "config" not in ckpt or "model" not in ckpt:
        raise RuntimeError("checkpoint 缺少 'config' 或 'model' 字段，请用训练脚本保存的 ckpt。")
    args = ckpt["config"]

    # 2) tokenizer
    tok = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # 3) 从 args 组装各模块参数（与你 train() 中一致）
    state_dim = args.env.state_dim
    if isinstance(state_dim, str):
        import ast
        state_dim = ast.literal_eval(state_dim)

    option_selector_args = dict(args.option_selector)
    option_selector_args['state_dim'] = state_dim
    option_selector_args['option_dim'] = args.option_dim
    option_selector_args['codebook_dim'] = args.codebook_dim
    state_reconstructor_args = dict(args.state_reconstructor)
    lang_reconstructor_args = dict(args.lang_reconstructor)
    decision_args = {
        'state_dim': state_dim,
        'action_dim': args.env.action_dim,
        'option_dim': args.option_dim,
        'discrete': args.env.discrete,
        'hidden_size': args.dt.hidden_size,
        'use_language': args.model.name == 'vanilla',
        'use_options': args.model.name != 'vanilla',
        'predict_q': args.use_iq,
        'max_length': args.model.K if 'option' in args.model.name else args.dt.max_length if hasattr(args.dt, 'max_length') else args.model.K,
        'max_ep_len': args.env.eval_episode_factor * (args.dt.max_length if hasattr(args.dt, 'max_length') else args.model.K),
        'eval': True,
    }

    # 4) 搭建 diffusion 模块与 trainer（与训练一致，但只用于推理依赖）
    if args.diffuser.diffusion == 'models.GaussianInvDynDiffusion':
        model_config = diffutils.Config(
            args.diffuser.model,
            horizon=args.diffuser.horizon,
            transition_dim=args.dt.hidden_size,
            cond_dim=args.dt.hidden_size,
            dim_mults=eval(args.diffuser.dim_mults),
            returns_condition=args.diffuser.returns_condition,
            dim=args.diffuser.dim,
            condition_dropout=args.diffuser.condition_dropout,
            calc_energy=args.diffuser.calc_energy,
            device=dev,
        )
        diffusion_config = diffutils.Config(
            args.diffuser.diffusion,
            horizon=args.diffuser.horizon,
            observation_dim=args.dt.hidden_size,
            action_dim=args.env.action_dim,
            n_timesteps=args.diffuser.n_diffusion_steps,
            loss_type=args.diffuser.loss_type,
            clip_denoised=args.diffuser.clip_denoised,
            predict_epsilon=args.diffuser.predict_epsilon,
            hidden_dim=args.diffuser.inv_hidden_dim,
            ar_inv=args.diffuser.ar_inv,
            train_only_inv=args.diffuser.train_only_inv,
            action_weight=args.diffuser.action_weight,
            loss_weights=args.diffuser.loss_weights,
            loss_discount=args.diffuser.loss_discount,
            returns_condition=args.diffuser.returns_condition,
            condition_guidance_w=args.diffuser.condition_guidance_w,
            device=dev,
        )
    elif args.diffuser.diffusion == 'models.ActionDiffusion':
        model_config = diffutils.Config(
            args.diffuser.model,
            horizon=args.diffuser.horizon,
            transition_dim=args.dt.hidden_size,
            cond_dim=args.dt.hidden_size,
            dim_mults=eval(args.diffuser.dim_mults),
            returns_condition=args.diffuser.returns_condition,
            dim=args.diffuser.dim,
            condition_dropout=args.diffuser.condition_dropout,
            calc_energy=args.diffuser.calc_energy,
            device=dev,
        )
        diffusion_config = diffutils.Config(
            args.diffuser.diffusion,
            horizon=args.diffuser.horizon,
            hidden_size=args.dt.hidden_size,
            action_dim=args.env.action_dim,
            n_timesteps=args.diffuser.n_diffusion_steps,
            loss_type=args.diffuser.loss_type,
            clip_denoised=args.diffuser.clip_denoised,
            predict_epsilon=args.diffuser.predict_epsilon,
            loss_discount=args.diffuser.loss_discount,
            returns_condition=args.diffuser.returns_condition,
            condition_guidance_w=args.diffuser.condition_guidance_w,
            device=dev,
        )
    else:
        model_config = diffutils.Config(
            args.diffuser.model,
            horizon=args.diffuser.horizon,
            transition_dim=2 * args.dt.hidden_size,
            cond_dim=args.dt.hidden_size,
            dim_mults=eval(args.diffuser.dim_mults),
            returns_condition=args.diffuser.returns_condition,
            dim=2 * args.diffuser.dim,
            condition_dropout=args.diffuser.condition_dropout,
            calc_energy=args.diffuser.calc_energy,
            device=dev,
        )
        diffusion_config = diffutils.Config(
            args.diffuser.diffusion,
            horizon=args.diffuser.horizon,
            observation_dim=args.dt.hidden_size,
            action_dim=args.env.action_dim,
            n_timesteps=args.diffuser.n_diffusion_steps,
            loss_type=args.diffuser.loss_type,
            clip_denoised=args.diffuser.clip_denoised,
            predict_epsilon=args.diffuser.predict_epsilon,
            action_weight=args.diffuser.action_weight,
            loss_weights=args.diffuser.loss_weights,
            loss_discount=args.diffuser.loss_discount,
            returns_condition=args.diffuser.returns_condition,
            condition_guidance_w=args.diffuser.condition_guidance_w,
            device=dev,
        )

    unet = model_config()
    diffusion_model = diffusion_config(unet)

    diff_trainer_args = dict(
        savepath=None,
        train_batch_size=args.diffuser.batch_size,
        train_lr=args.diffuser.learning_rate,
        gradient_accumulate_every=args.diffuser.gradient_accumulate_every,
        ema_decay=args.diffuser.ema_decay,
        sample_freq=args.diffuser.sample_freq,
        save_freq=args.diffuser.save_freq,
        log_freq=args.diffuser.log_freq,
        label_freq=int(args.diffuser.n_train_steps // max(1, args.diffuser.n_saves)),
        save_parallel=args.diffuser.save_parallel,
        bucket=args.diffuser.bucket,
        n_reference=args.diffuser.n_reference,
        train_device=dev,
        save_checkpoints=args.diffuser.save_checkpoints,
        update_ema_every=10,
        parallel=False,
    )
    diff_trainer = DiffTrainer(args, diffusion_model, diff_trainer_args)

    # 5) 构建 HRLModel 并加载权重
    hrl_model_args = dict(args.model)
    iq_args = args.iq
    m = HRLModel(
        args, option_selector_args, state_reconstructor_args,
        lang_reconstructor_args, decision_args, iq_args, diff_trainer, dev, **hrl_model_args
    ).to(dev)

    m.load_state_dict(ckpt['model'], strict=False)
    if pt_path is not None:
        m.diff_trainer.load(pt_path)
    m.eval()
    return m, tok


def init_runtime(stats_dir: str, ckpt_path: str, pt_path: Optional[str] = None, device_str: Optional[str] = None):
    global model, tokenizer, device, normalizer
    device = torch.device(device_str) if device_str else (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    model, tokenizer = build_model_from_checkpoint(ckpt_path, pt_path, device)

    normalizer = load_normalizer(stats_dir, device)
    print(f"[remote_server] Loaded normalizer from {stats_dir} with C={normalizer.channels()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats_dir", default=True, help="state_mean.npy/state_std.npy directory")
    parser.add_argument("--checkpoint", required=True, help="pth to your checkpoint file")
    parser.add_argument("--pt", default=None, help="path to your diffusion model pt file")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    init_runtime(args.stats_dir, args.checkpoint, args.pt, args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()