import os
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import DistilBertModel,DistilBertTokenizer
import numpy as np

from diffuser.utils.training import EMA
from diffuser.models.conditional_unet1d import ConditionalUnet1D
from diffuser.models.diffusion_policy_scheduler import SkillConditionDiffusion
from options.option_selector import OptionSelector
from src.policy.img_encoder import SingleImageObsEncoder

class SkillAttentionUnetLorelModel(nn.Module):
    def __init__(self, 
                 policy_args, 
                 diff_args,
                 device, 
                 **kwargs):
        super().__init__()

        self.device = device
        
        # --- Policy Params ---
        self.action_dim = policy_args['action_dim']
        self.option_dim = policy_args['option_dim']
        self.proprio_dim = policy_args['proprio_dim']

        self.horizon = policy_args['horizon']
        self.n_action_steps = policy_args['n_action_steps']
        self.n_obs_steps = policy_args['n_obs_steps']
        self.obs_as_global_cond = policy_args['obs_as_global_cond']
        self.option_as_global_cond = policy_args['option_as_global_cond']

        # --- Language Model ---
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.lm = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.lm.eval() # LM 通常冻结或只推理，看需求

        # --- 1) SingleImageObsEncoder ---
        obs_cfg = dict(policy_args.obs_encoder)
        self.obs_encoder = SingleImageObsEncoder(**obs_cfg)

        # 2) 计算 obs_feat_dim
        self.obs_feat_dim = self.obs_encoder.feat_dim
        
        # --- Option Selector ---
        opt_cfg = dict(policy_args.option_selector)
        opt_cfg['state_dim'] = self.obs_encoder.image_feat_dim
        self.option_selector = OptionSelector(**opt_cfg)

        # --- 3) 构建 Diffusion Model (集成在内部) ---
        self.diffusion_model = self._build_diffusion_model(diff_args)

        # --- 4) 配置 EMA (集成在内部) ---
        self.ema_decay = diff_args.ema_decay
        self.update_ema_every = diff_args.update_ema_every
        self.step_start_ema = 5 # 可以写在 config 里
        self.step = 0

        self.ema = EMA(self.ema_decay)
        self.ema_model = copy.deepcopy(self.diffusion_model)        # 深拷贝构建 EMA 模型
        self.ema_model.requires_grad_(False)    # ✅ 关键：EMA 模型不需要梯度，防止 DDP 把它当做需要训练的参数

        # --- Stats ---
        memmap_dir = kwargs.get('memmap_dir', None)
        assert memmap_dir is not None, "Please provide memmap_dir for ExpertDatasetLorel!"
        self.memmap_dir = memmap_dir
        self._load_stats(memmap_dir)

    # ----------------------- TRAIN FORWARD ----------------------- #
    def forward(self,
                langs,            # list[str]
                obs,              # dict
                actions,          # [B, K, Da]
                timesteps,        # [B, K]
                attention_mask,   # [B, K]
                ):
        device = self.device
        
        # ---------------- 0. 数据上 GPU ---------------- #
        B, K, Da = actions.shape
        image   = obs['images'].to(device)      # [B,K,3,H,W]
        proprio = obs['proprio'].to(device)    # [B,K,Dp]

        actions = actions.to(device)
        timesteps = timesteps.to(device)
        attention_mask = attention_mask.to(device)

        # ---------------- 1. 语言编码 ---------------- #
        with torch.no_grad():
            lm_input = self.tokenizer(
                text=list(langs),
                add_special_tokens=True,
                return_tensors='pt',
                padding=True
            ).to(device)

            lm_embeddings = self.lm(
                lm_input['input_ids'],
                lm_input['attention_mask']
            ).last_hidden_state

        word_embeddings = lm_embeddings[:, 1:, :]
        lang_mask = lm_input['attention_mask'][:, 1:]

        # ---------------- 2. 图像 + proprio 编码 ---------------- #
        B, K, C, H, W = image.shape
        image_flat = image.reshape(B*K, 3, H, W)
        proprio_flat = proprio.reshape(B*K, -1)

        obs_embeddings = self.obs_encoder(image_flat, proprio_flat)
        obs_embeddings = obs_embeddings.reshape(B, K, -1)

        T_obs = self.n_obs_steps
        state_embeddings = obs_embeddings[:, :T_obs, :self.obs_encoder.image_feat_dim]
        state_mask = attention_mask[:, :T_obs]
        timesteps = timesteps[:, :T_obs]
        
        # ---------------- 3. Option Selector ---------------- #
        selected_options, _, _, commitment_loss, entropy = \
            self.option_selector(
                word_embeddings, state_embeddings, 
                timesteps, state_mask, lang_mask
            )

        # ---------------- 4. 构造 Cond ---------------- #
        if self.obs_as_global_cond:
            global_cond = obs_embeddings[:, :T_obs].reshape(B, -1)
            local_cond = None

        elif not self.obs_as_global_cond and self.option_as_global_cond:
            global_cond = torch.cat(
                [
                    selected_options[:, :T_obs],   # [B,T_obs,option_dim]
                    obs_embeddings[:, :T_obs, -self.proprio_dim:]
                ],
                dim=-1
            ).reshape(B, -1)
            local_cond = None

        loss_mask = attention_mask.bool().unsqueeze(-1).expand(-1, -1, Da)

        # ---------------- 5. 扩散训练 (集成逻辑) ---------------- #
        # ✅ 直接调用内部 diffusion model 计算 Loss
        inner_diff_loss = self.diffusion_model.compute_loss(
            actions, global_cond=global_cond, local_cond=local_cond, loss_mask=loss_mask
        )

        # ---------------- 6. 验证/监控 (使用 EMA Model) ---------------- #
        with torch.no_grad():
            # ✅ 使用内部的 EMA model 采样
            action_pred = self.ema_model.condition_sample(
                global_cond=global_cond,
                local_cond=local_cond,
                horizon=K
            )

            val_loss = F.mse_loss(actions, action_pred, reduction='none')
            val_loss = val_loss * loss_mask.to(val_loss.dtype)
            denom = loss_mask.sum().clamp_min(1)
            action_pred_loss = val_loss.sum() / denom

        return {
            'action_pred_loss': action_pred_loss,   
            'commitment_loss': commitment_loss,
            'inner_diff_loss': inner_diff_loss, # 返回这个用于反向传播
            'entropy': entropy
        }

    # ------------------------------------------------------------------ #
    #                              INFERENCE                             #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def get_action(self,
                langs, obs, timesteps, attention_mask=None,
                ema_diffusion_model=None): # 保留接口兼容性
        
        device = self.device
        
        # ✅ 默认使用内部集成的 EMA model
        if ema_diffusion_model is None:
            ema_diffusion_model = self.ema_model

        # ---------------- 0. 拆出 obs ---------------- #
        image   = obs['image'].to(device)      # [B,K,3,H,W]
        proprio = obs['proprio'].to(device)    # [B,K,Dp]

        B, K = proprio.shape[0], proprio.shape[1]
        assert B == 1, "get_action 目前只实现 batch=1"

        # In Simulation Environment, obs already normalized
        if attention_mask is None:
            attention_mask = torch.ones(B, K, device=device, dtype=torch.long)
        attention_mask = attention_mask.to(device)
        timesteps = timesteps.to(device)

        # 1) LM Encode
        with torch.no_grad():
            lm_input = self.tokenizer(
                text=list(langs), add_special_tokens=True, 
                return_tensors='pt', padding=True
            ).to(device)
            lm_embeddings = self.lm(
                lm_input['input_ids'], lm_input['attention_mask']
            ).last_hidden_state

        word_embeddings = lm_embeddings[:, 1:, :]
        lang_mask = lm_input['attention_mask'][:, 1:]

        # 2) Obs Encode
        B, K, C, H, W = image.shape
        obs_embeddings = self.obs_encoder(
            image.reshape(B*K, 3, H, W),
            proprio.reshape(B*K, -1)
        ).reshape(B, K, -1)

        T_obs = self.n_obs_steps
        state_embeddings = obs_embeddings[:, :T_obs, :self.obs_encoder.image_feat_dim]
        state_mask = attention_mask[:, :T_obs]
        timesteps = timesteps[:, :T_obs]
        
        # 3) Option
        selected_options, option_idx, soft_option_idx = self.option_selector.get_option(
            word_embeddings, state_embeddings, 
            timesteps, state_mask, lang_mask
        )

        # 4) Cond
        if self.obs_as_global_cond:
            global_cond = obs_embeddings[:, :T_obs].reshape(B, -1)
            local_cond = None
        elif not self.obs_as_global_cond and self.option_as_global_cond:
            global_cond = torch.cat([selected_options[:, :T_obs],
                                     obs_embeddings[:, :T_obs, self.obs_encoder.image_feat_dim:]], 
                                     dim=-1).reshape(B, -1)
            local_cond = None
        else:
            raise NotImplementedError("Only support obs_as_global_cond or option_as_local_cond!")

        # 5) Sample
        actions = ema_diffusion_model.condition_sample(
            global_cond=global_cond,
            local_cond=local_cond,
            horizon=self.horizon
        )

        # Decode
        actions = self._normalize_inv_action(actions.squeeze(0))
        start = T_obs - 1
        actions = actions[start:start + self.n_action_steps]

        return actions, option_idx, soft_option_idx

    def _build_diffusion_model(self, diff_args):
        """直接构建 ConditionalUnet1D 和 SkillConditionDiffusion"""
        action_dim   = self.action_dim
        D_obs        = self.obs_feat_dim
        D_opt        = self.option_dim + self.obs_encoder.proprio_feat_dim
        n_obs_steps  = self.n_obs_steps

        # 决定各个 dim
        if self.obs_as_global_cond:
            input_dim       = action_dim
            global_cond_dim = D_obs * n_obs_steps
            local_cond_dim = None
        elif not self.obs_as_global_cond and self.option_as_global_cond:
            input_dim       = action_dim
            global_cond_dim = D_opt * n_obs_steps
            local_cond_dim  = None
        else:
            raise NotImplementedError("Only support obs_as_global_cond or option_as_local_cond!")

        # Backbone: U-Net
        unet = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=local_cond_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diff_args.diffusion_step_embed_dim,
            down_dims=diff_args.down_dims,
            kernel_size=diff_args.kernel_size,
            n_groups=diff_args.n_groups,
            cond_predict_scale=diff_args.cond_predict_scale,
        )

        # Scheduler / Wrapper
        diffusion_model = SkillConditionDiffusion(
            model=unet,
            horizon=self.horizon,
            action_dim=action_dim,
            n_timesteps=diff_args.n_diffusion_steps,
            clip_denoised=diff_args.clip_denoised,
            predict_epsilon=diff_args.predict_epsilon,
        )
        return diffusion_model

    # ----------------------- EMA Helper Functions ----------------------- #
    def update_ema(self):
        """外部 Trainer 在每次 optimizer.step() 后调用此方法"""
        self.step += 1
        if self.step % self.update_ema_every == 0:
            self._step_ema()

    def _step_ema(self):
        if self.step < self.step_start_ema:
            self.ema_model.load_state_dict(self.diffusion_model.state_dict())
        else:
            self.ema.update_model_average(self.ema_model, self.diffusion_model)

    def _load_stats(self, memmap_dir):
        action_offset = np.load(os.path.join(memmap_dir, "action_offset.npy")).astype(np.float32)
        action_scale = np.load(os.path.join(memmap_dir, "action_scale.npy")).astype(np.float32)

        # Register as buffer to avoid tracking in optimization
        self.register_buffer(
            'action_offset', torch.from_numpy(action_offset).view(1, -1)
        )
        self.register_buffer(
            'action_scale', torch.from_numpy(action_scale).view(1, -1)
        )

    def _normalize_inv_action(self, actions):
        # 反归一化操作： (actions - offset) / scale
        return (actions - self.action_offset) / self.action_scale