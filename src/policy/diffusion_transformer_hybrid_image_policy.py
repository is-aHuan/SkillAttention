import os
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import DistilBertModel, DistilBertTokenizer
import numpy as np

from diffuser.utils.training import EMA
from diffuser.models.transformer_diffusion import TransformerForDiffusion
from diffuser.models.diffusion_policy_scheduler import SkillConditionDiffusion
from options.option_selector import OptionSelector
from src.policy.img_encoder import MultiObsEncoder
from dataset.expert_dataset_realman import rot6d_to_matrix, matrix_to_euler_xyz

class SkillAttentionTransformerModel(nn.Module):
    def __init__(self, 
                 policy_args, 
                 diff_args,
                 device, 
                 **kwargs):
        super().__init__()

        self.device = device
        self.pred_action_steps_only = policy_args.get('pred_action_steps_only', False)
        
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
        self.lm.eval()

        # --- 1) MultiObsEncoder ---
        obs_cfg = dict(policy_args.obs_encoder)
        self.obs_encoder = MultiObsEncoder(**obs_cfg)

        assert self.obs_encoder.head_feat_dim == self.obs_encoder.wrist_feat_dim
        # 2) compute obs_feat_dim
        self.obs_feat_dim = (
            self.obs_encoder.head_feat_dim +
            self.obs_encoder.wrist_feat_dim +
            self.obs_encoder.proprio_feat_dim
        )
        
        # --- Option Selector ---
        opt_cfg = dict(policy_args.option_selector)
        opt_cfg['state_dim'] = self.obs_encoder.head_feat_dim
        self.option_selector = OptionSelector(**opt_cfg)

        # --- 3) build Diffusion Model ---
        self.diffusion_model = self._build_diffusion_model(diff_args)

        # --- 4) setup EMA ---
        self.ema_decay = diff_args.ema_decay
        self.update_ema_every = diff_args.update_ema_every
        self.step_start_ema = 5
        self.step = 0

        self.ema = EMA(self.ema_decay)
        self.ema_model = copy.deepcopy(self.diffusion_model)
        self.ema_model.requires_grad_(False)

        # --- Stats ---
        memmap_dir = kwargs.get('memmap_dir', None)
        assert memmap_dir is not None, "Please provide memmap_dir for ExpertDatasetRealMan!"
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

        # ---------------- 0. load dataset to device ---------------- #
        B, K, Da = actions.shape
        head = obs['head_image'].to(device)
        wrist = obs['wrist_image'].to(device)
        proprio = obs['proprio'].to(device)

        actions = actions.to(device)
        timesteps = timesteps.to(device)
        attention_mask = attention_mask.to(device)

        # ---------------- 1. language encoding ---------------- #
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

        # ---------------- 2. image + proprio encoding ---------------- #
        B, K, C, H, W = head.shape
        head_flat  = head.reshape(B*K, 3, H, W)
        wrist_flat = wrist.reshape(B*K, 3, H, W)

        obs_embeddings = self.obs_encoder(head_flat, wrist_flat, proprio.reshape(B*K, -1))
        obs_embeddings = obs_embeddings.reshape(B, K, -1)

        T_obs = self.n_obs_steps
        state_embeddings = torch.cat([obs_embeddings[:, :T_obs, :self.obs_encoder.head_feat_dim], 
                                      obs_embeddings[:, :T_obs, self.obs_encoder.head_feat_dim:self.obs_encoder.head_feat_dim*2]], 
                                      dim=1)
        state_mask = torch.cat([attention_mask[:, :T_obs], attention_mask[:, :T_obs]], dim=1)
        timesteps = torch.cat([timesteps[:, :T_obs], timesteps[:, :T_obs]], dim=1)

        # ---------------- 3. Option Selector ---------------- #
        selected_options, _, _, commitment_loss, entropy = \
            self.option_selector(
                word_embeddings, state_embeddings, 
                timesteps, state_mask, lang_mask
            )

        # ---------------- 4. construct Cond ---------------- #
        if self.obs_as_global_cond:
            # (mode A) obs as global_cond (flatten time), option as local_cond (keep time dim)
            nobs_features = obs_embeddings[:, :T_obs].reshape(B, T_obs, -1)      # [B, T_obs, D_cond]
            cond = nobs_features
            if self.pred_action_steps_only:
                start = T_obs - 1
                end = start + self.n_action_steps
                actions = actions[:,start:end]

        elif not self.obs_as_global_cond and self.option_as_global_cond:
            cond = torch.cat([selected_options[:, :T_obs],
                              selected_options[:, T_obs:], 
                              obs_embeddings[:, :T_obs, -self.proprio_dim:]], dim=-1)
            if self.pred_action_steps_only:
                start = T_obs - 1
                end = start + self.n_action_steps
                actions = actions[:,start:end]

        else:
            nobs_features = obs_embeddings.reshape(B, K, -1)
            actions = torch.cat([actions, nobs_features], dim=-1).detach()

        # diffusion loss mask: set padding to 0
        loss_mask = attention_mask.bool().unsqueeze(-1).expand(-1, -1, Da)   # [B, T, Da]

        # ---------------- 5. diffusion training (integrated logic) ---------------- #
        inner_diff_loss = self.diffusion_model.compute_loss(
            actions, global_cond=cond, loss_mask=loss_mask
        )

        # ---------------- 6. validation/monitoring (using EMA Model) ---------------- #
        with torch.no_grad():
            # ✅ sample using internal EMA model
            action_pred = self.ema_model.condition_sample(
                global_cond=cond,
                local_cond=None,
                horizon=K
            )

            # element-wise MSE
            val_loss = F.mse_loss(actions, action_pred, reduction='none')
            val_loss = val_loss * loss_mask.to(val_loss.dtype)
            denom = loss_mask.sum().clamp_min(1)
            action_pred_loss = val_loss.sum() / denom

        return {
            'action_pred_loss': action_pred_loss,   # for wandb visualization
            'commitment_loss': commitment_loss,    
            'inner_diff_loss': inner_diff_loss,     # for backward
            'entropy': entropy
        }

    # ------------------------------------------------------------------ #
    #                              INFERENCE                             #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def get_action(self,
                langs, obs, timesteps, attention_mask=None):
        
        device = self.device
        ema_diffusion_model = self.ema_model

        # ---------------- 0. unpack obs ---------------- #
        head   = obs['head_image'].to(device)
        wrist  = obs['wrist_image'].to(device)
        proprio = obs['proprio'].to(device)

        B, K = proprio.shape[0], proprio.shape[1]
        assert B == 1, "get_action currently only supports batch size 1"

        # normalize obs
        head = self._normalize_head_img(head)
        wrist = self._normalize_wrist_img(wrist)
        proprio = self._normalize_proprio(proprio)

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
        B, K, C, H, W = head.shape
        obs_embeddings = self.obs_encoder(
            head.reshape(B*K, 3, H, W),
            wrist.reshape(B*K, 3, H, W),
            proprio.reshape(B*K, -1)
        ).reshape(B, K, -1)

        T_obs = self.n_obs_steps
        state_embeddings = torch.cat([obs_embeddings[:, :T_obs, :self.obs_encoder.head_feat_dim],
                                      obs_embeddings[:, :T_obs, self.obs_encoder.head_feat_dim:self.obs_encoder.head_feat_dim*2]], 
                                      dim=1)
        state_mask = torch.cat([attention_mask[:, :T_obs], attention_mask[:, :T_obs]], dim=1)
        timesteps = torch.cat([timesteps[:, :T_obs], timesteps[:, :T_obs]], dim=1)

        # 3) Option
        selected_options, _, _ = self.option_selector.get_option(
            word_embeddings, state_embeddings, 
            timesteps, state_mask, lang_mask
        )

        horizon = self.horizon
        # 4) Cond
        if self.obs_as_global_cond:
            # (mode A) obs as global_cond (flatten time), option as local_cond (keep time dim)
            nobs_features = obs_embeddings[:, :T_obs].reshape(B, T_obs, -1)      # [B, T_obs, D_cond]
            cond = nobs_features
            if self.pred_action_steps_only:
                horizon = self.n_action_steps

        elif not self.obs_as_global_cond and self.option_as_global_cond:
            cond = torch.cat([selected_options[:, :T_obs],
                              selected_options[:, T_obs:], 
                              obs_embeddings[:, :T_obs, -self.proprio_dim:]], dim=-1)
            if self.pred_action_steps_only:
                horizon = self.n_action_steps

        else:
            nobs_features = obs_embeddings.reshape(B, K, -1)
            actions = torch.cat([actions, nobs_features], dim=-1).detach()

        # 5) Sample
        actions = ema_diffusion_model.condition_sample(
            global_cond=cond,
            horizon=horizon
        )

        # Decode
        actions = self._normalize_inv_action(actions.squeeze(0))
        pos = actions[:, :3].cpu().numpy()
        rot6d = actions[:, 3:9].cpu().numpy()
        grip = actions[:, 9:10].cpu().numpy()

        R = rot6d_to_matrix(rot6d)
        euler_xyz = matrix_to_euler_xyz(R)
        actions = np.concatenate([pos, euler_xyz, grip], axis=-1)

        start = T_obs - 1
        actions = actions[start:start + self.n_action_steps]
        return actions

    def _build_diffusion_model(self, diff_args):
        """Directly build TransformerForDiffusion and SkillConditionDiffusion"""
        action_dim   = self.action_dim
        n_obs_steps  = self.n_obs_steps

        # === 1) Decide each dim ===
        obs_feature_dim = self.obs_encoder.head_feat_dim + self.obs_encoder.wrist_feat_dim + self.obs_encoder.proprio_feat_dim
        
        if self.obs_as_global_cond:
            input_dim = action_dim
            # If obs is used as cond, input_dim equals action_dim, cond_dim is observation dim
            cond_dim = obs_feature_dim

        elif not self.obs_as_global_cond and self.option_as_global_cond:
            input_dim = action_dim
            cond_dim =  2 * self.option_dim + self.obs_encoder.proprio_feat_dim

        else:
            # If not global cond, usually obs and action are concatenated as input
            input_dim = action_dim + obs_feature_dim
            cond_dim = 0
            
        output_dim = input_dim 

        # === 2) Transformer Backbone ===
        backbone = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=self.horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            n_layer=diff_args.n_layer,
            n_head=diff_args.n_head,
            n_emb=diff_args.n_emb,
            p_drop_emb=diff_args.p_drop_emb,
            p_drop_attn=diff_args.p_drop_attn,
            causal_attn=diff_args.causal_attn,
            time_as_cond=diff_args.time_as_cond,
            obs_as_cond=diff_args.obs_as_cond, 
            n_cond_layers=diff_args.n_cond_layers
        )

        # === 3) Scheduler / Wrapper ===
        diffusion_model = SkillConditionDiffusion(
            model=backbone,
            horizon=self.horizon,
            action_dim=action_dim,
            n_timesteps=diff_args.n_diffusion_steps,
            clip_denoised=diff_args.clip_denoised,
            predict_epsilon=diff_args.predict_epsilon,
        )
        return diffusion_model

    # ----------------------- EMA Helper Functions ----------------------- #
    def update_ema(self):
        """Called by external Trainer after each optimizer.step()"""
        self.step += 1
        if self.step % self.update_ema_every == 0:
            self._step_ema()

    def _step_ema(self):
        if self.step < self.step_start_ema:
            self.ema_model.load_state_dict(self.diffusion_model.state_dict())
        else:
            self.ema.update_model_average(self.ema_model, self.diffusion_model)

    # ----------------------- STATS / NORM ----------------------- #
    def _load_stats(self, memmap_dir):
        head_offset = np.load(os.path.join(memmap_dir, "head_offset.npy")).astype(np.float32)
        head_scale = np.load(os.path.join(memmap_dir, "head_scale.npy")).astype(np.float32)

        wrist_offset = np.load(os.path.join(memmap_dir, "wrist_offset.npy")).astype(np.float32)
        wrist_scale = np.load(os.path.join(memmap_dir, "wrist_scale.npy")).astype(np.float32)

        proprio_offset = np.load(os.path.join(memmap_dir, "proprio_offset.npy")).astype(np.float32)
        proprio_scale = np.load(os.path.join(memmap_dir, "proprio_scale.npy")).astype(np.float32)
        
        action_offset = np.load(os.path.join(memmap_dir, "action_offset.npy")).astype(np.float32)
        action_scale = np.load(os.path.join(memmap_dir, "action_scale.npy")).astype(np.float32)

        self.register_buffer('head_offset', torch.from_numpy(head_offset).view(1, 1, -1, 1, 1))
        self.register_buffer('head_scale', torch.from_numpy(head_scale).view(1, 1, -1, 1, 1))
        self.register_buffer('wrist_offset', torch.from_numpy(wrist_offset).view(1, 1, -1, 1, 1))
        self.register_buffer('wrist_scale', torch.from_numpy(wrist_scale).view(1, 1, -1, 1, 1))
        self.register_buffer('proprio_offset', torch.from_numpy(proprio_offset).view(1, 1, -1))
        self.register_buffer('proprio_scale', torch.from_numpy(proprio_scale).view(1, 1, -1))
        self.register_buffer('action_offset', torch.from_numpy(action_offset).view(1, -1))
        self.register_buffer('action_scale', torch.from_numpy(action_scale).view(1, -1))

    def _normalize_head_img(self, head):
        return head * self.head_scale + self.head_offset

    def _normalize_wrist_img(self, wrist):
        return wrist * self.wrist_scale + self.wrist_offset

    def _normalize_proprio(self, proprio):
        return proprio * self.proprio_scale + self.proprio_offset

    def _normalize_inv_action(self, actions):
        return (actions - self.action_offset) / self.action_scale