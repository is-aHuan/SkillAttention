import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
)

class SkillConditionDiffusion(nn.Module):
    def __init__(self, model, horizon, action_dim,
                 n_timesteps=1000,
                 clip_denoised=False,
                 predict_epsilon=True,
                 condition_guidance_w=0.1,
                 returns_condition=True):
        super().__init__()

        self.horizon = horizon
        self.action_dim = action_dim
        self.model = model                      # ConditionalUnet1D
        self.predict_epsilon = predict_epsilon
        self.clip_denoised = clip_denoised
        self.n_timesteps = n_timesteps
        self.condition_guidance_w = condition_guidance_w
        self.returns_condition = returns_condition  # 用 classifier-free guidance 与否

        # --- diffusion 系数初始化（和你原来基本一样） ---
        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer(
            'posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer(
            'posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer(
            'posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))


    # ========= 前向/反向公式 =========
    def predict_start_from_noise(self, x_t, t, noise):
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_sample(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # ========= 调 Unet（有 local_cond/global_cond） =========
    def p_mean_variance(self, x, t, global_cond=None, local_cond=None):
        """
        x          : [B, H, Da]
        t          : [B]
        global_cond: [B, Dg]  option embedding
        local_cond : [B, H, Dl]  obs 编码
        """
        if self.returns_condition:
            eps_cond = self.model(
                x, t, local_cond=local_cond, global_cond=global_cond, use_dropout=False
            )
            eps_uncond = self.model(
                x, t, local_cond=local_cond, global_cond=global_cond, force_dropout=True
            )
            eps = eps_uncond + self.condition_guidance_w * (eps_cond - eps_uncond)
        else:
            eps = self.model(
                x, t, local_cond=local_cond, global_cond=global_cond
            )

        x_recon = self.predict_start_from_noise(x, t.detach().long(), eps)
        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, var, log_var = self.q_posterior(x_recon, x, t.long())
        return model_mean, var, log_var

    @torch.no_grad()
    def p_sample(self, x, t, global_cond=None, local_cond=None):
        model_mean, _, log_var = self.p_mean_variance(
            x, t, global_cond=global_cond, local_cond=local_cond
        )
        noise = torch.randn_like(x)
        return model_mean + (0.5 * log_var).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, global_cond=None, local_cond=None,
                      verbose=True, return_diffusion=False):
        device = self.betas.device
        B = shape[0]
        x = torch.randn(shape, device=device) * 0.5

        if return_diffusion:
            diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(self.n_timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, global_cond=global_cond, local_cond=local_cond)
            progress.update({'t': i})
            if return_diffusion:
                diffusion.append(x)
        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, global_cond=None, local_cond=None,
                           horizon=None, *_, **__):
        horizon = horizon or self.horizon
        B = global_cond.shape[0] if global_cond is not None else local_cond.shape[0]
        shape = (B, horizon, self.action_dim)
        return self.p_sample_loop(shape, global_cond=global_cond, local_cond=local_cond)

    # ========= 训练 loss（纯 diffusion） =========
    def p_losses(self, x_start, t, global_cond=None,
                 local_cond=None, valid_mask=None, condition_mask=None):
        """
        x_start       : [B, H, Da]  干净动作序列
        global_cond   : [B, Dg]
        local_cond    : [B, H, Dl]
        valid_mask    : [B, H, Da]  padding = 0, truth_data = 1
        condition_mask: [B, H, Da]  若你想把一部分动作当已知条件（一般可以全 False）
        """
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)

        if condition_mask is not None:
            x_noisy = torch.where(condition_mask, x_start, x_noisy)

        pred = self.model(
            x_noisy, t, local_cond=local_cond, global_cond=global_cond
        )

        target = noise if self.predict_epsilon else x_start

        if condition_mask is not None and not self.predict_epsilon:
            pred = torch.where(condition_mask, x_start, pred)

        if valid_mask is None:
            valid_mask = torch.ones_like(x_start, dtype=torch.bool)
        loss_mask = valid_mask if condition_mask is None else (valid_mask & (~condition_mask))

        sq_err = (pred - target) ** 2
        sq_err = sq_err * loss_mask

        denom = loss_mask.sum()
        loss = sq_err.mean() if denom.item() == 0 else sq_err.sum() / denom
        return loss, {'diff_loss': loss.detach()}

    def loss(self, actions, global_cond=None,
             local_cond=None, valid_mask=None, condition_mask=None):
        """
        actions       : [B, H, Da]  训练时的 GT 动作序列
        """
        B = actions.shape[0]
        t = torch.randint(0, self.n_timesteps, (B,), device=actions.device).long()
        return self.p_losses(actions, t,
                             global_cond=global_cond,
                             local_cond=local_cond,
                             valid_mask=valid_mask,
                             condition_mask=condition_mask)

    def forward(self, global_cond=None, local_cond=None, horizon=None):
        return self.conditional_sample(global_cond=global_cond,
                                       local_cond=local_cond,
                                       horizon=horizon)

