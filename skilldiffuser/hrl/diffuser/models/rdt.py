import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Mlp, RmsNorm, use_fused_attn

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)

class FinalLayer(nn.Module):
    """
    The final layer of RDT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = RmsNorm(hidden_size, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn_final = Mlp(in_features=hidden_size,
            hidden_features=hidden_size,
            out_features=out_channels, 
            act_layer=approx_gelu, drop=0)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.ffn_final(x)
        return x

class ADU(nn.Module):
    def __init__(self, model, horizon, hidden_size, action_dim, n_timesteps=200,
                 loss_type='l1', clip_denoised=False, predict_epsilon=True,
                 loss_discount=1.0, returns_condition=False,
                 condition_guidance_w=0.1):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim    # action_hidden_dim = 5
        self.hidden_size = hidden_size  # hidden_size = args.hidden_size = 128
        self.model = model # U-Net 模型
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        # --- Diffusion 过程相关的参数设置 ---
        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.final_layer = FinalLayer(hidden_size, action_dim)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # q(x_t | x_{t-1})
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        # --- 损失函数设置 ---
        # 这里的 loss_weights 需要根据 action_dim 进行调整
        loss_weights = self.get_loss_weights(loss_discount)
        self.loss_fn = Losses['state_l2'](loss_weights) # 假设 Losses 支持对 action 计算

    def get_loss_weights(self, discount):
        """为动作轨迹设置损失权重"""
        dim_weights = torch.ones(self.action_dim, dtype=torch.float32)
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, noisy_action, cond, t, returns=None):
        # cond 在这里是初始状态 (initial observation)
        if self.returns_condition:
            epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w * (epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, cond, t)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(noisy_action, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=noisy_action, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, noisy_action, cond, t, returns=None):
        b, *_, device = *noisy_action.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, noisy_action=noisy_action, cond=cond, t=t, returns=returns)
        noise = torch.randn_like(noisy_action)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *(1,) * (len(noisy_action.shape) - 1))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device
        batch_size = shape[0]
        noisy_action = torch.randn(shape, device=device) # x 是 action 序列

        if return_diffusion: diffusion = [noisy_action]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.action_adaptor(noisy_action)  # 将 action 序列投影到 hidden_size
            noisy_action = self.p_sample(x, noisy_action, cond, timesteps, returns)
            progress.update({'t': i})

            if return_diffusion: diffusion.append(noisy_action)

        progress.close()

        if return_diffusion:
            return noisy_action, torch.stack(diffusion, dim=1)
        else:
            return noisy_action

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        """
        cond: 初始状态 [ (batch_size, observation_dim) ]
        """
        device = self.betas.device
        batch_size = cond.shape[0]
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.action_dim)

        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start, cond, t, returns=None):
        # x_start 是真实的 action 序列
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # U-Net 预测噪声
        x_recon = self.model(x_noisy, cond, t, returns)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)
        
        return loss, info

    def loss(self, x, cond, returns=None):
        # x: 真实的 action 序列 (batch, horizon, action_dim)
        # cond: 初始状态 (batch, observation_dim)
        batch_size = len(x)
        a_t = x[:, :, :self.action_dim]
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(a_t, cond, t, returns)

    def forward(self, cond, *args, **kwargs):
        # cond 是初始状态
        return self.conditional_sample(cond, *args, **kwargs)
