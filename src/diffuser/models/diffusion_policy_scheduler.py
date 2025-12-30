import torch
from torch import nn
import torch.nn.functional as F
from diffusers import DDPMScheduler

class SkillConditionDiffusion(nn.Module):
    def __init__(self, 
                 model, 
                 horizon, 
                 action_dim,
                 n_timesteps=1000,
                 clip_denoised=True,
                 predict_epsilon=True,
    ):
        super().__init__()

        self.horizon = horizon
        self.action_dim = action_dim
        self.model = model  # ConditionalUnet1D
        self.predict_epsilon = predict_epsilon
        self.n_timesteps = n_timesteps

        if self.predict_epsilon:
            predict_type = "epsilon"
        else:
            predict_type = "sample"

        # 1. 初始化 Scheduler
        # DDPMScheduler 会自动处理 beta, alpha, posterior 等计算
        self.scheduler = DDPMScheduler(
            num_train_timesteps=n_timesteps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=clip_denoised,      # 对应原代码的 clip_denoised
            prediction_type=predict_type,    # 对应原代码的 predict_epsilon 判断
        )

    # =================================================================
    # 第一部分：训练 Loss 计算
    # =================================================================
    def compute_loss(self, x_start, global_cond=None, local_cond=None, loss_mask=None, **kwargs):
        """
        计算扩散模型的训练 Loss
        """
        device = x_start.device
        B = x_start.shape[0]

        # 1. 随机采样时间步 t
        t = torch.randint(0, self.n_timesteps, (B,), device=device).long()

        # 2. 生成高斯噪声
        noise = torch.randn_like(x_start)

        # 3. 前向加噪 (Forward Process)
        # scheduler.add_noise(original_samples, noise, timesteps)
        x_noisy = self.scheduler.add_noise(x_start, noise, t)

        # 4. 模型预测
        # 注意：model 需要接受 t 作为输入
        pred = self.model(x_noisy, t, global_cond=global_cond, local_cond=local_cond, **kwargs)

        # 5. 计算 Loss
        # 根据 prediction_type 决定 target 是噪声还是原始数据
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        else: # prediction_type == "sample"
            target = x_start

        loss = F.mse_loss(pred, target, reduction='none')
        
        if loss_mask is not None:
            loss = loss * loss_mask.type(loss.dtype)

        return loss.mean()

    # =================================================================
    # 第二部分：推理 Sampling
    # =================================================================
    @torch.no_grad()
    def sample(self, global_cond=None, local_cond=None, horizon=None, generator=None, **kwargs):
        """
        执行反向扩散过程生成样本
        """
        if global_cond is not None:
            device = global_cond.device
        elif local_cond is not None:
            device = local_cond.device
        else:
            # 从模型参数获取设备（模型肯定在 GPU/CPU 上）
            device = next(self.model.parameters()).device  # 关键修复！
        
        # 确定 Batch Size
        if global_cond is not None:
            B = global_cond.shape[0]
        elif local_cond is not None:
            B = local_cond.shape[0]
        else:
            B = 1
            
        horizon = horizon or self.horizon
        shape = (B, horizon, self.action_dim)

        # 1. 初始化高斯噪声 x_T
        latents = torch.randn(shape, device=device, generator=generator)

        # 2. 设置推理时间步 (通常推理时步数可以少于训练步数，这里保持一致)
        self.scheduler.set_timesteps(self.n_timesteps)

        # 3. 反向去噪循环
        for t in self.scheduler.timesteps:
            # 3.1 模型预测
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            
            # 预测噪声 (epsilon) 或 样本 (sample)
            model_output = self.model(latents, t_tensor, global_cond=global_cond, local_cond=local_cond, **kwargs)

            # 3.2 Scheduler Step (核心简化)
            latents = self.scheduler.step(model_output, t, latents).prev_sample

        return latents

    # forward loss
    def condition_sample(self, global_cond=None, local_cond=None, horizon=None, **kwargs):
        return self.sample(global_cond=global_cond, local_cond=local_cond, horizon=horizon, **kwargs)
    
    # sample interface
    def forward(self, actions, global_cond=None, local_cond=None, loss_mask=None):
        return self.compute_loss(actions, global_cond=global_cond, local_cond=local_cond, loss_mask=loss_mask)
