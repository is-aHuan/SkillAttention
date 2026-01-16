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

        # 1. Initialize Scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=n_timesteps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=clip_denoised,      
            prediction_type=predict_type,    
        )

    # =================================================================
    # First part: Loss
    # =================================================================
    def compute_loss(self, x_start, global_cond=None, local_cond=None, loss_mask=None, **kwargs):
        """
        Compute training loss for the diffusion model
        """
        device = x_start.device
        B = x_start.shape[0]

        # 1. Randomly sample time steps t
        t = torch.randint(0, self.n_timesteps, (B,), device=device).long()

        # 2. Generate Gaussian noise
        noise = torch.randn_like(x_start)

        # 3. Forward process (add noise)
        # scheduler.add_noise(original_samples, noise, timesteps)
        x_noisy = self.scheduler.add_noise(x_start, noise, t)

        # 4. Model prediction
        pred = self.model(x_noisy, t, global_cond=global_cond, local_cond=local_cond, **kwargs)

        # 5. Compute Loss
        # Determine target based on prediction_type
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        else: # prediction_type == "sample"
            target = x_start

        loss = F.mse_loss(pred, target, reduction='none')
        
        if loss_mask is not None:
            loss = loss * loss_mask.type(loss.dtype)

        return loss.mean()

    # =================================================================
    # Second part: Sampling
    # =================================================================
    @torch.no_grad()
    def sample(self, global_cond=None, local_cond=None, horizon=None, generator=None, **kwargs):
        """
        Perform the reverse diffusion process to generate samples
        """
        if global_cond is not None:
            device = global_cond.device
        elif local_cond is not None:
            device = local_cond.device
        else:
            # Get device from model parameters (model should be on GPU/CPU)
            device = next(self.model.parameters()).device  # Crucial fix!
        
        # Determine Batch Size
        if global_cond is not None:
            B = global_cond.shape[0]
        elif local_cond is not None:
            B = local_cond.shape[0]
        else:
            B = 1
            
        horizon = horizon or self.horizon
        shape = (B, horizon, self.action_dim)

        # 1. Initialize Gaussian noise x_T
        latents = torch.randn(shape, device=device, generator=generator)

        # 2. Set inference timesteps (usually fewer than training steps, but kept consistent here)
        self.scheduler.set_timesteps(self.n_timesteps)

        # 3. Reverse denoising loop
        for t in self.scheduler.timesteps:
            # 3.1 Model prediction
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            
            # Predict noise (epsilon) or sample based on scheduler config
            model_output = self.model(latents, t_tensor, global_cond=global_cond, local_cond=local_cond, **kwargs)

            # 3.2 Scheduler Step 
            latents = self.scheduler.step(model_output, t, latents).prev_sample

        return latents

    # forward loss
    def condition_sample(self, global_cond=None, local_cond=None, horizon=None, **kwargs):
        return self.sample(global_cond=global_cond, local_cond=local_cond, horizon=horizon, **kwargs)
    
    # sample interface
    def forward(self, actions, global_cond=None, local_cond=None, loss_mask=None):
        return self.compute_loss(actions, global_cond=global_cond, local_cond=local_cond, loss_mask=loss_mask)
