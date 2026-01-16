import torch
import time
import numpy as np
from tqdm import tqdm

class Trainer:

    def __init__(self, 
                 model, 
                 optimizer, 
                 train_loader, 
                 scheduler=None, 
                 device='cuda',
                 **kwargs):
        
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.scheduler = scheduler
        self.device = device

        self.start_time = time.time()
        
    def train_iteration(self, iter_num, print_logs=False):
        train_losses = []
        action_pred_losses = []
        commitment_losses = []
        inner_diffuser_losses = []
        entropies = []
        lang_entropies = []
        mutual_info = []
        logs = dict()

        train_start = time.time()

        model = self.model
        if hasattr(self.model, 'module'):
            model = self.model.module

        self.model.train()
        for batch in tqdm(self.train_loader):
            langs, obs, actions, timesteps, dones, attention_mask = batch

            B, L = actions.shape[0], actions.shape[1]
            assert L == model.horizon, f"L={L}, horizon={model.horizon} mismatch"

            actions = actions.float().to(self.device)
            timesteps = timesteps.long().to(self.device)
            dones = dones.long().to(self.device)
            attention_mask = attention_mask.long().to(self.device)

            # ==== Call model.forward (which already does diffusion training & sampling & evaluation) ====
            outputs = self.model(
                langs,
                obs,
                actions,
                timesteps,
                attention_mask=attention_mask,
            )

            action_pred_loss = outputs['action_pred_loss']       # Only for logging
            commitment_loss = outputs['commitment_loss']         # Training term
            inner_diff_loss = outputs['inner_diff_loss']         # Training term
            entropy, lang_entropy = outputs['entropy']           # If it's a tuple/list

            # Convert to scalar
            commitment_loss = commitment_loss.mean()
            inner_diff_loss = inner_diff_loss.mean()
            entropy = entropy.mean()
            lang_entropy = lang_entropy.mean()

            # ==== Total loss: only use training terms ====
            # loss = commitment_loss + 0.01 * outer_diff_loss
            # commitment_loss = 0 * commitment_loss.detach()  # No computation graph after detach()
            loss = commitment_loss + inner_diff_loss
            loss.backward()

            # Gradient accumulation & clip & step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.scheduler is not None:
                self.scheduler.step()

            # Modify here: handle DDP's .module
            raw_model = self.model.module if hasattr(self.model, "module") else self.model
            if hasattr(raw_model, "update_ema"):
                raw_model.update_ema()

            # ==== Statistics, use detach().item() ====
            train_losses.append(loss.detach().cpu().item())
            action_pred_losses.append(action_pred_loss.detach().cpu().item())
            inner_diffuser_losses.append(inner_diff_loss.detach().cpu().item())
            commitment_losses.append(commitment_loss.detach().cpu().item())
            entropies.append(entropy.detach().cpu().item())
            lang_entropies.append(lang_entropy.detach().cpu().item())
            mutual_info.append(entropies[-1] - lang_entropies[-1])


        # ==== Summarize logs ====
        logs['time/training'] = time.time() - train_start
        logs['time/total'] = time.time() - self.start_time
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)
        logs['training/action_pred_loss'] = np.mean(action_pred_losses)
        logs['training/commitment_loss'] = np.mean(commitment_losses)
        logs['training/inner_diff_loss'] = np.mean(inner_diffuser_losses)
        logs['training/entropy'] = np.mean(entropies)
        logs['training/lang_entropy'] = np.mean(lang_entropies)
        logs['training/MI'] = np.mean(entropies) - np.mean(lang_entropies)

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    # ------------------------------------------------------------------ #
    #                              SAVE / LOAD                          #
    # ------------------------------------------------------------------ #
    def save(self, iter_num, filepath, config):
        # First get the actual model (excluding DDP / DataParallel wrapping)
        if hasattr(self.model, 'module'):
            model = self.model.module
        else:
            model = self.model

        # Get the diffusion step from SkillAttentionModel
        step = model.step
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': self.optimizer.state_dict() if self.optimizer else None,
                'scheduler': self.scheduler.state_dict() if self.scheduler else None,
                'iter_num': iter_num,
                'train_dataset_max_length': getattr(self.train_loader.dataset, "max_length", None),
                'config': config,
                'step': step,   # ★ New addition
            },
            filepath
        )

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location='cpu')
        
        if hasattr(self.model, 'module'):
            raw_model = self.model.module
        else:
            raw_model = self.model

        # 2. Load parameters
        raw_model.load_state_dict(checkpoint['model'])

        if self.optimizer and checkpoint.get('optimizer') is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler and checkpoint.get('scheduler') is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        # 3. Restore diffusion step
        step = checkpoint.get('step', None)
        if step is not None:
            raw_model.step = int(step)

        return {
            'iter_num': checkpoint.get('iter_num', 0),
            'train_dataset_max_length': checkpoint.get('train_dataset_max_length', None),
            'config': checkpoint.get('config', None),
        }
