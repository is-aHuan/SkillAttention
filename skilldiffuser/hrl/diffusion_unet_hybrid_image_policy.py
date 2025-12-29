import wandb
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import DistilBertModel

from iq import IQMixin
from option_selector import OptionSelector
from img_encoder import Encoder
from dec_encoder import DecEncoder
# from utils import pad
# import diffuser.utils as diffutils


class HRLModel(nn.Module, IQMixin):
    def __init__(self, 
                 args, 
                 option_selector_args,
                 decision_args, 
                 diff_trainer,   
                 device, 
                 horizon=5, 
                 K=10, 
                 method='vanilla',
                 proprio_embed_dim=None,
                 **kwargs):
        super().__init__()

        self.args = args
        self.horizon = horizon
        self.K = K
        self.device = device
        self.method = method

        self.lm = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.lm.eval()

        self.state_dim = decision_args['state_dim']       # 这里其实是图像的 dim 或 shape
        self.action_dim = decision_args['action_dim']
        self.option_dim = decision_args['option_dim']
        # self.proprio_dim = decision_args.get('proprio_dim', None)
        self.proprio_dim = decision_args['option_dim']

        assert decision_args['use_options'] and not decision_args['use_language']
        self.option_selector = OptionSelector(
            lang_dim=self.lm.config.dim,
            method=self.method,
            **option_selector_args
        )

        # TODO proprio encoder, use MLP
        if self.proprio_dim is not None and proprio_embed_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, proprio_embed_dim)
            self.proprio_embed_dim = proprio_embed_dim

        # TODO vision encoder（图像编码器）
        self.img_encoder = Encoder(
            hidden_size=256,
        )

        # ===== 6. DiffTrainer（里面封装了 ConditionalUnet1D + EMA + optimizer）=====
        self.diff_trainer = diff_trainer
        self.layer_norm = nn.LayerNorm(self.lm.config.dim)

    # ------------------------------------------------------------------ #
    #                               TRAIN                                #
    # ------------------------------------------------------------------ #
    def forward(self, lm_input_ids, lm_attention_mask,
                obs_rgb1, obs_rgb2, obs_proprio,
                actions, timesteps, iter_num, ind, diff_train_num,
                attention_mask=None):
        """
        actions:     [B, T, action_dim]
        obs_proprio: [B, T, proprio_dim]
        obs_rgb1/2:  [B, T, C, H, W]
        timesteps:   [B, T]       # 环境时间步
        """

        device = self.device
        B, T, Da = actions.shape
        actions = actions.to(device)

        if attention_mask is None:
            attention_mask = torch.ones(B, T, device=device, dtype=torch.long)

        # ===== 1. 语言编码 =====
        with torch.no_grad():
            lm_embeddings = self.lm(
                lm_input_ids.to(device),
                lm_attention_mask.to(device)
            ).last_hidden_state
        word_embeddings = lm_embeddings[:, 1:, :]   # [B, L_text-1, D_lm]

        # ===== 2. 图像编码 → states =====
        # 假设 Encoder 支持两路图像输入，并输出 [B, T, state_dim]
        states = self.img_encoder(obs_rgb1.to(device), obs_rgb2.to(device))
        # states: [B, T, state_dim]

        # ===== 3. Option Selector：输出 options + state_embeddings =====
        if self.method == 'option':
            selected_options, _, _, commitment_loss, entropy, state_embeddings = self.option_selector(
                word_embeddings.mean(1, keepdim=True), states
            )
        else:
            selected_options, _, _, commitment_loss, entropy, state_embeddings = self.option_selector(
                word_embeddings, states, timesteps.to(device), attention_mask
            )

        # selected_options: [B, T//horizon, option_dim]
        # state_embeddings: [B, T, D_state_embed]   —— 图像 embedding 序列

        # ===== 4. local_cond = 图像 embedding + proprio =====
        obs_proprio = obs_proprio.to(device)
        if self.proprio_dim is not None:
            proprio_feat = self.proprio_encoder(obs_proprio)      # [B, T, D_prop_embed]
            local_cond = torch.cat([state_embeddings, proprio_feat], dim=-1)
        else:
            local_cond = state_embeddings                         # [B, T, D_state_embed]
        # local_cond: [B, T, D_local]

        # ===== 5. 按 horizon/K 切成子序列 =====
        assert T % self.K == 0, "T must be multiple of K"
        num_seq = T // self.K

        def chunk_3d(x):
            B_, T_, D_ = x.shape
            return x.reshape(B_ * num_seq, self.K, D_)

        def chunk_2d(x):
            B_, T_ = x.shape
            return x.reshape(B_ * num_seq, self.K)

        actions_seq    = chunk_3d(actions)         # [B_seq, K, Da]
        local_cond_seq = chunk_3d(local_cond)      # [B_seq, K, D_local]
        attn_seq       = chunk_2d(attention_mask)  # [B_seq, K]

        B_seq = actions_seq.shape[0]

        # ===== 6. global_cond：每个子序列一个 option =====
        # selected_options: [B, num_seq, option_dim]
        assert selected_options.shape[1] == num_seq, \
            f"selected_options.shape[1]={selected_options.shape[1]} != num_seq={num_seq}"
        global_cond_seq = selected_options.reshape(B_seq, self.option_dim)  # [B_seq, option_dim]

        # ===== 7. loss_mask：哪些时间步有效（padding 用 0）=====
        valid_mask = attn_seq.bool()             # [B_seq, K]
        loss_mask = valid_mask.unsqueeze(-1)     # [B_seq, K, 1]

        # ===== 8. 交给 DiffTrainer 训练扩散模型 =====
        #   batch 对应 diffusion_model.loss(actions, local_cond, global_cond, loss_mask)
        self.diff_log_interval = 100
        diff_loss_list = []

        for in_step in range(diff_train_num):
            step = iter_num * diff_train_num + ind * diff_train_num + in_step

            diff_loss = self.diff_trainer.train_iteration(
                (actions_seq, local_cond_seq, global_cond_seq, loss_mask)
            )
            diff_loss_list.append(diff_loss.detach().cpu().item())

            if step % self.diff_log_interval == 0:
                avg_diff_loss = sum(diff_loss_list) / len(diff_loss_list)
                diff_loss_list = []
                wandb.log({'inner/diff_loss': avg_diff_loss}, step=step)

        # 最后一轮的 diff_loss 用来返回
        return {
            'actions_pred': actions,          # 真正的预测可在 diffusion_model 里额外返回
            'actions': actions,
            'attention_mask': attention_mask,
            'commitment_loss': commitment_loss,
            'diff_loss': diff_loss,
            'entropy': entropy
        }


    # ------------------------------------------------------------------ #
    #                              INFERENCE                             #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def get_action(self, lm_input_ids, lm_attention_mask,
                   obs_rgb1, obs_rgb2, obs_proprio,
                   timesteps, attention_mask=None,
                   ema_diffusion_model=None):
        """
        在线预测一个 horizon 内的动作（比如长度 K）。
        这里先假定 batch=1，obs.shape = [1, T, ...]。
        """

        if ema_diffusion_model is None:
            # 用 DiffTrainer 里的 EMA 模型
            ema_diffusion_model = self.diff_trainer.ema_model

        device = self.device
        B, T = obs_proprio.shape[0], obs_proprio.shape[1]
        assert B == 1, "get_action 先实现 batch=1 的情况"

        if attention_mask is None:
            attention_mask = torch.ones(B, T, device=device, dtype=torch.long)

        # ===== 1. 语言编码 =====
        with torch.no_grad():
            lm_embeddings = self.lm(
                lm_input_ids.to(device),
                lm_attention_mask.to(device)
            ).last_hidden_state
        word_embeddings = lm_embeddings[:, 1:, :]

        # ===== 2. 图像编码 =====
        states = self.img_encoder(obs_rgb1.to(device), obs_rgb2.to(device))  # [1, T, state_dim]

        # ===== 3. option 选择 =====
        if self.method == 'option':
            selected_options, _, _, _, _, state_embeddings = self.option_selector(
                word_embeddings.mean(1, keepdim=True), states
            )
        else:
            selected_options, _, _, _, _, state_embeddings = self.option_selector(
                word_embeddings, states, timesteps.to(device), attention_mask
            )

        # 取最后一个 option 作为当前 horizon 的 global_cond
        global_cond = selected_options[:, -1, :]       # [1, option_dim]

        # ===== 4. local_cond = 图像 embedding + proprio =====
        obs_proprio = obs_proprio.to(device)
        if self.proprio_dim is not None:
            proprio_feat = self.proprio_encoder(obs_proprio)  # [1, T, D_prop_embed]
            local_cond = torch.cat([state_embeddings, proprio_feat], dim=-1)
        else:
            local_cond = state_embeddings                   # [1, T, D_state_embed]

        # ===== 5. 调用你的 diffusion 模型做采样 =====
        # 这里假设 ema_diffusion_model 有 sample(...) 接口：
        #   actions = ema_diffusion_model.sample(shape, local_cond, global_cond, mask=...)
        shape = (B, T, self.action_dim)
        mask = attention_mask.unsqueeze(-1).bool()

        actions = ema_diffusion_model.conditional_sample(
            shape=shape,
            local_cond=local_cond,
            global_cond=global_cond,
            mask=mask
        )
        # actions: [1, T, action_dim]

        # 只返回最后一个时间步的动作
        return actions[:, -1, :]


    def save(self, iter_num, filepath, config):
        if hasattr(self.model, 'module'):
            model = self.model.module

        torch.save({'model': model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'iter_num': iter_num,
                    'train_dataset_max_length': self.train_loader.dataset.max_length,
                    'config': config}, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        return {'iter_num': checkpoint['iter_num'], 'train_dataset_max_length': checkpoint['train_dataset_max_length'], 'config': checkpoint['config']}

    def count_parameters(self):
        total_num = sum(p.numel() for p in self.parameters())
        print(f'Total Parameters: {total_num / 1e9:.2f}B')