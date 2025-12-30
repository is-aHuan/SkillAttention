import numpy as np
import torch
from torch._C import Value
import torch.nn as nn
import torch.nn.functional as F
from src.options.option_transformer import OptionTransformer
from src.workspace.utils import pad, entropy
from src.options.vector_quantize_pytorch import VectorQuantize

# def _column_coverage_loss_from_logits(sim_logits, tau=0.7, eps=1e-8):
#     """
#     sim_logits: [B, T, C]  每个列（时间位）对 C 个码字的相似度
#     返回：标量 loss，鼓励每一列的使用分布接近均匀（最大熵）
#     """
#     # 软责任（温度 τ）
#     probs = F.softmax(sim_logits / tau, dim=-1)          # [B, T, C]
#     p_mean = probs.mean(dim=0)                           # [T, C]  每列在 batch 上的平均使用分布
#     # KL(Uniform || p_mean) = const - H(p_mean)；最小化即最大化熵
#     C = p_mean.size(-1)
#     uniform = p_mean.new_full((C,), 1.0 / C)
#     # cross-entropy(uniform, p_mean) = - sum (1/C) * log p_mean
#     loss_per_col = -(uniform * (p_mean.clamp_min(eps).log())).sum(dim=-1)  # [T]
#     return loss_per_col.mean()

# def _sim_logits(vq_module, option_preds):
#     """
#     用当前 codebook 和投影得到相似度 logits，形状 [B, T, C]
#     vq_module: VectorQuantize 实例 (self.Z 或 self.cross_attn_Z)
#     option_preds: [B, T, D]
#     """
#     x_proj = vq_module.project_in(option_preds)                 # [B, T, Cdim]
#     x_proj = F.normalize(x_proj, dim=-1)
#     cb = F.normalize(vq_module.codebook, dim=-1)               # [C, Cdim]
#     # [B, T, Cdim] x [Cdim, C] -> [B, T, C]
#     sim = torch.einsum('btc,kc->btk', x_proj, cb)
#     return sim

class OptionSelector(nn.Module):
    """
    输入:
      - word_embeddings: [B, L_text, D_lang]
      - img_embeddings:  [B, T, D_img_cat]   ← 已经把 head/wrist 两路图像 cat 好
    只用前 n_obs_steps 个 img_embeddings 作为条件。
    """

    def __init__(
            self,
            lang_dim,
            state_dim,      # 这里同样是 img_embeddings 的维度（int）
            num_options,
            option_dim,
            option_transformer=None,
            codebook_dim=128,
            kmeans_init=True,
            commitment_weight=0,
            **kwargs):

        super().__init__()

        self.option_dim = option_dim
        self.num_options = num_options

        # ========= VQ codebook =========
        self.Z = VectorQuantize(
            dim=option_dim,
            codebook_dim=codebook_dim,
            codebook_size=num_options,
            decay=0.99,
            commitment_weight=commitment_weight,
            kmeans_init=kmeans_init,
            cpc=False,
            threshold_ema_dead_code=3,
            use_cosine_sim=True,
            code_dropout_p=0.1,
            sample_top_k=4,
        )

        self.cross_attn_Z = VectorQuantize(
            dim=option_dim,
            codebook_dim=codebook_dim,
            codebook_size=num_options,
            decay=0.99,
            commitment_weight=commitment_weight,
            kmeans_init=kmeans_init,
            cpc=False,
            threshold_ema_dead_code=3,
            use_cosine_sim=True,
            code_dropout_p=0.1,
            sample_top_k=4,
        )

        option_transformer_args = {
            'state_dim': state_dim,
            'lang_dim': lang_dim,
            'option_dim': option_dim,
            'max_ep_len': option_transformer.max_ep_len,
            'n_layer': option_transformer.n_layer,
            'n_head': option_transformer.n_head,
            'n_inner': 4 * lang_dim,
            'activation_function': option_transformer.activation_function,
            'n_positions': option_transformer.n_positions,
            'resid_pdrop': option_transformer.dropout,
            'attn_pdrop': option_transformer.dropout,
            'output_attentions': True,
            'add_cross_attention': True,
        }
        self.option_dt = OptionTransformer(**option_transformer_args)

        self.alpha = nn.Parameter(torch.logit(torch.tensor(1.0)))

    def forward(self, 
                word_embeddings, 
                img_embeddings,
                timesteps, 
                attention_mask=None, 
                lang_mask=None,
                **kwargs):
        """
        img_embeddings: [B, 2 * n_obs_steps, D_img_cat]
        这里我们只用前 n_obs_steps image embeddings 作为条件
        """
        B, L, D = img_embeddings.shape

        option_hidden_states = self.Z.project_out(self.Z.codebook)  # [C, D_opt]
        dt_ret = self.option_dt(
            word_embeddings,
            img_embeddings,    # 当成 states
            timesteps,
            attention_mask,
            lang_mask,
            option_hidden_states
        )

        option_preds          = dt_ret[0]   # [B,L_lang+L_state,H]
        option_addition_preds = dt_ret[1]   # [B,L_lang+L_state,H]
        attention_weights     = dt_ret[2]   # [B,num_heads,L,L_lang+L_state]

        # ===== VQ =====
        options_vq, indices, commitment_loss = self.Z(option_preds)
        entropies = entropy(
            self.Z.codebook,
            options_vq,
            self.Z.project_in(option_preds)
        )

        options_addition, indices_addtion, commitment_loss_addition = self.cross_attn_Z(option_addition_preds)
        entropies_additon = entropy(
            self.cross_attn_Z.codebook,
            options_addition,
            self.cross_attn_Z.project_in(option_addition_preds)
        )

        alpha = 0.1 * torch.sigmoid(self.alpha)  # alpha ∈ (0, 0.1)
        options = alpha * option_addition_preds + (1 - alpha) * options_vq
        commitment_loss = alpha * commitment_loss_addition + (1 - alpha) * commitment_loss

        return options, indices, attention_weights, commitment_loss, entropies

    @torch.no_grad()
    def get_option(self,
                   word_embeddings,     # [B,Lt,D] or [Lt,D]
                   img_embeddings,      # [B,T,D] or [T,D]，这里当作 states
                   timesteps=None,      # [B,T] or [T]
                   state_mask=None,     # [B,T] or [T]，1=valid, 0=padding
                   lang_mask=None,      # [B,Lt] or [Lt]，1=valid, 0=padding
                   **kwargs):
        """
        推理接口：给出整段 n_obs 窗口观测
        返回最后一个 option 向量、对应的离散 idx、soft_idx。

        - word_embeddings: 语言 token 的 embedding    [B,Lt,D]
        - img_embeddings: 观测编码后的 state embedding [B,T,D]
        """

        # 1) constant_option 模式：直接用 codebook 某个索引
        if 'constant_option' in kwargs:
            idx = kwargs['constant_option']
            option_vec = self.Z.project_out(self.Z.codebook[idx])   # [D_opt]
            return option_vec, torch.tensor(idx, device=option_vec.device), None

        # 2) 统一 batch 维度
        if img_embeddings.dim() == 2:
            img_embeddings = img_embeddings.unsqueeze(0)    # [1,T,D]
        if word_embeddings.dim() == 2:
            word_embeddings = word_embeddings.unsqueeze(0)  # [1,Lt,D]

        B, T, _ = img_embeddings.shape
        device = img_embeddings.device

        # 3) timesteps
        if timesteps is None:
            timesteps = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # [B,T]
        elif timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(0)   # [1,T]

        # 4) state_mask / lang_mask
        if state_mask is None:
            state_mask = torch.ones(B, T, device=device, dtype=torch.long)
        elif state_mask.dim() == 1:
            state_mask = state_mask.unsqueeze(0)

        if lang_mask is None:
            L = word_embeddings.shape[1]
            lang_mask = torch.ones(B, L, device=device, dtype=torch.long)
        elif lang_mask.dim() == 1:
            lang_mask = lang_mask.unsqueeze(0)

        # 5) 调用训练时同一个 forward 接口
        #    注意这里假设 forward(word_emb, states, timesteps, state_mask, lang_mask, **kwargs)
        options, option_indx, soft_scores, _, _ = self.forward(
            word_embeddings,
            img_embeddings,
            timesteps,
            state_mask,
            lang_mask,
            **kwargs
        )
        # options:      [B,T,D_opt]
        # option_indx:  [B,T]
        # soft_scores:  [B,T,num_codes] 之类

        # 返回最后一个时间步的 option 向量和索引
        return options, option_indx, soft_scores
