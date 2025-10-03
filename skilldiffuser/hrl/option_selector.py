import numpy as np
import torch
from torch._C import Value
import torch.nn as nn
import torch.nn.functional as F
from option_transformer import OptionTransformer
from utils import pad, entropy
from img_encoder import Encoder

from vector_quantize_pytorch import VectorQuantize

def _column_coverage_loss_from_logits(sim_logits, tau=0.7, eps=1e-8):
    """
    sim_logits: [B, T, C]  每个列（时间位）对 C 个码字的相似度
    返回：标量 loss，鼓励每一列的使用分布接近均匀（最大熵）
    """
    # 软责任（温度 τ）
    probs = F.softmax(sim_logits / tau, dim=-1)          # [B, T, C]
    p_mean = probs.mean(dim=0)                           # [T, C]  每列在 batch 上的平均使用分布
    # KL(Uniform || p_mean) = const - H(p_mean)；最小化即最大化熵
    C = p_mean.size(-1)
    uniform = p_mean.new_full((C,), 1.0 / C)
    # cross-entropy(uniform, p_mean) = - sum (1/C) * log p_mean
    loss_per_col = -(uniform * (p_mean.clamp_min(eps).log())).sum(dim=-1)  # [T]
    return loss_per_col.mean()

def _sim_logits(vq_module, option_preds):
    """
    用当前 codebook 和投影得到相似度 logits，形状 [B, T, C]
    vq_module: VectorQuantize 实例 (self.Z 或 self.cross_attn_Z)
    option_preds: [B, T, D]
    """
    x_proj = vq_module.project_in(option_preds)                 # [B, T, Cdim]
    x_proj = F.normalize(x_proj, dim=-1)
    cb = F.normalize(vq_module.codebook, dim=-1)               # [C, Cdim]
    # [B, T, Cdim] x [Cdim, C] -> [B, T, C]
    sim = torch.einsum('btc,kc->btk', x_proj, cb)
    return sim

class OptionSelector(nn.Module):

    """
    This model takes in the language embedding and the state to output a z from a categorical distribution
    Use the VQ trick to pick an option z
    """

    def __init__(
            self, state_dim, num_options, option_dim, lang_dim, horizon, num_hidden=None, hidden_size=None,
            method='traj_option', option_transformer=None, codebook_dim=16, use_vq=True, kmeans_init=True,
            commitment_weight=0, **kwargs):

        # option_dim and codebook_dim are different because of the way the VQ package is designed
        # if they are different, then there is a projection operation that happens inside the VQ layer

        super().__init__()

        if num_hidden is not None:
            assert num_hidden >= 2, "We need at least two hidden layers!"

        self.state_dim = state_dim
        self.option_dim = option_dim
        self.use_vq = use_vq
        self.num_options = num_options

        self.horizon = horizon
        self.method = method   # whether to use full trajectory to get options or just current state
        self.hidden_size = 128
        hidden_size = 128

        if option_transformer:
            self.hidden_size = option_transformer.hidden_size

        self.Z = VectorQuantize(
            dim=option_dim,
            codebook_dim=codebook_dim,       # codebook vector size
            codebook_size=num_options,     # codebook size
            decay=0.98,             # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight=commitment_weight,   # the weight on the commitment loss
            kmeans_init=kmeans_init,   # use kmeans init
            cpc=False,
            threshold_ema_dead_code=3,  # should actively replace any codes that have an exponential moving average cluster size less than 2
            use_cosine_sim=True,   # l2 normalize the codes
            # orthogonal_reg_weight=0.1,  # orthogonal regularization weight
            code_dropout_p=0.2,
            sample_top_k=4,
        )

        self.cross_attn_Z = VectorQuantize(
            dim=option_dim,
            codebook_dim=codebook_dim,       # codebook vector size
            codebook_size=num_options,     # codebook size
            decay=0.99,             # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight=commitment_weight,   # the weight on the commitment loss
            kmeans_init=kmeans_init,   # use kmeans init
            cpc=False,
            threshold_ema_dead_code=3,  # should actively replace any codes that have an exponential moving average cluster size less than 2
            use_cosine_sim=True,   # l2 normalize the codes
            # orthogonal_reg_weight=0.1,  # orthogonal regularization weight
            code_dropout_p=0.2,
            sample_top_k=4,
        )   

        if self.method == 'traj_option':
            option_transformer_args = {'state_dim': state_dim,
                                       'lang_dim': lang_dim,
                                       'option_dim': option_dim,
                                       'hidden_size': option_transformer.hidden_size,
                                       'max_length': option_transformer.max_length,
                                       'max_ep_len': option_transformer.max_ep_len,
                                       'n_layer': option_transformer.n_layer,
                                       'n_head': option_transformer.n_head,
                                       'n_inner': 4*option_transformer.hidden_size,
                                       'activation_function': option_transformer.activation_function,
                                       'n_positions': option_transformer.n_positions,
                                       'resid_pdrop': option_transformer.dropout,
                                       'attn_pdrop': option_transformer.dropout,
                                       'output_attentions': True,  # option_transformer.output_attention,
                                       'add_cross_attention': True,
                                       }
            self.option_dt = OptionTransformer(**option_transformer_args)
            self.option_adaptor = nn.Linear(option_dim, option_dim // 2)
            self.option_addition_adaptor = nn.Linear(option_dim, option_dim // 2)

            self.alpha = nn.Parameter(torch.logit(torch.tensor(1.0)))  # learnable alpha for the linear combination of options
            
        else:
            if isinstance(state_dim, tuple):
                # LORL
                if state_dim[0] == 3:
                    # LORL Sawyer
                    self.embed_state = Encoder(hidden_size=hidden_size, ch=3, robot=False)
                else:
                    # LORL Franka
                    self.embed_state = Encoder(hidden_size=hidden_size, ch=12, robot=True)
            else:
                self.embed_state = nn.Linear(state_dim, hidden_size)

            z_layers = []
            for i in range(num_hidden):
                if i == 0:
                    z_layers.append(nn.Linear(2*hidden_size, hidden_size))
                elif i == num_hidden-1:
                    z_layers.append(nn.Linear(hidden_size, option_dim))
                else:
                    z_layers.append(nn.Linear(hidden_size, hidden_size))
            self.pred_options = nn.Sequential(*z_layers)
            self.embed_lang = nn.Linear(lang_dim, hidden_size)

    def forward(self, word_embeddings, states, timesteps=None, attention_mask=None, **kwargs):
        if self.method == 'traj_option':
            dt_ret = self.option_dt(word_embeddings, states, timesteps, attention_mask, self.Z.codebook)

            option_preds = dt_ret[0]
            option_addition_preds = dt_ret[1]
            attention_weights = dt_ret[2]  # shape: [batch_size, num_heads, query_len, key_len]
            state_embeddings = dt_ret[3]

            if self.training and self.horizon > 1:
                start = torch.randint(low=0, high=self.horizon, size=(1,), device=states.device).item()
            else:
                start = 0

            option_preds = option_preds[:, start::self.horizon, :]
            option_addition_preds = option_addition_preds[:, start::self.horizon, :]
            attention_weights = attention_weights[:, :, start::self.horizon, :].mean(dim=1) 

            if self.training:
                sim_logits_main = _sim_logits(self.Z, option_preds)             # [B, T, C]
                coverage_loss_main = _column_coverage_loss_from_logits(sim_logits_main, tau=0.7)

                # 如果也想约束 addition 分支（可选）
                sim_logits_add  = _sim_logits(self.cross_attn_Z, option_addition_preds)
                coverage_loss_add = _column_coverage_loss_from_logits(sim_logits_add, tau=0.7)

        else:
            ret_state_embeddings = self.embed_state(states).clone().detach()
            horizon_states = states[:, ::self.horizon, :]
            state_embeddings = self.embed_state(horizon_states)
            lang_embeddings = self.embed_lang(word_embeddings)  # these will be cls embeddings or word embeddings mean

            inp = torch.cat([lang_embeddings.repeat(
                1, state_embeddings.shape[1], 1), state_embeddings], dim=-1)
            option_preds = self.pred_options(inp)

            state_embeddings = ret_state_embeddings

        # attention_temperature = 0.01
        if self.use_vq:
            options_vq, indices, commitment_loss = self.Z(option_preds)
            entropies = entropy(self.Z.codebook, options_vq, self.Z.project_in(option_preds))

            options_addition, indices_addtion, commitment_loss_addition = self.cross_attn_Z(option_addition_preds)
            entropies_additon = entropy(self.cross_attn_Z.codebook, options_addition, self.cross_attn_Z.project_in(option_addition_preds))

            # alpha = 0.1 * torch.sigmoid(self.alpha)  # alpha ∈ (0, 0.1)
            # options = alpha * option_addition_preds + (1 - alpha) * options_vq

            options = torch.cat([self.option_adaptor(options_vq), self.option_addition_adaptor(options_addition)], dim=-1)
            # 更新 commitment_loss
            commitment_loss = commitment_loss + commitment_loss_addition * 0.1  # 使用不同的权重
            entropies = (entropies[0] + 0.1 * entropies_additon[0], entropies[1] + 0.1 * entropies_additon[1])

            if self.training:
                lambda_cov_main = 0.2   # 建议从 0.1~0.3 试
                lambda_cov_add  = 0.05  # addition 分支给小一点或先关掉
                commitment_loss = commitment_loss + lambda_cov_main * coverage_loss_main + lambda_cov_add * coverage_loss_add

            # indices = soft_counts
            # entropies = (marginal_entropy, cond_entropy)
        else:
            # TODO: For now simply return the first dim of option
            options, indices = option_preds, option_preds[:, :, 0]
            commitment_loss = None
            entropies = None

        return options, indices, attention_weights, commitment_loss, entropies, state_embeddings

    def get_option(self, word_embeddings, states, timesteps=None, **kwargs):

        if 'constant_option' in kwargs:
            return self.Z.project_out(
                self.Z.codebook[kwargs['constant_option']]), torch.tensor(
                kwargs['constant_option'])

        if self.method == 'traj_option':
            if isinstance(self.state_dim, tuple):
                states = states.reshape(1, -1, *self.state_dim)
            else:
                states = states.reshape(1, -1, self.state_dim)
            timesteps = timesteps.reshape(1, -1)
            max_length = self.option_dt.max_length
            # max_length = self.horizon

            if max_length is not None:
                states = states[:, -max_length:]
                timesteps = timesteps[:, -max_length:]

                # pad all tokens to sequence length
                attention_mask = pad(
                    torch.ones(1, states.shape[1]),
                    max_length).to(
                    dtype=torch.long, device=states.device).reshape(
                    1, -1)
                states = pad(states, max_length).to(dtype=torch.float32)
                timesteps = pad(timesteps, max_length).to(dtype=torch.long)
            else:
                attention_mask = None
                raise ValueError('Attention mask should not be none')

            options, option_indx, soft_idx, _, _, _ = self.forward(
                word_embeddings, states, timesteps, attention_mask=attention_mask, **kwargs)
        else:
            states = states[:, ::self.horizon, :]
            options, option_indx, soft_idx, _, _, _ = self.forward(
                word_embeddings, states, None, attention_mask=None, **kwargs)

        return options[0, -1], option_indx[0, -1], soft_idx[0, -1]
        # return options[0, -1], option_indx[-1]
