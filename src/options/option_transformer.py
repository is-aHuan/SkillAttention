import numpy as np
import torch
import torch.nn as nn
import transformers

from src.options.stage_fusion_transformer import StageFusionTransformer, SFTConfig

from src.options.trajectory_gpt2 import GPT2Model
from collections import OrderedDict

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    if not isinstance(pos, np.ndarray):
        pos = np.array(pos, dtype=np.float64)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_nd_sincos_pos_embed_from_grid(embed_dim, grid_sizes):
    """
    embed_dim: output dimension for each position
    grid_sizes: the grids sizes in each dimension (K,).
    out: (grid_sizes[0], ..., grid_sizes[K-1], D)
    """
    num_sizes = len(grid_sizes)
    # For grid size of 1, we do not need to add any positional embedding
    num_valid_sizes = len([x for x in grid_sizes if x > 1])
    emb = np.zeros(grid_sizes + (embed_dim,))
    # Uniformly divide the embedding dimension for each grid size
    dim_for_each_grid = embed_dim // num_valid_sizes
    # To make it even
    if dim_for_each_grid % 2 != 0:
        dim_for_each_grid -= 1
    valid_size_idx = 0
    for size_idx in range(num_sizes):
        grid_size = grid_sizes[size_idx]
        if grid_size <= 1:
            continue
        pos = np.arange(grid_size)
        posemb_shape = [1] * len(grid_sizes) + [dim_for_each_grid]
        posemb_shape[size_idx] = -1
        emb[..., valid_size_idx * dim_for_each_grid:(valid_size_idx + 1) * dim_for_each_grid] += \
            get_1d_sincos_pos_embed_from_grid(dim_for_each_grid, pos).reshape(posemb_shape)
        valid_size_idx += 1
    return emb


def get_multimodal_cond_pos_embed(embed_dim, mm_cond_lens: OrderedDict, 
                                  embed_modality=True):
    """
    Generate position embeddings for multimodal conditions. 
    
    mm_cond_lens: an OrderedDict containing 
        (modality name, modality token length) pairs.
        For `"image"` modality, the value can be a multi-dimensional tuple.
        If the length < 0, it means there is no position embedding for the modality or grid.
    embed_modality: whether to embed the modality information. Default is True.
    """
    num_modalities = len(mm_cond_lens)
    modality_pos_embed = np.zeros((num_modalities, embed_dim))
    if embed_modality:
        # Get embeddings for various modalites
        # We put it in the first half
        modality_sincos_embed = get_1d_sincos_pos_embed_from_grid(
            embed_dim // 2, torch.arange(num_modalities))
        modality_pos_embed[:, :embed_dim // 2] = modality_sincos_embed
        # The second half is for position embeddings
        pos_embed_dim = embed_dim // 2
    else:
        # The whole embedding is for position embeddings
        pos_embed_dim = embed_dim
    
    # Get embeddings for positions inside each modality
    c_pos_emb = np.zeros((0, embed_dim))
    for idx, (modality, cond_len) in enumerate(mm_cond_lens.items()):
        if modality == "image" and \
            (isinstance(cond_len, tuple) or isinstance(cond_len, list)):
            all_grid_sizes = tuple([abs(x) for x in cond_len])
            embed_grid_sizes = tuple([x if x > 0 else 1 for x in cond_len])
            cond_sincos_embed = get_nd_sincos_pos_embed_from_grid(
                pos_embed_dim, embed_grid_sizes)
            cond_pos_embed = np.zeros(all_grid_sizes + (embed_dim,))
            cond_pos_embed[..., -pos_embed_dim:] += cond_sincos_embed
            cond_pos_embed = cond_pos_embed.reshape((-1, embed_dim))
        else:
            cond_sincos_embed = get_1d_sincos_pos_embed_from_grid(
                pos_embed_dim, torch.arange(cond_len if cond_len > 0 else 1))
            cond_pos_embed = np.zeros((abs(cond_len), embed_dim))
            cond_pos_embed[:, -pos_embed_dim:] += cond_sincos_embed
        cond_pos_embed += modality_pos_embed[idx]
        c_pos_emb = np.concatenate([c_pos_emb, cond_pos_embed], axis=0)
    
    return c_pos_emb


class OptionTransformer(nn.Module):
    """
    This model uses GPT-2 to select options for every horizon-th state
    """

    def __init__(
            self,
            state_dim,
            lang_dim,
            option_dim,
            max_ep_len=4096,
            **kwargs):
        super().__init__()

        self.option_dim = option_dim
        self.hidden_size = lang_dim     # 通常我们设 hidden_size=lang_dim

        # 从 kwargs 里拿 n_layer / n_head / n_positions 等
        cfg_gpt = transformers.GPT2Config(
            vocab_size=1,              # 不用 vocab
            n_embd=lang_dim,        # = 768
            n_layer=kwargs.get("n_layer", 4),
            n_head=kwargs.get("n_head", 12),   # <<< 这里设 12 或 24
            n_positions=kwargs.get("n_positions", 1024),
            n_inner=kwargs.get("n_inner", 4 * lang_dim),
            resid_pdrop=kwargs.get("resid_pdrop", 0.1),
            attn_pdrop=kwargs.get("attn_pdrop", 0.1),
        )


        cfg_sft = SFTConfig(
            n_embd=lang_dim,
            n_head=kwargs.get("n_head", 24),
            n_positions=kwargs.get("n_positions", 512),
            attn_pdrop=kwargs.get("attn_pdrop", 0.1),
            resid_pdrop=kwargs.get("resid_pdrop", 0.1),
            add_cross_attention=kwargs.get("add_cross_attention", True),
            activation_function=kwargs.get("activation_function", "gelu"),
        )

        self.state_dim = state_dim
        self.output_attentions = kwargs["output_attentions"]

        # 图像 embedding -> hidden_size（例如 768）
        self.embed_state = nn.Linear(self.state_dim, self.hidden_size)

        # 语言 768 -> hidden_size（通常我们设 hidden_size=768，就等于 identity + LN）
        self.embed_lang = nn.Linear(lang_dim, self.hidden_size)

        # 时间步嵌入
        self.embed_timestep = nn.Embedding(max_ep_len, self.hidden_size)

        self.embed_ln = nn.LayerNorm(lang_dim)

        if kwargs["add_cross_attention"]:
            self.vq_codebook_ln = nn.LayerNorm(self.hidden_size)
            self.transformer = StageFusionTransformer(cfg_sft)
        else:
            self.transformer = GPT2Model(cfg_gpt)

        # 选项预测头
        self.predict_options_base = nn.Linear(self.hidden_size, self.option_dim)
        self.predict_options_add  = nn.Linear(self.hidden_size, self.option_dim)

        # codebook feature -> hidden_size（别再写死 128）
        self.option_adaptor = nn.Linear(option_dim, self.hidden_size)
        self.codebook_dropout_prob = 0.1

    def forward(self, 
                word_embeddings, 
                obs_embeddings, 
                timesteps, 
                attention_mask,
                lang_mask,
                option_hidden_states=None, 
                **kwargs):
        """
        word_embeddings: [B, L_text, lang_dim]  (比如 768)
        states:          [B, T, state_dim]
        timesteps:       [B, T]
        attention_mask:  [B, T]
        """
        B, T = timesteps.shape[0], timesteps.shape[1]
        num_tokens = word_embeddings.shape[1]

        # (1) 图像 → hidden_size
        state_embeddings = self.embed_state(obs_embeddings)      # [B,T,H]

        # (2) 文本 → hidden_size
        lang_embeddings = self.embed_lang(word_embeddings)  # [B,L,H]

        # (3) 时间嵌入 + 到 state 上
        time_embeddings = self.embed_timestep(timesteps)    # [B,T,H]
        state_embeddings = state_embeddings + time_embeddings

        # (4) 拼在一起（lang 在前）
        lang_mask = lang_mask.to(obs_embeddings.device)
        state_mask = attention_mask  # [B,T]
        encoder_attention_mask = None

        if option_hidden_states is not None:
            # option_hidden_states: [num_codes, codebook_dim]
            option_hidden_states = self.option_adaptor(option_hidden_states)   # [num_codes, H]

            B = obs_embeddings.shape[0]  # 或者直接用 word_embeddings.shape[0]

            # [num_codes, H] -> [1, num_codes, H] -> [B, num_codes, H]
            option_hidden_states = option_hidden_states.unsqueeze(0).expand(B, -1, -1)
            option_hidden_states = self.vq_codebook_ln(option_hidden_states)   # [B, num_codes, H]

            if self.training and self.codebook_dropout_prob > 0:
                # 现在 mask 应该也是 [B, num_codes]
                prob_tensor = torch.full(
                    option_hidden_states.shape[:2],   # [B, num_codes]
                    1 - self.codebook_dropout_prob,
                    device=option_hidden_states.device
                )
                encoder_attention_mask = torch.bernoulli(prob_tensor)
            else:
                encoder_attention_mask = torch.ones(
                    option_hidden_states.shape[:2],   # [B, num_codes]
                    device=option_hidden_states.device
                )


        # (5) multimodal infomation fusion by transformer
        transformer_outputs = self.transformer(
            lang_inputs=lang_embeddings,
            state_inputs=state_embeddings,
            lang_mask=lang_mask,
            state_mask=state_mask,
            encoder_hidden_states=option_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True
        )

        x_base = transformer_outputs['hidden_states'][-2]
        x_add  = transformer_outputs['last_hidden_state']

        # TODO [B, T_lang + T_state, H] or [B, T_state, H] ?
        traj_out     = x_base[:, num_tokens:, :]  # [B,T,H]
        traj_out_add = x_add[:,  num_tokens:, :]  # [B,T,H]

        option_preds     = self.predict_options_base(traj_out)
        option_preds_add = self.predict_options_add(traj_out_add)

        if self.output_attentions:
            cross_attentions = transformer_outputs['cross_attentions'][-1]
            cross_attentions = cross_attentions[:, :, num_tokens:, :].mean(dim=1)  # [B, T_state, T_option]
            return option_preds, option_preds_add, cross_attentions

        return option_preds, option_preds_add, None

