import numpy as np
import torch
from torch._C import Value
import torch.nn as nn
import torch.nn.functional as F
from src.options.option_transformer import OptionTransformer
from src.workspace.utils import pad, entropy
from src.options.vector_quantize_pytorch import VectorQuantize

class OptionSelector(nn.Module):
    """
    Inputs:
      - word_embeddings: [B, L_text, D_lang]
      - img_embeddings:  [B, T, D_img_cat]   <- head/wrist image streams already concatenated
    Only the first n_obs_steps of img_embeddings are used as conditioning.
    """

    def __init__(
            self,
            lang_dim,
            state_dim,      # img_embeddings dim
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
        """
        B, L, D = img_embeddings.shape

        option_hidden_states = self.Z.project_out(self.Z.codebook)  # [C, D_opt]
        dt_ret = self.option_dt(
            word_embeddings,
            img_embeddings,    # states
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
                   img_embeddings,      # [B,T,D] or [T,D], treated here as states
                   timesteps=None,      # [B,T] or [T]
                   state_mask=None,     # [B,T] or [T], 1=valid, 0=padding
                   lang_mask=None,      # [B,Lt] or [Lt], 1=valid, 0=padding
                   **kwargs):
        """
        Inference API: given an entire n_obs window of observations.
        Returns the last option vector, the corresponding discrete index, and soft_idx.

        - word_embeddings: language token embeddings    [B,Lt,D]
        - img_embeddings:  state embeddings after observation encoding [B,T,D]
        """

        # 1) constant_option mode: directly use a specific index from the codebook
        if 'constant_option' in kwargs:
            idx = kwargs['constant_option']
            option_vec = self.Z.project_out(self.Z.codebook[idx])   # [D_opt]
            return option_vec, torch.tensor(idx, device=option_vec.device), None

        # 2) unify batch dimension
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

        # 5) Call the same forward interface used during training
        #    Note we assume forward(word_emb, states, timesteps, state_mask, lang_mask, **kwargs)
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
        # soft_scores:  [B,T,num_codes] etc.

        return options, option_indx, soft_scores
