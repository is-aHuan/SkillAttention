import numpy as np
import torch
import torch.nn as nn
import transformers

from trajectory_gpt2_copy import GPT2Model as GPT2Model_copy
from trajectory_gpt2 import GPT2Model
from img_encoder import Encoder
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
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            **kwargs):
        super().__init__()

        self.option_dim = option_dim
        self.hidden_size = hidden_size

        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        self.state_dim = state_dim
        self.max_length = max_length
        self.output_attentions = kwargs["output_attentions"]

        if isinstance(state_dim, tuple):
            # LORL
            if state_dim[0] == 3:
                # LORL Sawyer
                self.embed_state = Encoder(hidden_size=hidden_size, ch=3, robot=False)
            elif state_dim[0] == 6:
                # RealMan RML63B
                self.embed_state = Encoder(hidden_size=hidden_size, ch=6, robot=True)
            else:
                # LORL Franka
                self.embed_state = Encoder(hidden_size=hidden_size, ch=12, robot=True)

        else:
            self.embed_state = nn.Linear(self.state_dim, hidden_size)

        self.embed_lang = nn.Linear(lang_dim, hidden_size)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        if kwargs["add_cross_attention"]:
            self.vq_codebook_ln = nn.LayerNorm(hidden_size)

            # note: the only difference between this GPT2Model and the default Huggingface version
            # is that the positional embeddings are removed (since we'll add those ourselves)
            self.transformer = GPT2Model_copy(config)
        else:
            self.transformer = GPT2Model(config)

        self.predict_options_base = torch.nn.Linear(hidden_size, self.option_dim)
        self.predict_options_add = torch.nn.Linear(hidden_size, self.option_dim)

        self.option_adaptor = torch.nn.Linear(16, 128)
        self.codebook_dropout_prob = 0.2

        # self.lang_pos_embed_config = [
        #         # Similarly, no initial pos embed for language
        #         ("lang", -config["dataset"]["tokenizer_max_length"]),
        #     ]
        # self.img_pos_embed_config = [
        #         # No initial pos embed in the last grid size
        #         # since we've already done in ViT
        #         ("image", (config["common"]["img_history_size"], 
        #             config["common"]["num_cameras"], 
        #             -self.embed_state.num_patches)),  
        #     ]

        # if self.lang_pos_embed_config is None:
        #     lang_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
        #         self.hidden_size, torch.arange(self.max_lang_cond_len))
        # else:
        #     lang_cond_pos_embed = get_multimodal_cond_pos_embed(
        #         embed_dim=self.hidden_size,
        #         mm_cond_lens=OrderedDict(self.lang_pos_embed_config),
        #         embed_modality=False
        #     )
        # self.lang_cond_pos_embed.data.copy_(
        #     torch.from_numpy(lang_cond_pos_embed).float().unsqueeze(0))
        
        # if self.img_pos_embed_config is None:
        #     img_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
        #         self.hidden_size, torch.arange(self.img_cond_len))
        # else:
        #     img_cond_pos_embed = get_multimodal_cond_pos_embed(
        #         embed_dim=self.hidden_size,
        #         mm_cond_lens=OrderedDict(self.img_pos_embed_config),
        #         embed_modality=False
        #     )
        # self.img_cond_pos_embed.data.copy_(
        #     torch.from_numpy(img_cond_pos_embed).float().unsqueeze(0))

    def forward(self, word_embeddings, states, timesteps, attention_mask, option_hidden_states=None, **kwargs):
        batch_size, seq_length = states.shape[0], states.shape[1]
        num_tokens = word_embeddings.shape[1]

        if attention_mask is None:
            raise ValueError('Should not have attention_mask NONE')
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        state_embeddings = self.embed_state(states)
        ret_state_embeddings = state_embeddings.clone().detach()
        lang_embeddings = self.embed_lang(word_embeddings)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings  # (batch_size, seq_length, hidden)
        lang_and_inputs = torch.cat([lang_embeddings, state_embeddings], dim=1)
        # LAYERNORM AFTER LANGUAGE
        stacked_inputs = self.embed_ln(lang_and_inputs)

        lang_mask = torch.ones((batch_size, num_tokens), device=states.device)
        lang_attn_mask = torch.cat([torch.ones((batch_size, num_tokens), device=states.device), attention_mask], dim=1)
        encoder_attention_mask = None

        # we feed in the input embeddings (not word indices as in NLP) to the model
        if option_hidden_states is not None:
            option_hidden_states = self.option_adaptor(option_hidden_states)
            option_hidden_states = self.vq_codebook_ln(option_hidden_states.view(1, *option_hidden_states.shape))

            # 使用 attention mask 进行随机丢弃（推荐）
            if self.training and self.codebook_dropout_prob > 0:
                prob_tensor = torch.full(
                    option_hidden_states.shape[:2], 
                    1 - self.codebook_dropout_prob, 
                    device=option_hidden_states.device
                )
                encoder_attention_mask = torch.bernoulli(prob_tensor)
            else:
                encoder_attention_mask = torch.ones(
                    option_hidden_states.shape[:2], 
                    device=option_hidden_states.device
                )
            
        transformer_outputs = self.transformer(
            # inputs_embeds=stacked_inputs,
            # attention_mask=lang_attn_mask,
            lang_inputs=lang_embeddings,
            state_inputs=state_embeddings,
            lang_mask=lang_mask,
            state_mask=attention_mask,
            encoder_hidden_states=option_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True
        )

        # x = transformer_outputs['last_hidden_state']
        # lang_out = x[:, :num_tokens, :].reshape(batch_size, num_tokens, self.hidden_size)
        # traj_out = x[:, num_tokens:, :].reshape(batch_size, seq_length, self.hidden_size)        

        x_base = transformer_outputs['hidden_states'][-2]   
        x_add = transformer_outputs['last_hidden_state']

        lang_out = x_base[:, :num_tokens, :].reshape(batch_size, num_tokens, self.hidden_size)
        traj_out = x_base[:, num_tokens:, :].reshape(batch_size, seq_length, self.hidden_size) 

        lang_out_add = x_add[:, :num_tokens, :].reshape(batch_size, num_tokens, self.hidden_size)
        traj_out_add = x_add[:, num_tokens:, :].reshape(batch_size, seq_length, self.hidden_size)   

        # get predictions
        # predict option logits given state
        option_preds = self.predict_options_base(traj_out)
        option_preds_add = self.predict_options_add(traj_out_add)

        if self.output_attentions:
            cross_attentions = transformer_outputs['cross_attentions'][-1][:, :, num_tokens:, :]
            return option_preds, option_preds_add, cross_attentions, ret_state_embeddings

        return option_preds, option_preds_add, None, ret_state_embeddings
