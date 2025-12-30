from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn

from .helpers import SinusoidalPosEmb, ModuleAttrMixin

logger = logging.getLogger(__name__)


class TransformerForDiffusionWithLanguage(ModuleAttrMixin):
    """
    Transformer diffusion backbone with optional language conditioning.

    - main tokens: action trajectory (B, T, input_dim)
    - memory tokens: [time] + (optional obs tokens) + (optional language tokens)
      decoder cross-attends to memory
    """
    def __init__(self,
            input_dim: int,
            output_dim: int,
            horizon: int,
            n_obs_steps: int = None,
            cond_dim: int = 0,

            # --- language ---
            lang_dim: int = 0,                 # >0 to enable language conditioning
            max_lang_tokens: int = 77,         # allocate pos_emb for up to this many tokens

            # --- transformer ---
            n_layer: int = 12,
            n_head: int = 12,
            n_emb: int = 768,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            causal_attn: bool = False,
            time_as_cond: bool = True,
            obs_as_cond: bool = False,
            n_cond_layers: int = 0,
        ) -> None:
        super().__init__()

        if n_obs_steps is None:
            n_obs_steps = horizon

        # ===== tokens count =====
        # main trunk length
        T = horizon
        T_cond = 1  # time token in memory

        if not time_as_cond:
            # original behavior: time token becomes part of main trunk (encoder-only BERT path)
            # NOTE: this class is designed for encoder-decoder conditioning; keep time_as_cond=True.
            raise ValueError("For language conditioning, set time_as_cond=True (encoder-decoder).")

        # obs_as_cond determined by cond_dim>0
        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            T_cond += n_obs_steps

        # language cond
        self.lang_as_cond = lang_dim > 0
        self.max_lang_tokens = int(max_lang_tokens)
        if self.lang_as_cond:
            T_cond += self.max_lang_tokens

        # ===== input stem =====
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # ===== cond encoder =====
        self.time_emb = SinusoidalPosEmb(n_emb)

        self.cond_obs_emb = None
        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        self.cond_lang_emb = None
        if self.lang_as_cond:
            self.cond_lang_emb = nn.Linear(lang_dim, n_emb)

        # memory positional emb (fixed max length = T_cond)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))

        # cond encoder module
        if n_cond_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_cond_layers)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(n_emb, 4 * n_emb),
                nn.Mish(),
                nn.Linear(4 * n_emb, n_emb),
            )

        # decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_layer)

        # ===== attention masks =====
        # tgt self-attn mask
        if causal_attn:
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)

            # optional memory_mask: keep original "time/obs alignment" if you want
            # Here we extend it to allow full access to language tokens (no restriction).
            if obs_as_cond:
                S_total = T_cond
                S_time_obs = 1 + n_obs_steps  # [time] + [obs...]
                t_idx = torch.arange(T)[:, None]                     # (T,1)
                s_idx = torch.arange(S_time_obs)[None, :]            # (1,S_time_obs)
                base = t_idx >= (s_idx - 1)                          # (T,S_time_obs) bool

                full = torch.ones(T, S_total, dtype=torch.bool)      # allow by default
                full[:, :S_time_obs] = base                          # restrict time/obs part
                mem_mask = full.float().masked_fill(full == 0, float("-inf")).masked_fill(full == 1, float(0.0))
                self.register_buffer("memory_mask", mem_mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        # constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_cond = True
        self.obs_as_cond = obs_as_cond
        self.causal_attn = causal_attn

        # init
        self.apply(self._init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = ["in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"]
            for name in weight_names:
                w = getattr(module, name, None)
                if w is not None:
                    torch.nn.init.normal_(w, mean=0.0, std=0.02)
            bias_names = ["in_proj_bias", "bias_k", "bias_v"]
            for name in bias_names:
                b = getattr(module, name, None)
                if b is not None:
                    torch.nn.init.zeros_(b)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusionWithLanguage):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError(f"Unaccounted module {module}")

    def forward(
        self,
        sample: torch.Tensor,                               # (B, T, input_dim)
        timestep: Union[torch.Tensor, float, int],          # (B,) or int
        global_cond: Optional[torch.Tensor] = None,         # (B, To, cond_dim) if obs_as_cond
        lang_ctx: Optional[torch.Tensor] = None,            # (B, L, lang_dim)  language token embeddings
        lang_mask: Optional[torch.Tensor] = None,           # (B, L)  1/True for valid, 0/False for pad
        **kwargs
    ):
        """
        output: (B, T, output_dim)
        """
        B, T, _ = sample.shape

        # ----- diffusion timestep embedding (time token) -----
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(B)
        time_tok = self.time_emb(timesteps).unsqueeze(1)     # (B,1,n_emb)

        # ----- main trunk tokens (actions) -----
        input_emb = self.input_emb(sample)                   # (B,T,n_emb)
        x = self.drop(input_emb + self.pos_emb[:, :T, :])    # (B,T,n_emb)

        # ----- build memory tokens -----
        mem_parts = [time_tok]

        if self.obs_as_cond:
            if global_cond is None:
                raise ValueError("obs_as_cond=True but global_cond is None")
            obs_tok = self.cond_obs_emb(global_cond)         # (B,To,n_emb)
            mem_parts.append(obs_tok)

        used_lang = False
        lang_len = 0
        if self.lang_as_cond and (lang_ctx is not None):
            # (B,L,lang_dim) -> (B,L,n_emb), keep up to max_lang_tokens
            lang_len = min(lang_ctx.shape[1], self.max_lang_tokens)
            lang_tok = self.cond_lang_emb(lang_ctx[:, :lang_len, :])
            mem_parts.append(lang_tok)
            used_lang = True

        memory_in = torch.cat(mem_parts, dim=1)              # (B,S,n_emb)
        S = memory_in.shape[1]
        memory_in = self.drop(memory_in + self.cond_pos_emb[:, :S, :])

        memory = self.encoder(memory_in)                     # (B,S,n_emb)

        # ----- padding mask for memory (only really needed for language padding) -----
        # PyTorch: True means "mask out / ignore"
        memory_kpm = None
        if used_lang and (lang_mask is not None):
            # build mask of shape (B,S)
            memory_kpm = torch.zeros(B, S, dtype=torch.bool, device=sample.device)

            offset = 1
            if self.obs_as_cond:
                offset += global_cond.shape[1]

            # lang_mask: True/1 valid, False/0 pad
            lm = lang_mask[:, :lang_len].to(torch.bool)
            memory_kpm[:, offset:offset + lang_len] = ~lm

        # ----- decode -----
        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=self.mask,                 # causal self-attn on actions if enabled
            memory_mask=self.memory_mask,       # optional structure mask (time/obs align). language part stays open.
            memory_key_padding_mask=memory_kpm  # masks out padded language tokens
        )

        # ----- head -----
        x = self.ln_f(x)
        x = self.head(x)
        return x
