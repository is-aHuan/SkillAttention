import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Basic Components: Attention / MLP / Block
# =========================

class Attention(nn.Module):
    """
    Simplified Multi-Head Attention:
      - Supports self-attn and cross-attn (controlled by is_cross_attention)
      - Only does padding mask, no causal mask
      - Input / output shape: hidden_states: [B, L, H]
    """

    def __init__(
        self,
        dim: int,
        n_head: int,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        is_cross_attention: bool = False,
    ):
        super().__init__()
        assert dim % n_head == 0, "dim must be divisible by n_head"
        self.dim = dim
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.is_cross_attention = is_cross_attention

        if is_cross_attention:
            # Q comes from the current layer hidden_states, K/V come from encoder_hidden_states
            self.q_proj = nn.Linear(dim, dim)
            self.kv_proj = nn.Linear(dim, 2 * dim)
        else:
            # Q/K/V all come from the same hidden_states
            self.qkv_proj = nn.Linear(dim, 3 * dim)

        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

    def _shape(self, x: torch.Tensor, B: int, L: int) -> torch.Tensor:
        # [B, L, H] -> [B, n_head, L, head_dim]
        return x.view(B, L, self.n_head, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, L_q, H]
        attention_mask: Optional[torch.Tensor] = None,   # [B, 1, 1, L_k] padding mask
        encoder_hidden_states: Optional[torch.Tensor] = None,  # [B, L_k, H]
        encoder_attention_mask: Optional[torch.Tensor] = None, # [B, 1, 1, L_k]
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns: (context, attn_weights)
          context: [B, L_q, H]
          attn_weights: [B, n_head, L_q, L_k] or None
        """
        B, L_q, H = hidden_states.shape

        if self.is_cross_attention and encoder_hidden_states is not None:
            # Cross-Attention: Q from hidden_states, K/V from encoder
            L_k = encoder_hidden_states.shape[1]

            q = self.q_proj(hidden_states)                        # [B, L_q, H]
            kv = self.kv_proj(encoder_hidden_states)              # [B, L_k, 2H]
            k, v = kv.split(H, dim=-1)                            # [B, L_k, H] each

            q = self._shape(q, B, L_q)                            # [B, h, L_q, d]
            k = self._shape(k, B, L_k)                            # [B, h, L_k, d]
            v = self._shape(v, B, L_k)                            # [B, h, L_k, d]

            attn_mask = encoder_attention_mask                    # [B,1,1,L_k] or None
        else:
            # Self-Attention: Q/K/V all come from hidden_states
            L_k = L_q
            qkv = self.qkv_proj(hidden_states)                    # [B, L_q, 3H]
            q, k, v = qkv.split(H, dim=-1)

            q = self._shape(q, B, L_q)
            k = self._shape(k, B, L_k)
            v = self._shape(v, B, L_k)

            attn_mask = attention_mask                            # [B,1,1,L_k] or None

        # [B, h, L_q, d] x [B, h, d, L_k] -> [B, h, L_q, L_k]
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        if attn_mask is not None:
            # attn_mask: [B, 1, 1, L_k] broadcast to [B, h, L_q, L_k]
            attn_scores = attn_scores + attn_mask

        attn_weights = F.softmax(attn_scores, dim=-1)   # [B, h, L_q, L_k]
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, v)         # [B, h, L_q, d]
        context = context.transpose(1, 2).contiguous().view(B, L_q, H)  # [B, L_q, H]
        context = self.out_proj(context)
        context = self.resid_dropout(context)

        if output_attentions:
            return context, attn_weights
        else:
            return context, None


class MLP(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_inner: Optional[int] = None,
        activation: str = "gelu",
        resid_pdrop: float = 0.1,
    ):
        super().__init__()
        inner_dim = n_inner if n_inner is not None else 4 * n_embd
        self.fc = nn.Linear(n_embd, inner_dim)
        self.proj = nn.Linear(inner_dim, n_embd)
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.act(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Single Layer Transformer Block:
      - Self-Attention
      - Option: Cross-Attention
      - MLP
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_inner: Optional[int] = None,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        add_cross_attention: bool = False,
        activation_function: str = "gelu",
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)
        self.attn = Attention(
            dim=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            is_cross_attention=False,
        )

        self.add_cross_attention = add_cross_attention
        if add_cross_attention:
            self.ln_cross_attn = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)
            self.cross_attn = Attention(
                dim=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                is_cross_attention=True,
            )

        self.ln_2 = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)
        self.mlp = MLP(
            n_embd=n_embd,
            n_inner=n_inner,
            activation=activation_function,
            resid_pdrop=resid_pdrop,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,                  # [B, L, H]
        attention_mask: Optional[torch.Tensor] = None,          # [B,1,1,L]
        encoder_hidden_states: Optional[torch.Tensor] = None,   # [Benc, L_code, H]
        encoder_attention_mask: Optional[torch.Tensor] = None,  # [Benc,1,1,L_code]
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns:
          hidden_states: [B, L, H]
          self_attn_weights: [B, n_head, L, L] or None
          cross_attn_weights: [B, n_head, L, L_code] or None
        """
        # Self-Attn
        residual = hidden_states
        x = self.ln_1(hidden_states)
        x, self_attn_weights = self.attn(
            x,
            attention_mask=attention_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=output_attentions,
        )
        hidden_states = residual + x

        cross_attn_weights = None
        if self.add_cross_attention and encoder_hidden_states is not None:
            residual = hidden_states
            x = self.ln_cross_attn(hidden_states)
            x, cross_attn_weights = self.cross_attn(
                x,
                attention_mask=None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            hidden_states = residual + x

        # MLP
        residual = hidden_states
        x = self.ln_2(hidden_states)
        x = self.mlp(x)
        hidden_states = residual + x

        return hidden_states, self_attn_weights, cross_attn_weights


# =========================
# Stage Fusion Transformer
# =========================

@dataclass
class SFTConfig:
    n_embd: int = 768
    n_head: int = 24
    n_inner: Optional[int] = None
    n_positions: int = 512          
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    add_cross_attention: bool = True
    activation_function: str = "gelu"


class StageFusionTransformer(nn.Module):
    """
    Three-stage fusion architecture:
      - Block0: self-attention over language tokens only
      - Inject state_inputs (images / state sequences)
      - Block1: self-attention over [lang + state] (global)
      - Block2: self-attention over [lang + state] plus cross-attention (codebook)
    Notes:
      - Do not add positional embeddings here:
        * lang_inputs: typically BERT outputs, already include positional information
        * state_inputs: upstream has already added time_embeddings
    """

    def __init__(self, config: SFTConfig):
        super().__init__()
        self.config = config
        H = config.n_embd

        self.block0 = Block(
            n_embd=H,
            n_head=config.n_head,
            n_inner=config.n_inner,
            attn_pdrop=config.attn_pdrop,
            resid_pdrop=config.resid_pdrop,
            layer_norm_epsilon=config.layer_norm_epsilon,
            add_cross_attention=False,
            activation_function=config.activation_function,
        )
        self.block1 = Block(
            n_embd=H,
            n_head=config.n_head,
            n_inner=config.n_inner,
            attn_pdrop=config.attn_pdrop,
            resid_pdrop=config.resid_pdrop,
            layer_norm_epsilon=config.layer_norm_epsilon,
            add_cross_attention=False,
            activation_function=config.activation_function,
        )
        self.block2 = Block(
            n_embd=H,
            n_head=config.n_head,
            n_inner=config.n_inner,
            attn_pdrop=config.attn_pdrop,
            resid_pdrop=config.resid_pdrop,
            layer_norm_epsilon=config.layer_norm_epsilon,
            add_cross_attention=config.add_cross_attention,
            activation_function=config.activation_function,
        )

        self.drop = nn.Dropout(config.resid_pdrop)
        self.ln_f = nn.LayerNorm(H, eps=config.layer_norm_epsilon)

    @staticmethod
    def _build_4d_mask(mask_2d: torch.Tensor) -> torch.Tensor:
        """
        mask_2d: [B, L], 1=keep, 0=pad
        Returns: [B, 1, 1, L], pad positions are set to -1e4
        """
        mask = mask_2d[:, None, None, :]         # [B,1,1,L]
        mask = (1.0 - mask.float()) * -1e4       # 1->0, 0->-1e4
        return mask

    def forward(
        self,
        lang_inputs: torch.Tensor,                # [B, L_text, H]  (BERT outputs)
        state_inputs: Optional[torch.Tensor] = None,  # [B, L_state, H] or None (already added time_embeddings)
        lang_mask: Optional[torch.Tensor] = None,     # [B, L_text], 1=keep,0=pad
        state_mask: Optional[torch.Tensor] = None,    # [B, L_state],1=keep,0=pad
        encoder_hidden_states: Optional[torch.Tensor] = None,   # codebook [Benc, L_code, H]
        encoder_attention_mask: Optional[torch.Tensor] = None,  # [Benc, L_code]
        output_hidden_states: bool = True,
        output_attentions: bool = True,
    ) -> Dict[str, Any]:
        """
        Returns:
          {
            "last_hidden_state": [B, L_total, H],
            "hidden_states": tuple([...]) or None,
            "attentions": tuple(self_attns) or None,
            "cross_attentions": tuple(cross_attns) or None,
          }
        """
        device = lang_inputs.device
        B, L_text, H = lang_inputs.shape

        if state_inputs is not None:
            assert state_inputs.shape[0] == B
            L_state = state_inputs.shape[1]
        else:
            L_state = 0

        # language mask
        if lang_mask is None:
            lang_mask = torch.ones(B, L_text, device=device)

        attn_mask_lang = self._build_4d_mask(lang_mask)   # [B,1,1,L_text]

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attns = () if (output_attentions and self.config.add_cross_attention) else None

        # ===== Block0: language only =====
        hidden_states = self.drop(lang_inputs)            # no pos added
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        hidden_states, self_attn0, _ = self.block0(
            hidden_states,
            attention_mask=attn_mask_lang,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=output_attentions,
        )
        if output_attentions:
            all_self_attns += (self_attn0,)

        # ===== state_inputs =====
        if state_inputs is not None:
            if state_mask is None:
                state_mask = torch.ones(B, L_state, device=device)

            hidden_states = torch.cat([hidden_states, state_inputs], dim=1)  # [B, L_text+L_state, H]
            full_mask_2d = torch.cat([lang_mask, state_mask], dim=1)         # [B, L_total]
            attn_mask_full = self._build_4d_mask(full_mask_2d)               # [B,1,1,L_total]
        else:
            attn_mask_full = attn_mask_lang

        L_total = hidden_states.shape[1]

        # ===== Block1: self-attention over [lang + state] =====
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        hidden_states, self_attn1, _ = self.block1(
            hidden_states,
            attention_mask=attn_mask_full,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=output_attentions,
        )
        if output_attentions:
            all_self_attns += (self_attn1,)

        # ===== Block2: self-attn + cross-attn(codebook) =====
        if encoder_hidden_states is not None:
            Benc, L_code, Henc = encoder_hidden_states.shape
            assert Henc == H, "codebook hidden size must match n_embd"

            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(Benc, L_code, device=device)
            enc_mask = self._build_4d_mask(encoder_attention_mask)   # [Benc,1,1,L_code]
        else:
            enc_mask = None

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        hidden_states, self_attn2, cross_attn2 = self.block2(
            hidden_states,
            attention_mask=attn_mask_full,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=enc_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            all_self_attns += (self_attn2,)
            if cross_attn2 is not None:
                all_cross_attns += (cross_attn2,)

        # final LN
        hidden_states = self.ln_f(hidden_states)          # [B, L_total, H]

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return {
            "last_hidden_state": hidden_states,          # [B, L_total, H]
            "hidden_states": all_hidden_states,          # tuple or None
            "attentions": all_self_attns,                # tuple or None
            "cross_attentions": all_cross_attns,         # tuple or None
        }
