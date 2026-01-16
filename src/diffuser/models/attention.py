import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner = heads * dim_head
        context_dim = context_dim if context_dim is not None else query_dim

        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_k = nn.Linear(context_dim, inner, bias=False)
        self.to_v = nn.Linear(context_dim, inner, bias=False)
        self.to_out = nn.Linear(inner, query_dim)

    def forward(self, x, context=None, context_mask=None):
        """
        x: (B, T, Cq)
        context: (B, N, Cc) or None (None -> self-attn)
        context_mask: (B, N) bool, True=keep, False=pad  (optional)
        """
        b, t, _ = x.shape
        context = x if context is None else context

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # (B, heads, T/N, dim_head)
        q = q.view(b, t, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(b, context.shape[1], self.heads, self.dim_head).transpose(1, 2)
        v = v.view(b, context.shape[1], self.heads, self.dim_head).transpose(1, 2)

        # PyTorch 2.x: scaled_dot_product_attention
        attn_mask = None
        if context_mask is not None:
            # Convert 0/1 valid mask to "padding = True" mask
            # True = positions to be masked = pad token
            attn_mask = (context_mask == 0)          # [B, L_k], bool
            attn_mask = attn_mask[:, None, None, :]   # [B, 1, 1, L_k] -> broadcast to (B, h, L_q, L_k)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # (B, heads, T, dim_head)
        out = out.transpose(1, 2).contiguous().view(b, t, self.heads * self.dim_head)
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, context_dim=None, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(dim, context_dim=None, heads=heads, dim_head=dim_head)          # self-attn
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = CrossAttention(dim, context_dim=context_dim, heads=heads, dim_head=dim_head)  # cross-attn
        self.ff = FeedForward(dim, dropout=dropout)

    def forward(self, x, context=None, context_mask=None):
        x = x + self.attn1(self.norm1(x), context=None)
        x = x + self.attn2(self.norm2(x), context=context, context_mask=context_mask)
        x = x + self.ff(x)
        return x

class SpatialTransformer1D(nn.Module):
    """
    LDM SpatialTransformer 1D version:
    (B,C,T) -> (B,T,inner) into transformer -> back to (B,C,T), residual connection
    """
    def __init__(self, channels, heads, dim_head, depth=1, context_dim=None, n_groups=8, dropout=0.0):
        super().__init__()
        inner = heads * dim_head

        self.norm = nn.GroupNorm(n_groups, channels)
        self.proj_in = nn.Conv1d(channels, inner, kernel_size=1)
        self.transformer = nn.ModuleList([
            BasicTransformerBlock(inner, heads=heads, dim_head=dim_head, context_dim=context_dim, dropout=dropout)
            for _ in range(depth)
        ])
        self.proj_out = nn.Conv1d(inner, channels, kernel_size=1)

    def forward(self, x, context=None, context_mask=None):
        # x: (B,C,T)
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)                         # (B,inner,T)
        x = einops.rearrange(x, 'b c t -> b t c')   # (B,T,inner)
        for blk in self.transformer:
            x = blk(x, context=context, context_mask=context_mask)
        x = einops.rearrange(x, 'b t c -> b c t')
        x = self.proj_out(x)
        return x + x_in
