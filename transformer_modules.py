import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x)->torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x)
        return output * self.weight

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._cached_emb = None
        self._cached_seq_len = None
        self._cached_device = None

    def forward(self, seq_len: int, device) -> torch.Tensor:
        if (
            self._cached_emb is not None
            and self._cached_seq_len == seq_len
            and self._cached_device == device
        ):
            return self._cached_emb

        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        emb = emb.unsqueeze(0).unsqueeze(0)

        self._cached_emb = emb
        self._cached_seq_len = seq_len
        self._cached_device = device

        return emb

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, freqs):
    return (x * freqs.cos()) + (rotate_half(x) * freqs.sin())

class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, max_seq_len = 512, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q:torch.Tensor = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        freqs = self.rope(seq_len, q.device)
        q = apply_rotary_pos_emb(q, freqs)
        k = apply_rotary_pos_emb(k, freqs)
        
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        # Add causal mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn_weights = attn_weights.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.dim
        )
        
        return self.out_proj(attn_output)

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or int(4 * dim * 2 / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            SwiGLU(dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ff_hidden_dim: Optional[int] = None,
        max_seq_len = 512,
        dropout: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        
        self.attn_norm = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, max_seq_len, dropout)
        
        self.ff_norm = RMSNorm(dim)
        self.ff = FeedForward(dim, ff_hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(self.attn_norm(x))
        y = h + self.ff(self.ff_norm(h))
        return y

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        dim: int,
        out_dim: int,
        num_layers: int,
        num_heads: int = 8,
        ff_hidden_dim: Optional[int] = None,
        max_seq_len = 512,
        dropout: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers

        self.proj_in = nn.Linear(inp_dim, dim)
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(dim, num_heads, ff_hidden_dim, max_seq_len, dropout)
            for _ in range(num_layers)
        ])

        self.proj_out = nn.Linear(dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        for layer in self.layers:
            x = layer(x)
        return self.proj_out(x)