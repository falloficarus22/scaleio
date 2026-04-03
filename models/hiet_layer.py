import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class HierarchicalEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        y_coords = torch.arange(h, device=device).float()
        x_coords = torch.arange(h, device=device).float()

        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        yy = yy / (h - 1) if h > 1 else yy
        xx = xx / (w - 1) if w > 1 else xx

        coords = torch.stack([yy.flatten(), xx.flatten()], dim=-1)

        encoding = []
        for i in range(self.d_model // 4):
            freq = 2.0**i
            encoding.append(torch.sin(coords[:, 0] * freq * math.pi))
            encoding.append(torch.cos(coords[:, 0] * freq * math.pi))
            encoding.append(torch.sin(coords[:, 1] * freq * math.pi))
            encoding.append(torch.cos(coords[:, 1] * freq * math.pi))

        if self.d_model % 4 != 0:
            encoding.append(torch.zeros_like(coords[:, 0]))

        delta_hw = torch.stack(encoding, dim=-1)

        return delta_hw


class ChannelSelfCorrection(nn.Module):
    def __init__(self, dim: int, num_heads: int = 0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        v = v.transpose(-2, -1)
        out = attn @ v

        out = out.transpose(-2, -1)

        out = out.reshape(B, N, C)
        out = self.proj(out)


class HiETLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (8, 8),
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.h, self.w = window_size

        self.hierarchical_encoding = HierarchicalEncoding(dim)
        self.input_proj = nn.Linear(dim, dim)
        self.split_dim = dim // 2

        self.csc = ChannelSelfCorrection(self.split_dim, num_heads)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def window_partition(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape

        x = x.view(B, H // self.h, self.h, W // self.w, self.w, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, self.h * self.w, C)

        return windows

    def window_reverse(self, windows: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B = int(windows.shape[0] / (H * W / (self.h * self.w)))

        x = windows.view(B, H // self.h, W // self.w, self.h, self.w, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H, W, -1)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape

        delta_hw = self.hierarchical_encoding(self.h, self.w, x.device)
        delta_hw = delta_hw.unsqueeze(0).expand(B, -1, -1)

        x_windows = self.window_partition(x)
        x_with_encoding = torch.cat([x_windows.delta_hw], dim=-1)
        x_proj = self.input_proj(x_with_encoding)

        q, v = torch.split(x_proj, self.split, dim=-1)
        attn_out = self.csc(q)
        x_attn = torch.cat([attn_out, v], dim=-1)
        x_attn = self.norm1(x_windows + x_attn)

        x_mlp = self.mlp(x_attn)
        x_out = self.norm2(x_attn + x_mlp)
        x_out = self.window_reverse(x_out, H, W)

        return x_out
