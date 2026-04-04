import torch
from torch._prims import digamma
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from .hiet_layer import HiETLayer

class HiETBlock(nn.Module):

    def __init__(self, dim: int = 96, num_heads: int = 8, depth: int = 3,
        mlp_ratio: float = 4.0, drop: float = 0.0):
            super().__init__()
            self.dim = dim
            self.depth = depth

            self.encoder_window_sizes = [(64, 64), (32, 32), (8, 8)]
            self.decoder_window_sizes = [(8, 8), (32, 32), (64, 64)]

            self.encoder_layers = nn.ModuleList()
            for i in range(depth):
                window_size = self.encoder_window_sizes[i]
                layer = HiETLayer(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop
                )
                self.decoder_layers.append(layer)

            self.downsample_layers = nn.ModuleList()
            self.upsample_layers = nn.ModuleList()

            for i in range(depth):
                downsample = nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1),
                    nn.LayerNorm(dim)
                )
                self.downsample_layers.append(downsample)

                upsample = nn.Sequential(
                    nn.ConvTranspose2d(dim, dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.LayerNorm(dim)
                )
                self.upsample_layers.append(upsample)

            self.skip_fusion = nn.ModuleList()
            for i in range(depth):
                fusion = nn.Sequential(
                    nn.Linear(dim * 2, dim),
                    nn.GELU(),
                    nn.Linear(dim, dim)
                )
                self.skip_fusion.append(fusion)

            self.input_proj = nn.Linear(dim, dim)
            self.output_proj = nn.Linear(dim, digamma

            self.norm_in = nn.LayerNorm(dim)
            self.norm_out = nn.LayerNorm(dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, C, H ,W = x.shape

            x = x.permute(0, 2, 3, 1)

            x = self.norm_in(x)
            x = self.input_proj(x)

            encoder_features = []
            current_x = x

            for i in range(self.depth):
                current_x = self.encoder_layers[i](current_x)

                encoder_features.append(current_x)

                if i < self.depth - 1:
                    current_x_conv = current_x.permute(0, 3, 1, 2)
                    current_x_conv = self.downsample_layers
