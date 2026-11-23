"""
ModernTCN: Temporal convolutional network with large depthwise-separable kernels.

Reference: Luo & Wang (2024), "ModernTCN: A Modern Pure Convolution
Structure for General Time Series Analysis"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DWConvBlock(nn.Module):
    """
    Large-kernel depthwise convolution followed by LayerNorm and two pointwise projections.
    """
    
    def __init__(
        self, d_model: int, kernel_size: int = 51, expand_ratio: int = 4
    ):
        super().__init__()
        
        d_inner = d_model * expand_ratio
        padding = (kernel_size - 1) // 2
        
        # Large-kernel depthwise convolution
        self.dwconv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=padding,
            groups=d_model,  # Depthwise
            bias=True
        )
        
        self.norm = nn.LayerNorm(d_model)
        
        # Pointwise channel expansion and projection
        self.pwconv1 = nn.Linear(d_model, d_inner)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(d_inner, d_model)
        
        # Learnable layer scale (ConvNeXt style)
        self.gamma = nn.Parameter(1e-4 * torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        residual = x
        
        # Depthwise conv (channels-first)
        h = x.transpose(1, 2)  # (B, d_model, L)
        h = self.dwconv(h)
        h = h.transpose(1, 2)  # (B, L, d_model)
        
        # Norm + Pointwise
        h = self.norm(h)
        h = self.pwconv1(h)
        h = self.act(h)
        h = self.pwconv2(h)
        
        # Layer scale + residual
        h = self.gamma * h
        return residual + h


class ModernTCNBlock(nn.Module):
    """
    ModernTCN block: Decoupled temporal and feature processing.
    
    Step 1: Large-kernel DWConv processes temporal patterns PER feature
    Step 2: Cross-feature mixing via pointwise convolution
    """
    
    def __init__(
        self, d_model: int, kernel_size: int = 51, expand_ratio: int = 4
    ):
        super().__init__()
        
        # Temporal processing (per-feature)
        self.temporal = DWConvBlock(d_model, kernel_size, expand_ratio)
        
        # Feature mixing
        self.feature_mix = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        self.gamma_mix = nn.Parameter(1e-4 * torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Temporal
        x = self.temporal(x)
        
        # Feature mixing with residual
        x = x + self.gamma_mix * self.feature_mix(x)
        return x


class ModernTCNForecaster(nn.Module):
    """
    ModernTCN-based Time Series Forecaster for HAB prediction.
    
    Architecture:
        Input → Stem Conv → N × ModernTCNBlock → Global Pool → Head
    
    Features:
        - Large-kernel (51) depthwise convolutions for wide receptive field
        - Layer scale and pointwise channel expansion
        - Station embedding for multi-site modeling
        - Gaussian output head for uncertainty quantification
    """
    
    def __init__(
        self,
        seq_len: int,
        n_features: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        kernel_size: int = 51,
        expand_ratio: int = 4,
        dropout_rate: float = 0.3,
        n_stations: int = 5,
        use_station_embedding: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.use_station_embedding = use_station_embedding
        
        # Stem: project features to hidden dim
        self.stem = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Station embedding
        if use_station_embedding:
            self.station_embedding = nn.Embedding(n_stations, hidden_dim // 4)
            head_input_dim = hidden_dim + hidden_dim // 4
        else:
            self.station_embedding = None
            head_input_dim = hidden_dim
        
        # ModernTCN blocks with progressively larger kernels
        actual_kernel = min(kernel_size, seq_len)
        if actual_kernel % 2 == 0:
            actual_kernel -= 1  # Ensure odd kernel
        
        self.layers = nn.ModuleList([
            ModernTCNBlock(hidden_dim, actual_kernel, expand_ratio)
            for _ in range(n_layers)
        ])
        
        self.norm_f = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Gaussian output head
        self.mean_predictor = nn.Linear(head_input_dim, 1)
        self.var_predictor = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus()
        )
        
        with torch.no_grad():
            self.var_predictor[-2].bias.fill_(1.0)
    
    def forward(
        self,
        x: torch.Tensor,
        station_ids: Optional[torch.Tensor] = None,
        mc_samples: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, n_features)
            station_ids: (batch,)
            mc_samples: Not used (API compatibility)
        
        Returns:
            mean: (batch, 1)
            var: (batch, 1)
        """
        # Stem
        h = self.stem(x)  # (B, L, hidden_dim)
        
        # ModernTCN blocks
        for layer in self.layers:
            h = layer(h)
        
        h = self.norm_f(h)
        
        # Global average pooling
        h = h.mean(dim=1)  # (B, hidden_dim)
        h = self.dropout(h)
        
        # Station embedding
        if self.use_station_embedding and station_ids is not None:
            station_emb = self.station_embedding(station_ids)
            h = torch.cat([h, station_emb], dim=1)
        
        # Gaussian output
        mean = self.mean_predictor(h)
        var = self.var_predictor(h) + 1e-6
        
        return mean, var
