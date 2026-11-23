"""
Mamba: Selective State Space Model for Time Series Forecasting.

Uses selective scan mechanism for efficient sequence modeling with
linear complexity. Designed for HAB prediction with station embedding
and Gaussian output head.

Reference: Gu & Dao (2023), "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SelectiveSSMBlock(nn.Module):
    """
    Simplified Selective State Space Model block.
    
    Implements the core Mamba mechanism:
    - Input-dependent selection (B, C, Δ are functions of input)
    - Discretized state space: x(t+1) = Āx(t) + B̄u(t), y(t) = Cx(t)
    """
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        
        # Linear projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Depthwise convolution for local context
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True
        )
        
        # SSM parameters (input-dependent)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # B, C, Δ
        
        # Learnable A (log-space for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # Dual projection: main path + gate
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_main, z = xz.chunk(2, dim=-1)  # Each (B, L, d_inner)
        
        # Causal convolution
        x_conv = x_main.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Causal: trim to seq_len
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_conv = F.silu(x_conv)
        
        # Input-dependent SSM parameters
        ssm_params = self.x_proj(x_conv)  # (B, L, 2*d_state + 1)
        B = ssm_params[:, :, :self.d_state]  # (B, L, d_state)
        C = ssm_params[:, :, self.d_state:2*self.d_state]  # (B, L, d_state)
        delta = F.softplus(ssm_params[:, :, -1:])  # (B, L, 1) - step size
        
        # Discretize A
        A = -torch.exp(self.A_log)  # (d_inner, d_state), negative for stability
        
        # Selective scan (sequential for correctness)
        y = self._selective_scan(x_conv, A, B, C, delta)
        
        # Skip connection with D
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        
        # Gate and output
        y = y * F.silu(z)
        return self.out_proj(y)
    
    def _selective_scan(
        self, u: torch.Tensor, A: torch.Tensor, 
        B: torch.Tensor, C: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        """
        Selective scan operation.
        
        Args:
            u: (B, L, d_inner) - input
            A: (d_inner, d_state) - state matrix
            B: (B, L, d_state) - input-dependent B
            C: (B, L, d_state) - input-dependent C
            delta: (B, L, 1) - step size
        """
        batch, seq_len, d_inner = u.shape
        d_state = A.shape[1]
        device = u.device
        
        # Discretize: Ā = exp(Δ * A), B̄ = Δ * B
        delta_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, d_inner, d_state)
        delta_B_u = (delta * B).unsqueeze(2) * u.unsqueeze(-1)  # (B, L, d_inner, d_state)
        
        # Sequential scan
        h = torch.zeros(batch, d_inner, d_state, device=device, dtype=u.dtype)
        ys = []
        
        for t in range(seq_len):
            h = delta_A[:, t] * h + delta_B_u[:, t]  # (B, d_inner, d_state)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
            ys.append(y_t)
        
        return torch.stack(ys, dim=1)  # (B, L, d_inner)


class MambaBlock(nn.Module):
    """Mamba block with residual connection and layer norm."""
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSMBlock(d_model, d_state, d_conv, expand)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))


class MambaForecaster(nn.Module):
    """
    Mamba-based Time Series Forecaster for HAB prediction.
    
    Architecture:
        Input → Linear Embedding → N × MambaBlock → Global Pool → Head
    
    Features:
        - Selective State Space for efficient long-range dependencies
        - Station embedding for multi-site modeling
        - Gaussian output head for uncertainty quantification
    """
    
    def __init__(
        self,
        seq_len: int,
        n_features: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout_rate: float = 0.3,
        n_stations: int = 5,
        use_station_embedding: bool = True,
        **kwargs  # Accept extra args for compatibility
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.use_station_embedding = use_station_embedding
        
        # Input embedding
        self.input_proj = nn.Linear(n_features, hidden_dim)
        
        # Station embedding
        if use_station_embedding:
            self.station_embedding = nn.Embedding(n_stations, hidden_dim // 4)
            head_input_dim = hidden_dim + hidden_dim // 4
        else:
            self.station_embedding = None
            head_input_dim = hidden_dim
        
        # Mamba layers
        self.layers = nn.ModuleList([
            MambaBlock(hidden_dim, d_state, d_conv, expand)
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
        # Input projection
        h = self.input_proj(x)  # (B, L, hidden_dim)
        
        # Mamba blocks
        for layer in self.layers:
            h = layer(h)
        
        h = self.norm_f(h)
        
        # Global average pooling over time
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
