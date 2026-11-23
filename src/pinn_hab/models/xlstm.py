"""
xLSTM: Extended Long Short-Term Memory for time-series forecasting.

Combines scalar sLSTM (exponential gating with a normalizer state) and
matrix mLSTM (matrix-valued memory cell) blocks from Beck et al. (2024).

Reference: Beck et al. (2024), "xLSTM: Extended Long Short-Term Memory"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class sLSTMCell(nn.Module):
    """
    sLSTM (scalar LSTM) with Exponential Gating.
    
    Key difference from vanilla LSTM:
    - Exponential input/forget gates for finer granularity
    - Normalizer state to prevent overflow
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Gates: input, forget, output, cell input
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(
        self, x_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Process full sequence.
        
        Args:
            x_seq: (batch, seq_len, input_size)
        Returns:
            h_seq: (batch, seq_len, hidden_size)
        """
        batch, seq_len, _ = x_seq.shape
        device = x_seq.device
        
        h = torch.zeros(batch, self.hidden_size, device=device)
        c = torch.zeros(batch, self.hidden_size, device=device)
        n = torch.ones(batch, self.hidden_size, device=device)  # normalizer state
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            combined = torch.cat([x_t, h], dim=1)
            gates = self.W(combined)
            
            i, f, o, z = gates.chunk(4, dim=1)
            
            # Exponential input gate
            i_gate = torch.exp(i)
            f_gate = torch.sigmoid(f)  # Sigmoid for forget to keep bounded
            o_gate = torch.sigmoid(o)
            z_tilde = torch.tanh(z)
            
            # Update cell state with normalizer
            c = f_gate * c + i_gate * z_tilde
            n = f_gate * n + i_gate  # normalizer tracks cumulative input scale
            
            # Normalized hidden state
            h_tilde = c / (n + 1e-8)  # stabilized division
            h = o_gate * self.layer_norm(h_tilde)
            
            outputs.append(h)
        
        return torch.stack(outputs, dim=1)


class mLSTMCell(nn.Module):
    """
    mLSTM cell storing a matrix state (d_k x d_v) updated via key-value outer products.
    """
    
    def __init__(self, input_size: int, hidden_size: int, head_dim: int = 16):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        
        # Query, Key, Value projections
        self.W_q = nn.Linear(input_size, head_dim, bias=False)
        self.W_k = nn.Linear(input_size, head_dim, bias=False)
        self.W_v = nn.Linear(input_size, hidden_size, bias=False)
        
        # Gates
        self.W_i = nn.Linear(input_size, 1)  # scalar input gate
        self.W_f = nn.Linear(input_size, 1)  # scalar forget gate
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)  # output gate
        
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Process full sequence.
        
        Args:
            x_seq: (batch, seq_len, input_size)
        Returns:
            h_seq: (batch, seq_len, hidden_size)
        """
        batch, seq_len, _ = x_seq.shape
        device = x_seq.device
        
        # Matrix memory: C is (batch, head_dim, hidden_size)
        C = torch.zeros(batch, self.head_dim, self.hidden_size, device=device)
        n = torch.zeros(batch, self.head_dim, 1, device=device)  # normalizer
        h = torch.zeros(batch, self.hidden_size, device=device)
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            
            # Projections
            q = self.W_q(x_t)  # (B, head_dim)
            k = self.W_k(x_t)  # (B, head_dim)
            v = self.W_v(x_t)  # (B, hidden_size)
            
            # Gates
            i_gate = torch.exp(self.W_i(x_t))  # (B, 1)
            f_gate = torch.sigmoid(self.W_f(x_t))  # (B, 1)
            
            # Update matrix memory: C = f*C + i*(k^T v)
            kv = k.unsqueeze(2) * v.unsqueeze(1)  # (B, head_dim, hidden_size)
            C = f_gate.unsqueeze(-1) * C + i_gate.unsqueeze(-1) * kv
            n = f_gate.unsqueeze(-1) * n + i_gate.unsqueeze(-1) * k.unsqueeze(2)
            
            # Retrieve from memory: h_tilde = C^T q / (n^T q)
            h_tilde = torch.bmm(q.unsqueeze(1), C).squeeze(1)  # (B, hidden_size)
            n_tilde = torch.bmm(q.unsqueeze(1), n).squeeze(1)  # (B, 1)
            h_tilde = h_tilde / (torch.abs(n_tilde) + 1e-8)
            
            # Output gate
            combined = torch.cat([x_t, h_tilde], dim=1)
            o_gate = torch.sigmoid(self.W_o(combined))
            h = o_gate * self.layer_norm(h_tilde)
            
            outputs.append(h)
        
        return torch.stack(outputs, dim=1)


class xLSTMBlock(nn.Module):
    """
    xLSTM block combining sLSTM and mLSTM with residual connection.
    """
    
    def __init__(
        self, d_model: int, block_type: str = 'slstm', head_dim: int = 16
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        
        if block_type == 'slstm':
            self.cell = sLSTMCell(d_model, d_model)
        elif block_type == 'mlstm':
            self.cell = mLSTMCell(d_model, d_model, head_dim=head_dim)
        else:
            raise ValueError(f"Unknown block type: {block_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cell(self.norm(x))


class xLSTMForecaster(nn.Module):
    """
    xLSTM-based Time Series Forecaster for HAB prediction.
    
    Architecture:
        Input → Linear Embedding → [sLSTM, mLSTM, sLSTM, ...] → Pool → Head
    
    Features:
        - Alternating sLSTM and mLSTM blocks
        - Station embedding for multi-site modeling
        - Gaussian output head for uncertainty quantification
    """
    
    def __init__(
        self,
        seq_len: int,
        n_features: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        head_dim: int = 16,
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
        
        # Input projection
        self.input_proj = nn.Linear(n_features, hidden_dim)
        
        # Station embedding
        if use_station_embedding:
            self.station_embedding = nn.Embedding(n_stations, hidden_dim // 4)
            head_input_dim = hidden_dim + hidden_dim // 4
        else:
            self.station_embedding = None
            head_input_dim = hidden_dim
        
        # Alternating sLSTM and mLSTM blocks
        block_types = ['slstm', 'mlstm'] * ((n_layers + 1) // 2)
        block_types = block_types[:n_layers]
        
        self.layers = nn.ModuleList([
            xLSTMBlock(hidden_dim, bt, head_dim) for bt in block_types
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
        h = self.input_proj(x)
        
        # xLSTM blocks
        for layer in self.layers:
            h = layer(h)
        
        h = self.norm_f(h)
        
        # Use last hidden state (recurrent model convention)
        h = h[:, -1, :]  # (B, hidden_dim)
        h = self.dropout(h)
        
        # Station embedding
        if self.use_station_embedding and station_ids is not None:
            station_emb = self.station_embedding(station_ids)
            h = torch.cat([h, station_emb], dim=1)
        
        # Gaussian output
        mean = self.mean_predictor(h)
        var = self.var_predictor(h) + 1e-6
        
        return mean, var
