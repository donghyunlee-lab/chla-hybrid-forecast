"""
Base model class for all prediction models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for HAB prediction models.
    
    All models should output (mean, variance) for uncertainty quantification.
    """
    
    def __init__(
        self,
        seq_len: int,
        n_features: int,
        hidden_dim: int = 64,
        n_stations: int = 5,
        use_station_embedding: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_stations = n_stations
        self.use_station_embedding = use_station_embedding
        self.dropout_rate = dropout_rate
        
        # Station embedding (optional)
        if use_station_embedding:
            self.station_embedding = nn.Embedding(n_stations, hidden_dim // 4)
        else:
            self.station_embedding = None
    
    @abstractmethod
    def forward(
        self, 
        x: torch.Tensor, 
        station_ids: Optional[torch.Tensor] = None,
        mc_samples: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, n_features)
            station_ids: Station IDs (batch,)
            mc_samples: Number of MC samples (for MC Dropout)
        
        Returns:
            Tuple of (mean, variance) predictions
        """
        pass
    
    def get_station_embedding(
        self, 
        station_ids: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Get station embedding if available."""
        if self.station_embedding is not None and station_ids is not None:
            return self.station_embedding(station_ids)
        return None
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GaussianHead(nn.Module):
    """
    Gaussian output head for heteroscedastic uncertainty.
    
    Outputs mean and variance predictions.
    """
    
    def __init__(self, input_dim: int, output_dim: int = 1, hidden_dim: int = 32):
        super().__init__()
        
        self.mean_predictor = nn.Linear(input_dim, output_dim)
        
        self.var_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()
        )
        
        # Initialize variance bias
        with torch.no_grad():
            self.var_predictor[-2].bias.fill_(1.0)
    
    def forward(self, x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features (batch, input_dim)
            eps: Small constant for numerical stability
        
        Returns:
            Tuple of (mean, variance)
        """
        mean = self.mean_predictor(x)
        var = self.var_predictor(x) + eps
        return mean, var
