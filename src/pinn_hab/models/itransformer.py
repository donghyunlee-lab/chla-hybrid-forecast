"""
iTransformer: Inverted Transformer for time-series forecasting (Liu et al., 2024).

Applies self-attention across the variable dimension rather than the time axis.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class iTransformer(nn.Module):
    """
    Inverted Transformer with station embedding and Gaussian output head.
    Applies self-attention across variables after 1D temporal encoding.
    """

    def __init__(
        self, 
        seq_len: int, 
        n_features: int, 
        hidden_dim: int = 64,
        n_heads: int = 4, 
        n_layers: int = 2, 
        dropout_rate: float = 0.3,
        n_stations: int = 5, 
        use_station_embedding: bool = True
    ):
        """
        Initialize iTransformer.
        
        Args:
            seq_len: Length of input sequences
            n_features: Number of input features
            hidden_dim: Hidden dimension size
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout_rate: Dropout probability
            n_stations: Number of stations for embedding
            use_station_embedding: Whether to use station embedding
        """
        super().__init__()

        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_stations = n_stations
        self.use_station_embedding = use_station_embedding
        self.dropout_rate = dropout_rate

        # Station Embedding
        if use_station_embedding:
            self.station_embedding = nn.Embedding(n_stations, hidden_dim // 4)
        else:
            self.station_embedding = None

        # 1D-CNN for local temporal pattern extraction
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(n_features, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Conv1d(hidden_dim // 2, n_features, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_features),
            nn.ReLU()
        )

        # Feature-wise embedding
        self.feature_embedding = nn.Linear(seq_len, hidden_dim)

        # Transformer encoder on variable dimension
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            batch_first=True,
            activation='gelu'
        )
        self.itransformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Temporal processor with station embedding integration
        self.embedding_dim = hidden_dim + (hidden_dim // 4 if use_station_embedding else 0)
        self.temporal_processor = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        # Gaussian output head
        self.mean_predictor = nn.Linear(hidden_dim // 2, 1)
        self.var_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus()
        )
        
        # Initialize variance bias
        with torch.no_grad():
            self.var_predictor[-2].bias.fill_(1.0)

    def forward(
        self, 
        x: torch.Tensor, 
        station_ids: Optional[torch.Tensor] = None, 
        mc_samples: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with Gaussian output.
        
        Args:
            x: Input tensor (batch_size, seq_len, n_features)
            station_ids: Station IDs (batch_size,)
            mc_samples: Not used (for API compatibility)
        
        Returns:
            mean: Predicted mean (batch_size, 1)
            var: Predicted variance (batch_size, 1)
        """
        eps = 1e-6

        # 1D-CNN: Extract local temporal patterns
        x_cnn_input = x.transpose(1, 2)  # (batch, n_features, seq_len)
        x_cnn_output = self.temporal_cnn(x_cnn_input)
        
        # Residual connection
        x_enhanced = x.transpose(1, 2) + x_cnn_output

        # Feature-wise embedding
        x_embedded = self.feature_embedding(x_enhanced)  # (batch, features, hidden_dim)

        # Transformer on variable dimension
        x_transformed = self.itransformer(x_embedded)

        # Global pooling across features
        x_pooled = x_transformed.mean(dim=1)

        # Integrate station embedding
        if self.use_station_embedding and station_ids is not None:
            station_emb = self.station_embedding(station_ids)
            x_pooled = torch.cat([x_pooled, station_emb], dim=1)

        # Temporal processing
        x_processed = self.temporal_processor(x_pooled)

        # Gaussian output
        mean = self.mean_predictor(x_processed)
        var = self.var_predictor(x_processed) + eps

        return mean, var
