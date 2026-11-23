"""Backbones with a multi-horizon head, an LSTM baseline, and the PINN container."""
from __future__ import annotations
import sys, torch, torch.nn as nn
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pinn_hab.models.factory import get_model          # noqa: E402
from physics import PhysicsParams                        # noqa: E402


class LSTMForecaster(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim=64, n_layers=2, dropout_rate=0.3, n_stations=6, use_station_embedding=True, n_horizons=3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, n_layers, batch_first=True, dropout=dropout_rate)
        self.use_station_embedding = use_station_embedding
        self.station_embedding = nn.Embedding(n_stations, hidden_dim // 4) if use_station_embedding else None
        self.dropout = nn.Dropout(dropout_rate)
        self.mean_predictor = nn.Linear(hidden_dim + (hidden_dim // 4 if use_station_embedding else 0), n_horizons)

    def forward(self, x, station_ids=None, mc_samples=1):
        h, _ = self.lstm(x); h = self.dropout(h[:, -1])
        if self.use_station_embedding and station_ids is not None:
            h = torch.cat([h, self.station_embedding(station_ids)], 1)
        return self.mean_predictor(h), None


def build_backbone(name: str, n_features: int, n_stations: int, seq_len: int, n_horizons: int, hidden_dim=64, n_layers=2, dropout=0.3):
    if name == "LSTM":
        return LSTMForecaster(seq_len, n_features, hidden_dim, n_layers, dropout, n_stations, True, n_horizons)
    cfg = {"model_type": name, "seq_len": seq_len, "hidden_dim": hidden_dim, "n_layers": n_layers, "n_heads": 4,
           "dropout_rate": dropout, "use_station_embedding": True}
    m = get_model(cfg, n_features=n_features, n_stations=n_stations)
    m.mean_predictor = nn.Linear(m.mean_predictor.in_features, n_horizons)   # multi-horizon head
    return m


class Net(nn.Module):
    def __init__(self, backbone: nn.Module, physics: PhysicsParams | None):
        super().__init__()
        self.backbone, self.physics = backbone, physics

    def forward(self, x, sid):
        out = self.backbone(x, sid)
        return out[0] if isinstance(out, tuple) else out

    def n_params(self, part="all"):
        if part == "backbone":
            return sum(p.numel() for p in self.backbone.parameters())
        return sum(p.numel() for p in self.parameters())
