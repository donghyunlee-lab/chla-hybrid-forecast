"""
Biogeochemical Physics Module for HAB (Harmful Algal Bloom) Dynamics.

Implements the governing ODE for algal growth:
    dC/dt = (μ - r - m) * C

Features:
1. Denormalization support for physics calculations
2. Sequence-level ODE residual computation
3. Autograd-based differentiation option
4. Collocation points support
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# Feature name patterns for dynamic index detection
PHYSICS_FEATURE_PATTERNS = {
    'temperature': ['Water Temperature', 'Temperature', '수온', 'Temp'],
    'nitrogen': ['TN', 'Total Nitrogen', '총질소', 'T-N'],
    'phosphorus': ['TP', 'Total Phosphorus', '총인', 'T-P'],
    'chlorophyll': ['Chl-a', 'Chlorophyll', '클로로필', 'Chl'],
    'solar_radiation': ['Solar', 'Radiation', '일사량', 'PAR', 'Light'],
}

# Default denormalization parameters (can be overridden)
DEFAULT_DENORM_PARAMS = {
    'temperature': {'mean': 20.0, 'std': 8.0, 'min': 0.0, 'max': 35.0},
    'nitrogen': {'mean': 2.0, 'std': 1.5, 'min': 0.0, 'max': 10.0},
    'phosphorus': {'mean': 0.1, 'std': 0.08, 'min': 0.0, 'max': 0.5},
    'chlorophyll': {'mean': 30.0, 'std': 40.0, 'min': 0.0, 'max': 300.0},
}


class EnhancedBiogeochemicalPhysics(nn.Module):
    """
    Biogeochemical Physics Module.
    
    Included components:
        1. Dynamic feature index mapping from feature_names
        2. Site-specific learnable parameters (μ_max per site)
        3. Denormalization for physically meaningful calculations
        4. Sequence-level ODE residual computation
        5. Light limitation and seasonal modulation
    """
    
    def __init__(
        self, 
        feature_names: Optional[List[str]] = None, 
        n_sites: int = 5,
        theta: float = 1.047, 
        freeze_params: bool = False,
        use_light_limitation: bool = True,
        use_seasonal_modulation: bool = True,
        use_site_specific: bool = True,
        use_denormalization: bool = True,
        denorm_params: Optional[Dict] = None,
        use_sequence_ode: bool = True
    ):
        """
        Initialize the physics module.
        
        Args:
            feature_names: List of feature names for dynamic index detection
            n_sites: Number of sites for site-specific parameters
            theta: van 't Hoff coefficient (default: 1.047)
            freeze_params: If True, fix parameters to initial values
            use_light_limitation: Enable light limitation factor
            use_seasonal_modulation: Enable seasonal modulation
            use_site_specific: Enable per-site μ_max
            use_denormalization: Enable denormalization for physics
            denorm_params: Custom denormalization parameters
            use_sequence_ode: Apply ODE across full sequence
        """
        super().__init__()
        
        self.n_sites = n_sites
        self.theta = theta
        self.use_light_limitation = use_light_limitation
        self.use_seasonal_modulation = use_seasonal_modulation
        self.use_site_specific = use_site_specific
        self.use_denormalization = use_denormalization
        self.use_sequence_ode = use_sequence_ode
        
        # Dynamic feature index mapping
        self.feature_indices = self._detect_feature_indices(feature_names)
        
        # Denormalization parameters (learnable or fixed)
        self.denorm_params = denorm_params or DEFAULT_DENORM_PARAMS
        self._register_denorm_buffers()
        
        # ===== LEARNABLE PARAMETERS =====
        
        # Parameter ranges for validation (in REAL units)
        # Literature-based ranges for biological parameters
        self.param_ranges = {
            'mu_max': (0.1, 1.5),      # day^-1 (growth rate)
            'K_N': (0.01, 0.5),        # mg/L - Bowie et al. (1985)
            'K_P': (0.001, 0.05),      # mg/L - Bowie et al. (1985)
            'K_I': (50.0, 1000.0),     # μmol/m²/s
            'r': (0.01, 0.2),          # day^-1 (respiration rate)
            'm': (0.005, 0.1),         # day^-1 (mortality rate)
            'seasonal_amplitude': (0.0, 0.5),
            'seasonal_phase': (0.0, 365.0),
        }
        
        # Site-specific maximum growth rate (μ_max) with hard constraints
        if use_site_specific:
            self.mu_max_embedding = nn.Embedding(n_sites, 1)
            # Initialize to center of valid range: (0.1 + 1.5) / 2 = 0.8
            mu_max_center = (self.param_ranges['mu_max'][0] + self.param_ranges['mu_max'][1]) / 2.0
            # Inverse sigmoid to get embedding value: sigmoid^-1(0.8) for range [0.1, 1.5]
            # sigmoid(x) maps to [0, 1], then scale to [min, max]
            # We want: sigmoid(x) * (max - min) + min = center
            # x = logit((center - min) / (max - min))
            mu_range = self.param_ranges['mu_max'][1] - self.param_ranges['mu_max'][0]
            mu_normalized = (mu_max_center - self.param_ranges['mu_max'][0]) / mu_range
            mu_logit = np.log(mu_normalized / (1 - mu_normalized + 1e-8))
            nn.init.constant_(self.mu_max_embedding.weight, mu_logit)
        else:
            # Initialize mu_max to center of range using inverse sigmoid
            mu_max_center = (self.param_ranges['mu_max'][0] + self.param_ranges['mu_max'][1]) / 2.0
            mu_range = self.param_ranges['mu_max'][1] - self.param_ranges['mu_max'][0]
            mu_normalized = (mu_max_center - self.param_ranges['mu_max'][0]) / mu_range
            mu_logit = np.log(mu_normalized / (1 - mu_normalized + 1e-8))
            self.mu_max = nn.Parameter(torch.tensor(mu_logit, dtype=torch.float32))
        
        # Half-saturation constants (Monod kinetics) - Sigmoid hard constraint
        # K_N: [0.01, 0.5] mg/L, initialize to center
        K_N_center = (self.param_ranges['K_N'][0] + self.param_ranges['K_N'][1]) / 2.0
        K_N_range = self.param_ranges['K_N'][1] - self.param_ranges['K_N'][0]
        K_N_normalized = (K_N_center - self.param_ranges['K_N'][0]) / K_N_range
        K_N_logit = np.log(K_N_normalized / (1 - K_N_normalized + 1e-8))
        self.K_N_raw = nn.Parameter(torch.tensor(K_N_logit, dtype=torch.float32))
        
        # K_P: [0.001, 0.05] mg/L, initialize to center
        K_P_center = (self.param_ranges['K_P'][0] + self.param_ranges['K_P'][1]) / 2.0
        K_P_range = self.param_ranges['K_P'][1] - self.param_ranges['K_P'][0]
        K_P_normalized = (K_P_center - self.param_ranges['K_P'][0]) / K_P_range
        K_P_logit = np.log(K_P_normalized / (1 - K_P_normalized + 1e-8))
        self.K_P_raw = nn.Parameter(torch.tensor(K_P_logit, dtype=torch.float32))
        
        # Light limitation parameter
        if use_light_limitation:
            self.K_I = nn.Parameter(torch.tensor(300.0, dtype=torch.float32))  # μmol/m²/s
        
        # Seasonal modulation parameters
        if use_seasonal_modulation:
            self.seasonal_amplitude = nn.Parameter(torch.tensor(0.3, dtype=torch.float32))
            self.seasonal_phase = nn.Parameter(torch.tensor(172.0, dtype=torch.float32))  # ~summer peak
        
        # Loss rates (day^-1) with hard constraints via Sigmoid scaling
        # Initialize to center of valid ranges
        r_center = (self.param_ranges['r'][0] + self.param_ranges['r'][1]) / 2.0  # (0.01 + 0.2) / 2 = 0.105
        r_range = self.param_ranges['r'][1] - self.param_ranges['r'][0]  # 0.19
        r_normalized = (r_center - self.param_ranges['r'][0]) / r_range  # (0.105 - 0.01) / 0.19 ≈ 0.5
        r_logit = np.log(r_normalized / (1 - r_normalized + 1e-8))
        self.r_raw = nn.Parameter(torch.tensor(r_logit, dtype=torch.float32))
        
        m_center = (self.param_ranges['m'][0] + self.param_ranges['m'][1]) / 2.0  # (0.005 + 0.1) / 2 = 0.0525
        m_range = self.param_ranges['m'][1] - self.param_ranges['m'][0]  # 0.095
        m_normalized = (m_center - self.param_ranges['m'][0]) / m_range  # (0.0525 - 0.005) / 0.095 ≈ 0.5
        m_logit = np.log(m_normalized / (1 - m_normalized + 1e-8))
        self.m_raw = nn.Parameter(torch.tensor(m_logit, dtype=torch.float32))
        
        if freeze_params:
            self._freeze_all_params()
            print("[INFO] Physics parameters are FROZEN")
    
    def _register_denorm_buffers(self):
        """Register denormalization parameters as buffers."""
        for var_name, params in self.denorm_params.items():
            self.register_buffer(
                f'{var_name}_mean', 
                torch.tensor(params.get('mean', 0.0), dtype=torch.float32)
            )
            self.register_buffer(
                f'{var_name}_std', 
                torch.tensor(params.get('std', 1.0), dtype=torch.float32)
            )
    
    def denormalize(
        self, 
        x: torch.Tensor, 
        var_type: str
    ) -> torch.Tensor:
        """
        Denormalize a variable to its physical scale.
        
        Args:
            x: Normalized tensor
            var_type: Variable type ('temperature', 'nitrogen', etc.)
        
        Returns:
            Denormalized tensor in physical units
        """
        if not self.use_denormalization:
            return x
        
        mean = getattr(self, f'{var_type}_mean', 0.0)
        std = getattr(self, f'{var_type}_std', 1.0)
        
        return x * std + mean
    
    def _detect_feature_indices(
        self, 
        feature_names: Optional[List[str]]
    ) -> Dict[str, Optional[int]]:
        """Dynamically detect feature indices from feature names."""
        indices = {
            'temperature': None,
            'nitrogen': None,
            'phosphorus': None,
            'chlorophyll': None,
            'solar_radiation': None,
        }
        
        if feature_names is None:
            print("[WARNING] No feature_names provided, using hardcoded indices")
            indices['temperature'] = 0
            indices['nitrogen'] = -2
            indices['phosphorus'] = -1
            indices['chlorophyll'] = -3
            return indices
        
        # Search for each feature type
        for feat_type, patterns in PHYSICS_FEATURE_PATTERNS.items():
            for idx, fname in enumerate(feature_names):
                for pattern in patterns:
                    if pattern.lower() in fname.lower():
                        indices[feat_type] = idx
                        break
                if indices[feat_type] is not None:
                    break
        
        detected = {k: v for k, v in indices.items() if v is not None}
        print(f"[Physics] Detected feature indices: {detected}")
        
        return indices
    
    def _freeze_all_params(self):
        """Freeze all learnable parameters."""
        for param in self.parameters():
            param.requires_grad = False
    
    def get_mu_max(self, station_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get site-specific or global μ_max with hard constraint via Sigmoid scaling.
        
        Returns μ_max in range [0.1, 1.5] day^-1
        """
        if self.use_site_specific and station_ids is not None:
            mu_max_raw = self.mu_max_embedding(station_ids).squeeze(-1)
        else:
            mu_max_raw = getattr(self, 'mu_max', torch.tensor(0.0, dtype=torch.float32))
        
        # Apply sigmoid and scale to [min, max]
        mu_max_sigmoid = torch.sigmoid(mu_max_raw)
        mu_min, mu_max = self.param_ranges['mu_max']
        mu_range = mu_max - mu_min
        mu_max_scaled = mu_min + mu_max_sigmoid * mu_range
        
        return mu_max_scaled
    
    def get_r(self) -> torch.Tensor:
        """
        Get respiration rate with hard constraint via Sigmoid scaling.
        
        Returns r in range [0.01, 0.2] day^-1
        """
        r_sigmoid = torch.sigmoid(self.r_raw)
        r_min, r_max = self.param_ranges['r']
        r_range = r_max - r_min
        r_scaled = r_min + r_sigmoid * r_range
        return r_scaled
    
    def get_m(self) -> torch.Tensor:
        """
        Get mortality rate with hard constraint via Sigmoid scaling.
        
        Returns m in range [0.005, 0.1] day^-1
        """
        m_sigmoid = torch.sigmoid(self.m_raw)
        m_min, m_max = self.param_ranges['m']
        m_range = m_max - m_min
        m_scaled = m_min + m_sigmoid * m_range
        return m_scaled
    
    def temperature_factor(self, T: torch.Tensor) -> torch.Tensor:
        """
        van 't Hoff temperature dependency: f(T) = θ^(T-20)
        
        Args:
            T: Temperature in Celsius (DENORMALIZED)
        """
        # Clamp temperature to prevent extreme values
        T_clamped = torch.clamp(T, min=0.0, max=40.0)
        return self.theta ** (T_clamped - 20.0)
    
    def get_K_N(self) -> torch.Tensor:
        """
        Get half-saturation constant for nitrogen with Sigmoid constraint.
        
        Returns K_N in range [0.01, 0.5] mg/L
        """
        K_N_sigmoid = torch.sigmoid(self.K_N_raw)
        K_N_min, K_N_max = self.param_ranges['K_N']
        return K_N_min + K_N_sigmoid * (K_N_max - K_N_min)
    
    def get_K_P(self) -> torch.Tensor:
        """
        Get half-saturation constant for phosphorus with Sigmoid constraint.
        
        Returns K_P in range [0.001, 0.05] mg/L
        """
        K_P_sigmoid = torch.sigmoid(self.K_P_raw)
        K_P_min, K_P_max = self.param_ranges['K_P']
        return K_P_min + K_P_sigmoid * (K_P_max - K_P_min)
    
    def nutrient_limitation(
        self, 
        N: torch.Tensor, 
        P: torch.Tensor
    ) -> torch.Tensor:
        """
        Monod kinetics for nutrient limitation (multiplicative model).
        
        Args:
            N: Total Nitrogen in mg/L (DENORMALIZED)
            P: Total Phosphorus in mg/L (DENORMALIZED)
        """
        K_N = self.get_K_N()
        K_P = self.get_K_P()
        
        # Ensure non-negative nutrients
        N = F.relu(N)
        P = F.relu(P)
        
        f_N = N / (K_N + N + 1e-8)
        f_P = P / (K_P + P + 1e-8)
        
        return f_N * f_P
    
    def light_limitation(self, I: Optional[torch.Tensor]) -> torch.Tensor:
        """Light saturation curve (Monod-type)."""
        if not self.use_light_limitation or I is None:
            return torch.tensor(1.0, dtype=torch.float32)
        
        K_I = F.softplus(self.K_I)
        I = F.relu(I)  # Ensure non-negative
        f_I = I / (K_I + I + 1e-8)
        return f_I
    
    def seasonal_modulation(self, doy: Optional[torch.Tensor]) -> torch.Tensor:
        """Seasonal growth modulation based on day-of-year."""
        if not self.use_seasonal_modulation or doy is None:
            return torch.tensor(1.0, dtype=torch.float32)
        
        A = torch.sigmoid(self.seasonal_amplitude)
        phi = self.seasonal_phase
        
        S = 1.0 + A * torch.sin(2 * np.pi * (doy - phi) / 365.0)
        return S
    
    def growth_rate(
        self, 
        T: torch.Tensor, 
        N: torch.Tensor, 
        P: torch.Tensor,
        I: Optional[torch.Tensor] = None, 
        doy: Optional[torch.Tensor] = None,
        station_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate algal growth rate μ.
        
        All inputs should be in PHYSICAL units (denormalized).
        """
        mu_max = self.get_mu_max(station_ids)
        f_T = self.temperature_factor(T)
        f_nutrient = self.nutrient_limitation(N, P)
        f_I = self.light_limitation(I)
        S = self.seasonal_modulation(doy)
        
        # Ensure all factors are tensors
        if not isinstance(f_I, torch.Tensor):
            f_I = torch.ones_like(T)
        if not isinstance(S, torch.Tensor):
            S = torch.ones_like(T)
        
        # Handle dimension mismatch
        if mu_max.dim() < f_T.dim():
            mu_max = mu_max.unsqueeze(-1)
        
        return mu_max * f_T * f_nutrient * f_I * S
    
    def net_rate(
        self, 
        T: torch.Tensor, 
        N: torch.Tensor, 
        P: torch.Tensor,
        I: Optional[torch.Tensor] = None, 
        doy: Optional[torch.Tensor] = None,
        station_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate net growth rate (μ - r - m).
        
        Uses hard-constrained parameters via Sigmoid scaling.
        """
        mu = self.growth_rate(T, N, P, I, doy, station_ids)
        r = self.get_r()
        m = self.get_m()
        return mu - r - m
    
    def extract_environmental_vars(
        self, 
        x_features: torch.Tensor,
        dates: Optional[torch.Tensor] = None,
        timestep: int = -1
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Extract and denormalize environmental variables from input features.
        
        Args:
            x_features: (batch, seq_len, n_features) - NORMALIZED
            dates: Optional tensor of dates for DOY extraction
            timestep: Which timestep to extract (-1 for last)
        
        Returns:
            Dict with T, N, P, C_current, I, doy - all DENORMALIZED
        """
        idx = self.feature_indices
        batch_size = x_features.shape[0]
        device = x_features.device
        
        # Extract normalized values
        T_norm = x_features[:, timestep, idx['temperature']] if idx['temperature'] is not None else torch.zeros(batch_size, device=device)
        N_norm = x_features[:, timestep, idx['nitrogen']] if idx['nitrogen'] is not None else torch.zeros(batch_size, device=device)
        P_norm = x_features[:, timestep, idx['phosphorus']] if idx['phosphorus'] is not None else torch.zeros(batch_size, device=device)
        C_norm = x_features[:, timestep, idx['chlorophyll']] if idx['chlorophyll'] is not None else torch.zeros(batch_size, device=device)
        
        # Denormalize to physical units
        T = self.denormalize(T_norm, 'temperature')
        N = self.denormalize(N_norm, 'nitrogen')
        P = self.denormalize(P_norm, 'phosphorus')
        C_current = self.denormalize(C_norm, 'chlorophyll')
        
        I = None
        if idx['solar_radiation'] is not None:
            I = x_features[:, timestep, idx['solar_radiation']]
        
        return {
            'T': T, 'N': N, 'P': P, 'C_current': C_current,
            'I': I, 'doy': dates
        }
    
    def extract_sequence_vars(
        self, 
        x_features: torch.Tensor,
        dates: Optional[torch.Tensor] = None
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Extract and denormalize environmental variables for FULL SEQUENCE.
        
        Args:
            x_features: (batch, seq_len, n_features) - NORMALIZED
            dates: Optional tensor of dates
        
        Returns:
            Dict with T, N, P, C - all (batch, seq_len), DENORMALIZED
        """
        idx = self.feature_indices
        batch_size, seq_len = x_features.shape[0], x_features.shape[1]
        device = x_features.device
        
        # Extract normalized values for full sequence
        T_norm = x_features[:, :, idx['temperature']] if idx['temperature'] is not None else torch.zeros(batch_size, seq_len, device=device)
        N_norm = x_features[:, :, idx['nitrogen']] if idx['nitrogen'] is not None else torch.zeros(batch_size, seq_len, device=device)
        P_norm = x_features[:, :, idx['phosphorus']] if idx['phosphorus'] is not None else torch.zeros(batch_size, seq_len, device=device)
        C_norm = x_features[:, :, idx['chlorophyll']] if idx['chlorophyll'] is not None else torch.zeros(batch_size, seq_len, device=device)
        
        # Denormalize
        T = self.denormalize(T_norm, 'temperature')
        N = self.denormalize(N_norm, 'nitrogen')
        P = self.denormalize(P_norm, 'phosphorus')
        C = self.denormalize(C_norm, 'chlorophyll')
        
        return {'T': T, 'N': N, 'P': P, 'C': C, 'doy': dates}
    
    def compute_sequence_ode_residual(
        self,
        x_features: torch.Tensor,
        station_ids: Optional[torch.Tensor] = None,
        dt: float = 1.0
    ) -> torch.Tensor:
        """
        Compute ODE residual across the FULL SEQUENCE.
        
        For consecutive timesteps t and t+1:
            C(t+1) ≈ C(t) + net_rate(t) * C(t) * dt
        
        Returns:
            Mean ODE residual across sequence
        """
        if not self.use_sequence_ode:
            return torch.tensor(0.0, device=x_features.device, dtype=torch.float32)
        
        seq_vars = self.extract_sequence_vars(x_features)
        T, N, P, C = seq_vars['T'], seq_vars['N'], seq_vars['P'], seq_vars['C']
        
        # Compute net rate for each timestep
        # net_rate: (batch, seq_len)
        net_rate = self.net_rate(T, N, P, station_ids=station_ids)
        
        # ODE residual: |C(t+1) - (C(t) + net_rate(t) * C(t) * dt)|
        C_t = C[:, :-1]           # (batch, seq_len-1)
        C_next = C[:, 1:]         # (batch, seq_len-1)
        rate_t = net_rate[:, :-1] # (batch, seq_len-1)
        
        # Expected C(t+1) from ODE
        C_expected = C_t + rate_t * C_t * dt
        
        # Residual with Huber loss (smooth L1) for robustness
        residual = F.smooth_l1_loss(C_next, C_expected, reduction='none')
        
        return residual.mean()
    
    def clamp_parameters(self):
        """
        Enforce physical constraints on parameters.
        
        Note: r, m, mu_max, K_N, K_P are now hard-constrained via Sigmoid,
        so no clamping is needed for these. Only clamp other parameters.
        """
        with torch.no_grad():
            # r, m, mu_max, K_N, K_P are already constrained by Sigmoid, skip them
            if self.use_light_limitation:
                self.K_I.clamp_(*self.param_ranges['K_I'])
            if self.use_seasonal_modulation:
                self.seasonal_amplitude.clamp_(*self.param_ranges['seasonal_amplitude'])
    
    def get_parameters_dict(self, station_id: Optional[int] = None) -> Dict:
        """Get current parameter values as dictionary (using hard-constrained values)."""
        params = {
            'K_N': self.get_K_N().item(),
            'K_P': self.get_K_P().item(),
            'r': self.get_r().item(),  # Use hard-constrained value
            'm': self.get_m().item(),   # Use hard-constrained value
            'theta': self.theta,
        }
        
        if self.use_site_specific and station_id is not None:
            station_tensor = torch.tensor([station_id], device=self.r_raw.device)
            params['mu_max'] = self.get_mu_max(station_tensor).item()
        elif not self.use_site_specific:
            params['mu_max'] = self.get_mu_max().item()
        
        if self.use_light_limitation:
            params['K_I'] = F.softplus(self.K_I).item()
        
        if self.use_seasonal_modulation:
            params['seasonal_amplitude'] = torch.sigmoid(self.seasonal_amplitude).item()
            params['seasonal_phase'] = self.seasonal_phase.item()
        
        return params
    
    def get_all_site_parameters(self) -> pd.DataFrame:
        """Get parameters for all sites as DataFrame."""
        if not self.use_site_specific:
            return pd.DataFrame([self.get_parameters_dict()])
        
        rows = []
        for site_id in range(self.n_sites):
            params = self.get_parameters_dict(site_id)
            params['site_id'] = site_id
            rows.append(params)
        return pd.DataFrame(rows)
    
    def set_denorm_params_from_scaler(
        self, 
        scaler, 
        feature_names: List[str]
    ):
        """
        Set denormalization parameters from a sklearn scaler.
        
        Args:
            scaler: sklearn StandardScaler or MinMaxScaler
            feature_names: List of feature names matching scaler order
        """
        idx = self.feature_indices
        
        for var_type, feat_idx in idx.items():
            if feat_idx is None:
                continue
            
            if hasattr(scaler, 'mean_'):
                # StandardScaler
                mean = scaler.mean_[feat_idx]
                std = scaler.scale_[feat_idx]
            elif hasattr(scaler, 'data_min_'):
                # MinMaxScaler - convert to mean/std approximation
                data_min = scaler.data_min_[feat_idx]
                data_max = scaler.data_max_[feat_idx]
                mean = (data_min + data_max) / 2
                std = (data_max - data_min) / 2
            else:
                continue
            
            setattr(self, f'{var_type}_mean', torch.tensor(mean, dtype=torch.float32))
            setattr(self, f'{var_type}_std', torch.tensor(std, dtype=torch.float32))
        
        print(f"[Physics] Denormalization parameters set from scaler")


# Backward compatibility alias
BiogeochemicalPhysics = EnhancedBiogeochemicalPhysics
