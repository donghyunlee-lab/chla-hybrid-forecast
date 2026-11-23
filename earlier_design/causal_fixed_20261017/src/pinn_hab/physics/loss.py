"""
Physics-Informed Loss Function for HAB Prediction.

Features:
1. Sequence-level ODE residual across full input
2. Gradient balancing between data and physics losses
3. Collocation points support for data-sparse regions
4. Curriculum learning schedule
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .biogeochemical import EnhancedBiogeochemicalPhysics


class EnhancedPhysicsInformedLoss(nn.Module):
    """
    Physics-Informed Loss Function.
    
    Features:
        1. Sequence-level ODE residual (not just last timestep)
        2. Multi-step prediction horizons (1, 3, 7 days)
        3. Curriculum learning with warmup + sigmoid schedule
        4. Gradient balancing option
        5. Collocation points for ODE enforcement
        6. Parameter regularization toward literature values
    
    Total Loss = Data Loss + λ(epoch) × Physics Loss + γ × Param Regularization
    """
    
    def __init__(
        self, 
        physics_module: EnhancedBiogeochemicalPhysics,
        w_physics_min: float = 0.01,
        w_physics_max: float = 0.3,
        curriculum_epochs: int = 70,
        warmup_epochs: int = 10,
        use_multi_step: bool = True,
        multi_step_horizons: List[int] = [1, 3, 7],
        param_regularization: float = 0.01,
        use_gradient_balancing: bool = True,
        use_collocation: bool = True,
        n_collocation_points: int = 20,
        sequence_ode_weight: float = 0.5,
        net_growth_threshold: float = -0.1,
        net_growth_penalty_weight: float = 1.0,
        # parameters for data-scarce settings
        n_train_samples: Optional[int] = None,
        use_adaptive_physics_weight: bool = True,
        adaptive_weighting_method: str = 'relobralo',  # 'relobralo' or 'ntk'
        base_n_samples: int = 10000  # Reference sample size for normalization
    ):
        """
        Initialize the loss function.
        
        Args:
            physics_module: EnhancedBiogeochemicalPhysics instance
            w_physics_min: Minimum physics weight (start of curriculum)
            w_physics_max: Maximum physics weight (end of curriculum)
            curriculum_epochs: Epochs to reach max physics weight
            warmup_epochs: Epochs before physics loss kicks in
            use_multi_step: Enable multi-step prediction residuals
            multi_step_horizons: Time horizons for multi-step (days)
            param_regularization: Weight for parameter regularization
            use_gradient_balancing: Enable dynamic gradient balancing
            use_collocation: Enable collocation points
            n_collocation_points: Number of collocation points per batch
            sequence_ode_weight: Weight for sequence ODE loss
            net_growth_threshold: Threshold for net growth penalty (default: -0.1)
            net_growth_penalty_weight: Weight for net growth penalty loss
            n_train_samples: Number of training samples (for adaptive weighting)
            use_adaptive_physics_weight: Enable adaptive physics weight based on data size
            adaptive_weighting_method: Method for adaptive weighting ('relobralo' or 'ntk')
            base_n_samples: Reference sample size for normalization
        """
        super().__init__()
        
        self.physics = physics_module
        self.w_physics_min = w_physics_min
        self.w_physics_max_base = w_physics_max  # Base maximum weight
        self.curriculum_epochs = curriculum_epochs
        self.warmup_epochs = warmup_epochs
        self.use_multi_step = use_multi_step
        self.multi_step_horizons = multi_step_horizons
        self.param_reg_weight = param_regularization
        self.use_gradient_balancing = use_gradient_balancing
        self.use_collocation = use_collocation
        self.n_collocation_points = n_collocation_points
        self.sequence_ode_weight = sequence_ode_weight
        self.net_growth_threshold = net_growth_threshold
        self.net_growth_penalty_weight = net_growth_penalty_weight
        
        # Adaptive physics weighting for data scarcity
        self.n_train_samples = n_train_samples
        self.use_adaptive_physics_weight = use_adaptive_physics_weight
        self.adaptive_weighting_method = adaptive_weighting_method
        self.base_n_samples = base_n_samples

        # Rerun fixes, applied in every phase:
        # - the prediction residual links the forecast to the last input day over the forecast lead time only
        #   (the original compared one forecast with 1-, 3- and 7-day projections);
        # - no pull of the kinetic parameters towards the centre of their ranges (it masked how far they would move);
        # - no penalty on net growth below -0.1 day^-1 (it pushed the kinetics towards growth regardless of the data).
        self.use_multi_step = False
        self.param_reg_weight = 0.0
        self.net_growth_penalty_weight = 0.0
        self.prediction_dt = float(os.environ.get('PINN_HORIZON', '1'))
        
        # Compute adaptive w_physics_max based on data size
        if self.use_adaptive_physics_weight and self.n_train_samples is not None:
            self.w_physics_max = self._compute_adaptive_physics_weight(
                self.n_train_samples, self.w_physics_max_base
            )
        else:
            self.w_physics_max = self.w_physics_max_base
        
        # Running gradient statistics for balancing
        self.register_buffer('grad_data_ema', torch.tensor(1.0))
        self.register_buffer('grad_physics_ema', torch.tensor(1.0))
        self.grad_ema_decay = 0.99
        
        # For ReLoBRaLo: running statistics
        self.register_buffer('relobralo_alpha', torch.tensor(0.5))
        self.relobralo_momentum = 0.9
    
    def _compute_adaptive_physics_weight(self, n_samples: int, base_max: float) -> float:
        """
        Compute adaptive physics weight based on data scarcity.
        
        Strategy: Less data → Higher physics weight (up to 2x base_max)
        Formula: w_max = base_max * (1 + (1 - min(n_samples / base_n_samples, 1.0)))
        
        Examples:
            - n_samples = 10000 (base): w_max = base_max * 1.0
            - n_samples = 5000: w_max = base_max * 1.5
            - n_samples = 1000: w_max = base_max * 1.9
            - n_samples = 100: w_max = base_max * 2.0 (capped)
        """
        if not self.use_adaptive_physics_weight:
            return base_max
        
        # Normalize sample count (0 = very sparse, 1 = abundant)
        sample_ratio = min(n_samples / self.base_n_samples, 1.0)
        
        # Inverse relationship: less data → more physics weight
        # Scale factor: [1.0, 2.0] range
        scale_factor = 1.0 + (1.0 - sample_ratio)
        
        adaptive_max = base_max * scale_factor
        
        # Cap at reasonable maximum (2x base or 0.6, whichever is lower)
        adaptive_max = min(adaptive_max, min(2.0 * base_max, 0.6))
        
        return float(adaptive_max)
    
    def update_adaptive_physics_weight(self, n_samples: int):
        """Update adaptive physics weight dynamically during training."""
        if self.use_adaptive_physics_weight:
            self.w_physics_max = self._compute_adaptive_physics_weight(
                n_samples, self.w_physics_max_base
            )
    
    def get_physics_weight(self, epoch: Optional[int]) -> float:
        """
        Curriculum learning: warmup, then a sigmoid schedule.
        
        Schedule:
            - epochs < warmup: w = w_min (very low, focus on data)
            - warmup <= epochs < curriculum: Smooth sigmoid ramp-up
            - epochs >= curriculum: w = w_max (full physics weight)
        
        The ramp-up is gradual early in the curriculum.
        """
        if epoch is None:
            return (self.w_physics_min + self.w_physics_max) / 2
        
        if epoch < self.warmup_epochs:
            return self.w_physics_min
        
        # sigmoid schedule after warmup
        effective_epoch = float(epoch - self.warmup_epochs)
        effective_curriculum = float(self.curriculum_epochs - self.warmup_epochs)
        
        if effective_curriculum <= 0:
            return self.w_physics_max
        
        # Normalized progress [0, 1]
        progress = effective_epoch / effective_curriculum
        
        # sigmoid: smooth ramp-up
        # Using tanh for smoother transition: tanh maps [-1, 1] → [0, 1] with sigmoid
        # Alternative: use polynomial easing for more control
        if progress < 0.3:
            # Early phase: very gradual increase (polynomial easing)
            eased = (progress / 0.3) ** 2
        else:
            # Later phase: sigmoid ramp
            x = ((progress - 0.3) / 0.7 - 0.5) * 8.0  # Steepness=8
            sigmoid_part = 1.0 / (1.0 + np.exp(-x))
            eased = 0.3 + 0.7 * sigmoid_part
        
        weight = float(self.w_physics_min + (self.w_physics_max - self.w_physics_min) * eased)
        return weight
    
    def compute_prediction_ode_residual(
        self, 
        C_pred: torch.Tensor, 
        C_current: torch.Tensor,
        net_rate: torch.Tensor, 
        dt: float = 1.0
    ) -> torch.Tensor:
        """
        Compute ODE residual for model prediction.
        
        Expected: C(t+dt) = C(t) + (μ - r - m) * C(t) * dt
        """
        # Rerun fix: the forecast is converted to mg/m3 like C_current (the original compared the normalized
        # forecast with a denormalized concentration), and the residual is ln C(t+dt) - ln C(t) - net_rate * dt.
        eps = 1e-3
        C_pred_phys = self.physics.denormalize(C_pred.reshape(-1), 'chlorophyll')
        res = torch.log(C_pred_phys + eps) - torch.log(C_current.reshape(-1) + eps) - net_rate.reshape(-1) * dt
        return F.smooth_l1_loss(res, torch.zeros_like(res), reduction='none')
    
    def compute_collocation_loss(
        self,
        x_features: torch.Tensor,
        station_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute ODE residual at collocation points (random timesteps).
        
        This enforces physics even in data-sparse regions.
        """
        if not self.use_collocation:
            return torch.tensor(0.0, device=x_features.device, dtype=torch.float32)
        
        batch_size, seq_len, n_features = x_features.shape
        device = x_features.device
        
        # Random collocation points within sequence
        n_colloc = min(self.n_collocation_points, seq_len - 1)
        colloc_indices = torch.randint(0, seq_len - 1, (n_colloc,), device=device)
        
        total_residual = torch.tensor(0.0, device=device, dtype=torch.float32)
        
        for t in colloc_indices:
            # Extract vars at timestep t
            env_t = self.physics.extract_environmental_vars(x_features, timestep=t.item())
            T, N, P = env_t['T'], env_t['N'], env_t['P']
            C_t = env_t['C_current']
            
            # Compute net rate at t
            net_rate = self.physics.net_rate(T, N, P, station_ids=station_ids)
            
            # Get C at t+1
            env_next = self.physics.extract_environmental_vars(x_features, timestep=t.item() + 1)
            C_next = env_next['C_current']
            
            # ODE residual
            res = torch.log(C_next + 1e-3) - torch.log(C_t + 1e-3) - net_rate * 1.0  # dt = 1 day, log space (rerun fix)
            residual = F.smooth_l1_loss(res, torch.zeros_like(res), reduction='mean')
            total_residual = total_residual + residual
        
        return total_residual / n_colloc if n_colloc > 0 else total_residual
    
    def compute_physics_loss(
        self, 
        C_pred: torch.Tensor, 
        x_features: torch.Tensor,
        station_ids: Optional[torch.Tensor] = None,
        dates: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive physics loss.
        
        Returns:
            Dict with individual physics loss components
        """
        device = x_features.device
        losses = {}
        
        # 1. Prediction ODE residual (model output vs physics)
        env_vars = self.physics.extract_environmental_vars(x_features, dates)
        T, N, P = env_vars['T'], env_vars['N'], env_vars['P']
        C_current = env_vars['C_current']
        I, doy = env_vars['I'], env_vars['doy']
        
        net_rate = self.physics.net_rate(T, N, P, I, doy, station_ids)
        
        if self.use_multi_step:
            residuals = []
            for dt in self.multi_step_horizons:
                weight = 1.0 / dt
                residual = self.compute_prediction_ode_residual(C_pred, C_current, net_rate, dt=float(dt))
                residuals.append(weight * residual.mean())
            
            losses['prediction_ode'] = torch.stack(residuals).sum()
        else:
            residual = self.compute_prediction_ode_residual(C_pred, C_current, net_rate, dt=self.prediction_dt)
            losses['prediction_ode'] = residual.mean()
        
        # 2. Sequence ODE residual (internal consistency)
        if self.physics.use_sequence_ode:
            seq_ode_loss = self.physics.compute_sequence_ode_residual(x_features, station_ids)
            losses['sequence_ode'] = self.sequence_ode_weight * seq_ode_loss
        else:
            losses['sequence_ode'] = torch.tensor(0.0, device=device, dtype=torch.float32)
        
        # 3. Collocation points loss
        losses['collocation'] = self.compute_collocation_loss(x_features, station_ids)
        
        # 4. Non-negativity constraint (soft barrier)
        losses['non_negative'] = F.relu(-self.physics.chl_log1p(C_pred)).mean()  # Chl-a < 0 only (rerun fix)
        
        # 5. Physical plausibility (Chl-a should have reasonable range)
        # Penalize predictions > 500 mg/m³ (extreme bloom)
        C_pred_denorm = self.physics.denormalize(C_pred.squeeze(), 'chlorophyll')
        extreme_penalty = F.relu(C_pred_denorm - 500).mean() * 0.01
        losses['plausibility'] = extreme_penalty
        
        # 6. Net Growth sign constraint penalty
        # Penalize when net growth (μ - r - m) falls below threshold
        # This ensures biological plausibility: growth should not be too negative
        net_rate = self.physics.net_rate(T, N, P, I, doy, station_ids)
        net_growth_violation = F.relu(self.net_growth_threshold - net_rate)  # Penalty if net_rate < threshold
        losses['net_growth_penalty'] = self.net_growth_penalty_weight * net_growth_violation.mean()
        
        return losses
    
    def compute_parameter_regularization(self) -> torch.Tensor:
        """Regularize learned parameters toward literature values."""
        # Use K_N_raw as the reference for device
        reg_loss = torch.tensor(0.0, device=self.physics.K_N_raw.device, dtype=torch.float32)
        ranges = self.physics.param_ranges
        
        # Regularize toward center of valid range
        # Note: All parameters now use getter methods (hard-constrained via Sigmoid)
        
        # 1. K_N and K_P
        for param_name in ['K_N', 'K_P']:
            getter = getattr(self.physics, f"get_{param_name}")
            val = getter()
            center = (ranges[param_name][0] + ranges[param_name][1]) / 2
            reg_loss = reg_loss + (val - center) ** 2
        
        # 2. r and m
        for param_name in ['r', 'm']:
            getter = getattr(self.physics, f"get_{param_name}")
            val = getter()
            center = (ranges[param_name][0] + ranges[param_name][1]) / 2
            reg_loss = reg_loss + (val - center) ** 2
            
        # 3. mu_max (if not site-specific)
        if not self.physics.use_site_specific:
            val = self.physics.get_mu_max()
            center = (ranges['mu_max'][0] + ranges['mu_max'][1]) / 2
            reg_loss = reg_loss + (val - center) ** 2
        
        return reg_loss
    
    def update_gradient_balance(
        self, 
        data_loss: torch.Tensor, 
        physics_loss: torch.Tensor,
        model_params: List[torch.Tensor]
    ) -> float:
        """
        Gradient balancing with ReLoBRaLo or NTK-based adaptive weighting.
        
        Methods:
        1. 'relobralo': ReLoBRaLo (Relative Loss Balancing with Random Lookback)
           - Uses exponential moving average with random lookback
           - More stable for data-scarce scenarios
        
        2. 'ntk': Neural Tangent Kernel inspired
           - Balances based on loss magnitudes and gradient norms
           - Better for small datasets
        """
        if not self.use_gradient_balancing:
            return 1.0
        
        try:
            if self.adaptive_weighting_method == 'relobralo':
                return self._relobralo_balancing(data_loss, physics_loss, model_params)
            elif self.adaptive_weighting_method == 'ntk':
                return self._ntk_balancing(data_loss, physics_loss, model_params)
            else:
                # Fallback to original GradNorm method
                return self._gradnorm_balancing(data_loss, physics_loss, model_params)
        except Exception as e:
            # Fallback to 1.0 if any error occurs
            return 1.0
    
    def _gradnorm_balancing(
        self,
        data_loss: torch.Tensor,
        physics_loss: torch.Tensor,
        model_params: List[torch.Tensor]
    ) -> float:
        """Original GradNorm balancing method."""
        grad_data = torch.autograd.grad(
            data_loss, model_params, 
            retain_graph=True, allow_unused=True
        )
        grad_physics = torch.autograd.grad(
            physics_loss, model_params,
            retain_graph=True, allow_unused=True
        )
        
        norm_data = sum(g.norm() for g in grad_data if g is not None)
        norm_physics = sum(g.norm() for g in grad_physics if g is not None)
        
        self.grad_data_ema = self.grad_ema_decay * self.grad_data_ema + (1 - self.grad_ema_decay) * norm_data.detach()
        self.grad_physics_ema = self.grad_ema_decay * self.grad_physics_ema + (1 - self.grad_ema_decay) * norm_physics.detach()
        
        balance = (self.grad_data_ema / (self.grad_physics_ema + 1e-8)).item()
        balance = np.clip(balance, 0.1, 10.0)
        
        return balance
    
    def _relobralo_balancing(
        self,
        data_loss: torch.Tensor,
        physics_loss: torch.Tensor,
        model_params: List[torch.Tensor]
    ) -> float:
        """
        ReLoBRaLo: Relative Loss Balancing with Random Lookback.
        
        More stable for data-scarce scenarios by using relative loss ratios
        instead of absolute gradient norms.
        """
        # Compute relative loss ratio
        loss_ratio = (data_loss.detach() / (physics_loss.detach() + 1e-8)).item()
        
        # Update alpha (balancing factor) using momentum
        # If data loss is much larger, increase physics weight (decrease alpha)
        # If physics loss is much larger, decrease physics weight (increase alpha)
        target_ratio = 1.0  # We want balanced losses
        
        # Adaptive alpha update
        alpha_update = target_ratio / (loss_ratio + 1e-8)
        alpha_update = np.clip(alpha_update, 0.5, 2.0)  # Reasonable bounds
        
        # Exponential moving average with momentum
        self.relobralo_alpha = (
            self.relobralo_momentum * self.relobralo_alpha + 
            (1 - self.relobralo_momentum) * torch.tensor(alpha_update, device=self.relobralo_alpha.device)
        )
        
        balance = self.relobralo_alpha.item()
        balance = np.clip(balance, 0.2, 5.0)  # Wider bounds for stability
        
        return balance
    
    def _ntk_balancing(
        self,
        data_loss: torch.Tensor,
        physics_loss: torch.Tensor,
        model_params: List[torch.Tensor]
    ) -> float:
        """
        NTK-inspired balancing: Uses both loss magnitudes and gradient information.
        
        Better for small datasets by considering the "effective capacity" of each loss.
        """
        # Compute gradient norms
        grad_data = torch.autograd.grad(
            data_loss, model_params,
            retain_graph=True, allow_unused=True
        )
        grad_physics = torch.autograd.grad(
            physics_loss, model_params,
            retain_graph=True, allow_unused=True
        )
        
        norm_data = sum(g.norm() for g in grad_data if g is not None)
        norm_physics = sum(g.norm() for g in grad_physics if g is not None)
        
        # Combine loss ratio and gradient ratio
        loss_ratio = (data_loss.detach() / (physics_loss.detach() + 1e-8)).item()
        grad_ratio = (norm_data / (norm_physics + 1e-8)).item()
        
        # NTK-inspired: balance based on both loss and gradient information
        # Weight the gradient ratio more for small datasets
        combined_ratio = 0.3 * loss_ratio + 0.7 * grad_ratio
        
        # Update EMAs
        self.grad_data_ema = self.grad_ema_decay * self.grad_data_ema + (1 - self.grad_ema_decay) * norm_data.detach()
        self.grad_physics_ema = self.grad_ema_decay * self.grad_physics_ema + (1 - self.grad_ema_decay) * norm_physics.detach()
        
        # Balance factor
        balance = (self.grad_data_ema / (self.grad_physics_ema + 1e-8)).item()
        
        # Adjust based on combined ratio
        balance = balance * (1.0 / (combined_ratio + 1e-8))
        balance = np.clip(balance, 0.1, 10.0)
        
        return balance
    
    def forward(
        self, 
        C_pred: torch.Tensor, 
        C_true: torch.Tensor,
        x_features: torch.Tensor, 
        var: Optional[torch.Tensor] = None,
        station_ids: Optional[torch.Tensor] = None, 
        dates: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
        model_params: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total physics-informed loss.
        
        Returns:
            Dict: All loss components and weights
        """
        device = x_features.device
        
        # Data loss (Gaussian NLL or MSE)
        if var is not None:
            var = torch.clamp(var, min=1e-6)
            # Use float32 version of pi
            pi = torch.tensor(np.pi, dtype=torch.float32, device=device)
            data_loss = 0.5 * torch.log(2.0 * pi * var) + \
                       0.5 * (C_true - C_pred) ** 2 / var
            data_loss = data_loss.mean()
        else:
            data_loss = F.mse_loss(C_pred, C_true)
        
        # Physics losses
        physics_losses = self.compute_physics_loss(C_pred, x_features, station_ids, dates)
        physics_loss_total = sum(physics_losses.values())
        
        # Parameter regularization
        param_reg = self.compute_parameter_regularization()
        
        # Get curriculum weight
        w_physics = self.get_physics_weight(epoch)
        
        # Optional gradient balancing
        grad_balance = 1.0
        if self.use_gradient_balancing and model_params is not None and epoch is not None and epoch > self.warmup_epochs:
            grad_balance = self.update_gradient_balance(data_loss, physics_loss_total, model_params)
        
        # Total loss
        effective_w_physics = w_physics * grad_balance
        k_pen = self.physics.k_penalty()                     # soft K_N/K_P constraint, zero in other modes (rerun fix)
        total_loss = data_loss + effective_w_physics * physics_loss_total + self.param_reg_weight * param_reg + k_pen
        
        return {
            'total': total_loss,
            'data': data_loss,
            'physics': physics_loss_total,
            'physics_prediction_ode': physics_losses['prediction_ode'],
            'physics_sequence_ode': physics_losses['sequence_ode'],
            'physics_collocation': physics_losses['collocation'],
            'physics_non_negative': physics_losses['non_negative'],
            'physics_net_growth_penalty': physics_losses['net_growth_penalty'],
            'param_reg': param_reg,
            'k_penalty': k_pen,
            'w_physics': torch.tensor(float(w_physics), device=device, dtype=torch.float32),
            'grad_balance': torch.tensor(float(grad_balance), device=device, dtype=torch.float32),
        }


# Alias for backward compatibility
PhysicsInformedLoss = EnhancedPhysicsInformedLoss
