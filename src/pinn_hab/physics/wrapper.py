"""
Physics-Informed Wrapper for Deep Learning Models.

Wraps any base model with physics-informed loss and curriculum learning.
Supports gradient balancing and collocation points.
"""

import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .biogeochemical import EnhancedBiogeochemicalPhysics
from .loss import EnhancedPhysicsInformedLoss


class EnhancedPhysicsInformedWrapper(nn.Module):
    """
    Physics-Informed Wrapper.
    
    Wraps any base model with enhanced physics-informed loss.
    Supports curriculum learning, gradient balancing, and collocation points.
    """
    
    def __init__(
        self, 
        base_model: nn.Module,
        physics_module: Optional[EnhancedBiogeochemicalPhysics] = None,
        w_physics_min: float = 0.01,
        w_physics_max: float = 0.3,
        curriculum_epochs: int = 70,
        warmup_epochs: int = 10,
        use_gradient_balancing: bool = True,
        use_collocation: bool = True,
        n_collocation_points: int = 20,
        feature_names: Optional[List[str]] = None,
        n_sites: int = 5,
        net_growth_threshold: float = -0.1,
        net_growth_penalty_weight: float = 1.0,
        # parameters for data-scarce settings
        n_train_samples: Optional[int] = None,
        use_adaptive_physics_weight: bool = True,
        adaptive_weighting_method: str = 'relobralo',
        base_n_samples: int = 10000
    ):
        """
        Initialize the wrapper.
        
        Args:
            base_model: Base DL model (GRU, iTransformer, etc.)
            physics_module: EnhancedBiogeochemicalPhysics instance
            w_physics_min: Minimum physics weight
            w_physics_max: Maximum physics weight
            curriculum_epochs: Epochs for curriculum learning
            warmup_epochs: Epochs before physics loss starts
            use_gradient_balancing: Enable gradient-based weight balancing
            use_collocation: Enable collocation points for ODE
            n_collocation_points: Number of collocation points per batch
            feature_names: Feature names for physics index detection
            n_sites: Number of sites for site-specific parameters
            net_growth_threshold: Threshold for net growth penalty (default: -0.1)
            net_growth_penalty_weight: Weight for net growth penalty loss
            n_train_samples: Number of training samples (for adaptive weighting)
            use_adaptive_physics_weight: Enable adaptive physics weight based on data size
            adaptive_weighting_method: Method for adaptive weighting ('relobralo' or 'ntk')
            base_n_samples: Reference sample size for normalization
        """
        super().__init__()
        
        self.base_model = base_model
        
        # Create physics module if not provided
        if physics_module is not None:
            self.physics = physics_module
        else:
            self.physics = EnhancedBiogeochemicalPhysics(
                feature_names=feature_names,
                n_sites=n_sites,
                use_denormalization=True,
                use_sequence_ode=True
            )
        
        # Create loss function with enhanced adaptive weighting
        self.loss_fn = EnhancedPhysicsInformedLoss(
            self.physics,
            w_physics_min=w_physics_min,
            w_physics_max=w_physics_max,
            curriculum_epochs=curriculum_epochs,
            warmup_epochs=warmup_epochs,
            use_gradient_balancing=use_gradient_balancing,
            use_collocation=use_collocation,
            n_collocation_points=n_collocation_points,
            net_growth_threshold=net_growth_threshold,
            net_growth_penalty_weight=net_growth_penalty_weight,
            n_train_samples=n_train_samples,
            use_adaptive_physics_weight=use_adaptive_physics_weight,
            adaptive_weighting_method=adaptive_weighting_method,
            base_n_samples=base_n_samples
        )
        
        self.model_type = type(base_model).__name__
        self.model_name = f"{self.model_type}-PINN-v3"
        self.current_epoch = 0
    
    def forward(
        self, 
        x: torch.Tensor, 
        station_ids: Optional[torch.Tensor] = None,
        mc_samples: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass (delegates to base model)."""
        return self.base_model(x, station_ids, mc_samples)
    
    def set_epoch(self, epoch: int):
        """Set current epoch for curriculum learning."""
        self.current_epoch = epoch
    
    def compute_loss(
        self, 
        x: torch.Tensor, 
        y_true: torch.Tensor,
        station_ids: Optional[torch.Tensor] = None,
        dates: Optional[torch.Tensor] = None,
        use_grad_balance: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute physics-informed loss with curriculum learning.
        
        Args:
            x: Input features (batch, seq_len, n_features)
            y_true: True chlorophyll-a (batch, 1)
            station_ids: Station IDs (batch,)
            dates: Day-of-year values (batch,)
            use_grad_balance: Whether to use gradient balancing
        
        Returns:
            Dict: All loss components
        """
        y_pred, var = self.forward(x, station_ids)
        
        # Get model parameters for gradient balancing
        model_params = None
        if use_grad_balance and self.loss_fn.use_gradient_balancing:
            model_params = list(self.base_model.parameters())
        
        losses = self.loss_fn(
            y_pred, y_true, x, var=var,
            station_ids=station_ids, dates=dates,
            epoch=self.current_epoch,
            model_params=model_params
        )
        
        return losses
    
    def get_physics_parameters(self, station_id: Optional[int] = None) -> Dict:
        """Get learned physics parameters."""
        return self.physics.get_parameters_dict(station_id)
    
    def get_all_site_parameters(self) -> pd.DataFrame:
        """Get physics parameters for all sites."""
        return self.physics.get_all_site_parameters()
    
    def clamp_physics_parameters(self):
        """Clamp physics parameters to valid ranges."""
        self.physics.clamp_parameters()
    
    def set_denorm_params_from_scaler(self, scaler, feature_names: List[str]):
        """Set denormalization parameters from a fitted feature scaler."""
        self.physics.set_denorm_params_from_scaler(scaler, feature_names)
    
    def get_physics_loss_breakdown(self) -> Dict[str, float]:
        """Get breakdown of physics loss components."""
        return {
            'curriculum_weight': self.loss_fn.get_physics_weight(self.current_epoch),
            'grad_balance': self.loss_fn.grad_data_ema.item() / (self.loss_fn.grad_physics_ema.item() + 1e-8),
            'warmup_epochs': self.loss_fn.warmup_epochs,
            'curriculum_epochs': self.loss_fn.curriculum_epochs,
        }
    
    def get_training_info(self) -> str:
        """Get human-readable training info."""
        w = self.loss_fn.get_physics_weight(self.current_epoch)
        info = f"Epoch {self.current_epoch}: w_physics={w:.4f}"
        
        if self.loss_fn.use_gradient_balancing:
            gb = self.loss_fn.grad_data_ema.item() / (self.loss_fn.grad_physics_ema.item() + 1e-8)
            info += f", grad_balance={gb:.2f}"
        
        return info


# Alias for backward compatibility
PhysicsInformedWrapper = EnhancedPhysicsInformedWrapper
