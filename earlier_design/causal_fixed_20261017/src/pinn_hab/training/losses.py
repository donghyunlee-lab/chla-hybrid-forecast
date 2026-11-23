"""
Loss functions for HAB prediction.
"""

import torch
import numpy as np


def gaussian_nll_loss(
    y_true: torch.Tensor,
    y_pred_mean: torch.Tensor,
    y_pred_var: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Gaussian Negative Log-Likelihood Loss for heteroscedastic uncertainty.
    
    NLL = 0.5 * log(2πσ²) + 0.5 * (y - μ)² / σ²
    
    Args:
        y_true: True values (batch, 1)
        y_pred_mean: Predicted mean (batch, 1)
        y_pred_var: Predicted variance (batch, 1)
        eps: Small constant for numerical stability
    
    Returns:
        Scalar loss
    """
    # Ensure variance is positive
    var = torch.clamp(y_pred_var, min=eps)
    
    # Gaussian NLL
    nll = 0.5 * torch.log(2 * np.pi * var) + \
          0.5 * (y_true - y_pred_mean) ** 2 / var
    
    return nll.mean()


def mse_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor
) -> torch.Tensor:
    """Mean Squared Error loss."""
    return torch.nn.functional.mse_loss(y_pred, y_true)


def rmse_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor
) -> torch.Tensor:
    """Root Mean Squared Error loss."""
    return torch.sqrt(mse_loss(y_true, y_pred))
