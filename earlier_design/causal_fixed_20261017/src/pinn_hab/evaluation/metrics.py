"""
Evaluation metrics for HAB prediction.
"""

import numpy as np
from typing import Dict


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Error."""
    return np.sqrt(np.mean((y_true.flatten() - y_pred.flatten()) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true.flatten() - y_pred.flatten()))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination (R²)."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    return 1 - (ss_res / (ss_tot + 1e-8))


def nse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency."""
    return r2_score(y_true, y_pred)  # NSE is equivalent to R²


def kge(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Kling-Gupta Efficiency.
    
    KGE = 1 - sqrt((r-1)² + (α-1)² + (β-1)²)
    
    Where:
        r = correlation coefficient
        α = std(pred) / std(obs)
        β = mean(pred) / mean(obs)
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Correlation
    r = np.corrcoef(y_true, y_pred)[0, 1]
    if np.isnan(r):
        r = 0
    
    # Variability ratio
    alpha = np.std(y_pred) / (np.std(y_true) + 1e-8)
    
    # Bias ratio
    beta = np.mean(y_pred) / (np.mean(y_true) + 1e-8)
    
    # KGE
    kge_value = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    
    return kge_value


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        Dictionary with all metrics
    """
    return {
        'RMSE': rmse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
        'R²': r2_score(y_true, y_pred),
        'NSE': nse(y_true, y_pred),
        'KGE': kge(y_true, y_pred),
        'MAPE': mape(y_true, y_pred)
    }


def compute_uncertainty_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Compute uncertainty-related metrics.
    
    Args:
        y_true: Ground truth values
        lower: Lower bounds of prediction interval
        upper: Upper bounds of prediction interval
        alpha: Miscoverage level
    
    Returns:
        Dictionary with PICP, PIAW, NIS
    """
    y_true = y_true.flatten()
    lower = lower.flatten()
    upper = upper.flatten()
    
    # Prediction Interval Coverage Probability
    covered = (y_true >= lower) & (y_true <= upper)
    picp = covered.mean()
    
    # Prediction Interval Average Width
    piaw = (upper - lower).mean()
    
    # Normalized Interval Score
    width = upper - lower
    penalty_lower = (2 / alpha) * (lower - y_true) * (y_true < lower)
    penalty_upper = (2 / alpha) * (y_true - upper) * (y_true > upper)
    nis = (width + penalty_lower + penalty_upper).mean()
    
    return {
        'PICP': picp,
        'PIAW': piaw,
        'NIS': nis,
        'target_coverage': 1 - alpha
    }
