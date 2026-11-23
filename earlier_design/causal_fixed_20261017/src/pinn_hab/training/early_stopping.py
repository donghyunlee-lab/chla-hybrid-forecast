"""
Early stopping callback for training.
"""

import numpy as np
import torch


class EarlyStopping:
    """
    Early stops training if validation loss doesn't improve.
    
    Args:
        patience: Number of epochs to wait before stopping
        verbose: Whether to print messages
        delta: Minimum change to qualify as improvement
        path: Path to save best model
    """
    
    def __init__(
        self, 
        patience: int = 15, 
        verbose: bool = False, 
        delta: float = 0,
        path: str = 'checkpoint.pth',
        trace_func=print
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss: float, model: torch.nn.Module):
        """Check if training should stop."""
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module):
        """Save model when validation loss improves."""
        if self.verbose:
            self.trace_func(
                f'Val loss decreased ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving...'
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
