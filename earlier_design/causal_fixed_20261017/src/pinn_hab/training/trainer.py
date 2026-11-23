"""
Training loop for HAB prediction models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Callable

from .early_stopping import EarlyStopping
from .losses import gaussian_nll_loss
from ..utils.device import device, cleanup_memory


def build_optimizer(model, learning_rate: float, weight_decay: float):
    """AdamW without weight decay on the kinetic (physics) parameters. Rerun fix: decay on the raw values
    pulled the kinetic parameters towards the centre of their transforms regardless of the data."""
    phys, other = [], []
    for name, prm in model.named_parameters():
        (phys if 'physics' in name else other).append(prm)
    groups = [{'params': other, 'weight_decay': weight_decay}]
    if phys:
        groups.append({'params': phys, 'weight_decay': 0.0})
    return optim.AdamW(groups, lr=learning_rate)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    save_path: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
    verbose: bool = True
) -> Dict:
    """
    Train a model with the given configuration.
    
    For data-scarce settings:
    - Dynamic physics weighting based on sample size
    - Detailed epoch-by-epoch loss logging
    - Adaptive gradient balancing (ReLoBRaLo/NTK)
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        save_path: Path to save best model
        progress_callback: Optional callback for progress updates
        verbose: Whether to print detailed logs
    
    Returns:
        Dict with training history and best metrics
    """
    # Move model to device
    model = model.to(device)
    
    # Training config
    epochs = config.get('epochs', 100)
    learning_rate = config.get('learning_rate', 1e-3)
    weight_decay = config.get('weight_decay', 1e-4)
    patience = config.get('patience', 15)
    
    # Calculate training sample size for adaptive weighting
    n_train_samples = len(train_loader.dataset) if hasattr(train_loader.dataset, '__len__') else None
    if n_train_samples is None:
        # Fallback: estimate from dataloader
        n_train_samples = len(train_loader) * train_loader.batch_size
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training Configuration")
        print(f"{'='*60}")
        print(f"Training samples: {n_train_samples}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {train_loader.batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"{'='*60}\n")
    
    # Optimizer
    optimizer = build_optimizer(model, learning_rate, weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=patience,
        verbose=True,
        path=save_path or 'best_model.pth'
    )
    
    # Training history (enhanced for detailed logging)
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': [],
        'data_loss': [],
        'physics_loss': [],
        'w_physics': [],
        'grad_balance': []
    }
    
    # Check if model has physics loss
    has_physics = hasattr(model, 'compute_loss')
    
    # Setup adaptive physics weighting if using PINN
    if has_physics and hasattr(model, 'loss_fn'):
        loss_fn = model.loss_fn
        if hasattr(loss_fn, 'update_adaptive_physics_weight'):
            loss_fn.update_adaptive_physics_weight(n_train_samples)
            if verbose:
                print(f"[Adaptive Weighting] Adjusted w_physics_max: {loss_fn.w_physics_max:.4f} "
                      f"(base: {loss_fn.w_physics_max_base:.4f}, samples: {n_train_samples})")
    
    if verbose and has_physics:
        print(f"[Physics-Informed Training] Enabled")
        if hasattr(model, 'loss_fn'):
            loss_fn = model.loss_fn
            print(f"  - Warmup epochs: {loss_fn.warmup_epochs}")
            print(f"  - Curriculum epochs: {loss_fn.curriculum_epochs}")
            print(f"  - Physics weight range: [{loss_fn.w_physics_min:.4f}, {loss_fn.w_physics_max:.4f}]")
            print(f"  - Gradient balancing: {loss_fn.use_gradient_balancing} ({loss_fn.adaptive_weighting_method})")
        print()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        train_data_losses = []
        train_physics_losses = []
        train_w_physics = []
        train_grad_balances = []
        
        for batch in train_loader:
            X = batch['X'].to(device)
            y = batch['y'].to(device)
            station_ids = batch.get('station_id')
            if station_ids is not None:
                station_ids = station_ids.to(device)
            
            optimizer.zero_grad()
            
            if has_physics:
                # Physics-informed model
                model.set_epoch(epoch)
                losses = model.compute_loss(X, y, station_ids)
                loss = losses['total']
                
                # Collect detailed loss components
                train_data_losses.append(losses.get('data', loss).item())
                train_physics_losses.append(losses.get('physics', torch.tensor(0.0)).item())
                train_w_physics.append(losses.get('w_physics', torch.tensor(0.0)).item())
                train_grad_balances.append(losses.get('grad_balance', torch.tensor(1.0)).item())
            else:
                # Standard model
                mean, var = model(X, station_ids)
                loss = gaussian_nll_loss(y, mean, var)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if has_physics:
                model.clamp_physics_parameters()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_data_loss = np.mean(train_data_losses) if train_data_losses else avg_train_loss
        avg_physics_loss = np.mean(train_physics_losses) if train_physics_losses else 0.0
        avg_w_physics = np.mean(train_w_physics) if train_w_physics else 0.0
        avg_grad_balance = np.mean(train_grad_balances) if train_grad_balances else 1.0
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                X = batch['X'].to(device)
                y = batch['y'].to(device)
                station_ids = batch.get('station_id')
                if station_ids is not None:
                    station_ids = station_ids.to(device)
                
                mean, var = model(X, station_ids)
                loss = gaussian_nll_loss(y, mean, var)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        if has_physics:
            history['data_loss'].append(avg_data_loss)
            history['physics_loss'].append(avg_physics_loss)
            history['w_physics'].append(avg_w_physics)
            history['grad_balance'].append(avg_grad_balance)
        
        # Scheduler step
        scheduler.step(avg_val_loss)
        
        # Early stopping
        early_stopping(avg_val_loss, model)
        
        # Progress callback
        if progress_callback:
            progress = int((epoch + 1) / epochs * 100)
            progress_callback(progress, f"Epoch {epoch+1}/{epochs} - Val Loss: {avg_val_loss:.4f}")
        
        # logging: epoch-by-epoch for physics-informed models
        if verbose:
            if has_physics:
                # Detailed logging for PINN models
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train: {avg_train_loss:.4f} (Data: {avg_data_loss:.4f}, "
                      f"Physics: {avg_physics_loss:.4f}) | "
                      f"Val: {avg_val_loss:.4f} | "
                      f"w_phys: {avg_w_physics:.4f} | "
                      f"grad_bal: {avg_grad_balance:.3f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            else:
                # Standard logging for base models
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"Epoch {epoch+1:3d}/{epochs} | "
                          f"Train: {avg_train_loss:.4f} | "
                          f"Val: {avg_val_loss:.4f} | "
                          f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        else:
            # Minimal logging
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")
        
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Memory cleanup
        if (epoch + 1) % 20 == 0:
            cleanup_memory()
    
    # Load best model
    if save_path:
        model.load_state_dict(torch.load(save_path))
    
    return {
        'history': history,
        'best_val_loss': early_stopping.val_loss_min,
        'final_epoch': epoch + 1
    }
