"""
PyTorch Dataset classes for HAB prediction.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple, Dict


class HABDataset(Dataset):
    """
    PyTorch Dataset for HAB (Harmful Algal Bloom) prediction.
    
    Wraps numpy arrays into a PyTorch-compatible dataset with
    support for station IDs (for panel data).
    """
    
    def __init__(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        station_ids: Optional[np.ndarray] = None,
        dates: Optional[np.ndarray] = None
    ):
        """
        Initialize dataset.
        
        Args:
            X: Input sequences (n_samples, seq_len, n_features)
            y: Target values (n_samples, 1)
            station_ids: Station IDs for each sample (n_samples,)
            dates: Date for each sample (n_samples,)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.station_ids = (
            torch.tensor(station_ids, dtype=torch.long) 
            if station_ids is not None else None
        )
        self.dates = dates
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with 'X', 'y', and optionally 'station_id'
        """
        item = {
            'X': self.X[idx],
            'y': self.y[idx]
        }
        
        if self.station_ids is not None:
            item['station_id'] = self.station_ids[idx]
        
        return item


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_station_ids: Optional[np.ndarray] = None,
    val_station_ids: Optional[np.ndarray] = None,
    test_station_ids: Optional[np.ndarray] = None,
    test_dates: Optional[np.ndarray] = None,
    batch_size: int = 32,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader, DataLoader, HABDataset]:
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        train_station_ids: Training station IDs
        val_station_ids: Validation station IDs
        test_station_ids: Test station IDs
        test_dates: Test dates
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes (default 2 for Apple Silicon)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, test_dataset)
    """
    import torch
    # pin_memory only useful for CUDA (DMA transfer); no-op/warning on MPS
    use_pin_memory = torch.cuda.is_available()
    
    train_dataset = HABDataset(X_train, y_train, train_station_ids)
    val_dataset = HABDataset(X_val, y_val, val_station_ids)
    test_dataset = HABDataset(X_test, y_test, test_station_ids, test_dates)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    return train_loader, val_loader, test_loader, test_dataset

