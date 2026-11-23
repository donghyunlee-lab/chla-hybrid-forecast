"""
Data transformation utilities for time series preprocessing.
"""

import numpy as np
from typing import Optional, Tuple


class SequenceTransform:
    """
    Transform time series data into sequences for model input.
    
    This class provides utilities for creating sliding window sequences
    from time series data.
    """
    
    def __init__(self, seq_len: int = 30, target_idx: int = 0):
        """
        Initialize sequence transform.
        
        Args:
            seq_len: Length of input sequences
            target_idx: Index of target variable in feature array
        """
        self.seq_len = seq_len
        self.target_idx = target_idx
    
    def create_sequences(
        self, 
        data: np.ndarray,
        station_ids: Optional[np.ndarray] = None,
        dates: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Create sliding window sequences from time series data.
        
        Args:
            data: Feature matrix (n_samples, n_features)
            station_ids: Station ID for each sample
            dates: Date for each sample
        
        Returns:
            Tuple of (X, y, station_ids_seq, dates_seq)
        """
        X, y, station_ids_seq, dates_seq = [], [], [], []
        
        if station_ids is None:
            # Single site mode
            for i in range(len(data) - self.seq_len):
                X.append(data[i:i+self.seq_len])
                y.append(data[i+self.seq_len, self.target_idx])
                if dates is not None:
                    dates_seq.append(dates[i+self.seq_len])
            
            return (
                np.array(X), 
                np.array(y).reshape(-1, 1), 
                None, 
                np.array(dates_seq) if dates is not None else None
            )
        
        # Panel data mode
        current_station = station_ids[0]
        start_idx = 0
        
        for i in range(len(data)):
            if station_ids[i] != current_station:
                self._process_site(
                    data[start_idx:i], current_station,
                    dates[start_idx:i] if dates is not None else None,
                    X, y, station_ids_seq, dates_seq
                )
                current_station = station_ids[i]
                start_idx = i
        
        # Process last site
        self._process_site(
            data[start_idx:], current_station,
            dates[start_idx:] if dates is not None else None,
            X, y, station_ids_seq, dates_seq
        )
        
        return (
            np.array(X), 
            np.array(y).reshape(-1, 1), 
            np.array(station_ids_seq),
            np.array(dates_seq) if dates is not None else None
        )
    
    def _process_site(
        self, 
        site_data: np.ndarray, 
        station_id: int,
        site_dates: Optional[np.ndarray],
        X: list, y: list, 
        station_ids_seq: list, 
        dates_seq: list
    ):
        """Process a single site's data."""
        for j in range(len(site_data) - self.seq_len):
            X.append(site_data[j:j+self.seq_len])
            y.append(site_data[j+self.seq_len, self.target_idx])
            station_ids_seq.append(station_id)
            if site_dates is not None:
                dates_seq.append(site_dates[j+self.seq_len])


class WindowAugmentation:
    """
    Data augmentation for time series through window manipulation.
    """
    
    @staticmethod
    def add_noise(data: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """Add Gaussian noise to data."""
        noise = np.random.normal(0, noise_level, data.shape)
        return data + noise
    
    @staticmethod
    def time_mask(
        data: np.ndarray, 
        mask_ratio: float = 0.1
    ) -> np.ndarray:
        """Randomly mask time steps."""
        data_aug = data.copy()
        n_mask = int(data.shape[1] * mask_ratio)
        
        for i in range(len(data)):
            mask_idx = np.random.choice(data.shape[1], n_mask, replace=False)
            data_aug[i, mask_idx, :] = 0
        
        return data_aug
