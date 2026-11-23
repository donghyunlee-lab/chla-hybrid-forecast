"""
Data Preprocessor for Multi-Site HAB (Harmful Algal Bloom) Prediction.

Supports panel data preprocessing with multiple sites and time series sequence generation.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import MinMaxScaler

from ..utils.korean_mapping import KOREAN_TO_ENGLISH_MAP


class DataPreprocessor:
    """
    Panel Data Preprocessor for Multi-Site HAB Prediction.
    
    Features:
        1. Multi-site data integration (panel data support)
        2. Station_ID tracking for site-specific modeling
        3. Site-aware sequence generation (prevents temporal leakage)
        4. Log transformation for high-variance features
        5. Feature engineering (diff, moving average)
    """

    def __init__(
        self, 
        data_path: str, 
        site_names: Optional[List[str]] = None, 
        apply_log_transform: bool = True, 
        use_panel_data: bool = True,
        sampling_rate: int = 1
    ):
        """
        Initialize the preprocessor.
        
        Args:
            data_path: Path to CSV data file
            site_names: List of site names to include (default: 5 Korean sites)
            apply_log_transform: Whether to apply log transformation
            use_panel_data: Whether to use panel data mode (multiple sites)
            sampling_rate: Subsampling rate for training data (1 = no subsampling)
        """
        self.data_path = data_path
        
        # Default Korean site names
        if site_names is None:
            site_names = ['공주', '대청호', '갑천', '부여', '용담호']
        self.site_names = site_names
        
        self.use_panel_data = use_panel_data
        self.sampling_rate = sampling_rate
        self.scaler = MinMaxScaler()
        self.feature_names: Optional[List[str]] = None
        self.apply_log_transform = apply_log_transform
        self.log_transform_cols: List[str] = []
        self.target_idx: Optional[int] = None
        self.station_id_map = {site: idx for idx, site in enumerate(site_names)}
        
        print(f"Panel Data Mode - {len(site_names)} sites")
        print(f"  Station ID mapping: {self.station_id_map}")
        if sampling_rate > 1:
            print(f"  [WARN] Sampling Rate: {sampling_rate} (Training data will be subsampled)")

    def load_data(self) -> pd.DataFrame:
        """
        Load and prepare data from CSV file.
        
        Returns:
            DataFrame with Station_ID column for panel data mode
        """
        df = pd.read_csv(self.data_path)
        
        # Translate Korean column names to English
        df = df.rename(columns=KOREAN_TO_ENGLISH_MAP)
        df['Date'] = pd.to_datetime(df['Date'])

        if self.use_panel_data:
            panel_dfs = []
            for site_name in self.site_names:
                site_df = df[df['Site'] == site_name].copy()
                if len(site_df) > 0:
                    site_df['Station_ID'] = self.station_id_map[site_name]
                    site_df = site_df.sort_values('Date').reset_index(drop=True)
                    panel_dfs.append(site_df)
                    print(f"  {site_name} (ID={self.station_id_map[site_name]}): {len(site_df)} records")
            
            if len(panel_dfs) == 0:
                raise ValueError(f"No data found for sites: {self.site_names}")
            
            # Stack all sites
            panel_df = pd.concat(panel_dfs, ignore_index=True)
            panel_df = panel_df.sort_values(['Station_ID', 'Date']).reset_index(drop=True)
            
            print(f"\nPanel Data Summary:")
            print(f"  Total records: {len(panel_df)}")
            print(f"  Date range: {panel_df['Date'].min()} to {panel_df['Date'].max()}")
            
            return panel_df
        else:
            # Single site mode
            site_name = self.site_names[0] if self.site_names else '공주'
            site_df = df[df['Site'] == site_name].copy()
            site_df = site_df.sort_values('Date').reset_index(drop=True)
            print(f"Loaded {len(site_df)} records for site: {site_name}")
            return site_df
    
    def create_missing_summer_data(
        self, 
        df: pd.DataFrame, 
        remove_months: List[int] = [7, 8]
    ) -> pd.DataFrame:
        """
        Create dataset with summer months removed for extrapolation testing.
        
        Args:
            df: Full dataframe with 'Date' column
            remove_months: List of months to remove (default: July-August)
        
        Returns:
            Filtered dataframe
        """
        df_copy = df.copy()
        
        train_start = pd.to_datetime('2013-01-01')
        train_end = pd.to_datetime('2022-12-31')
        
        train_mask = (df_copy['Date'] >= train_start) & (df_copy['Date'] <= train_end)
        summer_mask = df_copy['Date'].dt.month.isin(remove_months)
        
        remove_mask = train_mask & summer_mask
        df_filtered = df_copy[~remove_mask].reset_index(drop=True)
        
        removed_count = remove_mask.sum()
        print(f"\nMissing Summer Scenario:")
        print(f"  Removed {removed_count} samples from months {remove_months}")
        print(f"  Remaining: {len(df_filtered)} samples ({100*(1-removed_count/len(df)):.1f}%)")
        
        return df_filtered

    def preprocess(self, df: pd.DataFrame, fit_period: Optional[Tuple[str, str]] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Preprocess data with normalization and feature engineering.
        
        Args:
            df: Raw dataframe
            fit_period: (start, end) dates whose rows choose the log transforms and fit the scaler.
                Rerun fix: the training period, so no statistics of later years are used. None = all rows.
            
        Returns:
            Tuple of (normalized dataframe, feature names list)
        """
        if fit_period is not None:
            fit_mask = ((df['Date'] >= pd.to_datetime(fit_period[0])) & (df['Date'] <= pd.to_datetime(fit_period[1]))).values
        else:
            fit_mask = np.ones(len(df), dtype=bool)
        # Exclude metadata columns
        exclude_cols = ['Basin', 'Site', 'Date', 'Start Year', 'End Year', 'Filename', 'Site Name']
        if self.use_panel_data:
            exclude_cols.append('Station_ID')
        
        feature_cols = [
            col for col in df.columns 
            if col not in exclude_cols and df[col].dtype in [np.float64, np.float32, np.int64]
        ]

        # Exclude ECD measurement variables
        ecd_vars = [col for col in feature_cols if '[ECD]' in col]
        if ecd_vars:
            print(f"  Excluding ECD variables ({len(ecd_vars)}): {ecd_vars}")
            feature_cols = [col for col in feature_cols if '[ECD]' not in col]

        # Sensitivity run without the volatile organic compounds (PINN_DROP_VOC=1, rerun addition)
        if os.environ.get('PINN_DROP_VOC', '0') == '1':
            voc = [col for col in feature_cols if 'μg/L' in col]
            print(f"  Excluding VOC variables ({len(voc)}): {voc}")
            feature_cols = [col for col in feature_cols if 'μg/L' not in col]

        # Find target variable (Chlorophyll-a)
        target_col = None
        for col in feature_cols:
            if 'Chl-a' in col or 'Chlorophyll' in col or 'Chl' in col:
                target_col = col
                break
        
        if target_col:
            self.target_idx = feature_cols.index(target_col)
            print(f"Target variable: {target_col} (index: {self.target_idx})")

        df_numeric = df[feature_cols].copy()
        
        # Log transformation for high-variance features
        if self.apply_log_transform:
            high_var_cols = []
            for col in feature_cols:
                fit_col = df_numeric.loc[fit_mask, col]
                if fit_col.std() > 0:
                    cv = fit_col.std() / (fit_col.mean() + 1e-6)
                    if cv > 1.0 or col == target_col:
                        high_var_cols.append(col)
            
            for col in high_var_cols:
                if df_numeric[col].min() >= 0:
                    df_numeric[col] = np.log1p(df_numeric[col])
                    self.log_transform_cols.append(col)
                    print(f"  Applied log1p to: {col}")
        
        # Feature Engineering
        if target_col and target_col in df_numeric.columns:
            df_numeric[f'{target_col}_diff1'] = df_numeric[target_col].diff(1).fillna(0)
            df_numeric[f'{target_col}_ma7'] = df_numeric[target_col].rolling(
                window=7, min_periods=1
            ).mean().fillna(df_numeric[target_col])
            
            feature_cols.extend([f'{target_col}_diff1', f'{target_col}_ma7'])
            print(f"  Added: {target_col}_diff1, {target_col}_ma7")
        
        self.feature_names = feature_cols

        # Normalize features
        df_normalized = pd.DataFrame(
            self.scaler.fit(df_numeric[fit_mask]).transform(df_numeric),
            columns=feature_cols,
            index=df.index
        )

        print(f"Using {len(feature_cols)} features for modeling")
        return df_normalized, feature_cols
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized data back to original scale.
        
        Args:
            data: Normalized data (n_samples, n_features)
            
        Returns:
            Original scale data
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        data_denorm = self.scaler.inverse_transform(data)
        
        if self.apply_log_transform and self.log_transform_cols:
            for col in self.log_transform_cols:
                if col in self.feature_names:
                    col_idx = self.feature_names.index(col)
                    data_denorm[:, col_idx] = np.maximum(0, np.expm1(data_denorm[:, col_idx]))
        
        return data_denorm

    def create_sequences(
        self, 
        data: np.ndarray, 
        station_ids: Optional[np.ndarray] = None,
        seq_len: int = 30, 
        dates: Optional[np.ndarray] = None,
        horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Create time series sequences for model training.
        
        Args:
            data: Normalized feature matrix (N, n_features)
            station_ids: Station ID for each sample (N,)
            seq_len: Sequence length
            dates: Date array for alignment
        
        Returns:
            Tuple of (X, y, station_ids_seq, dates_seq)
        """
        X, y, station_ids_seq, dates_seq = [], [], [], []
        
        if station_ids is None:
            # Single site mode
            for i in range(len(data) - seq_len - horizon + 1):
                X.append(data[i:i+seq_len])
                y.append(data[i+seq_len+horizon-1, self.target_idx] if self.target_idx else data[i+seq_len+horizon-1, 0])
                if dates is not None:
                    dates_seq.append(dates[i+seq_len+horizon-1])
            
            return (
                np.array(X), 
                np.array(y).reshape(-1, 1), 
                None, 
                np.array(dates_seq) if dates is not None else None
            )
        
        # Panel data mode - generate sequences within each site
        current_station = station_ids[0]
        start_idx = 0
        
        for i in range(len(data)):
            if station_ids[i] != current_station:
                # Process previous site
                site_data = data[start_idx:i]
                site_station = current_station
                site_dates = dates[start_idx:i] if dates is not None else None
                
                for j in range(len(site_data) - seq_len - horizon + 1):
                    X.append(site_data[j:j+seq_len])
                    target_val = site_data[j+seq_len+horizon-1, self.target_idx] if self.target_idx else site_data[j+seq_len+horizon-1, 0]
                    y.append(target_val)
                    station_ids_seq.append(site_station)
                    if site_dates is not None:
                        dates_seq.append(site_dates[j+seq_len+horizon-1])
                
                current_station = station_ids[i]
                start_idx = i
        
        # Handle last site
        site_data = data[start_idx:]
        site_station = current_station
        site_dates = dates[start_idx:] if dates is not None else None
        
        for j in range(len(site_data) - seq_len - horizon + 1):
            X.append(site_data[j:j+seq_len])
            target_val = site_data[j+seq_len+horizon-1, self.target_idx] if self.target_idx else site_data[j+seq_len+horizon-1, 0]
            y.append(target_val)
            station_ids_seq.append(site_station)
            if site_dates is not None:
                dates_seq.append(site_dates[j+seq_len+horizon-1])
        
        return (
            np.array(X), 
            np.array(y).reshape(-1, 1), 
            np.array(station_ids_seq),
            np.array(dates_seq) if dates is not None else None
        )

    def split_train_test(
        self, 
        df: pd.DataFrame, 
        df_normalized: pd.DataFrame,
        seq_len: int = 30,
        horizon: int = 1,
        train_period: Tuple[str, str] = ('2013-01-01', '2022-12-31'),
        val_period: Tuple[str, str] = ('2023-01-01', '2023-12-31'),
        test_period: Tuple[str, str] = ('2024-01-01', '2024-12-31'),
        evaluation_focus: str = 'all'
    ) -> Dict:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Original dataframe
            df_normalized: Normalized dataframe
            seq_len: Sequence length
            train_period: Training date range
            val_period: Validation date range
            test_period: Test date range
            evaluation_focus: Filtering mode ('all', 'growth_phase', 'unseen_station', 'sparse_calibration')
        
        Returns:
            Dictionary with train/val/test data and metadata
        """
        train_start, train_end = pd.to_datetime(train_period[0]), pd.to_datetime(train_period[1])
        val_start, val_end = pd.to_datetime(val_period[0]), pd.to_datetime(val_period[1])
        test_start, test_end = pd.to_datetime(test_period[0]), pd.to_datetime(test_period[1])

        train_mask = (df['Date'] >= train_start) & (df['Date'] <= train_end)
        val_mask = (df['Date'] >= val_start) & (df['Date'] <= val_end)
        test_mask = (df['Date'] >= test_start) & (df['Date'] <= test_end)

        train_data = df_normalized[train_mask].values
        val_data = df_normalized[val_mask].values
        test_data = df_normalized[test_mask].values

        train_station_ids_seq = None
        val_station_ids_seq = None
        test_station_ids_seq = None

        if self.use_panel_data and 'Station_ID' in df.columns:
            train_station_ids = df[train_mask]['Station_ID'].values
            val_station_ids = df[val_mask]['Station_ID'].values
            test_station_ids = df[test_mask]['Station_ID'].values
            
            train_dates_raw = df[train_mask]['Date'].values
            val_dates_raw = df[val_mask]['Date'].values
            test_dates_raw = df[test_mask]['Date'].values

            X_train, y_train, train_station_ids_seq, _ = self.create_sequences(
                train_data, train_station_ids, seq_len, dates=train_dates_raw, horizon=horizon
            )
            X_val, y_val, val_station_ids_seq, _ = self.create_sequences(
                val_data, val_station_ids, seq_len, dates=val_dates_raw, horizon=horizon
            )
            X_test, y_test, test_station_ids_seq, test_dates_seq = self.create_sequences(
                test_data, test_station_ids, seq_len, dates=test_dates_raw, horizon=horizon
            )
            
            # Subsampling for low-frequency scenario
            if self.sampling_rate > 1:
                print(f"\n[Low-Frequency] Subsampling training data by rate {self.sampling_rate}...")
                X_train = X_train[::self.sampling_rate]
                y_train = y_train[::self.sampling_rate]
                if train_station_ids_seq is not None:
                    train_station_ids_seq = train_station_ids_seq[::self.sampling_rate]
                print(f"  Training samples: {X_train.shape[0]}")

            print(f"\nData Split Summary:")
            print(f"  Training: {X_train.shape[0]} sequences ({train_period[0]} ~ {train_period[1]})")
            print(f"  Validation: {X_val.shape[0]} sequences ({val_period[0]} ~ {val_period[1]})")
            print(f"  Test: {X_test.shape[0]} sequences ({test_period[0]} ~ {test_period[1]})")
        else:
            test_dates_raw = df[test_mask]['Date'].values
            X_train, y_train, _, _ = self.create_sequences(train_data, None, seq_len, horizon=horizon)
            X_val, y_val, _, _ = self.create_sequences(val_data, None, seq_len, horizon=horizon)
            X_test, y_test, _, test_dates_seq = self.create_sequences(
                test_data, None, seq_len, dates=test_dates_raw, horizon=horizon
            )
            
            if self.sampling_rate > 1:
                X_train = X_train[::self.sampling_rate]
                y_train = y_train[::self.sampling_rate]

            print(f"Training: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")

        test_dates = test_dates_seq

        # Apply evaluation focus filters
        if evaluation_focus == 'growth_phase':
            X_test, y_test, test_dates, test_station_ids_seq = self._filter_growth_phase(
                X_test, y_test, test_dates, test_station_ids_seq
            )
        elif evaluation_focus == 'unseen_station':
            (X_train, y_train, train_station_ids_seq, 
             X_val, y_val, val_station_ids_seq,
             X_test, y_test, test_dates, test_station_ids_seq) = self._filter_unseen_station(
                X_train, y_train, train_station_ids_seq,
                X_val, y_val, val_station_ids_seq,
                X_test, y_test, test_dates, test_station_ids_seq
            )
        elif evaluation_focus == 'sparse_calibration':
            X_train, y_train, train_station_ids_seq = self._filter_sparse_calibration(
                X_train, y_train, train_station_ids_seq
            )

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'train_station_ids': train_station_ids_seq,
            'val_station_ids': val_station_ids_seq,
            'test_station_ids': test_station_ids_seq,
            'test_dates': test_dates,
            'original_df': df,
            'df_normalized': df_normalized
        }
    
    def _filter_growth_phase(self, X_test, y_test, test_dates, test_station_ids_seq):
        """Filter test data to growth phase (May-Sept, rising trend)."""
        print(f"\n[Early-Warning] Filtering for Growth Phase (May-Sept, Rising trend)...")
        
        filter_indices = []
        for i, date_np in enumerate(test_dates):
            date = pd.to_datetime(date_np)
            if 5 <= date.month <= 9:
                current_val = y_test[i].item()
                prev_val = X_test[i, -1, self.target_idx].item()
                if current_val > prev_val:
                    filter_indices.append(i)
        
        if filter_indices:
            indices = np.array(filter_indices)
            X_test = X_test[indices]
            y_test = y_test[indices]
            test_dates = test_dates[indices]
            if test_station_ids_seq is not None:
                test_station_ids_seq = test_station_ids_seq[indices]
            print(f"  Test samples: {len(indices)} (Growth Phase only)")
        else:
            print("  [WARN] No samples match criteria, using full test set")
        
        return X_test, y_test, test_dates, test_station_ids_seq
    
    def _filter_unseen_station(
        self, X_train, y_train, train_station_ids,
        X_val, y_val, val_station_ids,
        X_test, y_test, test_dates, test_station_ids
    ):
        """Filter for unseen station (zero-shot) evaluation."""
        target_site_id = 4
        print(f"\n[Unseen Station] Holding out Site {target_site_id}...")
        
        if train_station_ids is not None:
            mask_train = train_station_ids != target_site_id
            X_train = X_train[mask_train]
            y_train = y_train[mask_train]
            train_station_ids = train_station_ids[mask_train]
            print(f"  Training (excluding site {target_site_id}): {len(X_train)}")
        
        if val_station_ids is not None:
            mask_val = val_station_ids != target_site_id
            X_val = X_val[mask_val]
            y_val = y_val[mask_val]
            val_station_ids = val_station_ids[mask_val]
            print(f"  Validation (excluding site {target_site_id}): {len(X_val)}")
        
        if test_station_ids is not None:
            mask_test = test_station_ids == target_site_id
            X_test = X_test[mask_test]
            y_test = y_test[mask_test]
            test_dates = test_dates[mask_test]
            test_station_ids = test_station_ids[mask_test]
            print(f"  Test (only site {target_site_id}): {len(X_test)}")
        
        return (X_train, y_train, train_station_ids,
                X_val, y_val, val_station_ids,
                X_test, y_test, test_dates, test_station_ids)
    
    def _filter_sparse_calibration(self, X_train, y_train, train_station_ids):
        """Filter for sparse calibration scenario."""
        target_site_id = 4
        print(f"\n[Sparse Calibration] Hybrid training data...")
        print("  - Source Sites (0~3): Daily Data")
        print("  - Target Site (4): Monthly Data")
        
        keep_mask = []
        site_4_count = 0
        site_4_kept = 0
        
        for i, station_id in enumerate(train_station_ids):
            if station_id != target_site_id:
                keep_mask.append(True)
            else:
                site_4_count += 1
                if (site_4_count - 1) % 30 == 0:
                    keep_mask.append(True)
                    site_4_kept += 1
                else:
                    keep_mask.append(False)
        
        keep_mask = np.array(keep_mask)
        X_train = X_train[keep_mask]
        y_train = y_train[keep_mask]
        if train_station_ids is not None:
            train_station_ids = train_station_ids[keep_mask]
        
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Target Site calibration: {site_4_kept} (from {site_4_count})")
        
        return X_train, y_train, train_station_ids
