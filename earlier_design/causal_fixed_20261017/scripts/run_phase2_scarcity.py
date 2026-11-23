#!/usr/bin/env python
"""
Phase 2: Data Scarcity & Transfer Learning.

Two steps:
  Step 1: Degradation Curve - Reduce training density (100% → 3%) and compare
          Pure DL vs PINN at each level.
  Step 2: Cross-Site Transfer Learning - Pre-train on source sites, fine-tune
          on held-out target site with Linear Probe vs Full Tune.

Prerequisites:
  - Phase 1 completed (best backbone selected in experiments/phase1/best_backbone.txt)

Usage:
    python scripts/run_phase2_scarcity.py
    python scripts/run_phase2_scarcity.py --step 1                   # Step 1 only
    python scripts/run_phase2_scarcity.py --step 2                   # Step 2 only
    python scripts/run_phase2_scarcity.py --backbone Mamba           # Override backbone
    python scripts/run_phase2_scarcity.py --epochs 10 --seeds 42     # Quick test
"""

import argparse
import json
import os
import sys
import random
import time
import copy
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pinn_hab.data import DataPreprocessor, create_dataloaders
from pinn_hab.models import get_model
from pinn_hab.training import train_model
from pinn_hab.training.transfer import (
    TransferLearningManager,
    TransferLearningConfig,
    FreezingStrategy,
    SamplingStrategy,
)
from pinn_hab.evaluation import compute_metrics
from pinn_hab.utils.device import device


# ---- Configuration ----

DEFAULT_SEEDS = [42, 123, 456]
DATA_DENSITIES = [1.0, 0.5, 0.25, 0.10, 0.03]  # 100% → 3%

DATA_CONFIG = {
    'path': 'data/imputed_daily_data.csv',
    'site_names': ['공주', '대청호', '갑천', '부여', '용담호'],
    'apply_log_transform': True,
    'use_panel_data': True,
}

MODEL_DEFAULTS = {
    'seq_len': 30,
    'hidden_dim': 64,
    'n_heads': 4,
    'n_layers': 2,
    'dropout_rate': 0.3,
    'use_station_embedding': True,
}

TRAIN_CONFIG = {
    'epochs': 100,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'patience': 15,
    'batch_size': 32,
}

PHYSICS_CONFIG = {
    'w_physics_min': 0.01,
    'w_physics_max': 0.3,
    'curriculum_epochs': 0,      # rerun fix: full physics weight from the first epoch;
    'warmup_epochs': 0,          # early stopping ended the original runs before the 10 + 70 epoch ramp
    'use_gradient_balancing': True,
    'use_collocation': True,
    'n_collocation_points': 20,
    'use_adaptive_physics_weight': True,
    'adaptive_weighting_method': 'relobralo',
}

SPLIT_CONFIG = {
    'train_period': ('2020-01-01', '2022-12-31'),
    'val_period': ('2023-01-01', '2023-12-31'),
    'test_period': ('2024-01-01', '2024-12-31'),
}

# Transfer learning config
SOURCE_SITES = [0, 1, 2, 3]   # Gongju, Daecheong, Gapcheon, Buyeo
TARGET_SITE = 4                # Yongdam

TRANSFER_FINETUNE_EPOCHS = 50
TRANSFER_FINETUNE_LR = 1e-4


# ---- Utilities ----

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_backbone(args_backbone: str, phase1_dir: Path) -> str:
    """Determine backbone from args or Phase 1 selection."""
    if args_backbone:
        return args_backbone
    
    selection_path = phase1_dir / 'best_backbone.txt'
    if selection_path.exists():
        backbone = selection_path.read_text().strip()
        print(f"Using Phase 1 selected backbone: {backbone}")
        return backbone
    
    default = 'iTransformer'
    print(f"[WARNING] No Phase 1 selection found. Using default: {default}")
    return default


def load_data():
    """Load and preprocess data (called once)."""
    preprocessor = DataPreprocessor(
        data_path=DATA_CONFIG['path'],
        site_names=DATA_CONFIG['site_names'],
        apply_log_transform=DATA_CONFIG['apply_log_transform'],
        use_panel_data=DATA_CONFIG['use_panel_data'],
    )
    
    df = preprocessor.load_data()
    # Rerun fixes: log transforms and scaler from the training period only; physical units for the
    # kinetics from that scaler.
    df_normalized, feature_names = preprocessor.preprocess(df, fit_period=SPLIT_CONFIG['train_period'])
    from pinn_hab.physics.biogeochemical import set_global_scaling
    set_global_scaling(preprocessor.scaler, feature_names, preprocessor.log_transform_cols)
    
    data = preprocessor.split_train_test(
        df, df_normalized,
        seq_len=MODEL_DEFAULTS['seq_len'],
        **SPLIT_CONFIG,
    )
    
    return data, feature_names, preprocessor


def subsample_data(data: dict, density: float, seed: int) -> dict:
    """
    Subsample training data to a given density (ratio).
    
    Stratified by station_id to preserve site balance.
    Validation and test sets remain unchanged.
    """
    if density >= 1.0:
        return data
    
    rng = np.random.RandomState(seed)
    
    X_train = data['X_train']
    y_train = data['y_train']
    station_ids = data['train_station_ids']
    
    n_total = len(X_train)
    n_target = max(1, int(n_total * density))
    
    # Stratified sampling by station
    if station_ids is not None:
        unique_stations = np.unique(station_ids)
        selected_indices = []
        
        for sid in unique_stations:
            mask = station_ids == sid
            site_indices = np.where(mask)[0]
            n_site = max(1, int(len(site_indices) * density))
            chosen = rng.choice(site_indices, size=n_site, replace=False)
            selected_indices.extend(chosen)
        
        selected_indices = np.sort(selected_indices)
    else:
        selected_indices = rng.choice(n_total, size=n_target, replace=False)
        selected_indices = np.sort(selected_indices)
    
    subsampled = dict(data)
    subsampled['X_train'] = X_train[selected_indices]
    subsampled['y_train'] = y_train[selected_indices]
    if station_ids is not None:
        subsampled['train_station_ids'] = station_ids[selected_indices]
    
    return subsampled


def create_loaders(data, batch_size=32):
    """Create DataLoaders from split data."""
    return create_dataloaders(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        data['X_test'], data['y_test'],
        data.get('train_station_ids'),
        data.get('val_station_ids'),
        data.get('test_station_ids'),
        batch_size=batch_size,
    )


def evaluate_model(model, test_loader):
    """Evaluate model on test set."""
    model.eval()
    model.to(device)
    
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            X = batch['X'].to(device)
            y = batch['y']
            station_ids = batch.get('station_id')
            if station_ids is not None:
                station_ids = station_ids.to(device)
            
            mean, var = model(X, station_ids)
            all_preds.append(mean.cpu().numpy())
            all_targets.append(y.numpy())
    
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    
    return compute_metrics(y_true, y_pred)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ---- Step 1: Degradation Curve ----

def run_step1(data, feature_names, backbone, seeds, epochs, output_dir):
    """
    Step 1: Degradation Curve.
    
    For each density level (100% → 3%):
      - Train Pure DL model
      - Train PINN model
      - Compare R², RMSE, NSE, KGE
    
    Key insight: PINN should degrade more gracefully than Pure DL
    as data becomes scarcer.
    
    Output: phase2_step1_degradation.csv
    """
    print_header(f"STEP 1: Degradation Curve ({backbone})")
    print(f"Data densities: {[f'{d:.0%}' for d in DATA_DENSITIES]}")
    
    results = []
    
    for density in DATA_DENSITIES:
        for model_variant in ['pure', 'pinn']:
            for seed in seeds:
                set_seed(seed)
                
                model_type = backbone if model_variant == 'pure' else f'{backbone}_PINN'
                n_train_full = data['X_train'].shape[0]
                
                # Subsample training data
                sub_data = subsample_data(data, density, seed)
                n_train_actual = sub_data['X_train'].shape[0]
                
                tag = f"density={density:.0%} | {model_type} | seed={seed}"
                print(f"\n--- {tag} ---")
                print(f"  Samples: {n_train_actual}/{n_train_full} ({n_train_actual/n_train_full:.1%})")
                
                # Create model
                config = {
                    'model_type': model_type,
                    **MODEL_DEFAULTS,
                }
                if model_variant == 'pinn':
                    config.update(PHYSICS_CONFIG)
                    config['n_train_samples'] = n_train_actual  # For adaptive weighting
                
                model = get_model(
                    config,
                    n_features=data['X_train'].shape[2],
                    n_stations=len(DATA_CONFIG['site_names']),
                    feature_names=feature_names,
                )
                
                # Create loaders
                train_loader, val_loader, test_loader, _ = create_loaders(
                    sub_data, batch_size=min(TRAIN_CONFIG['batch_size'], n_train_actual)
                )
                
                # Train
                train_cfg = {
                    'epochs': epochs,
                    'learning_rate': TRAIN_CONFIG['learning_rate'],
                    'weight_decay': TRAIN_CONFIG['weight_decay'],
                    'patience': TRAIN_CONFIG['patience'],
                }
                
                save_path = output_dir / f"step1_{model_variant}_d{int(density*100)}_s{seed}.pth"
                
                t_start = time.time()
                train_result = train_model(
                    model, train_loader, val_loader,
                    train_cfg, save_path=str(save_path),
                    verbose=False,
                )
                t_elapsed = time.time() - t_start
                
                # Evaluate
                metrics = evaluate_model(model, test_loader)
                
                row = {
                    'Backbone': backbone,
                    'Variant': model_variant.upper(),
                    'Density': density,
                    'Density_Label': f'{density:.0%}',
                    'N_Train': n_train_actual,
                    'Seed': seed,
                    **metrics,
                    'Best_Val_Loss': train_result['best_val_loss'],
                    'Final_Epoch': train_result['final_epoch'],
                    'Train_Time_s': round(t_elapsed, 1),
                }
                results.append(row)
                
                print(f"  R²={metrics['R²']:.4f} | RMSE={metrics['RMSE']:.4f} | "
                      f"NSE={metrics['NSE']:.4f} | KGE={metrics['KGE']:.4f} | "
                      f"Time={t_elapsed:.1f}s")
    
    # Save results
    df = pd.DataFrame(results)
    csv_path = output_dir / 'phase2_step1_degradation.csv'
    df.to_csv(csv_path, index=False)
    
    # Print summary
    print_header("STEP 1 DEGRADATION SUMMARY")
    
    summary = df.groupby(['Density_Label', 'Variant']).agg({
        'R²': ['mean', 'std'],
        'RMSE': ['mean', 'std'],
        'N_Train': 'first',
    }).round(4)
    print(summary.to_string())
    
    # Compute PINN advantage at each density
    print("\n--- PINN Advantage (ΔR² = R²_PINN - R²_Pure) ---")
    for density in DATA_DENSITIES:
        pure_r2 = df[(df['Density'] == density) & (df['Variant'] == 'PURE')]['R²'].mean()
        pinn_r2 = df[(df['Density'] == density) & (df['Variant'] == 'PINN')]['R²'].mean()
        delta = pinn_r2 - pure_r2
        marker = "[OK]" if delta > 0 else "[WARN]"
        print(f"  {density:>5.0%}: Pure R²={pure_r2:.4f} → PINN R²={pinn_r2:.4f} | "
              f"ΔR²={delta:+.4f} {marker}")
    
    print(f"\nResults saved to: {csv_path}")
    return df


# ---- Step 2: Cross-Site Transfer Learning ----

def split_by_site(data, source_sites, target_site):
    """
    Split data into source (pre-training) and target (fine-tuning) sets.
    
    Source: All training data from source sites
    Target train: Training data from target site only
    Target test: Test data from target site only
    """
    station_ids = data['train_station_ids']
    val_station_ids = data['val_station_ids']
    test_station_ids = data['test_station_ids']
    
    # Source training data (sites 0~3)
    source_mask = np.isin(station_ids, source_sites)
    source_data = {
        'X_train': data['X_train'][source_mask],
        'y_train': data['y_train'][source_mask],
        'train_station_ids': station_ids[source_mask],
    }
    
    # Source validation (sites 0~3)
    source_val_mask = np.isin(val_station_ids, source_sites)
    source_data['X_val'] = data['X_val'][source_val_mask]
    source_data['y_val'] = data['y_val'][source_val_mask]
    source_data['val_station_ids'] = val_station_ids[source_val_mask]
    
    # Source test (for reference)
    source_test_mask = np.isin(test_station_ids, source_sites)
    source_data['X_test'] = data['X_test'][source_test_mask]
    source_data['y_test'] = data['y_test'][source_test_mask]
    source_data['test_station_ids'] = test_station_ids[source_test_mask]
    
    # Target training data (site 4)
    target_mask = station_ids == target_site
    target_data = {
        'X_train': data['X_train'][target_mask],
        'y_train': data['y_train'][target_mask],
        'train_station_ids': station_ids[target_mask],
    }
    
    # Target validation (site 4)
    target_val_mask = val_station_ids == target_site
    target_data['X_val'] = data['X_val'][target_val_mask]
    target_data['y_val'] = data['y_val'][target_val_mask]
    target_data['val_station_ids'] = val_station_ids[target_val_mask]
    
    # Target test (site 4)
    target_test_mask = test_station_ids == target_site
    target_data['X_test'] = data['X_test'][target_test_mask]
    target_data['y_test'] = data['y_test'][target_test_mask]
    target_data['test_station_ids'] = test_station_ids[target_test_mask]
    
    print(f"  Source sites {source_sites}: "
          f"train={source_data['X_train'].shape[0]}, "
          f"val={source_data['X_val'].shape[0]}, "
          f"test={source_data['X_test'].shape[0]}")
    print(f"  Target site {target_site}: "
          f"train={target_data['X_train'].shape[0]}, "
          f"val={target_data['X_val'].shape[0]}, "
          f"test={target_data['X_test'].shape[0]}")
    
    return source_data, target_data


def apply_freezing(model, strategy: str):
    """
    Apply freezing strategy to model for fine-tuning.
    
    Args:
        strategy: 'linear_probe' or 'full_tune'
    """
    if strategy == 'linear_probe':
        # Freeze everything except the final prediction heads
        for name, param in model.named_parameters():
            if any(kw in name.lower() for kw in ['mean_predictor', 'var_predictor',
                                                    'mean_head', 'var_head',
                                                    'head', 'output']):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif strategy == 'full_tune':
        # Unfreeze everything
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  [{strategy}] Trainable: {n_trainable:,}/{n_total:,} "
          f"({n_trainable/n_total:.1%})")


def run_step2(data, feature_names, backbone, seeds, epochs, output_dir):
    """
    Step 2: Cross-Site Transfer Learning.
    
    Experiment matrix (for both Pure DL and PINN):
      - Baseline: Train from scratch on target site only
      - Transfer + Linear Probe: Pre-train on source → freeze backbone → fine-tune head
      - Transfer + Full Tune: Pre-train on source → fine-tune all layers
    
    Output: phase2_step2_transfer.csv
    """
    print_header(f"STEP 2: Cross-Site Transfer Learning ({backbone})")
    
    # Split data by site
    source_data, target_data = split_by_site(data, SOURCE_SITES, TARGET_SITE)
    
    ft_epochs = min(TRANSFER_FINETUNE_EPOCHS, epochs)
    
    # ---- Resume: load existing results ----
    csv_path = output_dir / 'phase2_step2_transfer.csv'
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        results = df_existing.to_dict('records')
        completed = set(zip(df_existing['Variant'], df_existing['Transfer_Mode'], df_existing['Seed']))
        print(f"  [RESUME] Loaded {len(results)} existing results, {len(completed)} completed")
    else:
        results = []
        completed = set()
    
    for model_variant in ['pure', 'pinn']:
        model_type_base = backbone if model_variant == 'pure' else f'{backbone}_PINN'
        
        for transfer_mode in ['scratch', 'linear_probe', 'full_tune']:
            for seed in seeds:
                variant_label = model_variant.upper()
                if (variant_label, transfer_mode, seed) in completed:
                    print(f"\n--- {model_type_base} | {transfer_mode} | seed={seed} --- [SKIP: already completed]")
                    continue
                
                set_seed(seed)
                
                label = f"{model_type_base} | {transfer_mode} | seed={seed}"
                print(f"\n--- {label} ---")
                
                # ---- Create Model ----
                config = {
                    'model_type': model_type_base,
                    **MODEL_DEFAULTS,
                }
                if model_variant == 'pinn':
                    config.update(PHYSICS_CONFIG)
                
                model = get_model(
                    config,
                    n_features=data['X_train'].shape[2],
                    n_stations=len(DATA_CONFIG['site_names']),
                    feature_names=feature_names,
                )
                
                t_start = time.time()
                pretrain_epochs_done = 0
                
                # ---- Phase A: Pre-train on source sites (if transfer) ----
                if transfer_mode != 'scratch':
                    print("  [Phase A] Pre-training on source sites...")
                    
                    source_train_loader, source_val_loader, _, _ = create_loaders(
                        source_data, batch_size=TRAIN_CONFIG['batch_size']
                    )
                    
                    pretrain_cfg = {
                        'epochs': epochs,
                        'learning_rate': TRAIN_CONFIG['learning_rate'],
                        'weight_decay': TRAIN_CONFIG['weight_decay'],
                        'patience': TRAIN_CONFIG['patience'],
                    }
                    
                    pretrain_save = output_dir / f"step2_pretrain_{model_variant}_{transfer_mode}_s{seed}.pth"
                    
                    pretrain_result = train_model(
                        model, source_train_loader, source_val_loader,
                        pretrain_cfg, save_path=str(pretrain_save),
                        verbose=False,
                    )
                    pretrain_epochs_done = pretrain_result['final_epoch']
                    print(f"  Pre-train done: {pretrain_epochs_done} epochs, "
                          f"val_loss={pretrain_result['best_val_loss']:.4f}")
                    
                    # Apply freezing strategy
                    apply_freezing(model, transfer_mode)
                
                # ---- Phase B: Train/Fine-tune on target site ----
                phase_b_label = "Fine-tuning" if transfer_mode != 'scratch' else "Training"
                print(f"  [Phase B] {phase_b_label} on target site...")
                
                target_train_loader, target_val_loader, target_test_loader, _ = create_loaders(
                    target_data,
                    batch_size=min(TRAIN_CONFIG['batch_size'], target_data['X_train'].shape[0]),
                )
                
                ft_cfg = {
                    'epochs': ft_epochs if transfer_mode != 'scratch' else epochs,
                    'learning_rate': TRANSFER_FINETUNE_LR if transfer_mode != 'scratch' else TRAIN_CONFIG['learning_rate'],
                    'weight_decay': TRAIN_CONFIG['weight_decay'],
                    'patience': TRAIN_CONFIG['patience'],
                }
                
                ft_save = output_dir / f"step2_ft_{model_variant}_{transfer_mode}_s{seed}.pth"
                
                ft_result = train_model(
                    model, target_train_loader, target_val_loader,
                    ft_cfg, save_path=str(ft_save),
                    verbose=False,
                )
                
                t_elapsed = time.time() - t_start
                
                # ---- Evaluate on target site test set ----
                metrics = evaluate_model(model, target_test_loader)
                
                row = {
                    'Backbone': backbone,
                    'Variant': variant_label,
                    'Transfer_Mode': transfer_mode,
                    'Seed': seed,
                    'Pretrain_Epochs': pretrain_epochs_done,
                    'Finetune_Epochs': ft_result['final_epoch'],
                    'N_Target_Train': target_data['X_train'].shape[0],
                    'N_Target_Test': target_data['X_test'].shape[0],
                    **metrics,
                    'Best_Val_Loss': ft_result['best_val_loss'],
                    'Total_Time_s': round(t_elapsed, 1),
                }
                results.append(row)
                
                # Save incrementally
                pd.DataFrame(results).to_csv(csv_path, index=False)
                
                print(f"  R²={metrics['R²']:.4f} | RMSE={metrics['RMSE']:.4f} | "
                      f"NSE={metrics['NSE']:.4f} | KGE={metrics['KGE']:.4f} | "
                      f"Total={t_elapsed:.1f}s")
    
    # Final save
    df = pd.DataFrame(results)
    
    # Print summary
    print_header("STEP 2 TRANSFER LEARNING SUMMARY")
    
    summary = df.groupby(['Variant', 'Transfer_Mode']).agg({
        'R²': ['mean', 'std'],
        'RMSE': ['mean', 'std'],
        'KGE': ['mean', 'std'],
    }).round(4)
    print(summary.to_string())
    
    # Compare PINN advantage per transfer mode
    print("\n--- PINN Advantage by Transfer Mode ---")
    for mode in ['scratch', 'linear_probe', 'full_tune']:
        pure_r2 = df[(df['Variant'] == 'PURE') & (df['Transfer_Mode'] == mode)]['R²'].mean()
        pinn_r2 = df[(df['Variant'] == 'PINN') & (df['Transfer_Mode'] == mode)]['R²'].mean()
        delta = pinn_r2 - pure_r2
        marker = "[OK]" if delta > 0 else "[WARN]"
        print(f"  {mode:>14s}: Pure R²={pure_r2:.4f} → PINN R²={pinn_r2:.4f} | "
              f"ΔR²={delta:+.4f} {marker}")
    
    print(f"\nResults saved to: {csv_path}")
    return df


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(
        description='Phase 2: Data Scarcity & Transfer Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--step', type=int, choices=[1, 2], default=None,
                        help='Run specific step (default: all)')
    parser.add_argument('--backbone', type=str, default=None,
                        help='Override backbone (default: Phase 1 selection)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs (default: 100)')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS,
                        help='Random seeds (default: 42 123 456)')
    parser.add_argument('--output-dir', type=str, default='experiments/phase2',
                        help='Output directory')
    parser.add_argument('--phase1-dir', type=str, default='experiments/phase1',
                        help='Phase 1 results directory')
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phase1_dir = Path(args.phase1_dir)
    
    backbone = get_backbone(args.backbone, phase1_dir)
    
    print_header("Five-station rerun - Phase 2: Data Scarcity & Transfer Learning")
    print(f"Backbone: {backbone}")
    print(f"Device:   {device}")
    print(f"Epochs:   {args.epochs}")
    print(f"Seeds:    {args.seeds}")
    print(f"Output:   {output_dir}")
    print(f"Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    print("\nLoading data...")
    data, feature_names, preprocessor = load_data()
    print(f"  Train: {data['X_train'].shape[0]} samples")
    print(f"  Val:   {data['X_val'].shape[0]} samples")
    print(f"  Test:  {data['X_test'].shape[0]} samples")
    
    # Save experiment config
    exp_config = {
        'backbone': backbone,
        'epochs': args.epochs,
        'seeds': args.seeds,
        'data_densities': DATA_DENSITIES,
        'source_sites': SOURCE_SITES,
        'target_site': TARGET_SITE,
        'model_defaults': MODEL_DEFAULTS,
        'train_config': TRAIN_CONFIG,
        'physics_config': PHYSICS_CONFIG,
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_dir / 'experiment_config.json', 'w') as f:
        json.dump(exp_config, f, indent=2, ensure_ascii=False)
    
    # Run steps
    if args.step is None or args.step == 1:
        run_step1(data, feature_names, backbone, args.seeds, args.epochs, output_dir)
    
    if args.step is None or args.step == 2:
        run_step2(data, feature_names, backbone, args.seeds, args.epochs, output_dir)
    
    print_header("PHASE 2 COMPLETE")
    print(f"All results saved to: {output_dir}/")


if __name__ == '__main__':
    main()
