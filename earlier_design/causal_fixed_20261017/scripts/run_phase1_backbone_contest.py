#!/usr/bin/env python
"""
Phase 1: Backbone Contest - Select the optimal backbone for PINN-HAB.

Three steps:
  Step 1: Pure data-driven performance comparison (4 backbones × 3 seeds)
  Step 2: Physics loss convergence test (select best backbone)
  Step 3: Ablation study (Case 0~3) on the selected backbone

Usage:
    python scripts/run_phase1_backbone_contest.py
    python scripts/run_phase1_backbone_contest.py --step 1            # Step 1 only
    python scripts/run_phase1_backbone_contest.py --step 2            # Step 2 only
    python scripts/run_phase1_backbone_contest.py --step 3 --backbone Mamba  # Step 3 with specific backbone
    python scripts/run_phase1_backbone_contest.py --epochs 10 --seeds 42  # Quick test
"""

import argparse
import json
import os
import sys
import random
import time
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
from pinn_hab.evaluation import compute_metrics
from pinn_hab.utils.device import device


# ---- Configuration ----

BACKBONES = ['iTransformer', 'Mamba', 'xLSTM', 'ModernTCN']
DEFAULT_SEEDS = [42, 123, 456]

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

# Forecast lead time in days. Rerun addition: the original pipeline forecasts 1 day ahead.
HORIZON = int(os.environ.get('PINN_HORIZON', '1'))

SPLIT_CONFIG = {
    'train_period': ('2020-01-01', '2022-12-31'),
    'val_period': ('2023-01-01', '2023-12-31'),
    'test_period': ('2024-01-01', '2024-12-31'),
}


# ---- Utilities ----

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def load_data():
    """Load and preprocess data (called once, reused across experiments)."""
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
        horizon=HORIZON,
        **SPLIT_CONFIG,
    )
    
    return data, feature_names, preprocessor


def create_loaders(data, batch_size=32):
    """Create DataLoaders from split data."""
    return create_dataloaders(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        data['X_test'], data['y_test'],
        data['train_station_ids'],
        data['val_station_ids'],
        data['test_station_ids'],
        batch_size=batch_size,
    )


def evaluate_model(model, test_loader, preprocessor=None):
    """Evaluate model on test set and return metrics."""
    model.eval()
    model.to(device)
    
    all_preds = []
    all_targets = []
    
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
    
    metrics = compute_metrics(y_true, y_pred)
    return metrics


def count_params(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ---- Step 1: Pure Data-Driven Performance Comparison ----

def run_step1(data, feature_names, seeds, epochs, output_dir):
    """
    Step 1: Compare 4 backbones WITHOUT physics constraints.
    
    For each backbone × seed, train and evaluate on test set.
    Output: phase1_step1_results.csv
    """
    print_header("STEP 1: Pure Data-Driven Backbone Comparison")
    
    # ---- Resume: load existing results ----
    csv_path = output_dir / 'phase1_step1_results.csv'
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        results = df_existing.to_dict('records')
        completed = set(zip(df_existing['Backbone'], df_existing['Seed']))
        print(f"  [RESUME] Loaded {len(results)} existing results, {len(completed)} completed")
    else:
        results = []
        completed = set()
    
    for backbone in BACKBONES:
        for seed in seeds:
            if (backbone, seed) in completed:
                print(f"\n--- {backbone} | Seed={seed} --- [SKIP: already completed]")
                continue
            
            set_seed(seed)
            print(f"\n--- {backbone} | Seed={seed} ---")
            
            # Create model
            config = {
                'model_type': backbone,
                **MODEL_DEFAULTS,
            }
            model = get_model(
                config,
                n_features=data['X_train'].shape[2],
                n_stations=len(DATA_CONFIG['site_names']),
                feature_names=feature_names,
            )
            
            total_params, trainable_params = count_params(model)
            print(f"  Parameters: {total_params:,} (trainable: {trainable_params:,})")
            
            # Create fresh loaders
            train_loader, val_loader, test_loader, _ = create_loaders(
                data, batch_size=TRAIN_CONFIG['batch_size']
            )
            
            # Train
            train_config = {
                'epochs': epochs,
                'learning_rate': TRAIN_CONFIG['learning_rate'],
                'weight_decay': TRAIN_CONFIG['weight_decay'],
                'patience': TRAIN_CONFIG['patience'],
            }
            
            save_path = output_dir / f"step1_{backbone}_seed{seed}.pth"
            
            t_start = time.time()
            train_result = train_model(
                model, train_loader, val_loader,
                train_config, save_path=str(save_path),
                verbose=False,
            )
            t_elapsed = time.time() - t_start
            
            # Evaluate
            metrics = evaluate_model(model, test_loader)
            
            row = {
                'Backbone': backbone,
                'Seed': seed,
                'Params': total_params,
                **metrics,
                'Best_Val_Loss': train_result['best_val_loss'],
                'Final_Epoch': train_result['final_epoch'],
                'Train_Time_s': round(t_elapsed, 1),
            }
            results.append(row)
            
            # Save incrementally
            pd.DataFrame(results).to_csv(csv_path, index=False)
            
            print(f"  R²={metrics['R²']:.4f} | RMSE={metrics['RMSE']:.4f} | "
                  f"NSE={metrics['NSE']:.4f} | KGE={metrics['KGE']:.4f} | "
                  f"Time={t_elapsed:.1f}s")
    
    # Final save
    df_results = pd.DataFrame(results)
    
    # Print summary
    print_header("STEP 1 SUMMARY")
    summary = df_results.groupby('Backbone').agg({
        'R²': ['mean', 'std'],
        'RMSE': ['mean', 'std'],
        'NSE': ['mean', 'std'],
        'KGE': ['mean', 'std'],
        'Params': 'first',
        'Train_Time_s': 'mean',
    }).round(4)
    print(summary.to_string())
    print(f"\nResults saved to: {csv_path}")
    
    return df_results


# ---- Step 2: Physics Loss Convergence Test ----

def run_step2(data, feature_names, seeds, epochs, output_dir, step1_results=None):
    """
    Step 2: Test physics loss convergence for each backbone.
    
    Score = 0.6 × R²_retention + 0.4 × Physics_stability
    Select the backbone with the highest score.
    
    Output: phase1_step2_results.csv, best_backbone.txt
    """
    print_header("STEP 2: Physics Loss Convergence Test")
    
    # Load Step 1 results if available (for R² retention calculation)
    if step1_results is None:
        step1_path = output_dir / 'phase1_step1_results.csv'
        if step1_path.exists():
            step1_results = pd.read_csv(step1_path)
        else:
            print("[WARNING] No Step 1 results found. Running with R²_retention = R²_PINN.")
    
    # Get pure R² per backbone (mean across seeds)
    pure_r2 = {}
    if step1_results is not None:
        for bb in BACKBONES:
            mask = step1_results['Backbone'] == bb
            pure_r2[bb] = step1_results.loc[mask, 'R²'].mean()
    
    # ---- Resume: load existing results ----
    csv_path = output_dir / 'phase1_step2_results.csv'
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        results = df_existing.to_dict('records')
        completed = set(zip(df_existing['Backbone'], df_existing['Seed']))
        print(f"  [RESUME] Loaded {len(results)} existing results, {len(completed)} completed")
    else:
        results = []
        completed = set()
    
    for backbone in BACKBONES:
        for seed in seeds:
            if (backbone, seed) in completed:
                print(f"\n--- {backbone}_PINN | Seed={seed} --- [SKIP: already completed]")
                continue
            
            set_seed(seed)
            print(f"\n--- {backbone}_PINN | Seed={seed} ---")
            
            # Create PINN model
            config = {
                'model_type': f'{backbone}_PINN',
                **MODEL_DEFAULTS,
                **PHYSICS_CONFIG,
            }
            model = get_model(
                config,
                n_features=data['X_train'].shape[2],
                n_stations=len(DATA_CONFIG['site_names']),
                feature_names=feature_names,
            )
            
            total_params, _ = count_params(model)
            
            # Create loaders
            train_loader, val_loader, test_loader, _ = create_loaders(
                data, batch_size=TRAIN_CONFIG['batch_size']
            )
            
            # Train
            train_config = {
                'epochs': epochs,
                'learning_rate': TRAIN_CONFIG['learning_rate'],
                'weight_decay': TRAIN_CONFIG['weight_decay'],
                'patience': TRAIN_CONFIG['patience'],
            }
            
            save_path = output_dir / f"step2_{backbone}_PINN_seed{seed}.pth"
            
            t_start = time.time()
            train_result = train_model(
                model, train_loader, val_loader,
                train_config, save_path=str(save_path),
                verbose=False,
            )
            t_elapsed = time.time() - t_start
            
            # Evaluate
            metrics = evaluate_model(model, test_loader)
            
            # Physics loss stability (std of last 20 epochs)
            physics_losses = train_result['history'].get('physics_loss', [])
            if len(physics_losses) >= 20:
                last_20 = physics_losses[-20:]
                physics_std = np.std(last_20)
                physics_mean = np.mean(last_20)
            elif len(physics_losses) > 0:
                last_20 = physics_losses
                physics_std = np.std(last_20)
                physics_mean = np.mean(last_20)
            else:
                physics_std = float('inf')
                physics_mean = 0.0
            
            # R² retention
            r2_pinn = metrics['R²']
            r2_pure = pure_r2.get(backbone, r2_pinn)
            r2_retention = r2_pinn / (r2_pure + 1e-8) if r2_pure > 0 else 0
            
            # Physics stability score (inverse of std, capped)
            physics_stability = 1.0 / (physics_std + 1e-6)
            physics_stability = min(physics_stability, 100)  # Cap at 100
            
            # Composite score
            score = 0.6 * min(r2_retention, 1.0) + 0.4 * (physics_stability / 100)
            
            row = {
                'Backbone': backbone,
                'Seed': seed,
                'Params': total_params,
                'R2_PINN': r2_pinn,
                'R2_Pure': r2_pure,
                'R2_Retention': r2_retention,
                'Physics_Loss_Mean': physics_mean,
                'Physics_Loss_Std': physics_std,
                'Physics_Stability': physics_stability,
                'Composite_Score': score,
                **{f'PINN_{k}': v for k, v in metrics.items()},
                'Best_Val_Loss': train_result['best_val_loss'],
                'Final_Epoch': train_result['final_epoch'],
                'Train_Time_s': round(t_elapsed, 1),
            }
            results.append(row)
            
            # Save incrementally
            pd.DataFrame(results).to_csv(csv_path, index=False)
            
            print(f"  R²={r2_pinn:.4f} (Retention={r2_retention:.2%}) | "
                  f"Physics: mean={physics_mean:.4f}, std={physics_std:.4f} | "
                  f"Score={score:.4f}")
    
    # Final save
    df_results = pd.DataFrame(results)
    
    # Select best backbone
    summary = df_results.groupby('Backbone').agg({
        'Composite_Score': 'mean',
        'R2_Retention': 'mean',
        'Physics_Stability': 'mean',
        'R2_PINN': 'mean',
    }).round(4)
    
    best_backbone = summary['R2_PINN'].idxmax()
    
    print_header("STEP 2 SUMMARY")
    print(summary.to_string())
    print(f"\n[BEST] BEST BACKBONE: {best_backbone} "
          f"(PINN R²={summary.loc[best_backbone, 'R2_PINN']:.4f})")
    
    # Save selection
    selection_path = output_dir / 'best_backbone.txt'
    with open(selection_path, 'w') as f:
        f.write(best_backbone)
    
    print(f"\nResults saved to: {csv_path}")
    print(f"Selection saved to: {selection_path}")
    
    return df_results, best_backbone


# ---- Step 3: Physics Ablation Study ----

def run_step3(data, feature_names, seeds, epochs, output_dir, backbone=None):
    """
    Step 3: Ablation on the selected backbone.
    
    Cases:
      Case 0: Pure deep learning (no physics)
      Case 1: + ODE Loss
      Case 2: + Site-specific mu_max embedding
      Case 3: + Sigmoid hard constraints (full PINN)
    
    Output: phase1_step3_ablation.csv
    """
    # Determine backbone
    if backbone is None:
        selection_path = output_dir / 'best_backbone.txt'
        if selection_path.exists():
            backbone = selection_path.read_text().strip()
        else:
            backbone = 'iTransformer'
            print(f"[WARNING] No backbone selection found. Using default: {backbone}")
    
    print_header(f"STEP 3: Ablation Study ({backbone})")
    
    # Define ablation cases
    cases = {
        'Case0': {
            'label': 'Pure DL (No Physics)',
            'model_type': backbone,  # No _PINN suffix
        },
        'Case1': {
            'label': '+ ODE Loss',
            'model_type': f'{backbone}_PINN',
            'physics_overrides': {
                # Basic ODE only, no site-specific, no hard constraints
                'use_collocation': False,
                'use_gradient_balancing': False,
                'freeze_physics_params': True,  # Fixed params (no learning)
            },
        },
        'Case2': {
            'label': '+ Site-Specific μ_max',
            'model_type': f'{backbone}_PINN',
            'physics_overrides': {
                'use_collocation': True,
                'use_gradient_balancing': True,
                'freeze_physics_params': False,  # Learnable params
            },
        },
        'Case3': {
            'label': '+ Hard Constraints (Full PINN)',
            'model_type': f'{backbone}_PINN',
            'physics_overrides': {
                'use_collocation': True,
                'use_gradient_balancing': True,
                'freeze_physics_params': False,
                # Hard constraints are always active via Sigmoid/Softplus in biogeochemical.py
            },
        },
    }
    
    # ---- Resume: load existing results ----
    csv_path = output_dir / 'phase1_step3_ablation.csv'
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        results = df_existing.to_dict('records')
        completed = set(zip(df_existing['Case'], df_existing['Seed']))
        print(f"  [RESUME] Loaded {len(results)} existing results, {len(completed)} completed")
    else:
        results = []
        completed = set()
    
    for case_id, case_cfg in cases.items():
        for seed in seeds:
            if (case_id, seed) in completed:
                print(f"\n--- {case_id}: {case_cfg['label']} | Seed={seed} --- [SKIP: already completed]")
                continue
            
            set_seed(seed)
            print(f"\n--- {case_id}: {case_cfg['label']} | Seed={seed} ---")
            
            # Build config
            config = {
                'model_type': case_cfg['model_type'],
                **MODEL_DEFAULTS,
            }
            
            if '_PINN' in case_cfg['model_type']:
                config.update(PHYSICS_CONFIG)
                config.update(case_cfg.get('physics_overrides', {}))
            
            model = get_model(
                config,
                n_features=data['X_train'].shape[2],
                n_stations=len(DATA_CONFIG['site_names']),
                feature_names=feature_names,
            )
            
            total_params, _ = count_params(model)
            
            # Create loaders
            train_loader, val_loader, test_loader, _ = create_loaders(
                data, batch_size=TRAIN_CONFIG['batch_size']
            )
            
            # Train
            train_config = {
                'epochs': epochs,
                'learning_rate': TRAIN_CONFIG['learning_rate'],
                'weight_decay': TRAIN_CONFIG['weight_decay'],
                'patience': TRAIN_CONFIG['patience'],
            }
            
            save_path = output_dir / f"step3_{case_id}_seed{seed}.pth"
            
            t_start = time.time()
            train_result = train_model(
                model, train_loader, val_loader,
                train_config, save_path=str(save_path),
                verbose=False,
            )
            t_elapsed = time.time() - t_start
            
            # Evaluate
            metrics = evaluate_model(model, test_loader)
            
            row = {
                'Backbone': backbone,
                'Case': case_id,
                'Case_Label': case_cfg['label'],
                'Seed': seed,
                'Params': total_params,
                **metrics,
                'Best_Val_Loss': train_result['best_val_loss'],
                'Final_Epoch': train_result['final_epoch'],
                'Train_Time_s': round(t_elapsed, 1),
            }
            results.append(row)
            
            # Save incrementally
            pd.DataFrame(results).to_csv(csv_path, index=False)
            
            print(f"  R²={metrics['R²']:.4f} | RMSE={metrics['RMSE']:.4f} | "
                  f"NSE={metrics['NSE']:.4f} | KGE={metrics['KGE']:.4f}")
    
    # Final save
    df_results = pd.DataFrame(results)
    
    # Summary
    print_header("STEP 3 ABLATION SUMMARY")
    summary = df_results.groupby(['Case', 'Case_Label']).agg({
        'R²': ['mean', 'std'],
        'RMSE': ['mean', 'std'],
        'NSE': ['mean', 'std'],
        'KGE': ['mean', 'std'],
    }).round(4)
    print(summary.to_string())
    print(f"\nResults saved to: {csv_path}")
    
    return df_results


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(
        description='Phase 1: Backbone Contest for PINN-HAB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--step', type=int, choices=[1, 2, 3], default=None,
                        help='Run specific step (default: all)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs (default: 100)')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS,
                        help='Random seeds (default: 42 123 456)')
    parser.add_argument('--backbone', type=str, default=None,
                        help='Override backbone for Step 3')
    parser.add_argument('--output-dir', type=str, default='experiments/phase1',
                        help='Output directory (default: experiments/phase1)')
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print_header("Five-station rerun - Phase 1: Backbone Contest")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {output_dir}")
    print(f"Horizon: {HORIZON} day(s) ahead")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data (once)
    print("\nLoading data...")
    data, feature_names, preprocessor = load_data()
    print(f"  Train: {data['X_train'].shape[0]} samples")
    print(f"  Val:   {data['X_val'].shape[0]} samples")
    print(f"  Test:  {data['X_test'].shape[0]} samples")
    print(f"  Features: {len(feature_names)}")
    
    # Save experiment config
    exp_config = {
        'epochs': args.epochs,
        'seeds': args.seeds,
        'backbones': BACKBONES,
        'model_defaults': MODEL_DEFAULTS,
        'train_config': TRAIN_CONFIG,
        'physics_config': PHYSICS_CONFIG,
        'data_config': {k: v for k, v in DATA_CONFIG.items() if k != 'path'},
        'split_config': SPLIT_CONFIG,
        'horizon': HORIZON,
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_dir / 'experiment_config.json', 'w') as f:
        json.dump(exp_config, f, indent=2, ensure_ascii=False)
    
    # Run steps
    step1_results = None
    best_backbone = args.backbone
    
    if args.step is None or args.step == 1:
        step1_results = run_step1(
            data, feature_names, args.seeds, args.epochs, output_dir
        )
    
    if args.step is None or args.step == 2:
        _, best_backbone = run_step2(
            data, feature_names, args.seeds, args.epochs, output_dir,
            step1_results=step1_results,
        )
    
    if args.step is None or args.step == 3:
        run_step3(
            data, feature_names, args.seeds, args.epochs, output_dir,
            backbone=best_backbone or args.backbone,
        )
    
    print_header("PHASE 1 COMPLETE")
    print(f"All results saved to: {output_dir}/")


if __name__ == '__main__':
    main()
