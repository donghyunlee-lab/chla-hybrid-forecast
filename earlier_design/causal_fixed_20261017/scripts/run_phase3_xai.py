#!/usr/bin/env python
"""
Phase 3: XAI & Ecological Validation.

Extracts and validates learned physical parameters from trained PINN models.

Steps:
  Step 1: Parameter extraction - Extract μ_max (per site), r, m and compare
          against ecological literature values.
  Step 2: Physical violation analysis - Compute ODE consistency, net growth
          anomalies, and constraint violation rates across test data.
  Step 3: Parameter dynamics - Analyze how physical parameters evolve during
          training (requires checkpoints from Phase 1/2).

Prerequisites:
  - Phase 1 completed (trained PINN model checkpoints exist)

Usage:
    python scripts/run_phase3_xai.py
    python scripts/run_phase3_xai.py --step 1          # Parameter extraction only
    python scripts/run_phase3_xai.py --step 2          # Violation analysis only
    python scripts/run_phase3_xai.py --step 3          # Dynamics tracking only
    python scripts/run_phase3_xai.py --backbone Mamba   # Override backbone
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pinn_hab.data import DataPreprocessor, create_dataloaders
from pinn_hab.models import get_model
from pinn_hab.training import train_model
from pinn_hab.evaluation import compute_metrics
from pinn_hab.utils.device import device


# ---- Configuration ----

DEFAULT_SEEDS = [42, 123, 456]

DATA_CONFIG = {
    'path': 'data/imputed_daily_data.csv',
    'site_names': ['공주', '대청호', '갑천', '부여', '용담호'],
    'apply_log_transform': True,
    'use_panel_data': True,
}

SITE_NAMES = {
    0: 'Gongju (공주)',
    1: 'Daecheong (대청호)',
    2: 'Gapcheon (갑천)',
    3: 'Buyeo (부여)',
    4: 'Yongdam (용담호)',
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

# Literature reference ranges for ecological validation
# Sources: Eppley (1972), Bowie et al. (1985), Chapra (1997)
LITERATURE_RANGES = {
    'mu_max': {
        'range': (0.5, 2.0),
        'unit': 'day⁻¹',
        'description': 'Max specific growth rate of freshwater algae',
        'references': 'Eppley (1972), Reynolds (2006)',
    },
    'r': {
        'range': (0.02, 0.15),
        'unit': 'day⁻¹',
        'description': 'Respiration loss rate',
        'references': 'Bowie et al. (1985)',
    },
    'm': {
        'range': (0.01, 0.05),
        'unit': 'day⁻¹',
        'description': 'Non-predatory mortality rate',
        'references': 'Chapra (1997)',
    },
    'K_N': {
        'range': (0.01, 0.5),
        'unit': 'mg/L',
        'description': 'Half-saturation for nitrogen',
        'references': 'Bowie et al. (1985)',
    },
    'K_P': {
        'range': (0.001, 0.05),
        'unit': 'mg/L',
        'description': 'Half-saturation for phosphorus',
        'references': 'Bowie et al. (1985)',
    },
}


# ---- Utilities ----

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_backbone(args_backbone, phase1_dir):
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


def create_loaders(data, batch_size=32):
    return create_dataloaders(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        data['X_test'], data['y_test'],
        data.get('train_station_ids'),
        data.get('val_station_ids'),
        data.get('test_station_ids'),
        batch_size=batch_size,
    )


def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def train_pinn_model(data, feature_names, backbone, seed, epochs, save_path):
    """Train a PINN model and return it."""
    set_seed(seed)
    
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
    
    train_loader, val_loader, _, _ = create_loaders(
        data, batch_size=TRAIN_CONFIG['batch_size']
    )
    
    train_cfg = {
        'epochs': epochs,
        'learning_rate': TRAIN_CONFIG['learning_rate'],
        'weight_decay': TRAIN_CONFIG['weight_decay'],
        'patience': TRAIN_CONFIG['patience'],
    }
    
    result = train_model(
        model, train_loader, val_loader,
        train_cfg, save_path=str(save_path),
        verbose=False,
    )
    
    return model, result


def load_or_train_model(data, feature_names, backbone, seed, epochs, output_dir):
    """Load existing checkpoint or train new model."""
    # Check Phase 1 checkpoints first
    phase1_path = Path('experiments/phase1') / f"step2_{backbone}_PINN_seed{seed}.pth"
    model_path = output_dir / f"pinn_{backbone}_seed{seed}.pth"
    
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
    model = model.to(device)
    
    # Try loading existing checkpoint
    for path in [phase1_path, model_path]:
        if path.exists():
            print(f"  Loading checkpoint: {path}")
            model.load_state_dict(torch.load(str(path), map_location=device))
            return model, path
    
    # Train new model if no checkpoint found
    print(f"  No checkpoint found. Training new model...")
    model, result = train_pinn_model(
        data, feature_names, backbone, seed, epochs, str(model_path)
    )
    print(f"  Trained for {result['final_epoch']} epochs, val_loss={result['best_val_loss']:.4f}")
    return model, model_path


# ---- Step 1: Parameter Extraction & Literature Comparison ----

def run_step1(data, feature_names, backbone, seeds, epochs, output_dir):
    """
    Step 1: Extract learned physical parameters and compare with literature.
    
    For each seed:
      - Load/train PINN model
      - Extract μ_max (per site), r, m, K_N, K_P
      - Compare against literature ranges
      - Report violation rates
    
    Output: phase3_step1_parameters.csv, phase3_step1_literature_comparison.csv
    """
    print_header(f"STEP 1: Parameter Extraction & Literature Comparison ({backbone})")
    
    all_params = []
    
    for seed in seeds:
        print(f"\n--- Seed={seed} ---")
        model, ckpt_path = load_or_train_model(
            data, feature_names, backbone, seed, epochs, output_dir
        )
        model.eval()
        
        # Check if model has physics module
        if not hasattr(model, 'physics'):
            print(f"  [SKIP] Model has no physics module")
            continue
        
        physics = model.physics
        
        # Extract global parameters
        r_val = physics.get_r().item()
        m_val = physics.get_m().item()
        K_N_val = physics.get_K_N().item()
        K_P_val = physics.get_K_P().item()
        
        # Extract site-specific μ_max
        for site_id in range(physics.n_sites):
            site_tensor = torch.tensor([site_id], device=device)
            mu_max_val = physics.get_mu_max(site_tensor).item()
            
            row = {
                'Backbone': backbone,
                'Seed': seed,
                'Site_ID': site_id,
                'Site_Name': SITE_NAMES.get(site_id, f'Site_{site_id}'),
                'mu_max': mu_max_val,
                'r': r_val,
                'm': m_val,
                'K_N': K_N_val,
                'K_P': K_P_val,
                'net_growth_max': mu_max_val - r_val - m_val,
                'checkpoint': str(ckpt_path),
            }
            all_params.append(row)
            
            print(f"  Site {site_id} ({SITE_NAMES.get(site_id, '?'):20s}): "
                  f"μ_max={mu_max_val:.4f} | r={r_val:.4f} | m={m_val:.4f} | "
                  f"K_N={K_N_val:.4f} | K_P={K_P_val:.4f}")
    
    if not all_params:
        print("[ERROR] No parameters extracted. Check that PINN models exist.")
        return None, None
    
    # Save raw parameters
    df_params = pd.DataFrame(all_params)
    params_csv = output_dir / 'phase3_step1_parameters.csv'
    df_params.to_csv(params_csv, index=False)
    
    # ---- Literature Comparison ----
    print_header("LITERATURE COMPARISON")
    
    comparison_rows = []
    for param_name, lit_info in LITERATURE_RANGES.items():
        lit_min, lit_max = lit_info['range']
        
        if param_name == 'mu_max':
            values = df_params['mu_max'].values
        elif param_name in df_params.columns:
            values = df_params[param_name].values
        else:
            continue
        
        learned_mean = np.mean(values)
        learned_std = np.std(values)
        learned_min = np.min(values)
        learned_max = np.max(values)
        
        # Violation: values outside literature range
        violations = ((values < lit_min) | (values > lit_max)).sum()
        violation_rate = violations / len(values) * 100
        
        in_range = "[OK]" if violation_rate == 0 else f"[WARN] ({violation_rate:.0f}%)"
        
        comparison_rows.append({
            'Parameter': param_name,
            'Unit': lit_info['unit'],
            'Lit_Min': lit_min,
            'Lit_Max': lit_max,
            'Learned_Mean': round(learned_mean, 4),
            'Learned_Std': round(learned_std, 4),
            'Learned_Min': round(learned_min, 4),
            'Learned_Max': round(learned_max, 4),
            'Violation_Rate_%': round(violation_rate, 1),
            'References': lit_info['references'],
        })
        
        print(f"  {param_name:10s} [{lit_info['unit']:>8s}]: "
              f"Learned={learned_mean:.4f}±{learned_std:.4f} "
              f"(range: {learned_min:.4f}~{learned_max:.4f}) | "
              f"Literature: [{lit_min}, {lit_max}] | {in_range}")
    
    df_comparison = pd.DataFrame(comparison_rows)
    comp_csv = output_dir / 'phase3_step1_literature_comparison.csv'
    df_comparison.to_csv(comp_csv, index=False)
    
    # Summary table by site
    print_header("SITE-WISE μ_max SUMMARY")
    site_summary = df_params.groupby('Site_Name').agg({
        'mu_max': ['mean', 'std'],
        'r': 'mean',
        'm': 'mean',
        'net_growth_max': 'mean',
    }).round(4)
    print(site_summary.to_string())
    
    print(f"\nParameters saved to: {params_csv}")
    print(f"Comparison saved to: {comp_csv}")
    
    return df_params, df_comparison


# ---- Step 2: Physical Violation Analysis ----

def run_step2(data, feature_names, backbone, seeds, epochs, output_dir):
    """
    Step 2: Physical violation analysis on test data.
    
    For each test sample:
      - Compute ODE residual |dC/dt_observed - (μ-r-m)·C|
      - Check if net growth rate is within plausible bounds
      - Count violations of physical constraints
    
    Output: phase3_step2_violations.csv, phase3_step2_summary.csv
    """
    print_header(f"STEP 2: Physical Violation Analysis ({backbone})")
    
    all_violations = []
    aggregate_stats = []
    
    for seed in seeds:
        print(f"\n--- Seed={seed} ---")
        model, ckpt_path = load_or_train_model(
            data, feature_names, backbone, seed, epochs, output_dir
        )
        model.eval()
        model.to(device)
        
        if not hasattr(model, 'physics'):
            print(f"  [SKIP] No physics module")
            continue
        
        physics = model.physics
        
        # Get test data
        test_X = torch.tensor(data['X_test'], dtype=torch.float32).to(device)
        test_y = torch.tensor(data['y_test'], dtype=torch.float32)
        test_station_ids = torch.tensor(data['test_station_ids'], dtype=torch.long).to(device)
        
        n_samples = test_X.shape[0]
        batch_size = 128
        
        ode_residuals = []
        net_rates_all = []
        predictions = []
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_X = test_X[i:i+batch_size]
                batch_sid = test_station_ids[i:i+batch_size]
                
                # Model predictions
                mean, var = model(batch_X, batch_sid)
                predictions.append(mean.cpu().numpy())
                
                # Extract environmental variables (last timestep)
                env_vars = physics.extract_environmental_vars(batch_X, timestep=-1)
                
                # Compute net rate at last timestep
                net_rate = physics.net_rate(
                    env_vars['T'], env_vars['N'], env_vars['P'],
                    station_ids=batch_sid
                )
                net_rates_all.append(net_rate.cpu().numpy())
                
                # Compute sequence ODE residual
                ode_res = physics.compute_sequence_ode_residual(
                    batch_X, station_ids=batch_sid
                )
                ode_residuals.append(ode_res.cpu().item())
        
        net_rates = np.concatenate(net_rates_all, axis=0).flatten()
        y_pred = np.concatenate(predictions, axis=0).flatten()
        y_true = test_y.numpy().flatten()
        
        # ---- Violation Analysis ----
        
        # 1. Net growth rate violations
        # Plausible range: net rate should rarely exceed 0.5 or drop below -0.3
        excessive_growth = (net_rates > 0.5).sum()
        excessive_decay = (net_rates < -0.3).sum()
        net_rate_violation_rate = (excessive_growth + excessive_decay) / len(net_rates) * 100
        
        # 2. Negative prediction violations
        neg_pred_rate = (y_pred < 0).sum() / len(y_pred) * 100
        
        # 3. ODE residual magnitude
        mean_ode_residual = np.mean(ode_residuals) if ode_residuals else 0.0
        
        # 4. Prediction metrics
        metrics = compute_metrics(y_true, y_pred)
        
        stats = {
            'Backbone': backbone,
            'Seed': seed,
            'N_Test_Samples': n_samples,
            'Mean_ODE_Residual': round(mean_ode_residual, 6),
            'Net_Rate_Mean': round(np.mean(net_rates), 4),
            'Net_Rate_Std': round(np.std(net_rates), 4),
            'Net_Rate_Min': round(np.min(net_rates), 4),
            'Net_Rate_Max': round(np.max(net_rates), 4),
            'Excessive_Growth_Rate_%': round(excessive_growth / len(net_rates) * 100, 2),
            'Excessive_Decay_Rate_%': round(excessive_decay / len(net_rates) * 100, 2),
            'Net_Rate_Violation_%': round(net_rate_violation_rate, 2),
            'Negative_Pred_%': round(neg_pred_rate, 2),
            **{f'Test_{k}': v for k, v in metrics.items()},
        }
        aggregate_stats.append(stats)
        
        print(f"  ODE Residual: {mean_ode_residual:.6f}")
        print(f"  Net Rate: {np.mean(net_rates):.4f} ± {np.std(net_rates):.4f} "
              f"(range: [{np.min(net_rates):.4f}, {np.max(net_rates):.4f}])")
        print(f"  Violations: net_rate={net_rate_violation_rate:.1f}%, "
              f"neg_pred={neg_pred_rate:.1f}%")
        print(f"  Test R²={metrics['R²']:.4f} | RMSE={metrics['RMSE']:.4f}")
    
    if not aggregate_stats:
        print("[ERROR] No violation analysis completed.")
        return None
    
    # Save
    df_stats = pd.DataFrame(aggregate_stats)
    stats_csv = output_dir / 'phase3_step2_violations.csv'
    df_stats.to_csv(stats_csv, index=False)
    
    # Summary
    print_header("STEP 2 VIOLATION SUMMARY")
    summary = df_stats[[
        'Seed', 'Mean_ODE_Residual', 'Net_Rate_Mean', 'Net_Rate_Violation_%',
        'Negative_Pred_%', 'Test_R²', 'Test_RMSE'
    ]]
    print(summary.to_string(index=False))
    
    # Aggregate
    print(f"\n  Avg ODE Residual: {df_stats['Mean_ODE_Residual'].mean():.6f}")
    print(f"  Avg Net Rate Violation: {df_stats['Net_Rate_Violation_%'].mean():.2f}%")
    print(f"  Avg Negative Prediction: {df_stats['Negative_Pred_%'].mean():.2f}%")
    print(f"\nResults saved to: {stats_csv}")
    
    return df_stats


# ---- Step 3: Parameter Dynamics During Training ----

def run_step3(data, feature_names, backbone, seed, epochs, output_dir):
    """
    Step 3: Track how physics parameters evolve during training.
    
    Trains a PINN model from scratch, recording parameter snapshots
    at every N epochs.
    
    Output: phase3_step3_dynamics.csv
    """
    print_header(f"STEP 3: Parameter Dynamics During Training ({backbone}, seed={seed})")
    
    set_seed(seed)
    
    # Create model
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
    model = model.to(device)
    
    if not hasattr(model, 'physics'):
        print("[ERROR] Model has no physics module")
        return None
    
    physics = model.physics
    
    # Create loaders
    train_loader, val_loader, _, _ = create_loaders(
        data, batch_size=TRAIN_CONFIG['batch_size']
    )
    
    # Training with parameter snapshots
    from pinn_hab.training.trainer import build_optimizer      # physics parameters without weight decay (rerun fix)
    optimizer = build_optimizer(model, TRAIN_CONFIG['learning_rate'], TRAIN_CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    snapshot_interval = max(1, epochs // 50)  # ~50 snapshots
    dynamics = []
    
    print(f"  Snapshot interval: every {snapshot_interval} epoch(s)")
    print(f"  Training for {epochs} epochs...\n")
    
    from pinn_hab.training.losses import gaussian_nll_loss
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        epoch_physics_losses = []
        
        for batch in train_loader:
            X = batch['X'].to(device)
            y = batch['y'].to(device)
            station_ids = batch.get('station_id')
            if station_ids is not None:
                station_ids = station_ids.to(device)
            
            optimizer.zero_grad()
            model.set_epoch(epoch)
            losses = model.compute_loss(X, y, station_ids)
            loss = losses['total']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.clamp_physics_parameters()
            
            epoch_losses.append(loss.item())
            epoch_physics_losses.append(losses.get('physics', torch.tensor(0.0)).item())
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                X = batch['X'].to(device)
                y = batch['y'].to(device)
                sid = batch.get('station_id')
                if sid is not None:
                    sid = sid.to(device)
                mean, var = model(X, sid)
                val_losses.append(gaussian_nll_loss(y, mean, var).item())
        
        avg_val_loss = np.mean(val_losses)
        scheduler.step(avg_val_loss)
        
        # Record snapshot
        if epoch % snapshot_interval == 0 or epoch == epochs - 1:
            with torch.no_grad():
                r_val = physics.get_r().item()
                m_val = physics.get_m().item()
                
                for site_id in range(physics.n_sites):
                    site_tensor = torch.tensor([site_id], device=device)
                    mu_max_val = physics.get_mu_max(site_tensor).item()
                    
                    snapshot = {
                        'Epoch': epoch,
                        'Site_ID': site_id,
                        'Site_Name': SITE_NAMES.get(site_id, f'Site_{site_id}'),
                        'mu_max': mu_max_val,
                        'r': r_val,
                        'm': m_val,
                        'K_N': physics.get_K_N().item(),
                        'K_P': physics.get_K_P().item(),
                        'net_growth_max': mu_max_val - r_val - m_val,
                        'Train_Loss': np.mean(epoch_losses),
                        'Physics_Loss': np.mean(epoch_physics_losses),
                        'Val_Loss': avg_val_loss,
                        'LR': optimizer.param_groups[0]['lr'],
                    }
                    dynamics.append(snapshot)
            
            if (epoch % (snapshot_interval * 5) == 0) or epoch == epochs - 1:
                print(f"  Epoch {epoch:3d}/{epochs} | "
                      f"Train: {np.mean(epoch_losses):.4f} | Val: {avg_val_loss:.4f} | "
                      f"μ_max[0]={dynamics[-physics.n_sites]['mu_max']:.4f} | "
                      f"r={r_val:.4f} | m={m_val:.4f}")
    
    # Save dynamics
    df_dynamics = pd.DataFrame(dynamics)
    dynamics_csv = output_dir / 'phase3_step3_dynamics.csv'
    df_dynamics.to_csv(dynamics_csv, index=False)
    
    # Summary: initial vs final
    print_header("PARAMETER EVOLUTION: Initial → Final")
    for site_id in range(physics.n_sites):
        site_data = df_dynamics[df_dynamics['Site_ID'] == site_id]
        initial = site_data.iloc[0]
        final = site_data.iloc[-1]
        
        print(f"  Site {site_id} ({SITE_NAMES.get(site_id, '?')}):")
        print(f"    μ_max: {initial['mu_max']:.4f} → {final['mu_max']:.4f} "
              f"(Δ={final['mu_max'] - initial['mu_max']:+.4f})")
        print(f"    r:     {initial['r']:.4f} → {final['r']:.4f}")
        print(f"    m:     {initial['m']:.4f} → {final['m']:.4f}")
        print(f"    net:   {initial['net_growth_max']:.4f} → {final['net_growth_max']:.4f}")
    
    print(f"\nDynamics saved to: {dynamics_csv}")
    return df_dynamics


# ---- Step 4: Parameter Constraint Ablation ----

def run_step4(data, feature_names, backbone, seeds, epochs, output_dir):
    """
    Step 4: Compare parameter constraint methods.
    
    Three constraint conditions:
      - no_constraint:  K_N/K_P unconstrained (raw nn.Parameter + softplus)
      - sigmoid:        K_N/K_P Sigmoid-constrained to literature range (default)
      - penalty:        K_N/K_P with penalty loss for range violations
    
    Output: phase3_step4_constraint_ablation.csv
    """
    print_header(f"STEP 4: Parameter Constraint Ablation ({backbone})")
    
    from pinn_hab.physics.biogeochemical import EnhancedBiogeochemicalPhysics
    import random
    
    # Rerun fix: the original 'no_constraint' was a sigmoid on a widened range started at its centre
    # (K_N 2.505), and 'penalty' never applied its penalty. Now all three modes keep the literature ranges as reference,
    # start at their centre (K_N 0.255, K_P 0.0255 mg/L) and differ only in how they bound K_N and K_P.
    constraint_modes = {
        'sigmoid': {'label': 'Sigmoid Hard Constraint (literature bounds)', 'mode': 'sigmoid'},
        'no_constraint': {'label': 'No Constraint (positive, centre x exp)', 'mode': 'none'},
        'penalty': {'label': 'Penalty Loss (soft constraint, lambda 10)', 'mode': 'soft', 'penalty_lambda': 10.0},
    }
    
    # ---- Resume: load existing results ----
    csv_path = output_dir / 'phase3_step4_constraint_ablation.csv'
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        results = df_existing.to_dict('records')
        completed = set(zip(df_existing['Constraint_Mode'], df_existing['Seed']))
        print(f"  [RESUME] Loaded {len(results)} existing results, {len(completed)} completed")
    else:
        results = []
        completed = set()
    
    for mode_name, mode_cfg in constraint_modes.items():
        for seed in seeds:
            if (mode_name, seed) in completed:
                print(f"\n--- {mode_name}: {mode_cfg['label']} | Seed={seed} --- [SKIP: already completed]")
                continue
            
            set_seed(seed)
            random.seed(seed)
            
            print(f"\n--- {mode_name}: {mode_cfg['label']} | Seed={seed} ---")
            
            # Create model with modified param ranges
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
            
            # Constraint mode on the physics module (rerun fix)
            if hasattr(model, 'physics'):
                model.physics.set_constraint_mode(mode_cfg['mode'], mode_cfg.get('penalty_lambda', 10.0))
            
            # Create loaders
            train_loader, val_loader, test_loader, _ = create_loaders(
                data, batch_size=TRAIN_CONFIG['batch_size']
            )
            
            # Train
            train_cfg = {
                'epochs': epochs,
                'learning_rate': TRAIN_CONFIG['learning_rate'],
                'weight_decay': TRAIN_CONFIG['weight_decay'],
                'patience': TRAIN_CONFIG['patience'],
            }
            
            save_path = output_dir / f"step4_{mode_name}_seed{seed}.pth"
            
            t_start = time.time()
            train_result = train_model(
                model, train_loader, val_loader,
                train_cfg, save_path=str(save_path),
                verbose=False,
            )
            t_elapsed = time.time() - t_start
            
            # Evaluate
            model.eval()
            model.to(device)
            all_preds, all_targets = [], []
            with torch.no_grad():
                for batch in test_loader:
                    X = batch['X'].to(device)
                    y = batch['y']
                    sid = batch.get('station_id')
                    if sid is not None:
                        sid = sid.to(device)
                    mean, var = model(X, sid)
                    all_preds.append(mean.cpu().numpy())
                    all_targets.append(y.numpy())
            
            y_pred = np.concatenate(all_preds)
            y_true = np.concatenate(all_targets)
            metrics = compute_metrics(y_true, y_pred)
            
            # Extract learned parameters
            physics = model.physics
            K_N_val = physics.get_K_N().item()
            K_P_val = physics.get_K_P().item()
            r_val = physics.get_r().item()
            m_val = physics.get_m().item()
            
            # Check literature violations
            K_N_in_lit = 0.01 <= K_N_val <= 0.5
            K_P_in_lit = 0.001 <= K_P_val <= 0.05
            
            row = {
                'Constraint_Mode': mode_name,
                'Constraint_Label': mode_cfg['label'],
                'Seed': seed,
                'K_N': round(K_N_val, 4),
                'K_P': round(K_P_val, 4),
                'K_N_In_Literature': K_N_in_lit,
                'K_P_In_Literature': K_P_in_lit,
                'r': round(r_val, 4),
                'm': round(m_val, 4),
                **metrics,
                'Best_Val_Loss': train_result['best_val_loss'],
                'Final_Epoch': train_result['final_epoch'],
                'Train_Time_s': round(t_elapsed, 1),
            }
            results.append(row)
            
            # Save incrementally
            pd.DataFrame(results).to_csv(csv_path, index=False)
            
            lit_status = f"K_N={'[OK]' if K_N_in_lit else '[OUT]'} K_P={'[OK]' if K_P_in_lit else '[OUT]'}"
            print(f"  K_N={K_N_val:.4f} K_P={K_P_val:.4f} | {lit_status}")
            print(f"  R²={metrics['R²']:.4f} | RMSE={metrics['RMSE']:.4f} | Time={t_elapsed:.1f}s")
    
    # Final save
    df = pd.DataFrame(results)
    
    # Summary
    print_header("STEP 4 CONSTRAINT ABLATION SUMMARY")
    summary = df.groupby('Constraint_Mode').agg({
        'R²': ['mean', 'std'],
        'RMSE': ['mean', 'std'],
        'K_N': ['mean', 'std'],
        'K_P': ['mean', 'std'],
        'K_N_In_Literature': 'mean',
        'K_P_In_Literature': 'mean',
    }).round(4)
    print(summary.to_string())
    print(f"\nResults saved to: {csv_path}")
    
    return df


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(
        description='Phase 3: XAI & Ecological Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4], default=None,
                        help='Run specific step (default: all)')
    parser.add_argument('--backbone', type=str, default=None,
                        help='Override backbone (default: Phase 1 selection)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs (default: 100)')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS,
                        help='Random seeds (default: 42 123 456)')
    parser.add_argument('--output-dir', type=str, default='experiments/phase3',
                        help='Output directory')
    parser.add_argument('--phase1-dir', type=str, default='experiments/phase1',
                        help='Phase 1 results directory (for checkpoint loading)')
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phase1_dir = Path(args.phase1_dir)
    
    backbone = get_backbone(args.backbone, phase1_dir)
    
    print_header("Five-station rerun - Phase 3: XAI & Ecological Validation")
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
    print(f"  Test:  {data['X_test'].shape[0]} samples")
    print(f"  Features: {len(feature_names)}")
    
    # Save experiment config
    exp_config = {
        'backbone': backbone,
        'epochs': args.epochs,
        'seeds': args.seeds,
        'literature_ranges': {k: {'range': v['range'], 'unit': v['unit']}
                              for k, v in LITERATURE_RANGES.items()},
        'model_defaults': MODEL_DEFAULTS,
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
    
    if args.step is None or args.step == 3:
        run_step3(data, feature_names, backbone, args.seeds[0], args.epochs, output_dir)
    
    if args.step is None or args.step == 4:
        run_step4(data, feature_names, backbone, args.seeds, args.epochs, output_dir)
    
    print_header("PHASE 3 COMPLETE")
    print(f"All results saved to: {output_dir}/")


if __name__ == '__main__':
    main()
