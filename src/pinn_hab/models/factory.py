"""
Model factory for creating and configuring prediction models.
"""

import torch.nn as nn
from typing import Dict, List, Optional, Type

from .itransformer import iTransformer
from .mamba import MambaForecaster
from .xlstm import xLSTMForecaster
from .modern_tcn import ModernTCNForecaster


# Model registry
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    'iTransformer': iTransformer,
    'Mamba': MambaForecaster,
    'xLSTM': xLSTMForecaster,
    'ModernTCN': ModernTCNForecaster,
}


def get_model(
    config: Dict,
    n_features: int,
    n_stations: int = 5,
    feature_names: Optional[List[str]] = None
) -> nn.Module:
    """
    Factory function to create model based on configuration.
    
    Supported models:
        - iTransformer: Transformer with attention on variable dimension
        - Mamba: Selective State Space Model
        - xLSTM: Extended LSTM with exponential gating and matrix memory
        - ModernTCN: Large-kernel depthwise-separable TCN
        - *_PINN: Any model wrapped with physics-informed loss
    
    Args:
        config: Configuration dictionary with model parameters
        n_features: Number of input features
        n_stations: Number of stations for embedding
        feature_names: List of feature names (for PINN models)
    
    Returns:
        Configured model instance
    """
    model_type = config.get('model_type', 'iTransformer')
    seq_len = config.get('seq_len', 30)
    hidden_dim = config.get('hidden_dim', 64)
    n_layers = config.get('n_layers', 2)
    n_heads = config.get('n_heads', 4)
    dropout_rate = config.get('dropout_rate', 0.3)
    use_station_embedding = config.get('use_station_embedding', True)
    
    # Check for PINN wrapper
    use_pinn = '_PINN' in model_type
    base_model_type = model_type.replace('_PINN', '')
    
    # Get base model class
    if base_model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {base_model_type}. "
                        f"Available: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[base_model_type]
    
    # Common args for all models
    common_args = dict(
        seq_len=seq_len,
        n_features=n_features,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        n_stations=n_stations,
        use_station_embedding=use_station_embedding,
    )
    
    # Model-specific args
    if base_model_type == 'iTransformer':
        base_model = model_class(
            **common_args,
            n_heads=n_heads,
            n_layers=n_layers,
        )
    elif base_model_type == 'Mamba':
        base_model = model_class(
            **common_args,
            n_layers=n_layers,
            d_state=config.get('d_state', 16),
            d_conv=config.get('d_conv', 4),
            expand=config.get('expand', 2),
        )
    elif base_model_type == 'xLSTM':
        base_model = model_class(
            **common_args,
            n_layers=n_layers,
            head_dim=config.get('head_dim', 16),
        )
    elif base_model_type == 'ModernTCN':
        base_model = model_class(
            **common_args,
            n_layers=n_layers,
            kernel_size=config.get('kernel_size', 51),
            expand_ratio=config.get('expand_ratio', 4),
        )
    else:
        # Generic fallback
        base_model = model_class(**common_args)
    
    # Wrap with physics-informed module if requested
    if use_pinn:
        from ..physics.wrapper import EnhancedPhysicsInformedWrapper
        from ..physics.biogeochemical import EnhancedBiogeochemicalPhysics
        
        # Create physics module
        physics_module = EnhancedBiogeochemicalPhysics(
            feature_names=feature_names or [],
            n_sites=n_stations,
            freeze_params=config.get('freeze_physics_params', False)
        )
        
        # Wrap model with enhanced adaptive weighting support
        model = EnhancedPhysicsInformedWrapper(
            base_model=base_model,
            physics_module=physics_module,
            w_physics_min=config.get('w_physics_min', 0.01),
            w_physics_max=config.get('w_physics_max', 0.3),
            curriculum_epochs=config.get('curriculum_epochs', 70),
            warmup_epochs=config.get('warmup_epochs', 10),
            use_gradient_balancing=config.get('use_gradient_balancing', True),
            use_collocation=config.get('use_collocation', True),
            n_collocation_points=config.get('n_collocation_points', 20),
            net_growth_threshold=config.get('net_growth_threshold', -0.1),
            net_growth_penalty_weight=config.get('net_growth_penalty_weight', 1.0),
            n_train_samples=config.get('n_train_samples', None),
            use_adaptive_physics_weight=config.get('use_adaptive_physics_weight', True),
            adaptive_weighting_method=config.get('adaptive_weighting_method', 'relobralo'),
            base_n_samples=config.get('base_n_samples', 10000)
        )
        
        print(f"Created {model_type} with Physics-Informed wrapper")
        return model
    
    print(f"Created {model_type} model")
    return base_model


def register_model(name: str, model_class: Type[nn.Module]):
    """
    Register a custom model class.
    
    Args:
        name: Model name for registry
        model_class: Model class to register
    """
    MODEL_REGISTRY[name] = model_class
    print(f"Registered model: {name}")
