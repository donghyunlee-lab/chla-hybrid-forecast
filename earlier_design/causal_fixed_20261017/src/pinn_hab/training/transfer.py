"""Transfer learning utilities for cross-site fine-tuning, parameter freezing and sparse target sampling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum


class FreezingStrategy(Enum):
    """Freezing strategies for transfer learning."""
    NONE = "none"                    # No freezing
    FULL = "full"                    # Freeze everything except head
    LINEAR_PROBING = "linear"        # Freeze all, only train head
    PHYSICS_AWARE = "physics"        # Keep physics module trainable
    GRADUAL = "gradual"              # Gradually unfreeze layers


class SamplingStrategy(Enum):
    """Sparse sampling strategies for target domain."""
    DAILY = 1
    WEEKLY = 7
    BIWEEKLY = 14
    MONTHLY = 30
    QUARTERLY = 90
    RANDOM = -1  # Random sampling


class TransferLearningConfig:
    """Configuration for transfer learning."""
    
    def __init__(
        self,
        source_sites: List[int] = [0, 1, 2, 3],
        target_site: int = 4,
        pretrain_epochs: int = 100,
        finetune_epochs: int = 50,
        pretrain_lr: float = 0.001,
        finetune_lr: float = 0.0001,
        freezing_strategy: FreezingStrategy = FreezingStrategy.PHYSICS_AWARE,
        sampling_strategy: SamplingStrategy = SamplingStrategy.MONTHLY,
        sampling_ratio: float = 1.0,  # For random sampling
        use_domain_adaptation: bool = False,
        domain_adaptation_weight: float = 0.1,
        gradual_unfreeze_epochs: int = 10,
        layer_wise_lr_decay: float = 0.9
    ):
        self.source_sites = source_sites
        self.target_site = target_site
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.pretrain_lr = pretrain_lr
        self.finetune_lr = finetune_lr
        self.freezing_strategy = freezing_strategy
        self.sampling_strategy = sampling_strategy
        self.sampling_ratio = sampling_ratio
        self.use_domain_adaptation = use_domain_adaptation
        self.domain_adaptation_weight = domain_adaptation_weight
        self.gradual_unfreeze_epochs = gradual_unfreeze_epochs
        self.layer_wise_lr_decay = layer_wise_lr_decay


class DomainAdaptationLoss(nn.Module):
    """
    Domain Adaptation Loss using Maximum Mean Discrepancy (MMD).
    
    Aligns source and target domain feature distributions.
    """
    
    def __init__(self, kernel: str = 'rbf', kernel_bandwidth: float = 1.0):
        super().__init__()
        self.kernel = kernel
        self.kernel_bandwidth = kernel_bandwidth
    
    def rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Radial Basis Function kernel."""
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        yy = torch.sum(y ** 2, dim=1, keepdim=True)
        xy = torch.mm(x, y.t())
        dist = xx + yy.t() - 2 * xy
        return torch.exp(-dist / (2 * self.kernel_bandwidth ** 2))
    
    def compute_mmd(
        self, 
        source_features: torch.Tensor, 
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Maximum Mean Discrepancy.
        
        MMD = E[k(xs, xs')] - 2E[k(xs, xt)] + E[k(xt, xt')]
        """
        n_source = source_features.shape[0]
        n_target = target_features.shape[0]
        
        if n_source == 0 or n_target == 0:
            return torch.tensor(0.0, device=source_features.device)
        
        # Compute kernel matrices
        K_ss = self.rbf_kernel(source_features, source_features)
        K_tt = self.rbf_kernel(target_features, target_features)
        K_st = self.rbf_kernel(source_features, target_features)
        
        # MMD computation
        mmd = (K_ss.sum() / (n_source * n_source) - 
               2 * K_st.sum() / (n_source * n_target) + 
               K_tt.sum() / (n_target * n_target))
        
        return mmd
    
    def forward(
        self, 
        source_features: torch.Tensor, 
        target_features: torch.Tensor
    ) -> torch.Tensor:
        return self.compute_mmd(source_features, target_features)


class DynamicSiteEmbedding(nn.Module):
    """
    Dynamic Site Embedding that can expand to new sites.
    
    Supports:
    - Adding new sites without retraining
    - Prototype-based initialization for new sites
    - Site similarity computation
    """
    
    def __init__(
        self, 
        n_initial_sites: int, 
        embedding_dim: int,
        max_sites: int = 20
    ):
        super().__init__()
        
        self.n_sites = n_initial_sites
        self.embedding_dim = embedding_dim
        self.max_sites = max_sites
        
        # Main embedding
        self.embedding = nn.Embedding(max_sites, embedding_dim)
        
        # Initialize with small values
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        
        # Site prototypes (for similarity-based init)
        self.register_buffer('site_prototypes', torch.zeros(max_sites, embedding_dim))
        self.register_buffer('site_counts', torch.zeros(max_sites))
    
    def forward(self, site_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings for site IDs."""
        return self.embedding(site_ids)
    
    def add_new_site(
        self, 
        site_id: int, 
        similar_sites: Optional[List[int]] = None,
        init_method: str = 'mean'
    ):
        """
        Add a new site with intelligent initialization.
        
        Args:
            site_id: ID for new site
            similar_sites: List of similar existing site IDs
            init_method: 'mean', 'nearest', or 'random'
        """
        if site_id >= self.max_sites:
            raise ValueError(f"Site ID {site_id} exceeds max sites {self.max_sites}")
        
        with torch.no_grad():
            if init_method == 'mean' and similar_sites:
                # Initialize as mean of similar sites
                similar_embeds = self.embedding.weight[similar_sites]
                new_embed = similar_embeds.mean(dim=0)
            elif init_method == 'nearest' and similar_sites:
                # Initialize as nearest neighbor
                new_embed = self.embedding.weight[similar_sites[0]]
            else:
                # Random initialization
                new_embed = torch.randn(self.embedding_dim) * 0.1
            
            self.embedding.weight[site_id] = new_embed
        
        self.n_sites = max(self.n_sites, site_id + 1)
        print(f"[Transfer] Added new site {site_id}, total sites: {self.n_sites}")
    
    def update_prototype(self, site_id: int, features: torch.Tensor):
        """Update site prototype with running mean."""
        with torch.no_grad():
            self.site_counts[site_id] += 1
            n = self.site_counts[site_id]
            
            # Running mean update
            self.site_prototypes[site_id] = (
                (n - 1) / n * self.site_prototypes[site_id] + 
                1 / n * features.mean(dim=0)
            )
    
    def get_site_similarity(self, site_a: int, site_b: int) -> float:
        """Compute cosine similarity between site embeddings."""
        embed_a = self.embedding.weight[site_a]
        embed_b = self.embedding.weight[site_b]
        
        similarity = F.cosine_similarity(embed_a.unsqueeze(0), embed_b.unsqueeze(0))
        return similarity.item()


class TransferLearningManager:
    """
    Manager for transfer learning with advanced features.
    
    Handles:
    - Sparse sampling with various strategies
    - Freezing/unfreezing with multiple strategies
    - Domain adaptation
    - Gradual unfreezing
    """
    
    def __init__(self, config: TransferLearningConfig):
        self.config = config
        self.da_loss = DomainAdaptationLoss() if config.use_domain_adaptation else None
        self.current_phase = "pretrain"
        self.unfreeze_schedule = []
    
    def get_sparse_indices(
        self, 
        all_indices: np.ndarray, 
        strategy: SamplingStrategy = None,
        ratio: float = None
    ) -> np.ndarray:
        """
        Get sparse indices based on sampling strategy.
        
        Args:
            all_indices: All available indices
            strategy: Sampling strategy (uses config default if None)
            ratio: Sampling ratio for random strategy
        
        Returns:
            Sparse indices array
        """
        strategy = strategy or self.config.sampling_strategy
        ratio = ratio or self.config.sampling_ratio
        n = len(all_indices)
        
        if strategy == SamplingStrategy.RANDOM:
            n_samples = int(n * ratio)
            return np.random.choice(all_indices, n_samples, replace=False)
        else:
            interval = strategy.value
            return all_indices[::interval]
    
    def apply_freezing(
        self, 
        model: nn.Module, 
        strategy: FreezingStrategy = None,
        physics_module_name: str = 'physics'
    ) -> Dict[str, bool]:
        """
        Apply freezing strategy to model.
        
        Returns dict mapping parameter names to frozen status.
        """
        strategy = strategy or self.config.freezing_strategy
        frozen_params = {}
        
        if strategy == FreezingStrategy.NONE:
            # No freezing
            for name, param in model.named_parameters():
                param.requires_grad = True
                frozen_params[name] = False
        
        elif strategy == FreezingStrategy.FULL:
            # Freeze all except head
            for name, param in model.named_parameters():
                if any(x in name.lower() for x in ['head', 'output', 'fc', 'projection']):
                    param.requires_grad = True
                    frozen_params[name] = False
                else:
                    param.requires_grad = False
                    frozen_params[name] = True
        
        elif strategy == FreezingStrategy.LINEAR_PROBING:
            # Freeze everything, only train final linear layer
            for name, param in model.named_parameters():
                if 'head.mean_head' in name or 'head.var_head' in name:
                    param.requires_grad = True
                    frozen_params[name] = False
                else:
                    param.requires_grad = False
                    frozen_params[name] = True
        
        elif strategy == FreezingStrategy.PHYSICS_AWARE:
            # Freeze base model, keep physics and head trainable
            for name, param in model.named_parameters():
                if any(x in name.lower() for x in [physics_module_name, 'head', 'output', 'physics']):
                    param.requires_grad = True
                    frozen_params[name] = False
                    print(f"  [Unfreeze] {name}")
                else:
                    param.requires_grad = False
                    frozen_params[name] = True
        
        elif strategy == FreezingStrategy.GRADUAL:
            # Initially freeze all, will be gradually unfrozen
            for name, param in model.named_parameters():
                param.requires_grad = False
                frozen_params[name] = True
            
            # Only unfreeze head initially
            for name, param in model.named_parameters():
                if any(x in name.lower() for x in ['head', 'output']):
                    param.requires_grad = True
                    frozen_params[name] = False
            
            # Build unfreeze schedule
            self._build_unfreeze_schedule(model)
        
        # Count frozen/unfrozen
        n_frozen = sum(frozen_params.values())
        n_total = len(frozen_params)
        print(f"  [Freezing] {n_frozen}/{n_total} parameters frozen ({strategy.value})")
        
        return frozen_params
    
    def _build_unfreeze_schedule(self, model: nn.Module):
        """Build schedule for gradual unfreezing."""
        # Group layers by depth (simplified)
        layer_groups = []
        current_group = []
        
        for name, param in model.named_parameters():
            if 'layer' in name.lower() or 'block' in name.lower():
                # Extract layer number
                import re
                match = re.search(r'layer[s]?[._]?(\d+)', name.lower())
                if match:
                    layer_num = int(match.group(1))
                    while len(layer_groups) <= layer_num:
                        layer_groups.append([])
                    layer_groups[layer_num].append(name)
                else:
                    current_group.append(name)
            else:
                current_group.append(name)
        
        if current_group:
            layer_groups.insert(0, current_group)
        
        # Reverse order (unfreeze from top to bottom)
        self.unfreeze_schedule = layer_groups[::-1]
    
    def gradual_unfreeze_step(
        self, 
        model: nn.Module, 
        epoch: int
    ) -> List[str]:
        """
        Perform gradual unfreezing based on epoch.
        
        Returns list of newly unfrozen parameter names.
        """
        if not self.unfreeze_schedule:
            return []
        
        epochs_per_group = max(1, self.config.gradual_unfreeze_epochs // len(self.unfreeze_schedule))
        group_idx = min(epoch // epochs_per_group, len(self.unfreeze_schedule) - 1)
        
        newly_unfrozen = []
        
        for i in range(group_idx + 1):
            for param_name in self.unfreeze_schedule[i]:
                for name, param in model.named_parameters():
                    if name == param_name and not param.requires_grad:
                        param.requires_grad = True
                        newly_unfrozen.append(name)
        
        return newly_unfrozen
    
    def get_layer_wise_lr(
        self, 
        model: nn.Module, 
        base_lr: float
    ) -> List[Dict]:
        """
        Get layer-wise learning rates with decay.
        
        Deeper layers get smaller learning rates.
        """
        param_groups = []
        decay = self.config.layer_wise_lr_decay
        
        # Group parameters by layer
        layer_params = {}
        other_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            import re
            match = re.search(r'layer[s]?[._]?(\d+)', name.lower())
            if match:
                layer_num = int(match.group(1))
                if layer_num not in layer_params:
                    layer_params[layer_num] = []
                layer_params[layer_num].append(param)
            else:
                other_params.append(param)
        
        # Assign LRs with decay
        max_layer = max(layer_params.keys()) if layer_params else 0
        
        for layer_num in sorted(layer_params.keys()):
            depth = max_layer - layer_num
            lr = base_lr * (decay ** depth)
            param_groups.append({
                'params': layer_params[layer_num],
                'lr': lr,
                'name': f'layer_{layer_num}'
            })
        
        # Other params get base LR
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': base_lr,
                'name': 'other'
            })
        
        return param_groups
    
    def compute_domain_loss(
        self, 
        source_features: torch.Tensor, 
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute domain adaptation loss."""
        if self.da_loss is None:
            return torch.tensor(0.0)
        
        return self.da_loss(source_features, target_features)
    
    def switch_to_finetune(self, model: nn.Module):
        """Switch from pretrain to finetune phase."""
        self.current_phase = "finetune"
        self.apply_freezing(model, self.config.freezing_strategy)
        print("[Transfer] Switched to fine-tuning phase")


def create_transfer_config(
    target_site: int = 4,
    source_sites: List[int] = None,
    sparse_mode: str = 'monthly',
    freeze_mode: str = 'physics',
    use_domain_adaptation: bool = False
) -> TransferLearningConfig:
    """
    Create transfer learning config from simple parameters.
    
    Args:
        target_site: Target site ID for fine-tuning
        source_sites: Source site IDs for pre-training
        sparse_mode: 'daily', 'weekly', 'monthly', 'quarterly', 'random'
        freeze_mode: 'none', 'full', 'linear', 'physics', 'gradual'
        use_domain_adaptation: Enable MMD-based domain adaptation
    """
    sampling_map = {
        'daily': SamplingStrategy.DAILY,
        'weekly': SamplingStrategy.WEEKLY,
        'biweekly': SamplingStrategy.BIWEEKLY,
        'monthly': SamplingStrategy.MONTHLY,
        'quarterly': SamplingStrategy.QUARTERLY,
        'random': SamplingStrategy.RANDOM
    }
    
    freeze_map = {
        'none': FreezingStrategy.NONE,
        'full': FreezingStrategy.FULL,
        'linear': FreezingStrategy.LINEAR_PROBING,
        'physics': FreezingStrategy.PHYSICS_AWARE,
        'gradual': FreezingStrategy.GRADUAL
    }
    
    return TransferLearningConfig(
        source_sites=source_sites or [0, 1, 2, 3],
        target_site=target_site,
        sampling_strategy=sampling_map.get(sparse_mode, SamplingStrategy.MONTHLY),
        freezing_strategy=freeze_map.get(freeze_mode, FreezingStrategy.PHYSICS_AWARE),
        use_domain_adaptation=use_domain_adaptation
    )
