"""
Monod biogeochemical rate equations, soft ODE residual losses and wrapper classes
for the earlier five-station PINN setup.
"""

from .biogeochemical import (
    EnhancedBiogeochemicalPhysics,
    BiogeochemicalPhysics,  # Alias
    PHYSICS_FEATURE_PATTERNS,
    DEFAULT_DENORM_PARAMS
)
from .loss import (
    EnhancedPhysicsInformedLoss,
    PhysicsInformedLoss  # Alias
)
from .wrapper import (
    EnhancedPhysicsInformedWrapper,
    PhysicsInformedWrapper  # Alias
)

__all__ = [
    # Main classes
    'EnhancedBiogeochemicalPhysics',
    'EnhancedPhysicsInformedLoss',
    'EnhancedPhysicsInformedWrapper',
    # Aliases for backward compatibility
    'BiogeochemicalPhysics',
    'PhysicsInformedLoss',
    'PhysicsInformedWrapper',
    # Constants
    'PHYSICS_FEATURE_PATTERNS',
    'DEFAULT_DENORM_PARAMS',
]
