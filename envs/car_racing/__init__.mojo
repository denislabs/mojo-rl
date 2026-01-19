"""CarRacing environment module.

This module provides both CPU and GPU implementations of CarRacing:
- CarRacingEnv: Original CPU implementation (from car_racing_v1.mojo)
- CarRacingV2: GPU-accelerated implementation using physics_gpu/car/

Usage:
    from envs.car_racing import CarRacingEnv, CarRacingV2
"""

# V1 CPU implementation (original)
from .car_racing_v1 import CarRacingEnv, CarRacingState, CarRacingAction

# V2 GPU-enabled components
from .constants import CRConstants
from .state import CarRacingV2State
from .action import CarRacingV2Action
from .track import TrackTileV2, TrackGenerator
from .car_racing_v2 import CarRacingV2
