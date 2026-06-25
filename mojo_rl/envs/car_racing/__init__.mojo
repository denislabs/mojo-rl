"""CarRacing environment module.

This module provides both CPU and GPU implementations of CarRacing:
- CarRacingEnv: Original CPU implementation (from car_racing_v1.mojo)
- CarRacing: GPU-accelerated implementation using physics2d/car/

Usage:
    from mojo_rl.envs.car_racing import CarRacingEnv, CarRacing
"""


from .constants import CRConstants
from .state import CarRacingState
from .action import CarRacingAction
from .track import TrackTile, TrackGenerator
from .car_racing import CarRacing
from .car_racing_mb import CarRacingMB
from .car_racing_discrete import CarRacingDiscrete
from .car_racing_pixel import CarRacingPixel
