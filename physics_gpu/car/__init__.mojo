"""CarRacing GPU physics module.

This module provides GPU-accelerated physics for CarRacing environments,
implementing slip-based tire friction physics for top-down car simulation.

Unlike the impulse-based collision physics used for LunarLander, CarRacing
uses a fundamentally different physics model:
- Slip-based tire friction (not impulse-based collision)
- Top-down view with no gravity (vs side-view with gravity)
- 1 hull + 4 derived wheel positions (not independent rigid bodies)
- Friction zone lookup from track tiles (not collision response)

Architecture:
- CarRacingLayout: Compile-time state layout for batched simulation
- WheelFriction: Slip-based tire friction model
- TileCollision: Point-in-quad friction zone lookup
- CarDynamics: Full car simulation step
- CarPhysicsKernel: Fused GPU kernel for physics step
"""

# Physics constants from Gymnasium car_dynamics.py
from .constants import (
    # Scale factor
    SIZE,
    # Engine and brake
    ENGINE_POWER,
    BRAKE_FORCE,
    # Wheel properties
    WHEEL_MOMENT_OF_INERTIA,
    WHEEL_RADIUS,
    WHEEL_WIDTH,
    # Friction
    FRICTION_LIMIT,
    FRICTION_COEF,
    ROAD_FRICTION,
    GRASS_FRICTION,
    # Steering
    STEERING_LIMIT,
    STEERING_MOTOR_SPEED,
    # Wheel positions (local coords)
    WHEEL_POS_FL_X,
    WHEEL_POS_FL_Y,
    WHEEL_POS_FR_X,
    WHEEL_POS_FR_Y,
    WHEEL_POS_RL_X,
    WHEEL_POS_RL_Y,
    WHEEL_POS_RR_X,
    WHEEL_POS_RR_Y,
    # Hull properties
    HULL_MASS,
    HULL_INERTIA,
    # Time step
    CAR_DT,
)

# Compile-time layout for state buffers
from .layout import CarRacingLayout

# Slip-based tire friction model
from .wheel_friction import WheelFriction

# Track tile collision for friction lookup
from .tile_collision import TileCollision

# Full car dynamics simulation
from .car_dynamics import CarDynamics

# Fused GPU kernel
from .car_kernel import CarPhysicsKernel
