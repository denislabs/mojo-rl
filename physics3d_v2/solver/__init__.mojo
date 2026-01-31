"""Physics3D v2 Constraint Solvers.

Two solver implementations:

1. ImpulseSolver (Bullet/Box2D style):
   - Split Impulse method
   - Separate velocity and position solving
   - Good for stable stacking

2. PGSSolver (MuJoCo style):
   - Projected Gauss-Seidel with spring-damper constraints
   - Soft contact dynamics
   - Configurable timeconst/dampratio/impedance
"""

from .gravity_solver import apply_gravity_gpu

# Impulse-based solver (Bullet/Box2D style)
from .impulse_solver import (
    solve_velocity_constraints,
    solve_position_constraints,
    solve_resting_contacts,
    solve_velocity_constraints_gpu,
    solve_position_constraints_gpu,
)

# PGS solver (MuJoCo style)
from .pgs_solver import (
    solve_constraints_pgs,
    solve_constraints_pgs_gpu,
    correct_positions,
    # Utility functions
    compute_spring_damper_params,
    compute_effective_mass,
    compute_reference_acceleration,
    compute_constraint_velocity,
    # Constants
    DEFAULT_TIMECONST,
    DEFAULT_DAMPRATIO,
    DEFAULT_IMPEDANCE,
)
