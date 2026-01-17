"""Physics engine trait definitions.

This module exports the core traits that define the physics engine architecture:
- Integrator: Velocity and position integration methods
- CollisionSystem: Collision detection algorithms
- ConstraintSolver: Contact and joint constraint resolution
"""

from .integrator import Integrator
from .collision import CollisionSystem
from .solver import ConstraintSolver
