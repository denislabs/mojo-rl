"""Traits for 3D physics components.

Defines interfaces for integrators, solvers, and collision detectors
that can be swapped for different implementations.
"""

from .integrator3d import Integrator3D
from .solver3d import ConstraintSolver3D
