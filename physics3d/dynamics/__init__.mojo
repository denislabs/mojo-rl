"""Dynamics computation for Generalized Coordinates engine.

This module computes:
- Mass matrix M(q) using CRBA (Composite Rigid Body Algorithm)
- Bias forces C(q, qdot) + g(q) (Coriolis + gravity)
"""

from .mass_matrix import compute_mass_matrix
from .bias_forces import compute_bias_forces
from .jacobian import compute_cdof, compute_contact_jacobian_row
