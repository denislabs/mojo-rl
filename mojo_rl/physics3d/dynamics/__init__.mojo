"""Dynamics computation for Generalized Coordinates engine.

This module computes:
- Mass matrix M(q) using CRBA (Composite Rigid Body Algorithm)
- Bias forces C(q, qdot) + g(q) (Coriolis + gravity)
- RNE velocity derivative d(bias)/d(qvel) for implicit integration
- LU factorization for non-symmetric systems
"""

from .bias_forces import compute_bias_forces
# Legacy CRBA/LDL/CoM-Jacobian/invweight0 (`mass_matrix`, `jacobian`) +
# `velocity_derivatives`/`lu_factorization` were deleted at the fields sunset —
# the fields path uses mass_matrix_fields / ldl_fields / cdof_fields /
# invweight_fields / qderiv_fields / lu_fields.
from .cfrc_ext import compute_cfrc_ext
