"""Dynamics computation for Generalized Coordinates engine.

This module computes:
- Mass matrix M(q) using CRBA (Composite Rigid Body Algorithm)
- Bias forces C(q, qdot) + g(q) (Coriolis + gravity)
- RNE velocity derivative d(bias)/d(qvel) for implicit integration
- LU factorization for non-symmetric systems
"""

from .mass_matrix import (
    compute_mass_matrix,
    compute_mass_matrix_full,
    ldl_factor,
    ldl_solve,
    compute_M_inv_from_ldl,
    compute_body_invweight0,
)
from .bias_forces import compute_bias_forces
from .jacobian import compute_cdof, compute_contact_jacobian_row, compute_composite_inertia
# Legacy slab `velocity_derivatives` + `lu_factorization` deleted at the P6
# sunset — the fields path uses `qderiv_fields` / `lu_fields`.
from .cfrc_ext import compute_cfrc_ext
