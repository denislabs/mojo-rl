"""Sheaf World Model with holonomy as an observable (SWM-H).

Research code. See `docs/SHEAF_WORLD_MODELS_V2.md` for the design and
`docs/SWM_IMPLEMENTATION_PLAN.md` for the phasing and the gates. Nothing
outside `mojo_rl/experimental/` should import from here.
"""

from .so_d import SqMat, skew_from_vector, cayley, expm_skew, householder
from .procrustes import (
    PairBatch,
    cross_covariance,
    polar_orthogonal_factor,
    procrustes_o_d,
    mean_squared_residual,
)
from .place_graph import (
    Edge,
    PlaceGraph,
    EDGE_ACTION,
    EDGE_INTERMODAL,
    EDGE_IDENTIFICATION,
)
from .sheaf_laplacian import (
    DenseSym,
    build_sheaf_laplacian,
    eigenvalues_ascending,
    kernel_dimension,
)
from .reference_io import (
    RefRow,
    load_reference,
    ref_scalar,
    ref_int,
    ref_indexed,
    ref_vector,
    ref_count,
)
