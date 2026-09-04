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
from .sheaf_inference import (
    InferenceConfig,
    frame_energy,
    frame_gradient,
    infer_frames,
    solve_frames_exact,
    frame_covariance_anisotropy,
)
from .observables import (
    gnc_weight,
    gnc_weights,
    GncSchedule,
    estimate_c_bar,
    classify,
    class_name,
    ClassificationLatch,
    confirm_by_independent_cycles,
    CLASS_NOMINAL,
    CLASS_ABERRANT,
    CLASS_OBSTRUCTION,
    CLASS_UNDECIDED,
    CLASS_CURVATURE_CONFIRMED,
)
from .planner import (
    FrameModel,
    PlannerConfig,
    Plan,
    plan,
    plan_exhaustive,
    plan_exhaustive_with_place_code,
    score_plan,
    MODEL_ORTHOGONAL,
    MODEL_TRANSLATION,
    MODEL_PLACE_LOOKUP,
    PLAN_FORWARD,
    PLAN_BACKWARD,
)
from .content import ContentChannel
from .so_d import fixed_subspace_dim
from .place_recognition import (
    PlaceMemory,
    RecogniserStats,
    score_recogniser,
    MATCH_NONE,
)
from .observables import (
    pairwise_consistent_in_group,
    closure_pair_composition,
    Z2Clique,
    maximal_clique_z2,
    CycleVerdict,
    classify_cycle,
)
from .world import SwmWorld
from .envs.klein_grid import KleinWorld, KleinWorldConfig
from .map_builder import (
    WalkRecord,
    OnlineLabeler,
    label_walk,
    count_labels,
    successor_conflicts,
    split_by_context,
    split_until_stable,
    MapScore,
    score_map,
    clone_graph,
    count_reversing,
)
from .cscg import Cscg, dense_labels
from .graph_planner import GraphPlan, plan_double_cover, nearest_centroid, clone_centroids
from .envs.pendulum_swm import PendulumSwm, PendulumSwmConfig
from .ablations import fit_free_lsq, orthogonality_defect
from .map_builder import MixtureFit, fit_transport_mixture, assignment_purity
from .envs.mobius_ring_nd import MobiusRingND, MobiusNDConfig, ACTION_FORWARD_ND
