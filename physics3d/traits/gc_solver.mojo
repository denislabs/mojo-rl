"""Constraint solver trait for Generalized Coordinates (GC) engine.

GcConstraintSolver defines the interface for constraint-based contact solving
in joint space. Implementations receive the predicted (unconstrained) velocity
and modify it in-place to satisfy contact constraints.

This follows MuJoCo's approach: detect contacts in Cartesian space, compute
contact Jacobians via cdof (spatial motion axes per DOF), then iteratively
solve for contact impulses.
"""

from layout import LayoutTensor, Layout

from ..types import ModelGC, DataGC


trait GcConstraintSolver(Movable & ImplicitlyCopyable):
    """Trait for constraint-based contact solvers in GC engine.

    Solvers modify the predicted velocity in-place to satisfy contact
    constraints (non-penetration + friction).

    The solve() method receives:
    - model/data: physics model and current state (with contacts detected)
    - M_inv: full dense inverse mass matrix (NV×NV)
    - cdof: spatial motion axes per DOF (6 * NV floats)
    - qvel: predicted velocity (modified in-place to be constrained)
    - dt: timestep
    """

    @staticmethod
    fn solve[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        V_SIZE: Int,
        M_SIZE: Int,
        CDOF_SIZE: Int,
    ](
        model: ModelGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        data: DataGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        M_inv: InlineArray[Scalar[DTYPE], M_SIZE],
        cdof: InlineArray[Scalar[DTYPE], CDOF_SIZE],
        mut qvel: InlineArray[Scalar[DTYPE], V_SIZE],
        dt: Scalar[DTYPE],
    ):
        """Solve contact constraints on CPU.

        Args:
            model: Static model configuration.
            data: Simulation state with contacts already detected.
            M_inv: Full dense inverse mass matrix (NV×NV, row-major).
            cdof: Spatial motion axes per DOF (6*NV entries).
            qvel: Predicted velocity, modified in-place to satisfy constraints.
            dt: Timestep for Baumgarte stabilization.
        """
        ...

    @staticmethod
    fn solve_gpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
        STATE_SIZE: Int,
        MODEL_SIZE: Int,
        V_SIZE: Int,
        M_SIZE: Int,
        CDOF_SIZE: Int,
        BATCH: Int,
    ](
        env: Int,
        state: LayoutTensor[
            DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        model: LayoutTensor[
            DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ],
        M_inv: InlineArray[Scalar[DTYPE], M_SIZE],
        cdof: InlineArray[Scalar[DTYPE], CDOF_SIZE],
        mut qvel: InlineArray[Scalar[DTYPE], V_SIZE],
        dt: Scalar[DTYPE],
    ):
        """Solve contact constraints on GPU (per-environment).

        Args:
            env: Environment index within the batch.
            state: GPU state buffer for all environments.
            model: GPU model buffer.
            M_inv: Full dense inverse mass matrix (NV×NV, row-major).
            cdof: Spatial motion axes per DOF (6*NV entries).
            qvel: Predicted velocity, modified in-place to satisfy constraints.
            dt: Timestep for Baumgarte stabilization.
        """
        ...
