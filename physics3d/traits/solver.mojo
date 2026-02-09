"""Constraint solver trait for Generalized Coordinates (GC) engine.

ConstraintSolver defines the interface for constraint-based contact solving
in joint space. Implementations receive the predicted (unconstrained) velocity
and modify it in-place to satisfy contact constraints.

This follows MuJoCo's approach: detect contacts in Cartesian space, compute
contact Jacobians via cdof (spatial motion axes per DOF), then iteratively
solve for contact impulses.
"""

from layout import LayoutTensor, Layout

from ..types import Model, Data


trait ConstraintSolver(Movable & ImplicitlyCopyable):
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
    fn solver_workspace_size[NV: Int, MAX_CONTACTS: Int]() -> Int:
        """Solver-specific workspace size in floats per environment.

        This workspace is allocated after M_inv (NV*NV) in the per-environment
        workspace buffer. Each solver declares how much device memory it needs
        for its MC-sized arrays (moved out of registers to reduce spilling).
        """
        ...

    @staticmethod
    fn solver_threads[
        NQ: Int,
        NV: Int,
        NBODY: Int,
        NJOINT: Int,
        MAX_CONTACTS: Int,
    ]() -> Int:
        """Number of threads for parallelization.

        This is the maximum number of threads that can be used for parallelization.
        """
        ...

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
        model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
        M_inv: InlineArray[Scalar[DTYPE], M_SIZE],
        cdof: InlineArray[Scalar[DTYPE], CDOF_SIZE],
        mut qvel: InlineArray[Scalar[DTYPE], V_SIZE],
        dt: Scalar[DTYPE],
    ):
        """Solve contact constraints on CPU.

        Args:
            model: Static model configuration.
            data: Mutable simulation state with contacts already detected.
                  Impulses are stored back for warm-starting next step.
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
        BATCH: Int,
        WS_SIZE: Int,
    ](
        state: LayoutTensor[
            DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        model: LayoutTensor[
            DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin
        ],
        workspace: LayoutTensor[
            DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
        ],
    ):
        """Solve contact constraints on GPU (per-environment).

        Args:
            state: GPU state buffer for all environments.
            model: GPU model buffer.
            workspace: Device memory workspace [BATCH, WS_SIZE] containing
                integrator temps, M_inv, and solver arrays per environment.
                Reads cdof and qvel_pred from workspace, modifies qvel_pred
                in-place to satisfy constraints.
        """
        ...
