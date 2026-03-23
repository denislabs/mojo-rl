# =============================================================================
# Optimizer Trait
# =============================================================================

from ..constants import dtype
from layout import Layout, LayoutTensor
from std.gpu.host import DeviceContext, DeviceBuffer


trait Optimizer(Movable & ImplicitlyCopyable):
    """Base trait for optimizers.

    Optimizers are stateless pure-computation types. All mutable state
    (parameter moments, etc.) is passed externally via LayoutTensor views.
    Hyperparameters (lr, beta1, etc.) are compile-time struct parameters.

    STATE_PER_PARAM defines how many state values are needed per parameter:
    - SGD: 1 (unused, but minimum for valid tensor dimensions)
    - Adam: 2 (m = first moment, v = second moment)
    - RMSprop: 1 (squared gradient moving average)

    step() and step_gpu() are @staticmethod - no instance needed.
    The caller tracks step_num and passes it in (used for bias correction).
    """

    comptime STATE_PER_PARAM: Int

    @staticmethod
    def step[
        PARAM_SIZE: Int
    ](
        mut params: LayoutTensor[
            dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
        ],
        grads: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
        mut state: LayoutTensor[
            dtype,
            Layout.row_major(PARAM_SIZE, Self.STATE_PER_PARAM),
            MutAnyOrigin,
        ],
        step_num: Int,
        lr_scale: Float64 = 1.0,
    ):
        """Perform one optimization step.

        Args:
            params: Flattened parameters to update (modified in place).
            grads: Flattened gradients.
            state: Optimizer state (e.g., moments). Layout: (PARAM_SIZE, STATE_PER_PARAM).
            step_num: Global step counter (1-based). Used for bias correction in Adam/AdamW.
            lr_scale: Multiplicative LR scale (default 1.0). Set < 1.0 for LR annealing.
        """
        ...

    # =========================================================================
    # GPU methods
    # =========================================================================

    @staticmethod
    def step_gpu[
        PARAM_SIZE: Int
    ](
        ctx: DeviceContext,
        mut params: LayoutTensor[
            dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin
        ],
        grads: LayoutTensor[dtype, Layout.row_major(PARAM_SIZE), MutAnyOrigin],
        mut state: LayoutTensor[
            dtype,
            Layout.row_major(PARAM_SIZE, Self.STATE_PER_PARAM),
            MutAnyOrigin,
        ],
        step_num: Int,
        lr_scale: Float64 = 1.0,
    ) raises:
        """Perform one optimization step on GPU.

        Args:
            ctx: GPU device context.
            params: Parameters [PARAM_SIZE] (modified in place).
            grads: Gradients [PARAM_SIZE].
            state: Optimizer state [PARAM_SIZE, STATE_PER_PARAM].
            step_num: Global step counter (1-based).
            lr_scale: Multiplicative LR scale (default 1.0). Set < 1.0 for LR annealing.
        """
        ...
