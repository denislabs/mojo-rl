"""GaussianLogProbOp: Gaussian log probability for continuous PPO (DiffOp).

Computes log probability of a continuous action under an unbounded Gaussian policy.
This is the continuous counterpart of CategoricalLogProbOp for discrete PPO.

Input layout:  [BATCH, 3 * action_dim] = [mean(A) || log_std(A) || action(A)]
Output layout: [BATCH, 1] = scalar log_prob per sample

  log_prob = sum_j(-0.5 * (log(2*pi) + 2*log_std_j + ((a_j - mean_j) / std_j)^2))

Cache layout:  [BATCH, 2 * action_dim] = [std_vals(A) || normalized_diff(A)]

Backward (vjp):
  grad_mean[j]    = grad_log_prob * (action_j - mean_j) / std_j^2
  grad_log_std[j] = grad_log_prob * (((action_j - mean_j) / std_j)^2 - 1)
  grad_action[j]  = 0  (frozen external input, no gradient)

Note: log_std is clamped to [-5, 2] for numerical stability.
"""

from ...constants import dtype, TPB
from ...autodiff.op import DiffOp, OpID
from layout import Layout, LayoutTensor
from std.math import exp, log
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext


comptime LOG_2PI: Float64 = 1.8378770664093453


struct GaussianLogProbOp[action_dim: Int](DiffOp):
    """Gaussian log probability for continuous PPO.

    Input: [BATCH, 3 * action_dim]
      = [mean(A) || log_std(A) || action(A)]

    Output: [BATCH, 1] = scalar log_prob per sample
      log_prob = sum_j(-0.5 * (log(2pi) + 2*log_std_j + ((a_j - mean_j) / std_j)^2))

    PARAM_SIZE = 0
    CACHE_SIZE = 2 * action_dim  (stores std_vals and normalized diff for backward)

    Backward:
      grad_mean[j] = grad_log_prob * (action_j - mean_j) / std_j^2
      grad_log_std[j] = grad_log_prob * (((action_j - mean_j) / std_j)^2 - 1)
      grad_action[j] = 0  (frozen external input, no gradient)

    Note: log_std is clamped to [-5, 2] for numerical stability.
    """

    comptime OP_ID: Int = OpID.USER_DEFINED._value + 9
    comptime IN_DIM: Int = 3 * Self.action_dim
    comptime OUT_DIM: Int = 1
    comptime PARAM_SIZE: Int = 0
    comptime CACHE_SIZE: Int = 2 * Self.action_dim  # [std(A) | normalized_diff(A)]
    comptime OP_WORKSPACE_PER_SAMPLE: Int = 0

    def __init__(out self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    # =========================================================================
    # CPU eval
    # =========================================================================

    @staticmethod
    def eval[
        BATCH: Int, dtype: DType = DType.float32
    ](
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        comptime A = Self.action_dim
        comptime LOG_STD_MIN: Float64 = -5.0
        comptime LOG_STD_MAX: Float64 = 2.0

        for b in range(BATCH):
            var total_log_prob: Float64 = 0.0

            for j in range(A):
                var mean = Float64(rebind[Scalar[dtype]](input[b, j]))
                var log_std = Float64(rebind[Scalar[dtype]](input[b, A + j]))
                var action = Float64(rebind[Scalar[dtype]](input[b, 2 * A + j]))

                # Clamp log_std for numerical stability
                if log_std < LOG_STD_MIN:
                    log_std = LOG_STD_MIN
                elif log_std > LOG_STD_MAX:
                    log_std = LOG_STD_MAX

                var std = exp(log_std)
                var normalized = (action - mean) / (std + 1e-6)

                # Gaussian log probability (no squashing)
                var log_gaussian = -0.5 * (
                    LOG_2PI + 2.0 * log_std + normalized * normalized
                )
                total_log_prob += log_gaussian

                # Cache for backward: [std(A) | normalized_diff(A)]
                cache[b, j] = Scalar[dtype](std)
                cache[b, A + j] = Scalar[dtype](normalized)

            output[b, 0] = Scalar[dtype](total_log_prob)

    # =========================================================================
    # CPU vjp
    # =========================================================================

    @staticmethod
    def vjp[
        BATCH: Int, dtype: DType = DType.float32
    ](
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        mut grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut grad_params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Backward through Gaussian log probability.

        grad_output[:, 0] = grad_log_prob (scalar per sample)

        Produces:
        grad_input[:, :A]       = grad_mean
        grad_input[:, A:2A]     = grad_log_std
        grad_input[:, 2A:3A]    = 0 (actions are frozen)
        """
        comptime A = Self.action_dim

        for b in range(BATCH):
            var glp = Float64(rebind[Scalar[dtype]](grad_output[b, 0]))

            for j in range(A):
                var std = Float64(rebind[Scalar[dtype]](cache[b, j]))
                var normalized = Float64(rebind[Scalar[dtype]](cache[b, A + j]))

                # grad_mean = grad_log_prob * normalized / std
                # = grad_log_prob * (action - mean) / std^2
                var grad_mean = glp * normalized / (std + 1e-6)

                # grad_log_std = grad_log_prob * (normalized^2 - 1)
                var grad_log_std = glp * (normalized * normalized - 1.0)

                grad_input[b, j] = Scalar[dtype](grad_mean)
                grad_input[b, A + j] = Scalar[dtype](grad_log_std)
                # Actions are frozen external input - no gradient
                grad_input[b, 2 * A + j] = Scalar[dtype](0.0)

    # =========================================================================
    # GPU eval
    # =========================================================================

    @always_inline
    @staticmethod
    def eval_kernel_impl[
        BATCH: Int, dtype: DType = DType.float32
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        """One thread per batch element. Loops over action_dim internally."""
        comptime assert dtype.is_floating_point(), "dtype must be floating point"
        comptime A = Self.action_dim
        comptime LOG_STD_MIN: Scalar[dtype] = -5.0
        comptime LOG_STD_MAX: Scalar[dtype] = 2.0

        var b = Int(block_dim.x * block_idx.x + thread_idx.x)
        if b >= BATCH:
            return

        var total_log_prob = Scalar[dtype](0.0)

        for j in range(A):
            var mean = rebind[Scalar[dtype]](input[b, j])
            var log_std = rebind[Scalar[dtype]](input[b, A + j])
            var action = rebind[Scalar[dtype]](input[b, 2 * A + j])

            # Clamp log_std
            if log_std < LOG_STD_MIN:
                log_std = LOG_STD_MIN
            elif log_std > LOG_STD_MAX:
                log_std = LOG_STD_MAX

            var std = exp(log_std)
            var normalized = (action - mean) / (std + Scalar[dtype](1e-6))

            # Gaussian log probability
            var log_gaussian = Scalar[dtype](-0.5) * (
                Scalar[dtype](LOG_2PI)
                + Scalar[dtype](2.0) * log_std
                + normalized * normalized
            )
            total_log_prob += log_gaussian

            # Cache: [std(A) | normalized_diff(A)]
            cache[b, j] = std
            cache[b, A + j] = normalized

        output[b, 0] = total_log_prob

    @staticmethod
    def eval_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        workspace: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ) raises:
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](input.ptr)
        var grid_x = (BATCH + TPB - 1) // TPB

        @always_inline
        def wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
            ],
        ):
            Self.eval_kernel_impl[BATCH, dtype](output, input, cache)

        ctx.enqueue_function[wrapper, wrapper](
            output,
            input_immut,
            cache,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # GPU vjp
    # =========================================================================

    @always_inline
    @staticmethod
    def vjp_kernel_impl[
        BATCH: Int, dtype: DType = DType.float32
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ],
    ):
        """One thread per (batch, action_dim) element."""
        comptime A = Self.action_dim
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * A:
            return

        var b = idx // A
        var j = idx % A

        var glp = rebind[Scalar[dtype]](grad_output[b, 0])
        var std = rebind[Scalar[dtype]](cache[b, j])
        var normalized = rebind[Scalar[dtype]](cache[b, A + j])

        # grad_mean = grad_log_prob * normalized / std
        grad_input[b, j] = glp * normalized / (std + Scalar[dtype](1e-6))

        # grad_log_std = grad_log_prob * (normalized^2 - 1)
        grad_input[b, A + j] = glp * (
            normalized * normalized - Scalar[dtype](1.0)
        )

        # Actions are frozen external input - no gradient
        grad_input[b, 2 * A + j] = Scalar[dtype](0.0)

    @staticmethod
    def vjp_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        mut grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut grad_params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        workspace: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ) raises:
        var grad_output_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ](grad_output.ptr)
        var cache_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ](cache.ptr)
        comptime total = BATCH * Self.action_dim
        var grid_x = (total + TPB - 1) // TPB

        @always_inline
        def wrapper(
            gi: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            go: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
            c: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.CACHE_SIZE),
                ImmutAnyOrigin,
            ],
        ):
            Self.vjp_kernel_impl[BATCH, dtype](gi, go, c)

        ctx.enqueue_function[wrapper, wrapper](
            grad_input,
            grad_output_immut,
            cache_immut,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )
