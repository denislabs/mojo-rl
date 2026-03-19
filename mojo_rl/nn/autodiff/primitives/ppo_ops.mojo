"""PPO DiffOps: CategoricalLogProbOp, RatioOp, ClipSurrogateOp.

These ops enable expressing PPO's clipped surrogate loss as an autodiff graph.

CategoricalLogProbOp: Computes log probability of a selected action from logits.
RatioOp: Computes importance sampling ratio exp(log_prob - old_log_prob).
ClipSurrogateOp: PPO clipped surrogate objective (negated for minimization).

Usage in discrete PPO autodiff graph:
    logits → CategoricalLogProbOp → log_prob
    [log_prob || old_log_prob] → RatioOp → ratio
    [ratio || advantage] → ClipSurrogateOp → loss
"""

from ...constants import dtype, TPB
from ...autodiff.op import DiffOp, OpID
from layout import Layout, LayoutTensor
from std.math import exp, log
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext


# =============================================================================
# CategoricalLogProbOp
# =============================================================================


struct CategoricalLogProbOp[num_actions: Int](DiffOp):
    """Log probability from categorical distribution.

    Input: [BATCH, num_actions + 1]
      - input[b, 0..num_actions-1] = raw logits
      - input[b, num_actions] = action index (packed as float, cast to Int)

    Output: [BATCH, 1] = log_softmax(logits)[action_index]

    PARAM_SIZE = 0
    CACHE_SIZE = num_actions + 1  (caches softmax probs + action index for backward)
    """

    comptime OP_ID: Int = OpID.USER_DEFINED._value + 6
    comptime IN_DIM: Int = Self.num_actions + 1
    comptime OUT_DIM: Int = 1
    comptime PARAM_SIZE: Int = 0
    comptime CACHE_SIZE: Int = Self.num_actions + 1
    comptime OP_WORKSPACE_PER_SAMPLE: Int = 0

    fn __init__(out self):
        pass

    fn __init__(out self, *, deinit take: Self):
        pass

    fn __init__(out self, *, copy: Self):
        pass

    # =========================================================================
    # CPU eval / vjp
    # =========================================================================

    @staticmethod
    fn eval[
        BATCH: Int
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
        comptime N = Self.num_actions
        for b in range(BATCH):
            # Read action index
            var action_idx = Int(
                Float64(rebind[Scalar[dtype]](input[b, N]))
            )

            # Find max logit for numerical stability
            var max_logit = Float64(rebind[Scalar[dtype]](input[b, 0]))
            for j in range(1, N):
                var lj = Float64(rebind[Scalar[dtype]](input[b, j]))
                if lj > max_logit:
                    max_logit = lj

            # Compute softmax probabilities
            var sum_exp: Float64 = 0.0
            for j in range(N):
                var lj = Float64(rebind[Scalar[dtype]](input[b, j]))
                sum_exp += exp(lj - max_logit)

            for j in range(N):
                var lj = Float64(rebind[Scalar[dtype]](input[b, j]))
                var prob_j = exp(lj - max_logit) / sum_exp
                cache[b, j] = Scalar[dtype](prob_j)

            # Cache action index
            cache[b, N] = Scalar[dtype](Float64(action_idx))

            # Output: log(probs[action_idx])
            var prob_a = Float64(rebind[Scalar[dtype]](cache[b, action_idx]))
            output[b, 0] = Scalar[dtype](log(prob_a))

    @staticmethod
    fn vjp[
        BATCH: Int
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
        """Backward: d(log_softmax[i])/d(logit[j]) = delta_ij - softmax_j."""
        comptime N = Self.num_actions
        for b in range(BATCH):
            var g = Float64(rebind[Scalar[dtype]](grad_output[b, 0]))
            var action_idx = Int(
                Float64(rebind[Scalar[dtype]](cache[b, N]))
            )

            for j in range(N):
                var prob_j = Float64(rebind[Scalar[dtype]](cache[b, j]))
                var indicator: Float64 = 1.0 if j == action_idx else 0.0
                grad_input[b, j] = Scalar[dtype](g * (indicator - prob_j))

            # No gradient for action index
            grad_input[b, N] = Scalar[dtype](0.0)

    # =========================================================================
    # GPU kernels
    # =========================================================================

    @always_inline
    @staticmethod
    fn eval_kernel_impl[
        BATCH: Int
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
        """One thread per batch element."""
        comptime N = Self.num_actions
        var b = Int(block_dim.x * block_idx.x + thread_idx.x)
        if b >= BATCH:
            return

        var action_idx = Int(rebind[Scalar[dtype]](input[b, N]))

        # Max logit for stability
        var max_logit = rebind[Scalar[dtype]](input[b, 0])
        for j in range(1, N):
            var lj = rebind[Scalar[dtype]](input[b, j])
            if lj > max_logit:
                max_logit = lj

        # Softmax
        var sum_exp = Scalar[dtype](0.0)
        for j in range(N):
            var lj = rebind[Scalar[dtype]](input[b, j])
            sum_exp += exp(lj - max_logit)

        for j in range(N):
            var lj = rebind[Scalar[dtype]](input[b, j])
            var prob_j = exp(lj - max_logit) / sum_exp
            cache[b, j] = prob_j

        cache[b, N] = Scalar[dtype](action_idx)
        output[b, 0] = log(rebind[Scalar[dtype]](cache[b, action_idx]))

    @always_inline
    @staticmethod
    fn backward_kernel_impl[
        BATCH: Int
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
        """One thread per batch element."""
        comptime N = Self.num_actions
        var b = Int(block_dim.x * block_idx.x + thread_idx.x)
        if b >= BATCH:
            return

        var g = rebind[Scalar[dtype]](grad_output[b, 0])
        var action_idx = Int(rebind[Scalar[dtype]](cache[b, N]))

        for j in range(N):
            var prob_j = rebind[Scalar[dtype]](cache[b, j])
            var indicator = Scalar[dtype](1.0) if j == action_idx else Scalar[
                dtype
            ](0.0)
            grad_input[b, j] = g * (indicator - prob_j)

        grad_input[b, N] = Scalar[dtype](0.0)

    # =========================================================================
    # GPU launchers
    # =========================================================================

    @staticmethod
    fn eval_gpu[
        BATCH: Int
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
        fn wrapper(
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
            Self.eval_kernel_impl[BATCH](output, input, cache)

        ctx.enqueue_function[wrapper, wrapper](
            output,
            input_immut,
            cache,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn vjp_gpu[
        BATCH: Int
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
        var grid_x = (BATCH + TPB - 1) // TPB

        @always_inline
        fn wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.CACHE_SIZE),
                ImmutAnyOrigin,
            ],
        ):
            Self.backward_kernel_impl[BATCH](grad_input, grad_output, cache)

        ctx.enqueue_function[wrapper, wrapper](
            grad_input,
            grad_output_immut,
            cache_immut,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )


# =============================================================================
# RatioOp
# =============================================================================


struct RatioOp[dim: Int = 1](DiffOp):
    """Importance sampling ratio: ratio = exp(log_prob - old_log_prob).

    Input: [BATCH, 2*dim] = [log_prob || old_log_prob]
      - log_prob: current policy log probability (differentiable)
      - old_log_prob: old policy log probability (frozen, no gradient)

    Output: [BATCH, dim] = exp(log_prob - old_log_prob)

    PARAM_SIZE = 0
    CACHE_SIZE = dim  (caches the ratio for backward)
    """

    comptime OP_ID: Int = OpID.USER_DEFINED._value + 7
    comptime IN_DIM: Int = 2 * Self.dim
    comptime OUT_DIM: Int = Self.dim
    comptime PARAM_SIZE: Int = 0
    comptime CACHE_SIZE: Int = Self.dim
    comptime OP_WORKSPACE_PER_SAMPLE: Int = 0

    fn __init__(out self):
        pass

    fn __init__(out self, *, deinit take: Self):
        pass

    fn __init__(out self, *, copy: Self):
        pass

    # =========================================================================
    # CPU eval / vjp
    # =========================================================================

    @staticmethod
    fn eval[
        BATCH: Int
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
        for b in range(BATCH):
            for i in range(Self.dim):
                var log_prob = Float64(
                    rebind[Scalar[dtype]](input[b, i])
                )
                var old_log_prob = Float64(
                    rebind[Scalar[dtype]](input[b, Self.dim + i])
                )
                var ratio = exp(log_prob - old_log_prob)
                output[b, i] = Scalar[dtype](ratio)
                cache[b, i] = Scalar[dtype](ratio)

    @staticmethod
    fn vjp[
        BATCH: Int
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
        """grad_log_prob = grad_output * ratio, grad_old_log_prob = 0."""
        for b in range(BATCH):
            for i in range(Self.dim):
                var g = Float64(rebind[Scalar[dtype]](grad_output[b, i]))
                var ratio = Float64(rebind[Scalar[dtype]](cache[b, i]))
                grad_input[b, i] = Scalar[dtype](g * ratio)
                grad_input[b, Self.dim + i] = Scalar[dtype](0.0)

    # =========================================================================
    # GPU kernels
    # =========================================================================

    @always_inline
    @staticmethod
    fn eval_kernel_impl[
        BATCH: Int
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
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return
        var b = idx // Self.dim
        var i = idx % Self.dim
        var log_prob = rebind[Scalar[dtype]](input[b, i])
        var old_log_prob = rebind[Scalar[dtype]](input[b, Self.dim + i])
        var ratio = exp(log_prob - old_log_prob)
        output[b, i] = ratio
        cache[b, i] = ratio

    @always_inline
    @staticmethod
    fn backward_kernel_impl[
        BATCH: Int
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
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return
        var b = idx // Self.dim
        var i = idx % Self.dim
        var g = rebind[Scalar[dtype]](grad_output[b, i])
        var ratio = rebind[Scalar[dtype]](cache[b, i])
        grad_input[b, i] = g * ratio
        grad_input[b, Self.dim + i] = Scalar[dtype](0.0)

    # =========================================================================
    # GPU launchers
    # =========================================================================

    @staticmethod
    fn eval_gpu[
        BATCH: Int
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
        var total = BATCH * Self.dim
        var grid_x = (total + TPB - 1) // TPB

        @always_inline
        fn wrapper(
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
            Self.eval_kernel_impl[BATCH](output, input, cache)

        ctx.enqueue_function[wrapper, wrapper](
            output,
            input_immut,
            cache,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn vjp_gpu[
        BATCH: Int
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
        var total = BATCH * Self.dim
        var grid_x = (total + TPB - 1) // TPB

        @always_inline
        fn wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.CACHE_SIZE),
                ImmutAnyOrigin,
            ],
        ):
            Self.backward_kernel_impl[BATCH](grad_input, grad_output, cache)

        ctx.enqueue_function[wrapper, wrapper](
            grad_input,
            grad_output_immut,
            cache_immut,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )


# =============================================================================
# ClipSurrogateOp
# =============================================================================


struct ClipSurrogateOp[eps: Float64 = 0.2](DiffOp):
    """PPO clipped surrogate objective (negated for minimization).

    Input: [BATCH, 2] = [ratio || advantage]
      - ratio: importance sampling ratio (differentiable)
      - advantage: GAE advantage (frozen, no gradient)

    Output: [BATCH, 1] = -min(ratio * adv, clip(ratio, 1-eps, 1+eps) * adv)
      Negated so that minimizing this = maximizing the surrogate.

    PARAM_SIZE = 0
    CACHE_SIZE = 1  (caches which branch was selected: 0=unclipped, 1=clipped)
    """

    comptime OP_ID: Int = OpID.USER_DEFINED._value + 8
    comptime IN_DIM: Int = 2
    comptime OUT_DIM: Int = 1
    comptime PARAM_SIZE: Int = 0
    comptime CACHE_SIZE: Int = 1
    comptime OP_WORKSPACE_PER_SAMPLE: Int = 0

    # Precompute clip bounds
    comptime LO: Float64 = 1.0 - Self.eps
    comptime HI: Float64 = 1.0 + Self.eps

    fn __init__(out self):
        pass

    fn __init__(out self, *, deinit take: Self):
        pass

    fn __init__(out self, *, copy: Self):
        pass

    # =========================================================================
    # CPU eval / vjp
    # =========================================================================

    @staticmethod
    fn eval[
        BATCH: Int
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
        """Forward: output = -min(ratio * adv, clip(ratio) * adv).

        Cache stores the precomputed gradient multiplier for ratio so that
        backward only needs: grad_ratio = grad_output * cache[b, 0].
        This avoids needing to re-access input (ratio, advantage) in vjp.

        Multiplier is:
          -advantage  if unclipped branch OR clipped branch with ratio in range
          0           if clipped branch with ratio outside [1-eps, 1+eps]
        """
        for b in range(BATCH):
            var ratio = Float64(rebind[Scalar[dtype]](input[b, 0]))
            var advantage = Float64(rebind[Scalar[dtype]](input[b, 1]))

            var surr1 = ratio * advantage

            # Clamp ratio
            var clipped_ratio = ratio
            if clipped_ratio < Self.LO:
                clipped_ratio = Self.LO
            if clipped_ratio > Self.HI:
                clipped_ratio = Self.HI

            var surr2 = clipped_ratio * advantage

            if surr1 <= surr2:
                # Unclipped branch: grad_ratio = -advantage
                output[b, 0] = Scalar[dtype](-surr1)
                cache[b, 0] = Scalar[dtype](-advantage)
            else:
                # Clipped branch selected
                output[b, 0] = Scalar[dtype](-surr2)
                if ratio >= Self.LO and ratio <= Self.HI:
                    # Inside clip range: gradient passes through
                    cache[b, 0] = Scalar[dtype](-advantage)
                else:
                    # Outside clip range: gradient zeroed (the clip effect)
                    cache[b, 0] = Scalar[dtype](0.0)

    @staticmethod
    fn vjp[
        BATCH: Int
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
        """Backward: grad_ratio = grad_output * cached_multiplier, grad_adv = 0."""
        for b in range(BATCH):
            var g = Float64(rebind[Scalar[dtype]](grad_output[b, 0]))
            var grad_mult = Float64(rebind[Scalar[dtype]](cache[b, 0]))
            grad_input[b, 0] = Scalar[dtype](g * grad_mult)
            grad_input[b, 1] = Scalar[dtype](0.0)  # no grad for advantage

    # =========================================================================
    # GPU kernels
    # =========================================================================

    @always_inline
    @staticmethod
    fn eval_kernel_impl[
        BATCH: Int
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
        var b = Int(block_dim.x * block_idx.x + thread_idx.x)
        if b >= BATCH:
            return

        var ratio = rebind[Scalar[dtype]](input[b, 0])
        var advantage = rebind[Scalar[dtype]](input[b, 1])
        var lo = Scalar[dtype](Self.LO)
        var hi = Scalar[dtype](Self.HI)

        var surr1 = ratio * advantage

        # Clamp ratio
        var clipped_ratio = ratio
        if clipped_ratio < lo:
            clipped_ratio = lo
        if clipped_ratio > hi:
            clipped_ratio = hi

        var surr2 = clipped_ratio * advantage

        var neg_adv = -advantage
        var zero = Scalar[dtype](0.0)

        if surr1 <= surr2:
            # Unclipped branch selected: grad_ratio = -advantage
            output[b, 0] = -surr1
            cache[b, 0] = neg_adv
        else:
            # Clipped branch selected
            output[b, 0] = -surr2
            if ratio >= lo and ratio <= hi:
                # Inside clip range: gradient passes through
                cache[b, 0] = neg_adv
            else:
                # Outside clip range: gradient zeroed
                cache[b, 0] = zero

    @always_inline
    @staticmethod
    fn backward_kernel_impl[
        BATCH: Int
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
        var b = Int(block_dim.x * block_idx.x + thread_idx.x)
        if b >= BATCH:
            return

        var g = rebind[Scalar[dtype]](grad_output[b, 0])
        var grad_mult = rebind[Scalar[dtype]](cache[b, 0])
        grad_input[b, 0] = g * grad_mult
        grad_input[b, 1] = Scalar[dtype](0.0)

    # =========================================================================
    # GPU launchers
    # =========================================================================

    @staticmethod
    fn eval_gpu[
        BATCH: Int
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
        fn wrapper(
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
            Self.eval_kernel_impl[BATCH](output, input, cache)

        ctx.enqueue_function[wrapper, wrapper](
            output,
            input_immut,
            cache,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )

    @staticmethod
    fn vjp_gpu[
        BATCH: Int
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
        var grid_x = (BATCH + TPB - 1) // TPB

        @always_inline
        fn wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.CACHE_SIZE),
                ImmutAnyOrigin,
            ],
        ):
            Self.backward_kernel_impl[BATCH](grad_input, grad_output, cache)

        ctx.enqueue_function[wrapper, wrapper](
            grad_input,
            grad_output_immut,
            cache_immut,
            grid_dim=(grid_x,),
            block_dim=(TPB,),
        )
