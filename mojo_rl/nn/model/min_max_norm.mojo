"""Per-sample min-max normalization with proper backward.

y_j = (x_j - min(x)) / (max(x) - min(x))

Used by MuZero (paper appendix Training, see muzero-general models.py:138-145).
The reference implementation does the min-max via PyTorch tensor ops, so
gradient flows through it naturally — the rep network learns to produce
well-spread outputs that play well with the normalization.

Our previous post-hoc `scale_hidden_kernel` did min-max as a standalone GPU
kernel outside the autodiff graph, so no gradient flowed back through it.
This caused activation explosion (10⁶ raw outputs) and sign-symmetric
representation collapse (different obs producing same-direction outputs)
because the network had no gradient signal about its raw output magnitudes
or directions. See docs/MUZERO_AUDIT.md Phase G post-mortem.

Math (per sample of dim N):
  m = min(x), M = max(x), s = clamp(M - m, ≥ ε)
  y_j = (x_j - m) / s

Backward (given grad_y, compute grad_x):
  Let G = Σ grad_y, Gy = Σ grad_y · y
  - For i ∉ {argmin, argmax}: grad_x[i] = grad_y[i] / s
  - For i = argmax:           grad_x[i] = (grad_y[argmax] - Gy) / s
  - For i = argmin:           grad_x[i] = (Gy + grad_y[argmin] - G) / s

Sum-zero invariant (verified):
  Σ grad_x = 0  (gradient is shift-invariant since y is shift-invariant in x)
"""

from ..constants import dtype, TPB
from .model import Model, PerfTimerPtr, NULL_PERF
from ..initializer import Initializer
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream
from std.gpu.memory import AddressSpace
from std.gpu.primitives import block


struct MinMaxNorm[dim: Int, EPSILON: Float64 = 1e-5](Model):
    """Per-sample min-max scaling: y = (x - min(x)) / (max(x) - min(x)).

    No learned parameters. CACHE = the input tensor (re-find min/max/argmin/
    argmax in backward; cheaper than caching indices and avoids int-stored-as-
    float fragility for large `dim`).

    Layout:
    - params: empty (PARAM_SIZE = 0)
    - cache:  [input copy (dim)] per sample
    """

    comptime IN_DIM: Int = Self.dim
    comptime OUT_DIM: Int = Self.dim
    comptime PARAM_SIZE: Int = 0
    comptime CACHE_SIZE: Int = Self.dim
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = 0
    comptime STATE_SIZE: Int = 0

    @staticmethod
    def initialize_params[
        INIT: Initializer, dtype: DType = DType.float32
    ](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """No-op: MinMaxNorm has no learned params."""
        pass

    @staticmethod
    def forward[
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        """Per-sample min-max scaling. Caches input for backward."""
        var eps = Scalar[dtype](Self.EPSILON)

        for batch in range(BATCH):
            # Find min, max
            var x0 = rebind[Scalar[dtype]](input[batch, 0])
            var min_val = x0
            var max_val = x0
            cache[batch, 0] = x0
            for i in range(1, Self.dim):
                var v = rebind[Scalar[dtype]](input[batch, i])
                cache[batch, i] = v
                if v < min_val:
                    min_val = v
                if v > max_val:
                    max_val = v

            # Scale (clamped to ≥ epsilon)
            var s = max_val - min_val
            if s < eps:
                s = eps
            var inv_s = Scalar[dtype](1.0) / s

            # y_i = (x_i - min) / s
            for i in range(Self.dim):
                var x = rebind[Scalar[dtype]](cache[batch, i])
                output[batch, i] = (x - min_val) * inv_s

    @staticmethod
    def forward[
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
    ):
        """Forward without caching (inference path)."""
        var eps = Scalar[dtype](Self.EPSILON)

        for batch in range(BATCH):
            var x0 = input[batch, 0]
            var min_val = x0
            var max_val = x0
            for i in range(1, Self.dim):
                var v = input[batch, i]
                if v < min_val:
                    min_val = v
                if v > max_val:
                    max_val = v
            var s = max_val - min_val
            if s < eps:
                s = eps
            var inv_s = output.element_type(1.0) / s
            for i in range(Self.dim):
                output[batch, i] = (input[batch, i] - min_val) * inv_s

    @staticmethod
    def backward[
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Per-sample backward — recovers min/max/argmin/argmax from cache."""
        var eps = Scalar[dtype](Self.EPSILON)

        for batch in range(BATCH):
            # Re-find min, max, argmin, argmax from cached input
            var x0 = rebind[Scalar[dtype]](cache[batch, 0])
            var min_val = x0
            var max_val = x0
            var argmin = 0
            var argmax = 0
            for i in range(1, Self.dim):
                var v = rebind[Scalar[dtype]](cache[batch, i])
                if v < min_val:
                    min_val = v
                    argmin = i
                if v > max_val:
                    max_val = v
                    argmax = i

            var s = max_val - min_val
            var degenerate = s < eps
            if degenerate:
                s = eps
            var inv_s = Scalar[dtype](1.0) / s

            # Degenerate case (max == min): all y_j are constants → grad_x = 0.
            # This handles both the truly-constant-input case and the
            # epsilon-clamped near-constant case where the formula is unstable.
            if degenerate:
                for i in range(Self.dim):
                    grad_input[batch, i] = Scalar[dtype](0.0)
                continue

            # Compute y_j and accumulate G = Σ grad_y, Gy = Σ grad_y · y_j
            var G = Scalar[dtype](0.0)
            var Gy = Scalar[dtype](0.0)
            for i in range(Self.dim):
                var x = rebind[Scalar[dtype]](cache[batch, i])
                var y = (x - min_val) * inv_s
                var dy = rebind[Scalar[dtype]](grad_output[batch, i])
                G = G + dy
                Gy = Gy + dy * y

            var dy_argmin = rebind[Scalar[dtype]](
                grad_output[batch, argmin]
            )
            var dy_argmax = rebind[Scalar[dtype]](
                grad_output[batch, argmax]
            )

            # Per-element grad
            for i in range(Self.dim):
                var dy = rebind[Scalar[dtype]](grad_output[batch, i])
                var dx: Scalar[dtype]
                if i == argmin and i == argmax:
                    # Degenerate (already handled above, but defensive)
                    dx = Scalar[dtype](0.0)
                elif i == argmin:
                    dx = (Gy + dy_argmin - G) * inv_s
                elif i == argmax:
                    dx = (dy_argmax - Gy) * inv_s
                else:
                    dx = dy * inv_s
                grad_input[batch, i] = dx

    # =========================================================================
    # GPU Kernel Implementations — one block per sample, threads parallelise dim
    # =========================================================================

    @always_inline
    @staticmethod
    def forward_kernel_impl[
        BATCH: Int, dtype: DType = DType.float32
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
        eps: Scalar[dtype],
    ):
        """Forward kernel with caching. Grid: (BATCH,), Block: (TPB,)."""
        var b = Int(block_idx.x)
        var local_i = Int(thread_idx.x)

        if b >= BATCH:
            return

        # Phase 1: per-thread local min/max via stride-TPB iteration
        var pos_inf = Scalar[dtype](1e30)
        var neg_inf = Scalar[dtype](-1e30)
        var my_min = pos_inf
        var my_max = neg_inf
        var idx = local_i
        while idx < Self.dim:
            var v = rebind[Scalar[dtype]](input[b, idx])
            if v < my_min:
                my_min = v
            if v > my_max:
                my_max = v
            idx += TPB

        # Block reductions to get per-sample min and max, broadcast to all threads
        var min_val = block.min[block_size=TPB, broadcast=True](val=my_min)
        var max_val = block.max[block_size=TPB, broadcast=True](val=my_max)

        var s = max_val - min_val
        if s < eps:
            s = eps
        var inv_s = Scalar[dtype](1.0) / s

        # Phase 2: write y_i = (x_i - min) / s and cache the input
        idx = local_i
        while idx < Self.dim:
            var x = rebind[Scalar[dtype]](input[b, idx])
            cache[b, idx] = x
            output[b, idx] = (x - min_val) * inv_s
            idx += TPB

    @always_inline
    @staticmethod
    def backward_kernel_impl[
        BATCH: Int, dtype: DType = DType.float32
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ],
        eps: Scalar[dtype],
    ):
        """Backward kernel — re-derives min/max/argmin/argmax from cached input.

        Uses block reductions for min/max (and their indices via paired key/value
        reductions). Grid: (BATCH,), Block: (TPB,).
        """
        var b = Int(block_idx.x)
        var local_i = Int(thread_idx.x)
        if b >= BATCH:
            return

        # Phase 1: per-thread local min/max + argmin/argmax
        var pos_inf = Scalar[dtype](1e30)
        var neg_inf = Scalar[dtype](-1e30)
        var my_min = pos_inf
        var my_max = neg_inf
        var my_argmin: Int = 0
        var my_argmax: Int = 0
        var idx = local_i
        while idx < Self.dim:
            var v = rebind[Scalar[dtype]](cache[b, idx])
            if v < my_min:
                my_min = v
                my_argmin = idx
            if v > my_max:
                my_max = v
                my_argmax = idx
            idx += TPB

        # Reduce min and max across the block (broadcast). For indices, we use
        # a separate pass: each thread checks if its local min equals the block
        # min, and uses block.min on the index (smallest index wins ties).
        var min_val = block.min[block_size=TPB, broadcast=True](val=my_min)
        var max_val = block.max[block_size=TPB, broadcast=True](val=my_max)

        # Resolve argmin/argmax via index reductions: threads that didn't hold
        # the global min/max emit a sentinel large index.
        var SENTINEL = Self.dim
        var my_argmin_signed: Int = my_argmin if my_min == min_val else SENTINEL
        var my_argmax_signed: Int = my_argmax if my_max == max_val else SENTINEL
        # Cast to a SIMD type that block.min supports
        var argmin = block.min[block_size=TPB, broadcast=True](
            val=Scalar[DType.int32](my_argmin_signed)
        )
        var argmax = block.min[block_size=TPB, broadcast=True](
            val=Scalar[DType.int32](my_argmax_signed)
        )

        var s = max_val - min_val
        var degenerate = s < eps
        if degenerate:
            s = eps
        var inv_s = Scalar[dtype](1.0) / s

        if degenerate:
            idx = local_i
            while idx < Self.dim:
                grad_input[b, idx] = Scalar[dtype](0.0)
                idx += TPB
            return

        # Phase 2: compute G = Σ grad_y, Gy = Σ grad_y · y via block reductions
        var my_G = Scalar[dtype](0.0)
        var my_Gy = Scalar[dtype](0.0)
        idx = local_i
        while idx < Self.dim:
            var x = rebind[Scalar[dtype]](cache[b, idx])
            var y = (x - min_val) * inv_s
            var dy = rebind[Scalar[dtype]](grad_output[b, idx])
            my_G = my_G + dy
            my_Gy = my_Gy + dy * y
            idx += TPB
        var G = block.sum[block_size=TPB, broadcast=True](val=my_G)
        var Gy = block.sum[block_size=TPB, broadcast=True](val=my_Gy)

        var dy_argmin = rebind[Scalar[dtype]](
            grad_output[b, Int(argmin)]
        )
        var dy_argmax = rebind[Scalar[dtype]](
            grad_output[b, Int(argmax)]
        )

        # Phase 3: per-element grad
        idx = local_i
        while idx < Self.dim:
            var dy = rebind[Scalar[dtype]](grad_output[b, idx])
            var dx: Scalar[dtype]
            if Int32(idx) == argmin and Int32(idx) == argmax:
                dx = Scalar[dtype](0.0)
            elif Int32(idx) == argmin:
                dx = (Gy + dy_argmin - G) * inv_s
            elif Int32(idx) == argmax:
                dx = (dy_argmax - Gy) * inv_s
            else:
                dx = dy * inv_s
            grad_input[b, idx] = dx
            idx += TPB

    @staticmethod
    def forward_gpu[
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """Launch forward pass on GPU."""
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](input.ptr)

        @parameter
        @always_inline
        def wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
            ],
            eps: Scalar[dtype],
        ):
            Self.forward_kernel_impl[BATCH, dtype](output, input, cache, eps)

        ctx.enqueue_function[wrapper](
            output,
            input_immut,
            cache,
            Scalar[dtype](Self.EPSILON),
            grid_dim=(BATCH,),
            block_dim=(TPB,),
        )

    @staticmethod
    def forward_gpu_no_cache[
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
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """Inference forward (no caching) — allocates a throwaway cache."""
        var dummy_cache = ctx.enqueue_create_buffer[dtype](BATCH * Self.dim)
        var dummy_cache_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ](dummy_cache.unsafe_ptr())
        Self.forward_gpu[BATCH, dtype](
            ctx, output, input, params, state, dummy_cache_t, workspace,
            perf, perf_slot,
        )

    @staticmethod
    def forward_gpu_no_cache_on_stream[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        stream: DeviceStream,
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
    ) raises:
        """Stream variant — delegates to default queue (no per-stream impl)."""
        Self.forward_gpu_no_cache[BATCH, dtype](
            ctx, output, input, params, state, workspace
        )

    @staticmethod
    def backward_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        mut grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """Launch backward pass on GPU."""
        var grad_output_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](grad_output.ptr)
        var cache_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](cache.ptr)

        @parameter
        @always_inline
        def wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ],
            eps: Scalar[dtype],
        ):
            Self.backward_kernel_impl[BATCH, dtype](
                grad_input, grad_output, cache, eps
            )

        ctx.enqueue_function[wrapper](
            grad_input,
            grad_output_immut,
            cache_immut,
            Scalar[dtype](Self.EPSILON),
            grid_dim=(BATCH,),
            block_dim=(TPB,),
        )
