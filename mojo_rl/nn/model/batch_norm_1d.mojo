"""BatchNorm1D: Batch Normalization for flat (Linear) tensors.

Normalizes per-feature across batch, matching the PyTorch/TF
`BatchNormalization` used in OFENet and other MLP-based RL architectures.

  y = gamma * (x - batch_mean) / sqrt(batch_var + eps) + beta   (training)
  y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta  (inference)

Training mode = forward WITH cache (uses batch stats, updates running stats).
Inference mode = forward WITHOUT cache (uses running stats, no update).

Params layout: [gamma(dim) | beta(dim)]                       (PARAM_SIZE = 2*dim)
State layout:  [running_mean(dim) | running_var(dim)]         (STATE_SIZE = 2*dim)
Cache layout per sample: [x_hat(dim) | batch_mean(dim) | batch_inv_std(dim)]

Phase 3 split: gamma/beta live in params (gradient-tracked), running_mean
and running_var live in the persistent model-state buffer (never touched by
the optimizer, updated via EMA inside the training-mode forward).

This is the 1D analogue of BatchNorm2D with channels=dim and spatial_size=1.
"""

from ..constants import dtype, TPB
from .model import Model, PerfTimerPtr, NULL_PERF
from ..initializer import Initializer
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.memory import AddressSpace
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream
from std.math import sqrt


struct BatchNorm1D[
    dim: Int,
    MOMENTUM: Float64 = 0.1,
    EPSILON: Float64 = 1e-5,
](Model):
    """Batch Normalization for flat activations (per-feature).

    Input: [BATCH, dim].
    Output: [BATCH, dim] (same shape).

    Parameters:
        dim: Feature dimension.
        MOMENTUM: EMA momentum for running stats (default 0.1).
        EPSILON: Numerical stability epsilon (default 1e-5).
    """

    comptime IN_DIM: Int = Self.dim
    comptime OUT_DIM: Int = Self.dim
    comptime PARAM_SIZE: Int = 2 * Self.dim  # gamma, beta (gradient-tracked)
    comptime CACHE_SIZE: Int = 3 * Self.dim  # x_hat + batch_mean + batch_inv_std
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = 0
    comptime STATE_SIZE: Int = 2 * Self.dim  # running_mean, running_var

    # Param offsets (within PARAM_SIZE)
    comptime GAMMA_OFF: Int = 0
    comptime BETA_OFF: Int = Self.dim
    # State offsets (within STATE_SIZE) — running stats live here post-Phase-3.
    comptime RMEAN_OFF: Int = 0
    comptime RVAR_OFF: Int = Self.dim
    # Cache offsets (per sample)
    comptime XHAT_OFF: Int = 0
    comptime CMEAN_OFF: Int = Self.dim
    comptime CINV_OFF: Int = 2 * Self.dim

    # =========================================================================
    # Initialization
    # =========================================================================

    @staticmethod
    def initialize_params[
        INIT: Initializer, dtype: DType = DType.float32
    ](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Initialize gamma=1, beta=0. Running stats are owned by `state`."""
        for i in range(Self.dim):
            params[Self.GAMMA_OFF + i] = Scalar[dtype](1.0)
            params[Self.BETA_OFF + i] = Scalar[dtype](0.0)

    @staticmethod
    def initialize_state[dtype: DType = DType.float32](
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
    ):
        """Initialize running_mean=0, running_var=1."""
        for i in range(Self.dim):
            state.ptr[Self.RMEAN_OFF + i] = Scalar[dtype](0.0)
            state.ptr[Self.RVAR_OFF + i] = Scalar[dtype](1.0)

    # =========================================================================
    # CPU Forward (training — with cache)
    # =========================================================================

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
        """Training forward: normalize using batch stats, update running stats."""
        var eps = Scalar[dtype](Self.EPSILON)
        var mom = Scalar[dtype](Self.MOMENTUM)
        var one_m = Scalar[dtype](1.0) - mom
        var n = Scalar[dtype](BATCH)

        for f in range(Self.dim):
            var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + f])
            var beta = rebind[Scalar[dtype]](params[Self.BETA_OFF + f])

            # 1. Compute batch mean for feature f
            var mean = Scalar[dtype](0.0)
            for b in range(BATCH):
                mean += rebind[Scalar[dtype]](input[b, f])
            mean = mean / n

            # 2. Compute batch variance for feature f
            var var_ = Scalar[dtype](0.0)
            for b in range(BATCH):
                var diff = rebind[Scalar[dtype]](input[b, f]) - mean
                var_ += diff * diff
            var_ = var_ / n

            # 3. Normalize, scale, shift
            var inv_std = Scalar[dtype](1.0) / Scalar[dtype](
                sqrt(Float64(var_ + eps))
            )
            for b in range(BATCH):
                var x = rebind[Scalar[dtype]](input[b, f])
                var x_hat = (x - mean) * inv_std
                cache[b, Self.XHAT_OFF + f] = x_hat
                output[b, f] = gamma * x_hat + beta
                cache[b, Self.CMEAN_OFF + f] = mean
                cache[b, Self.CINV_OFF + f] = inv_std

            # 4. Update running stats (EMA) in the persistent state buffer.
            var rm = rebind[Scalar[dtype]](state[Self.RMEAN_OFF + f])
            var rv = rebind[Scalar[dtype]](state[Self.RVAR_OFF + f])
            state.ptr[Self.RMEAN_OFF + f] = one_m * rm + mom * mean
            state.ptr[Self.RVAR_OFF + f] = one_m * rv + mom * var_

    # =========================================================================
    # CPU Forward (inference — no cache, uses running stats)
    # =========================================================================

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
        """Inference forward: use stored running stats, no update."""
        var eps = Scalar[dtype](Self.EPSILON)

        for f in range(Self.dim):
            var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + f])
            var beta = rebind[Scalar[dtype]](params[Self.BETA_OFF + f])
            var rmean = rebind[Scalar[dtype]](state[Self.RMEAN_OFF + f])
            var rvar = rebind[Scalar[dtype]](state[Self.RVAR_OFF + f])

            var inv_std = Scalar[dtype](1.0) / Scalar[dtype](
                sqrt(Float64(rvar + eps))
            )

            for b in range(BATCH):
                var x = rebind[Scalar[dtype]](input[b, f])
                output[b, f] = gamma * (x - rmean) * inv_std + beta

    # =========================================================================
    # CPU Backward
    # =========================================================================

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
        """Backward: compute grad_gamma, grad_beta, grad_input.

        Running stats live in `state` (not `params`) post-Phase-3, so there
        are no extra grad slots to keep zeroed.
        """
        var n = Scalar[dtype](BATCH)

        for f in range(Self.dim):
            var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + f])
            # Read batch_inv_std from sample 0's cache (same for all samples)
            var inv_std = rebind[Scalar[dtype]](cache[0, Self.CINV_OFF + f])

            # Accumulate grad_gamma and grad_beta, and intermediate sums
            var d_gamma = Scalar[dtype](0.0)
            var d_beta = Scalar[dtype](0.0)
            var sum_dy_gamma = Scalar[dtype](0.0)
            var sum_dy_gamma_xhat = Scalar[dtype](0.0)

            for b in range(BATCH):
                var dy = rebind[Scalar[dtype]](grad_output[b, f])
                var x_hat = rebind[Scalar[dtype]](cache[b, Self.XHAT_OFF + f])
                d_gamma += dy * x_hat
                d_beta += dy
                sum_dy_gamma += dy * gamma
                sum_dy_gamma_xhat += dy * gamma * x_hat

            # Accumulate into param grads (only gamma and beta)
            grads.ptr[Self.GAMMA_OFF + f] = (
                rebind[Scalar[dtype]](grads[Self.GAMMA_OFF + f]) + d_gamma
            )
            grads.ptr[Self.BETA_OFF + f] = (
                rebind[Scalar[dtype]](grads[Self.BETA_OFF + f]) + d_beta
            )

            # Compute grad_input
            for b in range(BATCH):
                var dy = rebind[Scalar[dtype]](grad_output[b, f])
                var x_hat = rebind[Scalar[dtype]](cache[b, Self.XHAT_OFF + f])
                var dx = inv_std * (
                    dy * gamma
                    - sum_dy_gamma / n
                    - x_hat * sum_dy_gamma_xhat / n
                )
                grad_input[b, f] = dx

    # =========================================================================
    # GPU Kernel Implementations
    # =========================================================================
    #
    # One block per feature. Threads parallel-reduce across BATCH.
    # =========================================================================

    @always_inline
    @staticmethod
    def forward_kernel_impl[
        BATCH: Int, dtype: DType = DType.float32,
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
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
    ):
        """Training forward kernel. Grid: (dim,), Block: (TPB,).

        Block-parallel reduction across BATCH per feature.
        """
        var f = Int(block_idx.x)
        if f >= Self.dim:
            return
        var tid = Int(thread_idx.x)

        var eps = Scalar[dtype](Self.EPSILON)
        var mom = Scalar[dtype](Self.MOMENTUM)
        var one_m = Scalar[dtype](1.0) - mom
        var n_f = Scalar[dtype](BATCH)
        var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + f])
        var beta = rebind[Scalar[dtype]](params[Self.BETA_OFF + f])

        var smem = LayoutTensor[
            dtype, Layout.row_major(TPB), MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()

        # Pass 1: Compute mean via block reduction
        var local_sum = Scalar[dtype](0.0)
        var idx = tid
        while idx < BATCH:
            local_sum += rebind[Scalar[dtype]](input[idx, f])
            idx += TPB
        smem[tid] = local_sum
        barrier()

        var stride = TPB // 2
        while stride > 0:
            if tid < stride:
                smem[tid] = smem[tid] + smem[tid + stride]
            barrier()
            stride = stride // 2

        var mean = rebind[Scalar[dtype]](smem[0]) / n_f
        barrier()

        # Pass 2: Compute variance via block reduction
        var local_var = Scalar[dtype](0.0)
        idx = tid
        while idx < BATCH:
            var diff = rebind[Scalar[dtype]](input[idx, f]) - mean
            local_var += diff * diff
            idx += TPB
        smem[tid] = local_var
        barrier()

        stride = TPB // 2
        while stride > 0:
            if tid < stride:
                smem[tid] = smem[tid] + smem[tid + stride]
            barrier()
            stride = stride // 2

        var var_ = rebind[Scalar[dtype]](smem[0]) / n_f
        var inv_std: Scalar[dtype] = 1.0 / sqrt(var_ + eps)
        barrier()

        # Pass 3: Normalize, scale, shift, cache (parallel scatter)
        idx = tid
        while idx < BATCH:
            var x = rebind[Scalar[dtype]](input[idx, f])
            var x_hat = (x - mean) * inv_std
            cache[idx, Self.XHAT_OFF + f] = x_hat
            cache[idx, Self.CMEAN_OFF + f] = mean
            cache[idx, Self.CINV_OFF + f] = inv_std
            output[idx, f] = gamma * x_hat + beta
            idx += TPB

        # Update running stats (thread 0 only) in the state buffer.
        if tid == 0:
            var rm = rebind[Scalar[dtype]](state[Self.RMEAN_OFF + f])
            var rv = rebind[Scalar[dtype]](state[Self.RVAR_OFF + f])
            state.ptr[Self.RMEAN_OFF + f] = one_m * rm + mom * mean
            state.ptr[Self.RVAR_OFF + f] = one_m * rv + mom * var_

    @always_inline
    @staticmethod
    def forward_kernel_impl_no_cache[
        BATCH: Int, dtype: DType = DType.float32,
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ],
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), ImmutAnyOrigin
        ],
    ):
        """Inference forward kernel using running stats (no update).

        Grid: (dim,), Block: (TPB,).
        """
        var f = Int(block_idx.x)
        if f >= Self.dim:
            return
        var tid = Int(thread_idx.x)

        var eps = Scalar[dtype](Self.EPSILON)
        var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + f])
        var beta = rebind[Scalar[dtype]](params[Self.BETA_OFF + f])
        var rmean = rebind[Scalar[dtype]](state[Self.RMEAN_OFF + f])
        var rvar = rebind[Scalar[dtype]](state[Self.RVAR_OFF + f])
        var inv_std: Scalar[dtype] = 1.0 / sqrt(rvar + eps)

        # Parallel scatter over batch
        var idx = tid
        while idx < BATCH:
            var x = rebind[Scalar[dtype]](input[idx, f])
            output[idx, f] = gamma * (x - rmean) * inv_std + beta
            idx += TPB

    @always_inline
    @staticmethod
    def backward_kernel_impl[
        BATCH: Int, dtype: DType = DType.float32,
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ],
        grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Backward kernel. Grid: (dim,), Block: (TPB,).

        Block-parallel reduction for gradient accumulation over batch.
        """
        var f = Int(block_idx.x)
        if f >= Self.dim:
            return
        var tid = Int(thread_idx.x)

        var n_f = Scalar[dtype](BATCH)
        var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + f])
        var inv_std = rebind[Scalar[dtype]](cache[0, Self.CINV_OFF + f])

        var smem = LayoutTensor[
            dtype, Layout.row_major(TPB), MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()

        # Pass 1: Accumulate 4 partial sums per thread
        var local_d_gamma = Scalar[dtype](0.0)
        var local_d_beta = Scalar[dtype](0.0)
        var local_sum_dy_g = Scalar[dtype](0.0)
        var local_sum_dy_g_xh = Scalar[dtype](0.0)

        var idx = tid
        while idx < BATCH:
            var dy = rebind[Scalar[dtype]](grad_output[idx, f])
            var xh = rebind[Scalar[dtype]](cache[idx, Self.XHAT_OFF + f])
            local_d_gamma += dy * xh
            local_d_beta += dy
            local_sum_dy_g += dy * gamma
            local_sum_dy_g_xh += dy * gamma * xh
            idx += TPB

        # Reduce d_gamma
        smem[tid] = local_d_gamma
        barrier()
        var stride = TPB // 2
        while stride > 0:
            if tid < stride:
                smem[tid] = smem[tid] + smem[tid + stride]
            barrier()
            stride = stride // 2
        var d_gamma = rebind[Scalar[dtype]](smem[0])
        barrier()

        # Reduce d_beta
        smem[tid] = local_d_beta
        barrier()
        stride = TPB // 2
        while stride > 0:
            if tid < stride:
                smem[tid] = smem[tid] + smem[tid + stride]
            barrier()
            stride = stride // 2
        var d_beta = rebind[Scalar[dtype]](smem[0])
        barrier()

        # Reduce sum_dy_g
        smem[tid] = local_sum_dy_g
        barrier()
        stride = TPB // 2
        while stride > 0:
            if tid < stride:
                smem[tid] = smem[tid] + smem[tid + stride]
            barrier()
            stride = stride // 2
        var sum_dy_g = rebind[Scalar[dtype]](smem[0])
        barrier()

        # Reduce sum_dy_g_xh
        smem[tid] = local_sum_dy_g_xh
        barrier()
        stride = TPB // 2
        while stride > 0:
            if tid < stride:
                smem[tid] = smem[tid] + smem[tid + stride]
            barrier()
            stride = stride // 2
        var sum_dy_g_xh = rebind[Scalar[dtype]](smem[0])
        barrier()

        # Accumulate param grads (thread 0 only)
        if tid == 0:
            grads.ptr[Self.GAMMA_OFF + f] = (
                rebind[Scalar[dtype]](grads[Self.GAMMA_OFF + f]) + d_gamma
            )
            grads.ptr[Self.BETA_OFF + f] = (
                rebind[Scalar[dtype]](grads[Self.BETA_OFF + f]) + d_beta
            )

        # Pass 2: Compute grad_input (parallel scatter)
        idx = tid
        while idx < BATCH:
            var dy = rebind[Scalar[dtype]](grad_output[idx, f])
            var xh = rebind[Scalar[dtype]](cache[idx, Self.XHAT_OFF + f])
            grad_input[idx, f] = inv_std * (
                dy * gamma - sum_dy_g / n_f - xh * sum_dy_g_xh / n_f
            )
            idx += TPB

    # =========================================================================
    # GPU Launchers
    # =========================================================================

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
        """GPU training forward: batch stats + running stats update."""
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](input.ptr)

        @parameter
        @always_inline
        def kernel_wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
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
        ):
            Self.forward_kernel_impl[BATCH, dtype](output, input, params, state, cache)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            output,
            input_immut,
            params,
            state,  # mutable — EMA-updated in kernel
            cache,
            grid_dim=(Self.dim,),
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
        """GPU inference forward: running stats, no update."""
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](input.ptr)
        var params_immut = LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ](params.ptr)
        var state_immut = LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), ImmutAnyOrigin
        ](state.ptr)

        @parameter
        @always_inline
        def kernel_wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
            params: LayoutTensor[
                dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
            ],
            state: LayoutTensor[
                dtype, Layout.row_major(Self.STATE_SIZE), ImmutAnyOrigin
            ],
        ):
            Self.forward_kernel_impl_no_cache[BATCH, dtype](
                output, input, params, state
            )

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            output,
            input_immut,
            params_immut,
            state_immut,
            grid_dim=(Self.dim,),
            block_dim=(TPB,),
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
        """GPU inference forward on stream — delegates to default stream."""
        Self.forward_gpu_no_cache[BATCH, dtype](ctx, output, input, params, state, workspace)

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
        """GPU backward pass."""
        var grad_output_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](grad_output.ptr)
        var params_immut = LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ](params.ptr)
        var cache_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ](cache.ptr)

        @parameter
        @always_inline
        def kernel_wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
            params: LayoutTensor[
                dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
            ],
            grads: LayoutTensor[
                dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
            ],
        ):
            Self.backward_kernel_impl[BATCH, dtype](
                grad_input, grad_output, params, cache, grads
            )

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            grad_input,
            grad_output_immut,
            params_immut,
            cache_immut,
            grads,
            grid_dim=(Self.dim,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # GPU inference-mode forward + backward (Phase 3.5)
    # =========================================================================
    # Forward uses running stats (no batch-stat reduction, no EMA update).
    # Backward applies the simpler `dx = γ·inv_std_r·dy` per-feature scale,
    # does NOT touch `grads` (BN params are frozen in inference mode; the
    # caller zeros their gradient slots).
    #
    # The `cache` parameter on the forward is unused — backward recomputes
    # `inv_std_r` directly from `state` so we don't need to materialize
    # anything per-sample.
    # =========================================================================

    @staticmethod
    def forward_gpu_inference_with_cache[
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
        """Inference forward: y = γ·(x−μ_r)·inv_std_r + β. No EMA, no cache writes."""
        Self.forward_gpu_no_cache[BATCH, dtype](
            ctx, output, input, params, state, workspace, perf, perf_slot
        )

    @always_inline
    @staticmethod
    def backward_inference_kernel_impl[
        BATCH: Int, dtype: DType = DType.float32,
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ],
        state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), ImmutAnyOrigin
        ],
    ):
        """Inference-mode backward kernel.

        Grid: (dim,), Block: (TPB,). One block per feature; threads scatter
        across BATCH applying the per-feature scale γ·inv_std_r.
        """
        var f = Int(block_idx.x)
        if f >= Self.dim:
            return
        var tid = Int(thread_idx.x)

        var eps = Scalar[dtype](Self.EPSILON)
        var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + f])
        var rvar = rebind[Scalar[dtype]](state[Self.RVAR_OFF + f])
        var inv_std: Scalar[dtype] = 1.0 / sqrt(rvar + eps)
        var scale = gamma * inv_std

        var idx = tid
        while idx < BATCH:
            var dy = rebind[Scalar[dtype]](grad_output[idx, f])
            grad_input[idx, f] = scale * dy
            idx += TPB

    @staticmethod
    def backward_gpu_inference[
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
        """Inference-mode GPU backward: dx = γ · inv_std_r · dy per feature.

        Does NOT touch `grads` — caller zeros BN gamma/beta slots if frozen.
        """
        var grad_output_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](grad_output.ptr)
        var params_immut = LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
        ](params.ptr)
        var state_immut = LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), ImmutAnyOrigin
        ](state.ptr)

        @parameter
        @always_inline
        def kernel_wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
            params: LayoutTensor[
                dtype, Layout.row_major(Self.PARAM_SIZE), ImmutAnyOrigin
            ],
            state: LayoutTensor[
                dtype, Layout.row_major(Self.STATE_SIZE), ImmutAnyOrigin
            ],
        ):
            Self.backward_inference_kernel_impl[BATCH, dtype](
                grad_input, grad_output, params, state
            )

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            grad_input,
            grad_output_immut,
            params_immut,
            state_immut,
            grid_dim=(Self.dim,),
            block_dim=(TPB,),
        )
