"""BatchNorm2D: Batch Normalization for convolutional layers.

Normalizes per-channel across batch × spatial dimensions, matching
the standard PyTorch/TF `BatchNormalization` used in AlphaZero.

  y = gamma * (x - batch_mean) / sqrt(batch_var + eps) + beta   (training)
  y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta  (inference)

Training mode = forward WITH cache (uses batch stats, updates running stats).
Inference mode = forward WITHOUT cache (uses running stats, no update).

Params layout: [gamma(C) | beta(C) | running_mean(C) | running_var(C)]
Cache layout per sample: [x_hat(C*S) | batch_mean(C) | batch_inv_std(C)]
  where S = H * W.

Running stats are stored in params but never receive gradients — the optimizer
leaves them unchanged, while forward (training) updates them via EMA.
"""

from ..constants import dtype, TPB
from .model import Model, PerfTimerPtr, NULL_PERF
from ..initializer import Initializer
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.memory import AddressSpace
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream
from std.math import sqrt


struct BatchNorm2D[
    channels: Int,
    height: Int,
    width: Int,
    MOMENTUM: Float64 = 0.1,
    EPSILON: Float64 = 1e-5,
](Model):
    """Batch Normalization for 2D convolutions (per-channel).

    Input: [BATCH, C*H*W] (flattened CHW from Conv2D).
    Output: [BATCH, C*H*W] (same shape).

    Parameters:
        channels: Number of channels (C).
        height: Spatial height (H).
        width: Spatial width (W).
        MOMENTUM: EMA momentum for running stats (default 0.1).
        EPSILON: Numerical stability epsilon (default 1e-5).
    """

    comptime S: Int = Self.height * Self.width  # Spatial size
    comptime IN_DIM: Int = Self.channels * Self.S
    comptime OUT_DIM: Int = Self.channels * Self.S
    comptime PARAM_SIZE: Int = 4 * Self.channels  # gamma, beta, running_mean, running_var
    comptime CACHE_SIZE: Int = Self.channels * Self.S + 2 * Self.channels  # x_hat + mean + inv_std
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = 0

    # Param offsets
    comptime GAMMA_OFF: Int = 0
    comptime BETA_OFF: Int = Self.channels
    comptime RMEAN_OFF: Int = 2 * Self.channels
    comptime RVAR_OFF: Int = 3 * Self.channels
    # Cache offsets (per sample)
    comptime XHAT_OFF: Int = 0
    comptime CMEAN_OFF: Int = Self.channels * Self.S
    comptime CINV_OFF: Int = Self.channels * Self.S + Self.channels

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
        """Initialize: gamma=1, beta=0, running_mean=0, running_var=1."""
        for c in range(Self.channels):
            params[Self.GAMMA_OFF + c] = Scalar[dtype](1.0)
            params[Self.BETA_OFF + c] = Scalar[dtype](0.0)
            params[Self.RMEAN_OFF + c] = Scalar[dtype](0.0)
            params[Self.RVAR_OFF + c] = Scalar[dtype](1.0)

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
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        """Training forward: normalize using batch stats, update running stats."""
        var eps = Scalar[dtype](Self.EPSILON)
        var mom = Scalar[dtype](Self.MOMENTUM)
        var one_m = Scalar[dtype](1.0) - mom
        var n = Scalar[dtype](BATCH * Self.S)

        for c in range(Self.channels):
            var c_off = c * Self.S
            var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + c])
            var beta = rebind[Scalar[dtype]](params[Self.BETA_OFF + c])

            # 1. Compute batch mean for channel c
            var mean = Scalar[dtype](0.0)
            for b in range(BATCH):
                for s in range(Self.S):
                    mean += rebind[Scalar[dtype]](input[b, c_off + s])
            mean = mean / n

            # 2. Compute batch variance for channel c
            var var_ = Scalar[dtype](0.0)
            for b in range(BATCH):
                for s in range(Self.S):
                    var diff = rebind[Scalar[dtype]](input[b, c_off + s]) - mean
                    var_ += diff * diff
            var_ = var_ / n

            # 3. Normalize, scale, shift
            var inv_std = Scalar[dtype](1.0) / Scalar[dtype](
                sqrt(Float64(var_ + eps))
            )
            for b in range(BATCH):
                for s in range(Self.S):
                    var x = rebind[Scalar[dtype]](input[b, c_off + s])
                    var x_hat = (x - mean) * inv_std
                    cache[b, Self.XHAT_OFF + c_off + s] = x_hat
                    output[b, c_off + s] = gamma * x_hat + beta
                # Store per-sample (replicated for simplicity)
                cache[b, Self.CMEAN_OFF + c] = mean
                cache[b, Self.CINV_OFF + c] = inv_std

            # 4. Update running stats (EMA)
            var rm = rebind[Scalar[dtype]](params[Self.RMEAN_OFF + c])
            var rv = rebind[Scalar[dtype]](params[Self.RVAR_OFF + c])
            params.ptr[Self.RMEAN_OFF + c] = one_m * rm + mom * mean
            params.ptr[Self.RVAR_OFF + c] = one_m * rv + mom * var_

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
    ):
        """Inference forward: always uses batch stats for normalization.

        This ensures BatchNorm normalizes properly even when running stats
        are not yet populated (e.g., fresh network in MCTS evaluation).
        """
        var eps = Scalar[dtype](Self.EPSILON)
        var n = Scalar[dtype](BATCH * Self.S)

        for c in range(Self.channels):
            var c_off = c * Self.S
            var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + c])
            var beta = rebind[Scalar[dtype]](params[Self.BETA_OFF + c])

            # Compute batch mean
            var mean = Scalar[dtype](0.0)
            for b in range(BATCH):
                for s in range(Self.S):
                    mean += rebind[Scalar[dtype]](input[b, c_off + s])
            mean = mean / n

            # Compute batch variance
            var var_ = Scalar[dtype](0.0)
            for b in range(BATCH):
                for s in range(Self.S):
                    var diff = rebind[Scalar[dtype]](input[b, c_off + s]) - mean
                    var_ += diff * diff
            var_ = var_ / n

            var inv_std = Scalar[dtype](1.0) / Scalar[dtype](
                sqrt(Float64(var_ + eps))
            )

            for b in range(BATCH):
                for s in range(Self.S):
                    var x = rebind[Scalar[dtype]](input[b, c_off + s])
                    output[b, c_off + s] = gamma * (x - mean) * inv_std + beta

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
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Backward: compute grad_gamma, grad_beta, grad_input.

        Running stats (grads[2*C:4*C]) are never written — stays zero.
        """
        var n = Scalar[dtype](BATCH * Self.S)

        for c in range(Self.channels):
            var c_off = c * Self.S
            var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + c])
            # Read batch_inv_std from sample 0's cache (same for all samples)
            var inv_std = rebind[Scalar[dtype]](cache[0, Self.CINV_OFF + c])

            # Accumulate grad_gamma and grad_beta, and intermediate sums
            var d_gamma = Scalar[dtype](0.0)
            var d_beta = Scalar[dtype](0.0)
            var sum_dy_gamma = Scalar[dtype](0.0)
            var sum_dy_gamma_xhat = Scalar[dtype](0.0)

            for b in range(BATCH):
                for s in range(Self.S):
                    var dy = rebind[Scalar[dtype]](grad_output[b, c_off + s])
                    var x_hat = rebind[Scalar[dtype]](
                        cache[b, Self.XHAT_OFF + c_off + s]
                    )
                    d_gamma += dy * x_hat
                    d_beta += dy
                    sum_dy_gamma += dy * gamma
                    sum_dy_gamma_xhat += dy * gamma * x_hat

            # Accumulate into param grads (only gamma and beta)
            grads.ptr[Self.GAMMA_OFF + c] = (
                rebind[Scalar[dtype]](grads[Self.GAMMA_OFF + c]) + d_gamma
            )
            grads.ptr[Self.BETA_OFF + c] = (
                rebind[Scalar[dtype]](grads[Self.BETA_OFF + c]) + d_beta
            )
            # grads[RMEAN_OFF + c] and grads[RVAR_OFF + c] stay zero

            # Compute grad_input
            for b in range(BATCH):
                for s in range(Self.S):
                    var dy = rebind[Scalar[dtype]](grad_output[b, c_off + s])
                    var x_hat = rebind[Scalar[dtype]](
                        cache[b, Self.XHAT_OFF + c_off + s]
                    )
                    var dx = inv_std * (
                        dy * gamma
                        - sum_dy_gamma / n
                        - x_hat * sum_dy_gamma_xhat / n
                    )
                    grad_input[b, c_off + s] = dx

    # =========================================================================
    # GPU Kernel Implementations
    # =========================================================================
    #
    # One block per channel. Single thread per block (like LayerNorm).
    # Each thread processes all BATCH * H * W elements for its channel.
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
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        """Training forward kernel. Grid: (channels,), Block: (TPB,).

        Block-parallel reduction across BATCH * spatial per channel.
        """
        var c = Int(block_idx.x)
        if c >= Self.channels:
            return
        var tid = Int(thread_idx.x)

        var c_off = c * Self.S
        var eps = Scalar[dtype](Self.EPSILON)
        var mom = Scalar[dtype](Self.MOMENTUM)
        var one_m = Scalar[dtype](1.0) - mom
        var n_f = Scalar[dtype](BATCH * Self.S)
        var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + c])
        var beta = rebind[Scalar[dtype]](params[Self.BETA_OFF + c])

        var smem = LayoutTensor[
            dtype, Layout.row_major(TPB), MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()

        # Pass 1: Compute mean via block reduction
        var local_sum = Scalar[dtype](0.0)
        var idx = tid
        while idx < BATCH * Self.S:
            var b = idx // Self.S
            var s = idx % Self.S
            local_sum += rebind[Scalar[dtype]](input[b, c_off + s])
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
        while idx < BATCH * Self.S:
            var b = idx // Self.S
            var s = idx % Self.S
            var diff = rebind[Scalar[dtype]](input[b, c_off + s]) - mean
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
        while idx < BATCH * Self.S:
            var b = idx // Self.S
            var s = idx % Self.S
            var x = rebind[Scalar[dtype]](input[b, c_off + s])
            var x_hat = (x - mean) * inv_std
            cache[b, Self.XHAT_OFF + c_off + s] = x_hat
            output[b, c_off + s] = gamma * x_hat + beta
            idx += TPB

        # Store cache stats and update running stats (thread 0 only)
        if tid == 0:
            for b in range(BATCH):
                cache[b, Self.CMEAN_OFF + c] = mean
                cache[b, Self.CINV_OFF + c] = inv_std
            var rm = rebind[Scalar[dtype]](params[Self.RMEAN_OFF + c])
            var rv = rebind[Scalar[dtype]](params[Self.RVAR_OFF + c])
            params.ptr[Self.RMEAN_OFF + c] = one_m * rm + mom * mean
            params.ptr[Self.RVAR_OFF + c] = one_m * rv + mom * var_

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
    ):
        """Inference forward kernel using batch stats (not running stats).

        Always computes batch statistics even in no-cache mode.
        This ensures BatchNorm normalizes properly during MCTS evaluation
        where running stats may not be populated yet.

        Grid: (channels,), Block: (TPB,).
        Block-parallel reduction across BATCH * spatial per channel.
        """
        var c = Int(block_idx.x)
        if c >= Self.channels:
            return
        var tid = Int(thread_idx.x)

        var c_off = c * Self.S
        var eps = Scalar[dtype](Self.EPSILON)
        var n_f = Scalar[dtype](BATCH * Self.S)
        var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + c])
        var beta = rebind[Scalar[dtype]](params[Self.BETA_OFF + c])

        var smem = LayoutTensor[
            dtype, Layout.row_major(TPB), MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()

        # Pass 1: Compute mean
        var local_sum = Scalar[dtype](0.0)
        var idx = tid
        while idx < BATCH * Self.S:
            var b = idx // Self.S
            var s = idx % Self.S
            local_sum += rebind[Scalar[dtype]](input[b, c_off + s])
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

        # Pass 2: Compute variance
        var local_var = Scalar[dtype](0.0)
        idx = tid
        while idx < BATCH * Self.S:
            var b = idx // Self.S
            var s = idx % Self.S
            var diff = rebind[Scalar[dtype]](input[b, c_off + s]) - mean
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

        # Pass 3: Normalize (parallel scatter)
        idx = tid
        while idx < BATCH * Self.S:
            var b = idx // Self.S
            var s = idx % Self.S
            var x = rebind[Scalar[dtype]](input[b, c_off + s])
            output[b, c_off + s] = gamma * (x - mean) * inv_std + beta
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
        """Backward kernel. Grid: (channels,), Block: (TPB,).

        Block-parallel reduction for gradient accumulation.
        """
        var c = Int(block_idx.x)
        if c >= Self.channels:
            return
        var tid = Int(thread_idx.x)

        var c_off = c * Self.S
        var n_f = Scalar[dtype](BATCH * Self.S)
        var gamma = rebind[Scalar[dtype]](params[Self.GAMMA_OFF + c])
        var inv_std = rebind[Scalar[dtype]](cache[0, Self.CINV_OFF + c])

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
        while idx < BATCH * Self.S:
            var b = idx // Self.S
            var s = idx % Self.S
            var dy = rebind[Scalar[dtype]](grad_output[b, c_off + s])
            var xh = rebind[Scalar[dtype]](cache[b, Self.XHAT_OFF + c_off + s])
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
            grads.ptr[Self.GAMMA_OFF + c] = (
                rebind[Scalar[dtype]](grads[Self.GAMMA_OFF + c]) + d_gamma
            )
            grads.ptr[Self.BETA_OFF + c] = (
                rebind[Scalar[dtype]](grads[Self.BETA_OFF + c]) + d_beta
            )

        # Pass 2: Compute grad_input (parallel scatter)
        idx = tid
        while idx < BATCH * Self.S:
            var b = idx // Self.S
            var s = idx % Self.S
            var dy = rebind[Scalar[dtype]](grad_output[b, c_off + s])
            var xh = rebind[Scalar[dtype]](cache[b, Self.XHAT_OFF + c_off + s])
            grad_input[b, c_off + s] = inv_std * (
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
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
            ],
        ):
            Self.forward_kernel_impl[BATCH, dtype](output, input, params, cache)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            output,
            input_immut,
            params,  # Mutable — running stats updated in kernel
            cache,
            grid_dim=(Self.channels,),
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
        ):
            Self.forward_kernel_impl_no_cache[BATCH, dtype](output, input, params)

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            output,
            input_immut,
            params_immut,
            grid_dim=(Self.channels,),
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
        workspace: DeviceBuffer[dtype],
    ) raises:
        """GPU inference forward on stream — delegates to default stream."""
        Self.forward_gpu_no_cache[BATCH, dtype](ctx, output, input, params, workspace)

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
            grid_dim=(Self.channels,),
            block_dim=(TPB,),
        )
