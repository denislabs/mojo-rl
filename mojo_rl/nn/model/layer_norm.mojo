from ..constants import dtype
from .model import Model, PerfTimerPtr, NULL_PERF
from ..initializer import Initializer
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream
from std.gpu.primitives import block
from std.math import sqrt
from ..constants import TPB


struct LayerNorm[dim: Int, EPSILON: Float64 = 1e-5](Model):
    """Layer Normalization: y = gamma * (x - mean) / sqrt(var + eps) + beta.

    Normalizes across the feature dimension (last dimension).

    Parameters (stored in params tensor):
    - gamma [dim]: Scale parameter, initialized to 1.0
    - beta [dim]: Shift parameter, initialized to 0.0

    PARAM_SIZE = 2 * dim (gamma + beta)
    CACHE_SIZE = dim + 2 (normalized values + inv_std + mean per sample)

    Layout:
    - params: [gamma (dim) | beta (dim)]
    - cache: [normalized (dim) | inv_std (1) | mean (1)] per sample
    """

    comptime IN_DIM: Int = Self.dim
    comptime OUT_DIM: Int = Self.dim
    comptime PARAM_SIZE: Int = 2 * Self.dim  # gamma + beta
    comptime CACHE_SIZE: Int = Self.dim + 2  # normalized + inv_std + mean
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = 0  # Leaf layer

    @staticmethod
    def initialize_params[
        INIT: Initializer, dtype: DType = DType.float32
    ](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Standard LayerNorm init: gamma = 1, beta = 0 (INIT is ignored).

        Previously called INIT on the whole parameter buffer, which gave
        gamma a random distribution (wrong — breaks the layer's identity
        behavior at init: `y = γ·x̂ + β` only reduces to `x̂` when `γ=1, β=0`).
        """
        # gamma [0:dim] = 1.0
        for i in range(Self.dim):
            params.ptr[i] = Scalar[dtype](1.0)
        # beta [dim:2*dim] = 0.0
        for i in range(Self.dim):
            params.ptr[Self.dim + i] = Scalar[dtype](0.0)

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
        mut state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        """Forward: y = gamma * normalize(x) + beta.

        Caches normalized values, inv_std, and mean for backward.
        """
        var eps = Scalar[dtype](Self.EPSILON)
        var n = Scalar[dtype](Self.dim)

        for batch in range(BATCH):
            # Compute mean
            var mean = rebind[Scalar[dtype]](input[batch, 0])
            for i in range(1, Self.dim):
                mean = mean + rebind[Scalar[dtype]](input[batch, i])
            mean = mean / n

            # Compute variance
            var diff0 = rebind[Scalar[dtype]](input[batch, 0]) - mean
            var var_ = diff0 * diff0
            for i in range(1, Self.dim):
                var diff = rebind[Scalar[dtype]](input[batch, i]) - mean
                var_ = var_ + diff * diff
            var_ = var_ / n

            # Compute inv_std
            var inv_std = Scalar[dtype](1.0 / sqrt(Float64(var_ + eps)))

            # Normalize and apply affine transform
            for i in range(Self.dim):
                var x_val = rebind[Scalar[dtype]](input[batch, i])
                var normalized = (x_val - mean) * inv_std
                # Cache normalized value
                cache[batch, i] = normalized
                # gamma at offset 0, beta at offset dim
                var gamma = rebind[Scalar[dtype]](params[i])
                var beta = rebind[Scalar[dtype]](params[Self.dim + i])
                output[batch, i] = gamma * normalized + beta

            # Cache inv_std and mean
            cache[batch, Self.dim] = inv_std
            cache[batch, Self.dim + 1] = mean

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
        mut state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
    ):
        """Forward pass without caching (for inference)."""
        var eps = Scalar[dtype](Self.EPSILON)
        var n = Scalar[dtype](Self.dim)

        for batch in range(BATCH):
            # Compute mean
            var mean = input[batch, 0]
            for i in range(1, Self.dim):
                mean = mean + input[batch, i]
            mean = mean / n

            # Compute variance
            var diff0 = input[batch, 0] - mean
            var var_ = diff0 * diff0
            for i in range(1, Self.dim):
                var diff = input[batch, i] - mean
                var_ = var_ + diff * diff
            var_ = var_ / n

            # Compute inv_std
            var inv_std: output.element_type = 1.0 / sqrt(var_ + eps)

            # Normalize and apply affine transform
            for i in range(Self.dim):
                var x_val = input[batch, i]
                var normalized = (x_val - mean) * inv_std
                var gamma = params[i]
                var beta = params[Self.dim + i]
                output[batch, i] = gamma * normalized + beta

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
        mut state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Backward pass for LayerNorm.

        Computes gradients for:
        - gamma (grads[0:dim])
        - beta (grads[dim:2*dim])
        - input (grad_input)
        """
        var n = Scalar[dtype](Self.dim)

        # First, accumulate parameter gradients across batch
        for batch in range(BATCH):
            for i in range(Self.dim):
                var normalized = cache[batch, i]
                var dy = grad_output[batch, i]

                # dgamma += dy * normalized
                var old_dgamma = grads[i]
                grads[i] = old_dgamma + dy * normalized
                # dbeta += dy
                var old_dbeta = grads[Self.dim + i]
                grads[Self.dim + i] = old_dbeta + dy

        # Then compute input gradients
        for batch in range(BATCH):
            var inv_std = cache[batch, Self.dim]

            # Compute intermediate values for this sample
            var sum_dy_gamma: grad_output.element_type = 0.0
            var sum_dy_gamma_norm: grad_output.element_type = 0.0

            for i in range(Self.dim):
                var gamma = params[i]
                var dy = grad_output[batch, i]
                var normalized = cache[batch, i]
                sum_dy_gamma = sum_dy_gamma + dy * gamma
                sum_dy_gamma_norm = sum_dy_gamma_norm + dy * gamma * normalized

            # Compute input gradients
            for i in range(Self.dim):
                var gamma = params[i]
                var dy = grad_output[batch, i]
                var normalized = cache[batch, i]

                var dx = inv_std * (
                    dy * gamma
                    - sum_dy_gamma / n
                    - normalized * sum_dy_gamma_norm / n
                )
                grad_input[batch, i] = dx

    # =========================================================================
    # GPU Kernel Implementations
    # =========================================================================
    #
    # LayerNorm requires per-sample statistics (mean, var), so we parallelize
    # over batches. Each block handles one sample.
    # =========================================================================

    @always_inline
    @staticmethod
    def forward_kernel_impl[
        BATCH: Int,
        dtype: DType = DType.float32,
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(2 * Self.dim), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim + 2), MutAnyOrigin
        ],
        eps: Scalar[dtype],
    ):
        """Forward pass kernel with caching.

        Grid: (BATCH,), Block: (TPB,). One block per sample, tree-reduces
        the per-dim sums across threads. Each thread handles a stride of
        TPB feature indices via `idx += TPB` (so `dim > TPB` is fine).
        """
        var b = Int(block_idx.x)
        var local_i = Int(thread_idx.x)

        if b >= BATCH:
            return

        var inv_dim = Scalar[dtype](1.0 / Float64(Self.dim))

        # Phase 1: compute mean (block-parallel reduction + broadcast)
        var my_sum = Scalar[dtype](0)
        var idx = local_i
        while idx < Self.dim:
            my_sum += rebind[Scalar[dtype]](input[b, idx])
            idx += TPB
        var mean_val = (
            block.sum[block_size=TPB, broadcast=True](val=my_sum) * inv_dim
        )

        # Phase 2: compute variance (block-parallel)
        var my_var = Scalar[dtype](0)
        idx = local_i
        while idx < Self.dim:
            var diff = rebind[Scalar[dtype]](input[b, idx]) - mean_val
            my_var += diff * diff
            idx += TPB
        var var_val = (
            block.sum[block_size=TPB, broadcast=True](val=my_var) * inv_dim
        )

        var inv_std = Scalar[dtype](1.0) / sqrt(var_val + eps)

        # Thread 0 writes inv_std and mean to cache
        if local_i == 0:
            cache[b, Self.dim] = inv_std
            cache[b, Self.dim + 1] = mean_val

        # Phase 3: normalize, scale, shift (parallel across features)
        idx = local_i
        while idx < Self.dim:
            var x_hat = (
                rebind[Scalar[dtype]](input[b, idx]) - mean_val
            ) * inv_std
            cache[b, idx] = x_hat
            output[b, idx] = rebind[Scalar[dtype]](
                params[idx]
            ) * x_hat + rebind[Scalar[dtype]](params[Self.dim + idx])
            idx += TPB

    @always_inline
    @staticmethod
    def forward_kernel_impl_no_cache[
        BATCH: Int,
        dtype: DType = DType.float32,
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(2 * Self.dim), ImmutAnyOrigin
        ],
        eps: Scalar[dtype],
    ):
        """Forward pass kernel without caching.

        Grid: (BATCH,), Block: (TPB,). Same block-parallel pattern as
        forward_kernel_impl but skips the cache writes.
        """
        var b = Int(block_idx.x)
        var local_i = Int(thread_idx.x)

        if b >= BATCH:
            return

        var inv_dim = Scalar[dtype](1.0 / Float64(Self.dim))

        var my_sum = Scalar[dtype](0)
        var idx = local_i
        while idx < Self.dim:
            my_sum += rebind[Scalar[dtype]](input[b, idx])
            idx += TPB
        var mean_val = (
            block.sum[block_size=TPB, broadcast=True](val=my_sum) * inv_dim
        )

        var my_var = Scalar[dtype](0)
        idx = local_i
        while idx < Self.dim:
            var diff = rebind[Scalar[dtype]](input[b, idx]) - mean_val
            my_var += diff * diff
            idx += TPB
        var var_val = (
            block.sum[block_size=TPB, broadcast=True](val=my_var) * inv_dim
        )

        var inv_std = Scalar[dtype](1.0) / sqrt(var_val + eps)

        idx = local_i
        while idx < Self.dim:
            var x_hat = (
                rebind[Scalar[dtype]](input[b, idx]) - mean_val
            ) * inv_std
            output[b, idx] = rebind[Scalar[dtype]](
                params[idx]
            ) * x_hat + rebind[Scalar[dtype]](params[Self.dim + idx])
            idx += TPB

    @always_inline
    @staticmethod
    def backward_kernel_impl[
        BATCH: Int,
        dtype: DType = DType.float32,
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(2 * Self.dim), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim + 2), ImmutAnyOrigin
        ],
        grads: LayoutTensor[
            dtype, Layout.row_major(2 * Self.dim), MutAnyOrigin
        ],
    ):
        """Per-sample input-gradient kernel.

        Grid: (BATCH,), Block: (TPB,). Block-parallel reduction over the
        feature dim to compute `mean(dxhat)` and `mean(dxhat * xhat)`,
        then per-feature `dx` emission. Param grads are accumulated by a
        separate kernel (see backward_param_kernel_impl).
        """
        var b = Int(block_idx.x)
        var local_i = Int(thread_idx.x)

        if b >= BATCH:
            return

        var inv_dim = Scalar[dtype](1.0 / Float64(Self.dim))
        var inv_std = rebind[Scalar[dtype]](cache[b, Self.dim])

        # Phase 1: compute mean(dx_hat) and mean(dx_hat * x_hat)
        var my_dxhat = Scalar[dtype](0)
        var my_dxhat_xhat = Scalar[dtype](0)
        var idx = local_i
        while idx < Self.dim:
            var g = rebind[Scalar[dtype]](grad_output[b, idx])
            var gamma = rebind[Scalar[dtype]](params[idx])
            var x_hat = rebind[Scalar[dtype]](cache[b, idx])
            var dx_hat = g * gamma
            my_dxhat += dx_hat
            my_dxhat_xhat += dx_hat * x_hat
            idx += TPB

        var mean_dxhat = (
            block.sum[block_size=TPB, broadcast=True](val=my_dxhat) * inv_dim
        )
        var mean_dxhat_xhat = (
            block.sum[block_size=TPB, broadcast=True](val=my_dxhat_xhat)
            * inv_dim
        )

        # Phase 2: compute dx (parallel per feature)
        idx = local_i
        while idx < Self.dim:
            var g = rebind[Scalar[dtype]](grad_output[b, idx])
            var gamma = rebind[Scalar[dtype]](params[idx])
            var x_hat = rebind[Scalar[dtype]](cache[b, idx])
            var dx_hat = g * gamma
            grad_input[b, idx] = inv_std * (
                dx_hat - mean_dxhat - x_hat * mean_dxhat_xhat
            )
            idx += TPB

    @always_inline
    @staticmethod
    def backward_param_kernel_impl[
        BATCH: Int,
        dtype: DType = DType.float32,
    ](
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim + 2), ImmutAnyOrigin
        ],
        grads: LayoutTensor[
            dtype, Layout.row_major(2 * Self.dim), MutAnyOrigin
        ],
    ):
        """Dgamma/dbeta reduction across batch.

        Grid: (dim,), Block: (TPB,). One block per output feature column,
        block-parallel sum over the BATCH rows. Replaces the previous
        single-thread serial kernel (grid=(1,), block=(1,)).
        """
        var col = Int(block_idx.x)
        var local_i = Int(thread_idx.x)

        if col >= Self.dim:
            return

        var my_dgamma = Scalar[dtype](0)
        var my_dbeta = Scalar[dtype](0)
        var batch_idx = local_i
        while batch_idx < BATCH:
            var g = rebind[Scalar[dtype]](grad_output[batch_idx, col])
            var x_hat = rebind[Scalar[dtype]](cache[batch_idx, col])
            my_dgamma += g * x_hat
            my_dbeta += g
            batch_idx += TPB

        var total_dgamma = block.sum[block_size=TPB, broadcast=False](
            val=my_dgamma
        )
        var total_dbeta = block.sum[block_size=TPB, broadcast=False](
            val=my_dbeta
        )
        if local_i == 0:
            grads[col] = grads[col] + total_dgamma[0]
            grads[Self.dim + col] = grads[Self.dim + col] + total_dbeta[0]

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
        mut state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        mut cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """Launch forward pass on GPU with caching."""
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](input.ptr)
        var params_immut = LayoutTensor[
            dtype, Layout.row_major(2 * Self.dim), ImmutAnyOrigin
        ](params.ptr)
        var eps_scalar = Scalar[dtype](Self.EPSILON)

        @parameter
        @always_inline
        def kernel_wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ],
            params: LayoutTensor[
                dtype, Layout.row_major(2 * Self.dim), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim + 2), MutAnyOrigin
            ],
            eps: Scalar[dtype],
        ):
            Self.forward_kernel_impl[BATCH, dtype](
                output, input, params, cache, eps
            )

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            output,
            input_immut,
            params_immut,
            cache,
            eps_scalar,
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
        mut state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
        perf: PerfTimerPtr = NULL_PERF,
        perf_slot: Int = 0,
    ) raises:
        """Launch forward pass on GPU without caching (for inference)."""
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
        ](input.ptr)
        var params_immut = LayoutTensor[
            dtype, Layout.row_major(2 * Self.dim), ImmutAnyOrigin
        ](params.ptr)
        var eps_scalar = Scalar[dtype](Self.EPSILON)

        @parameter
        @always_inline
        def kernel_wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ],
            params: LayoutTensor[
                dtype, Layout.row_major(2 * Self.dim), ImmutAnyOrigin
            ],
            eps: Scalar[dtype],
        ):
            Self.forward_kernel_impl_no_cache[BATCH, dtype](
                output, input, params, eps
            )

        ctx.enqueue_function[kernel_wrapper, kernel_wrapper](
            output,
            input_immut,
            params_immut,
            eps_scalar,
            grid_dim=(BATCH,),
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
        mut state: LayoutTensor[
            dtype, Layout.row_major(Self.STATE_SIZE), MutAnyOrigin
        ],
        workspace: DeviceBuffer[dtype],
    ) raises:
        """GPU forward on stream — delegates to default stream."""
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
        mut state: LayoutTensor[
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
        var params_immut = LayoutTensor[
            dtype, Layout.row_major(2 * Self.dim), ImmutAnyOrigin
        ](params.ptr)
        var cache_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.dim + 2), ImmutAnyOrigin
        ](cache.ptr)

        @parameter
        @always_inline
        def kernel_wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ],
            params: LayoutTensor[
                dtype, Layout.row_major(2 * Self.dim), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim + 2), ImmutAnyOrigin
            ],
            grads: LayoutTensor[
                dtype, Layout.row_major(2 * Self.dim), MutAnyOrigin
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
            grid_dim=(BATCH,),
            block_dim=(TPB,),
        )

        # Param gradients: dgamma, dbeta accumulated over batch (single thread)
        @parameter
        @always_inline
        def param_kernel_wrapper(
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.dim + 2), ImmutAnyOrigin
            ],
            grads: LayoutTensor[
                dtype, Layout.row_major(2 * Self.dim), MutAnyOrigin
            ],
        ):
            Self.backward_param_kernel_impl[BATCH, dtype](
                grad_output, cache, grads
            )

        ctx.enqueue_function[param_kernel_wrapper, param_kernel_wrapper](
            grad_output_immut,
            cache_immut,
            grads,
            grid_dim=(Self.dim,),
            block_dim=(TPB,),
        )
