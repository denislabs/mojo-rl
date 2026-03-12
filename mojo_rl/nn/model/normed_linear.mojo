from ..constants import dtype, TILE, TPB
from .model import Model, PerfTimerPtr, NULL_PERF
from ..initializer import Initializer
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream
from std.gpu.memory import AddressSpace
from std.gpu.primitives import block
from std.math import sqrt, exp, log


struct NormedLinear[in_dim: Int, out_dim: Int, EPSILON: Float64 = 1e-5](Model):
    """Fused NormedLinear block: Linear → LayerNorm → Mish.

    The base building block for all TDMPC2 MLPs.
    Computes: y = Mish(LayerNorm(x @ W + b)) in a single monolithic layer,
    eliminating intermediate allocations and kernel launches.

    Parameters:
        in_dim: Input dimension.
        out_dim: Output dimension.
        EPSILON: LayerNorm epsilon for numerical stability.

    Params layout: [W (in*out) | b (out) | gamma (out) | beta (out)]
    PARAM_SIZE = in*out + 3*out

    Cache layout per sample:
        [input (in) | ln_normalized (out) | inv_std (1) | mean (1)
         | tanh_sp (out) | ln_output (out)]
    CACHE_SIZE = in + 3*out + 2

    WORKSPACE_SIZE_PER_SAMPLE = out (intermediate linear output for GPU)
    """

    comptime IN_DIM: Int = Self.in_dim
    comptime OUT_DIM: Int = Self.out_dim

    # Params: W[in*out] + b[out] + gamma[out] + beta[out]
    comptime PARAM_SIZE: Int = Self.in_dim * Self.out_dim + 3 * Self.out_dim

    # Cache per sample: input[in] + ln_normalized[out] + inv_std[1] + mean[1]
    #                  + tanh_sp[out] + ln_output[out]
    comptime CACHE_SIZE: Int = Self.in_dim + 3 * Self.out_dim + 2

    # Workspace for GPU intermediate linear output
    comptime WORKSPACE_SIZE_PER_SAMPLE: Int = Self.out_dim

    # ---- Param offsets ----
    comptime _W_OFFSET: Int = 0
    comptime _B_OFFSET: Int = Self.in_dim * Self.out_dim
    comptime _GAMMA_OFFSET: Int = Self.in_dim * Self.out_dim + Self.out_dim
    comptime _BETA_OFFSET: Int = Self.in_dim * Self.out_dim + 2 * Self.out_dim

    # ---- Cache offsets (per sample) ----
    comptime _INPUT_OFFSET: Int = 0
    comptime _LN_NORM_OFFSET: Int = Self.in_dim
    comptime _INV_STD_OFFSET: Int = Self.in_dim + Self.out_dim
    comptime _MEAN_OFFSET: Int = Self.in_dim + Self.out_dim + 1
    comptime _TANH_SP_OFFSET: Int = Self.in_dim + Self.out_dim + 2
    comptime _LN_OUT_OFFSET: Int = Self.in_dim + 2 * Self.out_dim + 2

    fn __init__(out self):
        pass

    fn __init__(out self, *, deinit take: Self):
        pass

    fn __init__(out self, *, copy: Self):
        pass

    @staticmethod
    fn initialize_params[INIT: Initializer](
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        INIT.init[Self.PARAM_SIZE, Self.IN_DIM, Self.OUT_DIM](params)

    # =========================================================================
    # CPU Forward
    # =========================================================================

    @staticmethod
    fn forward[
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
        """Forward: y = Mish(LayerNorm(x @ W + b)) with caching."""
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](params.ptr)
        var eps = Scalar[dtype](Self.EPSILON)
        var n = Scalar[dtype](Self.out_dim)

        for batch in range(BATCH):
            # Cache input for dW computation
            for i in range(Self.in_dim):
                cache[batch, Self._INPUT_OFFSET + i] = input[batch, i]

            # --- Linear: z = x @ W + b ---
            for j in range(Self.out_dim):
                var acc = params[Self._B_OFFSET + j]  # bias
                for i in range(Self.in_dim):
                    acc += input[batch, i] * W[i, j]
                # Store linear output temporarily in the output buffer
                output[batch, j] = acc

            # --- LayerNorm: y = gamma * (z - mean) / sqrt(var + eps) + beta ---
            # Compute mean
            var mean = rebind[Scalar[dtype]](output[batch, 0])
            for j in range(1, Self.out_dim):
                mean = mean + rebind[Scalar[dtype]](output[batch, j])
            mean = mean / n

            # Compute variance
            var diff0 = rebind[Scalar[dtype]](output[batch, 0]) - mean
            var var_ = diff0 * diff0
            for j in range(1, Self.out_dim):
                var diff = rebind[Scalar[dtype]](output[batch, j]) - mean
                var_ = var_ + diff * diff
            var_ = var_ / n

            # Compute inv_std
            var inv_std = Scalar[dtype](1.0 / sqrt(Float64(var_ + eps)))

            # Normalize and apply affine transform
            for j in range(Self.out_dim):
                var z_val = rebind[Scalar[dtype]](output[batch, j])
                var normalized = (z_val - mean) * inv_std
                # Cache normalized for LN backward
                cache[batch, Self._LN_NORM_OFFSET + j] = normalized
                var gamma = rebind[Scalar[dtype]](params[Self._GAMMA_OFFSET + j])
                var beta = rebind[Scalar[dtype]](params[Self._BETA_OFFSET + j])
                var ln_out = gamma * normalized + beta
                # Store LN output for Mish backward
                cache[batch, Self._LN_OUT_OFFSET + j] = ln_out
                # Overwrite output with LN result (will be overwritten by Mish)
                output[batch, j] = ln_out

            # Cache inv_std and mean
            cache[batch, Self._INV_STD_OFFSET] = inv_std
            cache[batch, Self._MEAN_OFFSET] = mean

            # --- Mish: y = x * tanh(softplus(x)) ---
            for j in range(Self.out_dim):
                var x_val = Float64(rebind[Scalar[dtype]](output[batch, j]))
                var sp: Float64
                if x_val > 20.0:
                    sp = x_val
                else:
                    sp = log(1.0 + exp(x_val))
                var exp_sp = exp(sp)
                var exp_neg_sp = exp(-sp)
                var tanh_sp = (exp_sp - exp_neg_sp) / (exp_sp + exp_neg_sp)
                cache[batch, Self._TANH_SP_OFFSET + j] = Scalar[dtype](tanh_sp)
                output[batch, j] = Scalar[dtype](x_val * tanh_sp)

    @staticmethod
    fn forward[
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
    ):
        """Forward pass without caching (inference)."""
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](params.ptr)
        var eps = Scalar[dtype](Self.EPSILON)
        var n = Scalar[dtype](Self.out_dim)

        for batch in range(BATCH):
            # --- Linear ---
            for j in range(Self.out_dim):
                var acc = params[Self._B_OFFSET + j]
                for i in range(Self.in_dim):
                    acc += input[batch, i] * W[i, j]
                output[batch, j] = acc

            # --- LayerNorm ---
            var mean: output.element_type = output[batch, 0]
            for j in range(1, Self.out_dim):
                mean = mean + output[batch, j]
            mean = mean / n

            var diff0 = output[batch, 0] - mean
            var var_ = diff0 * diff0
            for j in range(1, Self.out_dim):
                var diff = output[batch, j] - mean
                var_ = var_ + diff * diff
            var_ = var_ / n

            var inv_std: output.element_type = 1.0 / sqrt(var_ + eps)

            for j in range(Self.out_dim):
                var normalized = (output[batch, j] - mean) * inv_std
                var gamma = params[Self._GAMMA_OFFSET + j]
                var beta = params[Self._BETA_OFFSET + j]
                output[batch, j] = gamma * normalized + beta

            # --- Mish ---
            for j in range(Self.out_dim):
                var x_val = Float64(rebind[Scalar[dtype]](output[batch, j]))
                var sp: Float64
                if x_val > 20.0:
                    sp = x_val
                else:
                    sp = log(1.0 + exp(x_val))
                var exp_sp = exp(sp)
                var exp_neg_sp = exp(-sp)
                var tanh_sp = (exp_sp - exp_neg_sp) / (exp_sp + exp_neg_sp)
                output[batch, j] = Scalar[dtype](x_val * tanh_sp)

    # =========================================================================
    # CPU Backward
    # =========================================================================

    @staticmethod
    fn backward[
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
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Backward through Mish → LayerNorm → Linear."""
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](params.ptr)
        var dW = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](grads.ptr)
        var n = Scalar[dtype](Self.out_dim)

        for batch in range(BATCH):
            # ---- Mish backward: d_ln_out = dy * d_mish ----
            # d_mish = tanh_sp + x * sigmoid(x) * (1 - tanh_sp^2)
            # where x = ln_output (input to Mish)
            for j in range(Self.out_dim):
                var tanh_sp = Float64(
                    rebind[Scalar[dtype]](
                        cache[batch, Self._TANH_SP_OFFSET + j]
                    )
                )
                var x_val = Float64(
                    rebind[Scalar[dtype]](
                        cache[batch, Self._LN_OUT_OFFSET + j]
                    )
                )
                var sigmoid_x = 1.0 / (1.0 + exp(-x_val))
                var d_mish = tanh_sp + x_val * sigmoid_x * (
                    1.0 - tanh_sp * tanh_sp
                )
                var dy = rebind[Scalar[dtype]](grad_output[batch, j])
                # Reuse grad_input temporarily? No, dimensions differ.
                # We'll store d_ln_out in output-sized temp. Use grad_output
                # buffer if mutable... but it's passed as non-mut for grads.
                # Actually grad_output IS mut in the signature. We can
                # overwrite it since we process each element once.
                # Store Mish gradient result back into grad_output buffer.
                grad_output[batch, j] = Scalar[dtype](Float64(dy) * d_mish)

            # ---- LayerNorm backward ----
            # grad_output now contains d_ln_out (gradient w.r.t. LN output)
            var inv_std = cache[batch, Self._INV_STD_OFFSET]

            # Accumulate dgamma, dbeta
            for j in range(Self.out_dim):
                var normalized = cache[batch, Self._LN_NORM_OFFSET + j]
                var dy = grad_output[batch, j]
                # dgamma += dy * normalized
                grads[Self._GAMMA_OFFSET + j] = (
                    grads[Self._GAMMA_OFFSET + j] + dy * normalized
                )
                # dbeta += dy
                grads[Self._BETA_OFFSET + j] = (
                    grads[Self._BETA_OFFSET + j] + dy
                )

            # Compute d_linear_out from LayerNorm backward
            var sum_dy_gamma: grad_output.element_type = 0.0
            var sum_dy_gamma_norm: grad_output.element_type = 0.0
            for j in range(Self.out_dim):
                var gamma = params[Self._GAMMA_OFFSET + j]
                var dy = grad_output[batch, j]
                var normalized = cache[batch, Self._LN_NORM_OFFSET + j]
                sum_dy_gamma = sum_dy_gamma + dy * gamma
                sum_dy_gamma_norm = (
                    sum_dy_gamma_norm + dy * gamma * normalized
                )

            # Now compute d_linear_out and use it for Linear backward inline
            # d_linear_out[j] = inv_std * (dy*gamma - sum_dy_gamma/n
            #                              - normalized * sum_dy_gamma_norm/n)
            # Then: dx = d_linear_out @ W.T, dW += x.T @ d_linear_out,
            #        db += d_linear_out

            # --- Linear backward: dx = d_linear_out @ W.T ---
            for i in range(Self.in_dim):
                var acc: grad_input.element_type = 0
                for j in range(Self.out_dim):
                    var gamma = params[Self._GAMMA_OFFSET + j]
                    var dy = grad_output[batch, j]
                    var normalized = cache[batch, Self._LN_NORM_OFFSET + j]
                    var d_lin = inv_std * (
                        dy * gamma
                        - sum_dy_gamma / n
                        - normalized * sum_dy_gamma_norm / n
                    )
                    acc += d_lin * W[i, j]
                grad_input[batch, i] = acc

            # --- Linear backward: dW += x.T @ d_linear_out ---
            for i in range(Self.in_dim):
                var cached_input = cache[batch, Self._INPUT_OFFSET + i]
                for j in range(Self.out_dim):
                    var gamma = params[Self._GAMMA_OFFSET + j]
                    var dy = grad_output[batch, j]
                    var normalized = cache[batch, Self._LN_NORM_OFFSET + j]
                    var d_lin = inv_std * (
                        dy * gamma
                        - sum_dy_gamma / n
                        - normalized * sum_dy_gamma_norm / n
                    )
                    dW[i, j] = dW[i, j] + cached_input * d_lin

            # --- Linear backward: db += d_linear_out ---
            for j in range(Self.out_dim):
                var gamma = params[Self._GAMMA_OFFSET + j]
                var dy = grad_output[batch, j]
                var normalized = cache[batch, Self._LN_NORM_OFFSET + j]
                var d_lin = inv_std * (
                    dy * gamma
                    - sum_dy_gamma / n
                    - normalized * sum_dy_gamma_norm / n
                )
                grads[Self._B_OFFSET + j] = grads[Self._B_OFFSET + j] + d_lin

    # =========================================================================
    # GPU Kernel Implementations
    # =========================================================================

    @always_inline
    @staticmethod
    fn forward_linear_kernel_impl[
        BATCH: Int,
    ](
        linear_out: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ],
        W: LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), ImmutAnyOrigin
        ],
        b: LayoutTensor[dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        """Tiled matmul: linear_out = input @ W + b, caches input.

        Grid: ((OUT_DIM + TILE - 1) // TILE, (BATCH + TILE - 1) // TILE)
        Block: (TILE, TILE)
        """
        var local_row = Int(thread_idx.y)
        var local_col = Int(thread_idx.x)
        var global_row = Int(block_idx.y) * TILE + local_row
        var global_col = Int(block_idx.x) * TILE + local_col

        var x_shared = LayoutTensor[
            dtype,
            Layout.row_major(TILE, TILE),
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ].stack_allocation()

        var W_shared = LayoutTensor[
            dtype,
            Layout.row_major(TILE, TILE),
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ].stack_allocation()

        var acc: linear_out.element_type = 0
        if global_col < Self.OUT_DIM:
            acc = b[global_col]

        for tile_idx in range((Self.IN_DIM + TILE - 1) // TILE):
            var x_col = tile_idx * TILE + local_col
            if global_row < BATCH and x_col < Self.IN_DIM:
                var x_val = input[global_row, x_col]
                x_shared[local_row, local_col] = x_val
                # Cache input (only first x-block to avoid races)
                if Int(block_idx.x) == 0:
                    cache[global_row, Self._INPUT_OFFSET + x_col] = x_val
            else:
                x_shared[local_row, local_col] = 0

            var W_row = tile_idx * TILE + local_row
            if W_row < Self.IN_DIM and global_col < Self.OUT_DIM:
                W_shared[local_row, local_col] = W[W_row, global_col]
            else:
                W_shared[local_row, local_col] = 0

            barrier()

            comptime for k in range(TILE):
                acc += x_shared[local_row, k] * W_shared[k, local_col]

            barrier()

        if global_row < BATCH and global_col < Self.OUT_DIM:
            linear_out[global_row, global_col] = acc

    @always_inline
    @staticmethod
    fn forward_linear_kernel_impl_no_cache[
        BATCH: Int,
    ](
        linear_out: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ],
        W: LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), ImmutAnyOrigin
        ],
        b: LayoutTensor[dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin],
    ):
        """Tiled matmul: linear_out = input @ W + b (no caching).

        Grid: ((OUT_DIM + TILE - 1) // TILE, (BATCH + TILE - 1) // TILE)
        Block: (TILE, TILE)
        """
        var local_row = Int(thread_idx.y)
        var local_col = Int(thread_idx.x)
        var global_row = Int(block_idx.y) * TILE + local_row
        var global_col = Int(block_idx.x) * TILE + local_col

        var x_shared = LayoutTensor[
            dtype,
            Layout.row_major(TILE, TILE),
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ].stack_allocation()

        var W_shared = LayoutTensor[
            dtype,
            Layout.row_major(TILE, TILE),
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ].stack_allocation()

        var acc: linear_out.element_type = 0
        if global_col < Self.OUT_DIM:
            acc = b[global_col]

        for tile_idx in range((Self.IN_DIM + TILE - 1) // TILE):
            var x_col = tile_idx * TILE + local_col
            if global_row < BATCH and x_col < Self.IN_DIM:
                x_shared[local_row, local_col] = input[global_row, x_col]
            else:
                x_shared[local_row, local_col] = 0

            var W_row = tile_idx * TILE + local_row
            if W_row < Self.IN_DIM and global_col < Self.OUT_DIM:
                W_shared[local_row, local_col] = W[W_row, global_col]
            else:
                W_shared[local_row, local_col] = 0

            barrier()

            comptime for k in range(TILE):
                acc += x_shared[local_row, k] * W_shared[k, local_col]

            barrier()

        if global_row < BATCH and global_col < Self.OUT_DIM:
            linear_out[global_row, global_col] = acc

    @always_inline
    @staticmethod
    fn forward_ln_mish_kernel_impl[
        BATCH: Int,
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        linear_out: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ],
        gamma: LayoutTensor[
            dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
        ],
        beta: LayoutTensor[
            dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
        eps: Scalar[dtype],
    ):
        """Fused LayerNorm + Mish kernel with caching.

        One block per sample (sequential over features).
        Grid: (BATCH,)
        Block: (1,)
        """
        var batch_idx = Int(block_idx.x)
        if batch_idx >= BATCH:
            return
        if thread_idx.x != 0:
            return

        var n = Scalar[dtype](Self.OUT_DIM)

        # Compute mean
        var mean: output.element_type = 0.0
        for j in range(Self.OUT_DIM):
            mean = mean + linear_out[batch_idx, j]
        mean = mean / n

        # Compute variance
        var var_: output.element_type = 0.0
        for j in range(Self.OUT_DIM):
            var diff = linear_out[batch_idx, j] - mean
            var_ = var_ + diff * diff
        var_ = var_ / n

        var inv_std: output.element_type = 1.0 / sqrt(var_ + eps)

        # LN + Mish fused
        for j in range(Self.OUT_DIM):
            var z_val = linear_out[batch_idx, j]
            var normalized = (z_val - mean) * inv_std
            cache[batch_idx, Self._LN_NORM_OFFSET + j] = normalized

            var ln_out = gamma[j] * normalized + beta[j]
            cache[batch_idx, Self._LN_OUT_OFFSET + j] = ln_out

            # Mish: y = x * tanh(softplus(x))
            var x_val = rebind[Scalar[DType.float32]](ln_out)
            var sp: Scalar[DType.float32]
            if x_val > 20.0:
                sp = x_val
            else:
                sp = log(1.0 + exp(x_val))
            var exp_sp = exp(sp)
            var exp_neg_sp = exp(-sp)
            var tanh_sp = (exp_sp - exp_neg_sp) / (exp_sp + exp_neg_sp)
            cache[batch_idx, Self._TANH_SP_OFFSET + j] = rebind[
                cache.element_type
            ](tanh_sp)
            output[batch_idx, j] = rebind[output.element_type](
                x_val * tanh_sp
            )

        cache[batch_idx, Self._INV_STD_OFFSET] = inv_std
        cache[batch_idx, Self._MEAN_OFFSET] = mean

    @always_inline
    @staticmethod
    fn forward_ln_mish_kernel_impl_no_cache[
        BATCH: Int,
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        linear_out: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ],
        gamma: LayoutTensor[
            dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
        ],
        beta: LayoutTensor[
            dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
        ],
        eps: Scalar[dtype],
    ):
        """Fused LayerNorm + Mish kernel without caching (inference).

        Grid: (BATCH,)
        Block: (1,)
        """
        var batch_idx = Int(block_idx.x)
        if batch_idx >= BATCH:
            return
        if thread_idx.x != 0:
            return

        var n = Scalar[dtype](Self.OUT_DIM)

        var mean: output.element_type = 0.0
        for j in range(Self.OUT_DIM):
            mean = mean + linear_out[batch_idx, j]
        mean = mean / n

        var var_: output.element_type = 0.0
        for j in range(Self.OUT_DIM):
            var diff = linear_out[batch_idx, j] - mean
            var_ = var_ + diff * diff
        var_ = var_ / n

        var inv_std: output.element_type = 1.0 / sqrt(var_ + eps)

        for j in range(Self.OUT_DIM):
            var z_val = linear_out[batch_idx, j]
            var normalized = (z_val - mean) * inv_std
            var ln_out = gamma[j] * normalized + beta[j]

            var x_val = rebind[Scalar[DType.float32]](ln_out)
            var sp: Scalar[DType.float32]
            if x_val > 20.0:
                sp = x_val
            else:
                sp = log(1.0 + exp(x_val))
            var exp_sp = exp(sp)
            var exp_neg_sp = exp(-sp)
            var tanh_sp = (exp_sp - exp_neg_sp) / (exp_sp + exp_neg_sp)
            output[batch_idx, j] = rebind[output.element_type](
                x_val * tanh_sp
            )

    @always_inline
    @staticmethod
    fn backward_mish_ln_kernel_impl[
        BATCH: Int,
    ](
        d_linear_out: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ],
        gamma: LayoutTensor[
            dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ],
        dgamma: LayoutTensor[
            dtype, Layout.row_major(Self.OUT_DIM), MutAnyOrigin
        ],
        dbeta: LayoutTensor[
            dtype, Layout.row_major(Self.OUT_DIM), MutAnyOrigin
        ],
    ):
        """Fused Mish backward + LayerNorm backward kernel.

        Computes d_linear_out and accumulates dgamma, dbeta.
        One block per sample.

        Grid: (BATCH,)
        Block: (1,)
        """
        var batch_idx = Int(block_idx.x)
        if batch_idx >= BATCH:
            return
        if thread_idx.x != 0:
            return

        var inv_std = cache[batch_idx, Self._INV_STD_OFFSET]
        var n = Scalar[dtype](Self.OUT_DIM)

        # First pass: compute Mish gradient and LN intermediate sums
        var sum_dy_gamma: d_linear_out.element_type = 0.0
        var sum_dy_gamma_norm: d_linear_out.element_type = 0.0

        for j in range(Self.OUT_DIM):
            # Mish backward
            var tanh_sp = rebind[Scalar[DType.float32]](
                cache[batch_idx, Self._TANH_SP_OFFSET + j]
            )
            var x_val = rebind[Scalar[DType.float32]](
                cache[batch_idx, Self._LN_OUT_OFFSET + j]
            )
            var sigmoid_x: Scalar[DType.float32] = 1.0 / (1.0 + exp(-x_val))
            var d_mish = tanh_sp + x_val * sigmoid_x * (
                1.0 - tanh_sp * tanh_sp
            )
            var dy = rebind[Scalar[DType.float32]](
                grad_output[batch_idx, j]
            )
            var d_ln_out = dy * d_mish  # gradient w.r.t. LN output

            # Accumulate dgamma, dbeta
            var normalized = cache[batch_idx, Self._LN_NORM_OFFSET + j]
            dgamma[j] = dgamma[j] + rebind[dgamma.element_type](
                d_ln_out
            ) * normalized
            dbeta[j] = dbeta[j] + rebind[dbeta.element_type](d_ln_out)

            # LN backward intermediate sums
            var g = gamma[j]
            sum_dy_gamma = sum_dy_gamma + rebind[d_linear_out.element_type](
                d_ln_out
            ) * g
            sum_dy_gamma_norm = sum_dy_gamma_norm + rebind[
                d_linear_out.element_type
            ](d_ln_out) * g * normalized

        # Second pass: compute d_linear_out
        for j in range(Self.OUT_DIM):
            # Recompute d_ln_out (Mish gradient)
            var tanh_sp = rebind[Scalar[DType.float32]](
                cache[batch_idx, Self._TANH_SP_OFFSET + j]
            )
            var x_val = rebind[Scalar[DType.float32]](
                cache[batch_idx, Self._LN_OUT_OFFSET + j]
            )
            var sigmoid_x: Scalar[DType.float32] = 1.0 / (1.0 + exp(-x_val))
            var d_mish = tanh_sp + x_val * sigmoid_x * (
                1.0 - tanh_sp * tanh_sp
            )
            var dy = rebind[Scalar[DType.float32]](
                grad_output[batch_idx, j]
            )
            var d_ln_out = rebind[d_linear_out.element_type](dy * d_mish)

            var g = gamma[j]
            var normalized = cache[batch_idx, Self._LN_NORM_OFFSET + j]

            var dx = inv_std * (
                d_ln_out * g
                - sum_dy_gamma / n
                - normalized * sum_dy_gamma_norm / n
            )
            d_linear_out[batch_idx, j] = dx

    @always_inline
    @staticmethod
    fn backward_linear_fused_kernel_impl[
        BATCH: Int,
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        dW: LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), MutAnyOrigin
        ],
        db: LayoutTensor[dtype, Layout.row_major(Self.OUT_DIM), MutAnyOrigin],
        d_linear_out: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ],
        W: LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ],
    ):
        """Fused Linear backward: dx, dW, db from d_linear_out.

        Grid partitioning:
        - Rows [0, dx_grid_y): compute grad_input = d_linear_out @ W.T
        - Rows [dx_grid_y, dx_grid_y + dW_grid_y): compute dW + db

        Grid: (max(dx_grid_x, dW_grid_x), dx_grid_y + dW_grid_y)
        Block: (TILE, TILE)
        """
        var local_row = Int(thread_idx.y)
        var local_col = Int(thread_idx.x)
        var block_y = Int(block_idx.y)
        var block_x = Int(block_idx.x)

        comptime dx_grid_x = (Self.IN_DIM + TILE - 1) // TILE
        comptime dx_grid_y = (BATCH + TILE - 1) // TILE
        comptime dW_grid_x = (Self.OUT_DIM + TILE - 1) // TILE
        comptime dW_grid_y = (Self.IN_DIM + TILE - 1) // TILE

        var shared_A = LayoutTensor[
            dtype,
            Layout.row_major(TILE, TILE),
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ].stack_allocation()

        var shared_B = LayoutTensor[
            dtype,
            Layout.row_major(TILE, TILE),
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ].stack_allocation()

        if block_y < dx_grid_y:
            # === dx: grad_input = d_linear_out @ W.T ===
            if block_x >= dx_grid_x:
                return

            var global_row = block_y * TILE + local_row
            var global_col = block_x * TILE + local_col

            var acc: grad_input.element_type = 0

            for tile_idx in range((Self.OUT_DIM + TILE - 1) // TILE):
                var dy_col = tile_idx * TILE + local_col
                if global_row < BATCH and dy_col < Self.OUT_DIM:
                    shared_A[local_row, local_col] = d_linear_out[
                        global_row, dy_col
                    ]
                else:
                    shared_A[local_row, local_col] = 0

                var W_col = tile_idx * TILE + local_row
                if W_col < Self.OUT_DIM and global_col < Self.IN_DIM:
                    shared_B[local_row, local_col] = W[global_col, W_col]
                else:
                    shared_B[local_row, local_col] = 0

                barrier()

                comptime for k in range(TILE):
                    acc += shared_A[local_row, k] * shared_B[k, local_col]

                barrier()

            if global_row < BATCH and global_col < Self.IN_DIM:
                grad_input[global_row, global_col] = acc

        else:
            # === dW + db ===
            var dW_block_y = block_y - dx_grid_y
            var dW_block_x = block_x

            if dW_block_y >= dW_grid_y or dW_block_x >= dW_grid_x:
                return

            var global_row = dW_block_y * TILE + local_row  # IN_DIM
            var global_col = dW_block_x * TILE + local_col  # OUT_DIM

            var dW_acc: dW.element_type = 0
            var db_acc: db.element_type = 0

            var num_tiles = (BATCH + TILE - 1) // TILE
            for tile_idx in range(num_tiles):
                # input.T tile (from cache)
                var batch_idx = tile_idx * TILE + local_col
                if global_row < Self.IN_DIM and batch_idx < BATCH:
                    shared_A[local_row, local_col] = cache[
                        batch_idx, Self._INPUT_OFFSET + global_row
                    ]
                else:
                    shared_A[local_row, local_col] = 0

                # d_linear_out tile
                var dy_row = tile_idx * TILE + local_row
                if dy_row < BATCH and global_col < Self.OUT_DIM:
                    var grad_val = d_linear_out[dy_row, global_col]
                    shared_B[local_row, local_col] = grad_val
                    if dW_block_y == 0:
                        db_acc += grad_val
                else:
                    shared_B[local_row, local_col] = 0

                barrier()

                comptime for k in range(TILE):
                    dW_acc += shared_A[local_row, k] * shared_B[k, local_col]

                barrier()

            if global_row < Self.IN_DIM and global_col < Self.OUT_DIM:
                dW[global_row, global_col] = dW_acc

            # db reduction
            if dW_block_y == 0 and global_col < Self.OUT_DIM:
                shared_A[local_row, local_col] = db_acc
                barrier()

                if local_row == 0:
                    var total = shared_A[0, local_col]
                    for r in range(1, TILE):
                        total += shared_A[r, local_col]
                    db[global_col] = total

    # =========================================================================
    # GPU Launchers
    # =========================================================================

    @staticmethod
    fn forward_gpu[
        BATCH: Int,
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
        """GPU forward with caching: Linear kernel → fused LN+Mish kernel."""
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), ImmutAnyOrigin
        ](params.ptr)
        var b = LayoutTensor[
            dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
        ](params.ptr + Self._B_OFFSET)
        var gamma = LayoutTensor[
            dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
        ](params.ptr + Self._GAMMA_OFFSET)
        var beta = LayoutTensor[
            dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
        ](params.ptr + Self._BETA_OFFSET)
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](input.ptr)

        # Workspace as intermediate linear output
        var linear_out_mut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ](workspace.unsafe_ptr())
        var linear_out_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ](workspace.unsafe_ptr())

        var eps_scalar = Scalar[dtype](Self.EPSILON)

        # Kernel 1: Linear matmul
        comptime grid_x = (Self.OUT_DIM + TILE - 1) // TILE
        comptime grid_y = (BATCH + TILE - 1) // TILE

        @always_inline
        fn linear_wrapper(
            linear_out: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
            W: LayoutTensor[
                dtype,
                Layout.row_major(Self.IN_DIM, Self.OUT_DIM),
                ImmutAnyOrigin,
            ],
            b: LayoutTensor[
                dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
            ],
        ):
            Self.forward_linear_kernel_impl[BATCH](
                linear_out, input, W, b, cache
            )

        ctx.enqueue_function[linear_wrapper, linear_wrapper](
            linear_out_mut,
            input_immut,
            W,
            b,
            cache,
            grid_dim=(grid_x, grid_y),
            block_dim=(TILE, TILE),
        )

        # Kernel 2: Fused LayerNorm + Mish
        @always_inline
        fn ln_mish_wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ],
            linear_out: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
            gamma: LayoutTensor[
                dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
            ],
            beta: LayoutTensor[
                dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
            ],
            eps: Scalar[dtype],
        ):
            Self.forward_ln_mish_kernel_impl[BATCH](
                output, linear_out, gamma, beta, cache, eps
            )

        ctx.enqueue_function[ln_mish_wrapper, ln_mish_wrapper](
            output,
            linear_out_immut,
            gamma,
            beta,
            cache,
            eps_scalar,
            grid_dim=(BATCH,),
            block_dim=(1,),
        )

    @staticmethod
    fn forward_gpu_no_cache[
        BATCH: Int,
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
        """GPU forward without caching (inference)."""
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), ImmutAnyOrigin
        ](params.ptr)
        var b = LayoutTensor[
            dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
        ](params.ptr + Self._B_OFFSET)
        var gamma = LayoutTensor[
            dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
        ](params.ptr + Self._GAMMA_OFFSET)
        var beta = LayoutTensor[
            dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
        ](params.ptr + Self._BETA_OFFSET)
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](input.ptr)

        var linear_out_mut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ](workspace.unsafe_ptr())
        var linear_out_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ](workspace.unsafe_ptr())

        var eps_scalar = Scalar[dtype](Self.EPSILON)

        # Kernel 1: Linear matmul (no cache)
        comptime grid_x = (Self.OUT_DIM + TILE - 1) // TILE
        comptime grid_y = (BATCH + TILE - 1) // TILE

        @always_inline
        fn linear_wrapper(
            linear_out: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
            W: LayoutTensor[
                dtype,
                Layout.row_major(Self.IN_DIM, Self.OUT_DIM),
                ImmutAnyOrigin,
            ],
            b: LayoutTensor[
                dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
            ],
        ):
            Self.forward_linear_kernel_impl_no_cache[BATCH](
                linear_out, input, W, b
            )

        ctx.enqueue_function[linear_wrapper, linear_wrapper](
            linear_out_mut,
            input_immut,
            W,
            b,
            grid_dim=(grid_x, grid_y),
            block_dim=(TILE, TILE),
        )

        # Kernel 2: Fused LN + Mish (no cache)
        @always_inline
        fn ln_mish_wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ],
            linear_out: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
            gamma: LayoutTensor[
                dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
            ],
            beta: LayoutTensor[
                dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
            ],
            eps: Scalar[dtype],
        ):
            Self.forward_ln_mish_kernel_impl_no_cache[BATCH](
                output, linear_out, gamma, beta, eps
            )

        ctx.enqueue_function[ln_mish_wrapper, ln_mish_wrapper](
            output,
            linear_out_immut,
            gamma,
            beta,
            eps_scalar,
            grid_dim=(BATCH,),
            block_dim=(1,),
        )

    @staticmethod
    fn forward_gpu_no_cache_on_stream[
        BATCH: Int,
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
        """GPU forward without caching — on DeviceStream."""
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), ImmutAnyOrigin
        ](params.ptr)
        var b = LayoutTensor[
            dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
        ](params.ptr + Self._B_OFFSET)
        var gamma = LayoutTensor[
            dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
        ](params.ptr + Self._GAMMA_OFFSET)
        var beta = LayoutTensor[
            dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
        ](params.ptr + Self._BETA_OFFSET)
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
        ](input.ptr)

        var linear_out_mut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ](workspace.unsafe_ptr())
        var linear_out_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ](workspace.unsafe_ptr())

        var eps_scalar = Scalar[dtype](Self.EPSILON)

        comptime grid_x = (Self.OUT_DIM + TILE - 1) // TILE
        comptime grid_y = (BATCH + TILE - 1) // TILE

        @always_inline
        fn linear_wrapper(
            linear_out: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), ImmutAnyOrigin
            ],
            W: LayoutTensor[
                dtype,
                Layout.row_major(Self.IN_DIM, Self.OUT_DIM),
                ImmutAnyOrigin,
            ],
            b: LayoutTensor[
                dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
            ],
        ):
            Self.forward_linear_kernel_impl_no_cache[BATCH](
                linear_out, input, W, b
            )

        var compiled_linear = ctx.compile_function[linear_wrapper, linear_wrapper]()
        stream.enqueue_function(
            compiled_linear,
            linear_out_mut,
            input_immut,
            W,
            b,
            grid_dim=(grid_x, grid_y),
            block_dim=(TILE, TILE),
        )

        @always_inline
        fn ln_mish_wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ],
            linear_out: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
            gamma: LayoutTensor[
                dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
            ],
            beta: LayoutTensor[
                dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
            ],
            eps: Scalar[dtype],
        ):
            Self.forward_ln_mish_kernel_impl_no_cache[BATCH](
                output, linear_out, gamma, beta, eps
            )

        var compiled_ln_mish = ctx.compile_function[ln_mish_wrapper, ln_mish_wrapper]()
        stream.enqueue_function(
            compiled_ln_mish,
            output,
            linear_out_immut,
            gamma,
            beta,
            eps_scalar,
            grid_dim=(BATCH,),
            block_dim=(1,),
        )

    @staticmethod
    fn backward_gpu[
        BATCH: Int,
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
        """GPU backward: fused Mish+LN kernel → fused Linear backward kernel."""
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), ImmutAnyOrigin
        ](params.ptr)
        var gamma = LayoutTensor[
            dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
        ](params.ptr + Self._GAMMA_OFFSET)
        var cache_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ](cache.ptr)
        var grad_output_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ](grad_output.ptr)
        var dgamma = LayoutTensor[
            dtype, Layout.row_major(Self.OUT_DIM), MutAnyOrigin
        ](grads.ptr + Self._GAMMA_OFFSET)
        var dbeta = LayoutTensor[
            dtype, Layout.row_major(Self.OUT_DIM), MutAnyOrigin
        ](grads.ptr + Self._BETA_OFFSET)

        # Workspace for d_linear_out [BATCH, OUT_DIM]
        var d_linear_out_mut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ](workspace.unsafe_ptr())
        var d_linear_out_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ](workspace.unsafe_ptr())

        # Kernel 1: Fused Mish + LN backward (per-sample)
        @always_inline
        fn mish_ln_backward_wrapper(
            d_linear_out: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
            gamma: LayoutTensor[
                dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
            ],
            dgamma: LayoutTensor[
                dtype, Layout.row_major(Self.OUT_DIM), MutAnyOrigin
            ],
            dbeta: LayoutTensor[
                dtype, Layout.row_major(Self.OUT_DIM), MutAnyOrigin
            ],
        ):
            Self.backward_mish_ln_kernel_impl[BATCH](
                d_linear_out, grad_output, gamma, cache, dgamma, dbeta
            )

        ctx.enqueue_function[
            mish_ln_backward_wrapper, mish_ln_backward_wrapper
        ](
            d_linear_out_mut,
            grad_output_immut,
            gamma,
            cache_immut,
            dgamma,
            dbeta,
            grid_dim=(BATCH,),
            block_dim=(1,),
        )

        # Kernel 2: Fused Linear backward (dx + dW + db)
        var dW = LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), MutAnyOrigin
        ](grads.ptr)
        var db = LayoutTensor[
            dtype, Layout.row_major(Self.OUT_DIM), MutAnyOrigin
        ](grads.ptr + Self._B_OFFSET)

        comptime dx_grid_x = (Self.IN_DIM + TILE - 1) // TILE
        comptime dx_grid_y = (BATCH + TILE - 1) // TILE
        comptime dW_grid_x = (Self.OUT_DIM + TILE - 1) // TILE
        comptime dW_grid_y = (Self.IN_DIM + TILE - 1) // TILE
        comptime fused_grid_x = dx_grid_x if dx_grid_x > dW_grid_x else dW_grid_x
        comptime fused_grid_y = dx_grid_y + dW_grid_y

        @always_inline
        fn linear_backward_wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            dW: LayoutTensor[
                dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), MutAnyOrigin
            ],
            db: LayoutTensor[
                dtype, Layout.row_major(Self.OUT_DIM), MutAnyOrigin
            ],
            d_linear_out: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
            W: LayoutTensor[
                dtype,
                Layout.row_major(Self.IN_DIM, Self.OUT_DIM),
                ImmutAnyOrigin,
            ],
            cache: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
            ],
        ):
            Self.backward_linear_fused_kernel_impl[BATCH](
                grad_input, dW, db, d_linear_out, W, cache
            )

        ctx.enqueue_function[
            linear_backward_wrapper, linear_backward_wrapper
        ](
            grad_input,
            dW,
            db,
            d_linear_out_immut,
            W,
            cache_immut,
            grid_dim=(fused_grid_x, fused_grid_y),
            block_dim=(TILE, TILE),
        )
