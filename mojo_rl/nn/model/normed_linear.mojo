from ..constants import (
    dtype,
    TILE,
    TPB,
    MMA_M,
    MMA_N,
    MMA_K,
    MMA_BLOCK_M,
    MMA_BLOCK_N,
    MMA_WARPS_M,
    MMA_WARPS_N,
    MMA_NUM_WARPS,
    MMA_BLOCK_THREADS,
)
from .model import Model, PerfTimerPtr, NULL_PERF
from ..initializer import Initializer
from layout import LayoutTensor, Layout
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext, DeviceBuffer, DeviceStream
from std.gpu.memory import AddressSpace
from std.gpu.primitives import block, lane_id
from std.gpu.compute.mma import mma
from std.sys import is_nvidia_gpu
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

    # ─── MMA / 2x2 matmul kernels (dispatched inside GPU wrapper) ──────
    # Grid: ((OUT_DIM + 31) // 32, (BATCH + 31) // 32)
    # Block: (256, 1) — same for both MMA and 2x2 paths

    @always_inline
    @staticmethod
    fn _linear_kernel_no_cache[
        BATCH: Int,
    ](
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
        b: LayoutTensor[dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin],
    ):
        """Dispatching matmul: MMA on NVIDIA, 2x2 register-tiled on Apple."""
        comptime if is_nvidia_gpu():
            # ── MMA tensor core path ──
            var tid = Int(thread_idx.x)
            var warp_id = tid // 32
            var warp_m = warp_id // MMA_WARPS_N
            var warp_n = warp_id % MMA_WARPS_N
            var block_row = Int(block_idx.y) * MMA_BLOCK_M
            var block_col = Int(block_idx.x) * MMA_BLOCK_N

            var a_smem = LayoutTensor[
                dtype,
                Layout.row_major(MMA_BLOCK_M, MMA_K),
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ].stack_allocation()
            var b_smem = LayoutTensor[
                dtype,
                Layout.row_major(MMA_K, MMA_BLOCK_N),
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ].stack_allocation()

            var acc = SIMD[DType.float32, 4](0)
            var lid = lane_id()
            var group_id = lid >> 2
            var group_lane = lid % 4
            comptime num_k_tiles = (Self.IN_DIM + MMA_K - 1) // MMA_K

            for k_tile in range(num_k_tiles):
                var k_off = k_tile * MMA_K
                var a_r = tid // MMA_K
                var a_c = tid % MMA_K
                if block_row + a_r < BATCH and k_off + a_c < Self.IN_DIM:
                    a_smem[a_r, a_c] = input[block_row + a_r, k_off + a_c]
                else:
                    a_smem[a_r, a_c] = 0
                var br = tid // MMA_BLOCK_N
                var bc = tid % MMA_BLOCK_N
                if k_off + br < Self.IN_DIM and block_col + bc < Self.OUT_DIM:
                    b_smem[br, bc] = W[k_off + br, block_col + bc]
                else:
                    b_smem[br, bc] = 0
                barrier()
                var warp_row = warp_m * MMA_M
                var a_frag = SIMD[DType.float32, 4](
                    rebind[Scalar[DType.float32]](
                        a_smem[warp_row + Int(group_id), Int(group_lane)]
                    ),
                    rebind[Scalar[DType.float32]](
                        a_smem[warp_row + Int(group_id) + 8, Int(group_lane)]
                    ),
                    rebind[Scalar[DType.float32]](
                        a_smem[warp_row + Int(group_id), Int(group_lane) + 4]
                    ),
                    rebind[Scalar[DType.float32]](
                        a_smem[
                            warp_row + Int(group_id) + 8, Int(group_lane) + 4
                        ]
                    ),
                )
                var warp_col = warp_n * MMA_N
                var b_frag = SIMD[DType.float32, 2](
                    rebind[Scalar[DType.float32]](
                        b_smem[Int(group_lane), warp_col + Int(group_id)]
                    ),
                    rebind[Scalar[DType.float32]](
                        b_smem[Int(group_lane) + 4, warp_col + Int(group_id)]
                    ),
                )
                mma(acc, a_frag, b_frag, acc)
                barrier()

            var r0 = block_row + warp_m * MMA_M + Int(group_id)
            var r1 = r0 + 8
            var c0 = block_col + warp_n * MMA_N + Int(group_lane * 2)
            var c1 = c0 + 1
            if r0 < BATCH and c0 < Self.OUT_DIM:
                linear_out[r0, c0] = rebind[Scalar[dtype]](acc[0]) + rebind[
                    Scalar[dtype]
                ](b[c0])
            if r0 < BATCH and c1 < Self.OUT_DIM:
                linear_out[r0, c1] = rebind[Scalar[dtype]](acc[1]) + rebind[
                    Scalar[dtype]
                ](b[c1])
            if r1 < BATCH and c0 < Self.OUT_DIM:
                linear_out[r1, c0] = rebind[Scalar[dtype]](acc[2]) + rebind[
                    Scalar[dtype]
                ](b[c0])
            if r1 < BATCH and c1 < Self.OUT_DIM:
                linear_out[r1, c1] = rebind[Scalar[dtype]](acc[3]) + rebind[
                    Scalar[dtype]
                ](b[c1])
        else:
            # ── 2x2 register-tiled path (Apple / non-MMA) ──
            comptime BT = 32
            comptime SK = 16
            var tid = Int(thread_idx.x)
            var sub_r = tid // 16
            var sub_c = tid % 16
            var block_row = Int(block_idx.y) * BT
            var block_col = Int(block_idx.x) * BT

            var a_smem = LayoutTensor[
                dtype,
                Layout.row_major(BT, SK),
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ].stack_allocation()
            var b_smem = LayoutTensor[
                dtype,
                Layout.row_major(SK, BT),
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ].stack_allocation()

            var acc00: Scalar[dtype] = 0
            var acc01: Scalar[dtype] = 0
            var acc10: Scalar[dtype] = 0
            var acc11: Scalar[dtype] = 0

            comptime num_k_tiles = (Self.IN_DIM + SK - 1) // SK
            for k_tile in range(num_k_tiles):
                var k_off = k_tile * SK
                # Load A tile (2 loads per thread for 32×16 with 256 threads)
                var a_r0 = tid // SK
                var a_c0 = tid % SK
                var a_r1 = (tid + 256) // SK
                var a_c1 = (tid + 256) % SK
                if block_row + a_r0 < BATCH and k_off + a_c0 < Self.IN_DIM:
                    a_smem[a_r0, a_c0] = input[
                        block_row + a_r0, k_off + a_c0
                    ]
                else:
                    a_smem[a_r0, a_c0] = 0
                if a_r1 < BT and block_row + a_r1 < BATCH and k_off + a_c1 < Self.IN_DIM:
                    a_smem[a_r1, a_c1] = input[
                        block_row + a_r1, k_off + a_c1
                    ]
                elif a_r1 < BT:
                    a_smem[a_r1, a_c1] = 0
                # Load B tile
                var b_r0 = tid // BT
                var b_c0 = tid % BT
                var b_r1 = (tid + 256) // BT
                var b_c1 = (tid + 256) % BT
                if k_off + b_r0 < Self.IN_DIM and block_col + b_c0 < Self.OUT_DIM:
                    b_smem[b_r0, b_c0] = W[
                        k_off + b_r0, block_col + b_c0
                    ]
                else:
                    b_smem[b_r0, b_c0] = 0
                if b_r1 < SK and k_off + b_r1 < Self.IN_DIM and block_col + b_c1 < Self.OUT_DIM:
                    b_smem[b_r1, b_c1] = W[
                        k_off + b_r1, block_col + b_c1
                    ]
                elif b_r1 < SK:
                    b_smem[b_r1, b_c1] = 0
                barrier()
                for k in range(SK):
                    if k_off + k < Self.IN_DIM:
                        var a0 = rebind[Scalar[dtype]](a_smem[sub_r * 2, k])
                        var a1 = rebind[Scalar[dtype]](
                            a_smem[sub_r * 2 + 1, k]
                        )
                        var b0 = rebind[Scalar[dtype]](b_smem[k, sub_c * 2])
                        var b1 = rebind[Scalar[dtype]](
                            b_smem[k, sub_c * 2 + 1]
                        )
                        acc00 += a0 * b0
                        acc01 += a0 * b1
                        acc10 += a1 * b0
                        acc11 += a1 * b1
                barrier()

            var gr0 = block_row + sub_r * 2
            var gc0 = block_col + sub_c * 2
            if gr0 < BATCH and gc0 < Self.OUT_DIM:
                linear_out[gr0, gc0] = acc00 + rebind[Scalar[dtype]](b[gc0])
            if gr0 < BATCH and gc0 + 1 < Self.OUT_DIM:
                linear_out[gr0, gc0 + 1] = acc01 + rebind[Scalar[dtype]](
                    b[gc0 + 1]
                )
            if gr0 + 1 < BATCH and gc0 < Self.OUT_DIM:
                linear_out[gr0 + 1, gc0] = acc10 + rebind[Scalar[dtype]](
                    b[gc0]
                )
            if gr0 + 1 < BATCH and gc0 + 1 < Self.OUT_DIM:
                linear_out[gr0 + 1, gc0 + 1] = acc11 + rebind[Scalar[dtype]](
                    b[gc0 + 1]
                )

    @always_inline
    @staticmethod
    fn _linear_kernel_with_cache[
        BATCH: Int,
    ](
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
        b: LayoutTensor[dtype, Layout.row_major(Self.OUT_DIM), ImmutAnyOrigin],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
        ],
    ):
        """Dispatching matmul with input caching: MMA on NVIDIA, 2x2 on Apple."""
        comptime if is_nvidia_gpu():
            # ── MMA tensor core path with input caching ──
            var tid = Int(thread_idx.x)
            var warp_id = tid // 32
            var warp_m = warp_id // MMA_WARPS_N
            var warp_n = warp_id % MMA_WARPS_N
            var block_row = Int(block_idx.y) * MMA_BLOCK_M
            var block_col = Int(block_idx.x) * MMA_BLOCK_N

            # Cache input (first column of blocks only)
            if Int(block_idx.x) == 0:
                for i in range(0, Self.IN_DIM, MMA_BLOCK_THREADS):
                    var col = i + tid
                    if col < Self.IN_DIM:
                        for r in range(MMA_BLOCK_M):
                            var gr = block_row + r
                            if gr < BATCH:
                                cache[gr, Self._INPUT_OFFSET + col] = input[
                                    gr, col
                                ]

            var a_smem = LayoutTensor[
                dtype,
                Layout.row_major(MMA_BLOCK_M, MMA_K),
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ].stack_allocation()
            var b_smem = LayoutTensor[
                dtype,
                Layout.row_major(MMA_K, MMA_BLOCK_N),
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ].stack_allocation()

            var acc = SIMD[DType.float32, 4](0)
            var lid = lane_id()
            var group_id = lid >> 2
            var group_lane = lid % 4
            comptime num_k_tiles = (Self.IN_DIM + MMA_K - 1) // MMA_K

            for k_tile in range(num_k_tiles):
                var k_off = k_tile * MMA_K
                var a_r = tid // MMA_K
                var a_c = tid % MMA_K
                if block_row + a_r < BATCH and k_off + a_c < Self.IN_DIM:
                    a_smem[a_r, a_c] = input[block_row + a_r, k_off + a_c]
                else:
                    a_smem[a_r, a_c] = 0
                var br = tid // MMA_BLOCK_N
                var bc = tid % MMA_BLOCK_N
                if k_off + br < Self.IN_DIM and block_col + bc < Self.OUT_DIM:
                    b_smem[br, bc] = W[k_off + br, block_col + bc]
                else:
                    b_smem[br, bc] = 0
                barrier()
                var warp_row = warp_m * MMA_M
                var a_frag = SIMD[DType.float32, 4](
                    rebind[Scalar[DType.float32]](
                        a_smem[warp_row + Int(group_id), Int(group_lane)]
                    ),
                    rebind[Scalar[DType.float32]](
                        a_smem[warp_row + Int(group_id) + 8, Int(group_lane)]
                    ),
                    rebind[Scalar[DType.float32]](
                        a_smem[warp_row + Int(group_id), Int(group_lane) + 4]
                    ),
                    rebind[Scalar[DType.float32]](
                        a_smem[
                            warp_row + Int(group_id) + 8, Int(group_lane) + 4
                        ]
                    ),
                )
                var warp_col = warp_n * MMA_N
                var b_frag = SIMD[DType.float32, 2](
                    rebind[Scalar[DType.float32]](
                        b_smem[Int(group_lane), warp_col + Int(group_id)]
                    ),
                    rebind[Scalar[DType.float32]](
                        b_smem[Int(group_lane) + 4, warp_col + Int(group_id)]
                    ),
                )
                mma(acc, a_frag, b_frag, acc)
                barrier()

            var r0 = block_row + warp_m * MMA_M + Int(group_id)
            var r1 = r0 + 8
            var c0 = block_col + warp_n * MMA_N + Int(group_lane * 2)
            var c1 = c0 + 1
            if r0 < BATCH and c0 < Self.OUT_DIM:
                linear_out[r0, c0] = rebind[Scalar[dtype]](acc[0]) + rebind[
                    Scalar[dtype]
                ](b[c0])
            if r0 < BATCH and c1 < Self.OUT_DIM:
                linear_out[r0, c1] = rebind[Scalar[dtype]](acc[1]) + rebind[
                    Scalar[dtype]
                ](b[c1])
            if r1 < BATCH and c0 < Self.OUT_DIM:
                linear_out[r1, c0] = rebind[Scalar[dtype]](acc[2]) + rebind[
                    Scalar[dtype]
                ](b[c0])
            if r1 < BATCH and c1 < Self.OUT_DIM:
                linear_out[r1, c1] = rebind[Scalar[dtype]](acc[3]) + rebind[
                    Scalar[dtype]
                ](b[c1])
        else:
            # ── 2x2 path with input caching (Apple / non-MMA) ──
            comptime BT = 32
            comptime SK = 16
            var tid = Int(thread_idx.x)
            var sub_r = tid // 16
            var sub_c = tid % 16
            var block_row = Int(block_idx.y) * BT
            var block_col = Int(block_idx.x) * BT

            # Cache input (first column of blocks only)
            if Int(block_idx.x) == 0:
                for i in range(0, Self.IN_DIM, MMA_BLOCK_THREADS):
                    var col = i + tid
                    if col < Self.IN_DIM:
                        for r in range(BT):
                            var gr = block_row + r
                            if gr < BATCH:
                                cache[gr, Self._INPUT_OFFSET + col] = input[
                                    gr, col
                                ]

            var a_smem = LayoutTensor[
                dtype,
                Layout.row_major(BT, SK),
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ].stack_allocation()
            var b_smem = LayoutTensor[
                dtype,
                Layout.row_major(SK, BT),
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ].stack_allocation()

            var acc00: Scalar[dtype] = 0
            var acc01: Scalar[dtype] = 0
            var acc10: Scalar[dtype] = 0
            var acc11: Scalar[dtype] = 0

            comptime num_k_tiles = (Self.IN_DIM + SK - 1) // SK
            for k_tile in range(num_k_tiles):
                var k_off = k_tile * SK
                var a_r0 = tid // SK
                var a_c0 = tid % SK
                var a_r1 = (tid + 256) // SK
                var a_c1 = (tid + 256) % SK
                if block_row + a_r0 < BATCH and k_off + a_c0 < Self.IN_DIM:
                    a_smem[a_r0, a_c0] = input[
                        block_row + a_r0, k_off + a_c0
                    ]
                else:
                    a_smem[a_r0, a_c0] = 0
                if a_r1 < BT and block_row + a_r1 < BATCH and k_off + a_c1 < Self.IN_DIM:
                    a_smem[a_r1, a_c1] = input[
                        block_row + a_r1, k_off + a_c1
                    ]
                elif a_r1 < BT:
                    a_smem[a_r1, a_c1] = 0
                var b_r0 = tid // BT
                var b_c0 = tid % BT
                var b_r1 = (tid + 256) // BT
                var b_c1 = (tid + 256) % BT
                if k_off + b_r0 < Self.IN_DIM and block_col + b_c0 < Self.OUT_DIM:
                    b_smem[b_r0, b_c0] = W[
                        k_off + b_r0, block_col + b_c0
                    ]
                else:
                    b_smem[b_r0, b_c0] = 0
                if b_r1 < SK and k_off + b_r1 < Self.IN_DIM and block_col + b_c1 < Self.OUT_DIM:
                    b_smem[b_r1, b_c1] = W[
                        k_off + b_r1, block_col + b_c1
                    ]
                elif b_r1 < SK:
                    b_smem[b_r1, b_c1] = 0
                barrier()
                for k in range(SK):
                    if k_off + k < Self.IN_DIM:
                        var a0 = rebind[Scalar[dtype]](a_smem[sub_r * 2, k])
                        var a1 = rebind[Scalar[dtype]](
                            a_smem[sub_r * 2 + 1, k]
                        )
                        var b0 = rebind[Scalar[dtype]](b_smem[k, sub_c * 2])
                        var b1 = rebind[Scalar[dtype]](
                            b_smem[k, sub_c * 2 + 1]
                        )
                        acc00 += a0 * b0
                        acc01 += a0 * b1
                        acc10 += a1 * b0
                        acc11 += a1 * b1
                barrier()

            var gr0 = block_row + sub_r * 2
            var gc0 = block_col + sub_c * 2
            if gr0 < BATCH and gc0 < Self.OUT_DIM:
                linear_out[gr0, gc0] = acc00 + rebind[Scalar[dtype]](b[gc0])
            if gr0 < BATCH and gc0 + 1 < Self.OUT_DIM:
                linear_out[gr0, gc0 + 1] = acc01 + rebind[Scalar[dtype]](
                    b[gc0 + 1]
                )
            if gr0 + 1 < BATCH and gc0 < Self.OUT_DIM:
                linear_out[gr0 + 1, gc0] = acc10 + rebind[Scalar[dtype]](
                    b[gc0]
                )
            if gr0 + 1 < BATCH and gc0 + 1 < Self.OUT_DIM:
                linear_out[gr0 + 1, gc0 + 1] = acc11 + rebind[Scalar[dtype]](
                    b[gc0 + 1]
                )

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

        Warp-parallel: 32 threads per sample for reduction.
        Grid: (BATCH,)
        Block: (32,)
        """
        var batch_idx = Int(block_idx.x)
        if batch_idx >= BATCH:
            return

        var tid = Int(thread_idx.x)
        comptime WARP = 32
        var n = Scalar[dtype](Self.OUT_DIM)

        # ── Pass 1: compute mean (warp-parallel reduction) ──
        var local_sum: Scalar[dtype] = 0.0
        for j in range(tid, Self.OUT_DIM, WARP):
            local_sum += rebind[Scalar[dtype]](linear_out[batch_idx, j])

        # Shared memory reduction for sum
        var smem = LayoutTensor[
            dtype,
            Layout.row_major(WARP),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()
        smem[tid] = local_sum
        barrier()
        if tid < 16:
            smem[tid] = smem[tid] + smem[tid + 16]
        barrier()
        if tid < 8:
            smem[tid] = smem[tid] + smem[tid + 8]
        barrier()
        if tid < 4:
            smem[tid] = smem[tid] + smem[tid + 4]
        barrier()
        if tid < 2:
            smem[tid] = smem[tid] + smem[tid + 2]
        barrier()
        if tid == 0:
            smem[0] = smem[0] + smem[1]
        barrier()
        var mean = rebind[Scalar[dtype]](smem[0]) / n

        # ── Pass 2: compute variance (warp-parallel reduction) ──
        var local_var: Scalar[dtype] = 0.0
        for j in range(tid, Self.OUT_DIM, WARP):
            var diff = rebind[Scalar[dtype]](linear_out[batch_idx, j]) - mean
            local_var += diff * diff

        # Shared memory reduction for variance
        smem[tid] = local_var
        barrier()
        if tid < 16:
            smem[tid] = smem[tid] + smem[tid + 16]
        barrier()
        if tid < 8:
            smem[tid] = smem[tid] + smem[tid + 8]
        barrier()
        if tid < 4:
            smem[tid] = smem[tid] + smem[tid + 4]
        barrier()
        if tid < 2:
            smem[tid] = smem[tid] + smem[tid + 2]
        barrier()
        if tid == 0:
            smem[0] = smem[0] + smem[1]
        barrier()
        var var_ = rebind[Scalar[dtype]](smem[0]) / n
        var inv_std = 1.0 / sqrt(var_ + eps)

        # ── Pass 3: normalize + Mish + cache (warp-parallel) ──
        for j in range(tid, Self.OUT_DIM, WARP):
            var z_val = rebind[Scalar[dtype]](linear_out[batch_idx, j])
            var normalized = (z_val - mean) * inv_std
            cache[batch_idx, Self._LN_NORM_OFFSET + j] = normalized

            var ln_out = rebind[Scalar[dtype]](gamma[j]) * normalized + rebind[
                Scalar[dtype]
            ](beta[j])
            cache[batch_idx, Self._LN_OUT_OFFSET + j] = ln_out

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

        # Store scalars (only thread 0)
        if tid == 0:
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

        Warp-parallel: 32 threads per sample for reduction.
        Grid: (BATCH,)
        Block: (32,)
        """
        var batch_idx = Int(block_idx.x)
        if batch_idx >= BATCH:
            return

        var tid = Int(thread_idx.x)
        comptime WARP = 32
        var n = Scalar[dtype](Self.OUT_DIM)

        # ── Pass 1: compute mean (warp-parallel reduction) ──
        var local_sum: Scalar[dtype] = 0.0
        for j in range(tid, Self.OUT_DIM, WARP):
            local_sum += rebind[Scalar[dtype]](linear_out[batch_idx, j])

        # Warp shuffle reduction for sum
        # Shared memory reduction for sum
        var smem = LayoutTensor[
            dtype,
            Layout.row_major(WARP),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ].stack_allocation()
        smem[tid] = local_sum
        barrier()
        if tid < 16:
            smem[tid] = smem[tid] + smem[tid + 16]
        barrier()
        if tid < 8:
            smem[tid] = smem[tid] + smem[tid + 8]
        barrier()
        if tid < 4:
            smem[tid] = smem[tid] + smem[tid + 4]
        barrier()
        if tid < 2:
            smem[tid] = smem[tid] + smem[tid + 2]
        barrier()
        if tid == 0:
            smem[0] = smem[0] + smem[1]
        barrier()
        var mean = rebind[Scalar[dtype]](smem[0]) / n

        # ── Pass 2: compute variance (warp-parallel reduction) ──
        var local_var: Scalar[dtype] = 0.0
        for j in range(tid, Self.OUT_DIM, WARP):
            var diff = rebind[Scalar[dtype]](linear_out[batch_idx, j]) - mean
            local_var += diff * diff

        # Shared memory reduction for variance
        smem[tid] = local_var
        barrier()
        if tid < 16:
            smem[tid] = smem[tid] + smem[tid + 16]
        barrier()
        if tid < 8:
            smem[tid] = smem[tid] + smem[tid + 8]
        barrier()
        if tid < 4:
            smem[tid] = smem[tid] + smem[tid + 4]
        barrier()
        if tid < 2:
            smem[tid] = smem[tid] + smem[tid + 2]
        barrier()
        if tid == 0:
            smem[0] = smem[0] + smem[1]
        barrier()
        var var_ = rebind[Scalar[dtype]](smem[0]) / n
        var inv_std = 1.0 / sqrt(var_ + eps)

        # ── Pass 3: normalize + Mish (warp-parallel) ──
        for j in range(tid, Self.OUT_DIM, WARP):
            var z_val = rebind[Scalar[dtype]](linear_out[batch_idx, j])
            var normalized = (z_val - mean) * inv_std
            var ln_out = rebind[Scalar[dtype]](gamma[j]) * normalized + rebind[
                Scalar[dtype]
            ](beta[j])

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
    fn _backward_dx_kernel[
        BATCH: Int,
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        d_linear_out: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ],
        W: LayoutTensor[
            dtype,
            Layout.row_major(Self.IN_DIM, Self.OUT_DIM),
            ImmutAnyOrigin,
        ],
    ):
        """Dispatching backward dx = d_linear_out @ W.T."""
        comptime if is_nvidia_gpu():
            var tid = Int(thread_idx.x)
            var warp_id = tid // 32
            var warp_m = warp_id // MMA_WARPS_N
            var warp_n = warp_id % MMA_WARPS_N
            var block_row = Int(block_idx.y) * MMA_BLOCK_M
            var block_col = Int(block_idx.x) * MMA_BLOCK_N

            var a_smem = LayoutTensor[
                dtype,
                Layout.row_major(MMA_BLOCK_M, MMA_K),
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ].stack_allocation()
            var b_smem = LayoutTensor[
                dtype,
                Layout.row_major(MMA_K, MMA_BLOCK_N),
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ].stack_allocation()

            var acc = SIMD[DType.float32, 4](0)
            var lid = lane_id()
            var group_id = lid >> 2
            var group_lane = lid % 4
            comptime num_k_tiles = (Self.OUT_DIM + MMA_K - 1) // MMA_K

            for k_tile in range(num_k_tiles):
                var k_off = k_tile * MMA_K
                # Load A = d_linear_out
                var a_r = tid // MMA_K
                var a_c = tid % MMA_K
                if block_row + a_r < BATCH and k_off + a_c < Self.OUT_DIM:
                    a_smem[a_r, a_c] = d_linear_out[
                        block_row + a_r, k_off + a_c
                    ]
                else:
                    a_smem[a_r, a_c] = 0
                # Load B = W.T (swap indices)
                var br = tid // MMA_BLOCK_N
                var bc = tid % MMA_BLOCK_N
                if k_off + br < Self.OUT_DIM and block_col + bc < Self.IN_DIM:
                    b_smem[br, bc] = W[block_col + bc, k_off + br]
                else:
                    b_smem[br, bc] = 0
                barrier()
                var warp_row = warp_m * MMA_M
                var a_frag = SIMD[DType.float32, 4](
                    rebind[Scalar[DType.float32]](
                        a_smem[warp_row + Int(group_id), Int(group_lane)]
                    ),
                    rebind[Scalar[DType.float32]](
                        a_smem[warp_row + Int(group_id) + 8, Int(group_lane)]
                    ),
                    rebind[Scalar[DType.float32]](
                        a_smem[warp_row + Int(group_id), Int(group_lane) + 4]
                    ),
                    rebind[Scalar[DType.float32]](
                        a_smem[
                            warp_row + Int(group_id) + 8, Int(group_lane) + 4
                        ]
                    ),
                )
                var warp_col = warp_n * MMA_N
                var b_frag = SIMD[DType.float32, 2](
                    rebind[Scalar[DType.float32]](
                        b_smem[Int(group_lane), warp_col + Int(group_id)]
                    ),
                    rebind[Scalar[DType.float32]](
                        b_smem[Int(group_lane) + 4, warp_col + Int(group_id)]
                    ),
                )
                mma(acc, a_frag, b_frag, acc)
                barrier()

            var r0 = block_row + warp_m * MMA_M + Int(group_id)
            var r1 = r0 + 8
            var c0 = block_col + warp_n * MMA_N + Int(group_lane * 2)
            var c1 = c0 + 1
            if r0 < BATCH and c0 < Self.IN_DIM:
                grad_input[r0, c0] = rebind[Scalar[dtype]](acc[0])
            if r0 < BATCH and c1 < Self.IN_DIM:
                grad_input[r0, c1] = rebind[Scalar[dtype]](acc[1])
            if r1 < BATCH and c0 < Self.IN_DIM:
                grad_input[r1, c0] = rebind[Scalar[dtype]](acc[2])
            if r1 < BATCH and c1 < Self.IN_DIM:
                grad_input[r1, c1] = rebind[Scalar[dtype]](acc[3])
        else:
            # 2x2 register-tiled fallback
            comptime BT = 32
            comptime SK = 16
            var tid = Int(thread_idx.x)
            var sub_r = tid // 16
            var sub_c = tid % 16
            var block_row = Int(block_idx.y) * BT
            var block_col = Int(block_idx.x) * BT
            var a_smem = LayoutTensor[
                dtype, Layout.row_major(BT, SK), MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ].stack_allocation()
            var b_smem = LayoutTensor[
                dtype, Layout.row_major(SK, BT), MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ].stack_allocation()
            var acc00: Scalar[dtype] = 0
            var acc01: Scalar[dtype] = 0
            var acc10: Scalar[dtype] = 0
            var acc11: Scalar[dtype] = 0
            for k_tile in range((Self.OUT_DIM + SK - 1) // SK):
                var k_off = k_tile * SK
                var a_r0 = tid // SK
                var a_c0 = tid % SK
                var a_r1 = (tid + 256) // SK
                var a_c1 = (tid + 256) % SK
                if block_row + a_r0 < BATCH and k_off + a_c0 < Self.OUT_DIM:
                    a_smem[a_r0, a_c0] = d_linear_out[block_row + a_r0, k_off + a_c0]
                else:
                    a_smem[a_r0, a_c0] = 0
                if a_r1 < BT and block_row + a_r1 < BATCH and k_off + a_c1 < Self.OUT_DIM:
                    a_smem[a_r1, a_c1] = d_linear_out[block_row + a_r1, k_off + a_c1]
                elif a_r1 < BT:
                    a_smem[a_r1, a_c1] = 0
                var b_r0 = tid // BT
                var b_c0 = tid % BT
                var b_r1 = (tid + 256) // BT
                var b_c1 = (tid + 256) % BT
                if k_off + b_r0 < Self.OUT_DIM and block_col + b_c0 < Self.IN_DIM:
                    b_smem[b_r0, b_c0] = W[block_col + b_c0, k_off + b_r0]
                else:
                    b_smem[b_r0, b_c0] = 0
                if b_r1 < SK and k_off + b_r1 < Self.OUT_DIM and block_col + b_c1 < Self.IN_DIM:
                    b_smem[b_r1, b_c1] = W[block_col + b_c1, k_off + b_r1]
                elif b_r1 < SK:
                    b_smem[b_r1, b_c1] = 0
                barrier()
                for k in range(SK):
                    if k_off + k < Self.OUT_DIM:
                        var a0 = rebind[Scalar[dtype]](a_smem[sub_r * 2, k])
                        var a1 = rebind[Scalar[dtype]](a_smem[sub_r * 2 + 1, k])
                        var b0 = rebind[Scalar[dtype]](b_smem[k, sub_c * 2])
                        var b1 = rebind[Scalar[dtype]](b_smem[k, sub_c * 2 + 1])
                        acc00 += a0 * b0
                        acc01 += a0 * b1
                        acc10 += a1 * b0
                        acc11 += a1 * b1
                barrier()
            var gr0 = block_row + sub_r * 2
            var gc0 = block_col + sub_c * 2
            if gr0 < BATCH and gc0 < Self.IN_DIM:
                grad_input[gr0, gc0] = acc00
            if gr0 < BATCH and gc0 + 1 < Self.IN_DIM:
                grad_input[gr0, gc0 + 1] = acc01
            if gr0 + 1 < BATCH and gc0 < Self.IN_DIM:
                grad_input[gr0 + 1, gc0] = acc10
            if gr0 + 1 < BATCH and gc0 + 1 < Self.IN_DIM:
                grad_input[gr0 + 1, gc0 + 1] = acc11

    @always_inline
    @staticmethod
    fn _backward_dW_kernel[
        BATCH: Int,
    ](
        dW: LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), MutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.CACHE_SIZE), ImmutAnyOrigin
        ],
        d_linear_out: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
        ],
    ):
        """Dispatching backward dW = input.T @ d_linear_out."""
        comptime if is_nvidia_gpu():
            var tid = Int(thread_idx.x)
            var warp_id = tid // 32
            var warp_m = warp_id // MMA_WARPS_N
            var warp_n = warp_id % MMA_WARPS_N
            var block_row = Int(block_idx.y) * MMA_BLOCK_M  # IN_DIM
            var block_col = Int(block_idx.x) * MMA_BLOCK_N  # OUT_DIM

            var a_smem = LayoutTensor[
                dtype,
                Layout.row_major(MMA_BLOCK_M, MMA_K),
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ].stack_allocation()
            var b_smem = LayoutTensor[
                dtype,
                Layout.row_major(MMA_K, MMA_BLOCK_N),
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ].stack_allocation()

            var acc = SIMD[DType.float32, 4](0)
            var lid = lane_id()
            var group_id = lid >> 2
            var group_lane = lid % 4
            comptime num_k_tiles = (BATCH + MMA_K - 1) // MMA_K

            for k_tile in range(num_k_tiles):
                var k_off = k_tile * MMA_K
                # Load A = input.T from cache
                var a_r = tid // MMA_K
                var a_c = tid % MMA_K
                if k_off + a_c < BATCH and block_row + a_r < Self.IN_DIM:
                    a_smem[a_r, a_c] = cache[
                        k_off + a_c, Self._INPUT_OFFSET + block_row + a_r
                    ]
                else:
                    a_smem[a_r, a_c] = 0
                # Load B = d_linear_out
                var br = tid // MMA_BLOCK_N
                var bc = tid % MMA_BLOCK_N
                if k_off + br < BATCH and block_col + bc < Self.OUT_DIM:
                    b_smem[br, bc] = d_linear_out[k_off + br, block_col + bc]
                else:
                    b_smem[br, bc] = 0
                barrier()
                var warp_row = warp_m * MMA_M
                var a_frag = SIMD[DType.float32, 4](
                    rebind[Scalar[DType.float32]](
                        a_smem[warp_row + Int(group_id), Int(group_lane)]
                    ),
                    rebind[Scalar[DType.float32]](
                        a_smem[warp_row + Int(group_id) + 8, Int(group_lane)]
                    ),
                    rebind[Scalar[DType.float32]](
                        a_smem[warp_row + Int(group_id), Int(group_lane) + 4]
                    ),
                    rebind[Scalar[DType.float32]](
                        a_smem[
                            warp_row + Int(group_id) + 8, Int(group_lane) + 4
                        ]
                    ),
                )
                var warp_col = warp_n * MMA_N
                var b_frag = SIMD[DType.float32, 2](
                    rebind[Scalar[DType.float32]](
                        b_smem[Int(group_lane), warp_col + Int(group_id)]
                    ),
                    rebind[Scalar[DType.float32]](
                        b_smem[Int(group_lane) + 4, warp_col + Int(group_id)]
                    ),
                )
                mma(acc, a_frag, b_frag, acc)
                barrier()

            var r0 = block_row + warp_m * MMA_M + Int(group_id)
            var r1 = r0 + 8
            var c0 = block_col + warp_n * MMA_N + Int(group_lane * 2)
            var c1 = c0 + 1
            if r0 < Self.IN_DIM and c0 < Self.OUT_DIM:
                dW[r0, c0] = rebind[Scalar[dtype]](acc[0])
            if r0 < Self.IN_DIM and c1 < Self.OUT_DIM:
                dW[r0, c1] = rebind[Scalar[dtype]](acc[1])
            if r1 < Self.IN_DIM and c0 < Self.OUT_DIM:
                dW[r1, c0] = rebind[Scalar[dtype]](acc[2])
            if r1 < Self.IN_DIM and c1 < Self.OUT_DIM:
                dW[r1, c1] = rebind[Scalar[dtype]](acc[3])
        else:
            # 2x2 register-tiled fallback
            comptime BT = 32
            comptime SK = 16
            var tid = Int(thread_idx.x)
            var sub_r = tid // 16
            var sub_c = tid % 16
            var block_row = Int(block_idx.y) * BT
            var block_col = Int(block_idx.x) * BT
            var a_smem = LayoutTensor[
                dtype, Layout.row_major(BT, SK), MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ].stack_allocation()
            var b_smem = LayoutTensor[
                dtype, Layout.row_major(SK, BT), MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ].stack_allocation()
            var acc00: Scalar[dtype] = 0
            var acc01: Scalar[dtype] = 0
            var acc10: Scalar[dtype] = 0
            var acc11: Scalar[dtype] = 0
            for k_tile in range((BATCH + SK - 1) // SK):
                var k_off = k_tile * SK
                var a_r0 = tid // SK
                var a_c0 = tid % SK
                var a_r1 = (tid + 256) // SK
                var a_c1 = (tid + 256) % SK
                if k_off + a_c0 < BATCH and block_row + a_r0 < Self.IN_DIM:
                    a_smem[a_r0, a_c0] = cache[k_off + a_c0, Self._INPUT_OFFSET + block_row + a_r0]
                else:
                    a_smem[a_r0, a_c0] = 0
                if a_r1 < BT and k_off + a_c1 < BATCH and block_row + a_r1 < Self.IN_DIM:
                    a_smem[a_r1, a_c1] = cache[k_off + a_c1, Self._INPUT_OFFSET + block_row + a_r1]
                elif a_r1 < BT:
                    a_smem[a_r1, a_c1] = 0
                var b_r0 = tid // BT
                var b_c0 = tid % BT
                var b_r1 = (tid + 256) // BT
                var b_c1 = (tid + 256) % BT
                if k_off + b_r0 < BATCH and block_col + b_c0 < Self.OUT_DIM:
                    b_smem[b_r0, b_c0] = d_linear_out[k_off + b_r0, block_col + b_c0]
                else:
                    b_smem[b_r0, b_c0] = 0
                if b_r1 < SK and k_off + b_r1 < BATCH and block_col + b_c1 < Self.OUT_DIM:
                    b_smem[b_r1, b_c1] = d_linear_out[k_off + b_r1, block_col + b_c1]
                elif b_r1 < SK:
                    b_smem[b_r1, b_c1] = 0
                barrier()
                for k in range(SK):
                    if k_off + k < BATCH:
                        var a0 = rebind[Scalar[dtype]](a_smem[sub_r * 2, k])
                        var a1 = rebind[Scalar[dtype]](a_smem[sub_r * 2 + 1, k])
                        var b0 = rebind[Scalar[dtype]](b_smem[k, sub_c * 2])
                        var b1 = rebind[Scalar[dtype]](b_smem[k, sub_c * 2 + 1])
                        acc00 += a0 * b0
                        acc01 += a0 * b1
                        acc10 += a1 * b0
                        acc11 += a1 * b1
                barrier()
            var gr0 = block_row + sub_r * 2
            var gc0 = block_col + sub_c * 2
            if gr0 < Self.IN_DIM and gc0 < Self.OUT_DIM:
                dW[gr0, gc0] = acc00
            if gr0 < Self.IN_DIM and gc0 + 1 < Self.OUT_DIM:
                dW[gr0, gc0 + 1] = acc01
            if gr0 + 1 < Self.IN_DIM and gc0 < Self.OUT_DIM:
                dW[gr0 + 1, gc0] = acc10
            if gr0 + 1 < Self.IN_DIM and gc0 + 1 < Self.OUT_DIM:
                dW[gr0 + 1, gc0 + 1] = acc11

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

        # Kernel 1: Linear matmul (with cache; dispatch inside wrapper)
        comptime grid_x = (Self.OUT_DIM + MMA_BLOCK_N - 1) // MMA_BLOCK_N
        comptime grid_y = (BATCH + MMA_BLOCK_M - 1) // MMA_BLOCK_M

        @always_inline
        fn linear_wrapper(
            linear_out: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.IN_DIM),
                ImmutAnyOrigin,
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
                dtype,
                Layout.row_major(BATCH, Self.CACHE_SIZE),
                MutAnyOrigin,
            ],
        ):
            Self._linear_kernel_with_cache[BATCH](
                linear_out, input, W, b, cache
            )

        ctx.enqueue_function[linear_wrapper, linear_wrapper](
            linear_out_mut,
            input_immut,
            W,
            b,
            cache,
            grid_dim=(grid_x, grid_y),
            block_dim=(MMA_BLOCK_THREADS, 1),
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
            block_dim=(32,),
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

        # Kernel 1: Linear matmul (no cache; dispatch inside wrapper)
        comptime grid_x = (Self.OUT_DIM + MMA_BLOCK_N - 1) // MMA_BLOCK_N
        comptime grid_y = (BATCH + MMA_BLOCK_M - 1) // MMA_BLOCK_M

        @always_inline
        fn linear_nc_wrapper(
            linear_out: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.IN_DIM),
                ImmutAnyOrigin,
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
            Self._linear_kernel_no_cache[BATCH](linear_out, input, W, b)

        ctx.enqueue_function[linear_nc_wrapper, linear_nc_wrapper](
            linear_out_mut,
            input_immut,
            W,
            b,
            grid_dim=(grid_x, grid_y),
            block_dim=(MMA_BLOCK_THREADS, 1),
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
            block_dim=(32,),
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

        comptime grid_x = (Self.OUT_DIM + MMA_BLOCK_N - 1) // MMA_BLOCK_N
        comptime grid_y = (BATCH + MMA_BLOCK_M - 1) // MMA_BLOCK_M

        @always_inline
        fn linear_stream_wrapper(
            linear_out: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.IN_DIM),
                ImmutAnyOrigin,
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
            Self._linear_kernel_no_cache[BATCH](linear_out, input, W, b)

        var compiled_linear = ctx.compile_function[
            linear_stream_wrapper, linear_stream_wrapper
        ]()
        stream.enqueue_function(
            compiled_linear,
            linear_out_mut,
            input_immut,
            W,
            b,
            grid_dim=(grid_x, grid_y),
            block_dim=(MMA_BLOCK_THREADS, 1),
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
            block_dim=(32,),
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
            block_dim=(32,),
        )

        # Kernel 2: dx = d_linear_out @ W.T (MMA on NVIDIA)
        var dW = LayoutTensor[
            dtype, Layout.row_major(Self.IN_DIM, Self.OUT_DIM), MutAnyOrigin
        ](grads.ptr)
        var db = LayoutTensor[
            dtype, Layout.row_major(Self.OUT_DIM), MutAnyOrigin
        ](grads.ptr + Self._B_OFFSET)

        comptime dx_grid_x = (Self.IN_DIM + MMA_BLOCK_N - 1) // MMA_BLOCK_N
        comptime dx_grid_y = (BATCH + MMA_BLOCK_M - 1) // MMA_BLOCK_M

        @always_inline
        fn dx_wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
            ],
            d_linear_out: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
            W: LayoutTensor[
                dtype,
                Layout.row_major(Self.IN_DIM, Self.OUT_DIM),
                ImmutAnyOrigin,
            ],
        ):
            Self._backward_dx_kernel[BATCH](grad_input, d_linear_out, W)

        ctx.enqueue_function[dx_wrapper, dx_wrapper](
            grad_input,
            d_linear_out_immut,
            W,
            grid_dim=(dx_grid_x, dx_grid_y),
            block_dim=(MMA_BLOCK_THREADS, 1),
        )

        # Kernel 3: dW = input.T @ d_linear_out (MMA on NVIDIA)
        comptime dW_grid_x = (Self.OUT_DIM + MMA_BLOCK_N - 1) // MMA_BLOCK_N
        comptime dW_grid_y = (Self.IN_DIM + MMA_BLOCK_M - 1) // MMA_BLOCK_M

        @always_inline
        fn dW_wrapper(
            dW: LayoutTensor[
                dtype,
                Layout.row_major(Self.IN_DIM, Self.OUT_DIM),
                MutAnyOrigin,
            ],
            cache: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.CACHE_SIZE),
                ImmutAnyOrigin,
            ],
            d_linear_out: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
        ):
            Self._backward_dW_kernel[BATCH](dW, cache, d_linear_out)

        ctx.enqueue_function[dW_wrapper, dW_wrapper](
            dW,
            cache_immut,
            d_linear_out_immut,
            grid_dim=(dW_grid_x, dW_grid_y),
            block_dim=(MMA_BLOCK_THREADS, 1),
        )

        # Kernel 4: db = sum(d_linear_out, axis=0)
        comptime db_blocks = (Self.OUT_DIM + TPB - 1) // TPB

        @always_inline
        fn db_wrapper(
            db: LayoutTensor[
                dtype, Layout.row_major(Self.OUT_DIM), MutAnyOrigin
            ],
            d_linear_out: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.OUT_DIM), ImmutAnyOrigin
            ],
        ):
            var col = Int(block_dim.x * block_idx.x + thread_idx.x)
            if col < Self.OUT_DIM:
                var acc: Scalar[dtype] = 0
                for ba in range(BATCH):
                    acc += rebind[Scalar[dtype]](d_linear_out[ba, col])
                db[col] = acc

        ctx.enqueue_function[db_wrapper, db_wrapper](
            db,
            d_linear_out_immut,
            grid_dim=(db_blocks,),
            block_dim=(TPB,),
        )
