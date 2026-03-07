"""Parameterized fused matmul + bias + activation op.

FusedMatMulBiasActivation[in_dim, out_dim, ACT] replaces the separate
FusedMatMulBiasReLU and FusedMatMulBiasTanh structs with a single
implementation parameterized on an Activation trait.

The activation-specific behavior is injected via Self.ACT.forward(),
Self.ACT.cache(), and Self.ACT.backward() — only ~8 lines differ between
activations.
"""

from ...constants import dtype, TILE, TPB
from ...autodiff.op import DiffOp, FusedOp, OpID
from .activation import Activation
from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.gpu.primitives import block


struct FusedMatMulBiasActivation[
    in_dim: Int, out_dim: Int, ACT: Activation
](FusedOp):
    """Fused y = act(x @ W + b) in a single operation.

    PARAM_SIZE = in_dim * out_dim + out_dim  (W then b)
    CACHE_SIZE = in_dim + out_dim  (input for dW, activation cache for backward)
    """

    comptime OP_ID: Int = Self.ACT.FUSED_OP_ID
    comptime IN_DIM: Int = Self.in_dim
    comptime OUT_DIM: Int = Self.out_dim
    comptime PARAM_SIZE: Int = Self.in_dim * Self.out_dim + Self.out_dim
    comptime CACHE_SIZE: Int = Self.in_dim + Self.out_dim
    comptime FUSED_COUNT: Int = 3

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
        """Forward: y = act(x @ W + b), cache input and activation state."""
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](params.ptr)
        var b = LayoutTensor[
            dtype, Layout.row_major(Self.out_dim), MutAnyOrigin
        ](params.ptr + Self.in_dim * Self.out_dim)

        for ba in range(BATCH):
            # Cache input
            for i in range(Self.in_dim):
                cache[ba, i] = input[ba, i]

            for j in range(Self.out_dim):
                var acc: output.element_type = b[j]
                for k in range(Self.in_dim):
                    acc += input[ba, k] * W[k, j]
                # Apply activation and cache
                var pre_act = rebind[Scalar[dtype]](acc)
                var act_out = Self.ACT.forward(pre_act)
                cache[ba, Self.in_dim + j] = Self.ACT.cache(pre_act, act_out)
                output[ba, j] = act_out

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
        """Backward with fused activation gradient."""
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](params.ptr)
        var dW = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](grad_params.ptr)
        var db = LayoutTensor[
            dtype, Layout.row_major(Self.out_dim), MutAnyOrigin
        ](grad_params.ptr + Self.in_dim * Self.out_dim)

        for ba in range(BATCH):
            # dx = masked_dy @ W.T
            for i in range(Self.in_dim):
                var acc: grad_output.element_type = 0
                for j in range(Self.out_dim):
                    var cache_val = rebind[Scalar[dtype]](cache[ba, Self.in_dim + j])
                    var grad_val = rebind[Scalar[dtype]](grad_output[ba, j])
                    var masked_dy = Self.ACT.backward(cache_val, grad_val)
                    acc += masked_dy * W[i, j]
                grad_input[ba, i] = acc

            # dW += x.T @ masked_dy, db += masked_dy
            for j in range(Self.out_dim):
                var cache_val = rebind[Scalar[dtype]](cache[ba, Self.in_dim + j])
                var grad_val = rebind[Scalar[dtype]](grad_output[ba, j])
                var masked_dy = Self.ACT.backward(cache_val, grad_val)
                db[j] = db[j] + masked_dy
                for i in range(Self.in_dim):
                    dW[i, j] = dW[i, j] + cache[ba, i] * masked_dy

    # =========================================================================
    # GPU kernel implementations
    # =========================================================================

    @always_inline
    @staticmethod
    fn eval_kernel_impl[
        BATCH: Int
    ](
        output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.out_dim), MutAnyOrigin
        ],
        input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
        ],
        W: LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), ImmutAnyOrigin
        ],
        b: LayoutTensor[dtype, Layout.row_major(Self.out_dim), ImmutAnyOrigin],
        cache: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.in_dim + Self.out_dim),
            MutAnyOrigin,
        ],
    ):
        """Fused forward: y = act(x @ W + b).

        Grid: ((out_dim + TILE - 1) // TILE, (BATCH + TILE - 1) // TILE)
        Block: (TILE, TILE)
        """
        var local_row = Int(thread_idx.y)
        var local_col = Int(thread_idx.x)
        var global_row = Int(block_idx.y) * TILE + local_row
        var global_col = Int(block_idx.x) * TILE + local_col

        var input_shared = LayoutTensor[
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

        # Init with bias
        var acc: Scalar[dtype] = 0
        if global_col < Self.out_dim:
            acc = rebind[Scalar[dtype]](b[global_col])

        comptime num_tiles = (Self.in_dim + TILE - 1) // TILE

        for tile_idx in range(num_tiles):
            var in_col = tile_idx * TILE + local_col
            if global_row < BATCH and in_col < Self.in_dim:
                var x_val = input[global_row, in_col]
                input_shared[local_row, local_col] = x_val
                if Int(block_idx.x) == 0:
                    cache[global_row, in_col] = x_val
            else:
                input_shared[local_row, local_col] = 0

            var W_row = tile_idx * TILE + local_row
            if W_row < Self.in_dim and global_col < Self.out_dim:
                W_shared[local_row, local_col] = W[W_row, global_col]
            else:
                W_shared[local_row, local_col] = 0

            barrier()

            comptime for k in range(TILE):
                acc += rebind[Scalar[dtype]](
                    input_shared[local_row, k]
                ) * rebind[Scalar[dtype]](W_shared[k, local_col])

            barrier()

        if global_row < BATCH and global_col < Self.out_dim:
            # Apply activation and cache
            var act_out = Self.ACT.forward(acc)
            cache[global_row, Self.in_dim + global_col] = Self.ACT.cache(
                acc, act_out
            )
            output[global_row, global_col] = act_out

    @always_inline
    @staticmethod
    fn backward_kernel_impl[
        BATCH: Int
    ](
        grad_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), MutAnyOrigin
        ],
        dW: LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ],
        db: LayoutTensor[dtype, Layout.row_major(Self.out_dim), MutAnyOrigin],
        grad_output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.out_dim), ImmutAnyOrigin
        ],
        W: LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), ImmutAnyOrigin
        ],
        cache: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.in_dim + Self.out_dim),
            ImmutAnyOrigin,
        ],
    ):
        """Fused backward with activation gradient, dual-region grid."""
        comptime dx_grid_x = (Self.in_dim + TILE - 1) // TILE
        comptime dx_grid_y = (BATCH + TILE - 1) // TILE
        comptime dW_grid_y = (Self.in_dim + TILE - 1) // TILE

        var local_row = Int(thread_idx.y)
        var local_col = Int(thread_idx.x)
        var block_y = Int(block_idx.y)

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
            # Region 1: dx = (dy * act_grad) @ W.T
            var global_row = block_y * TILE + local_row
            var global_col = Int(block_idx.x) * TILE + local_col

            var acc: Scalar[dtype] = 0
            comptime num_tiles = (Self.out_dim + TILE - 1) // TILE

            for tile_idx in range(num_tiles):
                var dy_col = tile_idx * TILE + local_col
                if global_row < BATCH and dy_col < Self.out_dim:
                    var grad_val = rebind[Scalar[dtype]](grad_output[global_row, dy_col])
                    var cache_val = rebind[Scalar[dtype]](cache[global_row, Self.in_dim + dy_col])
                    shared_A[local_row, local_col] = Self.ACT.backward(
                        cache_val, grad_val
                    )
                else:
                    shared_A[local_row, local_col] = 0

                var WT_row = tile_idx * TILE + local_row
                if global_col < Self.in_dim and WT_row < Self.out_dim:
                    shared_B[local_row, local_col] = W[global_col, WT_row]
                else:
                    shared_B[local_row, local_col] = 0

                barrier()

                comptime for k in range(TILE):
                    acc += rebind[Scalar[dtype]](
                        shared_A[local_row, k]
                    ) * rebind[Scalar[dtype]](shared_B[k, local_col])

                barrier()

            if global_row < BATCH and global_col < Self.in_dim:
                grad_input[global_row, global_col] = acc
        else:
            # Region 2: dW = cache_input.T @ masked_dy, db = sum(masked_dy)
            var dW_block_y = block_y - dx_grid_y
            var global_row = dW_block_y * TILE + local_row  # in_dim
            var global_col = Int(block_idx.x) * TILE + local_col  # out_dim

            var acc: Scalar[dtype] = 0
            var db_acc: Scalar[dtype] = 0
            comptime num_tiles = (BATCH + TILE - 1) // TILE

            for tile_idx in range(num_tiles):
                var batch_col = tile_idx * TILE + local_col
                if batch_col < BATCH and global_row < Self.in_dim:
                    shared_A[local_row, local_col] = cache[
                        batch_col, global_row
                    ]
                else:
                    shared_A[local_row, local_col] = 0

                var batch_row = tile_idx * TILE + local_row
                if batch_row < BATCH and global_col < Self.out_dim:
                    var grad_val = rebind[Scalar[dtype]](grad_output[batch_row, global_col])
                    var cache_val = rebind[Scalar[dtype]](cache[
                        batch_row, Self.in_dim + global_col
                    ])
                    var masked_grad = Self.ACT.backward(cache_val, grad_val)
                    shared_B[local_row, local_col] = masked_grad
                    if dW_block_y == 0:
                        db_acc += rebind[Scalar[dtype]](masked_grad)
                else:
                    shared_B[local_row, local_col] = 0

                barrier()

                comptime for k in range(TILE):
                    acc += rebind[Scalar[dtype]](
                        shared_A[local_row, k]
                    ) * rebind[Scalar[dtype]](shared_B[k, local_col])

                barrier()

            if global_row < Self.in_dim and global_col < Self.out_dim:
                dW[global_row, global_col] = acc

            if dW_block_y == 0 and global_col < Self.out_dim:
                shared_A[local_row, local_col] = db_acc
                barrier()
                if local_row == 0:
                    var total: Scalar[dtype] = 0
                    for r in range(TILE):
                        total += rebind[Scalar[dtype]](
                            shared_A[r, local_col]
                        )
                    db[global_col] = total

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
    ) raises:
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), ImmutAnyOrigin
        ](params.ptr)
        var b = LayoutTensor[
            dtype, Layout.row_major(Self.out_dim), ImmutAnyOrigin
        ](params.ptr + Self.in_dim * Self.out_dim)
        var input_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
        ](input.ptr)
        var cache_full = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.in_dim + Self.out_dim),
            MutAnyOrigin,
        ](cache.ptr)

        comptime grid_x = (Self.out_dim + TILE - 1) // TILE
        comptime grid_y = (BATCH + TILE - 1) // TILE

        @always_inline
        fn wrapper(
            output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.out_dim), MutAnyOrigin
            ],
            input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
            ],
            W: LayoutTensor[
                dtype,
                Layout.row_major(Self.in_dim, Self.out_dim),
                ImmutAnyOrigin,
            ],
            b: LayoutTensor[
                dtype, Layout.row_major(Self.out_dim), ImmutAnyOrigin
            ],
            cache: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.in_dim + Self.out_dim),
                MutAnyOrigin,
            ],
        ):
            Self.eval_kernel_impl[BATCH](output, input, W, b, cache)

        ctx.enqueue_function[wrapper, wrapper](
            output, input_immut, W, b, cache_full,
            grid_dim=(grid_x, grid_y),
            block_dim=(TILE, TILE),
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
    ) raises:
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), ImmutAnyOrigin
        ](params.ptr)
        var cache_immut = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Self.in_dim + Self.out_dim),
            ImmutAnyOrigin,
        ](cache.ptr)
        var grad_output_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.out_dim), ImmutAnyOrigin
        ](grad_output.ptr)
        var dW = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](grad_params.ptr)
        var db = LayoutTensor[
            dtype, Layout.row_major(Self.out_dim), MutAnyOrigin
        ](grad_params.ptr + Self.in_dim * Self.out_dim)

        comptime dx_grid_x = (Self.in_dim + TILE - 1) // TILE
        comptime dx_grid_y = (BATCH + TILE - 1) // TILE
        comptime dW_grid_x = (Self.out_dim + TILE - 1) // TILE
        comptime dW_grid_y = (Self.in_dim + TILE - 1) // TILE
        comptime grid_x = dx_grid_x if dx_grid_x > dW_grid_x else dW_grid_x
        comptime grid_y = dx_grid_y + dW_grid_y

        @always_inline
        fn bwd_wrapper(
            grad_input: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.in_dim), MutAnyOrigin
            ],
            dW: LayoutTensor[
                dtype,
                Layout.row_major(Self.in_dim, Self.out_dim),
                MutAnyOrigin,
            ],
            db: LayoutTensor[
                dtype, Layout.row_major(Self.out_dim), MutAnyOrigin
            ],
            grad_output: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.out_dim), ImmutAnyOrigin
            ],
            W: LayoutTensor[
                dtype,
                Layout.row_major(Self.in_dim, Self.out_dim),
                ImmutAnyOrigin,
            ],
            cache: LayoutTensor[
                dtype,
                Layout.row_major(BATCH, Self.in_dim + Self.out_dim),
                ImmutAnyOrigin,
            ],
        ):
            Self.backward_kernel_impl[BATCH](
                grad_input, dW, db, grad_output, W, cache
            )

        ctx.enqueue_function[bwd_wrapper, bwd_wrapper](
            grad_input, dW, db, grad_output_immut, W, cache_immut,
            grid_dim=(grid_x, grid_y),
            block_dim=(TILE, TILE),
        )
