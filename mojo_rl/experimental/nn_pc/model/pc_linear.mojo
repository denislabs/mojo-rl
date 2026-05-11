"""PCLinear — one PCN level (W matrix + bundled activation).

W is stored as `[in_dim, out_dim]` (matches nn.Linear convention exactly).
PCN predict goes top-down through W^T:

    a      = x_above @ W^T          # [B, OUT_DIM] @ [OUT_DIM, IN_DIM] = [B, IN_DIM]
    x_hat  = ACT(a)
    h      = eps * ACT'(a)          # gain-modulated error
    z      = h @ W                  # pulled-back message to latent above [B, OUT_DIM]
    grad_W = h^T @ x_above          # [IN_DIM, B] @ [B, OUT_DIM] = [IN_DIM, OUT_DIM]

For the readout, use ACT=PCIdentity:
    PCLinear[NUM_CLASSES, TOP_HIDDEN, PCIdentity]
"""

from layout import Layout, LayoutTensor
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.gpu.primitives import lane_id
from std.gpu.compute.mma import mma
from std.sys import is_nvidia_gpu

from mojo_rl.nn.constants import (
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
from mojo_rl.nn.initializer import Initializer

from ..predictive_model import PCActivation, PCReLU, PCLayer


struct PCLinear[
    in_dim: Int,
    out_dim: Int,
    ACT: PCActivation = PCReLU,
](PCLayer):
    """PCN level: W [in_dim, out_dim] + bundled activation `ACT` on the prediction.

    Naming matches `nn.Linear[in_dim, out_dim]`:
      - in_dim is the lower (predicted) dim — what `predict` produces
      - out_dim is the upper (latent above) dim — what `predict` consumes as `x_above`

    For the readout, pass `ACT=PCIdentity`.
    """

    comptime IN_DIM: Int = Self.in_dim
    comptime OUT_DIM: Int = Self.out_dim
    comptime PARAM_SIZE: Int = Self.in_dim * Self.out_dim

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

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
        """Init W via INIT with fan_in=in_dim, fan_out=out_dim (matches nn.Linear).
        """
        INIT.init[Self.PARAM_SIZE, Self.in_dim, Self.out_dim, dtype](params)

    # =========================================================================
    # Top-down prediction:  a = x_above @ W^T;  x_hat = ACT(a)
    # =========================================================================

    @staticmethod
    def predict[
        BATCH: Int, dtype: DType = DType.float32
    ](
        x_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut x_hat: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        mut a: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
    ):
        """a[b, i] = sum_j x_above[b, j] * W[i, j];  x_hat = ACT(a)."""
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](params.ptr)
        for b in range(BATCH):
            for i in range(Self.in_dim):
                var s: Scalar[dtype] = 0
                for j in range(Self.out_dim):
                    s += rebind[Scalar[dtype]](x_above[b, j]) * rebind[
                        Scalar[dtype]
                    ](W[i, j])
                a[b, i] = s
        Self.ACT.apply[BATCH, Self.in_dim, dtype](a, x_hat)

    # =========================================================================
    # Gain-modulated error:  h = eps * ACT'(a)
    # =========================================================================

    @staticmethod
    def gain_modulated_error[
        BATCH: Int, dtype: DType = DType.float32
    ](
        eps: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        a: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        mut h: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
    ):
        Self.ACT.apply_derivative_mul[BATCH, Self.in_dim, dtype](a, eps, h)

    # =========================================================================
    # Pull-back to latent above:  z = h @ W
    # =========================================================================

    @staticmethod
    def pull_back[
        BATCH: Int, dtype: DType = DType.float32
    ](
        h: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut z: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
    ):
        """z[b, j] = sum_i h[b, i] * W[i, j]."""
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](params.ptr)
        for b in range(BATCH):
            for j in range(Self.out_dim):
                var s: Scalar[dtype] = 0
                for i in range(Self.in_dim):
                    s += rebind[Scalar[dtype]](h[b, i]) * rebind[
                        Scalar[dtype]
                    ](W[i, j])
                z[b, j] = s

    # =========================================================================
    # Weight update:  W += scale * (h^T @ x_above)
    # =========================================================================

    @staticmethod
    def weight_grad_step[
        BATCH: Int, dtype: DType = DType.float32
    ](
        h: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        x_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        scale: Scalar[dtype],
    ):
        """W += scale * sum_b h[b, i] * x_above[b, j].

        Caller supplies signed scale:
          - non-readout (h = gain-modulated error): scale = +eta_learn / BATCH
            (PyTorch: grad = -h^T x_above / B; W -= eta * grad → W += eta/B * h^T x_above)
          - readout    (h = eps_sup, ACT=Identity): scale = -eta_learn / BATCH
            (PyTorch: grad = +eps_sup^T x_above / B; W -= eta * grad → W -= eta/B * eps^T x_above)
        """
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](params.ptr)
        for i in range(Self.in_dim):
            for j in range(Self.out_dim):
                var s: Scalar[dtype] = 0
                for b in range(BATCH):
                    s += rebind[Scalar[dtype]](h[b, i]) * rebind[
                        Scalar[dtype]
                    ](x_above[b, j])
                W[i, j] += scale * s

    # =========================================================================
    # GPU kernels — Apple register-tiled 2x2 (32×32 block, 16-elem K tile)
    # Modeled after nn/autodiff/primitives/matmul.mojo's _2x2 family.
    # =========================================================================

    @always_inline
    @staticmethod
    def _predict_kernel_2x2[
        BATCH: Int, dtype: DType,
    ](
        x_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.out_dim), ImmutAnyOrigin
        ],
        W: LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), ImmutAnyOrigin
        ],
        a: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), MutAnyOrigin
        ],
    ):
        """a = x_above @ W^T.  Output [BATCH, in_dim], reduce over out_dim.

        Mirrors MatMul.backward_dx_kernel_2x2 — same shape semantics.
        Grid: ((in_dim + 31) // 32, (BATCH + 31) // 32),  Block: (256, 1).
        """
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

        comptime num_k_tiles = (Self.out_dim + SK - 1) // SK

        for k_tile in range(num_k_tiles):
            var k_off = k_tile * SK

            # A = x_above [BT(BATCH), SK(out_dim)]
            var a_r0 = tid // SK
            var a_c0 = tid % SK
            var a_r1 = (tid + 256) // SK
            var a_c1 = (tid + 256) % SK
            if block_row + a_r0 < BATCH and k_off + a_c0 < Self.out_dim:
                a_smem[a_r0, a_c0] = x_above[block_row + a_r0, k_off + a_c0]
            else:
                a_smem[a_r0, a_c0] = 0
            if block_row + a_r1 < BATCH and k_off + a_c1 < Self.out_dim:
                a_smem[a_r1, a_c1] = x_above[block_row + a_r1, k_off + a_c1]
            else:
                a_smem[a_r1, a_c1] = 0

            # B = W^T [SK(out_dim), BT(in_dim)]: B[k, c] = W[block_col+c, k_off+k]
            var b_r0 = tid // BT
            var b_c0 = tid % BT
            var b_r1 = (tid + 256) // BT
            var b_c1 = (tid + 256) % BT
            if k_off + b_r0 < Self.out_dim and block_col + b_c0 < Self.in_dim:
                b_smem[b_r0, b_c0] = W[block_col + b_c0, k_off + b_r0]
            else:
                b_smem[b_r0, b_c0] = 0
            if k_off + b_r1 < Self.out_dim and block_col + b_c1 < Self.in_dim:
                b_smem[b_r1, b_c1] = W[block_col + b_c1, k_off + b_r1]
            else:
                b_smem[b_r1, b_c1] = 0

            barrier()

            for k in range(SK):
                if k_off + k < Self.out_dim:
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
        if gr0 < BATCH and gc0 < Self.in_dim:
            a[gr0, gc0] = acc00
        if gr0 < BATCH and gc0 + 1 < Self.in_dim:
            a[gr0, gc0 + 1] = acc01
        if gr0 + 1 < BATCH and gc0 < Self.in_dim:
            a[gr0 + 1, gc0] = acc10
        if gr0 + 1 < BATCH and gc0 + 1 < Self.in_dim:
            a[gr0 + 1, gc0 + 1] = acc11

    @always_inline
    @staticmethod
    def _pull_back_kernel_2x2[
        BATCH: Int, dtype: DType,
    ](
        h: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
        ],
        W: LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), ImmutAnyOrigin
        ],
        z: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.out_dim), MutAnyOrigin
        ],
    ):
        """z = h @ W.  Output [BATCH, out_dim], reduce over in_dim.

        Mirrors MatMul.eval_kernel_2x2 (sans cache/bias).
        Grid: ((out_dim + 31) // 32, (BATCH + 31) // 32),  Block: (256, 1).
        """
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

        comptime num_k_tiles = (Self.in_dim + SK - 1) // SK

        for k_tile in range(num_k_tiles):
            var k_off = k_tile * SK

            # A = h [BT(BATCH), SK(in_dim)]
            var a_r0 = tid // SK
            var a_c0 = tid % SK
            var a_r1 = (tid + 256) // SK
            var a_c1 = (tid + 256) % SK
            if block_row + a_r0 < BATCH and k_off + a_c0 < Self.in_dim:
                a_smem[a_r0, a_c0] = h[block_row + a_r0, k_off + a_c0]
            else:
                a_smem[a_r0, a_c0] = 0
            if block_row + a_r1 < BATCH and k_off + a_c1 < Self.in_dim:
                a_smem[a_r1, a_c1] = h[block_row + a_r1, k_off + a_c1]
            else:
                a_smem[a_r1, a_c1] = 0

            # B = W [SK(in_dim), BT(out_dim)]
            var b_r0 = tid // BT
            var b_c0 = tid % BT
            var b_r1 = (tid + 256) // BT
            var b_c1 = (tid + 256) % BT
            if k_off + b_r0 < Self.in_dim and block_col + b_c0 < Self.out_dim:
                b_smem[b_r0, b_c0] = W[k_off + b_r0, block_col + b_c0]
            else:
                b_smem[b_r0, b_c0] = 0
            if k_off + b_r1 < Self.in_dim and block_col + b_c1 < Self.out_dim:
                b_smem[b_r1, b_c1] = W[k_off + b_r1, block_col + b_c1]
            else:
                b_smem[b_r1, b_c1] = 0

            barrier()

            for k in range(SK):
                if k_off + k < Self.in_dim:
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
        if gr0 < BATCH and gc0 < Self.out_dim:
            z[gr0, gc0] = acc00
        if gr0 < BATCH and gc0 + 1 < Self.out_dim:
            z[gr0, gc0 + 1] = acc01
        if gr0 + 1 < BATCH and gc0 < Self.out_dim:
            z[gr0 + 1, gc0] = acc10
        if gr0 + 1 < BATCH and gc0 + 1 < Self.out_dim:
            z[gr0 + 1, gc0 + 1] = acc11

    @always_inline
    @staticmethod
    def _weight_grad_kernel_2x2[
        BATCH: Int, dtype: DType,
    ](
        h: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
        ],
        x_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.out_dim), ImmutAnyOrigin
        ],
        W: LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ],
        scale: Scalar[dtype],
    ):
        """W += scale * (h^T @ x_above).  Output [in_dim, out_dim], reduce over BATCH.

        Mirrors MatMul.backward_dW_kernel_2x2 but accumulates into W with a scale
        instead of overwriting.
        Grid: ((out_dim + 31) // 32, (in_dim + 31) // 32),  Block: (256, 1).
        """
        comptime BT = 32
        comptime SK = 16
        var tid = Int(thread_idx.x)
        var sub_r = tid // 16
        var sub_c = tid % 16
        var block_row = Int(block_idx.y) * BT  # in_dim axis
        var block_col = Int(block_idx.x) * BT  # out_dim axis

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

        comptime num_k_tiles = (BATCH + SK - 1) // SK

        for k_tile in range(num_k_tiles):
            var k_off = k_tile * SK

            # A = h^T [BT(in_dim), SK(BATCH)]: A[r, k] = h[k_off+k, block_row+r]
            var a_r0 = tid // SK
            var a_c0 = tid % SK
            var a_r1 = (tid + 256) // SK
            var a_c1 = (tid + 256) % SK
            if k_off + a_c0 < BATCH and block_row + a_r0 < Self.in_dim:
                a_smem[a_r0, a_c0] = h[k_off + a_c0, block_row + a_r0]
            else:
                a_smem[a_r0, a_c0] = 0
            if k_off + a_c1 < BATCH and block_row + a_r1 < Self.in_dim:
                a_smem[a_r1, a_c1] = h[k_off + a_c1, block_row + a_r1]
            else:
                a_smem[a_r1, a_c1] = 0

            # B = x_above [SK(BATCH), BT(out_dim)]
            var b_r0 = tid // BT
            var b_c0 = tid % BT
            var b_r1 = (tid + 256) // BT
            var b_c1 = (tid + 256) % BT
            if k_off + b_r0 < BATCH and block_col + b_c0 < Self.out_dim:
                b_smem[b_r0, b_c0] = x_above[k_off + b_r0, block_col + b_c0]
            else:
                b_smem[b_r0, b_c0] = 0
            if k_off + b_r1 < BATCH and block_col + b_c1 < Self.out_dim:
                b_smem[b_r1, b_c1] = x_above[k_off + b_r1, block_col + b_c1]
            else:
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
        if gr0 < Self.in_dim and gc0 < Self.out_dim:
            W[gr0, gc0] += scale * acc00
        if gr0 < Self.in_dim and gc0 + 1 < Self.out_dim:
            W[gr0, gc0 + 1] += scale * acc01
        if gr0 + 1 < Self.in_dim and gc0 < Self.out_dim:
            W[gr0 + 1, gc0] += scale * acc10
        if gr0 + 1 < Self.in_dim and gc0 + 1 < Self.out_dim:
            W[gr0 + 1, gc0 + 1] += scale * acc11

    # =========================================================================
    # GPU kernels — NVIDIA MMA (m16n8k8, 8-warp 32×32 block)
    # =========================================================================

    @always_inline
    @staticmethod
    def _predict_kernel_mma[
        BATCH: Int, dtype: DType,
    ](
        x_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.out_dim), ImmutAnyOrigin
        ],
        W: LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), ImmutAnyOrigin
        ],
        a: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), MutAnyOrigin
        ],
    ):
        """MMA predict: a = x_above @ W^T.  Mirrors MatMul.backward_dx_kernel_mma."""
        comptime if is_nvidia_gpu():
            var tid = Int(thread_idx.x)
            var warp_id = tid // 32
            var warp_m = warp_id // MMA_WARPS_N
            var warp_n = warp_id % MMA_WARPS_N

            var block_row = Int(block_idx.y) * MMA_BLOCK_M  # BATCH axis
            var block_col = Int(block_idx.x) * MMA_BLOCK_N  # in_dim axis

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

            comptime num_k_tiles = (Self.out_dim + MMA_K - 1) // MMA_K

            for k_tile in range(num_k_tiles):
                var k_off = k_tile * MMA_K

                # A = x_above [32, 8]
                var a_r = tid // MMA_K
                var a_c = tid % MMA_K
                var ga_r = block_row + a_r
                var ga_c = k_off + a_c
                if ga_r < BATCH and ga_c < Self.out_dim:
                    a_smem[a_r, a_c] = x_above[ga_r, ga_c]
                else:
                    a_smem[a_r, a_c] = 0

                # B = W^T [8, 32]: B[k, c] = W[block_col+c, k_off+k]
                var b_r = tid // MMA_BLOCK_N
                var b_c = tid % MMA_BLOCK_N
                var w_row = block_col + b_c  # in_dim
                var w_col = k_off + b_r       # out_dim
                if w_row < Self.in_dim and w_col < Self.out_dim:
                    b_smem[b_r, b_c] = W[w_row, w_col]
                else:
                    b_smem[b_r, b_c] = 0

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

            if r0 < BATCH and c0 < Self.in_dim:
                a[r0, c0] = acc[0].cast[dtype]()
            if r0 < BATCH and c1 < Self.in_dim:
                a[r0, c1] = acc[1].cast[dtype]()
            if r1 < BATCH and c0 < Self.in_dim:
                a[r1, c0] = acc[2].cast[dtype]()
            if r1 < BATCH and c1 < Self.in_dim:
                a[r1, c1] = acc[3].cast[dtype]()

    @always_inline
    @staticmethod
    def _pull_back_kernel_mma[
        BATCH: Int, dtype: DType,
    ](
        h: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
        ],
        W: LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), ImmutAnyOrigin
        ],
        z: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.out_dim), MutAnyOrigin
        ],
    ):
        """MMA pull-back: z = h @ W.  Mirrors MatMul.eval_kernel_mma (sans cache)."""
        comptime if is_nvidia_gpu():
            var tid = Int(thread_idx.x)
            var warp_id = tid // 32
            var warp_m = warp_id // MMA_WARPS_N
            var warp_n = warp_id % MMA_WARPS_N

            var block_row = Int(block_idx.y) * MMA_BLOCK_M  # BATCH
            var block_col = Int(block_idx.x) * MMA_BLOCK_N  # out_dim

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

            comptime num_k_tiles = (Self.in_dim + MMA_K - 1) // MMA_K

            for k_tile in range(num_k_tiles):
                var k_off = k_tile * MMA_K

                # A = h [32, 8]
                var a_r = tid // MMA_K
                var a_c = tid % MMA_K
                var ga_r = block_row + a_r
                var ga_c = k_off + a_c
                if ga_r < BATCH and ga_c < Self.in_dim:
                    a_smem[a_r, a_c] = h[ga_r, ga_c]
                else:
                    a_smem[a_r, a_c] = 0

                # B = W [8, 32]
                var b_r = tid // MMA_BLOCK_N
                var b_c = tid % MMA_BLOCK_N
                var gb_r = k_off + b_r
                var gb_c = block_col + b_c
                if gb_r < Self.in_dim and gb_c < Self.out_dim:
                    b_smem[b_r, b_c] = W[gb_r, gb_c]
                else:
                    b_smem[b_r, b_c] = 0

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

            if r0 < BATCH and c0 < Self.out_dim:
                z[r0, c0] = acc[0].cast[dtype]()
            if r0 < BATCH and c1 < Self.out_dim:
                z[r0, c1] = acc[1].cast[dtype]()
            if r1 < BATCH and c0 < Self.out_dim:
                z[r1, c0] = acc[2].cast[dtype]()
            if r1 < BATCH and c1 < Self.out_dim:
                z[r1, c1] = acc[3].cast[dtype]()

    @always_inline
    @staticmethod
    def _weight_grad_kernel_mma[
        BATCH: Int, dtype: DType,
    ](
        h: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
        ],
        x_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.out_dim), ImmutAnyOrigin
        ],
        W: LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ],
        scale: Scalar[dtype],
    ):
        """MMA weight-grad: W += scale * (h^T @ x_above).

        Mirrors MatMul.backward_dW_kernel_mma but accumulates into W with a
        scale instead of overwriting.
        """
        comptime if is_nvidia_gpu():
            var tid = Int(thread_idx.x)
            var warp_id = tid // 32
            var warp_m = warp_id // MMA_WARPS_N
            var warp_n = warp_id % MMA_WARPS_N

            var block_row = Int(block_idx.y) * MMA_BLOCK_M  # in_dim
            var block_col = Int(block_idx.x) * MMA_BLOCK_N  # out_dim

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

                # A = h^T [32, 8]: A[r, k] = h[k_off+k, block_row+r]
                var a_r = tid // MMA_K
                var a_c = tid % MMA_K
                if k_off + a_c < BATCH and block_row + a_r < Self.in_dim:
                    a_smem[a_r, a_c] = h[k_off + a_c, block_row + a_r]
                else:
                    a_smem[a_r, a_c] = 0

                # B = x_above [8, 32]
                var b_r = tid // MMA_BLOCK_N
                var b_c = tid % MMA_BLOCK_N
                var gb_r = k_off + b_r
                var gb_c = block_col + b_c
                if gb_r < BATCH and gb_c < Self.out_dim:
                    b_smem[b_r, b_c] = x_above[gb_r, gb_c]
                else:
                    b_smem[b_r, b_c] = 0

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

            if r0 < Self.in_dim and c0 < Self.out_dim:
                W[r0, c0] += scale * acc[0].cast[dtype]()
            if r0 < Self.in_dim and c1 < Self.out_dim:
                W[r0, c1] += scale * acc[1].cast[dtype]()
            if r1 < Self.in_dim and c0 < Self.out_dim:
                W[r1, c0] += scale * acc[2].cast[dtype]()
            if r1 < Self.in_dim and c1 < Self.out_dim:
                W[r1, c1] += scale * acc[3].cast[dtype]()

    # =========================================================================
    # Auto-dispatching launchers (MMA on NVIDIA, 2x2 on Apple)
    # =========================================================================

    @staticmethod
    def predict_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        x_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut x_hat: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        mut a: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
    ) raises:
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), ImmutAnyOrigin
        ](params.ptr)
        var x_above_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.out_dim), ImmutAnyOrigin
        ](x_above.ptr)

        comptime grid_x = (Self.in_dim + MMA_BLOCK_N - 1) // MMA_BLOCK_N
        comptime grid_y = (BATCH + MMA_BLOCK_M - 1) // MMA_BLOCK_M

        @parameter
        @always_inline
        def wrapper(
            x_above: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.out_dim), ImmutAnyOrigin
            ],
            W: LayoutTensor[
                dtype,
                Layout.row_major(Self.in_dim, Self.out_dim),
                ImmutAnyOrigin,
            ],
            a: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.in_dim), MutAnyOrigin
            ],
        ):
            comptime if is_nvidia_gpu():
                Self._predict_kernel_mma[BATCH, dtype](x_above, W, a)
            else:
                Self._predict_kernel_2x2[BATCH, dtype](x_above, W, a)

        ctx.enqueue_function[wrapper](
            x_above_immut, W, a,
            grid_dim=(grid_x, grid_y),
            block_dim=(MMA_BLOCK_THREADS, 1),
        )
        Self.ACT.apply_gpu[BATCH, Self.in_dim, dtype](ctx, a, x_hat)

    @staticmethod
    def gain_modulated_error_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        eps: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        a: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        mut h: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
    ) raises:
        Self.ACT.apply_derivative_mul_gpu[BATCH, Self.in_dim, dtype](
            ctx, a, eps, h
        )

    @staticmethod
    def pull_back_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        h: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        mut z: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
    ) raises:
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), ImmutAnyOrigin
        ](params.ptr)
        var h_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
        ](h.ptr)

        comptime grid_x = (Self.out_dim + MMA_BLOCK_N - 1) // MMA_BLOCK_N
        comptime grid_y = (BATCH + MMA_BLOCK_M - 1) // MMA_BLOCK_M

        @parameter
        @always_inline
        def wrapper(
            h: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
            ],
            W: LayoutTensor[
                dtype,
                Layout.row_major(Self.in_dim, Self.out_dim),
                ImmutAnyOrigin,
            ],
            z: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.out_dim), MutAnyOrigin
            ],
        ):
            comptime if is_nvidia_gpu():
                Self._pull_back_kernel_mma[BATCH, dtype](h, W, z)
            else:
                Self._pull_back_kernel_2x2[BATCH, dtype](h, W, z)

        ctx.enqueue_function[wrapper](
            h_immut, W, z,
            grid_dim=(grid_x, grid_y),
            block_dim=(MMA_BLOCK_THREADS, 1),
        )

    @staticmethod
    def weight_grad_step_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        h: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        x_above: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        mut params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        scale: Scalar[dtype],
    ) raises:
        var W = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
        ](params.ptr)
        var h_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
        ](h.ptr)
        var xa_immut = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.out_dim), ImmutAnyOrigin
        ](x_above.ptr)

        comptime grid_x = (Self.out_dim + MMA_BLOCK_N - 1) // MMA_BLOCK_N
        comptime grid_y = (Self.in_dim + MMA_BLOCK_M - 1) // MMA_BLOCK_M

        @parameter
        @always_inline
        def wrapper(
            h: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.in_dim), ImmutAnyOrigin
            ],
            x_above: LayoutTensor[
                dtype, Layout.row_major(BATCH, Self.out_dim), ImmutAnyOrigin
            ],
            W: LayoutTensor[
                dtype, Layout.row_major(Self.in_dim, Self.out_dim), MutAnyOrigin
            ],
            scale: Scalar[dtype],
        ):
            comptime if is_nvidia_gpu():
                Self._weight_grad_kernel_mma[BATCH, dtype](h, x_above, W, scale)
            else:
                Self._weight_grad_kernel_2x2[BATCH, dtype](h, x_above, W, scale)

        ctx.enqueue_function[wrapper](
            h_immut, xa_immut, W, scale,
            grid_dim=(grid_x, grid_y),
            block_dim=(MMA_BLOCK_THREADS, 1),
        )
