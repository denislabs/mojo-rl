"""PCEncoder — 2-layer MLP amortized posterior for hybrid PC (Tschantz 2023).

Encoder maps `[prev_z, action, obs]` → initial guess for the latent `z_t`.
Pairs with `PCTrainer.compute_grads_from_latents`: encoder produces the warm
start, K SGLD refinement steps then settle to the local energy minimum.

    z_pre  = W1 · input + b1
    z_hid  = tanh(z_pre)
    output = W2 · z_hid + b2

Matches the hand-rolled encoder used in the world-model amortized PC tests
(`test_pendulum_amortized_pc.mojo`, `test_mountain_car_amortized_pc.mojo`).
Hidden activation is tanh (no parametrization yet — extend if a real use case
needs ReLU/GELU).

Layout (single flat param tensor):

    [ W1 (IN×HID) | b1 (HID) | W2 (HID×OUT) | b2 (OUT) ]

CPU only for now. The PC inference loop is the GPU bottleneck; encoder
forward/backward is small (a few hundred params for typical world-model
sizes) and runs on host without dominating wall time.
"""

from layout import Layout, LayoutTensor
from std.math import sqrt, tanh
from std.random.philox import Random as PhiloxRandom
from std.sys import CompilationTarget
from linalg.matmul import matmul as max_matmul
from linalg.matmul.cpu.apple_accelerate import (
    get_cblas_f32_function,
    _CBLASOrder,
    _CBLASTranspose,
)
from layout.tile_tensor import lt_to_tt


struct PCEncoder[in_dim: Int, hidden_dim: Int, out_dim: Int]:
    """2-layer MLP encoder with tanh hidden activation.

    Caller owns all buffers (params, scratch, output) so they can be allocated
    once and reused across many batches.
    """

    comptime IN_DIM: Int = Self.in_dim
    comptime HIDDEN_DIM: Int = Self.hidden_dim
    comptime OUT_DIM: Int = Self.out_dim

    comptime W1_SIZE: Int = Self.in_dim * Self.hidden_dim
    comptime B1_SIZE: Int = Self.hidden_dim
    comptime W2_SIZE: Int = Self.hidden_dim * Self.out_dim
    comptime B2_SIZE: Int = Self.out_dim

    comptime W1_OFFSET: Int = 0
    comptime B1_OFFSET: Int = Self.W1_SIZE
    comptime W2_OFFSET: Int = Self.W1_SIZE + Self.B1_SIZE
    comptime B2_OFFSET: Int = Self.W1_SIZE + Self.B1_SIZE + Self.W2_SIZE

    comptime PARAM_SIZE: Int = (
        Self.W1_SIZE + Self.B1_SIZE + Self.W2_SIZE + Self.B2_SIZE
    )

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit take: Self):
        pass

    # =========================================================================
    # Initialization — Xavier-uniform on W1, W2; zeros on biases
    # =========================================================================

    @staticmethod
    def xavier_init[
        dtype: DType = DType.float32
    ](
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        seed: UInt64,
    ):
        """Xavier-uniform init for W1, W2; zero biases.

        W1 uses `seed`, W2 uses `seed + 1` (independent RNG streams).
        """
        # W1: bound = sqrt(6 / (fan_in + fan_out))
        var rng1 = PhiloxRandom(seed=seed, offset=UInt64(0))
        var bound1 = sqrt(
            Float64(6.0) / Float64(Self.in_dim + Self.hidden_dim)
        )
        for i in range(Self.W1_SIZE):
            var u = Float64(rng1.step_uniform()[0])
            params.ptr[Self.W1_OFFSET + i] = Scalar[dtype](
                (u * 2.0 - 1.0) * bound1
            )
        # b1 = 0
        for i in range(Self.B1_SIZE):
            params.ptr[Self.B1_OFFSET + i] = Scalar[dtype](0.0)
        # W2: bound = sqrt(6 / (hidden_dim + out_dim))
        var rng2 = PhiloxRandom(seed=seed + UInt64(1), offset=UInt64(0))
        var bound2 = sqrt(
            Float64(6.0) / Float64(Self.hidden_dim + Self.out_dim)
        )
        for i in range(Self.W2_SIZE):
            var u = Float64(rng2.step_uniform()[0])
            params.ptr[Self.W2_OFFSET + i] = Scalar[dtype](
                (u * 2.0 - 1.0) * bound2
            )
        # b2 = 0
        for i in range(Self.B2_SIZE):
            params.ptr[Self.B2_OFFSET + i] = Scalar[dtype](0.0)

    # =========================================================================
    # Forward:  output = W2 · tanh(W1 · input + b1) + b2
    # =========================================================================

    @staticmethod
    def forward[
        BATCH: Int, dtype: DType = DType.float32
    ](
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        enc_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        mut hidden_pre: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN_DIM), MutAnyOrigin
        ],
        mut hidden_act: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN_DIM), MutAnyOrigin
        ],
        mut output: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
    ):
        """Forward pass; caches `hidden_pre`, `hidden_act` for backward."""
        # Param sub-views over the flat slab (no rebind; like the block path).
        var W1 = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.hidden_dim), MutAnyOrigin
        ](params.ptr + Self.W1_OFFSET)
        var b1 = LayoutTensor[
            dtype, Layout.row_major(Self.hidden_dim), MutAnyOrigin
        ](params.ptr + Self.B1_OFFSET)
        var W2 = LayoutTensor[
            dtype, Layout.row_major(Self.hidden_dim, Self.out_dim), MutAnyOrigin
        ](params.ptr + Self.W2_OFFSET)
        var b2 = LayoutTensor[
            dtype, Layout.row_major(Self.out_dim), MutAnyOrigin
        ](params.ptr + Self.B2_OFFSET)

        # h_pre = input @ W1
        try:
            max_matmul[target="cpu"](
                lt_to_tt(hidden_pre), lt_to_tt(enc_input), lt_to_tt(W1), None
            )
        except:
            pass
        # bias add + tanh (scalar element indexing, identical iteration order)
        for b in range(BATCH):
            for j in range(Self.hidden_dim):
                hidden_pre[b, j] = hidden_pre[b, j] + b1[j]
            for k in range(Self.hidden_dim):
                hidden_act[b, k] = Scalar[dtype](
                    tanh(Float64(rebind[Scalar[dtype]](hidden_pre[b, k])))
                )

        # output = h_act @ W2
        try:
            max_matmul[target="cpu"](
                lt_to_tt(output), lt_to_tt(hidden_act), lt_to_tt(W2), None
            )
        except:
            pass
        for b in range(BATCH):
            for j in range(Self.out_dim):
                output[b, j] = output[b, j] + b2[j]

    # =========================================================================
    # Backward:  given dL/d(output), accumulate into grads (zero-initialized).
    # Standard MLP chain rule.
    # =========================================================================

    @staticmethod
    def backward[
        BATCH: Int, dtype: DType = DType.float32
    ](
        params: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
        enc_input: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.IN_DIM), MutAnyOrigin
        ],
        hidden_act: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.HIDDEN_DIM), MutAnyOrigin
        ],
        dz_out: LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.OUT_DIM), MutAnyOrigin
        ],
        mut grads: LayoutTensor[
            dtype, Layout.row_major(Self.PARAM_SIZE), MutAnyOrigin
        ],
    ):
        """Writes gradients into `grads` (overwrites; caller need not zero)."""
        for i in range(Self.PARAM_SIZE):
            grads.ptr[i] = Scalar[dtype](0.0)

        # Grad + param sub-views over the slabs (no rebind; like the block path).
        var dW1 = LayoutTensor[
            dtype, Layout.row_major(Self.in_dim, Self.hidden_dim), MutAnyOrigin
        ](grads.ptr + Self.W1_OFFSET)
        var db1 = LayoutTensor[
            dtype, Layout.row_major(Self.hidden_dim), MutAnyOrigin
        ](grads.ptr + Self.B1_OFFSET)
        var dW2 = LayoutTensor[
            dtype, Layout.row_major(Self.hidden_dim, Self.out_dim), MutAnyOrigin
        ](grads.ptr + Self.W2_OFFSET)
        var db2 = LayoutTensor[
            dtype, Layout.row_major(Self.out_dim), MutAnyOrigin
        ](grads.ptr + Self.B2_OFFSET)
        var W2 = LayoutTensor[
            dtype, Layout.row_major(Self.hidden_dim, Self.out_dim), MutAnyOrigin
        ](params.ptr + Self.W2_OFFSET)

        # dW2 = h_act^T @ dz_out
        comptime if CompilationTarget.is_macos() and dtype == DType.float32:
            # Apple Accelerate cblas FFI boundary (Scalar→Float32 + origin),
            # kept as-is per the nn.storage convention.
            try:
                var cblas_gemm = get_cblas_f32_function()
                cblas_gemm(
                    _CBLASOrder.ROW_MAJOR,
                    _CBLASTranspose.TRANSPOSE,
                    _CBLASTranspose.NO_TRANSPOSE,
                    Int32(Self.hidden_dim),
                    Int32(Self.out_dim),
                    Int32(BATCH),
                    Float32(1.0),
                    rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
                        hidden_act.ptr
                    ),
                    Int32(Self.hidden_dim),
                    rebind[UnsafePointer[Float32, ImmutAnyOrigin]](dz_out.ptr),
                    Int32(Self.out_dim),
                    Float32(0.0),
                    rebind[UnsafePointer[Float32, MutAnyOrigin]](dW2.ptr),
                    Int32(Self.out_dim),
                )
            except:
                pass
        else:
            # Portable: materialize h_act^T into an owned List, GEMM via lt_to_tt.
            var cT = List[Scalar[dtype]](
                length=Self.hidden_dim * BATCH, fill=Scalar[dtype](0)
            )
            for bi in range(BATCH):
                for i in range(Self.hidden_dim):
                    cT[i * BATCH + bi] = rebind[Scalar[dtype]](
                        hidden_act[bi, i]
                    )
            var cT_view = LayoutTensor[
                dtype, Layout.row_major(Self.hidden_dim, BATCH), MutAnyOrigin
            ](cT)
            try:
                max_matmul[target="cpu"](
                    lt_to_tt(dW2), lt_to_tt(cT_view), lt_to_tt(dz_out), None
                )
            except:
                pass
            _ = cT^

        # db2 = column-sum(dz_out) (stays in LayoutTensor element type)
        for j in range(Self.out_dim):
            var s = dz_out[0, j]
            for b in range(1, BATCH):
                s = s + dz_out[b, j]
            db2[j] = s

        # dh_act = dz_out @ W2^T (owned scratch List, no raw alloc)
        var dh_act = List[Scalar[dtype]](
            length=BATCH * Self.hidden_dim, fill=Scalar[dtype](0)
        )
        var dh_act_view = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.hidden_dim), MutAnyOrigin
        ](dh_act)
        try:
            max_matmul[transpose_b=True, target="cpu"](
                lt_to_tt(dh_act_view), lt_to_tt(dz_out), lt_to_tt(W2), None
            )
        except:
            pass

        # dh_pre = dh_act * (1 - h_act²)  (owned scratch List)
        var dh_pre = List[Scalar[dtype]](
            length=BATCH * Self.hidden_dim, fill=Scalar[dtype](0)
        )
        var dh_pre_view = LayoutTensor[
            dtype, Layout.row_major(BATCH, Self.hidden_dim), MutAnyOrigin
        ](dh_pre)
        for b in range(BATCH):
            for j in range(Self.hidden_dim):
                var ha = rebind[Scalar[dtype]](hidden_act[b, j])
                dh_pre_view[b, j] = rebind[Scalar[dtype]](
                    dh_act_view[b, j]
                ) * (Scalar[dtype](1) - ha * ha)

        # dW1 = input^T @ dh_pre
        comptime if CompilationTarget.is_macos() and dtype == DType.float32:
            # Apple Accelerate cblas FFI boundary, kept as-is.
            try:
                var cblas_gemm2 = get_cblas_f32_function()
                cblas_gemm2(
                    _CBLASOrder.ROW_MAJOR,
                    _CBLASTranspose.TRANSPOSE,
                    _CBLASTranspose.NO_TRANSPOSE,
                    Int32(Self.in_dim),
                    Int32(Self.hidden_dim),
                    Int32(BATCH),
                    Float32(1.0),
                    rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
                        enc_input.ptr
                    ),
                    Int32(Self.in_dim),
                    rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
                        dh_pre_view.ptr
                    ),
                    Int32(Self.hidden_dim),
                    Float32(0.0),
                    rebind[UnsafePointer[Float32, MutAnyOrigin]](dW1.ptr),
                    Int32(Self.hidden_dim),
                )
            except:
                pass
        else:
            # Portable: materialize input^T into an owned List, GEMM via lt_to_tt.
            var iT = List[Scalar[dtype]](
                length=Self.in_dim * BATCH, fill=Scalar[dtype](0)
            )
            for bi in range(BATCH):
                for i in range(Self.in_dim):
                    iT[i * BATCH + bi] = rebind[Scalar[dtype]](enc_input[bi, i])
            var iT_view = LayoutTensor[
                dtype, Layout.row_major(Self.in_dim, BATCH), MutAnyOrigin
            ](iT)
            try:
                max_matmul[target="cpu"](
                    lt_to_tt(dW1), lt_to_tt(iT_view), lt_to_tt(dh_pre_view), None
                )
            except:
                pass
            _ = iT^

        # db1 = column-sum(dh_pre) (stays in LayoutTensor element type)
        for j in range(Self.hidden_dim):
            var s = dh_pre_view[0, j]
            for b in range(1, BATCH):
                s = s + dh_pre_view[b, j]
            db1[j] = s

        _ = dh_pre^
        _ = dh_act^
