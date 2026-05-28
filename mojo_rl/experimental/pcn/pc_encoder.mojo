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

from layout import Layout, LayoutTensor, TileTensor, row_major
from std.math import sqrt, tanh
from std.memory import alloc
from std.random.philox import Random as PhiloxRandom
from std.sys import CompilationTarget, simd_width_of
from linalg.matmul import matmul as max_matmul
from linalg.matmul.cpu.apple_accelerate import (
    get_cblas_f32_function,
    _CBLASOrder,
    _CBLASTranspose,
)

comptime _SW = simd_width_of[DType.float32]()


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
        var inp_p = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
            enc_input.ptr
        )
        var hp_p = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
            hidden_pre.ptr
        )
        var ha_p = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
            hidden_act.ptr
        )
        var out_p = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
            output.ptr
        )
        var w1_p = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
            params.ptr + Self.W1_OFFSET
        )
        var b1_p = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
            params.ptr + Self.B1_OFFSET
        )
        var w2_p = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
            params.ptr + Self.W2_OFFSET
        )
        var b2_p = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
            params.ptr + Self.B2_OFFSET
        )

        # h_pre = input @ W1 + b1
        var inp_tt = TileTensor(inp_p, row_major[BATCH, Self.in_dim]())
        var w1_tt = TileTensor(w1_p, row_major[Self.in_dim, Self.hidden_dim]())
        var hp_tt = TileTensor(hp_p, row_major[BATCH, Self.hidden_dim]())
        try:
            max_matmul[target="cpu"](hp_tt, inp_tt, w1_tt, None)
        except:
            pass
        # bias add + tanh
        for b in range(BATCH):
            var off = b * Self.hidden_dim
            var j = 0
            comptime if dtype == DType.float32:
                while j + _SW <= Self.hidden_dim:
                    var v = hp_p.load[width=_SW](off + j) + b1_p.load[width=_SW](j)
                    hp_p.store(off + j, v)
                    j += _SW
            while j < Self.hidden_dim:
                hp_p[off + j] = hp_p[off + j] + b1_p[j]
                j += 1
            for k in range(Self.hidden_dim):
                ha_p[off + k] = Scalar[dtype](tanh(Float64(hp_p[off + k])))

        # output = h_act @ W2 + b2
        var ha_tt = TileTensor(ha_p, row_major[BATCH, Self.hidden_dim]())
        var w2_tt = TileTensor(w2_p, row_major[Self.hidden_dim, Self.out_dim]())
        var out_tt = TileTensor(out_p, row_major[BATCH, Self.out_dim]())
        try:
            max_matmul[target="cpu"](out_tt, ha_tt, w2_tt, None)
        except:
            pass
        for b in range(BATCH):
            var off = b * Self.out_dim
            var j = 0
            comptime if dtype == DType.float32:
                while j + _SW <= Self.out_dim:
                    out_p.store(
                        off + j,
                        out_p.load[width=_SW](off + j) + b2_p.load[width=_SW](j),
                    )
                    j += _SW
            while j < Self.out_dim:
                out_p[off + j] = out_p[off + j] + b2_p[j]
                j += 1

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

        var gp = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](grads.ptr)
        var pp = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](params.ptr)
        var ha_p = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
            hidden_act.ptr
        )
        var dz_p = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
            dz_out.ptr
        )
        var inp_p = rebind[UnsafePointer[Scalar[dtype], MutAnyOrigin]](
            enc_input.ptr
        )
        var dw2_p = gp + Self.W2_OFFSET
        var db2_p = gp + Self.B2_OFFSET
        var dw1_p = gp + Self.W1_OFFSET
        var db1_p = gp + Self.B1_OFFSET

        # dW2 = h_act^T @ dz_out
        comptime if CompilationTarget.is_macos() and dtype == DType.float32:
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
                    rebind[UnsafePointer[Float32, ImmutAnyOrigin]](ha_p),
                    Int32(Self.hidden_dim),
                    rebind[UnsafePointer[Float32, ImmutAnyOrigin]](dz_p),
                    Int32(Self.out_dim),
                    Float32(0.0),
                    rebind[UnsafePointer[Float32, MutAnyOrigin]](dw2_p),
                    Int32(Self.out_dim),
                )
            except:
                pass
        else:
            var cT_buf: UnsafePointer[
                Scalar[dtype], MutAnyOrigin
            ] = alloc[Scalar[dtype]](BATCH * Self.hidden_dim)
            for bi in range(BATCH):
                for i in range(Self.hidden_dim):
                    cT_buf[i * BATCH + bi] = ha_p[bi * Self.hidden_dim + i]
            var cT_tt = TileTensor(cT_buf, row_major[Self.hidden_dim, BATCH]())
            var dz_tt = TileTensor(dz_p, row_major[BATCH, Self.out_dim]())
            var dw2_tt = TileTensor(
                dw2_p, row_major[Self.hidden_dim, Self.out_dim](),
            )
            try:
                max_matmul[target="cpu"](dw2_tt, cT_tt, dz_tt, None)
            except:
                pass
            cT_buf.free()

        # db2 = column-sum(dz_out)
        for j in range(Self.out_dim):
            var s: Scalar[dtype] = 0
            for b in range(BATCH):
                s += dz_p[b * Self.out_dim + j]
            db2_p[j] = s

        # dh_act = dz_out @ W2^T
        var w2_p = pp + Self.W2_OFFSET
        var dh_act_buf: UnsafePointer[
            Scalar[dtype], MutAnyOrigin
        ] = alloc[Scalar[dtype]](BATCH * Self.hidden_dim)
        var dh_tt = TileTensor(
            dh_act_buf, row_major[BATCH, Self.hidden_dim](),
        )
        var dz_tt2 = TileTensor(dz_p, row_major[BATCH, Self.out_dim]())
        var w2_tt = TileTensor(
            w2_p, row_major[Self.hidden_dim, Self.out_dim](),
        )
        try:
            max_matmul[transpose_b=True, target="cpu"](dh_tt, dz_tt2, w2_tt, None)
        except:
            pass

        # dh_pre = dh_act * (1 - h_act²)
        var dh_pre_buf: UnsafePointer[
            Scalar[dtype], MutAnyOrigin
        ] = alloc[Scalar[dtype]](BATCH * Self.hidden_dim)
        for b in range(BATCH):
            for j in range(Self.hidden_dim):
                var idx = b * Self.hidden_dim + j
                var ha = ha_p[idx]
                dh_pre_buf[idx] = dh_act_buf[idx] * (
                    Scalar[dtype](1) - ha * ha
                )

        # dW1 = input^T @ dh_pre
        comptime if CompilationTarget.is_macos() and dtype == DType.float32:
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
                    rebind[UnsafePointer[Float32, ImmutAnyOrigin]](inp_p),
                    Int32(Self.in_dim),
                    rebind[UnsafePointer[Float32, ImmutAnyOrigin]](dh_pre_buf),
                    Int32(Self.hidden_dim),
                    Float32(0.0),
                    rebind[UnsafePointer[Float32, MutAnyOrigin]](dw1_p),
                    Int32(Self.hidden_dim),
                )
            except:
                pass
        else:
            var iT_buf: UnsafePointer[
                Scalar[dtype], MutAnyOrigin
            ] = alloc[Scalar[dtype]](BATCH * Self.in_dim)
            for bi in range(BATCH):
                for i in range(Self.in_dim):
                    iT_buf[i * BATCH + bi] = inp_p[bi * Self.in_dim + i]
            var iT_tt = TileTensor(iT_buf, row_major[Self.in_dim, BATCH]())
            var dp_tt = TileTensor(
                dh_pre_buf, row_major[BATCH, Self.hidden_dim](),
            )
            var dw1_tt = TileTensor(
                dw1_p, row_major[Self.in_dim, Self.hidden_dim](),
            )
            try:
                max_matmul[target="cpu"](dw1_tt, iT_tt, dp_tt, None)
            except:
                pass
            iT_buf.free()

        # db1 = column-sum(dh_pre)
        for j in range(Self.hidden_dim):
            var s: Scalar[dtype] = 0
            for b in range(BATCH):
                s += dh_pre_buf[b * Self.hidden_dim + j]
            db1_p[j] = s

        dh_pre_buf.free()
        dh_act_buf.free()
