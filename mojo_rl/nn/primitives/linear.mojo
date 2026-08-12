"""Linear — Module conformer (CPU + GPU). y = x @ W + b via max_matmul.

Each forward/vjp branches `comptime if target == "cpu"` (tracked `TileTensor`
over `.data`) `else` (device `LayoutTensor` via `lt_gpu` + a naive kernel).
The storage surface (`ref/mut Tensor`, `TensorRefs`) is identical on both
targets; the only GPU erasure is the kernel-arg `MutAnyOrigin`. Params are
`Param` (two `Tensor`s, cpu+dev).

bf16-FLOW (AMP "Step B"): `Linear[IN, OUT]` is fp32 (unchanged), while
`Linear[IN, OUT, DType.bfloat16]` is a bf16-flow linear whose ACTIVATIONS are
STORED and FLOW at bf16 (`ACT_DT == bfloat16`) — there is NO per-call x/out cast
(the legacy "cast-around" AMP tax is gone). Master weights/grads/bias STAY fp32
(`Param` is always `DT`); only a CACHED bf16 weight copy (`w_bf`, version-gated)
and a cached bf16 bias (`b_a`) are low-precision. The fp32 (ACT_DT == DT) path is
byte-for-byte the legacy NoAMP path; the bf16 path is GPU-only (cblas/CPU matmul
is fp32-only).

LIFETIME NOTE: a pack subscript (`inputs[k]`) returns a TEMPORARY ref. Building
a view from `inputs[k].data` directly and using it LATER dangles (the temporary
dies at the end of the statement; a later op clobbers the stack). So each body
first binds the element to a named `ref` (`ref in0 = inputs[0]`) that lives for
the whole function, then builds views from that.
"""

from std.sys import CompilationTarget
from std.gpu import global_idx, thread_idx, block_idx
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, TileTensor, row_major
from linalg.matmul import matmul as max_matmul
from linalg.matmul.cpu.apple_accelerate import (
    get_cblas_f32_function,
    _CBLASOrder,
    _CBLASTranspose,
)

from mojo_rl.nn.constants import DT
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from ..core.polyak import polyak_tensor


# ── kernels (non-GEMM ops; the three matmuls go through max_matmul) ──────
# `_bias_add_kernel` is dtype-parametric: the fp32 path calls it with DT, the
# bf16-flow path with bfloat16 (activation + cached bf16 bias both at ADT).
def _bias_add_kernel[
    B: Int, OUT: Int, ADT: DType = DT
](
    o: LayoutTensor[ADT, Layout.row_major(B, OUT), MutAnyOrigin],
    bias: LayoutTensor[ADT, Layout.row_major(OUT), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < B * OUT:
        o[idx // OUT, idx % OUT] += bias[idx % OUT]


# Naive grad_w transpose (one thread/elem, strided read). Still used by
# linear_relu / noisy_linear; linear + linear_act use the tiled B1' kernel below.
def _transpose_kernel[
    ROWS: Int, COLS: Int
](
    src: LayoutTensor[DT, Layout.row_major(ROWS, COLS), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(COLS, ROWS), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < ROWS * COLS:
        dst[idx % COLS, idx // COLS] = src[idx // COLS, idx % COLS]


comptime _T_TILE = 32
comptime _T_BR = 8        # 32x8 BLOCK_ROWS, 4 elems/thread (B1')


# Tiled grad_w transpose: 32x8 BLOCK_ROWS shared-mem tile (B1'). Coalesces the
# strided src read the naive kernel suffers; bench (RTX 5090) showed up to 1.65x
# on skinny grad_w (large ROWS·COLS), neutral on square. Launch with
# grid_dim=(ceildiv(COLS,32), ceildiv(ROWS,32)), block_dim=(32, 8).
# Dtype-parametric (`ADT`): the fp32 path transposes a DT activation; the bf16
# path transposes the bf16 forward-input directly (its source is already bf16).
def _transpose_tiled_kernel[
    ROWS: Int, COLS: Int, ADT: DType = DT
](
    src: LayoutTensor[ADT, Layout.row_major(ROWS, COLS), MutAnyOrigin],
    dst: LayoutTensor[ADT, Layout.row_major(COLS, ROWS), MutAnyOrigin],
):
    var tile = LayoutTensor[
        ADT,
        Layout.row_major(_T_TILE, _T_TILE + 1),   # +1 pad → no bank conflicts
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var cy = Int(block_idx.y) * _T_TILE       # tile origin in ROWS
    var cx = Int(block_idx.x) * _T_TILE       # tile origin in COLS
    var tx = Int(thread_idx.x)                # [0, _T_TILE)
    var ty = Int(thread_idx.y)                # [0, _T_BR)

    var c = cx + tx
    comptime for r in range(0, _T_TILE, _T_BR):
        var rr = cy + ty + r
        if rr < ROWS and c < COLS:
            tile[ty + r, tx] = rebind[Scalar[ADT]](src[rr, c])
    barrier()

    var r2 = cy + tx                          # dst col (coalesced, stride 1)
    comptime for r in range(0, _T_TILE, _T_BR):
        var c2 = cx + ty + r                  # dst row
        if r2 < ROWS and c2 < COLS:
            dst[c2, r2] = rebind[Scalar[ADT]](tile[tx, ty + r])


def _accum_kernel[
    N: Int
](
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        dst[i] += src[i]


# grad_b += colsum(go). Dtype-parametric on the grad_output activation (`ADT`):
# the bf16 path reads a bf16 `go` and accumulates into the FP32 `gb` (each
# element cast to DT before summing — the accumulator stays fp32).
def _lin_gb_kernel[
    B: Int, OUT: Int, ADT: DType = DT
](
    go: LayoutTensor[ADT, Layout.row_major(B, OUT), MutAnyOrigin],
    gb: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
):
    var j = Int(global_idx.x)
    if j < OUT:
        var s: Scalar[DT] = 0
        for b in range(B):
            s += rebind[Scalar[ADT]](go[b, j]).cast[DT]()
        gb[j] += s


comptime BF16 = DType.bfloat16


def _cast_f2b_kernel[
    N: Int
](
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    dst: LayoutTensor[BF16, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = src[i].cast[BF16]()


def _cast_b2f_kernel[
    N: Int
](
    src: LayoutTensor[BF16, Layout.row_major(N), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = src[i].cast[DT]()


def _pad_cols_kernel[
    ROWS: Int, SRC_COLS: Int, DST_COLS: Int
](
    src: LayoutTensor[DT, Layout.row_major(ROWS * SRC_COLS), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(ROWS * DST_COLS), MutAnyOrigin],
):
    """`dst[r, :SRC_COLS] = src[r]`, `dst[r, SRC_COLS:] = 0`.

    Serves BOTH K-alignment pads: the activation ([B, IN_] -> [B, K_PAD], a
    real per-row widening) and the weight ([IN_, OUT_] -> [K_PAD, OUT_], which
    is row-appending and so is the ROWS=1 flat case).
    """
    var i = Int(global_idx.x)
    if i < ROWS * DST_COLS:
        var r = i // DST_COLS
        var c = i % DST_COLS
        if c < SRC_COLS:
            dst[i] = src[r * SRC_COLS + c]
        else:
            dst[i] = Scalar[DT](0)


# ── Linear ─────────────────────────────────────────────────────────────
struct Linear[IN_: Int, OUT_: Int, ADT: DType = DT](Module):
    comptime ARITY = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.IN_)
    comptime OUT_DIM = Self.OUT_
    comptime W_SIZE = Self.IN_ * Self.OUT_

    # ── K-alignment padding (GPU fp32 forward) ───────────────────────────
    # `max_matmul` falls off a ~10x cliff when the CONTRACTION dim is
    # misaligned, on BOTH backends — Metal wants K%16, an RTX 5090 wants K%32,
    # so 32 satisfies both (it costs Metal ~4% against its own optimum and is
    # worth 7-24x on the 5090). Measured by
    # `benchmarks/bench_matmul_k_alignment.mojo`; run it before changing PAD_TO.
    #
    #     [268 x 518] @ [518 x 512]   1356 us (Metal)    206 us (5090)
    #     [268 x 544] @ [544 x 512]    150 us            27 us
    #
    # `IN_` is a concatenation width in most of the repo — TD-MPC2's
    # `ZA = LATENT + ACT` = 518, a SAC critic's `obs|act`, a two-hot `BINS`
    # vector — so it lands on an arbitrary K constantly.
    #
    # ⚠ The padding is INTERNAL. `IN_` and the checkpointed `weight` Param stay
    # at their logical size, so this changes no on-disk format and no existing
    # checkpoint. The alternative (widening the caller's concat) would have
    # changed every TD-MPC2 net shape and invalidated every checkpoint.
    comptime PAD_TO = 32
    comptime NEEDS_PAD = Self.IN_ % Self.PAD_TO != 0
    comptime K_PAD = (
        (Self.IN_ + Self.PAD_TO - 1) // Self.PAD_TO
    ) * Self.PAD_TO
    comptime WPAD_SIZE = Self.K_PAD * Self.OUT_
    # Activation-flow dtype (satisfies the Module trait). `Linear[IN, OUT]` =
    # fp32 (ACT_DT == DT, the legacy path); `Linear[IN, OUT, bfloat16]` flows
    # activations at bf16 (the AMP "Step B" memory win).
    comptime ACT_DT = Self.ADT

    @staticmethod
    def display_label() -> String:
        return String("Linear")
    comptime B_SIZE = Self.OUT_

    var weight: Param["weight", True, Self.W_SIZE]
    var bias: Param["bias", False, Self.B_SIZE]
    # grad_w scratch (lazy): cacheᵀ [IN, B] + dW_tmp [IN, OUT] for the
    # transpose + max_matmul + accumulate path (max_matmul rejects transpose_a).
    # Both STAY fp32 (master grad is fp32; the bf16 dW GEMM writes a fp32 output).
    var cacheT: Tensor
    var dW_tmp: Tensor
    # bf16-flow compute scratch (lazy; used only when ACT_DT == bf16 and
    # target == "gpu"). Master weights/grads/bias stay fp32 (`Param`); only these
    # low-precision copies are bf16. `w_bf` is the CACHED bf16 weight: recast from
    # `weight.val` only when the optimizer bumped `weight.val.version` since the
    # last cast (tracked by `_w_cast_version`), so the W cast happens ONCE per
    # optimizer step rather than every forward. `b_a` is the cached bf16 bias (the
    # tiny per-forward bias cast). `cacheT_bf` is the transposed bf16 fwd-input
    # (backward grad_w). No `x_bf`/`o_bf`/`go_bf`: activations ALREADY flow at bf16
    # (no input/output/grad_output cast — the whole point of bf16-flow).
    var w_bf: TensorImpl[Self.ADT]
    var b_a: TensorImpl[Self.ADT]        # cached bf16 bias (forward bias-add)
    var cacheT_bf: TensorImpl[Self.ADT]  # transposed-x bf16 (backward grad_w)
    var _w_cast_version: Int  # `weight.val.version` at last bf16 weight cast
    # K-alignment scratch (lazy; GPU fp32 forward only, and only when
    # `NEEDS_PAD`). `w_pad` is the zero-tailed [K_PAD, OUT_] weight, re-padded
    # only when the optimizer bumped `weight.val.version` — once per optimizer
    # step, not per forward, exactly like `w_bf` above. `x_pad` is the
    # zero-tailed [B, K_PAD] activation, which DOES have to be rebuilt every
    # forward because the input is a fresh upstream tensor each time.
    var w_pad: Tensor
    var x_pad: Tensor
    var _w_pad_version: Int
    # Capture mode (set via `set_attr["capture_recast"]`): when True, the bf16
    # weight recast in `_ensure_w_bf` is UNCONDITIONAL so the cast kernel is
    # always recorded into a CUDA graph and reads the live fp32 master on every
    # replay — the version gate would skip it on replay and serve STALE weights.
    # Off → the version-gated fast path (one cast per optimizer step).
    var _force_recast: Bool

    def __init__(out self):
        self.weight = Param["weight", True, Self.W_SIZE]()
        self.bias = Param["bias", False, Self.B_SIZE]()
        self.cacheT = Tensor()
        self.dW_tmp = Tensor()
        self.w_bf = TensorImpl[Self.ADT]()
        self.b_a = TensorImpl[Self.ADT]()
        self.cacheT_bf = TensorImpl[Self.ADT]()
        self._w_cast_version = -1  # < any real version → first forward casts
        self._force_recast = False
        self.w_pad = Tensor()
        self.x_pad = Tensor()
        self._w_pad_version = -1  # < any real version → first forward pads

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        comptime if ATTR == "capture_recast":
            self._force_recast = value != Scalar[DT](0.0)

    def _ensure_w_pad(mut self, c: DeviceContext) raises:
        """Ensure `w_pad` is `weight.val` with `K_PAD - IN_` rows of zeros
        appended. Re-pads ONLY when the optimizer bumped `val.version` since the
        last pad, so training pays it once per step and inference once ever.

        ⚠ `_force_recast` (CUDA-graph capture) must make this UNCONDITIONAL for
        the same reason it does for the bf16 cast: the version gate would skip
        the pad on replay and the GEMM would read a STALE weight.
        """
        self.w_pad.ensure_gpu(c, Self.WPAD_SIZE)
        if self._force_recast or self.weight.val.version != self._w_pad_version:
            # The weight is [IN_, OUT_] row-major, so padding the CONTRACTION
            # dim appends whole rows — a flat copy with a zero tail (ROWS=1).
            c.enqueue_function[
                _pad_cols_kernel[1, Self.W_SIZE, Self.WPAD_SIZE]
            ](
                self.weight.val.lt["gpu", Layout.row_major(Self.W_SIZE)](),
                self.w_pad.lt["gpu", Layout.row_major(Self.WPAD_SIZE)](),
                grid_dim=(Self.WPAD_SIZE + 255) // 256,
                block_dim=256,
            )
            self._w_pad_version = self.weight.val.version

    def _ensure_w_bf(mut self, c: DeviceContext) raises:
        """Ensure the cached bf16 weight `w_bf` reflects the current fp32
        `weight.val`. Recasts ONLY when the optimizer bumped `val.version` since
        the last cast (so the weight cast is ONCE per step, not per fwd/bwd).
        Shared by forward (the cast) and vjp (which REUSES it — no optimizer step
        intervenes between a fwd and its bwd, so the forward's cast is valid)."""
        self.w_bf.ensure_gpu(c, Self.W_SIZE)
        if self._force_recast or self.weight.val.version != self._w_cast_version:
            c.enqueue_function[_cast_f2b_kernel[Self.W_SIZE]](
                self.weight.val.lt["gpu", Layout.row_major(Self.W_SIZE)](),
                self.w_bf.lt["gpu", Layout.row_major(Self.W_SIZE)](),
                grid_dim=(Self.W_SIZE + 255) // 256,
                block_dim=256,
            )
            self._w_cast_version = self.weight.val.version

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var l = Self()
        l.weight = Param["weight", True, Self.W_SIZE].make[target](ctx)
        l.bias = Param["bias", False, Self.B_SIZE].make[target](ctx)
        INIT.init_weight[target](
            l.weight.val, Self.W_SIZE, Self.IN_, Self.OUT_, ctx
        )
        INIT.init_bias[target](l.bias.val, Self.B_SIZE, ctx)
        return l^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        # y = x @ W (max_matmul), then += bias.
        comptime if Self.ACT_DT == DT:
            # ── fp32 path (legacy NoAMP, byte-identical) ──
            # ACT_DT IS DT in this branch, but the compiler doesn't collapse the
            # opaque `Self.ACT_DT` parameter to `DT` for type-unification against
            # the fp32 weight/bias views — so rebind the activation refs (sound:
            # the dtypes are equal here). `TensorImpl[Self.ACT_DT]` ≡ `Tensor`.
            ref in0d = rebind[Tensor](in0)
            ref outd = rebind[Tensor](out)
            comptime if target == "cpu":
                outd.ensure(B * Self.OUT_)
                var x_v = TileTensor(in0d.data, row_major[B, Self.IN_]())
                var w_v = TileTensor(
                    self.weight.val.data, row_major[Self.IN_, Self.OUT_]()
                )
                var out_v = TileTensor(outd.data, row_major[B, Self.OUT_]())
                max_matmul[target="cpu"](out_v, x_v, w_v, None)
                var bt = TileTensor(self.bias.val.data, row_major[Self.OUT_]())
                for b in range(B):
                    for j in range(Self.OUT_):
                        out_v[b, j] += bt[j]
            else:
                var c = ctx.value()
                outd.ensure_gpu(c, B * Self.OUT_)
                var out_v = TileTensor(
                    outd.dev.value(), row_major[B, Self.OUT_]()
                )
                comptime if Self.NEEDS_PAD:
                    # Zero-pad the contraction dim to `K_PAD` — see the
                    # `PAD_TO` comment on the struct. The appended columns are
                    # exactly 0, so the dot products are unchanged; only the
                    # GEMM's tiling (and hence its fp32 reduction ORDER) moves,
                    # which can shift results by an ulp.
                    self._ensure_w_pad(c)
                    self.x_pad.ensure_gpu(c, B * Self.K_PAD)
                    c.enqueue_function[
                        _pad_cols_kernel[B, Self.IN_, Self.K_PAD]
                    ](
                        in0d.lt["gpu", Layout.row_major(B * Self.IN_)](),
                        self.x_pad.lt["gpu", Layout.row_major(B * Self.K_PAD)](),
                        grid_dim=(B * Self.K_PAD + 255) // 256,
                        block_dim=256,
                    )
                    var xp_v = TileTensor(
                        self.x_pad.dev.value(), row_major[B, Self.K_PAD]()
                    )
                    var wp_v = TileTensor(
                        self.w_pad.dev.value(),
                        row_major[Self.K_PAD, Self.OUT_](),
                    )
                    max_matmul[target="gpu"](out_v, xp_v, wp_v, c)
                else:
                    var x_v = TileTensor(
                        in0d.dev.value(), row_major[B, Self.IN_]()
                    )
                    var w_v = TileTensor(
                        self.weight.val.dev.value(),
                        row_major[Self.IN_, Self.OUT_](),
                    )
                    max_matmul[target="gpu"](out_v, x_v, w_v, c)
                var ol = outd.lt["gpu", Layout.row_major(B, Self.OUT_)]()
                var bl = self.bias.val.lt[
                    "gpu", Layout.row_major(Self.OUT_)
                ]()
                c.enqueue_function[_bias_add_kernel[B, Self.OUT_]](
                    ol, bl, grid_dim=(B * Self.OUT_ + 255) // 256, block_dim=256
                )
        else:
            # ── bf16-flow path (GPU-only) ──
            comptime assert (
                target == "gpu"
            ), "bf16-flow Linear is GPU-only"
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_)
            # x (in0) is ALREADY bf16 — no input cast. W: cached bf16 (recast
            # only on a version bump). bias: cheap per-forward DT→bf16 cast.
            self._ensure_w_bf(c)
            self.b_a.ensure_gpu(c, Self.B_SIZE)
            c.enqueue_function[_cast_f2b_kernel[Self.B_SIZE]](
                self.bias.val.lt["gpu", Layout.row_major(Self.B_SIZE)](),
                self.b_a.lt["gpu", Layout.row_major(Self.B_SIZE)](),
                grid_dim=(Self.B_SIZE + 255) // 256,
                block_dim=256,
            )
            var x_v = TileTensor(in0.dev.value(), row_major[B, Self.IN_]())
            var w_bf_v = TileTensor(
                self.w_bf.dev.value(), row_major[Self.IN_, Self.OUT_]()
            )
            var out_v = TileTensor(out.dev.value(), row_major[B, Self.OUT_]())
            # bf16-in → bf16-out GEMM (fp32 accumulation is automatic).
            max_matmul[target="gpu"](out_v, x_v, w_bf_v, c)
            var ol = out.lt["gpu", Layout.row_major(B, Self.OUT_)]()
            var bl = self.b_a.lt["gpu", Layout.row_major(Self.OUT_)]()
            c.enqueue_function[_bias_add_kernel[B, Self.OUT_, Self.ADT]](
                ol, bl, grid_dim=(B * Self.OUT_ + 255) // 256, block_dim=256
            )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi, Self.ACT_DT],
        mut grad_output: TensorImpl[Self.ACT_DT],
        grad_inputs: TensorRefs[1, ogi, Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref fin = forward_input[0]
        ref gin = grad_inputs[0]
        comptime if Self.ACT_DT == DT:
            # ── fp32 path (legacy NoAMP, byte-identical) ──
            # ACT_DT IS DT here — rebind the activation refs (sound; see forward).
            ref find = rebind[Tensor](fin)
            ref gind = rebind[Tensor](gin)
            ref god = rebind[Tensor](grad_output)
            comptime if target == "cpu":
                gind.ensure(B * Self.IN_)
                self.cacheT.ensure(Self.IN_ * B)
                self.dW_tmp.ensure(Self.W_SIZE)
                var x_v = TileTensor(find.data, row_major[B, Self.IN_]())
                var go_v = TileTensor(
                    god.data, row_major[B, Self.OUT_]()
                )
                var gi_v = TileTensor(gind.data, row_major[B, Self.IN_]())
                var w_v = TileTensor(
                    self.weight.val.data, row_major[Self.IN_, Self.OUT_]()
                )
                var gw_v = TileTensor(
                    self.weight.grd.data, row_major[Self.IN_, Self.OUT_]()
                )
                var gb_v = TileTensor(
                    self.bias.grd.data, row_major[Self.OUT_]()
                )
                # grad_b += colsum(go)
                for b in range(B):
                    for j in range(Self.OUT_):
                        gb_v[j] += go_v[b, j]
                # grad_w += xᵀ @ go. Apple-fp32: ONE fused cblas_sgemm (TRANSPOSE
                # A=x, beta=1 → no transpose buffer, no temp, no accumulate loop —
                # matches legacy Linear). Else: transpose into cacheT + max_matmul
                # + add.
                comptime IS_APPLE_F32 = (
                    CompilationTarget.is_macos() and DT == DType.float32
                )
                comptime if IS_APPLE_F32:
                    var cblas = get_cblas_f32_function()
                    cblas(
                        _CBLASOrder.ROW_MAJOR,
                        _CBLASTranspose.TRANSPOSE,
                        _CBLASTranspose.NO_TRANSPOSE,
                        Int32(Self.IN_),
                        Int32(Self.OUT_),
                        Int32(B),
                        Float32(1.0),
                        rebind[Pointer[Float32, ImmutAnyOrigin]](
                            find.data.unsafe_ptr()
                        ),
                        Int32(Self.IN_),
                        rebind[Pointer[Float32, ImmutAnyOrigin]](
                            god.data.unsafe_ptr()
                        ),
                        Int32(Self.OUT_),
                        Float32(1.0),
                        rebind[Pointer[Float32, MutAnyOrigin]](
                            self.weight.grd.data.unsafe_ptr()
                        ),
                        Int32(Self.OUT_),
                    )
                else:
                    var cT_v = TileTensor(
                        self.cacheT.data, row_major[Self.IN_, B]()
                    )
                    for b in range(B):
                        for k in range(Self.IN_):
                            cT_v[k, b] = x_v[b, k]
                    var dW_v = TileTensor(
                        self.dW_tmp.data, row_major[Self.IN_, Self.OUT_]()
                    )
                    max_matmul[target="cpu"](dW_v, cT_v, go_v, None)
                    for k in range(Self.IN_):
                        for j in range(Self.OUT_):
                            gw_v[k, j] += dW_v[k, j]
                # grad_input = go @ Wᵀ
                max_matmul[transpose_b=True, target="cpu"](
                    gi_v, go_v, w_v, None
                )
            else:
                var c = ctx.value()
                gind.ensure_gpu(c, B * Self.IN_)
                self.cacheT.ensure_gpu(c, Self.IN_ * B)
                self.dW_tmp.ensure_gpu(c, Self.W_SIZE)
                # grad_b += colsum(go)
                var gol = god.lt[
                    "gpu", Layout.row_major(B, Self.OUT_)
                ]()
                var gbl = self.bias.grd.lt[
                    "gpu", Layout.row_major(Self.OUT_)
                ]()
                c.enqueue_function[_lin_gb_kernel[B, Self.OUT_]](
                    gol, gbl, grid_dim=(Self.OUT_ + 255) // 256, block_dim=256
                )
                # grad_w += cacheᵀ @ go: transpose x → cacheT (B1' tiled, fp32),
                # then the two GEMMs (grad_w + grad_x), then the grad_w accumulate.
                var xl = find.lt["gpu", Layout.row_major(B, Self.IN_)]()
                var cTl = self.cacheT.lt[
                    "gpu", Layout.row_major(Self.IN_, B)
                ]()
                c.enqueue_function[_transpose_tiled_kernel[B, Self.IN_]](
                    xl,
                    cTl,
                    grid_dim=(
                        (Self.IN_ + _T_TILE - 1) // _T_TILE,
                        (B + _T_TILE - 1) // _T_TILE,
                    ),
                    block_dim=(_T_TILE, _T_BR),
                )
                var dW_v = TileTensor(
                    self.dW_tmp.dev.value(), row_major[Self.IN_, Self.OUT_]()
                )
                var gi_v = TileTensor(
                    gind.dev.value(), row_major[B, Self.IN_]()
                )
                var cT_v = TileTensor(
                    self.cacheT.dev.value(), row_major[Self.IN_, B]()
                )
                var go_v = TileTensor(
                    god.dev.value(), row_major[B, Self.OUT_]()
                )
                max_matmul[target="gpu"](dW_v, cT_v, go_v, c)
                var w_v = TileTensor(
                    self.weight.val.dev.value(),
                    row_major[Self.IN_, Self.OUT_](),
                )
                max_matmul[transpose_b=True, target="gpu"](gi_v, go_v, w_v, c)
                # grad_w += dW (accumulate into the fp32 master grad)
                var gwl = self.weight.grd.lt[
                    "gpu", Layout.row_major(Self.W_SIZE)
                ]()
                var dWl = self.dW_tmp.lt[
                    "gpu", Layout.row_major(Self.W_SIZE)
                ]()
                c.enqueue_function[_accum_kernel[Self.W_SIZE]](
                    gwl, dWl, grid_dim=(Self.W_SIZE + 255) // 256, block_dim=256
                )
        else:
            # ── bf16-flow path (GPU-only) ──
            comptime assert (
                target == "gpu"
            ), "bf16-flow Linear is GPU-only"
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.IN_)
            self.cacheT_bf.ensure_gpu(c, Self.IN_ * B)
            self.dW_tmp.ensure_gpu(c, Self.W_SIZE)
            # grad_b += colsum(go): bf16 go → fp32 master grad (fp32 accumulator).
            var gol = grad_output.lt["gpu", Layout.row_major(B, Self.OUT_)]()
            var gbl = self.bias.grd.lt["gpu", Layout.row_major(Self.OUT_)]()
            c.enqueue_function[_lin_gb_kernel[B, Self.OUT_, Self.ADT]](
                gol, gbl, grid_dim=(Self.OUT_ + 255) // 256, block_dim=256
            )
            # grad_w += cacheᵀ @ go. `fin`/`go` are ALREADY bf16 (no cast).
            # Transpose the bf16 fwd-input directly → bf16 cacheT_bf, then a
            # bf16-in → FP32-out GEMM into the fp32 dW_tmp, then accumulate into
            # the fp32 master grad. W reuses the forward's cached cast.
            self._ensure_w_bf(c)
            var xl = fin.lt["gpu", Layout.row_major(B, Self.IN_)]()
            var cTl = self.cacheT_bf.lt["gpu", Layout.row_major(Self.IN_, B)]()
            c.enqueue_function[_transpose_tiled_kernel[B, Self.IN_, Self.ADT]](
                xl,
                cTl,
                grid_dim=(
                    (Self.IN_ + _T_TILE - 1) // _T_TILE,
                    (B + _T_TILE - 1) // _T_TILE,
                ),
                block_dim=(_T_TILE, _T_BR),
            )
            var dW_v = TileTensor(
                self.dW_tmp.dev.value(), row_major[Self.IN_, Self.OUT_]()
            )
            var gi_v = TileTensor(gin.dev.value(), row_major[B, Self.IN_]())
            var cTb_v = TileTensor(
                self.cacheT_bf.dev.value(), row_major[Self.IN_, B]()
            )
            var gob_v = TileTensor(
                grad_output.dev.value(), row_major[B, Self.OUT_]()
            )
            var wb_v = TileTensor(
                self.w_bf.dev.value(), row_major[Self.IN_, Self.OUT_]()
            )
            # grad_w = cacheT_bfᵀ-form @ go → fp32 dW (bf16-in, fp32-out).
            max_matmul[target="gpu"](dW_v, cTb_v, gob_v, c)
            # grad_x = go @ Wᵀ → bf16 gin (bf16-in, bf16-out — gin flows at bf16).
            max_matmul[transpose_b=True, target="gpu"](gi_v, gob_v, wb_v, c)
            # grad_w += dW (accumulate into the fp32 master grad)
            var gwl = self.weight.grd.lt[
                "gpu", Layout.row_major(Self.W_SIZE)
            ]()
            var dWl = self.dW_tmp.lt["gpu", Layout.row_major(Self.W_SIZE)]()
            c.enqueue_function[_accum_kernel[Self.W_SIZE]](
                gwl, dWl, grid_dim=(Self.W_SIZE + 255) // 256, block_dim=256
            )

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (core/walkers.mojo auto-discovers the `weight` + `bias` Params).

    def polyak_from[
        target: StaticString
    ](
        mut self,
        mut src: Self,
        tau: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        polyak_tensor[target, Self.W_SIZE](
            self.weight.val, src.weight.val, tau, ctx
        )
        polyak_tensor[target, Self.B_SIZE](
            self.bias.val, src.bias.val, tau, ctx
        )
        # ⚠ `polyak_tensor` writes `weight.val` IN PLACE and does NOT bump
        # `val.version` — so both derived weight caches below would keep
        # serving the PRE-SYNC weight forever. That is invisible in a forward
        # numerics test (which never syncs) and shows up only as a target
        # network frozen at its init weights: `tests/deep_agents/
        # test_storage_dqn_gpu_smoke.mojo` went from eval 200 to eval 9.
        # Invalidate both caches so the next forward rebuilds them.
        self._w_pad_version = -1
        self._w_cast_version = -1
