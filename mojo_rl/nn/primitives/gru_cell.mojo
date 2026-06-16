"""GRUCell[IN_DIM, HIDDEN_DIM] — PyTorch-equivalent GRU cell as
BinaryModule (input, hidden) → new_hidden.

Math (PyTorch convention):
  r = σ(x · W_ir + b_ir + h · W_hr + b_hr)         reset gate
  z = σ(x · W_iz + b_iz + h · W_hz + b_hz)         update gate
  n = tanh(x · W_in + b_in + r ⊙ (h · W_hn + b_hn)) new candidate
  h' = (1 − z) ⊙ n + z ⊙ h

Storage convention (row-major):
  W_ih [IN, 3·H]  — columns 0..H = r, H..2H = z, 2H..3H = n
  W_hh [H,  3·H]  — same column split
  b_ih [3·H]
  b_hh [3·H]

Caches (BATCH-sized, allocated lazily):
  r [B, H], z [B, H], n [B, H]                    activations
  hn_pre [B, H]                                   W_hn·h + b_hn (pre-r-gate)

Backward (BinaryModule, mode-aware):
  Given dh' = grad_output (shape [B, H]):
    dz       = dh' ⊙ (h − n)
    dn       = dh' ⊙ (1 − z)
    d_pre_n  = dn ⊙ (1 − n²)
    d_in_n   = d_pre_n
    dr_x_hn  = d_pre_n              # the `r·hn` summand
    dr       = dr_x_hn ⊙ hn_pre
    d_hn     = dr_x_hn ⊙ r          # gradient on hn_pre (pre-r-gate)
    d_pre_r  = dr ⊙ r ⊙ (1 − r)
    d_pre_z  = dz ⊙ z ⊙ (1 − z)
    d_ir = d_pre_r,  d_hr = d_pre_r
    d_iz = d_pre_z,  d_hz = d_pre_z
    d_in = d_in_n              # only the input-projection part of n's pre-act
    # hn already accounts for the r-gate

  Param grads (mode == "all"):
    d_W_ih [IN, 3H] += x^T · [d_ir | d_iz | d_in]
    d_b_ih [3H]     += sum_B [d_ir | d_iz | d_in]
    d_W_hh [H,  3H] += h^T · [d_hr | d_hz | d_hn]
    d_b_hh [3H]     += sum_B [d_hr | d_hz | d_hn]

  Input grads:
    d_x = [d_ir | d_iz | d_in] · W_ih^T
    d_h = [d_hr | d_hz | d_hn] · W_hh^T + dh' ⊙ z   # last term: direct h
                                                       path through `z·h`

GPU path mirrors `LSTMCell`'s kernel layout (separate W_ih / W_hh / b
Param buffers). Forward is one block per sample (threads stride over
HIDDEN) writing a packed `[r | z | n | hn_pre]` cache (4·H wide).
Backward respects the nn param-grad-before-grad-input ordering so it is
alias-safe as a `Module`: a gate kernel fills the per-gate pre-activation
grads `d_comb` (x-side [0,3H) | h-side [3H,6H)); then dW_ih / dW_hh / db
kernels read x / h_prev; then a final kernel writes dx / dh (which may
alias x / h_prev) last. CPU and GPU are bit-parity-validated by
`tests/nn/test_gru_cell_gpu_parity.mojo`.

There is still no GPU *consumer* (DreamerV3 GPU world-model uses bespoke
fused RSSM ops, not this generic cell) — the GPU path exists for
CPU/GPU parity completeness, exercised by the parity test.
"""

from std.math import exp, tanh
from std.memory import alloc
from linalg.matmul import matmul as max_matmul
from std.gpu import thread_idx, block_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, TPB
from ..core import (
    Initializer,
    AMPPolicy,
    NoAMP,
    Param,
    Cache,
    ParamVisitor,
)
from ..core.module import Module, typed_view, typed_view_mut, mptr
from ..core.tensor_pack import TensorPack
from ..core.target_storage import require_ctx, TargetStorage, assert_tag_for, ensure_cpu_buffer


@always_inline
def _sigmoid(x: Scalar[DT]) -> Scalar[DT]:
    if x >= 0:
        var e = exp(-x)
        return Scalar[DT](1.0) / (Scalar[DT](1.0) + e)
    var e = exp(x)
    return e / (Scalar[DT](1.0) + e)


# ──────────────────────────────────────────────────────────────────────
# GPU kernels (separate W_ih / W_hh / b_ih / b_hh buffers — Param layout).
#   cache  [B, 4H] = [r | z | n | hn_pre]
#   d_comb [B, 6H] = x-side [d_ir | d_iz | d_in]  (cols 0..3H)
#                  | h-side [d_hr | d_hz | d_hn]  (cols 3H..6H)
# ──────────────────────────────────────────────────────────────────────


def _gru_fwd_kernel[
    BATCH: Int, IN_: Int, H: Int,
](
    x: LayoutTensor[DT, Layout.row_major(BATCH, IN_), MutAnyOrigin],
    W_ih: LayoutTensor[DT, Layout.row_major(IN_, 3 * H), MutAnyOrigin],
    W_hh: LayoutTensor[DT, Layout.row_major(H, 3 * H), MutAnyOrigin],
    b_ih: LayoutTensor[DT, Layout.row_major(3 * H), MutAnyOrigin],
    b_hh: LayoutTensor[DT, Layout.row_major(3 * H), MutAnyOrigin],
    h_prev: LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin],
    out_buf: LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, 4 * H), MutAnyOrigin],
):
    """One block per sample; threads stride over j ∈ [0, H)."""
    var bi = Int(block_idx.x)
    if bi >= BATCH:
        return
    var j = Int(thread_idx.x)
    while j < H:
        var ir = rebind[Scalar[DT]](b_ih[j])
        var iz = rebind[Scalar[DT]](b_ih[H + j])
        var in_pre = rebind[Scalar[DT]](b_ih[2 * H + j])
        var hr = rebind[Scalar[DT]](b_hh[j])
        var hz = rebind[Scalar[DT]](b_hh[H + j])
        var hn = rebind[Scalar[DT]](b_hh[2 * H + j])
        for k in range(IN_):
            var xv = rebind[Scalar[DT]](x[bi, k])
            ir += xv * rebind[Scalar[DT]](W_ih[k, j])
            iz += xv * rebind[Scalar[DT]](W_ih[k, H + j])
            in_pre += xv * rebind[Scalar[DT]](W_ih[k, 2 * H + j])
        for k in range(H):
            var hv = rebind[Scalar[DT]](h_prev[bi, k])
            hr += hv * rebind[Scalar[DT]](W_hh[k, j])
            hz += hv * rebind[Scalar[DT]](W_hh[k, H + j])
            hn += hv * rebind[Scalar[DT]](W_hh[k, 2 * H + j])
        var rg = _sigmoid(ir + hr)
        var zg = _sigmoid(iz + hz)
        var ng = tanh(in_pre + rg * hn)
        out_buf[bi, j] = (
            (Scalar[DT](1.0) - zg) * ng + zg * rebind[Scalar[DT]](h_prev[bi, j])
        )
        cache[bi, j] = rg
        cache[bi, H + j] = zg
        cache[bi, 2 * H + j] = ng
        cache[bi, 3 * H + j] = hn
        j += TPB


def _gru_bwd_gate_kernel[
    BATCH: Int, H: Int,
](
    go: LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, 4 * H), MutAnyOrigin],
    h_prev: LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin],
    d_comb: LayoutTensor[DT, Layout.row_major(BATCH, 6 * H), MutAnyOrigin],
):
    """Per-gate pre-activation grads from the cached activations."""
    var bi = Int(block_idx.x)
    if bi >= BATCH:
        return
    var one = Scalar[DT](1.0)
    var j = Int(thread_idx.x)
    while j < H:
        var rg = rebind[Scalar[DT]](cache[bi, j])
        var zg = rebind[Scalar[DT]](cache[bi, H + j])
        var ng = rebind[Scalar[DT]](cache[bi, 2 * H + j])
        var hn = rebind[Scalar[DT]](cache[bi, 3 * H + j])
        var hval = rebind[Scalar[DT]](h_prev[bi, j])
        var dh_now = rebind[Scalar[DT]](go[bi, j])

        var dz = dh_now * (hval - ng)
        var dn = dh_now * (one - zg)
        var d_pre_n = dn * (one - ng * ng)        # tanh'
        var dr = d_pre_n * hn
        var d_hn = d_pre_n * rg
        var d_pre_r = dr * rg * (one - rg)        # sigmoid'
        var d_pre_z = dz * zg * (one - zg)        # sigmoid'

        # x-side [0, 3H): d_ir | d_iz | d_in
        d_comb[bi, j] = d_pre_r
        d_comb[bi, H + j] = d_pre_z
        d_comb[bi, 2 * H + j] = d_pre_n
        # h-side [3H, 6H): d_hr | d_hz | d_hn
        d_comb[bi, 3 * H + j] = d_pre_r
        d_comb[bi, 4 * H + j] = d_pre_z
        d_comb[bi, 5 * H + j] = d_hn
        j += TPB


def _gru_dWih_kernel[
    BATCH: Int, IN_: Int, H: Int,
](
    x: LayoutTensor[DT, Layout.row_major(BATCH, IN_), MutAnyOrigin],
    d_comb: LayoutTensor[DT, Layout.row_major(BATCH, 6 * H), MutAnyOrigin],
    dW_ih: LayoutTensor[DT, Layout.row_major(IN_, 3 * H), MutAnyOrigin],
):
    """dW_ih[k,c] += Σ_b x[b,k]·d_comb_x[b,c]. Grid (IN_, 3H)."""
    var k_in = Int(block_idx.x)
    var c = Int(block_idx.y)
    if k_in >= IN_ or c >= 3 * H:
        return
    var my = Scalar[DT](0)
    var b = Int(thread_idx.x)
    while b < BATCH:
        my += rebind[Scalar[DT]](x[b, k_in]) * rebind[Scalar[DT]](
            d_comb[b, c]
        )
        b += TPB
    var total = block.sum[block_size=TPB, broadcast=False](val=my)
    if Int(thread_idx.x) == 0:
        dW_ih[k_in, c] = rebind[Scalar[DT]](dW_ih[k_in, c]) + total[0]


def _gru_dWhh_kernel[
    BATCH: Int, H: Int,
](
    h_prev: LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin],
    d_comb: LayoutTensor[DT, Layout.row_major(BATCH, 6 * H), MutAnyOrigin],
    dW_hh: LayoutTensor[DT, Layout.row_major(H, 3 * H), MutAnyOrigin],
):
    """dW_hh[k,c] += Σ_b h_prev[b,k]·d_comb_h[b,c]. Grid (H, 3H)."""
    var k_in = Int(block_idx.x)
    var c = Int(block_idx.y)
    if k_in >= H or c >= 3 * H:
        return
    var my = Scalar[DT](0)
    var b = Int(thread_idx.x)
    while b < BATCH:
        my += rebind[Scalar[DT]](h_prev[b, k_in]) * rebind[Scalar[DT]](
            d_comb[b, 3 * H + c]
        )
        b += TPB
    var total = block.sum[block_size=TPB, broadcast=False](val=my)
    if Int(thread_idx.x) == 0:
        dW_hh[k_in, c] = rebind[Scalar[DT]](dW_hh[k_in, c]) + total[0]


def _gru_db_kernel[
    BATCH: Int, H: Int,
](
    d_comb: LayoutTensor[DT, Layout.row_major(BATCH, 6 * H), MutAnyOrigin],
    db_ih: LayoutTensor[DT, Layout.row_major(3 * H), MutAnyOrigin],
    db_hh: LayoutTensor[DT, Layout.row_major(3 * H), MutAnyOrigin],
):
    """db_ih += Σ_b d_comb_x ; db_hh += Σ_b d_comb_h. Grid (6H,)."""
    var c = Int(block_idx.x)
    if c >= 6 * H:
        return
    var my = Scalar[DT](0)
    var b = Int(thread_idx.x)
    while b < BATCH:
        my += rebind[Scalar[DT]](d_comb[b, c])
        b += TPB
    var total = block.sum[block_size=TPB, broadcast=False](val=my)
    if Int(thread_idx.x) == 0:
        if c < 3 * H:
            db_ih[c] = rebind[Scalar[DT]](db_ih[c]) + total[0]
        else:
            db_hh[c - 3 * H] = rebind[Scalar[DT]](db_hh[c - 3 * H]) + total[0]


def _gru_bwd_input_kernel[
    BATCH: Int, IN_: Int, H: Int,
](
    d_comb: LayoutTensor[DT, Layout.row_major(BATCH, 6 * H), MutAnyOrigin],
    W_ih: LayoutTensor[DT, Layout.row_major(IN_, 3 * H), MutAnyOrigin],
    W_hh: LayoutTensor[DT, Layout.row_major(H, 3 * H), MutAnyOrigin],
    go: LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, 4 * H), MutAnyOrigin],
    dx: LayoutTensor[DT, Layout.row_major(BATCH, IN_), MutAnyOrigin],
    dh: LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin],
):
    """dx = d_comb_x · W_ihᵀ ; dh = d_comb_h · W_hhᵀ + go⊙z. Runs LAST so
    its writes to dx/dh (which may alias x/h_prev) follow every read of
    x/h_prev in the dW kernels."""
    var bi = Int(block_idx.x)
    if bi >= BATCH:
        return
    var jx = Int(thread_idx.x)
    while jx < IN_:
        var acc = Scalar[DT](0)
        for c in range(3 * H):
            acc += rebind[Scalar[DT]](d_comb[bi, c]) * rebind[Scalar[DT]](
                W_ih[jx, c]
            )
        dx[bi, jx] = acc
        jx += TPB
    var jh = Int(thread_idx.x)
    while jh < H:
        var acc = Scalar[DT](0)
        for c in range(3 * H):
            acc += rebind[Scalar[DT]](d_comb[bi, 3 * H + c]) * rebind[
                Scalar[DT]
            ](W_hh[jh, c])
        # Direct path through `z · h`: ∂h'/∂h_jh includes z_jh.
        acc += rebind[Scalar[DT]](go[bi, jh]) * rebind[Scalar[DT]](
            cache[bi, H + jh]
        )
        dh[bi, jh] = acc
        jh += TPB


# ──────────────────────────────────────────────────────────────────────
# GRUCell.
# ──────────────────────────────────────────────────────────────────────


struct GRUCell[IN_: Int, HIDDEN: Int](Module):
    comptime ARITY: Int = 2
    comptime IN_DIMS = Self._build_in_dims()
    comptime IN0_DIM = Self.IN_
    comptime OUT_DIM = Self.HIDDEN

    @staticmethod
    def _build_in_dims() -> InlineArray[Int, 2]:
        var d = InlineArray[Int, 2](fill=0)
        d[0] = Self.IN_
        d[1] = Self.HIDDEN
        return d
    comptime W_IH_SIZE = Self.IN_ * (3 * Self.HIDDEN)
    comptime W_HH_SIZE = Self.HIDDEN * (3 * Self.HIDDEN)
    comptime B_IH_SIZE = 3 * Self.HIDDEN

    var ts: TargetStorage

    # Parameters — Param fields are walked by reflection (for_each_param_auto).
    var W_ih: Param["W_ih", True,  Self.W_IH_SIZE]
    var W_hh: Param["W_hh", True,  Self.W_HH_SIZE]
    var b_ih: Param["b_ih", False, Self.B_IH_SIZE]
    var b_hh: Param["b_hh", False, Self.B_IH_SIZE]

    # Forward caches (CPU path).
    var _r_cache: List[Scalar[DT]]   # [BATCH, H]
    var _z_cache: List[Scalar[DT]]   # [BATCH, H]
    var _n_cache: List[Scalar[DT]]   # [BATCH, H]
    var _hn_pre:  List[Scalar[DT]]   # [BATCH, H]  W_hn·h + b_hn
    var _x_ptr: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]  # fwd x
    var _h_ptr: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]  # fwd h
    var _cache_batch: Int

    # GPU scratch (packed): cache [B, 4H], d_comb [B, 6H].
    var _cache: Cache["gru_cache"]   # [B, 4H] (device-only, lazy)
    var _dcomb: Cache["gru_dcomb"]   # [B, 6H] (device-only, lazy)

    # ------------------------------------------------------------------
    # Defaultable + factories.
    # ------------------------------------------------------------------

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()
        self.W_ih = Param["W_ih", True,  Self.W_IH_SIZE]()
        self.W_hh = Param["W_hh", True,  Self.W_HH_SIZE]()
        self.b_ih = Param["b_ih", False, Self.B_IH_SIZE]()
        self.b_hh = Param["b_hh", False, Self.B_IH_SIZE]()
        self._r_cache = List[Scalar[DT]]()
        self._z_cache = List[Scalar[DT]]()
        self._n_cache = List[Scalar[DT]]()
        self._hn_pre  = List[Scalar[DT]]()
        self._x_ptr = None
        self._h_ptr = None
        self._cache_batch = 0
        self._cache = Cache["gru_cache"]()
        self._dcomb = Cache["gru_dcomb"]()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory."""
        comptime assert target == "cpu" or target == "gpu", (
            "GRUCell: target must be 'cpu' or 'gpu'"
        )
        var g = Self()
        comptime if target == "cpu":
            g.ts = TargetStorage.make_cpu()
            g.W_ih = Param["W_ih", True,  Self.W_IH_SIZE].make_cpu()
            g.W_hh = Param["W_hh", True,  Self.W_HH_SIZE].make_cpu()
            g.b_ih = Param["b_ih", False, Self.B_IH_SIZE].make_cpu()
            g.b_hh = Param["b_hh", False, Self.B_IH_SIZE].make_cpu()
            INIT.init_weight(
                g.W_ih.value_unsafe_ptr_cpu(),
                Self.W_IH_SIZE, Self.IN_, 3 * Self.HIDDEN,
            )
            INIT.init_weight(
                g.W_hh.value_unsafe_ptr_cpu(),
                Self.W_HH_SIZE, Self.HIDDEN, 3 * Self.HIDDEN,
            )
            INIT.init_bias(g.b_ih.value_unsafe_ptr_cpu(), Self.B_IH_SIZE)
            INIT.init_bias(g.b_hh.value_unsafe_ptr_cpu(), Self.B_IH_SIZE)
        else:
            var ctx_v = require_ctx["GRUCell.make[target='gpu']"](ctx)
            g.ts = TargetStorage.make_gpu(ctx_v)
            g.W_ih = Param["W_ih", True,  Self.W_IH_SIZE].make_gpu(ctx_v)
            g.W_hh = Param["W_hh", True,  Self.W_HH_SIZE].make_gpu(ctx_v)
            g.b_ih = Param["b_ih", False, Self.B_IH_SIZE].make_gpu(ctx_v)
            g.b_hh = Param["b_hh", False, Self.B_IH_SIZE].make_gpu(ctx_v)
            Self._gpu_init_param[Self.W_IH_SIZE, Self.IN_, 3 * Self.HIDDEN, INIT](
                ctx_v, g.W_ih.val.dev.value(), is_bias=False
            )
            Self._gpu_init_param[Self.W_HH_SIZE, Self.HIDDEN, 3 * Self.HIDDEN, INIT](
                ctx_v, g.W_hh.val.dev.value(), is_bias=False
            )
            Self._gpu_init_param[Self.B_IH_SIZE, 0, 0, INIT](
                ctx_v, g.b_ih.val.dev.value(), is_bias=True
            )
            Self._gpu_init_param[Self.B_IH_SIZE, 0, 0, INIT](
                ctx_v, g.b_hh.val.dev.value(), is_bias=True
            )
        return g^

    @staticmethod
    def _gpu_init_param[
        N: Int, FAN_IN: Int, FAN_OUT: Int, INIT: Initializer,
    ](ctx: DeviceContext, dst: DeviceBuffer[DT], is_bias: Bool) raises:
        var host = List[Scalar[DT]](length=N, fill=0.0)
        var hp = mptr(host.unsafe_ptr())
        if is_bias:
            INIT.init_bias(hp, N)
        else:
            INIT.init_weight(hp, N, FAN_IN, FAN_OUT)
        var hb = ctx.enqueue_create_host_buffer[DT](N)
        ctx.synchronize()
        for k in range(N):
            hb.unsafe_ptr()[k] = host[k]
        ctx.enqueue_copy(dst, hb)
        ctx.synchronize()

    def _ensure_cache_gpu(mut self, batch: Int) raises:
        self._cache.ensure_gpu(self.ts.ctx.value(), batch * 4 * Self.HIDDEN)

    def _ensure_dcomb_gpu(mut self, batch: Int) raises:
        self._dcomb.ensure_gpu(self.ts.ctx.value(), batch * 6 * Self.HIDDEN)

    # ------------------------------------------------------------------
    # for_each_param / zero_grad: inherited from the Module default (S1,
    # 2026-06-07) — the default reflection-walks IsParam fields, which is
    # exactly what the old pure-`*_auto` overrides did. No override needed.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Cache management.
    # ------------------------------------------------------------------

    def _ensure_cache(mut self, batch: Int):
        var needed = batch * Self.HIDDEN
        if len(self._r_cache) < needed:
            self._r_cache.resize(needed, 0.0)
        if len(self._z_cache) < needed:
            self._z_cache.resize(needed, 0.0)
        if len(self._n_cache) < needed:
            self._n_cache.resize(needed, 0.0)
        if len(self._hn_pre) < needed:
            self._hn_pre.resize(needed, 0.0)
        self._cache_batch = batch

    # ------------------------------------------------------------------
    # Forward.
    # ------------------------------------------------------------------

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorPack[Self.ARITY],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["GRUCell", target](self.ts.target_tag)

        comptime H = Self.HIDDEN
        comptime THREE_H = 3 * Self.HIDDEN

        comptime if target == "gpu":
            self._ensure_cache_gpu(BATCH)
            var ctx = self.ts.ctx.value()
            var x_pg = inputs.ptr[0]()
            var h_pg = inputs.ptr[1]()
            self._x_ptr = x_pg
            self._h_ptr = h_pg
            var x_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.IN0_DIM), MutAnyOrigin
            ](x_pg)
            var h_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, H), MutAnyOrigin
            ](h_pg)
            var out_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, H), MutAnyOrigin
            ](output.ptr)
            var wih = LayoutTensor[
                DT, Layout.row_major(Self.IN0_DIM, THREE_H), MutAnyOrigin
            ](self.W_ih.val.dev.value())
            var whh = LayoutTensor[
                DT, Layout.row_major(H, THREE_H), MutAnyOrigin
            ](self.W_hh.val.dev.value())
            var bih = LayoutTensor[
                DT, Layout.row_major(THREE_H), MutAnyOrigin
            ](self.b_ih.val.dev.value())
            var bhh = LayoutTensor[
                DT, Layout.row_major(THREE_H), MutAnyOrigin
            ](self.b_hh.val.dev.value())
            var cc = LayoutTensor[
                DT, Layout.row_major(BATCH, 4 * H), MutAnyOrigin
            ](self._cache.dev.value())
            comptime kern = _gru_fwd_kernel[BATCH, Self.IN0_DIM, H]
            ctx.enqueue_function[kern](
                x_lt, wih, whh, bih, bhh, h_lt, out_lt, cc,
                grid_dim=(BATCH,), block_dim=(TPB,),
            )
            return

        self._ensure_cache(BATCH)

        var x_p = inputs.ptr[0]()
        var h_p = inputs.ptr[1]()
        var out_p = output.ptr
        self._x_ptr = x_p
        self._h_ptr = h_p

        var W_ih_p = self.W_ih.value_unsafe_ptr_cpu()
        var W_hh_p = self.W_hh.value_unsafe_ptr_cpu()
        var b_ih_p = self.b_ih.value_unsafe_ptr_cpu()
        var b_hh_p = self.b_hh.value_unsafe_ptr_cpu()
        var r_c = self._r_cache.unsafe_ptr()
        var z_c = self._z_cache.unsafe_ptr()
        var n_c = self._n_cache.unsafe_ptr()
        var hn_c = self._hn_pre.unsafe_ptr()

        # Gate pre-activations via BLAS (Apple Accelerate), like Linear /
        # NoisyLinear — previously naive per-(b,col) BATCH·IN·3H scalar
        # dot-products. The nonlinear gate logic (σ/tanh, r⊙hn coupling,
        # h'=(1-z)n+zh) stays scalar (O(BATCH·H)).
        #   ix = x @ W_ih  → [BATCH, 3H]   (bias added below)
        #   hx = h @ W_hh  → [BATCH, 3H]
        var x_tt = TileTensor(x_p, row_major[BATCH, Self.IN0_DIM]())
        var h_tt = TileTensor(h_p, row_major[BATCH, H]())
        var W_ih_tt = TileTensor(W_ih_p, row_major[Self.IN0_DIM, THREE_H]())
        var W_hh_tt = TileTensor(W_hh_p, row_major[H, THREE_H]())
        var ix_buf = alloc[Scalar[DT]](BATCH * THREE_H)
        var hx_buf = alloc[Scalar[DT]](BATCH * THREE_H)
        var ix_tt = TileTensor(ix_buf, row_major[BATCH, THREE_H]())
        var hx_tt = TileTensor(hx_buf, row_major[BATCH, THREE_H]())
        max_matmul[target="cpu"](ix_tt, x_tt, W_ih_tt, None)
        max_matmul[target="cpu"](hx_tt, h_tt, W_hh_tt, None)

        for b in range(BATCH):
            var g_off = b * THREE_H
            var h_off = b * H
            var out_off = b * H
            var c_off = b * H
            for col in range(H):
                var rg = _sigmoid(
                    ix_buf[g_off + col] + b_ih_p[col]
                    + hx_buf[g_off + col] + b_hh_p[col]
                )
                r_c[c_off + col] = rg

                var zg = _sigmoid(
                    ix_buf[g_off + H + col] + b_ih_p[H + col]
                    + hx_buf[g_off + H + col] + b_hh_p[H + col]
                )
                z_c[c_off + col] = zg

                var in_pre = ix_buf[g_off + 2 * H + col] + b_ih_p[2 * H + col]
                # hn_pre = full h-side pre-activation for the n branch.
                var hn_p = hx_buf[g_off + 2 * H + col] + b_hh_p[2 * H + col]
                hn_c[c_off + col] = hn_p

                var ng = tanh(in_pre + rg * hn_p)
                n_c[c_off + col] = ng

                out_p[out_off + col] = (
                    (Scalar[DT](1.0) - zg) * ng + zg * h_p[h_off + col]
                )
        ix_buf.free()
        hx_buf.free()

    # ------------------------------------------------------------------
    # Backward.
    # ------------------------------------------------------------------

    def vjp[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        grad_inputs: TensorPack[Self.ARITY],
    ) raises:
        """Combined backward (S7) — the two phases in fixed order. Single
        source of truth for direct callers; the recurrent unroll + any
        orchestrator that calls the phases directly gets the
        param-before-input order structurally."""
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        self.vjp_param_grads[target, BATCH, POLICY=POLICY, mode=mode](
            grad_output
        )
        self.vjp_grad_input[target, BATCH, POLICY=POLICY, mode=mode](
            grad_output, grad_inputs
        )

    def vjp_param_grads[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        """Phase 1 (S7): dW_ih / dW_hh / db (mode=all). Reads cached x/h
        BEFORE `vjp_grad_input` clobbers their slabs. GPU: the gate kernel
        fills the persistent `_dcomb` ALWAYS (phase 2's input kernel reads
        it), then the dW/db kernels (mode=all). CPU: nothing to do under
        input_only (the gate grads are recomputed in phase 2)."""
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["GRUCell", target](self.ts.target_tag)

        comptime H = Self.HIDDEN
        comptime THREE_H = 3 * Self.HIDDEN

        comptime if target == "gpu":
            self._ensure_dcomb_gpu(BATCH)
            var ctx = self.ts.ctx.value()
            var x_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.IN0_DIM), MutAnyOrigin
            ](self._x_ptr.value())
            var h_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, H), MutAnyOrigin
            ](self._h_ptr.value())
            var go_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, H), MutAnyOrigin
            ](grad_output.ptr)
            var cc = LayoutTensor[
                DT, Layout.row_major(BATCH, 4 * H), MutAnyOrigin
            ](self._cache.dev.value())
            var dcomb = LayoutTensor[
                DT, Layout.row_major(BATCH, 6 * H), MutAnyOrigin
            ](self._dcomb.dev.value())

            # Gate kernel → d_comb (reads cache, h_prev, go). ALWAYS — it
            # is shared preprocessing that phase 2's input kernel reads
            # out of the persistent `_dcomb`.
            comptime gk = _gru_bwd_gate_kernel[BATCH, H]
            ctx.enqueue_function[gk](
                go_lt, cc, h_lt, dcomb, grid_dim=(BATCH,), block_dim=(TPB,),
            )
            # Param grads (read x / h_prev) — enqueued here so they
            # precede dx/dh (phase 2), which clobber the aliased slabs.
            comptime if mode == "all":
                var dwih = LayoutTensor[
                    DT, Layout.row_major(Self.IN0_DIM, THREE_H), MutAnyOrigin
                ](self.W_ih.grd.dev.value())
                var dwhh = LayoutTensor[
                    DT, Layout.row_major(H, THREE_H), MutAnyOrigin,
                ](self.W_hh.grd.dev.value())
                var dbih = LayoutTensor[
                    DT, Layout.row_major(THREE_H), MutAnyOrigin
                ](self.b_ih.grd.dev.value())
                var dbhh = LayoutTensor[
                    DT, Layout.row_major(THREE_H), MutAnyOrigin
                ](self.b_hh.grd.dev.value())
                comptime wk = _gru_dWih_kernel[BATCH, Self.IN0_DIM, H]
                ctx.enqueue_function[wk](
                    x_lt, dcomb, dwih,
                    grid_dim=(Self.IN0_DIM, THREE_H), block_dim=(TPB,),
                )
                comptime hk = _gru_dWhh_kernel[BATCH, H]
                ctx.enqueue_function[hk](
                    h_lt, dcomb, dwhh,
                    grid_dim=(H, THREE_H), block_dim=(TPB,),
                )
                comptime bk = _gru_db_kernel[BATCH, H]
                ctx.enqueue_function[bk](
                    dcomb, dbih, dbhh,
                    grid_dim=(6 * H,), block_dim=(TPB,),
                )
            return

        comptime if mode == "all":
            var x_p = self._x_ptr.value()
            var h_p = self._h_ptr.value()
            var dW_ih_p = self.W_ih.grad_unsafe_ptr_cpu()
            var dW_hh_p = self.W_hh.grad_unsafe_ptr_cpu()
            var db_ih_p = self.b_ih.grad_unsafe_ptr_cpu()
            var db_hh_p = self.b_hh.grad_unsafe_ptr_cpu()
            var r_c = self._r_cache.unsafe_ptr()
            var z_c = self._z_cache.unsafe_ptr()
            var n_c = self._n_cache.unsafe_ptr()
            var hn_c = self._hn_pre.unsafe_ptr()
            var go_p = grad_output.ptr

            # Per-(b,col) gate gradients (scalar, O(BATCH·H)) → d_ix/d_hx
            # [BATCH, 3H] + bias accumulate. Reads caches + cached h only
            # (NOT the dx/dh slabs). d_ix/d_hx are local — phase 2
            # recomputes them (the loop is cheap vs the matmuls).
            var d_ix_buf = alloc[Scalar[DT]](BATCH * THREE_H)
            var d_hx_buf = alloc[Scalar[DT]](BATCH * THREE_H)
            for b in range(BATCH):
                var c_off = b * H
                var g_off = b * THREE_H
                for col in range(H):
                    var dh_now = go_p[c_off + col]
                    var rg = r_c[c_off + col]
                    var zg = z_c[c_off + col]
                    var ng = n_c[c_off + col]
                    var hn_v = hn_c[c_off + col]
                    var h_val = h_p[c_off + col]

                    var dz = dh_now * (h_val - ng)
                    var dn = dh_now * (Scalar[DT](1.0) - zg)
                    var d_pre_n = dn * (Scalar[DT](1.0) - ng * ng)  # tanh'
                    var d_in_n = d_pre_n
                    var dr = d_pre_n * hn_v          # gradient on r
                    var d_hn = d_pre_n * rg          # gradient on hn_pre
                    var d_pre_r = dr * rg * (Scalar[DT](1.0) - rg)  # sigmoid'
                    var d_pre_z = dz * zg * (Scalar[DT](1.0) - zg)  # sigmoid'

                    d_ix_buf[g_off + col]         = d_pre_r
                    d_ix_buf[g_off + H + col]     = d_pre_z
                    d_ix_buf[g_off + 2 * H + col] = d_in_n
                    d_hx_buf[g_off + col]         = d_pre_r
                    d_hx_buf[g_off + H + col]     = d_pre_z
                    d_hx_buf[g_off + 2 * H + col] = d_hn

                    db_ih_p[col]         += d_pre_r
                    db_ih_p[H + col]     += d_pre_z
                    db_ih_p[2 * H + col] += d_in_n
                    db_hh_p[col]         += d_pre_r
                    db_hh_p[H + col]     += d_pre_z
                    db_hh_p[2 * H + col] += d_hn

            var d_ix_tt = TileTensor(d_ix_buf, row_major[BATCH, THREE_H]())
            var d_hx_tt = TileTensor(d_hx_buf, row_major[BATCH, THREE_H]())

            # dW_ih += xᵀ @ d_ix, dW_hh += hᵀ @ d_hx via BLAS. Transpose
            # x/h FIRST: that consumes the reads of x_p/h_p before phase 2
            # clobbers the aliased input slabs.
            var xT_buf = alloc[Scalar[DT]](Self.IN0_DIM * BATCH)
            var hT_buf = alloc[Scalar[DT]](H * BATCH)
            for b in range(BATCH):
                for k in range(Self.IN0_DIM):
                    xT_buf[k * BATCH + b] = x_p[b * Self.IN0_DIM + k]
                for k in range(H):
                    hT_buf[k * BATCH + b] = h_p[b * H + k]
            var xT_tt = TileTensor(xT_buf, row_major[Self.IN0_DIM, BATCH]())
            var hT_tt = TileTensor(hT_buf, row_major[H, BATCH]())
            var dWih_buf = alloc[Scalar[DT]](Self.IN0_DIM * THREE_H)
            var dWhh_buf = alloc[Scalar[DT]](H * THREE_H)
            var dWih_tt = TileTensor(dWih_buf, row_major[Self.IN0_DIM, THREE_H]())
            var dWhh_tt = TileTensor(dWhh_buf, row_major[H, THREE_H]())
            max_matmul[target="cpu"](dWih_tt, xT_tt, d_ix_tt, None)
            max_matmul[target="cpu"](dWhh_tt, hT_tt, d_hx_tt, None)
            for i in range(Self.IN0_DIM * THREE_H):
                dW_ih_p[i] += dWih_buf[i]
            for i in range(H * THREE_H):
                dW_hh_p[i] += dWhh_buf[i]
            xT_buf.free()
            hT_buf.free()
            dWih_buf.free()
            dWhh_buf.free()
            d_ix_buf.free()
            d_hx_buf.free()

    def vjp_grad_input[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        grad_inputs: TensorPack[Self.ARITY],
    ) raises:
        """Phase 2 (S7): dx = d_ix @ W_ihᵀ, dh = d_hx @ W_hhᵀ + z-path.
        Writes grad_inputs[0]/[1] (alias the cached x/h — safe after phase
        1's reads). GPU reads the persistent `_dcomb` filled by phase 1;
        CPU RECOMPUTES the gate grads (cheap scalar loop; reads only
        persistent caches + cached h + grad_output → bit-identical) so the
        phases need no shared CPU scratch. Runs in both modes."""
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["GRUCell", target](self.ts.target_tag)

        comptime H = Self.HIDDEN
        comptime THREE_H = 3 * Self.HIDDEN

        comptime if target == "gpu":
            self._ensure_dcomb_gpu(BATCH)
            var ctx = self.ts.ctx.value()
            var go_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, H), MutAnyOrigin
            ](grad_output.ptr)
            var dx_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.IN0_DIM), MutAnyOrigin
            ](grad_inputs.ptr[0]())
            var dh_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, H), MutAnyOrigin
            ](grad_inputs.ptr[1]())
            var cc = LayoutTensor[
                DT, Layout.row_major(BATCH, 4 * H), MutAnyOrigin
            ](self._cache.dev.value())
            var dcomb = LayoutTensor[
                DT, Layout.row_major(BATCH, 6 * H), MutAnyOrigin
            ](self._dcomb.dev.value())
            var wih = LayoutTensor[
                DT, Layout.row_major(Self.IN0_DIM, THREE_H), MutAnyOrigin
            ](self.W_ih.val.dev.value())
            var whh = LayoutTensor[
                DT, Layout.row_major(H, THREE_H), MutAnyOrigin
            ](self.W_hh.val.dev.value())

            # Input grads (write dx/dh) — reads `_dcomb` filled in phase 1.
            comptime ik = _gru_bwd_input_kernel[BATCH, Self.IN0_DIM, H]
            ctx.enqueue_function[ik](
                dcomb, wih, whh, go_lt, cc, dx_lt, dh_lt,
                grid_dim=(BATCH,), block_dim=(TPB,),
            )
            return

        var h_p = self._h_ptr.value()
        var W_ih_p = self.W_ih.value_unsafe_ptr_cpu()
        var W_hh_p = self.W_hh.value_unsafe_ptr_cpu()
        var r_c = self._r_cache.unsafe_ptr()
        var z_c = self._z_cache.unsafe_ptr()
        var n_c = self._n_cache.unsafe_ptr()
        var hn_c = self._hn_pre.unsafe_ptr()

        var go_p = grad_output.ptr
        var dx_p = grad_inputs.ptr[0]()
        var dh_p = grad_inputs.ptr[1]()

        var W_ih_tt = TileTensor(W_ih_p, row_major[Self.IN0_DIM, THREE_H]())
        var W_hh_tt = TileTensor(W_hh_p, row_major[H, THREE_H]())

        # Recompute the per-(b,col) gate gradients into d_ix/d_hx (NO bias
        # accumulation — that was phase 1). Reads caches + cached h; this
        # read precedes the dx/dh writes below that clobber the aliased
        # slabs.
        var d_ix_buf = alloc[Scalar[DT]](BATCH * THREE_H)
        var d_hx_buf = alloc[Scalar[DT]](BATCH * THREE_H)
        for b in range(BATCH):
            var c_off = b * H
            var g_off = b * THREE_H
            for col in range(H):
                var dh_now = go_p[c_off + col]
                var rg = r_c[c_off + col]
                var zg = z_c[c_off + col]
                var ng = n_c[c_off + col]
                var hn_v = hn_c[c_off + col]
                var h_val = h_p[c_off + col]

                var dz = dh_now * (h_val - ng)
                var dn = dh_now * (Scalar[DT](1.0) - zg)
                var d_pre_n = dn * (Scalar[DT](1.0) - ng * ng)  # tanh'
                var d_in_n = d_pre_n
                var dr = d_pre_n * hn_v          # gradient on r
                var d_hn = d_pre_n * rg          # gradient on hn_pre
                var d_pre_r = dr * rg * (Scalar[DT](1.0) - rg)  # sigmoid'
                var d_pre_z = dz * zg * (Scalar[DT](1.0) - zg)  # sigmoid'

                d_ix_buf[g_off + col]         = d_pre_r
                d_ix_buf[g_off + H + col]     = d_pre_z
                d_ix_buf[g_off + 2 * H + col] = d_in_n
                d_hx_buf[g_off + col]         = d_pre_r
                d_hx_buf[g_off + H + col]     = d_pre_z
                d_hx_buf[g_off + 2 * H + col] = d_hn

        var d_ix_tt = TileTensor(d_ix_buf, row_major[BATCH, THREE_H]())
        var d_hx_tt = TileTensor(d_hx_buf, row_major[BATCH, THREE_H]())

        # Input grads (write dx/dh, which alias x/h → after the reads
        # above). dx = d_ix @ W_ihᵀ ; dh = d_hx @ W_hhᵀ (transpose_b
        # matmul, overwrite), then add the direct ∂h'/∂h_col = z_col path.
        var dx_tt = TileTensor(dx_p, row_major[BATCH, Self.IN0_DIM]())
        var dh_tt = TileTensor(dh_p, row_major[BATCH, H]())
        max_matmul[transpose_b=True, target="cpu"](dx_tt, d_ix_tt, W_ih_tt, None)
        max_matmul[transpose_b=True, target="cpu"](dh_tt, d_hx_tt, W_hh_tt, None)
        for b in range(BATCH):
            var c_off = b * H
            for col in range(H):
                dh_p[c_off + col] += go_p[c_off + col] * z_c[c_off + col]
        d_ix_buf.free()
        d_hx_buf.free()
