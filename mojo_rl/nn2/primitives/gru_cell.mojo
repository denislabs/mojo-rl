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
Backward respects the nn2 param-grad-before-grad-input ordering so it is
alias-safe as a `Module`: a gate kernel fills the per-gate pre-activation
grads `d_comb` (x-side [0,3H) | h-side [3H,6H)); then dW_ih / dW_hh / db
kernels read x / h_prev; then a final kernel writes dx / dh (which may
alias x / h_prev) last. CPU and GPU are bit-parity-validated by
`tests/nn2/test_gru_cell_gpu_parity.mojo`.

There is still no GPU *consumer* (DreamerV3 GPU world-model uses bespoke
fused RSSM ops, not this generic cell) — the GPU path exists for
CPU/GPU parity completeness, exercised by the parity test.
"""

from std.math import exp, tanh
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
    for_each_param_auto,
    zero_grad_auto,
    ParamVisitor,
)
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import TargetStorage, assert_tag_for, ensure_cpu_buffer


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
    var _cache_dev: Optional[DeviceBuffer[DT]]
    var _cache_dev_n: Int
    var _dcomb_dev: Optional[DeviceBuffer[DT]]
    var _dcomb_dev_n: Int

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
        self._cache_dev = None
        self._cache_dev_n = 0
        self._dcomb_dev = None
        self._dcomb_dev_n = 0

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
            if not ctx:
                raise Error("GRUCell.make[target='gpu']: ctx required")
            var ctx_v = ctx.value()
            g.ts = TargetStorage.make_gpu(ctx_v)
            g.W_ih = Param["W_ih", True,  Self.W_IH_SIZE].make_gpu(ctx_v)
            g.W_hh = Param["W_hh", True,  Self.W_HH_SIZE].make_gpu(ctx_v)
            g.b_ih = Param["b_ih", False, Self.B_IH_SIZE].make_gpu(ctx_v)
            g.b_hh = Param["b_hh", False, Self.B_IH_SIZE].make_gpu(ctx_v)
            Self._gpu_init_param[Self.W_IH_SIZE, Self.IN_, 3 * Self.HIDDEN, INIT](
                ctx_v, g.W_ih.value_dev.value(), is_bias=False
            )
            Self._gpu_init_param[Self.W_HH_SIZE, Self.HIDDEN, 3 * Self.HIDDEN, INIT](
                ctx_v, g.W_hh.value_dev.value(), is_bias=False
            )
            Self._gpu_init_param[Self.B_IH_SIZE, 0, 0, INIT](
                ctx_v, g.b_ih.value_dev.value(), is_bias=True
            )
            Self._gpu_init_param[Self.B_IH_SIZE, 0, 0, INIT](
                ctx_v, g.b_hh.value_dev.value(), is_bias=True
            )
            g._cache_dev = ctx_v.enqueue_create_buffer[DT](1)
            g._dcomb_dev = ctx_v.enqueue_create_buffer[DT](1)
        return g^

    @staticmethod
    def _gpu_init_param[
        N: Int, FAN_IN: Int, FAN_OUT: Int, INIT: Initializer,
    ](ctx: DeviceContext, dst: DeviceBuffer[DT], is_bias: Bool) raises:
        var host = List[Scalar[DT]](length=N, fill=0.0)
        var hp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            host.unsafe_ptr()
        )
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
        var needed = batch * 4 * Self.HIDDEN
        if self._cache_dev_n < needed:
            self._cache_dev = self.ts.ctx.value().enqueue_create_buffer[DT](
                needed
            )
            self._cache_dev_n = needed

    def _ensure_dcomb_gpu(mut self, batch: Int) raises:
        var needed = batch * 6 * Self.HIDDEN
        if self._dcomb_dev_n < needed:
            self._dcomb_dev = self.ts.ctx.value().enqueue_create_buffer[DT](
                needed
            )
            self._dcomb_dev_n = needed

    # ------------------------------------------------------------------
    # Param-walker overrides (param-bearing leaf).
    # ------------------------------------------------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        for_each_param_auto[Self, V, target](self, prefix, visitor)

    def zero_grad[target: StaticString](mut self) raises:
        zero_grad_auto[Self, target](self)

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
        var *inputs: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
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
            var x_pg = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                inputs[0].ptr
            )
            var h_pg = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                inputs[1].ptr
            )
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
            ](rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr))
            var wih = LayoutTensor[
                DT, Layout.row_major(Self.IN0_DIM, THREE_H), MutAnyOrigin
            ](self.W_ih.value_dev.value())
            var whh = LayoutTensor[
                DT, Layout.row_major(H, THREE_H), MutAnyOrigin
            ](self.W_hh.value_dev.value())
            var bih = LayoutTensor[
                DT, Layout.row_major(THREE_H), MutAnyOrigin
            ](self.b_ih.value_dev.value())
            var bhh = LayoutTensor[
                DT, Layout.row_major(THREE_H), MutAnyOrigin
            ](self.b_hh.value_dev.value())
            var cc = LayoutTensor[
                DT, Layout.row_major(BATCH, 4 * H), MutAnyOrigin
            ](self._cache_dev.value())
            comptime kern = _gru_fwd_kernel[BATCH, Self.IN0_DIM, H]
            ctx.enqueue_function[kern](
                x_lt, wih, whh, bih, bhh, h_lt, out_lt, cc,
                grid_dim=(BATCH,), block_dim=(TPB,),
            )
            return

        self._ensure_cache(BATCH)

        var x_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](inputs[0].ptr)
        var h_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](inputs[1].ptr)
        var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
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

        # For each row b:
        #   1. compute ix_j = sum_k x[b,k]*W_ih[k,j] + b_ih[j]  for j in [0, 3H)
        #   2. compute hx_j = sum_k h[b,k]*W_hh[k,j] + b_hh[j]  for j in [0, 3H)
        #   3. r = σ(ix[0:H]  + hx[0:H])
        #      z = σ(ix[H:2H] + hx[H:2H])
        #      hn_pre = hx[2H:3H]
        #      n = tanh(ix[2H:3H] + r ⊙ hn_pre)
        #      h' = (1-z)*n + z*h
        for b in range(BATCH):
            var x_off = b * Self.IN0_DIM
            var h_off = b * H
            var out_off = b * H
            var c_off = b * H

            # Slot scratch on stack via inline loops (no large temporaries).
            for col in range(H):
                # ir + hr
                var ir: Scalar[DT] = b_ih_p[col]
                var hr: Scalar[DT] = b_hh_p[col]
                for k in range(Self.IN0_DIM):
                    ir += x_p[x_off + k] * W_ih_p[k * THREE_H + col]
                for k in range(H):
                    hr += h_p[h_off + k] * W_hh_p[k * THREE_H + col]
                var rg = _sigmoid(ir + hr)
                r_c[c_off + col] = rg

                # iz + hz
                var iz: Scalar[DT] = b_ih_p[H + col]
                var hz: Scalar[DT] = b_hh_p[H + col]
                for k in range(Self.IN0_DIM):
                    iz += x_p[x_off + k] * W_ih_p[k * THREE_H + H + col]
                for k in range(H):
                    hz += h_p[h_off + k] * W_hh_p[k * THREE_H + H + col]
                var zg = _sigmoid(iz + hz)
                z_c[c_off + col] = zg

                # in_pre = sum + b_in (only x-side part)
                var in_pre: Scalar[DT] = b_ih_p[2 * H + col]
                for k in range(Self.IN0_DIM):
                    in_pre += x_p[x_off + k] * W_ih_p[k * THREE_H + 2 * H + col]
                # hn_pre = sum + b_hn (h-side part — full pre-activation
                # for the n branch, before the r gate).
                var hn_p: Scalar[DT] = b_hh_p[2 * H + col]
                for k in range(H):
                    hn_p += h_p[h_off + k] * W_hh_p[k * THREE_H + 2 * H + col]
                hn_c[c_off + col] = hn_p

                var ng = tanh(in_pre + rg * hn_p)
                n_c[c_off + col] = ng

                # h' = (1 − z) * n + z * h
                out_p[out_off + col] = (
                    (Scalar[DT](1.0) - zg) * ng + zg * h_p[h_off + col]
                )

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
        mut *grad_inputs: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
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
            ](rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output.ptr))
            var dx_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.IN0_DIM), MutAnyOrigin
            ](rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_inputs[0].ptr))
            var dh_lt = LayoutTensor[
                DT, Layout.row_major(BATCH, H), MutAnyOrigin
            ](rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_inputs[1].ptr))
            var cc = LayoutTensor[
                DT, Layout.row_major(BATCH, 4 * H), MutAnyOrigin
            ](self._cache_dev.value())
            var dcomb = LayoutTensor[
                DT, Layout.row_major(BATCH, 6 * H), MutAnyOrigin
            ](self._dcomb_dev.value())
            var wih = LayoutTensor[
                DT, Layout.row_major(Self.IN0_DIM, THREE_H), MutAnyOrigin
            ](self.W_ih.value_dev.value())
            var whh = LayoutTensor[
                DT, Layout.row_major(H, THREE_H), MutAnyOrigin
            ](self.W_hh.value_dev.value())

            # 1. Gate kernel → d_comb (reads cache, h_prev, go).
            comptime gk = _gru_bwd_gate_kernel[BATCH, H]
            ctx.enqueue_function[gk](
                go_lt, cc, h_lt, dcomb, grid_dim=(BATCH,), block_dim=(TPB,),
            )
            # 2-4. Param grads (read x / h_prev) — BEFORE dx/dh writes.
            comptime if mode == "all":
                var dwih = LayoutTensor[
                    DT, Layout.row_major(Self.IN0_DIM, THREE_H), MutAnyOrigin
                ](self.W_ih.grad_dev.value())
                var dwhh = LayoutTensor[
                    DT, Layout.row_major(H, THREE_H), MutAnyOrigin,
                ](self.W_hh.grad_dev.value())
                var dbih = LayoutTensor[
                    DT, Layout.row_major(THREE_H), MutAnyOrigin
                ](self.b_ih.grad_dev.value())
                var dbhh = LayoutTensor[
                    DT, Layout.row_major(THREE_H), MutAnyOrigin
                ](self.b_hh.grad_dev.value())
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
            # 5. Input grads (write dx/dh — may alias x/h, so run LAST).
            comptime ik = _gru_bwd_input_kernel[BATCH, Self.IN0_DIM, H]
            ctx.enqueue_function[ik](
                dcomb, wih, whh, go_lt, cc, dx_lt, dh_lt,
                grid_dim=(BATCH,), block_dim=(TPB,),
            )
            return

        var x_p = self._x_ptr.value()
        var h_p = self._h_ptr.value()
        var W_ih_p = self.W_ih.value_unsafe_ptr_cpu()
        var W_hh_p = self.W_hh.value_unsafe_ptr_cpu()
        var dW_ih_p = self.W_ih.grad_unsafe_ptr_cpu()
        var dW_hh_p = self.W_hh.grad_unsafe_ptr_cpu()
        var db_ih_p = self.b_ih.grad_unsafe_ptr_cpu()
        var db_hh_p = self.b_hh.grad_unsafe_ptr_cpu()
        var r_c = self._r_cache.unsafe_ptr()
        var z_c = self._z_cache.unsafe_ptr()
        var n_c = self._n_cache.unsafe_ptr()
        var hn_c = self._hn_pre.unsafe_ptr()

        var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output.ptr)
        var dx_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_inputs[0].ptr)
        var dh_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_inputs[1].ptr)

        # PARAM-GRAD-FIRST INVARIANT: like Linear, x_p and h_p MAY alias
        # the orchestrator's input slabs that dx_p / dh_p write to. We
        # compute all parameter grads (which read x, h) and the input
        # grads (which read W) into stack scratch first, then write
        # dx, dh as the final step. Since dx and dh are produced from
        # the per-row d_pre_* signals (not from x, h directly), it's safe
        # to interleave per-row.

        # Initialize grad inputs to zero (we accumulate into them per element).
        for b in range(BATCH):
            for k in range(Self.IN0_DIM):
                dx_p[b * Self.IN0_DIM + k] = 0.0
            for k in range(H):
                dh_p[b * H + k] = 0.0

        for b in range(BATCH):
            var x_off = b * Self.IN0_DIM
            var h_off = b * H
            var c_off = b * H

            for col in range(H):
                var dh_now = go_p[c_off + col]
                var rg = r_c[c_off + col]
                var zg = z_c[c_off + col]
                var ng = n_c[c_off + col]
                var hn_v = hn_c[c_off + col]
                var h_val = h_p[h_off + col]

                # Gate / candidate gradients.
                var dz = dh_now * (h_val - ng)
                var dn = dh_now * (Scalar[DT](1.0) - zg)
                var d_pre_n = dn * (Scalar[DT](1.0) - ng * ng)  # tanh'

                # Split d_pre_n across input-projection (d_in_n) and
                # `r * hn` summand.
                var d_in_n = d_pre_n
                var dr_x_hn = d_pre_n
                var dr = dr_x_hn * hn_v          # gradient on r
                var d_hn = dr_x_hn * rg          # gradient on hn_pre

                var d_pre_r = dr * rg * (Scalar[DT](1.0) - rg)  # sigmoid'
                var d_pre_z = dz * zg * (Scalar[DT](1.0) - zg)  # sigmoid'

                # Combined d-vectors per gate index.
                var d_ir = d_pre_r
                var d_iz = d_pre_z
                var d_in_g = d_in_n
                var d_hr_g = d_pre_r
                var d_hz_g = d_pre_z
                var d_hn_g = d_hn

                # ----- Param grads (mode == "all" only) -----
                comptime if mode == "all":
                    # b_ih: sum across batch
                    db_ih_p[col]         += d_ir
                    db_ih_p[H + col]     += d_iz
                    db_ih_p[2 * H + col] += d_in_g
                    db_hh_p[col]         += d_hr_g
                    db_hh_p[H + col]     += d_hz_g
                    db_hh_p[2 * H + col] += d_hn_g

                    # W_ih [IN, 3H] += x^T · d_ix
                    for k in range(Self.IN0_DIM):
                        var xv = x_p[x_off + k]
                        dW_ih_p[k * THREE_H + col]         += xv * d_ir
                        dW_ih_p[k * THREE_H + H + col]     += xv * d_iz
                        dW_ih_p[k * THREE_H + 2 * H + col] += xv * d_in_g
                    # W_hh [H, 3H] += h^T · d_hx
                    for k in range(H):
                        var hv = h_p[h_off + k]
                        dW_hh_p[k * THREE_H + col]         += hv * d_hr_g
                        dW_hh_p[k * THREE_H + H + col]     += hv * d_hz_g
                        dW_hh_p[k * THREE_H + 2 * H + col] += hv * d_hn_g

                # ----- Input grads -----
                # d_x[k] += d_ir·W_ih[k, col] + d_iz·W_ih[k, H+col] + d_in_g·W_ih[k, 2H+col]
                for k in range(Self.IN0_DIM):
                    dx_p[x_off + k] += (
                        d_ir   * W_ih_p[k * THREE_H + col]
                        + d_iz * W_ih_p[k * THREE_H + H + col]
                        + d_in_g * W_ih_p[k * THREE_H + 2 * H + col]
                    )
                # d_h[k] += d_hr_g·W_hh[k, col] + d_hz_g·W_hh[k, H+col] + d_hn_g·W_hh[k, 2H+col]
                # Plus the direct path through `z · h` — only for k = col.
                for k in range(H):
                    dh_p[h_off + k] += (
                        d_hr_g   * W_hh_p[k * THREE_H + col]
                        + d_hz_g * W_hh_p[k * THREE_H + H + col]
                        + d_hn_g * W_hh_p[k * THREE_H + 2 * H + col]
                    )
                # Direct path: ∂h'/∂h_col = z_col
                dh_p[h_off + col] += dh_now * zg
