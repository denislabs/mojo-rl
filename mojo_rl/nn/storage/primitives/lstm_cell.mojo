"""LSTMCell[IN_, HIDDEN] — PyTorch-equivalent LSTM cell (storage surface).

Transformed from legacy `nn.primitives.lstm_cell` (surface-only change): the
gate math, the CPU BLAS pre-activation path, and the four GPU kernels are
carried over VERBATIM; only the surface (UnsafePointer/TileTensor view args +
`TargetStorage` → owning `Tensor` storage cells + `Param`/`make[target,INIT]`)
changes.

Unlike a feed-forward `Module`, an LSTM threads TWO states (h, c) across time
and is trained with BPTT, so it exposes an explicit recurrent API
(`step_forward` / `step_backward` / `step_forward_no_cache`) rather than the
single-input/single-output `Module.forward`. The caller owns the (h, c) state
`Tensor`s and a per-timestep cache `Tensor`, and runs the BPTT loop (see
`LSTMSeq`). Its `Module.forward` / `vjp` / `vjp_param_grads` RAISE — use the step
API (the raising stubs satisfy the trait; the recurrent backward is the bespoke
`step_backward`, not orchestrator-driven).

Parameters are `Param` fields, so the reflection `for_each_param` / `zero_grad`
defaults walk them and the cell composes with the storage `Adam` / checkpointing.

Math (PyTorch convention, gates packed [i | f | g | o], each HIDDEN):
    preact = x · W_ih + h_prev · W_hh + b           [BATCH, 4·H]
    i = σ(preact[0:H]),  f = σ(preact[H:2H])
    g = tanh(preact[2H:3H]),  o = σ(preact[3H:4H])
    c_t = f ⊙ c_prev + i ⊙ g
    h_t = o ⊙ tanh(c_t)

Storage (row-major):
    W_ih [IN, 4·H]   W_hh [H, 4·H]   b [4·H]
Cache (per timestep, BATCH-major): [i | f | g | o | tanh(c_t)], 5·H wide.
"""

from std.math import exp, tanh
from std.gpu import thread_idx, block_idx, global_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, TileTensor, row_major
from linalg.matmul import matmul as max_matmul

from mojo_rl.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from ..loss.sac import polyak_tensor


@always_inline
def _sigmoid(x: Scalar[DT]) -> Scalar[DT]:
    if x >= 0:
        var e = exp(-x)
        return Scalar[DT](1.0) / (Scalar[DT](1.0) + e)
    var e = exp(x)
    return e / (Scalar[DT](1.0) + e)


# ──────────────────────────────────────────────────────────────────────
# GPU kernels (separate W_ih / W_hh / b buffers — Param layout). Carried
# VERBATIM from legacy (args MutAnyOrigin = the GPU kernel ABI).
# ──────────────────────────────────────────────────────────────────────


def _lstm_gate_fwd_kernel[
    BATCH: Int, H: Int, CACHE: Int, WITH_CACHE: Bool,
](
    ix: LayoutTensor[DT, Layout.row_major(BATCH, 4 * H), MutAnyOrigin],
    hx: LayoutTensor[DT, Layout.row_major(BATCH, 4 * H), MutAnyOrigin],
    b: LayoutTensor[DT, Layout.row_major(4 * H), MutAnyOrigin],
    c_prev: LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin],
    h_t: LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin],
    c_t: LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, CACHE), MutAnyOrigin],
):
    """Elementwise gate/cell from the two GEMM pre-activations (ix = x·W_ih,
    hx = h_prev·W_hh). One thread per (sample, hidden unit)."""
    var gid = Int(global_idx.x)
    if gid >= BATCH * H:
        return
    var bi = gid // H
    var j = gid % H
    var i_pre = rebind[Scalar[DT]](ix[bi, j]) + rebind[Scalar[DT]](
        hx[bi, j]
    ) + rebind[Scalar[DT]](b[j])
    var f_pre = rebind[Scalar[DT]](ix[bi, H + j]) + rebind[Scalar[DT]](
        hx[bi, H + j]
    ) + rebind[Scalar[DT]](b[H + j])
    var g_pre = rebind[Scalar[DT]](ix[bi, 2 * H + j]) + rebind[Scalar[DT]](
        hx[bi, 2 * H + j]
    ) + rebind[Scalar[DT]](b[2 * H + j])
    var o_pre = rebind[Scalar[DT]](ix[bi, 3 * H + j]) + rebind[Scalar[DT]](
        hx[bi, 3 * H + j]
    ) + rebind[Scalar[DT]](b[3 * H + j])

    var i_val = _sigmoid(i_pre)
    var f_val = _sigmoid(f_pre)
    var g_val = tanh(g_pre)
    var o_val = _sigmoid(o_pre)
    var c_new = f_val * rebind[Scalar[DT]](c_prev[bi, j]) + i_val * g_val
    var tc = tanh(c_new)
    c_t[bi, j] = c_new
    h_t[bi, j] = o_val * tc
    comptime if WITH_CACHE:
        cache[bi, j] = i_val
        cache[bi, H + j] = f_val
        cache[bi, 2 * H + j] = g_val
        cache[bi, 3 * H + j] = o_val
        cache[bi, 4 * H + j] = tc


def _lstm_gate_grad_kernel[
    BATCH: Int, H: Int, CACHE: Int,
](
    dh: LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin],
    dc: LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin],
    c_prev: LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, CACHE), MutAnyOrigin],
    dc_prev: LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin],
    d_comb: LayoutTensor[DT, Layout.row_major(BATCH, 4 * H), MutAnyOrigin],
):
    """Gate-grad math → combined pre-activation grad d_comb [BATCH, 4H] + the
    cell-state grad dc_prev. One thread per (sample, hidden unit). The dx /
    dh_prev / dW matmuls then run through max_matmul."""
    var gid = Int(global_idx.x)
    if gid >= BATCH * H:
        return
    var bi = gid // H
    var j = gid % H
    var one = Scalar[DT](1.0)
    var i_v = rebind[Scalar[DT]](cache[bi, j])
    var f_v = rebind[Scalar[DT]](cache[bi, H + j])
    var g_v = rebind[Scalar[DT]](cache[bi, 2 * H + j])
    var o_v = rebind[Scalar[DT]](cache[bi, 3 * H + j])
    var tc = rebind[Scalar[DT]](cache[bi, 4 * H + j])
    var dh_j = rebind[Scalar[DT]](dh[bi, j])
    var dc_j = rebind[Scalar[DT]](dc[bi, j])

    var do_post = dh_j * tc
    var dc_total = dc_j + dh_j * o_v * (one - tc * tc)
    var df_post = dc_total * rebind[Scalar[DT]](c_prev[bi, j])
    var di_post = dc_total * g_v
    var dg_post = dc_total * i_v
    dc_prev[bi, j] = dc_total * f_v

    d_comb[bi, j]         = di_post * i_v * (one - i_v)
    d_comb[bi, H + j]     = df_post * f_v * (one - f_v)
    d_comb[bi, 2 * H + j] = dg_post * (one - g_v * g_v)
    d_comb[bi, 3 * H + j] = do_post * o_v * (one - o_v)


def _lstm_transpose_kernel[
    ROWS: Int, COLS: Int,
](
    src: LayoutTensor[DT, Layout.row_major(ROWS, COLS), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(COLS, ROWS), MutAnyOrigin],
):
    """dst[COLS, ROWS] = src[ROWS, COLS]ᵀ (captures x / h_prev for the dW GEMMs
    before dx / dh_prev clobber an aliased input slab)."""
    var idx = Int(global_idx.x)
    if idx < ROWS * COLS:
        dst[idx % COLS, idx // COLS] = src[idx // COLS, idx % COLS]


def _lstm_accum_kernel[
    N: Int,
](
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    """dst += src (accumulate dW_tmp into the persistent param grad)."""
    var i = Int(global_idx.x)
    if i < N:
        dst[i] += src[i]


def _lstm_db_kernel[
    BATCH: Int, H: Int,
](
    d_comb: LayoutTensor[DT, Layout.row_major(BATCH, 4 * H), MutAnyOrigin],
    db: LayoutTensor[DT, Layout.row_major(4 * H), MutAnyOrigin],
):
    """Accumulate db += Σ_b d_comb. Grid (4H,)."""
    var k = Int(block_idx.x)
    if k >= 4 * H:
        return
    var my = Scalar[DT](0)
    var b = Int(thread_idx.x)
    while b < BATCH:
        my += rebind[Scalar[DT]](d_comb[b, k])
        b += TPB
    var total = block.sum[block_size=TPB, broadcast=False](val=my)
    if Int(thread_idx.x) == 0:
        db[k] = rebind[Scalar[DT]](db[k]) + total[0]


# ──────────────────────────────────────────────────────────────────────
# LSTMCell.
# ──────────────────────────────────────────────────────────────────────


struct LSTMCell[IN_: Int, HIDDEN: Int](Module):
    comptime ARITY: Int = 2
    comptime IN_DIMS = Self._build_in_dims()
    comptime OUT_DIM = 2 * Self.HIDDEN  # packed [h ; c]

    @staticmethod
    def _build_in_dims() -> InlineArray[Int, 2]:
        var d = InlineArray[Int, 2](fill=0)
        d[0] = Self.IN_
        d[1] = 2 * Self.HIDDEN
        return d

    comptime W_IH_SIZE = Self.IN_ * (4 * Self.HIDDEN)
    comptime W_HH_SIZE = Self.HIDDEN * (4 * Self.HIDDEN)
    comptime B_SIZE = 4 * Self.HIDDEN
    comptime CACHE_SIZE = 5 * Self.HIDDEN  # [i | f | g | o | tanh_c]

    var W_ih: Param["W_ih", True,  Self.W_IH_SIZE]
    var W_hh: Param["W_hh", True,  Self.W_HH_SIZE]
    var b:    Param["b",    False, Self.B_SIZE]
    var _dcomb: Tensor  # GPU d_combined scratch (lazy by BATCH)
    # GPU forward GEMM scratch (lazy by BATCH): ix = x·W_ih, hx = h_prev·W_hh.
    var _ix: Tensor
    var _hx: Tensor
    # GPU backward GEMM scratch (lazy): xᵀ/h_prevᵀ + dW_ih/dW_hh temps for the
    # transpose + max_matmul + accumulate path (max_matmul rejects transpose_a).
    var _xT: Tensor
    var _hT: Tensor
    var _dWih_tmp: Tensor
    var _dWhh_tmp: Tensor

    # ------------------------------------------------------------------
    # Defaultable + factory.
    # ------------------------------------------------------------------

    def __init__(out self):
        self.W_ih = Param["W_ih", True,  Self.W_IH_SIZE]()
        self.W_hh = Param["W_hh", True,  Self.W_HH_SIZE]()
        self.b    = Param["b",    False, Self.B_SIZE]()
        self._dcomb = Tensor()
        self._ix = Tensor()
        self._hx = Tensor()
        self._xT = Tensor()
        self._hT = Tensor()
        self._dWih_tmp = Tensor()
        self._dWhh_tmp = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Unified CPU/GPU factory (Param.make + INIT, mirrors Linear)."""
        comptime assert target == "cpu" or target == "gpu", (
            "LSTMCell: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        m.W_ih = Param["W_ih", True,  Self.W_IH_SIZE].make[target](ctx)
        m.W_hh = Param["W_hh", True,  Self.W_HH_SIZE].make[target](ctx)
        m.b    = Param["b",    False, Self.B_SIZE].make[target](ctx)
        INIT.init_weight[target](
            m.W_ih.val, Self.W_IH_SIZE, Self.IN_, 4 * Self.HIDDEN, ctx
        )
        INIT.init_weight[target](
            m.W_hh.val, Self.W_HH_SIZE, Self.HIDDEN, 4 * Self.HIDDEN, ctx
        )
        INIT.init_bias[target](m.b.val, Self.B_SIZE, ctx)
        return m^

    # ------------------------------------------------------------------
    # for_each_param / zero_grad: inherit the Module reflection defaults
    # (auto-discover the W_ih / W_hh / b Params).
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Module conformance — recurrent cell uses the step API instead. The
    # raising stubs satisfy the trait (legacy raised identically).
    # ------------------------------------------------------------------

    def forward[
        target: StaticString, B: Int, o: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        raise Error(
            "LSTMCell is recurrent — use step_forward/step_backward "
            "(see LSTMSeq), not Module.forward"
        )

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        raise Error(
            "LSTMCell is recurrent — use step_backward "
            "(see LSTMSeq), not Module.vjp"
        )

    # ------------------------------------------------------------------
    # GPU scratch.
    # ------------------------------------------------------------------

    def _ensure_dcomb_gpu(mut self, ctx: DeviceContext, batch: Int) raises:
        self._dcomb.ensure_gpu(ctx, batch * 4 * Self.HIDDEN)

    # ------------------------------------------------------------------
    # Recurrent step API. Args are owning `Tensor` storage cells; each
    # method builds its typed view internally (CPU `TileTensor` over
    # `.data + offset`, GPU device `LayoutTensor` over a `create_sub_buffer`
    # at the offset). The per-tensor element offsets (default 0 for the
    # standalone cell) let `LSTMSeq` pass shared timestep-major buffers +
    # per-step offsets — zero-copy, faithful to the legacy pointer-offset
    # slicing. `ctx` is the GPU DeviceContext (ignored on CPU).
    # ------------------------------------------------------------------

    def step_forward[target: StaticString, BATCH: Int](
        mut self,
        mut x: Tensor,
        mut h: Tensor,
        mut c: Tensor,
        mut cache: Tensor,
        ctx: Optional[DeviceContext] = None,
        x_off: Int = 0,
        h_prev_off: Int = 0,
        c_prev_off: Int = 0,
        h_t_off: Int = 0,
        c_t_off: Int = 0,
        cache_off: Int = 0,
    ) raises:
        """One LSTM step; writes h[h_t_off..], c[c_t_off..], and the backward
        cache ([i | f | g | o | tanh_c], 5·H wide). The recurrent state h / c is
        ONE `Tensor` each, carrying the read slab (h_prev_off / c_prev_off) AND
        the write slab (h_t_off / c_t_off) — this single-arg form is what lets
        `LSTMSeq` thread the recurrence through a shared `h_buf`/`c_buf` without
        an exclusivity violation (the same buffer can't be two `mut` args). The
        standalone cell passes h / c sized 2·BATCH·H (prev=slab 0, out=slab 1)."""
        comptime H = Self.HIDDEN
        comptime FOURH = 4 * Self.HIDDEN

        comptime if target == "cpu":
            h.ensure(h_t_off + BATCH * H)
            c.ensure(c_t_off + BATCH * H)
            cache.ensure(cache_off + BATCH * Self.CACHE_SIZE)
            var xv = TileTensor(
                x.data.unsafe_ptr() + x_off, row_major[BATCH, Self.IN_]()
            )
            var hp = TileTensor(
                h.data.unsafe_ptr() + h_prev_off,
                row_major[BATCH, Self.HIDDEN](),
            )
            var b_p = self.b.val.data.unsafe_ptr()
            var cc_p = cache.data.unsafe_ptr() + cache_off
            var ct_p = c.data.unsafe_ptr() + c_t_off
            var ht_p = h.data.unsafe_ptr() + h_t_off
            var cp_p = c.data.unsafe_ptr() + c_prev_off
            # Gate pre-activations via BLAS (Apple Accelerate), mirroring
            # Linear's CPU path. ix = x @ W_ih, hx = h_prev @ W_hh. The gate
            # nonlinearities below stay scalar (O(BATCH·H)).
            var ix_list = List[Scalar[DT]](length=BATCH * FOURH, fill=Scalar[DT](0))
            var hx_list = List[Scalar[DT]](length=BATCH * FOURH, fill=Scalar[DT](0))
            var ix_tt = TileTensor(ix_list, row_major[BATCH, FOURH]())
            var hx_tt = TileTensor(hx_list, row_major[BATCH, FOURH]())
            var Wih_tt = TileTensor(self.W_ih.val.data, row_major[Self.IN_, FOURH]())
            var Whh_tt = TileTensor(self.W_hh.val.data, row_major[Self.HIDDEN, FOURH]())
            max_matmul[target="cpu"](ix_tt, xv, Wih_tt, None)
            max_matmul[target="cpu"](hx_tt, hp, Whh_tt, None)
            for bi in range(BATCH):
                for k in range(FOURH):
                    var pre: Scalar[DT] = (
                        ix_list[bi * FOURH + k] + hx_list[bi * FOURH + k] + b_p[k]
                    )
                    var act: Scalar[DT]
                    if k < 3 * H:
                        act = _sigmoid(pre) if k < 2 * H else tanh(pre)
                    else:
                        act = _sigmoid(pre)
                    cc_p[bi * Self.CACHE_SIZE + k] = act
                for j in range(H):
                    var base = bi * Self.CACHE_SIZE
                    var i_v = cc_p[base + j]
                    var f_v = cc_p[base + H + j]
                    var g_v = cc_p[base + 2 * H + j]
                    var o_v = cc_p[base + 3 * H + j]
                    var c_new = f_v * cp_p[bi * H + j] + i_v * g_v
                    var tc = tanh(c_new)
                    ct_p[bi * H + j] = c_new
                    ht_p[bi * H + j] = o_v * tc
                    cc_p[base + 4 * H + j] = tc
        else:
            var dctx = ctx.value()
            h.ensure_gpu(dctx, h_t_off + BATCH * H)
            c.ensure_gpu(dctx, c_t_off + BATCH * H)
            cache.ensure_gpu(dctx, cache_off + BATCH * Self.CACHE_SIZE)
            self._ix.ensure_gpu(dctx, BATCH * FOURH)
            self._hx.ensure_gpu(dctx, BATCH * FOURH)
            var xb = x.dev.value().create_sub_buffer[DT](x_off, BATCH * Self.IN_)
            var hpb = h.dev.value().create_sub_buffer[DT](h_prev_off, BATCH * H)
            var cpb = c.dev.value().create_sub_buffer[DT](c_prev_off, BATCH * H)
            var htb = h.dev.value().create_sub_buffer[DT](h_t_off, BATCH * H)
            var ctb = c.dev.value().create_sub_buffer[DT](c_t_off, BATCH * H)
            var ccb = cache.dev.value().create_sub_buffer[DT](
                cache_off, BATCH * Self.CACHE_SIZE
            )
            # ix = x @ W_ih ; hx = h_prev @ W_hh  (two GEMMs, was a hand-rolled
            # per-thread inner product over IN_+H for all 4 gates).
            var x_v = TileTensor(xb, row_major[BATCH, Self.IN_]())
            var hp_v = TileTensor(hpb, row_major[BATCH, Self.HIDDEN]())
            var Wih_v = TileTensor(
                self.W_ih.val.dev.value(), row_major[Self.IN_, FOURH]()
            )
            var Whh_v = TileTensor(
                self.W_hh.val.dev.value(), row_major[Self.HIDDEN, FOURH]()
            )
            var ix_v = TileTensor(self._ix.dev.value(), row_major[BATCH, FOURH]())
            var hx_v = TileTensor(self._hx.dev.value(), row_major[BATCH, FOURH]())
            max_matmul[target="gpu"](ix_v, x_v, Wih_v, dctx)
            max_matmul[target="gpu"](hx_v, hp_v, Whh_v, dctx)
            # elementwise gates + cell update + cache.
            comptime gk = _lstm_gate_fwd_kernel[BATCH, H, Self.CACHE_SIZE, True]
            comptime nblk = (BATCH * H + TPB - 1) // TPB
            dctx.enqueue_function[gk](
                LayoutTensor[DT, Layout.row_major(BATCH, FOURH), MutAnyOrigin](
                    self._ix.dev.value()
                ),
                LayoutTensor[DT, Layout.row_major(BATCH, FOURH), MutAnyOrigin](
                    self._hx.dev.value()
                ),
                self.b.val.lt["gpu", Layout.row_major(FOURH)](),
                LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin](cpb),
                LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin](htb),
                LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin](ctb),
                LayoutTensor[DT, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin](ccb),
                grid_dim=(nblk,), block_dim=(TPB,),
            )

    def step_forward_no_cache[target: StaticString, BATCH: Int](
        mut self,
        mut x: Tensor,
        mut h_prev: Tensor,
        mut c_prev: Tensor,
        mut h_t: Tensor,
        mut c_t: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Inference step (no cache) — for eval / sampling (offset 0 only)."""
        comptime H = Self.HIDDEN
        comptime FOURH = 4 * Self.HIDDEN

        comptime if target == "cpu":
            h_t.ensure(BATCH * H)
            c_t.ensure(BATCH * H)
            var xv = TileTensor(x.data, row_major[BATCH, Self.IN_]())
            var hp = TileTensor(h_prev.data, row_major[BATCH, Self.HIDDEN]())
            var cp_p = c_prev.data.unsafe_ptr()
            var b_p = self.b.val.data.unsafe_ptr()
            var ct_p = c_t.data.unsafe_ptr()
            var ht_p = h_t.data.unsafe_ptr()
            # BLAS gate pre-activations (see step_forward).
            var ix_list = List[Scalar[DT]](length=BATCH * FOURH, fill=Scalar[DT](0))
            var hx_list = List[Scalar[DT]](length=BATCH * FOURH, fill=Scalar[DT](0))
            var ix_tt = TileTensor(ix_list, row_major[BATCH, FOURH]())
            var hx_tt = TileTensor(hx_list, row_major[BATCH, FOURH]())
            var Wih_tt = TileTensor(self.W_ih.val.data, row_major[Self.IN_, FOURH]())
            var Whh_tt = TileTensor(self.W_hh.val.data, row_major[Self.HIDDEN, FOURH]())
            max_matmul[target="cpu"](ix_tt, xv, Wih_tt, None)
            max_matmul[target="cpu"](hx_tt, hp, Whh_tt, None)
            for bi in range(BATCH):
                var gates = InlineArray[Scalar[DT], 4 * Self.HIDDEN](fill=0.0)
                for k in range(FOURH):
                    var pre: Scalar[DT] = (
                        ix_list[bi * FOURH + k] + hx_list[bi * FOURH + k] + b_p[k]
                    )
                    if k < 3 * H:
                        gates[k] = _sigmoid(pre) if k < 2 * H else tanh(pre)
                    else:
                        gates[k] = _sigmoid(pre)
                for j in range(H):
                    var c_new = gates[H + j] * cp_p[bi * H + j] + gates[j] * gates[2 * H + j]
                    ct_p[bi * H + j] = c_new
                    ht_p[bi * H + j] = gates[3 * H + j] * tanh(c_new)
        else:
            var c = ctx.value()
            h_t.ensure_gpu(c, BATCH * H)
            c_t.ensure_gpu(c, BATCH * H)
            self._ix.ensure_gpu(c, BATCH * FOURH)
            self._hx.ensure_gpu(c, BATCH * FOURH)
            # ix = x @ W_ih ; hx = h_prev @ W_hh, then gates (WITH_CACHE=False).
            var x_v = TileTensor(x.dev.value(), row_major[BATCH, Self.IN_]())
            var hp_v = TileTensor(
                h_prev.dev.value(), row_major[BATCH, Self.HIDDEN]()
            )
            var Wih_v = TileTensor(
                self.W_ih.val.dev.value(), row_major[Self.IN_, FOURH]()
            )
            var Whh_v = TileTensor(
                self.W_hh.val.dev.value(), row_major[Self.HIDDEN, FOURH]()
            )
            var ix_v = TileTensor(self._ix.dev.value(), row_major[BATCH, FOURH]())
            var hx_v = TileTensor(self._hx.dev.value(), row_major[BATCH, FOURH]())
            max_matmul[target="gpu"](ix_v, x_v, Wih_v, c)
            max_matmul[target="gpu"](hx_v, hp_v, Whh_v, c)
            # cache unused (WITH_CACHE=False) — alias h_t's buffer as a dummy.
            comptime gk = _lstm_gate_fwd_kernel[BATCH, H, Self.CACHE_SIZE, False]
            comptime nblk = (BATCH * H + TPB - 1) // TPB
            var dummy = LayoutTensor[
                DT, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin
            ](h_t.dev.value())
            c.enqueue_function[gk](
                LayoutTensor[DT, Layout.row_major(BATCH, FOURH), MutAnyOrigin](
                    self._ix.dev.value()
                ),
                LayoutTensor[DT, Layout.row_major(BATCH, FOURH), MutAnyOrigin](
                    self._hx.dev.value()
                ),
                self.b.val.lt["gpu", Layout.row_major(FOURH)](),
                c_prev.lt["gpu", Layout.row_major(BATCH, H)](),
                h_t.lt["gpu", Layout.row_major(BATCH, H)](),
                c_t.lt["gpu", Layout.row_major(BATCH, H)](),
                dummy,
                grid_dim=(nblk,), block_dim=(TPB,),
            )

    def step_backward[target: StaticString, BATCH: Int](
        mut self,
        mut dh: Tensor,
        mut dc: Tensor,
        mut x: Tensor,
        mut h_prev: Tensor,
        mut c_prev: Tensor,
        mut cache: Tensor,
        mut dx: Tensor,
        mut dh_prev: Tensor,
        mut dc_prev: Tensor,
        ctx: Optional[DeviceContext] = None,
        dh_off: Int = 0,
        dc_off: Int = 0,
        x_off: Int = 0,
        h_prev_off: Int = 0,
        c_prev_off: Int = 0,
        cache_off: Int = 0,
        dx_off: Int = 0,
        dh_prev_off: Int = 0,
        dc_prev_off: Int = 0,
    ) raises:
        """One BPTT step. `dh`/`dc` are incoming grads w.r.t. h_t / c_t
        (pass dc=0 at the last timestep). Writes dx, dh_prev, dc_prev and
        ACCUMULATES into the cell's parameter grads. `dh_prev`/`dc_prev`
        must not alias `h_prev`/`c_prev`."""
        comptime H = Self.HIDDEN
        comptime FOURH = 4 * Self.HIDDEN

        comptime if target == "cpu":
            dx.ensure(dx_off + BATCH * Self.IN_)
            dh_prev.ensure(dh_prev_off + BATCH * H)
            dc_prev.ensure(dc_prev_off + BATCH * H)
            var cc_p = cache.data.unsafe_ptr() + cache_off
            var dh_p = dh.data.unsafe_ptr() + dh_off
            var dc_p = dc.data.unsafe_ptr() + dc_off
            var cp_p = c_prev.data.unsafe_ptr() + c_prev_off
            var dcp_p = dc_prev.data.unsafe_ptr() + dc_prev_off
            var dW_ih_p = self.W_ih.grd.data.unsafe_ptr()
            var dW_hh_p = self.W_hh.grd.data.unsafe_ptr()
            var db_p = self.b.grd.data.unsafe_ptr()
            var xv = TileTensor(
                x.data.unsafe_ptr() + x_off, row_major[BATCH, Self.IN_]()
            )
            var hp = TileTensor(
                h_prev.data.unsafe_ptr() + h_prev_off,
                row_major[BATCH, Self.HIDDEN](),
            )
            # Gate-grad math (unchanged) produces the combined per-(b,k) preact
            # gradient into a [BATCH, 4H] buffer `d_pre`, so the dW / dx / dh_prev
            # work below runs through BLAS like Linear's CPU backward. The
            # cell-state grad path `dcp = dc_total ⊙ f` stays scalar.
            var dpre_list = List[Scalar[DT]](length=BATCH * FOURH, fill=Scalar[DT](0))
            for bi in range(BATCH):
                for j in range(H):
                    var cbase = bi * Self.CACHE_SIZE
                    var i_v = cc_p[cbase + j]
                    var f_v = cc_p[cbase + H + j]
                    var g_v = cc_p[cbase + 2 * H + j]
                    var o_v = cc_p[cbase + 3 * H + j]
                    var tc = cc_p[cbase + 4 * H + j]
                    var dh_j = dh_p[bi * H + j]
                    var dc_j = dc_p[bi * H + j]
                    var do_post = dh_j * tc
                    var dc_total = dc_j + dh_j * o_v * (Scalar[DT](1.0) - tc * tc)
                    var df_post = dc_total * cp_p[bi * H + j]
                    var di_post = dc_total * g_v
                    var dg_post = dc_total * i_v
                    dcp_p[bi * H + j] = dc_total * f_v
                    var base = bi * FOURH
                    dpre_list[base + j]         = di_post * i_v * (Scalar[DT](1.0) - i_v)
                    dpre_list[base + H + j]     = df_post * f_v * (Scalar[DT](1.0) - f_v)
                    dpre_list[base + 2 * H + j] = dg_post * (Scalar[DT](1.0) - g_v * g_v)
                    dpre_list[base + 3 * H + j] = do_post * o_v * (Scalar[DT](1.0) - o_v)
            var dpre_tt = TileTensor(dpre_list, row_major[BATCH, FOURH]())

            # dW_ih += xᵀ @ d_pre, dW_hh += h_prevᵀ @ d_pre, db += Σ_b d_pre.
            # Transpose x / h_prev into contiguous [IN, BATCH] / [H, BATCH]
            # FIRST — reads cached x / h_prev BEFORE the dx / dh_prev matmuls
            # below clobber the aliased input slabs. Matmul into a temp then
            # accumulate.
            var xT_list = List[Scalar[DT]](length=Self.IN_ * BATCH, fill=Scalar[DT](0))
            for bi in range(BATCH):
                for j in range(Self.IN_):
                    xT_list[j * BATCH + bi] = xv[bi, j]
            var hT_list = List[Scalar[DT]](length=Self.HIDDEN * BATCH, fill=Scalar[DT](0))
            for bi in range(BATCH):
                for j in range(H):
                    hT_list[j * BATCH + bi] = hp[bi, j]
            var dWih_tmp_list = List[Scalar[DT]](length=Self.IN_ * FOURH, fill=Scalar[DT](0))
            var dWhh_tmp_list = List[Scalar[DT]](length=Self.HIDDEN * FOURH, fill=Scalar[DT](0))
            var xT_tt = TileTensor(xT_list, row_major[Self.IN_, BATCH]())
            var hT_tt = TileTensor(hT_list, row_major[Self.HIDDEN, BATCH]())
            var dWih_tmp_tt = TileTensor(dWih_tmp_list, row_major[Self.IN_, FOURH]())
            var dWhh_tmp_tt = TileTensor(dWhh_tmp_list, row_major[Self.HIDDEN, FOURH]())
            max_matmul[target="cpu"](dWih_tmp_tt, xT_tt, dpre_tt, None)
            max_matmul[target="cpu"](dWhh_tmp_tt, hT_tt, dpre_tt, None)
            for idx in range(Self.IN_ * FOURH):
                dW_ih_p[idx] += dWih_tmp_list[idx]
            for idx in range(Self.HIDDEN * FOURH):
                dW_hh_p[idx] += dWhh_tmp_list[idx]
            # db — cheap O(BATCH·4H) reduction; keep scalar.
            for k in range(FOURH):
                var sb: Scalar[DT] = 0.0
                for bi in range(BATCH):
                    sb += dpre_list[bi * FOURH + k]
                db_p[k] += sb

            # dx = d_pre @ W_ihᵀ, dh_prev = d_pre @ W_hhᵀ via BLAS. These WRITE
            # the slabs that x / h_prev alias — safe now (dW reads done).
            var dxv = TileTensor(
                dx.data.unsafe_ptr() + dx_off, row_major[BATCH, Self.IN_]()
            )
            var dhp = TileTensor(
                dh_prev.data.unsafe_ptr() + dh_prev_off,
                row_major[BATCH, Self.HIDDEN](),
            )
            var Wih_tt = TileTensor(self.W_ih.val.data, row_major[Self.IN_, FOURH]())
            var Whh_tt = TileTensor(self.W_hh.val.data, row_major[Self.HIDDEN, FOURH]())
            max_matmul[transpose_b=True, target="cpu"](dxv, dpre_tt, Wih_tt, None)
            max_matmul[transpose_b=True, target="cpu"](dhp, dpre_tt, Whh_tt, None)
        else:
            var c = ctx.value()
            dx.ensure_gpu(c, dx_off + BATCH * Self.IN_)
            dh_prev.ensure_gpu(c, dh_prev_off + BATCH * H)
            dc_prev.ensure_gpu(c, dc_prev_off + BATCH * H)
            self._ensure_dcomb_gpu(c, BATCH)
            self._xT.ensure_gpu(c, Self.IN_ * BATCH)
            self._hT.ensure_gpu(c, Self.HIDDEN * BATCH)
            self._dWih_tmp.ensure_gpu(c, Self.W_IH_SIZE)
            self._dWhh_tmp.ensure_gpu(c, Self.W_HH_SIZE)
            var dhb = dh.dev.value().create_sub_buffer[DT](dh_off, BATCH * H)
            var dcb = dc.dev.value().create_sub_buffer[DT](dc_off, BATCH * H)
            var cpb = c_prev.dev.value().create_sub_buffer[DT](c_prev_off, BATCH * H)
            var ccb = cache.dev.value().create_sub_buffer[DT](
                cache_off, BATCH * Self.CACHE_SIZE
            )
            var xb = x.dev.value().create_sub_buffer[DT](x_off, BATCH * Self.IN_)
            var hpb = h_prev.dev.value().create_sub_buffer[DT](h_prev_off, BATCH * H)
            var dxb = dx.dev.value().create_sub_buffer[DT](dx_off, BATCH * Self.IN_)
            var dhpb = dh_prev.dev.value().create_sub_buffer[DT](dh_prev_off, BATCH * H)
            var dcpb = dc_prev.dev.value().create_sub_buffer[DT](dc_prev_off, BATCH * H)

            # Phase 1: gate-grad math → d_comb [BATCH, 4H] + dc_prev.
            var dcomb = LayoutTensor[
                DT, Layout.row_major(BATCH, FOURH), MutAnyOrigin
            ](self._dcomb.dev.value())
            comptime gk = _lstm_gate_grad_kernel[BATCH, H, Self.CACHE_SIZE]
            comptime nbh = (BATCH * H + TPB - 1) // TPB
            c.enqueue_function[gk](
                LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin](dhb),
                LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin](dcb),
                LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin](cpb),
                LayoutTensor[DT, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin](ccb),
                LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin](dcpb),
                dcomb,
                grid_dim=(nbh,), block_dim=(TPB,),
            )

            # Capture xᵀ / h_prevᵀ BEFORE dx / dh_prev writes (may alias x /
            # h_prev under LSTMSeq's shared buffers).
            comptime txk = _lstm_transpose_kernel[BATCH, Self.IN_]
            comptime nbx = (BATCH * Self.IN_ + TPB - 1) // TPB
            c.enqueue_function[txk](
                LayoutTensor[DT, Layout.row_major(BATCH, Self.IN_), MutAnyOrigin](xb),
                self._xT.lt["gpu", Layout.row_major(Self.IN_, BATCH)](),
                grid_dim=(nbx,), block_dim=(TPB,),
            )
            comptime thk = _lstm_transpose_kernel[BATCH, H]
            c.enqueue_function[thk](
                LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin](hpb),
                self._hT.lt["gpu", Layout.row_major(H, BATCH)](),
                grid_dim=(nbh,), block_dim=(TPB,),
            )

            # dx = d_comb @ W_ihᵀ ; dh_prev = d_comb @ W_hhᵀ.
            var dcomb_tt = TileTensor(
                self._dcomb.dev.value(), row_major[BATCH, FOURH]()
            )
            var Wih_tt = TileTensor(
                self.W_ih.val.dev.value(), row_major[Self.IN_, FOURH]()
            )
            var Whh_tt = TileTensor(
                self.W_hh.val.dev.value(), row_major[Self.HIDDEN, FOURH]()
            )
            var dx_tt = TileTensor(dxb, row_major[BATCH, Self.IN_]())
            var dhp_tt = TileTensor(dhpb, row_major[BATCH, Self.HIDDEN]())
            max_matmul[transpose_b=True, target="gpu"](dx_tt, dcomb_tt, Wih_tt, c)
            max_matmul[transpose_b=True, target="gpu"](dhp_tt, dcomb_tt, Whh_tt, c)

            # dW_ih += xᵀ @ d_comb ; dW_hh += h_prevᵀ @ d_comb (temp + accumulate).
            var xT_tt = TileTensor(
                self._xT.dev.value(), row_major[Self.IN_, BATCH]()
            )
            var hT_tt = TileTensor(
                self._hT.dev.value(), row_major[Self.HIDDEN, BATCH]()
            )
            var dWih_tmp_tt = TileTensor(
                self._dWih_tmp.dev.value(), row_major[Self.IN_, FOURH]()
            )
            var dWhh_tmp_tt = TileTensor(
                self._dWhh_tmp.dev.value(), row_major[Self.HIDDEN, FOURH]()
            )
            max_matmul[target="gpu"](dWih_tmp_tt, xT_tt, dcomb_tt, c)
            max_matmul[target="gpu"](dWhh_tmp_tt, hT_tt, dcomb_tt, c)
            comptime aih = _lstm_accum_kernel[Self.W_IH_SIZE]
            c.enqueue_function[aih](
                self.W_ih.grd.lt["gpu", Layout.row_major(Self.W_IH_SIZE)](),
                self._dWih_tmp.lt["gpu", Layout.row_major(Self.W_IH_SIZE)](),
                grid_dim=((Self.W_IH_SIZE + TPB - 1) // TPB,), block_dim=(TPB,),
            )
            comptime ahh = _lstm_accum_kernel[Self.W_HH_SIZE]
            c.enqueue_function[ahh](
                self.W_hh.grd.lt["gpu", Layout.row_major(Self.W_HH_SIZE)](),
                self._dWhh_tmp.lt["gpu", Layout.row_major(Self.W_HH_SIZE)](),
                grid_dim=((Self.W_HH_SIZE + TPB - 1) // TPB,), block_dim=(TPB,),
            )

            # db += Σ_b d_comb (column reduction; not a GEMM).
            comptime bk = _lstm_db_kernel[BATCH, H]
            c.enqueue_function[bk](
                dcomb,
                self.b.grd.lt["gpu", Layout.row_major(FOURH)](),
                grid_dim=(FOURH,), block_dim=(TPB,),
            )

    # ------------------------------------------------------------------
    # polyak_from — soft-update all three weight slabs (mirror Linear).
    # ------------------------------------------------------------------

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        polyak_tensor[target, Self.W_IH_SIZE](
            self.W_ih.val, src.W_ih.val, tau, ctx
        )
        polyak_tensor[target, Self.W_HH_SIZE](
            self.W_hh.val, src.W_hh.val, tau, ctx
        )
        polyak_tensor[target, Self.B_SIZE](self.b.val, src.b.val, tau, ctx)
