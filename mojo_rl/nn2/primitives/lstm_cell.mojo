"""LSTMCell[IN_, HIDDEN] — PyTorch-equivalent LSTM cell for nn2.

Unlike a feed-forward `Module`, an LSTM threads TWO states (h, c) across
time and is trained with BPTT, so it exposes an explicit recurrent API
(`step_forward` / `step_backward` / `step_forward_no_cache`) rather than
the single-input/single-output `Module.forward`. The caller owns the
(h, c) state and a per-timestep cache buffer, and runs the BPTT loop
(see `examples/nn2/lstm/`). This matches the legacy `mojo_rl.nn` LSTMCell
and the nn2 `GRUCell` math conventions.

Parameters are nn2 `Param` fields, so `for_each_param` / `zero_grad`
work and the cell composes with nn2 `Adam` / checkpointing. The cell
still conforms to `Module` (for the optimizer's `M: Module` bound), but
its `Module.forward` / `vjp` raise — use the step API.

Each step method takes a `target` ("cpu" or "gpu"). The GPU path mirrors
the legacy kernels: forward is one block per sample (threads stride over
HIDDEN); backward is an input/state kernel (d_combined + dx + dh_prev +
dc_prev, with a block barrier) followed by three block-reduction kernels
that accumulate dW_ih / dW_hh / db across the batch.

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
from std.memory import alloc
from std.gpu import thread_idx, block_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major
from linalg.matmul import matmul as max_matmul

from ..constants import DT, TPB
from ..core import (
    Initializer,
    AMPPolicy,
    NoAMP,
    Param,
    ParamVisitor,
)
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import require_ctx, TargetStorage, assert_tag_for


@always_inline
def _sigmoid(x: Scalar[DT]) -> Scalar[DT]:
    if x >= 0:
        var e = exp(-x)
        return Scalar[DT](1.0) / (Scalar[DT](1.0) + e)
    var e = exp(x)
    return e / (Scalar[DT](1.0) + e)


# ──────────────────────────────────────────────────────────────────────
# GPU kernels (separate W_ih / W_hh / b buffers — nn2 Param layout).
# ──────────────────────────────────────────────────────────────────────


def _lstm_fwd_kernel[
    BATCH: Int, IN_: Int, H: Int, CACHE: Int, WITH_CACHE: Bool,
](
    x: LayoutTensor[DT, Layout.row_major(BATCH, IN_), MutAnyOrigin],
    W_ih: LayoutTensor[DT, Layout.row_major(IN_, 4 * H), MutAnyOrigin],
    W_hh: LayoutTensor[DT, Layout.row_major(H, 4 * H), MutAnyOrigin],
    b: LayoutTensor[DT, Layout.row_major(4 * H), MutAnyOrigin],
    h_prev: LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin],
    c_prev: LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin],
    h_t: LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin],
    c_t: LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, CACHE), MutAnyOrigin],
):
    """One block per sample; threads stride over j ∈ [0, H)."""
    var bi = Int(block_idx.x)
    if bi >= BATCH:
        return
    var j = Int(thread_idx.x)
    while j < H:
        var i_pre = Scalar[DT](0)
        var f_pre = Scalar[DT](0)
        var g_pre = Scalar[DT](0)
        var o_pre = Scalar[DT](0)
        for jj in range(IN_):
            var xv = rebind[Scalar[DT]](x[bi, jj])
            i_pre += xv * rebind[Scalar[DT]](W_ih[jj, j])
            f_pre += xv * rebind[Scalar[DT]](W_ih[jj, H + j])
            g_pre += xv * rebind[Scalar[DT]](W_ih[jj, 2 * H + j])
            o_pre += xv * rebind[Scalar[DT]](W_ih[jj, 3 * H + j])
        for jj in range(H):
            var hv = rebind[Scalar[DT]](h_prev[bi, jj])
            i_pre += hv * rebind[Scalar[DT]](W_hh[jj, j])
            f_pre += hv * rebind[Scalar[DT]](W_hh[jj, H + j])
            g_pre += hv * rebind[Scalar[DT]](W_hh[jj, 2 * H + j])
            o_pre += hv * rebind[Scalar[DT]](W_hh[jj, 3 * H + j])
        i_pre += rebind[Scalar[DT]](b[j])
        f_pre += rebind[Scalar[DT]](b[H + j])
        g_pre += rebind[Scalar[DT]](b[2 * H + j])
        o_pre += rebind[Scalar[DT]](b[3 * H + j])

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
        j += TPB


def _lstm_bwd_input_kernel[
    BATCH: Int, IN_: Int, H: Int, CACHE: Int,
](
    dh: LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin],
    dc: LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin],
    c_prev: LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin],
    W_ih: LayoutTensor[DT, Layout.row_major(IN_, 4 * H), MutAnyOrigin],
    W_hh: LayoutTensor[DT, Layout.row_major(H, 4 * H), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, CACHE), MutAnyOrigin],
    dx: LayoutTensor[DT, Layout.row_major(BATCH, IN_), MutAnyOrigin],
    dh_prev: LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin],
    dc_prev: LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin],
    d_comb: LayoutTensor[DT, Layout.row_major(BATCH, 4 * H), MutAnyOrigin],
):
    """One block per sample. Phase 1: d_comb + dc_prev. Barrier. Phase 2:
    dx = d_comb @ W_ih^T. Phase 3: dh_prev = d_comb @ W_hh^T."""
    var bi = Int(block_idx.x)
    if bi >= BATCH:
        return
    var one = Scalar[DT](1.0)

    var j = Int(thread_idx.x)
    while j < H:
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
        j += TPB

    block.barrier()

    var jx = Int(thread_idx.x)
    while jx < IN_:
        var acc = Scalar[DT](0)
        for k in range(4 * H):
            acc += rebind[Scalar[DT]](d_comb[bi, k]) * rebind[Scalar[DT]](
                W_ih[jx, k]
            )
        dx[bi, jx] = acc
        jx += TPB

    var jh = Int(thread_idx.x)
    while jh < H:
        var acc = Scalar[DT](0)
        for k in range(4 * H):
            acc += rebind[Scalar[DT]](d_comb[bi, k]) * rebind[Scalar[DT]](
                W_hh[jh, k]
            )
        dh_prev[bi, jh] = acc
        jh += TPB


def _lstm_dWih_kernel[
    BATCH: Int, IN_: Int, H: Int,
](
    x: LayoutTensor[DT, Layout.row_major(BATCH, IN_), MutAnyOrigin],
    d_comb: LayoutTensor[DT, Layout.row_major(BATCH, 4 * H), MutAnyOrigin],
    dW_ih: LayoutTensor[DT, Layout.row_major(IN_, 4 * H), MutAnyOrigin],
):
    """Accumulate dW_ih += xᵀ · d_comb. Grid (IN_, 4H), block-reduce over BATCH."""
    var j_in = Int(block_idx.x)
    var k = Int(block_idx.y)
    if j_in >= IN_ or k >= 4 * H:
        return
    var my = Scalar[DT](0)
    var b = Int(thread_idx.x)
    while b < BATCH:
        my += rebind[Scalar[DT]](x[b, j_in]) * rebind[Scalar[DT]](d_comb[b, k])
        b += TPB
    var total = block.sum[block_size=TPB, broadcast=False](val=my)
    if Int(thread_idx.x) == 0:
        dW_ih[j_in, k] = rebind[Scalar[DT]](dW_ih[j_in, k]) + total[0]


def _lstm_dWhh_kernel[
    BATCH: Int, H: Int,
](
    h_prev: LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin],
    d_comb: LayoutTensor[DT, Layout.row_major(BATCH, 4 * H), MutAnyOrigin],
    dW_hh: LayoutTensor[DT, Layout.row_major(H, 4 * H), MutAnyOrigin],
):
    """Accumulate dW_hh += h_prevᵀ · d_comb. Grid (H, 4H)."""
    var j_in = Int(block_idx.x)
    var k = Int(block_idx.y)
    if j_in >= H or k >= 4 * H:
        return
    var my = Scalar[DT](0)
    var b = Int(thread_idx.x)
    while b < BATCH:
        my += rebind[Scalar[DT]](h_prev[b, j_in]) * rebind[Scalar[DT]](
            d_comb[b, k]
        )
        b += TPB
    var total = block.sum[block_size=TPB, broadcast=False](val=my)
    if Int(thread_idx.x) == 0:
        dW_hh[j_in, k] = rebind[Scalar[DT]](dW_hh[j_in, k]) + total[0]


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

    var ts: TargetStorage
    var W_ih: Param["W_ih", True,  Self.W_IH_SIZE]
    var W_hh: Param["W_hh", True,  Self.W_HH_SIZE]
    var b:    Param["b",    False, Self.B_SIZE]
    var _dcomb_dev: Optional[DeviceBuffer[DT]]  # GPU d_combined scratch
    var _dcomb_n: Int

    # ------------------------------------------------------------------
    # Defaultable + factories.
    # ------------------------------------------------------------------

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()
        self.W_ih = Param["W_ih", True,  Self.W_IH_SIZE]()
        self.W_hh = Param["W_hh", True,  Self.W_HH_SIZE]()
        self.b    = Param["b",    False, Self.B_SIZE]()
        self._dcomb_dev = None
        self._dcomb_n = 0

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory."""
        comptime assert target == "cpu" or target == "gpu", (
            "LSTMCell: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        comptime if target == "cpu":
            m.ts = TargetStorage.make_cpu()
            m.W_ih = Param["W_ih", True,  Self.W_IH_SIZE].make_cpu()
            m.W_hh = Param["W_hh", True,  Self.W_HH_SIZE].make_cpu()
            m.b    = Param["b",    False, Self.B_SIZE].make_cpu()
            INIT.init_weight(
                m.W_ih.value_unsafe_ptr_cpu(),
                Self.W_IH_SIZE, Self.IN_, 4 * Self.HIDDEN,
            )
            INIT.init_weight(
                m.W_hh.value_unsafe_ptr_cpu(),
                Self.W_HH_SIZE, Self.HIDDEN, 4 * Self.HIDDEN,
            )
            INIT.init_bias(m.b.value_unsafe_ptr_cpu(), Self.B_SIZE)
        else:
            var ctx_v = require_ctx["LSTMCell.make[target='gpu']"](ctx)
            m.ts = TargetStorage.make_gpu(ctx_v)
            m.W_ih = Param["W_ih", True,  Self.W_IH_SIZE].make_gpu(ctx_v)
            m.W_hh = Param["W_hh", True,  Self.W_HH_SIZE].make_gpu(ctx_v)
            m.b    = Param["b",    False, Self.B_SIZE].make_gpu(ctx_v)
            Self._gpu_init_param[Self.W_IH_SIZE, Self.IN_, 4 * Self.HIDDEN, INIT](
                ctx_v, m.W_ih.val.dev.value(), is_bias=False
            )
            Self._gpu_init_param[Self.W_HH_SIZE, Self.HIDDEN, 4 * Self.HIDDEN, INIT](
                ctx_v, m.W_hh.val.dev.value(), is_bias=False
            )
            Self._gpu_init_param[Self.B_SIZE, 0, 0, INIT](
                ctx_v, m.b.val.dev.value(), is_bias=True
            )
            m._dcomb_dev = ctx_v.enqueue_create_buffer[DT](1)
        return m^

    @staticmethod
    def _gpu_init_param[
        N: Int, FAN_IN: Int, FAN_OUT: Int, INIT: Initializer,
    ](ctx: DeviceContext, dst: DeviceBuffer[DT], is_bias: Bool) raises:
        var host = List[Scalar[DT]](length=N, fill=0.0)
        var hp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](host.unsafe_ptr())
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

    # ------------------------------------------------------------------
    # for_each_param / zero_grad: inherited from the Module default (S1,
    # 2026-06-07) — the default reflection-walks IsParam fields, replacing
    # the old pure-`*_auto` overrides. nn2 Adam / zero_grad / checkpoint
    # work unchanged.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Module conformance — recurrent cell uses the step API instead.
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
        raise Error(
            "LSTMCell is recurrent — use step_forward/step_backward "
            "(see examples/nn2/lstm), not Module.forward"
        )

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
        raise Error(
            "LSTMCell is recurrent — use step_backward "
            "(see examples/nn2/lstm), not Module.vjp"
        )

    # ------------------------------------------------------------------
    # GPU scratch.
    # ------------------------------------------------------------------

    def _ensure_dcomb_gpu(mut self, batch: Int) raises:
        var needed = batch * 4 * Self.HIDDEN
        if self._dcomb_n < needed:
            self._dcomb_dev = self.ts.ctx.value().enqueue_create_buffer[DT](needed)
            self._dcomb_n = needed

    # ------------------------------------------------------------------
    # Recurrent step API.
    # ------------------------------------------------------------------

    def step_forward[target: StaticString, BATCH: Int](
        mut self,
        x: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        h_prev: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        c_prev: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut h_t: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut c_t: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut cache: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        """One LSTM step; writes h_t, c_t, and the backward cache
        ([i | f | g | o | tanh_c], 5·H wide)."""
        assert_tag_for["LSTMCell", target](self.ts.target_tag)
        comptime H = Self.HIDDEN
        comptime FOURH = 4 * Self.HIDDEN

        comptime if target == "cpu":
            var xv = typed_view[BATCH, Self.IN_](x)
            var hp = typed_view[BATCH, Self.HIDDEN](h_prev)
            var cp = typed_view[BATCH, Self.HIDDEN](c_prev)
            var ht = typed_view_mut[BATCH, Self.HIDDEN](h_t)
            var ct = typed_view_mut[BATCH, Self.HIDDEN](c_t)
            var cc = typed_view_mut[BATCH, Self.CACHE_SIZE](cache)
            var W_ih_p = self.W_ih.value_unsafe_ptr_cpu()
            var W_hh_p = self.W_hh.value_unsafe_ptr_cpu()
            var b_p = self.b.value_unsafe_ptr_cpu()
            # Gate pre-activations via BLAS (Apple Accelerate), mirroring
            # Linear's CPU path — was naive BATCH·(IN+H)·4H scalar dots.
            #   ix = x @ W_ih  [BATCH, 4H],  hx = h_prev @ W_hh  [BATCH, 4H].
            # The gate nonlinearities below stay scalar (O(BATCH·H)).
            var ix_buf = alloc[Scalar[DT]](BATCH * FOURH)
            var hx_buf = alloc[Scalar[DT]](BATCH * FOURH)
            var ix_tt = TileTensor(ix_buf, row_major[BATCH, FOURH]())
            var hx_tt = TileTensor(hx_buf, row_major[BATCH, FOURH]())
            var Wih_tt = TileTensor(W_ih_p, row_major[Self.IN_, FOURH]())
            var Whh_tt = TileTensor(W_hh_p, row_major[Self.HIDDEN, FOURH]())
            max_matmul[target="cpu"](ix_tt, xv, Wih_tt, None)
            max_matmul[target="cpu"](hx_tt, hp, Whh_tt, None)
            for bi in range(BATCH):
                for k in range(FOURH):
                    var pre: Scalar[DT] = (
                        ix_buf[bi * FOURH + k] + hx_buf[bi * FOURH + k] + b_p[k]
                    )
                    var act: Scalar[DT]
                    if k < 3 * H:
                        act = _sigmoid(pre) if k < 2 * H else tanh(pre)
                    else:
                        act = _sigmoid(pre)
                    cc[bi, k] = act
                for j in range(H):
                    var i_v = cc[bi, j]
                    var f_v = cc[bi, H + j]
                    var g_v = cc[bi, 2 * H + j]
                    var o_v = cc[bi, 3 * H + j]
                    var c_new = f_v * cp[bi, j] + i_v * g_v
                    var tc = tanh(c_new)
                    ct[bi, j] = c_new
                    ht[bi, j] = o_v * tc
                    cc[bi, 4 * H + j] = tc
            hx_buf.free()
            ix_buf.free()
        else:
            var ctx = self.ts.ctx.value()
            var x_lt = LayoutTensor[DT, Layout.row_major(BATCH, Self.IN_), MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](x.ptr)
            )
            var hp_lt = LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](h_prev.ptr)
            )
            var cp_lt = LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](c_prev.ptr)
            )
            var ht_lt = LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](h_t.ptr)
            )
            var ct_lt = LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](c_t.ptr)
            )
            var cc_lt = LayoutTensor[DT, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](cache.ptr)
            )
            var wih = LayoutTensor[DT, Layout.row_major(Self.IN_, FOURH), MutAnyOrigin](self.W_ih.val.dev.value())
            var whh = LayoutTensor[DT, Layout.row_major(H, FOURH), MutAnyOrigin](self.W_hh.val.dev.value())
            var bb = LayoutTensor[DT, Layout.row_major(FOURH), MutAnyOrigin](self.b.val.dev.value())
            comptime kern = _lstm_fwd_kernel[BATCH, Self.IN_, H, Self.CACHE_SIZE, True]
            ctx.enqueue_function[kern](
                x_lt, wih, whh, bb, hp_lt, cp_lt, ht_lt, ct_lt, cc_lt,
                grid_dim=(BATCH,), block_dim=(TPB,),
            )

    def step_forward_no_cache[target: StaticString, BATCH: Int](
        mut self,
        x: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        h_prev: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        c_prev: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut h_t: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut c_t: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        """Inference step (no cache) — for eval / sampling."""
        assert_tag_for["LSTMCell", target](self.ts.target_tag)
        comptime H = Self.HIDDEN
        comptime FOURH = 4 * Self.HIDDEN

        comptime if target == "cpu":
            var xv = typed_view[BATCH, Self.IN_](x)
            var hp = typed_view[BATCH, Self.HIDDEN](h_prev)
            var cp = typed_view[BATCH, Self.HIDDEN](c_prev)
            var ht = typed_view_mut[BATCH, Self.HIDDEN](h_t)
            var ct = typed_view_mut[BATCH, Self.HIDDEN](c_t)
            var W_ih_p = self.W_ih.value_unsafe_ptr_cpu()
            var W_hh_p = self.W_hh.value_unsafe_ptr_cpu()
            var b_p = self.b.value_unsafe_ptr_cpu()
            # BLAS gate pre-activations (see step_forward).
            var ix_buf = alloc[Scalar[DT]](BATCH * FOURH)
            var hx_buf = alloc[Scalar[DT]](BATCH * FOURH)
            var ix_tt = TileTensor(ix_buf, row_major[BATCH, FOURH]())
            var hx_tt = TileTensor(hx_buf, row_major[BATCH, FOURH]())
            var Wih_tt = TileTensor(W_ih_p, row_major[Self.IN_, FOURH]())
            var Whh_tt = TileTensor(W_hh_p, row_major[Self.HIDDEN, FOURH]())
            max_matmul[target="cpu"](ix_tt, xv, Wih_tt, None)
            max_matmul[target="cpu"](hx_tt, hp, Whh_tt, None)
            for bi in range(BATCH):
                var gates = InlineArray[Scalar[DT], 4 * Self.HIDDEN](fill=0.0)
                for k in range(FOURH):
                    var pre: Scalar[DT] = (
                        ix_buf[bi * FOURH + k] + hx_buf[bi * FOURH + k] + b_p[k]
                    )
                    if k < 3 * H:
                        gates[k] = _sigmoid(pre) if k < 2 * H else tanh(pre)
                    else:
                        gates[k] = _sigmoid(pre)
                for j in range(H):
                    var c_new = gates[H + j] * cp[bi, j] + gates[j] * gates[2 * H + j]
                    ct[bi, j] = c_new
                    ht[bi, j] = gates[3 * H + j] * tanh(c_new)
            hx_buf.free()
            ix_buf.free()
        else:
            var ctx = self.ts.ctx.value()
            var x_lt = LayoutTensor[DT, Layout.row_major(BATCH, Self.IN_), MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](x.ptr)
            )
            var hp_lt = LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](h_prev.ptr)
            )
            var cp_lt = LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](c_prev.ptr)
            )
            var ht_lt = LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](h_t.ptr)
            )
            var ct_lt = LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](c_t.ptr)
            )
            var wih = LayoutTensor[DT, Layout.row_major(Self.IN_, FOURH), MutAnyOrigin](self.W_ih.val.dev.value())
            var whh = LayoutTensor[DT, Layout.row_major(H, FOURH), MutAnyOrigin](self.W_hh.val.dev.value())
            var bb = LayoutTensor[DT, Layout.row_major(FOURH), MutAnyOrigin](self.b.val.dev.value())
            # Reuse the fused kernel with WITH_CACHE=False (cache view unused).
            var dummy = LayoutTensor[DT, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](h_t.ptr)
            )
            comptime kern = _lstm_fwd_kernel[BATCH, Self.IN_, H, Self.CACHE_SIZE, False]
            ctx.enqueue_function[kern](
                x_lt, wih, whh, bb, hp_lt, cp_lt, ht_lt, ct_lt, dummy,
                grid_dim=(BATCH,), block_dim=(TPB,),
            )

    def step_backward[target: StaticString, BATCH: Int](
        mut self,
        dh: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        dc: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        x: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        h_prev: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        c_prev: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        cache: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut dx: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut dh_prev: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut dc_prev: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        """One BPTT step. `dh`/`dc` are incoming grads w.r.t. h_t / c_t
        (pass dc=0 at the last timestep). Writes dx, dh_prev, dc_prev and
        ACCUMULATES into the cell's parameter grads. `dh_prev`/`dc_prev`
        must not alias `h_prev`/`c_prev`."""
        assert_tag_for["LSTMCell", target](self.ts.target_tag)
        comptime H = Self.HIDDEN
        comptime FOURH = 4 * Self.HIDDEN

        comptime if target == "cpu":
            var dh_v = typed_view[BATCH, Self.HIDDEN](dh)
            var dc_v = typed_view[BATCH, Self.HIDDEN](dc)
            var xv = typed_view[BATCH, Self.IN_](x)
            var hp = typed_view[BATCH, Self.HIDDEN](h_prev)
            var cp = typed_view[BATCH, Self.HIDDEN](c_prev)
            var cc = typed_view[BATCH, Self.CACHE_SIZE](cache)
            var dxv = typed_view_mut[BATCH, Self.IN_](dx)
            var dhp = typed_view_mut[BATCH, Self.HIDDEN](dh_prev)
            var dcp = typed_view_mut[BATCH, Self.HIDDEN](dc_prev)
            var W_ih_p = self.W_ih.value_unsafe_ptr_cpu()
            var W_hh_p = self.W_hh.value_unsafe_ptr_cpu()
            var dW_ih_p = self.W_ih.grad_unsafe_ptr_cpu()
            var dW_hh_p = self.W_hh.grad_unsafe_ptr_cpu()
            var db_p = self.b.grad_unsafe_ptr_cpu()
            # Gate-grad math (unchanged) produces the combined per-(b,k) preact
            # gradient. Write it into a [BATCH, 4H] buffer `d_pre` so the dW /
            # dx / dh_prev work below can run through BLAS like Linear's CPU
            # backward (was naive BATCH·(IN+H)·4H scalar loops). The cell-state
            # grad path `dcp = dc_total ⊙ f` stays scalar (O(BATCH·H)).
            var dpre_buf = alloc[Scalar[DT]](BATCH * FOURH)
            for bi in range(BATCH):
                for j in range(H):
                    var i_v = cc[bi, j]
                    var f_v = cc[bi, H + j]
                    var g_v = cc[bi, 2 * H + j]
                    var o_v = cc[bi, 3 * H + j]
                    var tc = cc[bi, 4 * H + j]
                    var dh_j = dh_v[bi, j]
                    var dc_j = dc_v[bi, j]
                    var do_post = dh_j * tc
                    var dc_total = dc_j + dh_j * o_v * (Scalar[DT](1.0) - tc * tc)
                    var df_post = dc_total * cp[bi, j]
                    var di_post = dc_total * g_v
                    var dg_post = dc_total * i_v
                    dcp[bi, j] = dc_total * f_v
                    var base = bi * FOURH
                    dpre_buf[base + j]         = di_post * i_v * (Scalar[DT](1.0) - i_v)
                    dpre_buf[base + H + j]     = df_post * f_v * (Scalar[DT](1.0) - f_v)
                    dpre_buf[base + 2 * H + j] = dg_post * (Scalar[DT](1.0) - g_v * g_v)
                    dpre_buf[base + 3 * H + j] = do_post * o_v * (Scalar[DT](1.0) - o_v)
            var dpre_tt = TileTensor(dpre_buf, row_major[BATCH, FOURH]())

            # ── dW_ih += xᵀ @ d_pre, dW_hh += h_prevᵀ @ d_pre, db += Σ_b d_pre.
            # Transpose x / h_prev into contiguous [IN, BATCH] / [H, BATCH]
            # FIRST — this reads the cached x / h_prev BEFORE the dx / dh_prev
            # matmuls below clobber the aliased input slabs (leaf
            # backward-order invariant). Matmul into a temp then accumulate.
            var xT_buf = alloc[Scalar[DT]](Self.IN_ * BATCH)
            for bi in range(BATCH):
                for j in range(Self.IN_):
                    xT_buf[j * BATCH + bi] = xv[bi, j]
            var hT_buf = alloc[Scalar[DT]](Self.HIDDEN * BATCH)
            for bi in range(BATCH):
                for j in range(H):
                    hT_buf[j * BATCH + bi] = hp[bi, j]
            var dWih_tmp = alloc[Scalar[DT]](Self.IN_ * FOURH)
            var dWhh_tmp = alloc[Scalar[DT]](Self.HIDDEN * FOURH)
            var xT_tt = TileTensor(xT_buf, row_major[Self.IN_, BATCH]())
            var hT_tt = TileTensor(hT_buf, row_major[Self.HIDDEN, BATCH]())
            var dWih_tmp_tt = TileTensor(dWih_tmp, row_major[Self.IN_, FOURH]())
            var dWhh_tmp_tt = TileTensor(dWhh_tmp, row_major[Self.HIDDEN, FOURH]())
            max_matmul[target="cpu"](dWih_tmp_tt, xT_tt, dpre_tt, None)
            max_matmul[target="cpu"](dWhh_tmp_tt, hT_tt, dpre_tt, None)
            for idx in range(Self.IN_ * FOURH):
                dW_ih_p[idx] += dWih_tmp[idx]
            for idx in range(Self.HIDDEN * FOURH):
                dW_hh_p[idx] += dWhh_tmp[idx]
            # db — cheap O(BATCH·4H) reduction; keep scalar.
            for k in range(FOURH):
                var sb: Scalar[DT] = 0.0
                for bi in range(BATCH):
                    sb += dpre_buf[bi * FOURH + k]
                db_p[k] += sb
            dWhh_tmp.free()
            dWih_tmp.free()
            hT_buf.free()
            xT_buf.free()

            # ── dx = d_pre @ W_ihᵀ, dh_prev = d_pre @ W_hhᵀ via BLAS. These
            # WRITE the slabs that x / h_prev alias — safe now (dW reads done).
            var Wih_tt = TileTensor(W_ih_p, row_major[Self.IN_, FOURH]())
            var Whh_tt = TileTensor(W_hh_p, row_major[Self.HIDDEN, FOURH]())
            max_matmul[transpose_b=True, target="cpu"](dxv, dpre_tt, Wih_tt, None)
            max_matmul[transpose_b=True, target="cpu"](dhp, dpre_tt, Whh_tt, None)
            dpre_buf.free()
        else:
            var ctx = self.ts.ctx.value()
            self._ensure_dcomb_gpu(BATCH)
            var dh_lt = LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](dh.ptr)
            )
            var dc_lt = LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](dc.ptr)
            )
            var x_lt = LayoutTensor[DT, Layout.row_major(BATCH, Self.IN_), MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](x.ptr)
            )
            var hp_lt = LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](h_prev.ptr)
            )
            var cp_lt = LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](c_prev.ptr)
            )
            var cc_lt = LayoutTensor[DT, Layout.row_major(BATCH, Self.CACHE_SIZE), MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](cache.ptr)
            )
            var dx_lt = LayoutTensor[DT, Layout.row_major(BATCH, Self.IN_), MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](dx.ptr)
            )
            var dhp_lt = LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](dh_prev.ptr)
            )
            var dcp_lt = LayoutTensor[DT, Layout.row_major(BATCH, H), MutAnyOrigin](
                rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](dc_prev.ptr)
            )
            var dcomb = LayoutTensor[DT, Layout.row_major(BATCH, FOURH), MutAnyOrigin](
                self._dcomb_dev.value()
            )
            var wih = LayoutTensor[DT, Layout.row_major(Self.IN_, FOURH), MutAnyOrigin](self.W_ih.val.dev.value())
            var whh = LayoutTensor[DT, Layout.row_major(H, FOURH), MutAnyOrigin](self.W_hh.val.dev.value())
            var dwih = LayoutTensor[DT, Layout.row_major(Self.IN_, FOURH), MutAnyOrigin](self.W_ih.grd.dev.value())
            var dwhh = LayoutTensor[DT, Layout.row_major(H, FOURH), MutAnyOrigin](self.W_hh.grd.dev.value())
            var dbb = LayoutTensor[DT, Layout.row_major(FOURH), MutAnyOrigin](self.b.grd.dev.value())

            comptime ik = _lstm_bwd_input_kernel[BATCH, Self.IN_, H, Self.CACHE_SIZE]
            ctx.enqueue_function[ik](
                dh_lt, dc_lt, cp_lt, wih, whh, cc_lt, dx_lt, dhp_lt, dcp_lt, dcomb,
                grid_dim=(BATCH,), block_dim=(TPB,),
            )
            comptime wk = _lstm_dWih_kernel[BATCH, Self.IN_, H]
            ctx.enqueue_function[wk](
                x_lt, dcomb, dwih, grid_dim=(Self.IN_, FOURH), block_dim=(TPB,),
            )
            comptime hk = _lstm_dWhh_kernel[BATCH, H]
            ctx.enqueue_function[hk](
                hp_lt, dcomb, dwhh, grid_dim=(H, FOURH), block_dim=(TPB,),
            )
            comptime bk = _lstm_db_kernel[BATCH, H]
            ctx.enqueue_function[bk](
                dcomb, dbb, grid_dim=(FOURH,), block_dim=(TPB,),
            )
