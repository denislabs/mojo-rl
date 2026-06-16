"""LSTMSeq[VOCAB, HIDDEN, SEQ] — a fixed-length LSTM encoder as a standard
`Module` (nn's `nn.LSTM`-over-a-window analog).

`LSTMCell` is recurrent and exposes a step API (`step_forward`/
`step_backward`); its `Module.forward`/`vjp` raise. `LSTMSeq` wraps the
unroll + BPTT behind the standard `Module` interface so an LSTM LM is a
plain `Module` and trains through the generic `Trainer` /
`AutoregressiveTrainer` with no hand-written loop:

    Sequential[LSTMSeq[VOCAB, HIDDEN, SEQ], Tokenwise[SEQ, Linear[HIDDEN, VOCAB]]]

I/O matches the rest of the sequence stack (BATCH-major, SEQ folded into
the feature dim, per-token sub-vectors contiguous):
    forward : input  [BATCH, SEQ·VOCAB]  → output [BATCH, SEQ·HIDDEN]
    vjp     : grad_output [BATCH, SEQ·HIDDEN] → grad_input [BATCH, SEQ·VOCAB]

The cell wants a contiguous `[BATCH, ·]` per timestep, but the I/O is
BATCH-major, so each forward/backward transposes the sequence to
timestep-major `[SEQ, BATCH, ·]` once at the boundary (CPU: host loop;
GPU: one kernel), runs the contiguous-slice step loop, and transposes
back. The per-timestep h/c states + caches live in owned `Cache` fields
(lazy by BATCH) so BPTT can read them. `for_each_param` recurses into the
cell so the optimizer trains `W_ih`/`W_hh`/`b` (the default reflection
would skip the `cell` Module field).

CPU + GPU (the cell's steps support both). The GPU path is an *unfused*
RNN — SEQ sequential per-step kernel launches — so it's correct but slow
for long SEQ (same as a non-cuDNN RNN); keep SEQ modest. `mode ==
"input_only"` is unsupported (the cell's `step_backward` always
accumulates param grads); LM training only ever uses `mode == "all"`.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from ..constants import DT, TPB
from ..core import Initializer, AMPPolicy, NoAMP, Cache, ParamVisitor
from ..core.module import Module, typed_view, typed_view_mut, mptr
from ..core.tensor_pack import TensorPack
from ..core.target_storage import require_ctx, TargetStorage, assert_tag_for
from .lstm_cell import LSTMCell


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — sequence transpose (BATCH-major ↔ timestep-major) + add.
# ──────────────────────────────────────────────────────────────────────


def _bsd_to_sbd_kernel[
    BATCH: Int, SEQ: Int, D: Int
](
    src: UnsafePointer[Scalar[DT], MutAnyOrigin],
    dst: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    """`dst[t,b,d] = src[b,t,d]`. src `[BATCH, SEQ·D]`, dst `[SEQ, BATCH, D]`."""
    var idx = Int(global_idx.x)
    comptime total = BATCH * SEQ * D
    if idx >= total:
        return
    var b = idx // (SEQ * D)
    var rem = idx % (SEQ * D)
    var t = rem // D
    var d = rem % D
    dst[t * BATCH * D + b * D + d] = src[idx]


def _sbd_to_bsd_kernel[
    BATCH: Int, SEQ: Int, D: Int
](
    src: UnsafePointer[Scalar[DT], MutAnyOrigin],
    dst: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    """`dst[b,t,d] = src[t,b,d]`. src `[SEQ, BATCH, D]`, dst `[BATCH, SEQ·D]`."""
    var idx = Int(global_idx.x)
    comptime total = BATCH * SEQ * D
    if idx >= total:
        return
    var b = idx // (SEQ * D)
    var rem = idx % (SEQ * D)
    var t = rem // D
    var d = rem % D
    dst[idx] = src[t * BATCH * D + b * D + d]


def _add_kernel(
    a: UnsafePointer[Scalar[DT], MutAnyOrigin],
    b: UnsafePointer[Scalar[DT], MutAnyOrigin],
    dst: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n: Int,
):
    """`dst[i] = a[i] + b[i]`."""
    var i = Int(global_idx.x)
    if i < n:
        dst[i] = a[i] + b[i]


struct LSTMSeq[VOCAB: Int, HIDDEN: Int, SEQ: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.SEQ * Self.VOCAB)
    comptime OUT_DIM = Self.SEQ * Self.HIDDEN
    comptime Cell = LSTMCell[Self.VOCAB, Self.HIDDEN]
    comptime CACHE_SIZE = Self.Cell.CACHE_SIZE

    @staticmethod
    def display_label() -> String:
        return String("LSTMSeq")

    var cell: Self.Cell
    # Forward scratch (lazy by BATCH): timestep-major input + per-step h/c +
    # the cell's per-step backward cache.
    var x_seq: Cache["x_seq"]          # [SEQ, BATCH, VOCAB]
    var h_buf: Cache["h_buf"]          # [SEQ+1, BATCH, HIDDEN]
    var c_buf: Cache["c_buf"]          # [SEQ+1, BATCH, HIDDEN]
    var cache_buf: Cache["cache_buf"]  # [SEQ, BATCH, CACHE_SIZE]
    # Backward scratch.
    var dh_seq: Cache["dh_seq"]        # [SEQ, BATCH, HIDDEN]
    var dx_seq: Cache["dx_seq"]        # [SEQ, BATCH, VOCAB]
    var dh_recur: Cache["dh_recur"]    # [BATCH, HIDDEN] carry
    var dc_recur: Cache["dc_recur"]
    var dh_prev: Cache["dh_prev"]
    var dc_prev: Cache["dc_prev"]
    var dh_t: Cache["dh_t"]            # [BATCH, HIDDEN] working
    var ts: TargetStorage

    def __init__(out self):
        self.cell = Self.Cell()
        self.x_seq = Cache["x_seq"]()
        self.h_buf = Cache["h_buf"]()
        self.c_buf = Cache["c_buf"]()
        self.cache_buf = Cache["cache_buf"]()
        self.dh_seq = Cache["dh_seq"]()
        self.dx_seq = Cache["dx_seq"]()
        self.dh_recur = Cache["dh_recur"]()
        self.dc_recur = Cache["dc_recur"]()
        self.dh_prev = Cache["dh_prev"]()
        self.dc_prev = Cache["dc_prev"]()
        self.dh_t = Cache["dh_t"]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None,) raises -> Self:
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "LSTMSeq: target must be 'cpu' or 'gpu'"
        var m = Self()
        m.cell = Self.Cell.make[target, INIT](ctx=ctx)
        comptime if target == "cpu":
            m.ts = TargetStorage.make_cpu()
        else:
            m.ts = TargetStorage.make_gpu(
                require_ctx["LSTMSeq.make[target='gpu']"](ctx)
            )
        return m^

    # ----- scratch sizing -------------------------------------------------

    def _ensure[target: StaticString](mut self, batch: Int) raises:
        comptime H = Self.HIDDEN
        comptime V = Self.VOCAB
        comptime C = Self.CACHE_SIZE
        comptime if target == "cpu":
            self.x_seq.ensure_cpu(Self.SEQ * batch * V)
            self.h_buf.ensure_cpu((Self.SEQ + 1) * batch * H)
            self.c_buf.ensure_cpu((Self.SEQ + 1) * batch * H)
            self.cache_buf.ensure_cpu(Self.SEQ * batch * C)
            self.dh_seq.ensure_cpu(Self.SEQ * batch * H)
            self.dx_seq.ensure_cpu(Self.SEQ * batch * V)
            self.dh_recur.ensure_cpu(batch * H)
            self.dc_recur.ensure_cpu(batch * H)
            self.dh_prev.ensure_cpu(batch * H)
            self.dc_prev.ensure_cpu(batch * H)
            self.dh_t.ensure_cpu(batch * H)
        else:
            var ctx = self.ts.ctx.value()
            self.x_seq.ensure_gpu(ctx, Self.SEQ * batch * V)
            self.h_buf.ensure_gpu(ctx, (Self.SEQ + 1) * batch * H)
            self.c_buf.ensure_gpu(ctx, (Self.SEQ + 1) * batch * H)
            self.cache_buf.ensure_gpu(ctx, Self.SEQ * batch * C)
            self.dh_seq.ensure_gpu(ctx, Self.SEQ * batch * H)
            self.dx_seq.ensure_gpu(ctx, Self.SEQ * batch * V)
            self.dh_recur.ensure_gpu(ctx, batch * H)
            self.dc_recur.ensure_gpu(ctx, batch * H)
            self.dh_prev.ensure_gpu(ctx, batch * H)
            self.dc_prev.ensure_gpu(ctx, batch * H)
            self.dh_t.ensure_gpu(ctx, batch * H)

    # ----- Forward: unroll the cell over SEQ -----------------------------

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
        assert_tag_for["LSTMSeq", target](self.ts.target_tag)
        comptime H = Self.HIDDEN
        comptime V = Self.VOCAB
        comptime C = Self.CACHE_SIZE
        var input_v = inputs.tile[0, BATCH, Self.SEQ * V]()
        var output_v = typed_view_mut[BATCH, Self.SEQ * H](output)
        self._ensure[target](BATCH)

        var ip = mptr(input_v.ptr)
        var op = mptr(output_v.ptr)
        var xs = self.x_seq.target_ptr[target]()
        var hb = self.h_buf.target_ptr[target]()
        var cb = self.c_buf.target_ptr[target]()
        var cab = self.cache_buf.target_ptr[target]()

        comptime if target == "cpu":
            for i in range(BATCH * H):  # zero initial h_0 / c_0
                hb[i] = 0.0
                cb[i] = 0.0
            for b in range(BATCH):  # transpose input BSD → SBD
                for t in range(Self.SEQ):
                    for d in range(V):
                        xs[t * BATCH * V + b * V + d] = ip[
                            b * Self.SEQ * V + t * V + d
                        ]
        else:
            var ctx = self.ts.ctx.value()
            self.h_buf.dev.value().enqueue_fill(0.0)
            self.c_buf.dev.value().enqueue_fill(0.0)
            comptime nb = (BATCH * Self.SEQ * V + TPB - 1) // TPB
            ctx.enqueue_function[_bsd_to_sbd_kernel[BATCH, Self.SEQ, V]](
                ip, xs, grid_dim=nb, block_dim=TPB
            )

        for t in range(Self.SEQ):
            var x_t = TileTensor(xs + t * BATCH * V, row_major[BATCH, V]())
            var hp = TileTensor(hb + t * BATCH * H, row_major[BATCH, H]())
            var cp = TileTensor(cb + t * BATCH * H, row_major[BATCH, H]())
            var ht = TileTensor(hb + (t + 1) * BATCH * H, row_major[BATCH, H]())
            var ct = TileTensor(cb + (t + 1) * BATCH * H, row_major[BATCH, H]())
            var cc = TileTensor(cab + t * BATCH * C, row_major[BATCH, C]())
            self.cell.step_forward[target, BATCH](x_t, hp, cp, ht, ct, cc)

        # transpose h_buf[1:] (SBD) → output (BSD)
        comptime if target == "cpu":
            for b in range(BATCH):
                for t in range(Self.SEQ):
                    for d in range(H):
                        op[b * Self.SEQ * H + t * H + d] = hb[
                            (t + 1) * BATCH * H + b * H + d
                        ]
        else:
            var ctx = self.ts.ctx.value()
            comptime nb = (BATCH * Self.SEQ * H + TPB - 1) // TPB
            ctx.enqueue_function[_sbd_to_bsd_kernel[BATCH, Self.SEQ, H]](
                hb + BATCH * H, op, grid_dim=nb, block_dim=TPB
            )

    # ----- Backward: BPTT over SEQ ---------------------------------------

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
        comptime assert (
            mode == "all"
        ), "LSTMSeq supports mode='all' only (BPTT always accumulates params)"
        assert_tag_for["LSTMSeq", target](self.ts.target_tag)
        comptime H = Self.HIDDEN
        comptime V = Self.VOCAB
        comptime C = Self.CACHE_SIZE
        var grad_output_v = typed_view[BATCH, Self.SEQ * H](grad_output)
        var grad_input_v = grad_inputs.tile[0, BATCH, Self.SEQ * V]()
        self._ensure[target](BATCH)

        var gop = mptr(grad_output_v.ptr)
        var gip = mptr(grad_input_v.ptr)
        var xs = self.x_seq.target_ptr[target]()
        var hb = self.h_buf.target_ptr[target]()
        var cb = self.c_buf.target_ptr[target]()
        var cab = self.cache_buf.target_ptr[target]()
        var dhs = self.dh_seq.target_ptr[target]()
        var dxs = self.dx_seq.target_ptr[target]()
        var dh_rec = self.dh_recur.target_ptr[target]()
        var dc_rec = self.dc_recur.target_ptr[target]()
        var dh_pr = self.dh_prev.target_ptr[target]()
        var dc_pr = self.dc_prev.target_ptr[target]()
        var dh_tp = self.dh_t.target_ptr[target]()

        # transpose grad_output (BSD) → dh_seq (SBD); zero the recur carries.
        comptime if target == "cpu":
            for b in range(BATCH):
                for t in range(Self.SEQ):
                    for d in range(H):
                        dhs[t * BATCH * H + b * H + d] = gop[
                            b * Self.SEQ * H + t * H + d
                        ]
            for i in range(BATCH * H):
                dh_rec[i] = 0.0
                dc_rec[i] = 0.0
        else:
            var ctx = self.ts.ctx.value()
            comptime nb = (BATCH * Self.SEQ * H + TPB - 1) // TPB
            ctx.enqueue_function[_bsd_to_sbd_kernel[BATCH, Self.SEQ, H]](
                gop, dhs, grid_dim=nb, block_dim=TPB
            )
            self.dh_recur.dev.value().enqueue_fill(0.0)
            self.dc_recur.dev.value().enqueue_fill(0.0)

        for tt in range(Self.SEQ):
            var t = Self.SEQ - 1 - tt
            # dh_t = dh_seq[t] + dh_recur ; dc_t = dc_recur (passed directly).
            comptime if target == "cpu":
                for i in range(BATCH * H):
                    dh_tp[i] = dhs[t * BATCH * H + i] + dh_rec[i]
            else:
                var ctx = self.ts.ctx.value()
                comptime nb = (BATCH * H + TPB - 1) // TPB
                ctx.enqueue_function[_add_kernel](
                    dhs + t * BATCH * H, dh_rec, dh_tp, BATCH * H,
                    grid_dim=nb, block_dim=TPB,
                )

            var dh_tt = TileTensor(dh_tp, row_major[BATCH, H]())
            var dc_tt = TileTensor(dc_rec, row_major[BATCH, H]())
            var x_t = TileTensor(xs + t * BATCH * V, row_major[BATCH, V]())
            var hp = TileTensor(hb + t * BATCH * H, row_major[BATCH, H]())
            var cp = TileTensor(cb + t * BATCH * H, row_major[BATCH, H]())
            var cc = TileTensor(cab + t * BATCH * C, row_major[BATCH, C]())
            var dx_tt = TileTensor(dxs + t * BATCH * V, row_major[BATCH, V]())
            var dhp_tt = TileTensor(dh_pr, row_major[BATCH, H]())
            var dcp_tt = TileTensor(dc_pr, row_major[BATCH, H]())
            self.cell.step_backward[target, BATCH](
                dh_tt, dc_tt, x_t, hp, cp, cc, dx_tt, dhp_tt, dcp_tt
            )

            # carry: dh_recur = dh_prev, dc_recur = dc_prev.
            comptime if target == "cpu":
                for i in range(BATCH * H):
                    dh_rec[i] = dh_pr[i]
                    dc_rec[i] = dc_pr[i]
            else:
                var ctx = self.ts.ctx.value()
                ctx.enqueue_copy(
                    self.dh_recur.dev.value(), self.dh_prev.dev.value()
                )
                ctx.enqueue_copy(
                    self.dc_recur.dev.value(), self.dc_prev.dev.value()
                )

        # transpose dx_seq (SBD) → grad_input (BSD)
        comptime if target == "cpu":
            for b in range(BATCH):
                for t in range(Self.SEQ):
                    for d in range(V):
                        gip[b * Self.SEQ * V + t * V + d] = dxs[
                            t * BATCH * V + b * V + d
                        ]
        else:
            var ctx = self.ts.ctx.value()
            comptime nb = (BATCH * Self.SEQ * V + TPB - 1) // TPB
            ctx.enqueue_function[_sbd_to_bsd_kernel[BATCH, Self.SEQ, V]](
                dxs, gip, grid_dim=nb, block_dim=TPB
            )

    # ----- Param walkers (recurse into the cell) -------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["LSTMSeq", target](self.ts.target_tag)
        var sep = "." if prefix.byte_length() > 0 else ""
        self.cell.for_each_param[target, V](prefix + sep + "cell", visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["LSTMSeq", target](self.ts.target_tag)
        self.cell.zero_grad[target]()
