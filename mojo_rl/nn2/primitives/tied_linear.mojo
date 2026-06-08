"""TiedLinear[IN, OUT] — a bias-less Linear whose weight is *borrowed* from
another leaf (weight tying, nanoGPT's `lm_head.weight = wte.weight`).

The canonical use is the GPT LM head sharing the token embedding's table.
The embedding owns a `[VOCAB, EMBED]` weight; the LM head is the same map
applied transposed (`out = x @ W^T`). PyTorch ties by making both modules
reference one `nn.Parameter`; nn2 leaves own private `Param` storage, so
instead `TiedLinear`:

  * owns **no** weight `Param` (and no bias) — so reflection
    (`for_each_param`) finds nothing here and the optimizer collects the
    shared weight exactly once, via the *source* leaf;
  * holds **borrowed pointers** to the source weight's value + grad
    buffers (`src_val_ptr` / `src_grd_ptr`), wired once after construction
    via `tie_to` (the analog of `lm_head.weight = wte.weight` in __init__);
  * reads the source value transposed in forward / grad-input, and
    **accumulates** its weight gradient straight into the source grad
    buffer (`+=`) in `vjp`. Both leaves accumulating into one grad buffer
    is correct because nn2 grads are `+=` and `zero_grad` clears the
    buffer once per step (the source leaf owns + zeroes it).

Net effect: the standard `Trainer.train_gpu` / `train_step` works with
zero tying code in the loop — no grad-fold, no per-step copy. The source
weight stays in its `[OUT, IN]` ( = `[VOCAB, EMBED]` ) layout; the
transpose is encapsulated in the matmul orientation here.

Shapes (IN = EMBED, OUT = VOCAB; source weight is `[OUT, IN]`):
  * forward:        out[b,v]  = Σ_e x[b,e]·Wsrc[v,e]      (= x @ Wsrcᵀ)
  * vjp grad-input: dx[b,e]   = Σ_v dout[b,v]·Wsrc[v,e]   (= dout @ Wsrc)
  * vjp grad-weight: dWsrc[v,e] += Σ_b dout[b,v]·x[b,e]   (accumulate)

Pointer-stability rule: `tie_to` captures raw pointers, so it must run
*after* the owning model reaches its final home (e.g. inside the
`Trainer`) and after any model load. Re-wiring is cheap and idempotent.
GPU device pointers are stable for the buffer's lifetime; the source leaf
keeps the buffer alive. fp32 compute only (no AMP) for v1.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major
from linalg.matmul import matmul as max_matmul

from ..constants import DT, TPB
from ..core import Initializer, AMPPolicy, NoAMP
from ..core.module import Module, typed_view, typed_view_mut, mptr
from ..core.tensor_pack import TensorPack
from ..core.target_storage import require_ctx, TargetStorage, assert_tag_for


# ──────────────────────────────────────────────────────────────────────
# GPU param-grad kernel — accumulate dWsrc[v,e] += Σ_b dout[b,v]·x[b,e]
# into the borrowed source grad buffer. One thread per (v, e); serial over
# BATCH. Mirrors Embedding's `_embedding_grad_w_kernel` (the source leaf's
# own half of the tied weight uses the identical pattern), so the two
# accumulate symmetrically into the shared buffer.
# ──────────────────────────────────────────────────────────────────────


def _tied_grad_w_kernel[
    BATCH: Int, OUT: Int, IN: Int
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, OUT), MutAnyOrigin],
    cache_in: LayoutTensor[DT, Layout.row_major(BATCH, IN), MutAnyOrigin],
    grad_w: LayoutTensor[DT, Layout.row_major(OUT, IN), MutAnyOrigin],
):
    var gid = Int(global_idx.x)
    comptime total = OUT * IN
    if gid >= total:
        return
    var v = gid // IN
    var e = gid % IN
    var acc: Scalar[DT] = 0.0
    for b in range(BATCH):
        acc += rebind[Scalar[DT]](grad_output[b, v]) * rebind[Scalar[DT]](
            cache_in[b, e]
        )
    grad_w[v, e] = rebind[Scalar[DT]](grad_w[v, e]) + acc


struct TiedLinear[IN: Int, OUT: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.IN)
    comptime OUT_DIM = Self.OUT
    comptime W_SIZE = Self.IN * Self.OUT

    @staticmethod
    def display_label() -> String:
        return String("TiedLinear")

    # No owned weight/bias Param — the weight is borrowed (reflection finds
    # nothing here, so the optimizer never double-counts the shared weight).
    # `tie_to` sets these to the source weight's value + grad buffers.
    var src_val_ptr: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]
    var src_grd_ptr: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]

    # Forward-time alias of the orchestrator's input slab (no copy), read by
    # the param-grad pass. The combined `vjp` runs param-grad BEFORE
    # grad-input, so grad-input writing the (possibly aliased) slab can't
    # clobber the cache first — same invariant as Linear.
    var _cached_input_ptr: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]

    var ts: TargetStorage

    def __init__(out self):
        self.src_val_ptr = None
        self.src_grd_ptr = None
        self._cached_input_ptr = None
        self.ts = TargetStorage.make_uninit()

    # ----- Factory (INIT unused — no owned params) ------------------------

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None,) raises -> Self:
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "TiedLinear: target must be 'cpu' or 'gpu'"
        var t = Self()
        comptime if target == "cpu":
            t.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["TiedLinear.make[target='gpu']"](ctx)
            t.ts = TargetStorage.make_gpu(ctx_v)
        return t^

    # ----- Tie wiring -----------------------------------------------------

    def tie_to(
        mut self,
        val_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        grd_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ):
        """Point this head at the source weight's value + grad buffers
        (laid out `[OUT, IN]`). Call once after the owning model settles
        (and after any load); device pointers stay valid for the buffer's
        lifetime. The caller extracts the pointers target-appropriately."""
        self.src_val_ptr = val_ptr
        self.src_grd_ptr = grd_ptr

    def _src_val(self) raises -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        if not self.src_val_ptr:
            raise Error(
                "TiedLinear: not wired — call tie_to(...) before forward/vjp"
            )
        return self.src_val_ptr.value()

    # ----- Forward: out = x @ Wsrcᵀ ---------------------------------------

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
        comptime assert (
            POLICY.compute_dtype == DT
        ), "TiedLinear supports fp32 compute only"
        assert_tag_for["TiedLinear", target](self.ts.target_tag)
        var input_v = inputs.tile[0, BATCH, Self.IN]()
        var output_v = typed_view_mut[BATCH, Self.OUT](output)
        self._cached_input_ptr = input_v.ptr  # alias slab for backward

        var w_tt = TileTensor(self._src_val(), row_major[Self.OUT, Self.IN]())
        comptime if target == "cpu":
            # out[b,v] = Σ_e x[b,e]·Wsrc[v,e]  (transpose_b: contract IN).
            max_matmul[transpose_b=True, target="cpu"](
                output_v, input_v, w_tt, None
            )
        else:
            max_matmul[transpose_b=True, target="gpu"](
                output_v, input_v, w_tt, self.ts.ctx.value()
            )

    # ----- Backward (combined; Embedding-style single vjp) ----------------

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
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        comptime assert (
            POLICY.compute_dtype == DT
        ), "TiedLinear supports fp32 compute only"
        assert_tag_for["TiedLinear", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT](grad_output)
        var grad_input_v = grad_inputs.tile[0, BATCH, Self.IN]()
        var w_tt = TileTensor(self._src_val(), row_major[Self.OUT, Self.IN]())

        # ── (1) grad-weight FIRST (reads cached input slab) ───────────────
        # Must precede grad-input, which writes the (possibly aliased) slab.
        comptime if mode == "all":
            if not self.src_grd_ptr:
                raise Error("TiedLinear.vjp: not wired — call tie_to(...)")
            if not self._cached_input_ptr:
                raise Error("TiedLinear.vjp: no cached input — run forward")
            var x_ptr = self._cached_input_ptr.value()
            var gw_ptr = self.src_grd_ptr.value()
            comptime if target == "cpu":
                var go = grad_output_v
                var x = TileTensor(x_ptr, row_major[BATCH, Self.IN]())
                var grd = TileTensor(gw_ptr, row_major[Self.OUT, Self.IN]())
                for v in range(Self.OUT):
                    for e in range(Self.IN):
                        var acc: Scalar[DT] = 0.0
                        for b in range(BATCH):
                            acc += go[b, v] * x[b, e]
                        grd[v, e] += acc
            else:
                var ctx = self.ts.ctx.value()
                var go_lt = LayoutTensor[
                    DT, Layout.row_major(BATCH, Self.OUT), MutAnyOrigin
                ](grad_output_v.ptr)
                var x_lt = LayoutTensor[
                    DT, Layout.row_major(BATCH, Self.IN), MutAnyOrigin
                ](x_ptr)
                var gw_lt = LayoutTensor[
                    DT, Layout.row_major(Self.OUT, Self.IN), MutAnyOrigin
                ](gw_ptr)
                comptime total = Self.OUT * Self.IN
                comptime n_blocks = (total + TPB - 1) // TPB
                comptime kernel = _tied_grad_w_kernel[BATCH, Self.OUT, Self.IN]
                ctx.enqueue_function[kernel](
                    go_lt, x_lt, gw_lt, grid_dim=n_blocks, block_dim=TPB
                )

        # ── (2) grad-input = dout @ Wsrc  (always) ────────────────────────
        comptime if target == "cpu":
            max_matmul[target="cpu"](grad_input_v, grad_output_v, w_tt, None)
        else:
            max_matmul[target="gpu"](
                grad_input_v, grad_output_v, w_tt, self.ts.ctx.value()
            )
