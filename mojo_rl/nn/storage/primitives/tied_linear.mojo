"""TiedLinear[IN, OUT] — a bias-less Linear whose weight is *borrowed* from
another leaf (weight tying, nanoGPT's `lm_head.weight = wte.weight`).

Transformed from legacy `nn.primitives.TiedLinear` onto the storage surface.
The canonical use is the GPT LM head sharing the token embedding's table: the
embedding owns a `[VOCAB, EMBED]` weight; the LM head is the same map applied
transposed (`out = x @ Wᵀ`). PyTorch ties by making both modules reference one
`nn.Parameter`; storage leaves own private `Param` storage, so instead
`TiedLinear`:

  * owns **no** weight `Param` (and no bias) — so reflection
    (`for_each_param`) finds nothing here (no `IsParam` field) and the
    optimizer / checkpoint walk collects the shared weight EXACTLY ONCE, via
    the *source* leaf (the embedding). `for_each_param`/`zero_grad` therefore
    inherit the Module reflection defaults, which reflect to a no-op here;
  * holds a **borrowed reference to the source weight's storage cells**
    (`src_val` / `src_grd`), wired once after construction via `tie_to`
    (the analog of `lm_head.weight = wte.weight` in __init__);
  * reads the source value transposed in forward / grad-input, and
    **accumulates** its weight gradient straight into the source grad
    `Tensor` (`+=`) in `vjp`. Both leaves accumulating into one grad cell is
    correct because storage grads are `+=` and `zero_grad` clears the cell
    once per step (the source leaf owns + zeroes it).

BORROWING REPRESENTATION (the crux): the borrow is the SAFE `std.memory.Pointer`
(single-element, no arithmetic, bounds-safe deref — NOT a raw `UnsafePointer`),
holding a `Pointer[Tensor, MutAnyOrigin]` to the owner's `Param.val` /
`Param.grd` *storage cells*. Dereferencing yields a mutable `Tensor`, so
forward/vjp build their typed views with the SAME storage-surface calls every
other leaf uses — `TileTensor(t.data, …)` / cblas on CPU, `t.lt["gpu", layout]()`
on GPU. `tie_to` takes the source cells BY `ref` and builds the `Pointer`
internally, so no pointer is constructed at the call site.

The origin is the wildcard `MutAnyOrigin` (not a TRACKED origin `o` like
`TensorRefs`) DELIBERATELY: a tracked `Pointer[Tensor, o]` would force a
borrow-origin TYPE PARAMETER onto this struct, which would ripple into every
combinator that holds it (`Sequential`/`Tokenwise`/… are parametric over
`*MODULES: Module` and would each have to thread `o`). The wildcard keeps the
struct origin-free so it conforms to `Module` and drops into a generic
`Sequential` trained by the standard `Trainer` loop. The trade: the wildcard
does not pin the owner, so the POINTER-STABILITY RULE below is load-bearing.
This is the storage-clean evolution of the legacy raw `UnsafePointer[Scalar]`
to the val/grad buffers: safe single-element reference, Tensor-cell granularity,
fp32 compute only (no AMP) for v1.

POINTER-STABILITY RULE (load-bearing): `tie_to` captures a wildcard-origin
reference to the owner's `Param.val` / `Param.grd` *cells*, so the OWNER must
outlive every forward/vjp on this head. In real use that is automatic — the
source leaf (embedding) lives inside the same model struct that owns this head
— but it means `tie_to` must run after the owning model reaches its final home
(and after any load), and a standalone owner cell (e.g. in a test) must be kept
alive past the tied calls. A destroyed owner leaves a dangling reference; on GPU
that surfaces as the head reading a freed device buffer (zeros / empty `.dev`).

Shapes (IN = EMBED, OUT = VOCAB; source weight is `[OUT, IN]`):
  * forward:        out[b,v]  = Σ_e x[b,e]·Wsrc[v,e]      (= x @ Wsrcᵀ)
  * vjp grad-input: dx[b,e]   = Σ_v dout[b,v]·Wsrc[v,e]   (= dout @ Wsrc)
  * vjp grad-weight: dWsrc[v,e] += Σ_b dout[b,v]·x[b,e]   (accumulate)
"""

from std.sys import CompilationTarget
from std.memory import Pointer
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, TileTensor, row_major
from linalg.matmul import matmul as max_matmul
from linalg.matmul.cpu.apple_accelerate import (
    get_cblas_f32_function,
    _CBLASOrder,
    _CBLASTranspose,
)

from mojo_rl.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


# ──────────────────────────────────────────────────────────────────────
# GPU param-grad kernel — accumulate dWsrc[v,e] += Σ_b dout[b,v]·x[b,e]
# into the borrowed source grad buffer. One thread per (v, e); serial over
# BATCH. Carried over verbatim from legacy (mirrors Embedding's
# `_embedding_grad_w_kernel`, so the two accumulate symmetrically into the
# shared buffer).
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


struct TiedLinear[IN_: Int, OUT_: Int](Module):
    comptime ARITY = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.IN_)
    comptime OUT_DIM = Self.OUT_
    comptime W_SIZE = Self.IN_ * Self.OUT_

    # No owned weight/bias Param — the weight is borrowed (reflection finds no
    # `IsParam` field here, so the optimizer never double-counts the shared
    # weight). `tie_to` points these at the source weight's val/grad cells.
    # SAFE `Pointer` (not raw `UnsafePointer`); wildcard origin so the struct
    # stays origin-free (see module docstring).
    var src_val: Optional[Pointer[Tensor, MutAnyOrigin]]
    var src_grd: Optional[Pointer[Tensor, MutAnyOrigin]]

    def __init__(out self):
        self.src_val = None
        self.src_grd = None

    # ----- Factory (INIT unused — no owned params) ------------------------

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "TiedLinear: target must be 'cpu' or 'gpu'"
        return Self()

    # ----- Tie wiring -----------------------------------------------------

    def tie_to(mut self, ref val: Tensor, ref grd: Tensor):
        """Point this head at the source weight's value + grad storage cells
        (the owner's `Param.val` / `Param.grd`, laid out `[OUT, IN]`). Pass the
        cells BY `ref` (e.g. `head.tie_to(emb.weight.val, emb.weight.grd)`);
        the safe `Pointer` is built internally — no pointer at the call site.
        Call once after the owning model settles in its final home (and after
        any load); the source leaf keeps the cells alive. Idempotent. The
        captured origin is the wildcard `MutAnyOrigin` (see module docstring),
        so the POINTER-STABILITY RULE applies: the owner must outlive this head.
        """
        self.src_val = rebind[Pointer[Tensor, MutAnyOrigin]](Pointer(to=val))
        self.src_grd = rebind[Pointer[Tensor, MutAnyOrigin]](Pointer(to=grd))

    def _val(self) raises -> Pointer[Tensor, MutAnyOrigin]:
        if not self.src_val:
            raise Error(
                "TiedLinear: not wired — call tie_to(...) before forward/vjp"
            )
        return self.src_val.value()

    def _grd(self) raises -> Pointer[Tensor, MutAnyOrigin]:
        if not self.src_grd:
            raise Error(
                "TiedLinear: not wired — call tie_to(...) before forward/vjp"
            )
        return self.src_grd.value()

    # ----- Forward: out = x @ Wsrcᵀ ---------------------------------------

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime assert (
            POLICY.compute_dtype == DT
        ), "TiedLinear supports fp32 compute only"
        ref in0 = inputs[0]
        ref w = self._val()[]  # source weight cell [OUT, IN]
        comptime if target == "cpu":
            out.ensure(B * Self.OUT_)
            var x_v = TileTensor(in0.data, row_major[B, Self.IN_]())
            var w_v = TileTensor(w.data, row_major[Self.OUT_, Self.IN_]())
            var out_v = TileTensor(out.data, row_major[B, Self.OUT_]())
            # out[b,v] = Σ_e x[b,e]·Wsrc[v,e]  (transpose_b: contract IN).
            max_matmul[transpose_b=True, target="cpu"](out_v, x_v, w_v, None)
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_)
            var x_v = TileTensor(in0.dev.value(), row_major[B, Self.IN_]())
            var w_v = TileTensor(w.dev.value(), row_major[Self.OUT_, Self.IN_]())
            var out_v = TileTensor(out.dev.value(), row_major[B, Self.OUT_]())
            # out[b,v] = Σ_e x[b,e]·Wsrc[v,e]  (transpose_b: contract IN).
            max_matmul[transpose_b=True, target="gpu"](out_v, x_v, w_v, c)

    # ----- Backward (combined; Embedding-style single vjp) ----------------

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime assert (
            POLICY.compute_dtype == DT
        ), "TiedLinear supports fp32 compute only"
        ref fin = forward_input[0]
        ref gin = grad_inputs[0]
        ref w = self._val()[]  # source weight cell [OUT, IN]
        ref gw = self._grd()[]  # source grad cell [OUT, IN] (accumulate into)
        comptime if target == "cpu":
            gin.ensure(B * Self.IN_)
            var go_v = TileTensor(
                grad_output.data, row_major[B, Self.OUT_]()
            )
            var gi_v = TileTensor(gin.data, row_major[B, Self.IN_]())
            var w_v = TileTensor(w.data, row_major[Self.OUT_, Self.IN_]())
            # ── (1) grad-weight: dWsrc[v,e] += Σ_b dout[b,v]·x[b,e] = doutᵀ@x.
            # Apple-fp32: ONE fused cblas_sgemm (TRANSPOSE A=dout, beta=1 → no
            # transpose buffer, no temp, no accumulate loop). Else: naive loops.
            comptime IS_APPLE_F32 = (
                CompilationTarget.is_macos() and DT == DType.float32
            )
            comptime if IS_APPLE_F32:
                var cblas = get_cblas_f32_function()
                cblas(
                    _CBLASOrder.ROW_MAJOR,
                    _CBLASTranspose.TRANSPOSE,
                    _CBLASTranspose.NO_TRANSPOSE,
                    Int32(Self.OUT_),
                    Int32(Self.IN_),
                    Int32(B),
                    Float32(1.0),
                    rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
                        grad_output.data.unsafe_ptr()
                    ),
                    Int32(Self.OUT_),
                    rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
                        fin.data.unsafe_ptr()
                    ),
                    Int32(Self.IN_),
                    Float32(1.0),
                    rebind[UnsafePointer[Float32, MutAnyOrigin]](
                        gw.data.unsafe_ptr()
                    ),
                    Int32(Self.IN_),
                )
            else:
                var x_v = TileTensor(fin.data, row_major[B, Self.IN_]())
                var gw_v = TileTensor(
                    gw.data, row_major[Self.OUT_, Self.IN_]()
                )
                for v in range(Self.OUT_):
                    for e in range(Self.IN_):
                        var acc: Scalar[DT] = 0.0
                        for b in range(B):
                            acc += go_v[b, v] * x_v[b, e]
                        gw_v[v, e] += acc
            # ── (2) grad-input = dout @ Wsrc.
            max_matmul[target="cpu"](gi_v, go_v, w_v, None)
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.IN_)
            # ── (1) grad-weight kernel (accumulate into source grad cell).
            var go_lt = grad_output.lt[
                "gpu", Layout.row_major(B, Self.OUT_)
            ]()
            var x_lt = fin.lt["gpu", Layout.row_major(B, Self.IN_)]()
            var gw_lt = gw.lt["gpu", Layout.row_major(Self.OUT_, Self.IN_)]()
            comptime total = Self.W_SIZE
            comptime n_blocks = (total + TPB - 1) // TPB
            c.enqueue_function[_tied_grad_w_kernel[B, Self.OUT_, Self.IN_]](
                go_lt, x_lt, gw_lt, grid_dim=n_blocks, block_dim=TPB
            )
            # ── (2) grad-input = dout @ Wsrc.
            var go_v = TileTensor(
                grad_output.dev.value(), row_major[B, Self.OUT_]()
            )
            var w_v = TileTensor(w.dev.value(), row_major[Self.OUT_, Self.IN_]())
            var gi_v = TileTensor(gin.dev.value(), row_major[B, Self.IN_]())
            max_matmul[target="gpu"](gi_v, go_v, w_v, c)

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (no `IsParam` field here → reflects to a no-op; the shared weight is
    # walked exactly once, by the source leaf that owns it).
