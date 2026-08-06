"""`PairwiseDot[D, BATCH]` and `RowDot[D]` — products of two ACTIVATIONS.

Every matmul elsewhere in `nn` is weight x activation. These two are not: both
inputs are activations, and both receive gradients. Forward-Backward
representations need exactly that, twice over —

    M = F(s,a,z) . B(s+)^T      the successor-measure matrix, [BATCH, BATCH]
    O = B(s)     . B(s+)^T      the orthonormality regulariser, same shape

so one primitive serves both loss terms of `docs/BFM_ZERO_SHOT_RL.md` §6
component 2. `RowDot` is its diagonal-only sibling, for the terms that pair
row i with row i and nothing else:

    a[i] = F(s_i,a_i,z_i) . B(s'_i)     the anchor term, [BATCH, 1]

⚠ **The anchor term is not a regulariser.** `-2·E[F(s,a,z)^T B(s')]` falls out
of expanding the square of the successor measure. Dropping it leaves a model
whose loss still descends and whose policy still moves, encoding nothing — see
§11, where it is ranked among the silent failures. `RowDot` exists so that term
is as cheap as it is meant to be: taking the diagonal of a `PairwiseDot` would
compute BATCH² dot products to keep BATCH of them.

Shapes and gradients:

    PairwiseDot   in ([B, D], [B, D]) -> out [B, B]
        M[i,j] = sum_k A[i,k]·C[j,k]
        dA     = G·C          i.e. dA[i,k] = sum_j G[i,j]·C[j,k]
        dC     = G^T·A        i.e. dC[j,k] = sum_i G[i,j]·A[i,k]

    RowDot        in ([B, D], [B, D]) -> out [B, 1]
        r[i]   = sum_k A[i,k]·C[i,k]
        dA[i,k] = G[i,0]·C[i,k];   dC[i,k] = G[i,0]·A[i,k]

Neither caches anything: the vjp reads `forward_input`, which the graph already
holds. That is why `MSEPerSample` — the closest existing 2-input leaf — carries
a `cache_diff` and these do not.

**`BATCH` is a compile-time parameter of `PairwiseDot`, not just of `forward`.**
The output row width IS the batch size, and `Module.OUT_DIM` is a struct-level
comptime Int, so it cannot be otherwise. `forward` asserts `B == BATCH` rather
than letting a mismatched combinator silently size a buffer to the wrong width.

⚠ Kernels are deliberately NAIVE — one thread per output element, an inner loop
over the contracted axis, no shared-memory tiling. This is not an oversight to
optimise away later: heavy blocked kernels hard-crash the Metal compiler on
Apple (see the project's GPU notes), and at the sizes involved the arithmetic
is not the cost. [1024, 1024] fp32 is 4 MB and roughly three are live at once.
Revisit only with a measurement, and only on NVIDIA.

⚠ `max_matmul` with `transpose_b` is NOT used here, and could not be: it
silently miscomputes at N=1. `PairwiseDot`'s N is BATCH, so a single-row
inference path reusing this primitive would land exactly on that bug.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


# ──────────────────────────────────────────────────────────────────────
# PairwiseDot kernels
# ──────────────────────────────────────────────────────────────────────


def _pd_forward_kernel[BATCH: Int, D: Int](
    a: LayoutTensor[DT, Layout.row_major(BATCH, D), MutAnyOrigin],
    c: LayoutTensor[DT, Layout.row_major(BATCH, D), MutAnyOrigin],
    m: LayoutTensor[DT, Layout.row_major(BATCH, BATCH), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * BATCH:
        return
    var i = idx // BATCH
    var j = idx % BATCH
    var s = Scalar[DT](0)
    for k in range(D):
        s += rebind[Scalar[DT]](a[i, k]) * rebind[Scalar[DT]](c[j, k])
    m[i, j] = s


def _pd_da_kernel[BATCH: Int, D: Int](
    g: LayoutTensor[DT, Layout.row_major(BATCH, BATCH), MutAnyOrigin],
    c: LayoutTensor[DT, Layout.row_major(BATCH, D), MutAnyOrigin],
    ga: LayoutTensor[DT, Layout.row_major(BATCH, D), MutAnyOrigin],
):
    # dA[i,k] = sum_j G[i,j]·C[j,k]
    var idx = Int(global_idx.x)
    if idx >= BATCH * D:
        return
    var i = idx // D
    var k = idx % D
    var s = Scalar[DT](0)
    for j in range(BATCH):
        s += rebind[Scalar[DT]](g[i, j]) * rebind[Scalar[DT]](c[j, k])
    ga[i, k] = s


def _pd_dc_kernel[BATCH: Int, D: Int](
    g: LayoutTensor[DT, Layout.row_major(BATCH, BATCH), MutAnyOrigin],
    a: LayoutTensor[DT, Layout.row_major(BATCH, D), MutAnyOrigin],
    gc: LayoutTensor[DT, Layout.row_major(BATCH, D), MutAnyOrigin],
):
    # dC[j,k] = sum_i G[i,j]·A[i,k]  — note G is read down its COLUMN j.
    var idx = Int(global_idx.x)
    if idx >= BATCH * D:
        return
    var j = idx // D
    var k = idx % D
    var s = Scalar[DT](0)
    for i in range(BATCH):
        s += rebind[Scalar[DT]](g[i, j]) * rebind[Scalar[DT]](a[i, k])
    gc[j, k] = s


struct PairwiseDot[D_: Int, BATCH: Int](Module):
    """`out[i, j] = sum_k A[i, k] · C[j, k]`, both inputs differentiable."""

    comptime ARITY = 2
    comptime IN_DIMS = InlineArray[Int, 2](fill=Self.D_)
    comptime OUT_DIM = Self.BATCH

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Unified CPU/GPU factory. INIT accepted for `make[target, INIT]`
        uniformity but ignored (no params)."""
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[2, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime assert B == Self.BATCH, (
            "PairwiseDot: forward batch must equal the BATCH parameter — the"
            " output row width IS the batch size, so a mismatch would size"
            " every downstream buffer wrongly rather than merely process fewer"
            " rows"
        )
        ref a = inputs[0]
        ref c = inputs[1]
        comptime if target == "cpu":
            out.ensure(B * B)
            var a_t = TileTensor(a.data, row_major[B, Self.D_]())
            var c_t = TileTensor(c.data, row_major[B, Self.D_]())
            var m_t = TileTensor(out.data, row_major[B, B]())
            for i in range(B):
                for j in range(B):
                    var s = Scalar[DT](0)
                    for k in range(Self.D_):
                        s += a_t[i, k] * c_t[j, k]
                    m_t[i, j] = s
        else:
            var dc = ctx.value()
            out.ensure_gpu(dc, B * B)
            comptime lay_in = Layout.row_major(B, Self.D_)
            comptime lay_m = Layout.row_major(B, B)
            comptime n_blocks = (B * B + TPB - 1) // TPB
            dc.enqueue_function[_pd_forward_kernel[B, Self.D_]](
                a.lt["gpu", lay_in](),
                c.lt["gpu", lay_in](),
                out.lt["gpu", lay_m](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[2, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[2, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime assert B == Self.BATCH, "PairwiseDot: vjp batch must equal BATCH"
        ref a = forward_input[0]
        ref c = forward_input[1]
        ref ga = grad_inputs[0]
        ref gc = grad_inputs[1]
        comptime if target == "cpu":
            ga.ensure(B * Self.D_)
            gc.ensure(B * Self.D_)
            var a_t = TileTensor(a.data, row_major[B, Self.D_]())
            var c_t = TileTensor(c.data, row_major[B, Self.D_]())
            var g_t = TileTensor(grad_output.data, row_major[B, B]())
            var ga_t = TileTensor(ga.data, row_major[B, Self.D_]())
            var gc_t = TileTensor(gc.data, row_major[B, Self.D_]())
            for i in range(B):
                for k in range(Self.D_):
                    var s = Scalar[DT](0)
                    for j in range(B):
                        s += g_t[i, j] * c_t[j, k]
                    ga_t[i, k] = s
            for j in range(B):
                for k in range(Self.D_):
                    var s = Scalar[DT](0)
                    for i in range(B):
                        s += g_t[i, j] * a_t[i, k]
                    gc_t[j, k] = s
        else:
            var dc = ctx.value()
            ga.ensure_gpu(dc, B * Self.D_)
            gc.ensure_gpu(dc, B * Self.D_)
            comptime lay_in = Layout.row_major(B, Self.D_)
            comptime lay_m = Layout.row_major(B, B)
            comptime n_blocks = (B * Self.D_ + TPB - 1) // TPB
            dc.enqueue_function[_pd_da_kernel[B, Self.D_]](
                grad_output.lt["gpu", lay_m](),
                c.lt["gpu", lay_in](),
                ga.lt["gpu", lay_in](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )
            dc.enqueue_function[_pd_dc_kernel[B, Self.D_]](
                grad_output.lt["gpu", lay_m](),
                a.lt["gpu", lay_in](),
                gc.lt["gpu", lay_in](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (param-less leaf -> no-op).


# ──────────────────────────────────────────────────────────────────────
# RowDot kernels
# ──────────────────────────────────────────────────────────────────────


def _rd_forward_kernel[BATCH: Int, D: Int](
    a: LayoutTensor[DT, Layout.row_major(BATCH, D), MutAnyOrigin],
    c: LayoutTensor[DT, Layout.row_major(BATCH, D), MutAnyOrigin],
    o: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i >= BATCH:
        return
    var s = Scalar[DT](0)
    for k in range(D):
        s += rebind[Scalar[DT]](a[i, k]) * rebind[Scalar[DT]](c[i, k])
    o[i, 0] = s


def _rd_backward_kernel[BATCH: Int, D: Int](
    g: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    a: LayoutTensor[DT, Layout.row_major(BATCH, D), MutAnyOrigin],
    c: LayoutTensor[DT, Layout.row_major(BATCH, D), MutAnyOrigin],
    ga: LayoutTensor[DT, Layout.row_major(BATCH, D), MutAnyOrigin],
    gc: LayoutTensor[DT, Layout.row_major(BATCH, D), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * D:
        return
    var i = idx // D
    var k = idx % D
    var go = rebind[Scalar[DT]](g[i, 0])
    ga[i, k] = go * rebind[Scalar[DT]](c[i, k])
    gc[i, k] = go * rebind[Scalar[DT]](a[i, k])


struct RowDot[D_: Int](Module):
    """`out[i, 0] = sum_k A[i, k] · C[i, k]` — the diagonal of `PairwiseDot`,
    computed without the other BATCH²-BATCH entries."""

    comptime ARITY = 2
    comptime IN_DIMS = InlineArray[Int, 2](fill=Self.D_)
    comptime OUT_DIM = 1

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[2, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref a = inputs[0]
        ref c = inputs[1]
        comptime if target == "cpu":
            out.ensure(B)
            var a_t = TileTensor(a.data, row_major[B, Self.D_]())
            var c_t = TileTensor(c.data, row_major[B, Self.D_]())
            var o_t = TileTensor(out.data, row_major[B, 1]())
            for i in range(B):
                var s = Scalar[DT](0)
                for k in range(Self.D_):
                    s += a_t[i, k] * c_t[i, k]
                o_t[i, 0] = s
        else:
            var dc = ctx.value()
            out.ensure_gpu(dc, B)
            comptime lay_in = Layout.row_major(B, Self.D_)
            comptime lay_o = Layout.row_major(B, 1)
            comptime n_blocks = (B + TPB - 1) // TPB
            dc.enqueue_function[_rd_forward_kernel[B, Self.D_]](
                a.lt["gpu", lay_in](),
                c.lt["gpu", lay_in](),
                out.lt["gpu", lay_o](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[2, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[2, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref a = forward_input[0]
        ref c = forward_input[1]
        ref ga = grad_inputs[0]
        ref gc = grad_inputs[1]
        comptime if target == "cpu":
            ga.ensure(B * Self.D_)
            gc.ensure(B * Self.D_)
            var a_t = TileTensor(a.data, row_major[B, Self.D_]())
            var c_t = TileTensor(c.data, row_major[B, Self.D_]())
            var g_t = TileTensor(grad_output.data, row_major[B, 1]())
            var ga_t = TileTensor(ga.data, row_major[B, Self.D_]())
            var gc_t = TileTensor(gc.data, row_major[B, Self.D_]())
            for i in range(B):
                var go = g_t[i, 0]
                for k in range(Self.D_):
                    ga_t[i, k] = go * c_t[i, k]
                    gc_t[i, k] = go * a_t[i, k]
        else:
            var dc = ctx.value()
            ga.ensure_gpu(dc, B * Self.D_)
            gc.ensure_gpu(dc, B * Self.D_)
            comptime lay_in = Layout.row_major(B, Self.D_)
            comptime lay_o = Layout.row_major(B, 1)
            comptime n_blocks = (B * Self.D_ + TPB - 1) // TPB
            dc.enqueue_function[_rd_backward_kernel[B, Self.D_]](
                grad_output.lt["gpu", lay_o](),
                a.lt["gpu", lay_in](),
                c.lt["gpu", lay_in](),
                ga.lt["gpu", lay_in](),
                gc.lt["gpu", lay_in](),
                grid_dim=n_blocks,
                block_dim=TPB,
            )
