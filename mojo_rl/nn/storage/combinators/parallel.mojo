"""Parallel[*BRANCHES] — N-ary fan-out then column-concat (storage, CPU + GPU).

Runs each branch on the SAME input and concatenates the per-branch outputs along
columns: `out = [B0(x) | B1(x) | … ]`. Backward splits grad_output into per-branch
column slices, runs each branch's vjp into a shared gi_temp, and accumulates the
per-branch grad-inputs.

Variadic from the start — the storage `TensorPack[N]` (whose `__getitem__`
MutAnyOrigin pin makes each slot a safe branch output) + comptime cumulative
offsets make the N-ary form as clean as the old hardcoded 2-branch one. The
2-branch `Parallel[A, B]` (e.g. the actor's `[mu | log_std]` heads) is just N=2.
`BranchConcat` is a parametric alias of this.

  ARITY  = 1            (one input, N branches)
  OUT_DIM = Σ BRANCHES[i].OUT_DIM
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, TPB, CPU_SIMD_W
from ..core.initializer import Initializer
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.tensor_pack import TensorPack
from ..core.module import Module
from ..core.param import ParamVisitor
from ..primitives.linear import _accum_kernel  # dst[i] += src[i]


# ── comptime variadic helpers ──────────────────────────────────────────
def _total_out_dim[*BRANCHES: Module]() -> Int:
    var s = 0
    comptime for i in range(BRANCHES.size):
        s += BRANCHES[i].OUT_DIM
    return s


def _cumulative_offset[index: Int, *BRANCHES: Module]() -> Int:
    var s = 0
    comptime for j in range(index):
        s += BRANCHES[j].OUT_DIM
    return s


# ── per-branch offset write/read into the packed buffer ─────────────────
def _par_write_kernel[
    B: Int, OI: Int, OD: Int
](
    slab: LayoutTensor[DT, Layout.row_major(B, OI), MutAnyOrigin],
    packed: LayoutTensor[DT, Layout.row_major(B, OD), MutAnyOrigin],
    off: Int,
):
    var idx = Int(global_idx.x)
    if idx < B * OI:
        var bi = idx // OI
        var ji = idx % OI
        packed[bi, off + ji] = rebind[Scalar[DT]](slab[bi, ji])


def _par_read_kernel[
    B: Int, OI: Int, OD: Int
](
    packed: LayoutTensor[DT, Layout.row_major(B, OD), MutAnyOrigin],
    slab: LayoutTensor[DT, Layout.row_major(B, OI), MutAnyOrigin],
    off: Int,
):
    var idx = Int(global_idx.x)
    if idx < B * OI:
        var bi = idx // OI
        var ji = idx % OI
        slab[bi, ji] = rebind[Scalar[DT]](packed[bi, off + ji])


# ── 2-branch concat/split kernels — kept for SkipConcat's input|inner merge.
def _par_concat_kernel[
    B: Int, OA: Int, OB: Int
](
    a: LayoutTensor[DT, Layout.row_major(B, OA), MutAnyOrigin],
    bb: LayoutTensor[DT, Layout.row_major(B, OB), MutAnyOrigin],
    packed: LayoutTensor[DT, Layout.row_major(B, OA + OB), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = B * (OA + OB)
    if idx < total:
        var bi = idx // (OA + OB)
        var ji = idx % (OA + OB)
        if ji < OA:
            packed[bi, ji] = rebind[Scalar[DT]](a[bi, ji])
        else:
            packed[bi, ji] = rebind[Scalar[DT]](bb[bi, ji - OA])


def _par_split_kernel[
    B: Int, OA: Int, OB: Int
](
    packed: LayoutTensor[DT, Layout.row_major(B, OA + OB), MutAnyOrigin],
    a: LayoutTensor[DT, Layout.row_major(B, OA), MutAnyOrigin],
    bb: LayoutTensor[DT, Layout.row_major(B, OB), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = B * (OA + OB)
    if idx < total:
        var bi = idx // (OA + OB)
        var ji = idx % (OA + OB)
        if ji < OA:
            a[bi, ji] = rebind[Scalar[DT]](packed[bi, ji])
        else:
            bb[bi, ji - OA] = rebind[Scalar[DT]](packed[bi, ji])


struct Parallel[*BRANCHES: Module](Module):
    comptime ARITY = 1
    comptime N = Self.BRANCHES.size
    comptime IN = Self.BRANCHES[0].IN_DIMS[0]
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.BRANCHES[0].IN_DIMS[0])
    comptime OUT_DIM = _total_out_dim[*Self.BRANCHES]()

    var branches: Tuple[*Self.BRANCHES]
    var slabs: TensorPack[Self.N]
    var gi_temp: Tensor

    def __init__(out self):
        comptime assert Self.N >= 1, "Parallel requires >= 1 branch"
        comptime for i in range(Self.N):
            comptime assert (
                Self.BRANCHES[i].IN_DIMS[0] == Self.BRANCHES[0].IN_DIMS[0]
            ), "Parallel: all BRANCHES must share IN_DIM"
        self.branches = Tuple[*Self.BRANCHES]()
        self.slabs = TensorPack[Self.N]()
        self.gi_temp = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var p = Self()
        comptime for i in range(Self.N):
            p.branches[i] = Self.BRANCHES[i].make[target, INIT](ctx)
        return p^

    def forward[
        target: StaticString, B: Int, o: MutOrigin
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime for i in range(Self.N):
            self.branches[i].forward[target, B](
                TensorRefs[Self.BRANCHES[i].ARITY](in0), self.slabs[i], ctx
            )
        comptime if target == "cpu":
            out.ensure(B * Self.OUT_DIM)
            comptime for i in range(Self.N):
                comptime off = _cumulative_offset[i, *Self.BRANCHES]()
                comptime oi = Self.BRANCHES[i].OUT_DIM
                for b in range(B):
                    for j in range(oi):
                        out.data[b * Self.OUT_DIM + off + j] = self.slabs[
                            i
                        ].data[b * oi + j]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_DIM)
            comptime for i in range(Self.N):
                comptime off = _cumulative_offset[i, *Self.BRANCHES]()
                comptime oi = Self.BRANCHES[i].OUT_DIM
                c.enqueue_function[_par_write_kernel[B, oi, Self.OUT_DIM]](
                    self.slabs[i].lt["gpu", Layout.row_major(B, oi)](),
                    out.lt["gpu", Layout.row_major(B, Self.OUT_DIM)](),
                    off,
                    grid_dim=(B * oi + TPB - 1) // TPB,
                    block_dim=TPB,
                )

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin
    ](
        mut self,
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref fin = forward_input[0]
        ref gin = grad_inputs[0]
        comptime NIN = B * Self.IN
        comptime if target == "cpu":
            gin.ensure(NIN)
            for k in range(NIN):
                gin.data[k] = Scalar[DT](0)
        else:
            var c0 = ctx.value()
            gin.ensure_gpu(c0, NIN)
            gin.dev.value().enqueue_fill(Scalar[DT](0))

        comptime for i in range(Self.N):
            comptime off = _cumulative_offset[i, *Self.BRANCHES]()
            comptime oi = Self.BRANCHES[i].OUT_DIM
            comptime if target == "cpu":
                for b in range(B):
                    for j in range(oi):
                        self.slabs[i].data[b * oi + j] = grad_output.data[
                            b * Self.OUT_DIM + off + j
                        ]
            else:
                var c = ctx.value()
                c.enqueue_function[_par_read_kernel[B, oi, Self.OUT_DIM]](
                    grad_output.lt["gpu", Layout.row_major(B, Self.OUT_DIM)](),
                    self.slabs[i].lt["gpu", Layout.row_major(B, oi)](),
                    off,
                    grid_dim=(B * oi + TPB - 1) // TPB,
                    block_dim=TPB,
                )
            self.branches[i].vjp[target, B](
                TensorRefs[Self.BRANCHES[i].ARITY](fin),
                self.slabs[i],
                TensorRefs[Self.BRANCHES[i].ARITY](self.gi_temp),
                ctx,
            )
            comptime if target == "cpu":
                var gp = gin.data.unsafe_ptr()
                var ap = self.gi_temp.data.unsafe_ptr()
                comptime W = CPU_SIMD_W
                var k = 0
                while k + W <= NIN:
                    gp.store(k, gp.load[width=W](k) + ap.load[width=W](k))
                    k += W
                while k < NIN:
                    gp[k] = gp[k] + ap[k]
                    k += 1
            else:
                var c = ctx.value()
                c.enqueue_function[_accum_kernel[NIN]](
                    gin.lt["gpu", Layout.row_major(NIN)](),
                    self.gi_temp.lt["gpu", Layout.row_major(NIN)](),
                    grid_dim=(NIN + TPB - 1) // TPB,
                    block_dim=TPB,
                )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext]) raises:
        comptime for i in range(Self.N):
            self.branches[i].for_each_param[target](visitor, ctx)

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        comptime for i in range(Self.N):
            self.branches[i].zero_grad[target](ctx)

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT], ctx: Optional[DeviceContext]
    ) raises:
        comptime for i in range(Self.N):
            self.branches[i].polyak_from[target](src.branches[i], tau, ctx)
