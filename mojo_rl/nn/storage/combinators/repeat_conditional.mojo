"""RepeatConditional[N, Inner] — chain N copies of a 2-input conditional block
(storage surface).

Stacks `Inner` (ARITY=2, `forward(x, c)`, dim-preserving) N times: the main
stream `x` chains block→block while the conditioning `c` is **broadcast** to
every block (same input). On backward, grad_x chains in reverse and grad_c is
**accumulated** across all N blocks (c fans out to every layer, so its gradient
is the sum of each layer's contribution):

    x_0 = x;  x_{i+1} = Inner_i(x_i, c);  out = x_N
    grad_c = Σ_i (∂Inner_i/∂c)·grad_{x_{i+1}}

The LeWM AR-predictor stack: `RepeatConditional[DEPTH,
ConditionalTransformerBlock[...]]`. Each block owns its own params
(shared=False, like `Repeat`).

Storage wiring (§B0): a block's two inputs must share ONE origin, so they are
read from a single owning pool — `pool` holds the x-copy (slot 0), the mid
activations (slots 1..N-1) and the broadcast c-copy (slot N), all `MutAnyOrigin`
via `TensorPack.__getitem__`. The backward grad pool `gpool` holds the per-block
grad_x slots (0..N-1) + a reused grad_c temp (slot N); each block's grad_c temp
is ACCUMULATED into the caller's grad_inputs[1] (elementwise, origin-free), and
block 0's grad_x is copied out to grad_inputs[0]. Mirrors `Repeat`'s mid-slab
reuse + `ComputeGraph`'s fan-out accumulation kernel.
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, TPB
from ..core.initializer import Initializer
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.tensor_pack import TensorPack
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.walkers import join_name
from ..core.amp import AMPPolicy, NoAMP


def _rc_accum_kernel[
    N: Int
](
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    """dst[i] += src[i] — grad_c fan-out accumulation on device."""
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = rebind[Scalar[DT]](dst[i]) + rebind[Scalar[DT]](src[i])


struct RepeatConditional[N: Int, Inner: Module](Module):
    comptime ARITY = 2
    comptime D = Self.Inner.OUT_DIM
    comptime IN_DIMS = InlineArray[Int, 2](fill=Self.D)
    comptime OUT_DIM = Self.Inner.OUT_DIM

    var children: List[Self.Inner]
    # Forward pool: slot 0 = x copy, slots 1..N-1 = block(0..N-2) outputs,
    # slot N = broadcast c copy. (§B0: each block reads two slots of ONE pool.)
    var pool: TensorPack[Self.N + 1]
    # Backward pool: slots 0..N-1 = per-block grad_x, slot N = grad_c temp.
    var gpool: TensorPack[Self.N + 1]

    def __init__(out self):
        comptime assert Self.N >= 1, "RepeatConditional requires N >= 1"
        comptime assert (
            Self.Inner.ARITY == 2
        ), "RepeatConditional: Inner must be ARITY=2 (forward(x, c))"
        comptime assert (
            Self.Inner.IN_DIMS[0] == Self.Inner.OUT_DIM
            and Self.Inner.IN_DIMS[1] == Self.Inner.OUT_DIM
        ), "RepeatConditional: Inner must be dim-preserving (IN0==IN1==OUT)"
        self.children = List[Self.Inner]()
        self.pool = TensorPack[Self.N + 1]()
        self.gpool = TensorPack[Self.N + 1]()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var r = Self()
        for _ in range(Self.N):
            r.children.append(Self.Inner.make[target, INIT](ctx))
        return r^

    @staticmethod
    def _copy_into[
        target: StaticString
    ](
        mut dst: Tensor, mut src: Tensor, n: Int,
        ctx: Optional[DeviceContext],
    ) raises:
        """dst[0:n] = src[0:n] (target-aware copy into a pool slot)."""
        comptime if target == "cpu":
            dst.ensure(n)
            for q in range(n):
                dst.data[q] = src.data[q]
        else:
            var c = ctx.value()
            dst.ensure_gpu(c, n)
            c.enqueue_copy(dst.dev.value(), src.dev.value())

    @staticmethod
    def _accum_into[
        target: StaticString, NN: Int
    ](
        mut dst: Tensor, mut src: Tensor,
        ctx: Optional[DeviceContext],
    ) raises:
        """dst[0:NN] += src[0:NN] (target-aware grad_c fan-out)."""
        comptime if target == "cpu":
            for q in range(NN):
                dst.data[q] += src.data[q]
        else:
            ctx.value().enqueue_function[_rc_accum_kernel[NN]](
                dst.lt["gpu", Layout.row_major(NN)](),
                src.lt["gpu", Layout.row_major(NN)](),
                grid_dim=(NN + TPB - 1) // TPB,
                block_dim=TPB,
            )

    @staticmethod
    def _zero[
        target: StaticString
    ](mut dst: Tensor, n: Int, ctx: Optional[DeviceContext]) raises:
        comptime if target == "cpu":
            dst.ensure(n)
            for q in range(n):
                dst.data[q] = Scalar[DT](0)
        else:
            dst.ensure_gpu(ctx.value(), n)
            dst.dev.value().enqueue_fill(Scalar[DT](0))

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[2, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime n = B * Self.D
        # Seed pool: slot 0 = x copy, slot N = broadcast c copy.
        Self._copy_into[target](self.pool[0], inputs[0], n, ctx)
        Self._copy_into[target](self.pool[Self.N], inputs[1], n, ctx)
        comptime for i in range(Self.N):
            comptime if i == Self.N - 1:
                self.children[i].forward[target, B, POLICY=POLICY](
                    TensorRefs[Self.Inner.ARITY](self.pool[i], self.pool[Self.N]),
                    out,
                    ctx,
                )
            else:
                self.children[i].forward[target, B, POLICY=POLICY](
                    TensorRefs[Self.Inner.ARITY](self.pool[i], self.pool[Self.N]),
                    self.pool[i + 1],
                    ctx,
                )

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        forward_input: TensorRefs[2, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[2, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime n = B * Self.D
        # grad_c accumulator (caller slot) starts at zero; each block adds its
        # grad_c temp (gpool[N]) into it.
        Self._zero[target](grad_inputs[1], n, ctx)
        # Reverse-topo: block i's output-grad is grad_output (last) or block
        # i+1's already-computed grad_x (gpool[i+1]). Its grad_inputs scatter
        # into (gpool[i], gpool[N]=grad_c temp) — both from gpool (§B0).
        comptime for j in range(Self.N):
            comptime i = Self.N - 1 - j
            comptime if i == Self.N - 1:
                self.children[i].vjp[target, B, POLICY=POLICY](
                    TensorRefs[Self.Inner.ARITY](
                        self.pool[i], self.pool[Self.N]
                    ),
                    grad_output,
                    TensorRefs[Self.Inner.ARITY](
                        self.gpool[i], self.gpool[Self.N]
                    ),
                    ctx,
                )
            else:
                self.children[i].vjp[target, B, POLICY=POLICY](
                    TensorRefs[Self.Inner.ARITY](
                        self.pool[i], self.pool[Self.N]
                    ),
                    self.gpool[i + 1],
                    TensorRefs[Self.Inner.ARITY](
                        self.gpool[i], self.gpool[Self.N]
                    ),
                    ctx,
                )
            Self._accum_into[target, B * Self.D](
                grad_inputs[1], self.gpool[Self.N], ctx
            )
        # Block 0's grad_x (gpool[0]) is the gradient w.r.t. the x input.
        Self._copy_into[target](grad_inputs[0], self.gpool[0], n, ctx)

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        for i in range(Self.N):
            self.children[i].for_each_param[target](
                visitor, ctx, join_name(prefix, String(i))
            )

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        for i in range(Self.N):
            self.children[i].for_each_state[target](
                visitor, ctx, join_name(prefix, String(i))
            )

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        for i in range(Self.N):
            self.children[i].zero_grad[target](ctx)

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT], ctx: Optional[DeviceContext]
    ) raises:
        for i in range(Self.N):
            self.children[i].polyak_from[target](src.children[i], tau, ctx)
