"""StopGradParams[Inner] — let grad flow through, freeze Inner's params.

Forward: passthrough to `inner.forward`.
Backward: grad flows to grad_input as normal, but Inner's params receive NO grad
from this path (their grads end the call exactly as they began it). Inner's params
stay VISIBLE to the optimizer (for_each_param passes through), so OTHER loss paths
can still update them — contrast `primitives/StopGrad`, which zeros grad_input.

The storage `Module.vjp` computes input-grad and param-grad together, so rather
than ripple an `input_only` mode through every leaf, this wrapper SNAPSHOTs
Inner's param grads, runs the full `inner.vjp` (correct grad_input, param grads
accumulate), then RESTOREs the snapshot — undoing only this path's param-grad
contribution while preserving any prior accumulation. Net result is identical to
legacy's `vjp[mode="input_only"]`.
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from ..core.initializer import Initializer
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.walkers import join_name
from ..core.amp import AMPPolicy, NoAMP


struct _GradStash(ParamVisitor):
    """Two-pass param-grad save/restore. First walk (restoring=False) copies each
    param's grad into `saved`; second walk (restoring=True) copies it back."""
    var saved: List[Tensor]
    var restoring: Bool
    var idx: Int

    def __init__(out self):
        self.saved = List[Tensor]()
        self.restoring = False
        self.idx = 0

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        if not self.restoring:
            var t = Tensor.alloc(N)
            comptime if target == "cpu":
                for k in range(N):
                    t.data[k] = grad.data[k]
            else:
                t.ensure_gpu(ctx.value(), N)
                # Size-exact sub-buffer copy — `grad` may be larger than N
                # (monotone ensure_gpu); whole-buffer copies error on the
                # size mismatch. Mirrors compute_graph's fix.
                var g_src = grad.dev.value().create_sub_buffer[DT](0, N)
                var t_dst = t.dev.value().create_sub_buffer[DT](0, N)
                ctx.value().enqueue_copy(t_dst, g_src)
            self.saved.append(t^)
        else:
            comptime if target == "cpu":
                for k in range(N):
                    grad.data[k] = self.saved[self.idx].data[k]
            else:
                var s_src = self.saved[self.idx].dev.value(
                ).create_sub_buffer[DT](0, N)
                var g_dst = grad.dev.value().create_sub_buffer[DT](0, N)
                ctx.value().enqueue_copy(g_dst, s_src)
            self.idx += 1


struct StopGradParams[Inner: Module](Module):
    comptime ARITY = Self.Inner.ARITY
    comptime IN_DIMS = Self.Inner.IN_DIMS
    comptime OUT_DIM = Self.Inner.OUT_DIM
    # Passthrough wrapper — activation dtype is the wrapped child's.
    comptime ACT_DT = Self.Inner.ACT_DT

    var inner: Self.Inner

    def __init__(out self):
        self.inner = Self.Inner()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var s = Self()
        s.inner = Self.Inner.make[target, INIT](ctx)
        return s^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self, inputs: TensorRefs[Self.ARITY, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        # Self.ARITY/ACT_DT == Inner's (definitionally), but distinct to the
        # checker — rebind the whole pack + the mut output to the child types.
        comptime ci = Self.Inner.ACT_DT
        comptime cn = Self.Inner.ARITY
        self.inner.forward[target, B, POLICY=POLICY](
            rebind[TensorRefs[cn, o, ci]](inputs),
            rebind[TensorImpl[ci]](out),
            ctx,
        )

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP
    ](
        mut self, forward_input: TensorRefs[Self.ARITY, ofi, Self.ACT_DT],
        mut grad_output: TensorImpl[Self.ACT_DT],
        grad_inputs: TensorRefs[Self.ARITY, ogi, Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime ci = Self.Inner.ACT_DT
        comptime cn = Self.Inner.ARITY
        var stash = _GradStash()
        self.inner.for_each_param[target](stash, ctx)  # snapshot
        self.inner.vjp[target, B, POLICY=POLICY](
            rebind[TensorRefs[cn, ofi, ci]](forward_input),
            rebind[TensorImpl[ci]](grad_output),
            rebind[TensorRefs[cn, ogi, ci]](grad_inputs),
            ctx,
        )
        stash.restoring = True
        stash.idx = 0
        self.inner.for_each_param[target](stash, ctx)  # restore (freeze params)

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        # Params stay visible: other loss paths / the optimizer still see them.
        self.inner.for_each_param[target](
            visitor, ctx, join_name(prefix, String(0))
        )

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        # States pass through normally (no grads to stash).
        self.inner.for_each_state[target](
            visitor, ctx, join_name(prefix, String(0))
        )

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        self.inner.zero_grad[target](ctx)

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT], ctx: Optional[DeviceContext]
    ) raises:
        self.inner.polyak_from[target](src.inner, tau, ctx)
