"""Module — the storage-passing Module trait (N-ary, CPU + GPU).

forward/vjp take a borrowing `TensorRefs[ARITY, o]` input pack (origin-tracked,
inferred via `o`) + a single `Tensor` output/grad. The pack's origin `o` is the
ONE origin parameter on the surface — TRACKED (not the wildcard), threaded so
the referenced inputs stay live across the call. Inside, CPU views are tracked
`TileTensor`s over `.data`; GPU views are `lt_gpu` device tensors whose only
erasure is the kernel-arg `MutAnyOrigin` (the ABI). `grad_output` is `mut` so
the GPU view can be built. `ctx` is the GPU `DeviceContext` (ignored on CPU).
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from .initializer import Initializer
from .tensor import Tensor
from .tensor_refs import TensorRefs
from .param import ParamVisitor
from .walkers import for_each_param_auto, zero_grad_auto
from .state import for_each_state_auto


trait Module(Defaultable & Movable & ImplicitlyDeletable):
    comptime ARITY: Int
    comptime IN_DIMS: InlineArray[Int, Self.ARITY]
    comptime OUT_DIM: Int

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU (impls
        raise at runtime if missing). Params are allocated AND initialized with
        `INIT` at construction (no separate reinit pass); combinators thread
        `[target, INIT]` to their children."""
        ...

    def forward[target: StaticString, B: Int, o: MutOrigin](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ...

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ...

    def for_each_param[target: StaticString, V: ParamVisitor](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        """Default: reflection-walk every `IsParam` field of the concrete
        leaf and dispatch the visitor. Param-less leaves reflect to a no-op;
        Param-bearing leaves no longer need to override (forgetting it can no
        longer silently skip params in checkpoint/optimizer walks).
        Combinators + wrapper leaves (children are Module-typed, not IsParam)
        still override to recurse into children. `prefix` (default empty) is
        the dotted path so far; the walker composes `prefix.<param_name>`."""
        for_each_param_auto[Self, V, target](self, visitor, ctx, prefix)

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        """Default: reflection-walk every `IsParam` field and zero its grad.
        Param-less leaves reflect to a no-op; combinators override to recurse."""
        zero_grad_auto[Self, target](self, ctx)

    def for_each_state[target: StaticString, V: ParamVisitor](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        """Default: reflection-walk every `IsState` field of the concrete leaf
        and dispatch the visitor. The checkpoint path runs this right after
        `for_each_param`, so State fields (e.g. BatchNorm running stats) are
        persisted; the optimizer path (`for_each_param`) never reaches them.
        State-less leaves reflect to a no-op; combinators override to recurse."""
        for_each_state_auto[Self, V, target](self, visitor, ctx, prefix)

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        """Soft-update this (target) module's params toward `src` (online):
        `p_self = tau·p_src + (1-tau)·p_self`. Default no-op (param-less
        leaves + leaves whose polyak isn't exercised yet); param leaves
        override on their `Param`s, combinators recurse into children."""
        pass
