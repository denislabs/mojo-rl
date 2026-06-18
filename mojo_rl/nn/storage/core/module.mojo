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
from .tensor import Tensor
from .tensor_refs import TensorRefs
from .param import ParamVisitor


trait Module(Defaultable & Movable & ImplicitlyDeletable):
    comptime ARITY: Int
    comptime IN_DIMS: InlineArray[Int, Self.ARITY]
    comptime OUT_DIM: Int

    @staticmethod
    def make_cpu() raises -> Self:
        ...

    @staticmethod
    def make_gpu(ctx: DeviceContext) raises -> Self:
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
        mut self, mut visitor: V, ctx: Optional[DeviceContext]
    ) raises:
        pass

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        pass

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
