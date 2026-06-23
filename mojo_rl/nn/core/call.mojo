"""call_forward / call_vjp — invoke a GENERICALLY-typed Module without the AMP
activation-dtype leaking into the caller.

The Module trait's `forward`/`vjp` take activation buffers at the module's OWN
`Self.ACT_DT`. A caller that holds the net as a CONCRETE type is fine (ACT_DT is
concrete = DT for fp32 nets). But a GENERIC caller — an agent block/primitive
over an opaque `M: Module` — can't satisfy `TensorRefs[M.ARITY, o, M.ACT_DT]` /
`TensorImpl[M.ACT_DT]` with its fp32 buffers, because the checker treats
`M.ACT_DT` as opaque even when it's `DT`. That's the "rebind dance" agents would
otherwise have to write at every call site.

These helpers contain that rebind ONCE. The caller passes its own buffers (any
dtype `BDT` — `DT`/fp32 in practice); the helper reinterprets the small pointer
pack + the buffer container to the net's `M.ACT_DT` and dispatches. Sound under
the invariant `BDT == M.ACT_DT` (a no-op at NoAMP, where both are `DT`; for a
bf16 net the caller supplies bf16 buffers). Only pointers/containers are
reinterpreted — the underlying `Tensor` never moves.

Usage (replaces a direct generic call):
    # was: net.forward[target, B, POLICY=P](TensorRefs[NET.ARITY](x), out, ctx)
    call_forward[target, B, POLICY=P](net, TensorRefs[NET.ARITY](x), out, ctx)
    # was: net.vjp[target, B](fin_refs, grad_out, gin_refs, ctx)
    call_vjp[target, B](net, fin_refs, grad_out, gin_refs, ctx)
"""

from std.gpu.host import DeviceContext

from .module import Module
from .tensor import TensorImpl
from .tensor_refs import TensorRefs
from .amp import AMPPolicy, NoAMP


def call_forward[
    M: Module, N: Int, o: MutOrigin, BDT: DType, //,
    target: StaticString, B: Int, POLICY: AMPPolicy = NoAMP,
](
    mut net: M,
    refs: TensorRefs[N, o, BDT],
    mut out: TensorImpl[BDT],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Call `net.forward` from generic code: rebind the (BDT) input pack + output
    to the net's `M.ACT_DT`. `M`/`N`/`o`/`BDT` infer from the args."""
    net.forward[target, B, POLICY=POLICY](
        rebind[TensorRefs[M.ARITY, o, M.ACT_DT]](refs),
        rebind[TensorImpl[M.ACT_DT]](out),
        ctx,
    )


def call_vjp[
    M: Module, NF: Int, fi: MutOrigin, NG: Int, gi: MutOrigin, BDT: DType, //,
    target: StaticString, B: Int, POLICY: AMPPolicy = NoAMP,
](
    mut net: M,
    fin_refs: TensorRefs[NF, fi, BDT],
    mut grad_out: TensorImpl[BDT],
    gin_refs: TensorRefs[NG, gi, BDT],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Call `net.vjp` from generic code: rebind the (BDT) forward-input pack,
    grad-output, and grad-input pack to the net's `M.ACT_DT`."""
    net.vjp[target, B, POLICY=POLICY](
        rebind[TensorRefs[M.ARITY, fi, M.ACT_DT]](fin_refs),
        rebind[TensorImpl[M.ACT_DT]](grad_out),
        rebind[TensorRefs[M.ARITY, gi, M.ACT_DT]](gin_refs),
        ctx,
    )
