"""Repeat[N, Inner] correctness (storage surface, CPU + GPU).

Repeat[3, Linear[D,D]] must equal Sequential[Linear,Linear,Linear] (identical
deterministic init), and Repeat[1, Linear] must equal a single Linear — forward
AND vjp grad-input, bit-close.

Run: pixi run -e apple mojo run -I . tests/nn/test_repeat_storage.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.combinators.repeat import Repeat


comptime D = 5
comptime B = 4
comptime N = B * D
comptime REP3 = Repeat[3, Linear[D, D]]
comptime SEQ3 = Sequential[Linear[D, D], Linear[D, D], Linear[D, D]]
comptime REP1 = Repeat[1, Linear[D, D]]


def _close(ref a: Tensor, ref b: Tensor) -> Bool:
    for i in range(N):
        if abs(a.data[i] - b.data[i]) > Scalar[DT](1e-4):
            return False
    return True


def _check[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    var rep = REP3.make[target, Deterministic](ctx)
    var seq = SEQ3.make[target, Deterministic](ctx)
    var rep1 = REP1.make[target, Deterministic](ctx)
    var lin = Linear[D, D].make[target, Deterministic](ctx)

    var x = Tensor.alloc(N)
    var go = Tensor.alloc(N)
    var go2 = Tensor.alloc(N)
    var go3 = Tensor.alloc(N)
    var go4 = Tensor.alloc(N)
    for i in range(N):
        x.data[i] = Scalar[DT]((i % 5) - 2) * 0.2
        go.data[i] = Scalar[DT](((i * 3) % 7) - 3) * 0.15
        go2.data[i] = go.data[i]
        go3.data[i] = go.data[i]
        go4.data[i] = go.data[i]
    var o_r = Tensor.alloc(N); var o_s = Tensor.alloc(N)
    var o_r1 = Tensor.alloc(N); var o_l = Tensor.alloc(N)
    var gi_r = Tensor.alloc(N); var gi_s = Tensor.alloc(N)
    var gi_r1 = Tensor.alloc(N); var gi_l = Tensor.alloc(N)

    comptime if target == "gpu":
        x.upload(ctx.value())
        go.upload(ctx.value()); go2.upload(ctx.value())
        go3.upload(ctx.value()); go4.upload(ctx.value())

    rep.forward[target, B](TensorRefs[1](x), o_r, ctx)
    seq.forward[target, B](TensorRefs[1](x), o_s, ctx)
    rep1.forward[target, B](TensorRefs[1](x), o_r1, ctx)
    lin.forward[target, B](TensorRefs[1](x), o_l, ctx)
    rep.vjp[target, B](TensorRefs[1](x), go, TensorRefs[1](gi_r), ctx)
    seq.vjp[target, B](TensorRefs[1](x), go2, TensorRefs[1](gi_s), ctx)
    rep1.vjp[target, B](TensorRefs[1](x), go3, TensorRefs[1](gi_r1), ctx)
    lin.vjp[target, B](TensorRefs[1](x), go4, TensorRefs[1](gi_l), ctx)

    comptime if target == "gpu":
        o_r.download(ctx.value()); o_s.download(ctx.value())
        o_r1.download(ctx.value()); o_l.download(ctx.value())
        gi_r.download(ctx.value()); gi_s.download(ctx.value())
        gi_r1.download(ctx.value()); gi_l.download(ctx.value())

    return (
        _close(o_r, o_s) and _close(gi_r, gi_s)
        and _close(o_r1, o_l) and _close(gi_r1, gi_l)
    )


def main() raises:
    print("Repeat[N, Inner] storage correctness")
    var oc = _check["cpu"](None)
    print("  CPU:", "OK" if oc else "FAIL")
    var c = DeviceContext()
    var og = _check["gpu"](Optional(c))
    print("  GPU:", "OK" if og else "FAIL")
    assert_true(oc and og, "Repeat correctness")
    print("REPEAT OK")
