"""Tokenwise[SEQ_LEN, Inner] correctness (storage surface, CPU + GPU).

Tokenwise[SEQ, Linear].forward[B] over a flat slab must equal a standalone
Linear.forward[B*SEQ] over the same slab (batch reinterpretation) — forward
AND vjp grad-input, bit-close.

Run: pixi run -e apple mojo run -I . tests/nn/test_tokenwise_storage.mojo
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.combinators.tokenwise import Tokenwise


comptime SEQ = 3
comptime DI = 4
comptime DO = 5
comptime B = 2
comptime BS = B * SEQ
comptime TW = Tokenwise[SEQ, Linear[DI, DO]]
comptime NX = B * SEQ * DI  # = BS*DI
comptime NO = B * SEQ * DO  # = BS*DO


def _check[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime TOL = Scalar[DT](1e-4)
    var tw = TW.make[target, Deterministic](ctx)
    var lin = Linear[DI, DO].make[target, Deterministic](ctx)

    var x = Tensor.alloc(NX)
    var go = Tensor.alloc(NO)
    var go2 = Tensor.alloc(NO)
    for i in range(NX):
        x.data[i] = Scalar[DT]((i % 5) - 2) * 0.2
    for i in range(NO):
        go.data[i] = Scalar[DT](((i * 3) % 7) - 3) * 0.15
        go2.data[i] = go.data[i]
    var o_tw = Tensor.alloc(NO); var o_l = Tensor.alloc(NO)
    var gi_tw = Tensor.alloc(NX); var gi_l = Tensor.alloc(NX)

    comptime if target == "gpu":
        x.upload(ctx.value()); go.upload(ctx.value()); go2.upload(ctx.value())

    tw.forward[target, B](TensorRefs[1](x), o_tw, ctx)
    lin.forward[target, BS](TensorRefs[1](x), o_l, ctx)
    tw.vjp[target, B](TensorRefs[1](x), go, TensorRefs[1](gi_tw), ctx)
    lin.vjp[target, BS](TensorRefs[1](x), go2, TensorRefs[1](gi_l), ctx)

    comptime if target == "gpu":
        o_tw.download(ctx.value()); o_l.download(ctx.value())
        gi_tw.download(ctx.value()); gi_l.download(ctx.value())

    var ok = True
    for i in range(NO):
        if abs(o_tw.data[i] - o_l.data[i]) > TOL:
            ok = False
    for i in range(NX):
        if abs(gi_tw.data[i] - gi_l.data[i]) > TOL:
            ok = False
    return ok


def main() raises:
    print("Tokenwise[SEQ_LEN, Inner] storage correctness")
    var oc = _check["cpu"](None)
    print("  CPU:", "OK" if oc else "FAIL")
    var c = DeviceContext()
    var og = _check["gpu"](Optional(c))
    print("  GPU:", "OK" if og else "FAIL")
    assert_true(oc and og, "Tokenwise correctness")
    print("TOKENWISE OK")
