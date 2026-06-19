"""Parallel[A, B] correctness (storage surface, CPU + GPU).

vs standalone identical branches (deterministic init):
  forward:  out[:, :OUT_A] == A(x),  out[:, OUT_A:] == B(x)
  backward: grad_input == A.vjp(go[:, :OUT_A]) + B.vjp(go[:, OUT_A:])

Run: pixi run -e apple mojo run -I . tests/nn/test_parallel_storage.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.combinators.parallel import Parallel


comptime D = 5
comptime DA = 3
comptime DB = 2
comptime OUT = DA + DB
comptime B = 4
comptime PAR = Parallel[Linear[D, DA], Linear[D, DB]]


def _check[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime TOL = Scalar[DT](1e-4)
    var par = PAR.make_cpu() if target == "cpu" else PAR.make_gpu(ctx.value())
    var a = Linear[D, DA].make_cpu() if target == "cpu" else Linear[D, DA].make_gpu(ctx.value())
    var bb = Linear[D, DB].make_cpu() if target == "cpu" else Linear[D, DB].make_gpu(ctx.value())

    var x = Tensor.alloc(B * D)
    var go = Tensor.alloc(B * OUT)
    for i in range(B * D):
        x.data[i] = Scalar[DT]((i % 5) - 2) * 0.3
    for i in range(B * OUT):
        go.data[i] = Scalar[DT](((i * 3) % 7) - 3) * 0.2
    # standalone branch grad-output halves
    var go_a = Tensor.alloc(B * DA)
    var go_b = Tensor.alloc(B * DB)
    for b in range(B):
        for j in range(DA):
            go_a.data[b * DA + j] = go.data[b * OUT + j]
        for j in range(DB):
            go_b.data[b * DB + j] = go.data[b * OUT + DA + j]

    var out = Tensor.alloc(B * OUT)
    var oa = Tensor.alloc(B * DA)
    var ob = Tensor.alloc(B * DB)
    var gi = Tensor.alloc(B * D)
    var gia = Tensor.alloc(B * D)
    var gib = Tensor.alloc(B * D)

    comptime if target == "gpu":
        x.upload(ctx.value()); go.upload(ctx.value())
        go_a.upload(ctx.value()); go_b.upload(ctx.value())

    par.forward[target, B](TensorRefs[1].of1(x), out, ctx)
    a.forward[target, B](TensorRefs[1].of1(x), oa, ctx)
    bb.forward[target, B](TensorRefs[1].of1(x), ob, ctx)
    par.vjp[target, B](TensorRefs[1].of1(x), go, TensorRefs[1].of1(gi), ctx)
    a.vjp[target, B](TensorRefs[1].of1(x), go_a, TensorRefs[1].of1(gia), ctx)
    bb.vjp[target, B](TensorRefs[1].of1(x), go_b, TensorRefs[1].of1(gib), ctx)

    comptime if target == "gpu":
        out.download(ctx.value()); oa.download(ctx.value()); ob.download(ctx.value())
        gi.download(ctx.value()); gia.download(ctx.value()); gib.download(ctx.value())

    var ok = True
    for b in range(B):
        for j in range(DA):
            if abs(out.data[b * OUT + j] - oa.data[b * DA + j]) > TOL:
                ok = False
        for j in range(DB):
            if abs(out.data[b * OUT + DA + j] - ob.data[b * DB + j]) > TOL:
                ok = False
    for i in range(B * D):
        if abs(gi.data[i] - (gia.data[i] + gib.data[i])) > TOL:
            ok = False
    return ok


def main() raises:
    print("Parallel[A, B] storage correctness")
    var oc = _check["cpu"](None)
    print("  CPU:", "OK" if oc else "FAIL")
    var c = DeviceContext()
    var og = _check["gpu"](Optional(c))
    print("  GPU:", "OK" if og else "FAIL")
    assert_true(oc and og, "Parallel correctness")
    print("PARALLEL OK")
