"""BranchConcat[*BRANCHES] correctness (storage surface, CPU + GPU).

3 branches D->2, D->3, D->1. vs standalone identical branches (deterministic
init):
  forward:  out == [b0(x) | b1(x) | b2(x)]
  backward: grad_input == Σ b_i.vjp(grad_output[:, off_i:off_i+O_i])

Run: pixi run -e apple mojo run -I . tests/nn/test_branch_concat_storage.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.combinators.branch_concat import BranchConcat


comptime D = 5
comptime B = 4
comptime O0 = 2
comptime O1 = 3
comptime O2 = 1
comptime OUT = O0 + O1 + O2  # 6
comptime BC = BranchConcat[Linear[D, O0], Linear[D, O1], Linear[D, O2]]


def _check[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime TOL = Scalar[DT](1e-4)
    var bc = BC.make_cpu() if target == "cpu" else BC.make_gpu(ctx.value())
    var b0 = Linear[D, O0].make_cpu() if target == "cpu" else Linear[D, O0].make_gpu(ctx.value())
    var b1 = Linear[D, O1].make_cpu() if target == "cpu" else Linear[D, O1].make_gpu(ctx.value())
    var b2 = Linear[D, O2].make_cpu() if target == "cpu" else Linear[D, O2].make_gpu(ctx.value())

    var x = Tensor.alloc(B * D)
    var go = Tensor.alloc(B * OUT)
    for i in range(B * D):
        x.data[i] = Scalar[DT]((i % 5) - 2) * 0.25
    for i in range(B * OUT):
        go.data[i] = Scalar[DT](((i * 3) % 7) - 3) * 0.2
    # per-branch grad-output slices
    var g0 = Tensor.alloc(B * O0)
    var g1 = Tensor.alloc(B * O1)
    var g2 = Tensor.alloc(B * O2)
    for b in range(B):
        for j in range(O0):
            g0.data[b * O0 + j] = go.data[b * OUT + j]
        for j in range(O1):
            g1.data[b * O1 + j] = go.data[b * OUT + O0 + j]
        for j in range(O2):
            g2.data[b * O2 + j] = go.data[b * OUT + O0 + O1 + j]

    var out = Tensor.alloc(B * OUT)
    var o0 = Tensor.alloc(B * O0); var o1 = Tensor.alloc(B * O1); var o2 = Tensor.alloc(B * O2)
    var gi = Tensor.alloc(B * D)
    var gi0 = Tensor.alloc(B * D); var gi1 = Tensor.alloc(B * D); var gi2 = Tensor.alloc(B * D)

    comptime if target == "gpu":
        x.upload(ctx.value()); go.upload(ctx.value())
        g0.upload(ctx.value()); g1.upload(ctx.value()); g2.upload(ctx.value())

    bc.forward[target, B](TensorRefs[1].of1(x), out, ctx)
    b0.forward[target, B](TensorRefs[1].of1(x), o0, ctx)
    b1.forward[target, B](TensorRefs[1].of1(x), o1, ctx)
    b2.forward[target, B](TensorRefs[1].of1(x), o2, ctx)
    bc.vjp[target, B](TensorRefs[1].of1(x), go, TensorRefs[1].of1(gi), ctx)
    b0.vjp[target, B](TensorRefs[1].of1(x), g0, TensorRefs[1].of1(gi0), ctx)
    b1.vjp[target, B](TensorRefs[1].of1(x), g1, TensorRefs[1].of1(gi1), ctx)
    b2.vjp[target, B](TensorRefs[1].of1(x), g2, TensorRefs[1].of1(gi2), ctx)

    comptime if target == "gpu":
        out.download(ctx.value())
        o0.download(ctx.value()); o1.download(ctx.value()); o2.download(ctx.value())
        gi.download(ctx.value())
        gi0.download(ctx.value()); gi1.download(ctx.value()); gi2.download(ctx.value())

    var ok = True
    for b in range(B):
        for j in range(O0):
            if abs(out.data[b * OUT + j] - o0.data[b * O0 + j]) > TOL:
                ok = False
        for j in range(O1):
            if abs(out.data[b * OUT + O0 + j] - o1.data[b * O1 + j]) > TOL:
                ok = False
        for j in range(O2):
            if abs(out.data[b * OUT + O0 + O1 + j] - o2.data[b * O2 + j]) > TOL:
                ok = False
    for i in range(B * D):
        if abs(gi.data[i] - (gi0.data[i] + gi1.data[i] + gi2.data[i])) > TOL:
            ok = False
    return ok


def main() raises:
    print("BranchConcat[*BRANCHES] storage correctness")
    var oc = _check["cpu"](None)
    print("  CPU:", "OK" if oc else "FAIL")
    var c = DeviceContext()
    var og = _check["gpu"](Optional(c))
    print("  GPU:", "OK" if og else "FAIL")
    assert_true(oc and og, "BranchConcat correctness")
    print("BRANCH_CONCAT OK")
