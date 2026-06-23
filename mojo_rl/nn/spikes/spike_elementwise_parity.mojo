"""Elementwise parity harness (CPU): forward + vjp vs a direct ElementOp
reference, over the owns_cache=False (ReLU/Mish/GELU/Swish) and
owns_cache=True (Tanh/Sigmoid) families. Validates the storage plumbing
(ensure/SIMD-ptr/TensorRefs + the recompute-y backward) — the math itself is
the reused legacy ElementOp.

Run: pixi run mojo run -I . mojo_rl/nn/storage/spike_elementwise_parity.mojo
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.element_op import ElementOp
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.primitives.elementwise import Elementwise
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.ops.relu_op import ReLUOp
from mojo_rl.nn.primitives.ops.tanh_op import TanhOp
from mojo_rl.nn.primitives.ops.sigmoid_op import SigmoidOp
from mojo_rl.nn.primitives.ops.gelu_op import GELUOp
from mojo_rl.nn.primitives.ops.mish_op import MishOp


def _check[
    OP: ElementOp, B: Int, DIM: Int, target: StaticString
](name: String, ctx: Optional[DeviceContext]) raises -> Bool:
    comptime M = B * DIM
    var x = Tensor.alloc(M)
    for i in range(M):
        x.data[i] = (
            Scalar[DT]((i % 9) - 4) * 0.37
        )  # spans negatives + positives
    var go = Tensor.alloc(M)
    for i in range(M):
        go.data[i] = Scalar[DT]((i % 5) - 2) * 0.5
    var out = Tensor.alloc(M)
    var gi = Tensor.alloc(M)

    comptime if target == "cpu":
        var leaf = Elementwise[DIM, OP].make["cpu", Deterministic]()
        leaf.forward["cpu", B](TensorRefs[1](x), out, None)
        leaf.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), None)
    else:
        var c = ctx.value()
        var leaf = Elementwise[DIM, OP].make["gpu", Deterministic](Optional(c))
        x.upload(c)
        go.upload(c)
        leaf.forward["gpu", B](TensorRefs[1](x), out, ctx)
        leaf.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), ctx)
        out.download(c)
        gi.download(c)

    var ok = True
    for i in range(M):
        var ref_y = OP.forward_scalar(x.data[i])
        var c = ref_y if OP.owns_cache else x.data[i]
        var ref_gi = OP.backward_scalar(c, go.data[i])
        if abs(out.data[i] - ref_y) > 1e-6 or abs(gi.data[i] - ref_gi) > 1e-6:
            ok = False
    if ok:
        print("  ", name, "OK")
    else:
        print("  ", name, "FAIL")
    return ok


def _run[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    var ok = True
    ok = _check[ReLUOp, 4, 7, target]("ReLU   ", ctx) and ok
    ok = _check[TanhOp, 4, 7, target]("Tanh   ", ctx) and ok
    ok = _check[SigmoidOp, 4, 7, target]("Sigmoid", ctx) and ok
    ok = _check[GELUOp, 4, 7, target]("GELU   ", ctx) and ok
    ok = _check[MishOp, 4, 7, target]("Mish   ", ctx) and ok
    return ok


def main() raises:
    print("Elementwise parity (CPU):")
    var all_ok = _run["cpu"](None)
    print("Elementwise parity (GPU):")
    var c = DeviceContext()
    all_ok = _run["gpu"](Optional(c)) and all_ok
    if all_ok:
        print("ELEMENTWISE PARITY OK")
    else:
        print("ELEMENTWISE PARITY FAIL")
