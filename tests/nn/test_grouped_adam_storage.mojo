"""GroupedAdam (single-kernel, contiguous arena) vs per-param Adam parity (GPU).

Two identically-initialized models trained K steps with identical inputs/grads —
one with per-param Adam (N launches/step), one with GroupedAdam (1 launch/step
over the arena). Same per-element math → param values must match to fp tolerance,
proving the arena rebind + grouped kernel are correct.

Run: pixi run -e apple mojo run -I . tests/nn/test_grouped_adam_storage.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.param import ParamVisitor
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.combinators.sequential import Sequential
from mojo_rl.nn.storage.optimizer.adam import Adam
from mojo_rl.nn.storage.optimizer.grouped_adam import GroupedAdam


comptime D = 4
comptime H = 6
comptime O = 3
comptime B = 5
comptime K = 4
comptime NET = Sequential[Linear[D, H], Linear[H, O]]


struct _ValCapture(ParamVisitor):
    var vals: List[Scalar[DT]]

    def __init__(out self):
        self.vals = List[Scalar[DT]]()

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        param.download(ctx.value())
        for i in range(N):
            self.vals.append(param.data[i])


def _feed(mut x: Tensor, mut go: Tensor, step: Int, ctx: DeviceContext) raises:
    for i in range(B * D):
        x.data[i] = Scalar[DT](((i + step) % 5) - 2) * 0.3
    for i in range(B * O):
        go.data[i] = Scalar[DT](((i * 3 + step) % 7) - 3) * 0.4
    x.upload(ctx)
    go.upload(ctx)


def main() raises:
    print("GroupedAdam vs per-param Adam parity (GPU)")
    var c = DeviceContext()

    # Model A — per-param Adam.
    var a = NET.make["gpu", Deterministic](Optional(c))
    var optA = Adam(lr=1e-2)
    for step in range(K):
        var x = Tensor.alloc(B * D); var go = Tensor.alloc(B * O)
        _feed(x, go, step, c)
        var out = Tensor.alloc(B * O); var gi = Tensor.alloc(B * D)
        a.zero_grad["gpu"](Optional(c))
        a.forward["gpu", B](TensorRefs[1](x), out, Optional(c))
        a.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), Optional(c))
        optA.step["gpu"](a, Optional(c))

    # Model B — GroupedAdam (arena, single kernel).
    var b = NET.make["gpu", Deterministic](Optional(c))
    var optB = GroupedAdam(lr=1e-2)
    optB.adopt(b, c)
    for step in range(K):
        var x = Tensor.alloc(B * D); var go = Tensor.alloc(B * O)
        _feed(x, go, step, c)
        var out = Tensor.alloc(B * O); var gi = Tensor.alloc(B * D)
        optB.zero_grad()
        b.forward["gpu", B](TensorRefs[1](x), out, Optional(c))
        b.vjp["gpu", B](TensorRefs[1](x), go, TensorRefs[1](gi), Optional(c))
        optB.step(c)

    var ca = _ValCapture(); a.for_each_param["gpu"](ca, Optional(c))
    var cb = _ValCapture(); b.for_each_param["gpu"](cb, Optional(c))
    if len(ca.vals) != len(cb.vals) or len(ca.vals) == 0:
        print("  FAIL: param count mismatch")
        assert_true(False, "param counts")
    var max_d = Scalar[DT](0.0)
    var moved = False
    for i in range(len(ca.vals)):
        var d = abs(ca.vals[i] - cb.vals[i])
        if d > max_d:
            max_d = d
        if abs(ca.vals[i]) > Scalar[DT](1e-4):
            moved = True  # params actually updated (not all-zero)
    print("  params:", len(ca.vals), " max|A-B| =", max_d, " updated:", moved)
    assert_true(moved and max_d < Scalar[DT](1e-5), "GroupedAdam parity")
    print("GROUPED ADAM OK")
