"""Phase B optimizer extras: schedules + AdamW alias + grad-norm clipping.

- LinearWarmupSchedule: ramp 0→lr over warmup, then constant; warmup=0 → const.
- AdamW alias constructs + steps (decoupled decay = Adam(wd>0), gated by tests
  test_adam_storage +decay/-decay).
- clip_grad_norm (CPU+GPU): populate grads via forward+vjp, then
    n0 = clip(max=1e9)  -> true norm, no clip
    n1 = clip(max=n0/2) -> returns n0 (pre-clip), grads now scaled by 0.5
    n2 = clip(max=1e9)  -> returns ~n0/2 (the rescaled norm)
  so n2 ≈ n0/2 validates BOTH the norm computation and the in-place scaling.

Run: pixi run -e apple mojo run -I . tests/nn/test_optimizer_extras_storage.mojo
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.optimizer.schedules import LinearWarmupSchedule
from mojo_rl.nn.optimizer.adam import Adam, AdamW
from mojo_rl.nn.optimizer.grad_clip import clip_grad_norm


comptime D = 4
comptime H = 6
comptime O = 3
comptime B = 5
comptime NET = Sequential[Linear[D, H], Linear[H, O]]


def _schedules_ok() -> Bool:
    var s = LinearWarmupSchedule.make(Scalar[DT](0.1), 10)
    var ok = True
    ok = ok and s.lr_at(0) == Scalar[DT](0.0)
    ok = ok and abs(s.lr_at(5) - Scalar[DT](0.05)) < Scalar[DT](1e-6)
    ok = ok and abs(s.lr_at(10) - Scalar[DT](0.1)) < Scalar[DT](1e-6)
    ok = ok and abs(s.lr_at(99) - Scalar[DT](0.1)) < Scalar[DT](1e-6)
    var c = LinearWarmupSchedule.make(Scalar[DT](0.2), 0)
    ok = ok and c.lr_at(0) == Scalar[DT](0.2)  # warmup=0 → constant
    return ok


def _clip_ok[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    var net = NET.make[target, Deterministic](ctx)
    # Populate param grads via a forward + vjp with a nonzero grad_output.
    var x = Tensor.alloc(B * D)
    var go = Tensor.alloc(B * O)
    for i in range(B * D):
        x.data[i] = Scalar[DT]((i % 5) - 2) * 0.3
    for i in range(B * O):
        go.data[i] = Scalar[DT](((i * 3) % 7) - 3) * 0.4
    var out = Tensor.alloc(B * O)
    var gi = Tensor.alloc(B * D)
    comptime if target == "gpu":
        x.upload(ctx.value()); go.upload(ctx.value())
    net.forward[target, B](TensorRefs[1](x), out, ctx)
    net.vjp[target, B](TensorRefs[1](x), go, TensorRefs[1](gi), ctx)

    var n0 = clip_grad_norm[target](net, Scalar[DT](1e9), ctx)  # no clip
    if not (n0 > Scalar[DT](0.0)):
        return False
    var n1 = clip_grad_norm[target](net, n0 * Scalar[DT](0.5), ctx)  # halve
    if abs(n1 - n0) > Scalar[DT](1e-3):  # returns PRE-clip norm
        return False
    var n2 = clip_grad_norm[target](net, Scalar[DT](1e9), ctx)  # measure again
    return abs(n2 - n0 * Scalar[DT](0.5)) < Scalar[DT](1e-3)


def main() raises:
    print("Phase B optimizer extras (schedules / AdamW / grad_clip)")
    var so = _schedules_ok()
    print("  schedules:", "OK" if so else "FAIL")

    # AdamW alias is usable (decoupled-decay math gated by test_adam_storage).
    var aw = AdamW(lr=1e-3, wd=0.01)
    var adam_ok = aw.wd == Scalar[DT](0.01)
    print("  AdamW alias:", "OK" if adam_ok else "FAIL")

    var cc = _clip_ok["cpu"](None)
    print("  clip CPU:", "OK" if cc else "FAIL")
    var c = DeviceContext()
    var cg = _clip_ok["gpu"](Optional(c))
    print("  clip GPU:", "OK" if cg else "FAIL")

    assert_true(so and adam_ok and cc and cg, "Phase B optimizer extras")
    print("OPTIMIZER EXTRAS OK")
