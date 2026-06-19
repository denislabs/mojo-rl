"""polyak_from for NoisyLinear + Conv2D (target-net leaves) — CPU + GPU.

Regression guard for the Stage-5 polyak bug: Param-bearing leaves used as target
nets MUST override polyak_from (the Module default is a no-op that silently
freezes the target). LinearReLU was fixed; this gates the other two off-policy
target-net leaves (NoisyLinear → Rainbow/Noisy-DQN; Conv2D → pixel critics).

Two Deterministic-init instances are identical; perturb the online's first param
by Δ, polyak(tau=0.5) → target must move by 0.5·Δ on that param (non-zero).

Run:
  pixi run mojo run -I . tests/nn/test_polyak_from_leaves_storage.mojo
  pixi run -e apple mojo run -I . tests/nn/test_polyak_from_leaves_storage.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.primitives.noisy_linear import NoisyLinear
from mojo_rl.nn.storage.primitives.conv2d import Conv2D
from mojo_rl.nn.storage.core.initializer import Deterministic


def test_noisy_linear[target: StaticString](ctx: Optional[DeviceContext]) raises:
    print("NoisyLinear.polyak_from", target, "...")
    comptime IN = 4
    comptime OUT = 3
    var online = NoisyLinear[IN, OUT].make[target, Deterministic](ctx)
    var tgt = NoisyLinear[IN, OUT].make[target, Deterministic](ctx)
    comptime if target == "gpu":
        online.mu_w.val.download(ctx.value())
        tgt.mu_w.val.download(ctx.value())
    var before = tgt.mu_w.val.data[0]
    online.mu_w.val.data[0] = before + Scalar[DT](0.8)
    comptime if target == "gpu":
        online.mu_w.val.upload(ctx.value())
    tgt.polyak_from[target](online, Scalar[DT](0.5), ctx)
    comptime if target == "gpu":
        tgt.mu_w.val.download(ctx.value())
    var expect = before + Scalar[DT](0.5) * Scalar[DT](0.8)
    assert_true(abs(tgt.mu_w.val.data[0] - expect) < 1e-5, "noisy polyak mu_w[0]")
    print("  ok")


def test_conv2d[target: StaticString](ctx: Optional[DeviceContext]) raises:
    print("Conv2D.polyak_from", target, "...")
    comptime IC = 2
    comptime OC = 3
    comptime K = 3
    comptime HW = 8
    var online = Conv2D[IC, OC, K, 1, 1, HW, HW].make[target, Deterministic](ctx)
    var tgt = Conv2D[IC, OC, K, 1, 1, HW, HW].make[target, Deterministic](ctx)
    comptime if target == "gpu":
        online.weight.val.download(ctx.value())
        tgt.weight.val.download(ctx.value())
    var before = tgt.weight.val.data[0]
    online.weight.val.data[0] = before + Scalar[DT](0.8)
    comptime if target == "gpu":
        online.weight.val.upload(ctx.value())
    tgt.polyak_from[target](online, Scalar[DT](0.5), ctx)
    comptime if target == "gpu":
        tgt.weight.val.download(ctx.value())
    var expect = before + Scalar[DT](0.5) * Scalar[DT](0.8)
    assert_true(abs(tgt.weight.val.data[0] - expect) < 1e-5, "conv polyak w[0]")
    print("  ok")


def main() raises:
    print("=" * 60)
    print("polyak_from target-net leaves gate")
    print("=" * 60)
    test_noisy_linear["cpu"](None)
    test_conv2d["cpu"](None)
    var c = DeviceContext()
    test_noisy_linear["gpu"](Optional(c))
    test_conv2d["gpu"](Optional(c))
    print("ALL PASSED")
