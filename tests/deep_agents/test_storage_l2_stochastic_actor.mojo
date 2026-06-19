"""L2 gate — storage StochasticActor forward/vjp + toy train (CPU + GPU).

StochasticActor[OBS, ACT, *TRUNK] = trunk (Sequential) + Parallel[Linear,Linear]
heads -> [B, 2*ACT] packed [mu | log_std]. Verifies the migrated storage Module:
  - forward produces a finite [B, 2*ACT] output.
  - a toy MSE-to-fixed-target descent lowers the loss (forward + vjp + for_each_param
    all wired through the trunk and heads).

Run:
  pixi run mojo run -I . tests/deep_agents/test_storage_l2_stochastic_actor.mojo
  pixi run -e apple mojo run -I . tests/deep_agents/test_storage_l2_stochastic_actor.mojo
"""

from std.math import isnan, isinf
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.primitives.linear_relu import LinearReLU
from mojo_rl.nn.storage.core.initializer import Xavier
from mojo_rl.nn.storage.optimizer.adam import Adam
from mojo_rl.nn.storage.loss.mse_loss import MSELoss
from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor


comptime OBS = 3
comptime ACT = 2
comptime H = 32
comptime B = 8
comptime OUTD = 2 * ACT
comptime ACTOR = StochasticActor[OBS, ACT, LinearReLU[OBS, H], LinearReLU[H, H]]


def test_cpu() raises:
    print("StochasticActor CPU forward + toy train ...")
    var actor = ACTOR.make["cpu", Xavier]()
    var obs = Tensor.alloc(B * OBS)
    for i in range(B * OBS):
        obs.data[i] = Scalar[DT]((i % 7) - 3) * 0.2
    var out = Tensor.alloc(B * OUTD)
    actor.forward["cpu", B](TensorRefs[1](obs), out)
    var finite = True
    for i in range(B * OUTD):
        if isnan(out.data[i]) or isinf(out.data[i]):
            finite = False
    assert_true(finite, "actor forward finite [B,2*ACT]")

    # toy train: fit out -> fixed target
    var tgt = Tensor.alloc(B * OUTD)
    for i in range(B * OUTD):
        tgt.data[i] = Scalar[DT]((i % 5) - 2) * 0.3
    var mse = MSELoss[OUTD].make_cpu()
    var opt = Adam(lr=1e-2)
    var grad = Tensor.alloc(B * OUTD)
    var ginp = Tensor.alloc(B * OBS)
    var first: Scalar[DT] = 0
    var last: Scalar[DT] = 0
    for step in range(120):
        actor.zero_grad["cpu"](None)
        actor.forward["cpu", B](TensorRefs[1](obs), out)
        var loss = mse.forward["cpu", B](out, tgt, None)
        if step == 0:
            first = loss
        last = loss
        mse.vjp["cpu", B](out, tgt, grad, None)
        actor.vjp["cpu", B](TensorRefs[1](obs), grad, TensorRefs[1](ginp))
        opt.step["cpu", M=ACTOR](actor)
    print("  loss", first, "->", last)
    assert_true(last < first * 0.5, "actor toy train lowers loss")
    print("  ok")


def test_gpu() raises:
    print("StochasticActor GPU forward + toy train ...")
    var c = DeviceContext()
    var actor = ACTOR.make["gpu", Xavier](Optional(c))
    var obs = Tensor.alloc(B * OBS)
    for i in range(B * OBS):
        obs.data[i] = Scalar[DT]((i % 7) - 3) * 0.2
    obs.upload(c)
    var out = Tensor.alloc_gpu(c, B * OUTD)
    var tgt = Tensor.alloc(B * OUTD)
    for i in range(B * OUTD):
        tgt.data[i] = Scalar[DT]((i % 5) - 2) * 0.3
    tgt.upload(c)
    var mse = MSELoss[OUTD].make_gpu(c)
    var opt = Adam(lr=1e-2)
    var grad = Tensor.alloc_gpu(c, B * OUTD)
    var ginp = Tensor.alloc_gpu(c, B * OBS)
    var first: Scalar[DT] = 0
    var last: Scalar[DT] = 0
    for step in range(120):
        actor.zero_grad["gpu"](Optional(c))
        actor.forward["gpu", B](TensorRefs[1](obs), out, Optional(c))
        var loss = mse.forward["gpu", B](out, tgt, Optional(c))
        if step == 0:
            first = loss
        last = loss
        mse.vjp["gpu", B](out, tgt, grad, Optional(c))
        actor.vjp["gpu", B](TensorRefs[1](obs), grad, TensorRefs[1](ginp), Optional(c))
        opt.step["gpu", M=ACTOR](actor, Optional(c))
    print("  loss", first, "->", last)
    assert_true(last < first * 0.6, "actor toy train lowers loss (gpu)")
    print("  ok")


def main() raises:
    print("=" * 60)
    print("L2 storage StochasticActor gate")
    print("=" * 60)
    test_cpu()
    test_gpu()
    print("ALL PASSED")
