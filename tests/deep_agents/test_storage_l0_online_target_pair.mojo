"""L0 gate — storage OnlineTargetPair: init hard-copy + polyak soft-update.

Verifies (CPU + GPU):
  1. after make[target, Xavier], online and target params are IDENTICAL
     (hard copy via polyak_from tau=1.0), despite random init.
  2. polyak_step(tau): target moves to tau·online + (1-tau)·target. After
     perturbing online by +Δ then stepping tau, target shifts by tau·Δ.

Net under test = Sequential[Linear, ReLU, Linear] (the critic shape).

Run:
  pixi run mojo run -I . tests/deep_agents/test_storage_l0_online_target_pair.mojo
  pixi run -e apple mojo run -I . tests/deep_agents/test_storage_l0_online_target_pair.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.activations import ReLU
from mojo_rl.nn.storage.combinators.sequential import Sequential
from mojo_rl.nn.storage.core.initializer import Xavier
from mojo_rl.deep_agents.core.online_target_pair import OnlineTargetPair


comptime IN = 4
comptime H = 8
comptime W0 = IN * H  # first Linear weight count
comptime Net = Sequential[Linear[IN, H], ReLU[H], Linear[H, 1]]


def test_cpu() raises:
    print("OnlineTargetPair CPU: hard-copy + polyak ...")
    var pair = OnlineTargetPair[Net].make["cpu", Xavier]()
    # 1. identical after make
    var identical = True
    for i in range(W0):
        if (pair.online.children[0].weight.val.data[i]
                != pair.target_net.children[0].weight.val.data[i]):
            identical = False
    assert_true(identical, "online==target after hard copy")

    # 2. perturb online by +0.5, polyak tau=0.25 → target shifts +0.125
    var before = List[Scalar[DT]](capacity=W0)
    for i in range(W0):
        before.append(pair.target_net.children[0].weight.val.data[i])
        pair.online.children[0].weight.val.data[i] += Scalar[DT](0.5)
    pair.polyak_step["cpu"](Scalar[DT](0.25))
    var ok = True
    for i in range(W0):
        var expect = before[i] + Scalar[DT](0.25) * Scalar[DT](0.5)
        if abs(pair.target_net.children[0].weight.val.data[i] - expect) > 1e-6:
            ok = False
    assert_true(ok, "polyak target = tau·online + (1-tau)·target")
    print("  ok")


def test_gpu() raises:
    print("OnlineTargetPair GPU: hard-copy + polyak ...")
    var c = DeviceContext()
    var pair = OnlineTargetPair[Net].make["gpu", Xavier](Optional(c))
    # download both first-Linear weights, check identical
    pair.online.children[0].weight.val.download(c)
    pair.target_net.children[0].weight.val.download(c)
    var identical = True
    for i in range(W0):
        if (pair.online.children[0].weight.val.data[i]
                != pair.target_net.children[0].weight.val.data[i]):
            identical = False
    assert_true(identical, "gpu online==target after hard copy")

    # capture target before, perturb online on host+reupload, polyak, verify
    var before = List[Scalar[DT]](capacity=W0)
    for i in range(W0):
        before.append(pair.target_net.children[0].weight.val.data[i])
        pair.online.children[0].weight.val.data[i] += Scalar[DT](0.5)
    pair.online.children[0].weight.val.upload(c)
    pair.polyak_step["gpu"](Scalar[DT](0.25), Optional(c))
    pair.target_net.children[0].weight.val.download(c)
    var ok = True
    for i in range(W0):
        var expect = before[i] + Scalar[DT](0.25) * Scalar[DT](0.5)
        if abs(pair.target_net.children[0].weight.val.data[i] - expect) > 1e-5:
            ok = False
    assert_true(ok, "gpu polyak target = tau·online + (1-tau)·target")
    print("  ok")


def main() raises:
    print("=" * 60)
    print("L0 storage OnlineTargetPair gate")
    print("=" * 60)
    test_cpu()
    test_gpu()
    print("ALL PASSED")
