"""AlphaZero CNN + ResNet torsos — build + forward shape/finiteness smoke.

Confirms the conv/resnet net variants share the MLP's external contract
(IN_DIMS[0]==OBS, OUT_DIM==ACT+1) and produce finite [policy | value] output
through the GPU forward path, in eval (BN running-stats) mode.

Run (Apple Metal):
    pixi run -e apple mojo run -I . tests/deep_agents/test_az_nets_cnn_resnet_smoke.mojo
"""

from std.testing import assert_equal, assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.deep_agents.alphazero.nets import (
    AZMLPNet, AZTicTacToeCNN, AZTicTacToeResNet,
)


def _forward_finite[NET: Module, OBS: Int, W: Int, B: Int](
    ctx: DeviceContext, name: String
) raises:
    var net = NET.make["gpu", INIT=Kaiming](ctx=ctx)
    net.set_attr["training"](Scalar[DT](0.0))  # eval mode (BN running stats)

    var obs = ctx.enqueue_create_buffer[DT](B * OBS)
    var out = ctx.enqueue_create_buffer[DT](B * W)
    var obs_h = ctx.enqueue_create_host_buffer[DT](B * OBS)
    var out_h = ctx.enqueue_create_host_buffer[DT](B * W)
    ctx.synchronize()

    # A plausible canonical obs: a couple of one-hot planes set.
    for i in range(B * OBS):
        obs_h.unsafe_ptr()[i] = Scalar[DT](0.0)
    for b in range(B):
        obs_h.unsafe_ptr()[b * OBS + 0] = Scalar[DT](1.0)   # mine @ cell0
        obs_h.unsafe_ptr()[b * OBS + 9 + 1] = Scalar[DT](1.0)  # opp @ cell1
        for c in range(2, 9):
            obs_h.unsafe_ptr()[b * OBS + 18 + c] = Scalar[DT](1.0)  # empty
    ctx.enqueue_copy(obs, obs_h)
    ctx.synchronize()

    var in_t = TileTensor(
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](obs.unsafe_ptr()),
        row_major[B, OBS](),
    )
    var out_t = TileTensor(
        rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](out.unsafe_ptr()),
        row_major[B, W](),
    )
    net.forward["gpu", B](in_t, output=out_t)
    ctx.enqueue_copy(out_h, out)
    ctx.synchronize()

    var all_finite = True
    for i in range(B * W):
        var v = Float64(out_h.unsafe_ptr()[i])
        if not (v == v) or v > 1e30 or v < -1e30:
            all_finite = False
    assert_true(all_finite, name + ": non-finite output")
    print(name, " OUT_DIM=", W, " forward finite OK")


def main() raises:
    comptime OBS = 27
    comptime ACT = 9
    comptime W = ACT + 1
    comptime B = 8

    # Contract: all three torsos expose IN_DIMS[0]==OBS and OUT_DIM==ACT+1.
    assert_equal(AZMLPNet[OBS, ACT, 32].OUT_DIM, W, "MLP OUT_DIM")
    assert_equal(AZTicTacToeCNN[16, 32].OUT_DIM, W, "CNN OUT_DIM")
    assert_equal(AZTicTacToeResNet[16, 2, 32].OUT_DIM, W, "ResNet OUT_DIM")
    assert_equal(AZTicTacToeCNN[16, 32].IN_DIMS[0], OBS, "CNN IN_DIM")
    assert_equal(AZTicTacToeResNet[16, 2, 32].IN_DIMS[0], OBS, "ResNet IN_DIM")

    var ctx = DeviceContext()
    _forward_finite[AZMLPNet[OBS, ACT, 32], OBS, W, B](ctx, "MLP")
    _forward_finite[AZTicTacToeCNN[16, 32], OBS, W, B](ctx, "CNN")
    _forward_finite[AZTicTacToeResNet[16, 2, 32], OBS, W, B](ctx, "ResNet")
    print("AZ CNN/ResNet nets smoke: OK")
