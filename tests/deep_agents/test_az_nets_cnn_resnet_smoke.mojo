"""AlphaZero CNN + ResNet torsos — build + forward shape/finiteness smoke.

Confirms the conv/resnet net variants share the MLP's external contract
(IN_DIMS[0]==OBS, OUT_DIM==ACT+1) and produce finite [policy | value] output
through the GPU forward path, in eval (BN running-stats) mode.

Run (Apple Metal):
    pixi run -e apple mojo run -I . tests/deep_agents/test_az_nets_cnn_resnet_smoke.mojo
"""

from std.testing import assert_equal, assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.tensor import Tensor, TensorImpl
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.deep_agents.alphazero.nets import (
    AZMLPNet, AZTicTacToeCNN, AZTicTacToeResNet,
)


def _forward_finite[NET: Module, OBS: Int, W: Int, B: Int](
    ctx: DeviceContext, name: String
) raises:
    var net = NET.make["gpu", Kaiming](Optional(ctx))
    net.set_attr["training"](Scalar[DT](0.0))  # eval mode (BN running stats)

    # A plausible canonical obs: a couple of one-hot planes set (storage
    # Tensor). Allocated in the net's own activation dtype — `forward`
    # takes `TensorRefs[..., NET.ACT_DT]`, which the fp32 `Tensor` alias
    # only satisfies for an fp32 net.
    comptime ADT = NET.ACT_DT
    var obs_t = TensorImpl[ADT].alloc(B * OBS)
    for b in range(B):
        obs_t.data[b * OBS + 0] = Scalar[ADT](1.0)       # mine @ cell0
        obs_t.data[b * OBS + 9 + 1] = Scalar[ADT](1.0)   # opp @ cell1
        for c in range(2, 9):
            obs_t.data[b * OBS + 18 + c] = Scalar[ADT](1.0)  # empty
    obs_t.upload(ctx)

    var out_t = TensorImpl[ADT].alloc_gpu(ctx, B * W)
    # `TensorRefs`' ADT defaults to `DT`; `forward` wants the net's own
    # activation dtype, which no longer unifies with the default implicitly.
    net.forward["gpu", B](
        TensorRefs[NET.ARITY, ADT=ADT](obs_t), out_t, Optional(ctx)
    )
    out_t.download(ctx)

    var all_finite = True
    for i in range(B * W):
        var v = Float64(out_t.data[i])
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
