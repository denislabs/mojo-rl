"""Dreamer 4 perceptual feature loss — CPU↔GPU parity gate.

`perceptual_feature_loss_gpu` runs the ResNet-20 backbone on device (host glue for
unpatchify / gray-replicate / feature-MSE / patchify). This checks it matches the
CPU `perceptual_feature_loss`: build identical (Deterministic) CPU + GPU backbones,
CALIBRATE both with the SAME train-mode forwards (so BN running stats match — a
random eval backbone explodes otherwise), then compare the perceptual loss + the
patch-space gradient.

Run: pixi run -e apple mojo run -I . tests/nn/test_dreamer4_perceptual_loss_gpu.mojo
"""

from std.math import abs
from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.core.checkpoint import save_params, load_params
from mojo_rl.nn.models.cifar_feature_net import CifarBackbone
from mojo_rl.deep_agents.dreamer4.perceptual_loss import (
    perceptual_feature_loss, perceptual_feature_loss_gpu,
)
from mojo_rl.deep_agents.dreamer4.shortcut_loss import _mao


def main() raises:
    print("Dreamer4 perceptual loss CPU↔GPU parity gate")
    comptime C_IMG = 1
    comptime H = 32
    comptime W = 32
    comptime PATCH = 8
    comptime BT = 4
    comptime NP = (H // PATCH) * (W // PATCH)
    comptime DP = C_IMG * PATCH * PATCH
    comptime NPD = BT * NP * DP
    comptime IMG3 = 3 * H * W
    comptime OUT = CifarBackbone[H, W].OUT_DIM

    var c = DeviceContext()
    var bb_c = CifarBackbone[H, W].make["cpu", Deterministic](None)
    var bb_g = CifarBackbone[H, W].make["gpu", Deterministic](Optional(c))

    # Calibrate the CPU backbone's BN running stats, then copy params + stats to
    # the GPU backbone via a checkpoint round-trip so BOTH are the SAME function.
    # (Calibrating each separately leaves ~1e-5 running-stat Δ, which near ReLU
    # kinks makes the two slightly different functions → the grads then diverge;
    # the GPU eval-BN vjp itself is validated in test_batch_norm_2d_eval_vjp_gpu.)
    var cal = Tensor.alloc(BT * IMG3)
    for bt in range(BT):
        for i in range(H * W):
            var v = Scalar[DT]((((bt * 131 + i) % 13) - 6)) * 0.05 + 0.5
            for k in range(3):
                cal.data[bt * IMG3 + k * H * W + i] = v
    var cal_out_c = Tensor.alloc(BT * OUT)
    bb_c.set_attr["training"](Scalar[DT](1.0))
    for _ in range(50):
        bb_c.forward["cpu", BT](TensorRefs[1](cal), cal_out_c, None)
    save_params["cpu"](bb_c, String("/tmp/bb_perc_parity.ckpt"), None)
    load_params["gpu"](bb_g, String("/tmp/bb_perc_parity.ckpt"), Optional(c))

    var pred = List[Scalar[DT]](length=NPD, fill=Scalar[DT](0))
    var tgt = List[Scalar[DT]](length=NPD, fill=Scalar[DT](0))
    for i in range(NPD):
        pred[i] = Scalar[DT]((i % 13) - 6) * 0.05 + 0.5
        tgt[i] = Scalar[DT]((i % 11) - 5) * 0.04 + 0.5

    var grad_c = List[Scalar[DT]](length=NPD, fill=Scalar[DT](0))
    var loss_c = perceptual_feature_loss[BT, C_IMG, H, W, PATCH](
        _mao(pred.unsafe_ptr()), _mao(tgt.unsafe_ptr()), bb_c,
        _mao(grad_c.unsafe_ptr()),
    )
    var grad_g = List[Scalar[DT]](length=NPD, fill=Scalar[DT](0))
    var loss_g = perceptual_feature_loss_gpu[BT, C_IMG, H, W, PATCH](
        _mao(pred.unsafe_ptr()), _mao(tgt.unsafe_ptr()), bb_g,
        _mao(grad_g.unsafe_ptr()), c,
    )

    var dloss = abs(loss_c - loss_g)
    var dgrad: Float64 = 0.0
    for i in range(NPD):
        var d = abs(Float64(grad_c[i]) - Float64(grad_g[i]))
        if d > dgrad:
            dgrad = d
    print("  loss  cpu=", loss_c, " gpu=", loss_g, " Δ=", dloss)
    print("  grad  max|Δ| =", dgrad)
    var ok = (loss_c == loss_c) and (loss_g == loss_g)
    ok = ok and (dloss < 1.0e-3) and (dgrad < 5.0e-3)
    print("  CPU↔GPU perceptual parity:", "OK" if ok else "FAIL")
    assert_true(ok, "perceptual loss CPU/GPU parity")
    print("DREAMER4 PERCEPTUAL LOSS GPU PARITY OK")
