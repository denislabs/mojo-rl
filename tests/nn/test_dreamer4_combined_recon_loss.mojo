"""Dreamer 4 combined tokenizer recon loss: MSE + 0.2·perceptual (CPU gate).

Checks `masked_recon_plus_perceptual_loss` (paper eq. 5 wiring):
  • combined loss + combined patch-space gradient are finite and non-zero,
  • the returned (mse, perceptual) terms are finite,
  • with perc_weight == 0 it reduces to `masked_recon_loss` EXACTLY (same grad,
    same MSE) — i.e. the perceptual term is purely additive.

Run: pixi run mojo run -I . tests/nn/test_dreamer4_combined_recon_loss.mojo
"""

from std.math import abs
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.models.cifar_feature_net import CifarBackbone
from mojo_rl.deep_agents.dreamer4.recon_loss import masked_recon_loss
from mojo_rl.deep_agents.dreamer4.perceptual_loss import (
    masked_recon_plus_perceptual_loss,
)
from mojo_rl.deep_agents.dreamer4.shortcut_loss import _mao


def main() raises:
    print("Dreamer4 combined recon loss (MSE + 0.2·perceptual) gate (CPU)")
    comptime C_IMG = 1
    comptime H = 32
    comptime W = 32
    comptime PATCH = 8
    comptime BT = 4
    comptime NP = (H // PATCH) * (W // PATCH)   # 16
    comptime DP = C_IMG * PATCH * PATCH         # 64
    comptime NPD = BT * NP * DP

    comptime BB = CifarBackbone[H, W]
    var bb = BB.make["cpu", Deterministic](None)

    # calibrate BN running stats (eval-mode conditioning; see perceptual gate)
    comptime IMG3 = 3 * H * W
    var cal = Tensor.alloc(BT * IMG3)
    for bt in range(BT):
        for i in range(H * W):
            var v = Scalar[DT]((((bt * 131 + i) % 13) - 6)) * 0.05 + 0.5
            for k in range(3):
                cal.data[bt * IMG3 + k * H * W + i] = v
    var cal_out = Tensor.alloc(BT * BB.OUT_DIM)
    bb.set_attr["training"](Scalar[DT](1.0))
    for _ in range(50):
        bb.forward["cpu", BT](TensorRefs[1](cal), cal_out, None)

    var pred = List[Scalar[DT]](length=NPD, fill=Scalar[DT](0))
    var tgt = List[Scalar[DT]](length=NPD, fill=Scalar[DT](0))
    for i in range(NPD):
        pred[i] = Scalar[DT]((i % 13) - 6) * 0.05 + 0.5
        tgt[i] = Scalar[DT]((i % 11) - 5) * 0.04 + 0.5
    # MAE keep flags: mask roughly half the patches (0 = dropped → in MSE).
    var keep = List[Scalar[DT]](length=BT * NP, fill=Scalar[DT](1))
    for j in range(BT * NP):
        if j % 2 == 0:
            keep[j] = Scalar[DT](0)

    var grad = List[Scalar[DT]](length=NPD, fill=Scalar[DT](0))
    var grad_perc = List[Scalar[DT]](length=NPD, fill=Scalar[DT](0))

    # combined, w = 0.2
    var lv = masked_recon_plus_perceptual_loss[BT, C_IMG, H, W, PATCH](
        _mao(pred.unsafe_ptr()), _mao(tgt.unsafe_ptr()), _mao(keep.unsafe_ptr()),
        bb, 0.2, _mao(grad.unsafe_ptr()), _mao(grad_perc.unsafe_ptr()),
    )
    print("  mse =", lv[0], " perceptual =", lv[1])
    var ok = (lv[0] == lv[0]) and (lv[1] == lv[1]) and (lv[1] > 0.0)
    var gsum: Float64 = 0.0
    for i in range(NPD):
        if not (grad[i] == grad[i]):
            ok = False
        gsum += abs(Float64(grad[i]))
    print("  Σ|combined grad| =", gsum)
    ok = ok and (gsum > 0.0)

    # w = 0 must reduce to masked_recon_loss exactly (same grad, same mse).
    var grad_ref = List[Scalar[DT]](length=NPD, fill=Scalar[DT](0))
    var mse_ref = masked_recon_loss[NP, DP, BT](
        _mao(pred.unsafe_ptr()), _mao(tgt.unsafe_ptr()),
        _mao(keep.unsafe_ptr()), _mao(grad_ref.unsafe_ptr()),
    )
    var grad0 = List[Scalar[DT]](length=NPD, fill=Scalar[DT](0))
    var lv0 = masked_recon_plus_perceptual_loss[BT, C_IMG, H, W, PATCH](
        _mao(pred.unsafe_ptr()), _mao(tgt.unsafe_ptr()), _mao(keep.unsafe_ptr()),
        bb, 0.0, _mao(grad0.unsafe_ptr()), _mao(grad_perc.unsafe_ptr()),
    )
    var maxd: Float64 = 0.0
    for i in range(NPD):
        var d = abs(Float64(grad0[i]) - Float64(grad_ref[i]))
        if d > maxd:
            maxd = d
    var reduce_ok = (abs(lv0[0] - mse_ref) < 1.0e-9) and (lv0[1] == 0.0) and (
        maxd < 1.0e-12
    )
    print("  w=0 reduction: mse Δ =", abs(lv0[0] - mse_ref),
          " grad max|Δ| =", maxd)

    print("  combined finite + grad nonzero:", "OK" if ok else "FAIL")
    print("  w=0 reduces to masked MSE:", "OK" if reduce_ok else "FAIL")
    assert_true(ok, "combined recon loss finite + nonzero grad")
    assert_true(reduce_ok, "perc_weight=0 reduces to masked_recon_loss")
    print("DREAMER4 COMBINED RECON LOSS GATE OK")
