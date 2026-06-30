"""Dreamer4 perceptual (LPIPS-style) feature loss — correctness gate (CPU).

Validates `perceptual_feature_loss` (frozen CifarBackbone feature-MSE) end to end
with a RANDOM-INIT backbone (no checkpoint needed): finite loss, non-zero grad,
and — the real check — finite-difference agreement of the gradient w.r.t. the
tokenizer's patch-space output (the vjp chain: gray-replicate / unpatchify /
backbone.vjp / patchify).

The backbone runs in BN-EVAL mode (frozen, running-stat normalization — what a
trained checkpoint uses). A random 20-layer net in eval with init stats (0,1)
explodes, so we first CALIBRATE the running stats with a few train-mode forwards
on representative images (this is what training would produce). The validated
eval-mode BN backward is gated separately in `test_batch_norm_2d_eval_vjp.mojo`.

The backbone is piecewise-linear (Conv / BN / ReLU), so per-entry relative FD
error EXPLODES near a ReLU kink wherever the true slope is tiny — a property of
finite-differencing a kinked function, not a vjp bug. The gradient check is
therefore restricted to the SIGNIFICANT-magnitude gradient entries
(|grad| > 0.3·max|grad|), where the local slope is large/stable and central
differences are reliable, and aggregated as a relative-L2 error.

Run: pixi run mojo run -I . tests/nn/test_dreamer4_perceptual_loss.mojo
"""

from std.math import abs
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.models.cifar_feature_net import CifarBackbone
from mojo_rl.deep_agents.dreamer4.perceptual_loss import perceptual_feature_loss
from mojo_rl.deep_agents.dreamer4.shortcut_loss import _mao


def main() raises:
    comptime C_IMG = 1
    comptime H = 32
    comptime W = 32
    comptime PATCH = 8
    comptime BT = 8                              # B*T frames (stable batch stats)
    comptime NP = (H // PATCH) * (W // PATCH)    # 16
    comptime DP = C_IMG * PATCH * PATCH          # 64
    comptime NPD = BT * NP * DP                  # 8192

    print("Dreamer4 perceptual feature-loss gate (CPU)")

    comptime BB = CifarBackbone[H, W]
    var bb = BB.make["cpu", Deterministic](None)

    # Calibrate BN running stats so eval-mode normalization is well-conditioned
    # (a random 20-layer net in eval with init stats (0,1) explodes; a trained
    # backbone has sensible stats). Run train-mode forwards on representative
    # gray-replicated images; momentum 0.1 → ~50 iters converges the EMA.
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
    # perceptual_feature_loss sets BN-eval internally and uses these stats.

    var pred = List[Scalar[DT]](length=NPD, fill=Scalar[DT](0))
    var tgt = List[Scalar[DT]](length=NPD, fill=Scalar[DT](0))
    for i in range(NPD):
        pred[i] = Scalar[DT]((i % 13) - 6) * 0.05 + 0.5
        tgt[i] = Scalar[DT]((i % 11) - 5) * 0.04 + 0.5
    var grad = List[Scalar[DT]](length=NPD, fill=Scalar[DT](0))

    var loss = perceptual_feature_loss[BT, C_IMG, H, W, PATCH](
        _mao(pred.unsafe_ptr()), _mao(tgt.unsafe_ptr()), bb,
        _mao(grad.unsafe_ptr()),
    )
    print("  loss =", loss)
    var ok = (loss == loss) and (loss > 0.0)
    var gsum: Float64 = 0.0
    for i in range(NPD):
        if not (grad[i] == grad[i]):
            ok = False
        gsum += abs(Float64(grad[i]))
    print("  Σ|grad| =", gsum)
    if not (gsum > 0.0):
        ok = False

    # Sample a spread of entries; FD-check the SIGNIFICANT-magnitude ones.
    comptime NSAMP = 32
    comptime STRIDE = NPD // NSAMP
    var an_s = List[Float64]()
    var fd_s = List[Float64]()
    var maxabs: Float64 = 0.0
    var eps = Scalar[DT](1.0e-4)
    var grad2 = List[Scalar[DT]](length=NPD, fill=Scalar[DT](0))
    for s in range(NSAMP):
        var k = s * STRIDE
        var saved = pred[k]
        pred[k] = saved + eps
        var lp = perceptual_feature_loss[BT, C_IMG, H, W, PATCH](
            _mao(pred.unsafe_ptr()), _mao(tgt.unsafe_ptr()), bb,
            _mao(grad2.unsafe_ptr()),
        )
        pred[k] = saved - eps
        var lm = perceptual_feature_loss[BT, C_IMG, H, W, PATCH](
            _mao(pred.unsafe_ptr()), _mao(tgt.unsafe_ptr()), bb,
            _mao(grad2.unsafe_ptr()),
        )
        pred[k] = saved
        var an = Float64(grad[k])
        var fd = (lp - lm) / (2.0 * Float64(eps))
        an_s.append(an)
        fd_s.append(fd)
        if abs(an) > maxabs:
            maxabs = abs(an)

    # Aggregate relative-L2 over the significant-gradient subset.
    var thresh = 0.3 * maxabs
    var num: Float64 = 0.0
    var den: Float64 = 0.0
    var ncheck = 0
    for s in range(NSAMP):
        if abs(an_s[s]) > thresh:
            var d = fd_s[s] - an_s[s]
            num += d * d
            den += an_s[s] * an_s[s]
            ncheck += 1
            print("  [sig] an =", an_s[s], " fd =", fd_s[s])
    var rel_l2 = (num**0.5) / (den**0.5 + 1.0e-9)
    print("  checked", ncheck, "significant entries; rel-L2 =", rel_l2)
    var fd_ok = (ncheck >= 4) and (rel_l2 < 0.05)

    print("  loss finite+positive, grad nonzero:", "OK" if ok else "FAIL")
    print("  vjp finite-difference agreement:", "OK" if fd_ok else "FAIL")
    assert_true(ok, "perceptual loss finite + nonzero grad")
    assert_true(fd_ok, "perceptual loss vjp matches finite difference")
    print("DREAMER4 PERCEPTUAL LOSS GATE OK")
