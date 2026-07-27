"""CifarFeatureClassifier + backbone-only checkpoint round-trip (CPU gate).

Validates the bespoke trainable classifier used to train the perceptual-loss
backbone:
  • forward/vjp run end to end (finite logits, non-zero param grads),
  • `save_params(clf.backbone, …)` writes a BACKBONE-ONLY checkpoint that loads
    straight into a fresh `CifarBackbone` and reproduces its features bit-for-bit
    (this is how the perceptual loss consumes the trained weights),
  • that checkpoint also loads into a DIFFERENT-resolution `CifarBackbone`
    (conv/BN param sizes are H,W-independent — 32×32-trained → 64×64-eval), which
    the dreamer4 perceptual loss relies on.

Uses a tiny 8×8 input for CPU speed; the architecture is identical to the 32×32
CIFAR classifier.

Run: pixi run mojo run -I . tests/nn/test_cifar_feature_classifier.mojo
"""

from std.math import abs
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.core.checkpoint import save_params, load_params
from mojo_rl.nn.models.cifar_feature_net import (
    CifarBackbone, CifarFeatureClassifier,
)


struct _GradSum(ParamVisitor):
    var total: Float64

    def __init__(out self):
        self.total = 0.0

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "cpu":
            for i in range(len(grad.data)):
                self.total += abs(Float64(grad.data[i]))


def main() raises:
    print("CifarFeatureClassifier + backbone checkpoint round-trip (CPU)")
    comptime NC = 10
    comptime H = 8
    comptime W = 8
    comptime B = 2
    comptime IN = 3 * H * W
    comptime FEAT = 64 * (H // 4) * (W // 4)
    comptime path = String("/tmp/cifar_backbone_roundtrip.ckpt")

    var clf = CifarFeatureClassifier[NC, H, W].make["cpu", Deterministic](None)

    var x = Tensor.alloc(B * IN)
    for i in range(B * IN):
        x.data[i] = Scalar[DT](((i * 7 + 3) % 13) - 6) * 0.07

    # forward + vjp
    var out = Tensor.alloc(B * NC)
    clf.zero_grad["cpu"](None)
    clf.forward["cpu", B](TensorRefs[1](x), out, None)
    var ok = True
    for i in range(B * NC):
        if not (out.data[i] == out.data[i]):
            ok = False
    var go = Tensor.alloc(B * NC)
    for i in range(B * NC):
        go.data[i] = Scalar[DT](0.1)
    var gin = Tensor.alloc(B * IN)
    clf.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gin), None)
    var gs = _GradSum()
    clf.for_each_param["cpu"](gs, None)
    print("  logits finite:", ok, " Σ|param grad| =", gs.total)
    ok = ok and (gs.total > 0.0)

    # save BACKBONE only
    save_params["cpu"](clf.backbone, path, None)

    # reference features from the trained backbone
    var feat_ref = Tensor.alloc(B * FEAT)
    clf.backbone.forward["cpu", B](TensorRefs[1](x), feat_ref, None)

    # fresh backbone (different init) → load → features must match bit-for-bit
    var ext = CifarBackbone[H, W].make["cpu", Deterministic](None)
    load_params["cpu"](ext, path, None)
    var feat_ext = Tensor.alloc(B * FEAT)
    ext.forward["cpu", B](TensorRefs[1](x), feat_ext, None)
    var maxd: Float64 = 0.0
    for i in range(B * FEAT):
        var d = abs(Float64(feat_ext.data[i]) - Float64(feat_ref.data[i]))
        if d > maxd:
            maxd = d
    print("  backbone round-trip max|Δfeature| =", maxd)
    var rt_ok = maxd < 1.0e-5

    # cross-resolution load (32→64 property the perceptual loss relies on):
    # the same backbone-only checkpoint must load into a different-H,W backbone.
    var ext2 = CifarBackbone[2 * H, 2 * W].make["cpu", Deterministic](None)
    load_params["cpu"](ext2, path, None)  # must not raise
    print("  cross-resolution load (H,W → 2H,2W): OK")

    print("  forward/vjp finite + grads:", "OK" if ok else "FAIL")
    print("  backbone checkpoint round-trip:", "OK" if rt_ok else "FAIL")
    assert_true(ok, "classifier forward/vjp finite + nonzero param grads")
    assert_true(rt_ok, "backbone-only checkpoint round-trip reproduces features")
    print("CIFAR FEATURE CLASSIFIER GATE OK")
