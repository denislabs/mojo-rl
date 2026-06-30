"""LeWM nn nets smoke (storage) — shapes + non-zero grads, CPU.

Builds LeWMEncoder, ActionEmbedder, ARPredictor, PredProj at toy dims;
runs forward + vjp on the storage surface (TensorRefs inputs + Tensor out);
asserts finite outputs and that backward delivers non-zero gradients to
(almost) every parameter. The ARPredictor case exercises the storage
`RepeatConditional` (ARITY=2 conditional block stack).
"""

from std.math import isnan, isinf
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn import Tensor, TensorRefs, TensorPack, ParamVisitor, Kaiming
from mojo_rl.experimental.lewm.encoder import (
    LeWMEncoder, ActionEmbedder, ARPredictor, PredProj,
)


# toy config
comptime IN_CH = 4
comptime IMG = 8
comptime PATCH = 4
comptime N_PATCHES = (IMG // PATCH) * (IMG // PATCH)  # 4
comptime HIDDEN = 8
comptime ENC_HEADS = 2
comptime ENC_LAYERS = 2
comptime EMB = 8
comptime PROJ_H = 16
comptime FF_MULT = 2

comptime T = 4
comptime ACT = 3
comptime SMOOTHED = 8
comptime H = 3
comptime PRED_HEADS = 2
comptime PRED_FF = 16
comptime DEPTH = 2

comptime B = 2
comptime BT = B * T


def _det(i: Int, scale: Float64) -> Scalar[DT]:
    var v = (Float64((i * 2654435761) % 1000) / 500.0) - 1.0
    return Scalar[DT](v * scale)


def _filled(n: Int, seed: Int, scale: Float64) raises -> Tensor:
    var t = Tensor.alloc(n)
    for k in range(n):
        t.data[k] = _det(k + seed, scale)
    return t^


def _finite(t: Tensor, n: Int) -> Bool:
    for i in range(n):
        if isnan(t.data[i]) or isinf(t.data[i]):
            return False
    return True


struct FillVisitor(ParamVisitor):
    """Overwrite every param with a small non-zero deterministic value, so an
    AdaLN-zero block's conditioning path is active (grad_c ≠ 0)."""
    var counter: Int

    def __init__(out self):
        self.counter = 0

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        for i in range(N):
            param.data[i] = _det(self.counter + 11, 0.3)
            self.counter += 1


struct GradStats(ParamVisitor):
    var n_params: Int
    var n_nonzero: Int

    def __init__(out self):
        self.n_params = 0
        self.n_nonzero = 0

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        self.n_params += 1
        var any: Bool = False
        for i in range(N):
            if grad.data[i] != Scalar[DT](0.0):
                any = True
                break
        if any:
            self.n_nonzero += 1


def test_encoder() raises:
    print("test_encoder ...")
    comptime IN = IN_CH * IMG * IMG
    comptime OUT = EMB
    var m = LeWMEncoder[
        IN_CH, IMG, PATCH, N_PATCHES, HIDDEN, ENC_HEADS, ENC_LAYERS,
        EMB, PROJ_H, FF_MULT
    ].make["cpu", Kaiming]()

    var x = _filled(BT * IN, 1, 1.0)
    var y = Tensor.alloc(BT * OUT)
    m.forward["cpu", BT](TensorRefs[1](x), y, None)
    assert_true(_finite(y, BT * OUT), "encoder output finite")

    var go = _filled(BT * OUT, 3, 1.0)
    var gx = Tensor.alloc(BT * IN)
    m.vjp["cpu", BT](TensorRefs[1](x), go, TensorRefs[1](gx), None)
    var gs = GradStats()
    m.for_each_param["cpu", GradStats](gs, None)
    print("   params=", gs.n_params, " nonzero-grad=", gs.n_nonzero)
    assert_true(gs.n_params > 0, "encoder has params")
    assert_true(gs.n_nonzero * 10 >= gs.n_params * 9,
                "≥90% of encoder params receive grad")
    _ = m^
    print("  ok")


def test_action_embedder() raises:
    print("test_action_embedder ...")
    comptime IN = T * ACT
    comptime OUT = T * EMB
    var m = ActionEmbedder[T, ACT, SMOOTHED, EMB, FF_MULT].make["cpu", Kaiming]()
    var x = _filled(B * IN, 1, 1.0)
    var y = Tensor.alloc(B * OUT)
    m.forward["cpu", B](TensorRefs[1](x), y, None)
    assert_true(_finite(y, B * OUT), "action_embedder output finite")
    var go = _filled(B * OUT, 3, 1.0)
    var gx = Tensor.alloc(B * IN)
    m.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gx), None)
    var gs = GradStats()
    m.for_each_param["cpu", GradStats](gs, None)
    print("   params=", gs.n_params, " nonzero-grad=", gs.n_nonzero)
    assert_true(gs.n_nonzero * 10 >= gs.n_params * 9, "AE grads")
    _ = m^
    print("  ok")


def test_pred_proj() raises:
    print("test_pred_proj ...")
    comptime D = H * EMB
    var m = PredProj[H, EMB, PROJ_H].make["cpu", Kaiming]()
    var x = _filled(B * D, 1, 1.0)
    var y = Tensor.alloc(B * D)
    m.forward["cpu", B](TensorRefs[1](x), y, None)
    assert_true(_finite(y, B * D), "pred_proj output finite")
    var go = _filled(B * D, 3, 1.0)
    var gx = Tensor.alloc(B * D)
    m.vjp["cpu", B](TensorRefs[1](x), go, TensorRefs[1](gx), None)
    var gs = GradStats()
    m.for_each_param["cpu", GradStats](gs, None)
    print("   params=", gs.n_params, " nonzero-grad=", gs.n_nonzero)
    assert_true(gs.n_nonzero * 10 >= gs.n_params * 9, "pred_proj grads")
    _ = m^
    print("  ok")


def test_ar_predictor() raises:
    """ARITY=2 RepeatConditional[DEPTH, ConditionalTransformerBlock] —
    forward(x, c) over the H-token context + conditioning, then vjp; both x and
    c grads must be finite and (nearly) every block param must receive grad."""
    print("test_ar_predictor ...")
    comptime D = H * EMB
    var m = ARPredictor[EMB, PRED_HEADS, H, PRED_FF, DEPTH].make[
        "cpu", Kaiming
    ]()
    # AdaLN-zero init zeroes the conditioning path → grad_c=0 at init. Fill
    # params with non-zero values so the c-path (and grad_c) is active.
    var fv = FillVisitor()
    m.for_each_param["cpu", FillVisitor](fv, None)
    # §B0: a binary module's two inputs must come from one owner (pool).
    var inp = TensorPack[2]()
    inp[0].ensure(B * D); inp[1].ensure(B * D)
    for k in range(B * D):
        inp[0].data[k] = _det(k + 1, 1.0)
        inp[1].data[k] = _det(k + 7, 1.0)
    var y = Tensor.alloc(B * D)
    m.forward["cpu", B](TensorRefs[2](inp[0], inp[1]), y, None)
    assert_true(_finite(y, B * D), "ar_predictor output finite")

    var go = _filled(B * D, 3, 1.0)
    var gpk = TensorPack[2]()
    m.vjp["cpu", B](
        TensorRefs[2](inp[0], inp[1]), go,
        TensorRefs[2](gpk[0], gpk[1]), None,
    )
    assert_true(_finite(gpk[0], B * D), "ar_predictor grad_x finite")
    assert_true(_finite(gpk[1], B * D), "ar_predictor grad_c finite")
    # c fans out to every block, so its accumulated grad must be non-zero.
    var gc_any: Bool = False
    for i in range(B * D):
        if gpk[1].data[i] != Scalar[DT](0.0):
            gc_any = True
            break
    assert_true(gc_any, "ar_predictor grad_c is non-zero (fan-out)")
    var gs = GradStats()
    m.for_each_param["cpu", GradStats](gs, None)
    print("   params=", gs.n_params, " nonzero-grad=", gs.n_nonzero)
    assert_true(gs.n_params > 0, "ar_predictor has params")
    assert_true(gs.n_nonzero * 10 >= gs.n_params * 9, "ar_predictor grads")
    _ = m^
    print("  ok")


def main() raises:
    print("=" * 70)
    print("LeWM nn nets smoke (storage, CPU)")
    print("=" * 70)
    test_encoder()
    test_action_embedder()
    test_pred_proj()
    test_ar_predictor()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
