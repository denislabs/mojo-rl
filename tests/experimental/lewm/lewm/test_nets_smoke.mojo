"""LeWM nn nets smoke (Phase C) — shapes + non-zero grads, CPU.

Builds LeWMEncoder, ActionEmbedder, ARPredictor, PredProj at toy dims;
runs forward + vjp; asserts finite outputs and that backward delivers
non-zero gradients to (almost) every parameter.
"""

from std.memory import alloc
from std.math import isnan, isinf
from std.gpu.memory import AddressSpace
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.core import ParamVisitor
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


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


def _det(i: Int, scale: Float64) -> Scalar[DT]:
    var v = (Float64((i * 2654435761) % 1000) / 500.0) - 1.0
    return Scalar[DT](v * scale)


struct GradStats(ParamVisitor):
    var n_params: Int
    var n_nonzero: Int

    def __init__(out self):
        self.n_params = 0
        self.n_nonzero = 0

    def visit(
        mut self, name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int, apply_decay: Bool,
    ) raises:
        self.n_params += 1
        var g = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad.ptr)
        var any: Bool = False
        for i in range(n_elems):
            if g[i] != Scalar[DT](0.0):
                any = True
                break
        if any:
            self.n_nonzero += 1


def _finite(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int) -> Bool:
    for i in range(n):
        if isnan(p[i]) or isinf(p[i]):
            return False
    return True


def test_encoder() raises:
    print("test_encoder ...")
    comptime IN = IN_CH * IMG * IMG
    comptime OUT = EMB
    var m = LeWMEncoder[
        IN_CH, IMG, PATCH, N_PATCHES, HIDDEN, ENC_HEADS, ENC_LAYERS,
        EMB, PROJ_H, FF_MULT
    ].make[target="cpu", INIT=Kaiming]()

    var x = _a(BT * IN); var y = _a(BT * OUT)
    var go = _a(BT * OUT); var gx = _a(BT * IN)
    for k in range(BT * IN):
        x[k] = _det(k + 1, 1.0)
    for k in range(BT * OUT):
        go[k] = _det(k + 3, 1.0)
    var x_t = TileTensor(x, row_major[BT, IN]())
    var y_t = TileTensor(y, row_major[BT, OUT]())
    m.forward["cpu", BT](x_t, output=y_t)
    assert_true(_finite(y, BT * OUT), "encoder output finite")

    var go_t = TileTensor(go, row_major[BT, OUT]())
    var gx_t = TileTensor(gx, row_major[BT, IN]())
    m.vjp["cpu", BT](go_t, gx_t)
    var gs = GradStats()
    m.for_each_param["cpu", GradStats]("enc", gs)
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
    var m = ActionEmbedder[T, ACT, SMOOTHED, EMB, FF_MULT].make[
        target="cpu", INIT=Kaiming
    ]()
    var x = _a(B * IN); var y = _a(B * OUT)
    var go = _a(B * OUT); var gx = _a(B * IN)
    for k in range(B * IN):
        x[k] = _det(k + 1, 1.0)
    for k in range(B * OUT):
        go[k] = _det(k + 3, 1.0)
    var x_t = TileTensor(x, row_major[B, IN]())
    var y_t = TileTensor(y, row_major[B, OUT]())
    m.forward["cpu", B](x_t, output=y_t)
    assert_true(_finite(y, B * OUT), "action_embedder output finite")
    var go_t = TileTensor(go, row_major[B, OUT]())
    var gx_t = TileTensor(gx, row_major[B, IN]())
    m.vjp["cpu", B](go_t, gx_t)
    var gs = GradStats()
    m.for_each_param["cpu", GradStats]("ae", gs)
    print("   params=", gs.n_params, " nonzero-grad=", gs.n_nonzero)
    assert_true(gs.n_nonzero * 10 >= gs.n_params * 9, "AE grads")
    _ = m^
    print("  ok")


def test_pred_proj() raises:
    print("test_pred_proj ...")
    comptime D = H * EMB
    var m = PredProj[H, EMB, PROJ_H].make[target="cpu", INIT=Kaiming]()
    var x = _a(B * D); var y = _a(B * D)
    var go = _a(B * D); var gx = _a(B * D)
    for k in range(B * D):
        x[k] = _det(k + 1, 1.0)
        go[k] = _det(k + 3, 1.0)
    var x_t = TileTensor(x, row_major[B, D]())
    var y_t = TileTensor(y, row_major[B, D]())
    m.forward["cpu", B](x_t, output=y_t)
    assert_true(_finite(y, B * D), "pred_proj output finite")
    var go_t = TileTensor(go, row_major[B, D]())
    var gx_t = TileTensor(gx, row_major[B, D]())
    m.vjp["cpu", B](go_t, gx_t)
    var gs = GradStats()
    m.for_each_param["cpu", GradStats]("pp", gs)
    print("   params=", gs.n_params, " nonzero-grad=", gs.n_nonzero)
    assert_true(gs.n_nonzero * 10 >= gs.n_params * 9, "pred_proj grads")
    _ = m^
    print("  ok")


def main() raises:
    print("=" * 70)
    print("LeWM nn nets smoke (Phase C, CPU)")
    print("=" * 70)
    test_encoder()
    test_action_embedder()
    test_pred_proj()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
