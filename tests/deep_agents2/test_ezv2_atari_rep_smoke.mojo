"""EZv2-Atari representation tower smoke (Stage 2) — shapes + finite grads, CPU.

Builds `EZRepNetResNetAtari[IN_CH=12, C=64]` (the official EZv2 Atari DownSample +
1-block RepresentationNetwork), asserts the flat in/out dims match the spatial
geometry (12×96×96 → [64,6,6] = 2304), runs forward + vjp on a tiny batch, and
checks outputs are finite and ≥90% of parameters receive a non-zero gradient.

Run:
    pixi run -e apple mojo run -I . tests/deep_agents2/test_ezv2_atari_rep_smoke.mojo
"""

from std.memory import alloc
from std.math import isnan, isinf
from std.gpu.memory import AddressSpace
from std.testing import assert_true, assert_equal
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.core import ParamVisitor
from mojo_rl.deep_agents2.efficient_zero_v2.nets_atari import EZRepNetResNetAtari


comptime IN_CH = 12      # 4 stacked RGB frames
comptime C = 64          # num_channels
comptime IMG = 96
comptime B = 2

comptime Rep = EZRepNetResNetAtari[IN_CH, C]
comptime IN = IN_CH * IMG * IMG     # 110592
comptime OUT = C * 6 * 6            # 2304


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


def _det(i: Int, scale: Float64) -> Scalar[DT]:
    var v = (Float64((i * 2654435761) % 1000) / 500.0) - 1.0
    return Scalar[DT](v * scale)


def _finite(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int) -> Bool:
    for i in range(n):
        if isnan(p[i]) or isinf(p[i]):
            return False
    return True


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


def main() raises:
    print("=" * 70)
    print("EZv2-Atari representation tower smoke (Stage 2, CPU)")
    print("=" * 70)

    # ── compile-time geometry ───────────────────────────────────────
    print("Rep.IN_DIMS[0] =", Rep.IN_DIMS[0], " (expect", IN, ")")
    print("Rep.OUT_DIM   =", Rep.OUT_DIM, " (expect", OUT, ")")
    assert_equal(Rep.IN_DIMS[0], IN, "rep input dim = 12*96*96")
    assert_equal(Rep.OUT_DIM, OUT, "rep latent dim = 64*6*6 = 2304")

    var m = Rep.make[target="cpu", INIT=Kaiming]()
    # BatchNorm train mode (EZ toggles eval around MCTS, train around update).
    m.set_attr["training"](Scalar[DT](1.0))

    var x = _a(B * IN); var y = _a(B * OUT)
    var go = _a(B * OUT); var gx = _a(B * IN)
    for k in range(B * IN):
        x[k] = _det(k + 1, 1.0)
    for k in range(B * OUT):
        go[k] = _det(k + 3, 1.0)

    var x_t = TileTensor(x, row_major[B, IN]())
    var y_t = TileTensor(y, row_major[B, OUT]())
    m.forward["cpu", B](x_t, output=y_t)
    assert_true(_finite(y, B * OUT), "rep output finite")

    var go_t = TileTensor(go, row_major[B, OUT]())
    var gx_t = TileTensor(gx, row_major[B, IN]())
    m.vjp["cpu", B](go_t, gx_t)
    assert_true(_finite(gx, B * IN), "rep grad_input finite")

    var gs = GradStats()
    m.for_each_param["cpu", GradStats]("rep", gs)
    print("   params=", gs.n_params, " nonzero-grad=", gs.n_nonzero)
    assert_true(gs.n_params > 0, "rep has params")
    assert_true(gs.n_nonzero * 10 >= gs.n_params * 9,
                "≥90% of rep params receive grad")
    _ = m^

    print("=" * 70)
    print("PASSED")
    print("=" * 70)
