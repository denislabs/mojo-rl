"""EZv2-Atari representation tower smoke (Stage 2) — shapes + finite grads, CPU.

Builds `EZRepNetResNetAtari[IN_CH=12, C=64]` (the official EZv2 Atari DownSample +
1-block RepresentationNetwork), asserts the flat in/out dims match the spatial
geometry (12×96×96 → [64,6,6] = 2304), runs forward + vjp on a tiny batch, and
checks outputs are finite and ≥90% of parameters receive a non-zero gradient.

Run:
    pixi run -e apple mojo run -I . tests/deep_agents/test_ezv2_atari_rep_smoke.mojo
"""

from std.math import isnan, isinf
from std.testing import assert_true, assert_equal
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.deep_agents.efficient_zero_v2.nets_atari import EZRepNetResNetAtari


comptime IN_CH = 12      # 4 stacked RGB frames
comptime C = 64          # num_channels
comptime IMG = 96
comptime B = 2

comptime Rep = EZRepNetResNetAtari[IN_CH, C]
comptime IN = IN_CH * IMG * IMG     # 110592
comptime OUT = C * 6 * 6            # 2304


def _det(i: Int, scale: Float64) -> Scalar[DT]:
    var v = (Float64((i * 2654435761) % 1000) / 500.0) - 1.0
    return Scalar[DT](v * scale)


def _finite(p: List[Scalar[DT]], n: Int) -> Bool:
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


def main() raises:
    print("=" * 70)
    print("EZv2-Atari representation tower smoke (Stage 2, CPU)")
    print("=" * 70)

    # ── compile-time geometry ───────────────────────────────────────
    print("Rep.IN_DIMS[0] =", Rep.IN_DIMS[0], " (expect", IN, ")")
    print("Rep.OUT_DIM   =", Rep.OUT_DIM, " (expect", OUT, ")")
    assert_equal(Rep.IN_DIMS[0], IN, "rep input dim = 12*96*96")
    assert_equal(Rep.OUT_DIM, OUT, "rep latent dim = 64*6*6 = 2304")

    var m = Rep.make["cpu", Kaiming]()
    # BatchNorm train mode (EZ toggles eval around MCTS, train around update).
    m.set_attr["training"](Scalar[DT](1.0))

    var x = Tensor.alloc(B * IN)
    var y = Tensor.alloc(B * OUT)
    var go = Tensor.alloc(B * OUT)
    var gx = Tensor.alloc(B * IN)
    for k in range(B * IN):
        x.data[k] = _det(k + 1, 1.0)
    for k in range(B * OUT):
        go.data[k] = _det(k + 3, 1.0)

    m.forward["cpu", B](TensorRefs[Rep.ARITY](x), y, None)
    assert_true(_finite(y.data, B * OUT), "rep output finite")

    m.vjp["cpu", B](
        TensorRefs[Rep.ARITY](x), go, TensorRefs[Rep.ARITY](gx), None
    )
    assert_true(_finite(gx.data, B * IN), "rep grad_input finite")

    var gs = GradStats()
    m.for_each_param["cpu"](gs, None)
    print("   params=", gs.n_params, " nonzero-grad=", gs.n_nonzero)
    assert_true(gs.n_params > 0, "rep has params")
    assert_true(gs.n_nonzero * 10 >= gs.n_params * 9,
                "≥90% of rep params receive grad")
    _ = m^

    print("=" * 70)
    print("PASSED")
    print("=" * 70)
