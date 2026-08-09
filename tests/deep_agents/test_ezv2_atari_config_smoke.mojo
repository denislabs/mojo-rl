"""EZv2-Atari config + prediction-head smoke (Stage 2) — CPU.

Asserts the five-net dim contract of `EZV2AtariConfig` composes (the flat
LATENT=2304 interface the planner/agent rely on), and runs the conv prediction
head forward + vjp with finite output + ≥80% params receiving grad.

Run:
    pixi run -e apple mojo run -I . tests/deep_agents/test_ezv2_atari_config_smoke.mojo
"""

from std.math import isnan, isinf
from std.testing import assert_true, assert_equal
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT, LAYOUT_NCHW
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.deep_agents.efficient_zero_v2.config_atari import EZV2AtariConfig


comptime FRAMES = 4
comptime ACT = 18
comptime Cfg = EZV2AtariConfig[FRAMES, ACT, LAYOUT=LAYOUT_NCHW]
comptime LATENT = Cfg.LATENT     # 2304
comptime BINS = Cfg.BINS         # 601
comptime PROJ = Cfg.PROJ         # 1024
comptime B = 2


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
        for i in range(N):
            if grad.data[i] != Scalar[DT](0.0):
                self.n_nonzero += 1
                return


def main() raises:
    print("=" * 70)
    print("EZv2-Atari config + prediction-head smoke (Stage 2, CPU)")
    print("=" * 70)

    # ── five-net dim contract ───────────────────────────────────────
    print("OBS          =", Cfg.OBS, " (FRAMES*3*96*96)")
    print("LATENT/BINS  =", LATENT, "/", BINS)
    assert_equal(Cfg.OBS, FRAMES * 3 * 96 * 96, "OBS = stacked RGB 96x96")
    assert_equal(Cfg.Rep.IN_DIMS[0], Cfg.OBS, "Rep IN = OBS")
    assert_equal(Cfg.Rep.OUT_DIM, LATENT, "Rep OUT = LATENT")
    assert_equal(Cfg.Dyn.IN_DIMS[0], LATENT + ACT, "Dyn IN = LATENT+ACT")
    assert_equal(Cfg.Dyn.OUT_DIM, LATENT + BINS, "Dyn OUT = LATENT+BINS")
    assert_equal(Cfg.Pred.IN_DIMS[0], LATENT, "Pred IN = LATENT")
    assert_equal(Cfg.Pred.OUT_DIM, ACT + BINS, "Pred OUT = ACT+BINS")
    assert_equal(Cfg.Proj.IN_DIMS[0], LATENT, "Proj IN = LATENT")
    assert_equal(Cfg.Proj.OUT_DIM, PROJ, "Proj OUT = PROJ")
    assert_equal(Cfg.Predh.IN_DIMS[0], PROJ, "Predh IN = PROJ")
    assert_equal(Cfg.Predh.OUT_DIM, PROJ, "Predh OUT = PROJ")
    print("  ✓ five-net dim contract consistent")

    # ── prediction head forward + grad ──────────────────────────────
    comptime PIN = LATENT
    comptime POUT = ACT + BINS
    var m = Cfg.Pred.make["cpu", Kaiming]()
    m.set_attr["training"](Scalar[DT](1.0))

    var x = Tensor.alloc(B * PIN)
    var y = Tensor.alloc(B * POUT)
    var go = Tensor.alloc(B * POUT)
    var gx = Tensor.alloc(B * PIN)
    for k in range(B * PIN):
        x.data[k] = _det(k + 1, 0.3)
    for k in range(B * POUT):
        go.data[k] = _det(k + 3, 1.0)

    m.forward["cpu", B](TensorRefs[Cfg.Pred.ARITY](x), y, None)
    assert_true(_finite(y.data, B * POUT), "pred output finite")

    m.vjp["cpu", B](
        TensorRefs[Cfg.Pred.ARITY](x), go, TensorRefs[Cfg.Pred.ARITY](gx), None
    )
    assert_true(_finite(gx.data, B * PIN), "pred grad_input finite")

    var gs = GradStats()
    m.for_each_param["cpu"](gs, None)
    print("   pred params=", gs.n_params, " nonzero-grad=", gs.n_nonzero)
    assert_true(gs.n_nonzero * 10 >= gs.n_params * 8,
                "≥80% of pred params receive grad")
    _ = m^

    print("=" * 70)
    print("PASSED")
    print("=" * 70)
