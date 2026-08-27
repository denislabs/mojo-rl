"""EZv2-Atari NHWC rep-tower integration smoke (Phase 2, CPU + GPU).

Proves the FULL channels-last rep tower composes and runs: with the Atari config
flipped to LAYOUT=NHWC, `Cfg.Rep` = the NHWC conv tower (Conv2D/BN2D/AvgPool/
ResBlock all NHWC) + the ToNCHW boundary adapter, output a canonical NCHW [64,6,6]
= 2304 latent. The per-primitive NCHW-vs-NHWC parity is proven elsewhere
(test_conv2d / batch_norm_2d / pool_2d / to_nchw _nhwc_parity); this checks they
COMPOSE — forward + vjp run, outputs finite, ≥90% params receive grad. The NCHW
config Rep is covered by test_ezv2_atari_rep_smoke; this file is the NHWC twin.

Run:
    pixi run -e apple mojo run -I . tests/deep_agents/test_ezv2_atari_rep_nhwc_smoke.mojo
"""

from std.math import isnan, isinf
from std.sys import has_accelerator
from std.testing import assert_true, assert_equal
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT, LAYOUT_NHWC
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.deep_agents.efficient_zero_v2.config_atari import EZV2AtariConfig

comptime B = 2
comptime Cfg = EZV2AtariConfig[FRAMES=4, ACT=18, LAYOUT=LAYOUT_NHWC]
comptime Rep = Cfg.Rep                  # NHWC tower + ToNCHW adapter
comptime IN = Cfg.OBS                   # 110592
comptime OUT = Cfg.LATENT               # 2304


def _det(i: Int) -> Scalar[DT]:
    return Scalar[DT]((Float64((i * 2654435761) % 1000) / 500.0) - 1.0)


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
                break


def test_nhwc_cpu() raises:
    var m = Rep.make["cpu", Kaiming]()
    m.set_attr["training"](Scalar[DT](1.0))
    var x = Tensor.alloc(B * IN)
    var y = Tensor.alloc(B * OUT)
    var go = Tensor.alloc(B * OUT)
    var gx = Tensor.alloc(B * IN)
    for k in range(B * IN):
        x.data[k] = _det(k + 1)
    for k in range(B * OUT):
        go.data[k] = _det(k + 3)
    m.forward["cpu", B](TensorRefs[Rep.ARITY](x), y, None)
    assert_true(_finite(y.data, B * OUT), "NHWC rep forward finite")
    m.vjp["cpu", B](
        TensorRefs[Rep.ARITY](x), go, TensorRefs[Rep.ARITY](gx), None
    )
    assert_true(_finite(gx.data, B * IN), "NHWC rep grad_input finite")
    var gs = GradStats()
    m.for_each_param["cpu"](gs, None)
    print("  NHWC CPU params=", gs.n_params, " nonzero-grad=", gs.n_nonzero)
    assert_true(
        gs.n_nonzero * 10 >= gs.n_params * 9, "≥90% NHWC params get grad"
    )
    _ = m^


def test_nhwc_gpu() raises:
    var c = DeviceContext()
    var m = Rep.make["gpu", Kaiming](Optional(c))
    m.set_attr["training"](Scalar[DT](1.0))
    var x = Tensor.alloc(B * IN)
    var y = Tensor.alloc(B * OUT)
    var go = Tensor.alloc(B * OUT)
    var gx = Tensor.alloc(B * IN)
    for k in range(B * IN):
        x.data[k] = _det(k + 1)
    for k in range(B * OUT):
        go.data[k] = _det(k + 3)
    x.upload(c)
    go.upload(c)
    m.forward["gpu", B](TensorRefs[Rep.ARITY](x), y, Optional(c))
    y.download(c)
    assert_true(_finite(y.data, B * OUT), "NHWC rep GPU forward finite")
    m.vjp["gpu", B](
        TensorRefs[Rep.ARITY](x), go, TensorRefs[Rep.ARITY](gx), Optional(c)
    )
    gx.download(c)
    assert_true(_finite(gx.data, B * IN), "NHWC rep GPU grad_input finite")
    print("  NHWC GPU fwd+vjp finite: True")
    _ = m^


def main() raises:
    print("=" * 70)
    print("EZv2-Atari NHWC rep-tower integration smoke")
    print("=" * 70)
    assert_equal(Rep.IN_DIMS[0], IN, "rep in dim 110592")
    assert_equal(Rep.OUT_DIM, OUT, "rep latent dim 2304 (ToNCHW preserves)")
    print("CPU:")
    test_nhwc_cpu()
    comptime if has_accelerator():
        print("GPU:")
        test_nhwc_gpu()
    else:
        print("No accelerator — CPU only")
    print("=" * 70)
    print("PASSED")
    print("=" * 70)
