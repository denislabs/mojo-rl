"""EZv2-Atari conv-dynamics smoke (Stage 2) — shapes + finite grads, CPU.

Builds `EZDynNetAtari[ACT=18, BINS=601]` (the ComputeGraph spatial dynamics
wrapped as a single-input Module), asserts the flat adapter contract
(IN=LATENT+ACT=2322, OUT=LATENT+BINS=2905), runs forward + vjp on a tiny batch,
and checks output + grad_input are finite and ≥80% of params receive a non-zero
gradient. The wrapper's vjp must copy the slot input-gradient into grad_inputs[0]
(the unroll BPTT needs ∂/∂z), so a non-zero grad_input is the key check.

Run:
    pixi run -e apple mojo run -I . tests/deep_agents/test_ezv2_atari_dynamics_smoke.mojo
"""

from std.math import isnan, isinf
from std.testing import assert_true, assert_equal
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.deep_agents.efficient_zero_v2.nets_atari import (
    EZDynNetAtari, EZ_LATENT,
)


comptime ACT = 18
comptime BINS = 601
comptime B = 2

comptime Dyn = EZDynNetAtari[ACT, BINS]
comptime IN = EZ_LATENT + ACT      # 2322
comptime OUT = EZ_LATENT + BINS    # 2905


def _det(i: Int, scale: Float64) -> Scalar[DT]:
    var v = (Float64((i * 2654435761) % 1000) / 500.0) - 1.0
    return Scalar[DT](v * scale)


def _finite(p: List[Scalar[DT]], n: Int) -> Bool:
    for i in range(n):
        if isnan(p[i]) or isinf(p[i]):
            return False
    return True


def _any_nonzero(p: List[Scalar[DT]], n: Int) -> Bool:
    for i in range(n):
        if p[i] != Scalar[DT](0.0):
            return True
    return False


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
    print("EZv2-Atari conv-dynamics smoke (Stage 2, CPU)")
    print("=" * 70)

    print("Dyn.IN_DIMS[0] =", Dyn.IN_DIMS[0], " (expect", IN, ")")
    print("Dyn.OUT_DIM   =", Dyn.OUT_DIM, " (expect", OUT, ")")
    assert_equal(Dyn.IN_DIMS[0], IN, "dyn input = LATENT+ACT")
    assert_equal(Dyn.OUT_DIM, OUT, "dyn output = LATENT+BINS")

    var m = Dyn.make["cpu", Kaiming]()
    m.set_attr["training"](Scalar[DT](1.0))

    var x = Tensor.alloc(B * IN)
    var y = Tensor.alloc(B * OUT)
    var go = Tensor.alloc(B * OUT)
    var gx = Tensor.alloc(B * IN)
    for k in range(B * IN):
        x.data[k] = _det(k + 1, 0.3)
    for k in range(B * OUT):
        go.data[k] = _det(k + 3, 1.0)

    m.forward["cpu", B](TensorRefs[Dyn.ARITY](x), y, None)
    assert_true(_finite(y.data, B * OUT), "dyn output finite")

    m.vjp["cpu", B](
        TensorRefs[Dyn.ARITY](x), go, TensorRefs[Dyn.ARITY](gx), None
    )
    assert_true(_finite(gx.data, B * IN), "dyn grad_input finite")
    # The latent half of grad_input must be non-zero (BPTT ∂/∂z copy-back).
    assert_true(_any_nonzero(gx.data, B * EZ_LATENT), "dyn grad_input(z) non-zero")

    var gs = GradStats()
    m.for_each_param["cpu"](gs, None)
    print("   params=", gs.n_params, " nonzero-grad=", gs.n_nonzero)
    assert_true(gs.n_params > 0, "dyn has params")
    assert_true(gs.n_nonzero * 10 >= gs.n_params * 8,
                "≥80% of dyn params receive grad")
    _ = m^

    print("=" * 70)
    print("PASSED")
    print("=" * 70)
