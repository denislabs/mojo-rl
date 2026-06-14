"""EZv2-Atari conv-dynamics smoke (Stage 2) — shapes + finite grads, CPU.

Builds `EZDynNetAtari[ACT=18, BINS=601]` (the ComputeGraph spatial dynamics
wrapped as a single-input Module), asserts the flat adapter contract
(IN=LATENT+ACT=2322, OUT=LATENT+BINS=2905), runs forward + vjp on a tiny batch,
and checks output + grad_input are finite and ≥80% of params receive a non-zero
gradient. The wrapper's vjp must copy the slot input-gradient into grad_inputs[0]
(the unroll BPTT needs ∂/∂z), so a non-zero grad_input is the key check.

Run:
    pixi run -e apple mojo run -I . tests/deep_agents2/test_ezv2_atari_dynamics_smoke.mojo
"""

from std.memory import alloc
from std.math import isnan, isinf
from std.gpu.memory import AddressSpace
from std.testing import assert_true, assert_equal
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.core import ParamVisitor
from mojo_rl.deep_agents2.efficient_zero_v2.nets_atari import (
    EZDynNetAtari, EZ_LATENT,
)


comptime ACT = 18
comptime BINS = 601
comptime B = 2

comptime Dyn = EZDynNetAtari[ACT, BINS]
comptime IN = EZ_LATENT + ACT      # 2322
comptime OUT = EZ_LATENT + BINS    # 2905


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


def _any_nonzero(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int) -> Bool:
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
    print("EZv2-Atari conv-dynamics smoke (Stage 2, CPU)")
    print("=" * 70)

    print("Dyn.IN_DIMS[0] =", Dyn.IN_DIMS[0], " (expect", IN, ")")
    print("Dyn.OUT_DIM   =", Dyn.OUT_DIM, " (expect", OUT, ")")
    assert_equal(Dyn.IN_DIMS[0], IN, "dyn input = LATENT+ACT")
    assert_equal(Dyn.OUT_DIM, OUT, "dyn output = LATENT+BINS")

    var m = Dyn.make[target="cpu", INIT=Kaiming]()
    m.set_attr["training"](Scalar[DT](1.0))

    var x = _a(B * IN); var y = _a(B * OUT)
    var go = _a(B * OUT); var gx = _a(B * IN)
    for k in range(B * IN):
        x[k] = _det(k + 1, 0.3)
    for k in range(B * OUT):
        go[k] = _det(k + 3, 1.0)

    var x_t = TileTensor(x, row_major[B, IN]())
    var y_t = TileTensor(y, row_major[B, OUT]())
    m.forward["cpu", B](x_t, output=y_t)
    assert_true(_finite(y, B * OUT), "dyn output finite")

    var go_t = TileTensor(go, row_major[B, OUT]())
    var gx_t = TileTensor(gx, row_major[B, IN]())
    m.vjp["cpu", B](go_t, gx_t)
    assert_true(_finite(gx, B * IN), "dyn grad_input finite")
    # The latent half of grad_input must be non-zero (BPTT ∂/∂z copy-back).
    assert_true(_any_nonzero(gx, B * EZ_LATENT), "dyn grad_input(z) non-zero")

    var gs = GradStats()
    m.for_each_param["cpu", GradStats]("dyn", gs)
    print("   params=", gs.n_params, " nonzero-grad=", gs.n_nonzero)
    assert_true(gs.n_params > 0, "dyn has params")
    assert_true(gs.n_nonzero * 10 >= gs.n_params * 8,
                "≥80% of dyn params receive grad")
    _ = m^

    print("=" * 70)
    print("PASSED")
    print("=" * 70)
