"""`GradPenalty` gate — the WGAN-GP / R1 parameter gradient, at first order.

`mojo_rl/nn/loss/grad_penalty.mojo` computes ∂/∂θ mean_i(‖∇ₓD_i‖ − t)² from
three forward and two backward passes (a directional finite difference along
the stop-gradient unit gradient — exact up to O(ε²), see the module). Two
independent oracles pin it, and a third check pins the call-order contract:

  [1] the finite-difference directional derivative `h_i` matches the exact
      ‖∇ₓD_i‖ read off the probe vjp, per row (the O(ε²) claim, measured).
  [2] the θ-gradient matches PARAMETER central differences of the exact
      penalty on a smooth net (Linear → Tanh → Linear → Tanh → Linear),
      every coordinate. The oracle evaluates the penalty EXACTLY (forward +
      vjp, host norms) at θ ± h — no directional trick in the oracle, so the
      two sides share nothing but the primitives' vjps.
  [3] closed form on a LINEAR D: ∇ₓD = w for every row, so
      ∂P/∂w = 2·coef·(‖w‖ − 1)·w/‖w‖ and ∂P/∂b = 0. Pins the VALUE.
  [4] a ReLU net (Linear → ReLU → Linear → ReLU → Linear): same FD oracle,
      gated on the FRACTION of coordinates within tolerance — a parameter
      step can flip a ReLU mask and the penalty jumps there; a per-element
      gate would be undecidable (the discontinuity lesson in memory).
  [5] the contract: grads after `apply` then a loss vjp equal the sum of
      the two taken separately. `apply` zeroes what came before it — this
      is the executable form of "apply FIRST".

FD hygiene, both traps from memory: every perturbation is restored and the
penalty is re-evaluated at the end to the SAME value; the nominal step is
2h in the denominator.

Run:
    pixi run mojo run -I . tests/nn/test_grad_penalty.mojo
"""

from std.math import abs, sqrt
from std.random import random_float64, seed
from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.call import call_forward, call_vjp
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.initializer import Xavier
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import ReLU, Tanh
from mojo_rl.nn.loss.grad_penalty import GradPenalty


comptime IN = 5
comptime H = 8
comptime B = 4
comptime SEED = 20260908
comptime COEF = 10.0
comptime TARGET = 1.0

comptime TanhNet = Sequential[
    Linear[IN, H], Tanh[H], Linear[H, H], Tanh[H], Linear[H, 1]
]
comptime ReLUNet = Sequential[
    Linear[IN, H], ReLU[H], Linear[H, H], ReLU[H], Linear[H, 1]
]
comptime LinNet = Linear[IN, 1]


# ── visitors ────────────────────────────────────────────────────────────


struct _Sizes(ParamVisitor):
    var names: List[String]
    var sizes: List[Int]

    def __init__(out self):
        self.names = List[String]()
        self.sizes = List[Int]()

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        self.names.append(name)
        self.sizes.append(N)


struct _Coord(ParamVisitor):
    """Read (`set=False`) or write (`set=True`) coordinate `j` of the
    `pidx`-th visited param. Writes are ABSOLUTE: restoring by adding
    `+h, -2h, +h` is not bit-exact in fp32 and the restoration check below
    caught exactly that on the first run."""
    var pidx: Int
    var j: Int
    var value: Scalar[DT]
    var set: Bool
    var cur: Int

    def __init__(out self, pidx: Int, j: Int, value: Scalar[DT], set: Bool):
        self.pidx = pidx
        self.j = j
        self.value = value
        self.set = set
        self.cur = 0

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        if self.cur == self.pidx:
            if self.set:
                param.data[self.j] = self.value
            else:
                self.value = param.data[self.j]
        self.cur += 1


struct _ReadGrads(ParamVisitor):
    var vals: List[List[Scalar[DT]]]
    var params: List[List[Scalar[DT]]]

    def __init__(out self):
        self.vals = List[List[Scalar[DT]]]()
        self.params = List[List[Scalar[DT]]]()

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        var g = List[Scalar[DT]](capacity=N)
        var p = List[Scalar[DT]](capacity=N)
        for i in range(N):
            g.append(grad.data[i] if grad.n >= N else Scalar[DT](0))
            p.append(param.data[i])
        self.vals.append(g^)
        self.params.append(p^)


# ── helpers ─────────────────────────────────────────────────────────────


def _rand(n: Int) raises -> Tensor:
    var t = Tensor.alloc(n)
    for i in range(n):
        t.data[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
    return t^


def _exact_penalty[M: Module](mut net: M, mut x: Tensor) raises -> Float64:
    """coef · mean_i (‖∇ₓD_i‖ − t)², with ∇ₓD from the vjp and the norm on
    the host. Zeroes the grads it accumulated."""
    var out = Tensor()
    call_forward["cpu", B](net, TensorRefs[1, MutAnyOrigin](x), out, None)
    var ones = Tensor.alloc(B)
    for i in range(B):
        ones.data[i] = Scalar[DT](1.0)
    var g = Tensor()
    call_vjp["cpu", B](
        net, TensorRefs[1, MutAnyOrigin](x), ones,
        TensorRefs[1, MutAnyOrigin](g), None,
    )
    net.zero_grad["cpu"](None)
    var s = Float64(0)
    for i in range(B):
        var acc = Float64(0)
        for k in range(IN):
            var v = Float64(g.data[i * IN + k])
            acc += v * v
        var r = sqrt(acc) - TARGET
        s += r * r
    return COEF * s / Float64(B)


def _grads[M: Module](mut net: M) raises -> _ReadGrads:
    var r = _ReadGrads()
    net.for_each_param["cpu"](r, None)
    return r^


def _fd_check[
    M: Module
](mut net: M, mut x: Tensor, h: Scalar[DT], label: String) raises -> Tuple[Int, Int, Float64]:
    """Analytic (GradPenalty) vs parameter central differences of the exact
    penalty, every coordinate. Returns (within tolerance, total, worst rel)."""
    var gp = GradPenalty[IN, B].make["cpu"](None, 1e-2, TARGET)
    net.zero_grad["cpu"](None)
    var p_an = gp.apply["cpu", M](net, x, COEF, want_loss=True)
    var an = _grads(net)
    var sizes = _Sizes()
    net.for_each_param["cpu"](sizes, None)
    var p0 = _exact_penalty(net, x)
    print("      ", label, ": exact P", p0, "  apply's P", p_an)
    assert_true(
        abs(p0 - p_an) < 2e-3 * (abs(p0) + 1e-2),
        label + ": apply's penalty value disagrees with the exact one",
    )
    var ok = 0
    var total = 0
    var worst = Float64(0)
    for k in range(len(sizes.sizes)):
        for j in range(sizes.sizes[k]):
            var rd = _Coord(k, j, Scalar[DT](0), False)
            net.for_each_param["cpu"](rd, None)
            var orig = rd.value
            var plus = _Coord(k, j, orig + h, True)
            net.for_each_param["cpu"](plus, None)
            var pp = _exact_penalty(net, x)
            var minus = _Coord(k, j, orig - h, True)
            net.for_each_param["cpu"](minus, None)
            var pm = _exact_penalty(net, x)
            var back = _Coord(k, j, orig, True)
            net.for_each_param["cpu"](back, None)
            var fd = (pp - pm) / (2.0 * Float64(h))
            var a = Float64(an.vals[k][j])
            var d = abs(fd - a)
            var rel = d / (abs(fd) + 1e-2)
            if rel > worst:
                worst = rel
            if d <= 3e-2 * (abs(fd) + 1e-2):
                ok += 1
            total += 1
    var p1 = _exact_penalty(net, x)
    assert_true(
        p1 == p0,
        label + ": the penalty changed after the FD sweep (" + String(p0)
        + " -> " + String(p1) + ") — a perturbation was not restored",
    )
    return (ok, total, worst)


# ── tests ───────────────────────────────────────────────────────────────


def test_directional_matches_exact_norm() raises:
    print("[1] finite-difference directional derivative == exact ‖∇ₓD‖ per row ...")
    seed(SEED)
    var net = TanhNet.make["cpu", Xavier](None)
    var x = _rand(B * IN)
    var gp = GradPenalty[IN, B].make["cpu"](None, 1e-2, TARGET)
    _ = gp.apply["cpu", TanhNet](net, x, COEF, want_loss=True)
    gp.read_norms["cpu"]()
    var worst = Float64(0)
    for i in range(B):
        var n = Float64(gp.gnorm.data[i])
        var h = Float64(gp.h.data[i])
        var rel = abs(h - n) / (n + 1e-6)
        print("      row", i, " exact", n, " fd", h, " rel", rel)
        if rel > worst:
            worst = rel
    assert_true(worst < 2e-3, "directional FD disagrees with the exact norm by " + String(worst))
    print("      OK")


def test_theta_gradient_vs_fd_smooth() raises:
    print("[2] θ-gradient vs parameter central differences, Tanh net, every coordinate ...")
    seed(SEED + 1)
    var net = TanhNet.make["cpu", Xavier](None)
    var x = _rand(B * IN)
    var r = _fd_check[TanhNet](net, x, Scalar[DT](3e-3), String("tanh"))
    print("      within tolerance:", r[0], "/", r[1], "  worst rel", r[2])
    assert_true(r[0] == r[1], "a coordinate's θ-gradient disagrees with the FD oracle")
    print("      OK")


def test_linear_closed_form() raises:
    print("[3] linear D: ∂P/∂w = 2·coef·(‖w‖−1)·w/‖w‖, ∂P/∂b = 0 ...")
    seed(SEED + 2)
    var net = LinNet.make["cpu", Xavier](None)
    var x = _rand(B * IN)
    var gp = GradPenalty[IN, B].make["cpu"](None, 1e-2, TARGET)
    net.zero_grad["cpu"](None)
    var p = gp.apply["cpu", LinNet](net, x, COEF, want_loss=True)
    var g = _grads(net)
    # Param order of Linear: the IN-sized one is the weight, the 1-sized the bias.
    var wi = 0 if len(g.params[0]) == IN else 1
    var bi = 1 - wi
    var wn = Float64(0)
    for k in range(IN):
        wn += Float64(g.params[wi][k]) * Float64(g.params[wi][k])
    wn = sqrt(wn)
    var want_p = COEF * (wn - TARGET) * (wn - TARGET)
    print("      ‖w‖", wn, "  P", p, "  closed form", want_p)
    assert_true(abs(p - want_p) < 1e-4 * (abs(want_p) + 1.0), "penalty value off the closed form")
    var worst = Float64(0)
    for k in range(IN):
        var want = 2.0 * COEF * (wn - TARGET) * Float64(g.params[wi][k]) / wn
        var d = abs(Float64(g.vals[wi][k]) - want)
        if d > worst:
            worst = d
    var db = abs(Float64(g.vals[bi][0]))
    print("      worst |∂P/∂w − closed form|", worst, "  |∂P/∂b|", db)
    assert_true(worst < 1e-4 * (abs(2.0 * COEF * (wn - TARGET)) + 1.0), "weight gradient off the closed form")
    assert_true(db < 1e-6, "bias gradient is not zero")
    print("      OK")


def test_theta_gradient_vs_fd_relu() raises:
    print("[4] θ-gradient vs FD, ReLU net — gated on the fraction within tolerance ...")
    seed(SEED + 3)
    var net = ReLUNet.make["cpu", Xavier](None)
    var x = _rand(B * IN)
    var r = _fd_check[ReLUNet](net, x, Scalar[DT](3e-3), String("relu"))
    var frac = Float64(r[0]) / Float64(r[1])
    print("      within tolerance:", r[0], "/", r[1], " (", frac, ")  worst rel", r[2])
    assert_true(frac >= 0.95, "too many ReLU-net coordinates disagree with the FD oracle")
    print("      OK")


def test_apply_first_contract() raises:
    print("[5] apply then a loss vjp == the two taken separately (linearity) ...")
    seed(SEED + 4)
    var net = TanhNet.make["cpu", Xavier](None)
    var x = _rand(B * IN)
    var cot = _rand(B)
    var gp = GradPenalty[IN, B].make["cpu"](None, 1e-2, TARGET)

    net.zero_grad["cpu"](None)
    _ = gp.apply["cpu", TanhNet](net, x, COEF, want_loss=False)
    var ga = _grads(net)

    net.zero_grad["cpu"](None)
    var out = Tensor()
    call_forward["cpu", B](net, TensorRefs[1, MutAnyOrigin](x), out, None)
    var sink = Tensor()
    call_vjp["cpu", B](net, TensorRefs[1, MutAnyOrigin](x), cot, TensorRefs[1, MutAnyOrigin](sink), None)
    var gb = _grads(net)

    net.zero_grad["cpu"](None)
    _ = gp.apply["cpu", TanhNet](net, x, COEF, want_loss=False)
    call_forward["cpu", B](net, TensorRefs[1, MutAnyOrigin](x), out, None)
    call_vjp["cpu", B](net, TensorRefs[1, MutAnyOrigin](x), cot, TensorRefs[1, MutAnyOrigin](sink), None)
    var gc = _grads(net)

    var worst = Float64(0)
    for k in range(len(ga.vals)):
        for j in range(len(ga.vals[k])):
            var d = abs(Float64(gc.vals[k][j]) - Float64(ga.vals[k][j]) - Float64(gb.vals[k][j]))
            if d > worst:
                worst = d
    print("      worst |(apply+loss) − apply − loss| =", worst)
    assert_true(worst < 1e-5, "apply-then-loss is not the sum — the contract is broken")
    print("      OK")


def main() raises:
    print("=" * 70)
    print("GradPenalty — first-order WGAN-GP / R1 gradient")
    print("=" * 70)
    test_directional_matches_exact_norm()
    test_theta_gradient_vs_fd_smooth()
    test_linear_closed_form()
    test_theta_gradient_vs_fd_relu()
    test_apply_first_contract()
    print("\n[PASS] GradPenalty gate")
