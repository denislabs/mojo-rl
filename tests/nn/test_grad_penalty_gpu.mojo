"""`GradPenalty` GPU parity — the device path against the CPU path.

The CPU gate (`test_grad_penalty.mojo`) carries the oracles; this one asks
only that the GPU kernels (`_row_unit_kernel`, `_shift_kernel`, `_cot_kernel`)
and the device forward/vjp chain produce the same parameter gradients and the
same penalty value as the host path on the same net and input.

⚠ The band is fp32 GPU-vs-CPU on a Tanh MLP — 1e-4 on Apple. A 5090 run
that lands outside it is TF32 on the matmuls, not a defect: compare the
weights bit-for-bit first (the memory note on TF32 bands).

Run:
    pixi run -e apple mojo run -I . tests/nn/test_grad_penalty_gpu.mojo
"""

from std.math import abs
from std.random import random_float64, seed
from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.core.initializer import Xavier
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import Tanh
from mojo_rl.nn.loss.grad_penalty import GradPenalty


comptime IN = 6
comptime H = 16
comptime B = 8
comptime SEED = 20260908
comptime COEF = 10.0
comptime Net = Sequential[Linear[IN, H], Tanh[H], Linear[H, H], Tanh[H], Linear[H, 1]]


struct _Read(ParamVisitor):
    var vals: List[List[Scalar[DT]]]

    def __init__(out self):
        self.vals = List[List[Scalar[DT]]]()

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "gpu":
            grad.download(ctx.value())
        var g = List[Scalar[DT]](capacity=N)
        for i in range(N):
            g.append(grad.data[i])
        self.vals.append(g^)


def main() raises:
    print("=" * 70)
    print("GradPenalty GPU parity")
    print("=" * 70)
    var ctx = DeviceContext()
    var octx = Optional[DeviceContext](ctx)

    seed(SEED)
    var net_c = Net.make["cpu", Xavier](None)
    seed(SEED)
    var net_g = Net.make["gpu", Xavier](octx)

    seed(SEED + 1)
    var x = Tensor.alloc(B * IN)
    for i in range(B * IN):
        x.data[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
    var xg = Tensor.alloc(B * IN)
    for i in range(B * IN):
        xg.data[i] = x.data[i]
    xg.upload(ctx)

    var gp_c = GradPenalty[IN, B].make["cpu"](None, 1e-2, 1.0)
    var gp_g = GradPenalty[IN, B].make["gpu"](octx, 1e-2, 1.0)
    net_c.zero_grad["cpu"](None)
    net_g.zero_grad["gpu"](octx)
    var pc = gp_c.apply["cpu", Net](net_c, x, COEF, want_loss=True)
    var pg = gp_g.apply["gpu", Net](net_g, xg, COEF, want_loss=True)
    ctx.synchronize()
    print("   penalty: cpu", pc, "  gpu", pg)
    assert_true(abs(pc - pg) < 1e-4 * (abs(pc) + 1.0), "penalty value differs across devices")

    var rc = _Read()
    net_c.for_each_param["cpu"](rc, None)
    var rg = _Read()
    net_g.for_each_param["gpu"](rg, octx)
    var worst = Float64(0)
    var n = 0
    for k in range(len(rc.vals)):
        for j in range(len(rc.vals[k])):
            var d = abs(Float64(rc.vals[k][j]) - Float64(rg.vals[k][j]))
            if d > worst:
                worst = d
            n += 1
    print("   grads compared:", n, "  worst |cpu − gpu| =", worst)
    assert_true(n > 0, "no parameters compared")
    assert_true(worst < 1e-4, "GPU gradient differs from CPU by " + String(worst))

    gp_g.read_norms["gpu"]()
    gp_c.read_norms["cpu"]()
    var wn = Float64(0)
    for i in range(B):
        var d = abs(Float64(gp_c.gnorm.data[i]) - Float64(gp_g.gnorm.data[i]))
        if d > wn:
            wn = d
    print("   worst |‖∇ₓD‖ cpu − gpu| =", wn)
    assert_true(wn < 1e-4, "probe gradient norm differs across devices")
    print("\n[PASS] GradPenalty GPU parity")
