"""ISOLATION TEST (run on NVIDIA): does Parallel branch 0 update on device like
branch 1?

The MuZero pred net is `Sequential[torso, Parallel[policy_head, value_head]]`.
In training the VALUE head (branch 1) learns but the POLICY head (branch 0) stays
uniform. This reproduces the minimal core: `Parallel[Linear[IN,7],
Linear[IN,51]]` (policy-like + value-like), seed BOTH output slices with nonzero
grad, run one forward→vjp→Adam step on GPU, and check BOTH branches' weights
moved on device.

  branch0 Δ ≈ 0 while branch1 Δ > 0  → Parallel GPU vjp drops/corrupts branch 0
                                       (the policy-death bug; expected NVIDIA-only)
  both Δ > 0                          → Parallel device update is fine; bug is
                                       elsewhere (value/two-hot, MCTS targets)

No Conv2D / ReLU, so it compiles even while the AMP elementwise work is in
flight. Run: pixi run -e nvidia mojo run -I . tests/nn/test_parallel_branch_update_gpu.mojo
(and -e apple to compare — BatchNorm-style bugs were Apple-OK / NVIDIA-broken).
"""

from std.gpu.host import DeviceContext
from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.combinators.parallel import Parallel
from mojo_rl.nn.primitives.linear import Linear


struct WSum(ParamVisitor):
    """Sum |weight| of branch-0 ('0.weight') and branch-1 ('1.weight')."""
    var b0: Scalar[DT]
    var b1: Scalar[DT]

    def __init__(out self):
        self.b0 = 0
        self.b1 = 0

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        if name == "0.weight" or name == "1.weight":
            comptime if target == "gpu":
                param.download(ctx.value())
            var s = Scalar[DT](0)
            for i in range(N):
                s += abs(param.data[i])
            if name == "0.weight":
                self.b0 = s
            else:
                self.b1 = s


def _run[target: StaticString](ctx: Optional[DeviceContext]) raises:
    comptime IN = 32
    comptime A = 7      # policy-like
    comptime Bv = 51    # value-like
    comptime OUT = A + Bv
    comptime B = 16
    comptime Net = Parallel[Linear[IN, A], Linear[IN, Bv]]
    var net = Net.make[target, Kaiming](ctx)
    var opt = Adam(lr=Scalar[DT](1e-2))

    var x = Tensor.alloc(B * IN)
    for i in range(B * IN):
        x.data[i] = Scalar[DT](0.1) * Scalar[DT](((i * 7) % 13) - 6)
    var out = Tensor()
    var gin = Tensor()
    comptime if target == "gpu":
        x.ensure_gpu(ctx.value(), B * IN); x.upload(ctx.value())
        out.ensure_gpu(ctx.value(), B * OUT)
        gin.ensure_gpu(ctx.value(), B * IN)

    print("  [", target, "] before:")
    var r0 = WSum(); net.for_each_param[target](r0, ctx)
    print("    branch0 sum|w| =", r0.b0, " branch1 sum|w| =", r0.b1)

    for _ in range(10):
        net.zero_grad[target](ctx)
        net.forward[target, B](TensorRefs[1](x), out, ctx)
        # grad_output: BOTH slices nonzero (policy slice [0,A), value slice [A,OUT))
        var gout = Tensor.alloc(B * OUT)
        for b in range(B):
            for a in range(A):
                gout.data[b * OUT + a] = Scalar[DT](0.05) * Scalar[DT]((a % 3) - 1)
            for j in range(Bv):
                gout.data[b * OUT + A + j] = Scalar[DT](0.05) * Scalar[DT]((j % 3) - 1)
        comptime if target == "gpu":
            gout.ensure_gpu(ctx.value(), B * OUT); gout.upload(ctx.value())
        net.vjp[target, B](TensorRefs[1](x), gout, TensorRefs[1](gin), ctx)
        opt.begin_step()
        net.for_each_param[target](opt, ctx)
        comptime if target == "gpu":
            ctx.value().synchronize()

    print("  [", target, "] after 10 steps:")
    var r1 = WSum(); net.for_each_param[target](r1, ctx)
    print("    branch0 sum|w| =", r1.b0, " branch1 sum|w| =", r1.b1)
    var d0 = abs(r1.b0 - r0.b0)
    var d1 = abs(r1.b1 - r0.b1)
    print("    Δbranch0 =", d0, "  Δbranch1 =", d1)
    if d0 < Scalar[DT](1e-6) and d1 > Scalar[DT](1e-6):
        print("    >>> BRANCH 0 (policy) DID NOT UPDATE — Parallel vjp BUG <<<")
    elif d0 > Scalar[DT](1e-6) and d1 > Scalar[DT](1e-6):
        print("    >>> both branches updated — Parallel OK on", target, "<<<")
    else:
        print("    >>> unexpected:", d0, d1, "<<<")


def main() raises:
    print("CPU:")
    _run["cpu"](None)
    print("GPU:")
    var ctx = DeviceContext()
    _run["gpu"](Optional(ctx))
