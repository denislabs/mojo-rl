"""The runtime-length GPU kernels behind `ParamVisitorRef` (grad clip's sum
of squares and scale, SGD, DreamerOpt's AGC partials and update, zero-init's
output scale) against the CPU path of the same visitors, on one model with
identical weights and grads. Each kernel replaced a per-size instantiation;
this is the gate that the raw-pointer twin computes the same thing."""
from std.math import abs
from max.gpu.host import DeviceContext
from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.param import ParamVisitor
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.optimizer.sgd import SGD
from mojo_rl.nn.optimizer.dreamer_opt import DreamerOpt
from mojo_rl.nn.optimizer.grad_clip import clip_grad_norm
from mojo_rl.deep_agents.dreamerv3.zero_init import scale_output_module

comptime NET = Sequential[Linear[37, 53], Linear[53, 29], Linear[29, 7]]
comptime TOL = Scalar[DT](2e-5)


struct _SetGrad(ParamVisitor):
    var seed: Int
    var ctx: Optional[DeviceContext]
    def __init__(out self, ctx: Optional[DeviceContext]):
        self.seed = 0
        self.ctx = ctx
    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool, ctx: Optional[DeviceContext],
    ) raises:
        for k in range(N):
            grad.data[k] = Scalar[DT](0.3) * Scalar[DT]((k + self.seed) % 17 - 8)
        self.seed += 3
        if self.ctx:
            grad.upload(self.ctx.value())


struct _Snapshot(ParamVisitor):
    var vals: List[Scalar[DT]]
    var grads: List[Scalar[DT]]
    var ctx: Optional[DeviceContext]
    def __init__(out self, ctx: Optional[DeviceContext]):
        self.vals = List[Scalar[DT]]()
        self.grads = List[Scalar[DT]]()
        self.ctx = ctx
    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool, ctx: Optional[DeviceContext],
    ) raises:
        if self.ctx:
            param.download(self.ctx.value())
            grad.download(self.ctx.value())
        for k in range(N):
            self.vals.append(param.data[k])
            self.grads.append(grad.data[k])


def worst(a: List[Scalar[DT]], b: List[Scalar[DT]]) raises -> Scalar[DT]:
    if len(a) != len(b) or len(a) == 0:
        raise Error("snapshot length mismatch")
    var w = Scalar[DT](0)
    for k in range(len(a)):
        var d = abs(a[k] - b[k])
        if d > w:
            w = d
    return w


def check(label: String, mut c: NET, mut g: NET, ctx: DeviceContext, mut n_fail: Int) raises:
    var sc = _Snapshot(None)
    var sg = _Snapshot(Optional(ctx))
    c.for_each_param["cpu"](sc, None)
    g.for_each_param["gpu"](sg, Optional(ctx))
    var wv = worst(sc.vals, sg.vals)
    var wg = worst(sc.grads, sg.grads)
    var ok = wv <= TOL and wg <= TOL
    if not ok:
        n_fail += 1
    print(("PASS " if ok else "FAIL ") + label, " values:", len(sc.vals), " worst param diff:", wv, " worst grad diff:", wg)


def main() raises:
    var ctx = DeviceContext()
    var c = NET.make["cpu", Deterministic](None)
    var g = NET.make["gpu", Deterministic](Optional(ctx))
    var setc = _SetGrad(None)
    var setg = _SetGrad(Optional(ctx))
    c.for_each_param["cpu"](setc, None)
    g.for_each_param["gpu"](setg, Optional(ctx))
    var n_fail = 0
    check("initial weights + grads identical on both targets", c, g, ctx, n_fail)
    # grad clip: sum-of-squares + scale kernels
    var nc = clip_grad_norm["cpu"](c, Scalar[DT](0.5), None)
    var ng = clip_grad_norm["gpu"](g, Scalar[DT](0.5), Optional(ctx))
    ctx.synchronize()
    print("clip norms cpu/gpu:", nc, ng)
    if abs(nc - ng) > Scalar[DT](1e-3) * nc:
        n_fail += 1
        print("FAIL clip norm disagrees")
    check("grad clip (sum_sq + scale kernels)", c, g, ctx, n_fail)
    # SGD with decay
    var sc_opt = SGD(lr=Scalar[DT](1e-2), wd=Scalar[DT](1e-3))
    var sg_opt = SGD(lr=Scalar[DT](1e-2), wd=Scalar[DT](1e-3))
    for _ in range(3):
        sc_opt.step["cpu"](c, None)
        sg_opt.step["gpu"](g, Optional(ctx))
    ctx.synchronize()
    check("SGD x3 (sgd kernel)", c, g, ctx, n_fail)
    # DreamerOpt (AGC partials + update kernels)
    var dc = DreamerOpt(lr=Scalar[DT](1e-3))
    var dg = DreamerOpt(lr=Scalar[DT](1e-3))
    for _ in range(3):
        dc.step["cpu"](c, None)
        dg.step["gpu"](g, Optional(ctx))
    ctx.synchronize()
    check("DreamerOpt x3 (agc + update kernels)", c, g, ctx, n_fail)
    # zero-init output scale
    scale_output_module["cpu"](c, String("2.weight"), String("2.bias"), Scalar[DT](0.25), None)
    scale_output_module["gpu"](g, String("2.weight"), String("2.bias"), Scalar[DT](0.25), Optional(ctx))
    ctx.synchronize()
    check("scale_output_module (scale kernel)", c, g, ctx, n_fail)
    if n_fail > 0:
        raise Error("FAIL: " + String(n_fail) + " check(s) disagree between CPU and the runtime-length GPU kernels")
    print("ALL PASSED: runtime-length GPU kernels match the CPU visitors")
