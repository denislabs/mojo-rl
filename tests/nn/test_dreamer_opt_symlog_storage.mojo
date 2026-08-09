"""DreamerV3 storage foundations gate: `Symlog` primitive + `DreamerOpt`.

- Symlog[D] = Elementwise[D, SymlogOp]: forward = sign(x)·log(1+|x|), backward =
  go/(1+|x|). Exact known-value check + CPU/GPU parity.
- DreamerOpt (rms→momentum→lr + per-leaf AGC; mu→Param.m, nu→Param.v): stepping a
  Linear net on an MSE-gradient must REDUCE the loss (optimizer works + AGC/bias-
  correction wired right), and the CPU and GPU paths must land on ~identical params
  (compared via the post-training forward output).

Run: pixi run -e apple mojo run -I . tests/nn/test_dreamer_opt_symlog_storage.mojo
"""

from std.math import log
from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.symlog import Symlog
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.optimizer.dreamer_opt import DreamerOpt


comptime SD = 6  # symlog width (B=1)
comptime D = 4
comptime H = 6
comptime O = 3
comptime B = 5
comptime K = 12  # optimizer steps
comptime NET = Sequential[Linear[D, H], Linear[H, O]]


def _abs(x: Scalar[DT]) -> Scalar[DT]:
    return x if x >= Scalar[DT](0) else -x


# ── Symlog ──────────────────────────────────────────────────────────────
def _symlog_ok[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    var net = Symlog[SD].make[target, Deterministic](ctx)
    var x = Tensor.alloc(SD)
    var vals = [
        Scalar[DT](-2.0), Scalar[DT](-0.5), Scalar[DT](0.0),
        Scalar[DT](0.5), Scalar[DT](2.0), Scalar[DT](7.0),
    ]
    for i in range(SD):
        x.data[i] = vals[i]
    var out = Tensor.alloc(SD)
    var go = Tensor.alloc(SD)
    var gi = Tensor.alloc(SD)
    for i in range(SD):
        go.data[i] = Scalar[DT](1.0)
    comptime if target == "gpu":
        x.upload(ctx.value()); go.upload(ctx.value())
    net.forward[target, 1](TensorRefs[1](x), out, ctx)
    net.vjp[target, 1](TensorRefs[1](x), go, TensorRefs[1](gi), ctx)
    comptime if target == "gpu":
        out.download(ctx.value()); gi.download(ctx.value())
    var one = Scalar[DT](1.0)
    for i in range(SD):
        var xv = vals[i]
        var ax = _abs(xv)
        var sgn = one if xv >= Scalar[DT](0) else -one
        var exp_f = sgn * log(one + ax)
        var exp_b = one / (one + ax)
        if _abs(out.data[i] - exp_f) > Scalar[DT](1e-5):
            return False
        if _abs(gi.data[i] - exp_b) > Scalar[DT](1e-5):
            return False
    return True


# ── DreamerOpt: converge an MSE + record final output for parity ──────────
def _dreamer_train[
    target: StaticString
](
    ctx: Optional[DeviceContext], mut final_out: List[Scalar[DT]]
) raises -> Bool:
    var net = NET.make[target, Deterministic](ctx)
    var opt = DreamerOpt(lr=Scalar[DT](1e-2))

    var x = Tensor.alloc(B * D)
    var tgt = Tensor.alloc(B * O)
    for i in range(B * D):
        x.data[i] = Scalar[DT]((i % 5) - 2) * 0.3
    for i in range(B * O):
        tgt.data[i] = Scalar[DT](((i * 3) % 7) - 3) * 0.2
    var out = Tensor.alloc(B * O)
    var go = Tensor.alloc(B * O)
    var gi = Tensor.alloc(B * D)
    comptime if target == "gpu":
        x.upload(ctx.value())

    # initial loss
    net.forward[target, B](TensorRefs[1](x), out, ctx)
    comptime if target == "gpu":
        out.download(ctx.value())
    var loss0 = Scalar[DT](0.0)
    for i in range(B * O):
        var d0 = out.data[i] - tgt.data[i]
        loss0 += d0 * d0

    for _ in range(K):
        net.forward[target, B](TensorRefs[1](x), out, ctx)
        comptime if target == "gpu":
            out.download(ctx.value())
        for i in range(B * O):
            go.data[i] = out.data[i] - tgt.data[i]  # ∝ MSE gradient
        comptime if target == "gpu":
            go.upload(ctx.value())
        opt.zero_grad[target, NET](net, ctx)
        net.vjp[target, B](TensorRefs[1](x), go, TensorRefs[1](gi), ctx)
        opt.step[target, NET](net, ctx)

    net.forward[target, B](TensorRefs[1](x), out, ctx)
    comptime if target == "gpu":
        out.download(ctx.value())
    var lossK = Scalar[DT](0.0)
    for i in range(B * O):
        var dk = out.data[i] - tgt.data[i]
        lossK += dk * dk
    for i in range(B * O):
        final_out.append(out.data[i])
    print("    [", target, "] loss0=", loss0, " lossK=", lossK)
    return lossK < loss0 * Scalar[DT](0.9)  # clearly reduced


def main() raises:
    print("DreamerV3 storage foundations (Symlog + DreamerOpt)")
    var c = DeviceContext()

    var s_cpu = _symlog_ok["cpu"](None)
    var s_gpu = _symlog_ok["gpu"](Optional(c))
    print("  Symlog  CPU:", "OK" if s_cpu else "FAIL",
          " GPU:", "OK" if s_gpu else "FAIL")

    var out_cpu = List[Scalar[DT]]()
    var out_gpu = List[Scalar[DT]]()
    var d_cpu = _dreamer_train["cpu"](None, out_cpu)
    var d_gpu = _dreamer_train["gpu"](Optional(c), out_gpu)
    print("  DreamerOpt converges  CPU:", "OK" if d_cpu else "FAIL",
          " GPU:", "OK" if d_gpu else "FAIL")

    var parity = True
    for i in range(len(out_cpu)):
        if _abs(out_cpu[i] - out_gpu[i]) > Scalar[DT](1e-4):
            parity = False
    print("  DreamerOpt CPU/GPU parity:", "OK" if parity else "FAIL")

    assert_true(
        s_cpu and s_gpu and d_cpu and d_gpu and parity,
        "DreamerV3 storage foundations",
    )
    print("DREAMER FOUNDATIONS OK")
