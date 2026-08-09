"""Isolated GPU BatchNorm2D running-stat regression test.

Localizes the CIFAR/ResNet eval collapse (train climbs, test pinned ~random)
seen on NVIDIA but NOT Apple. Feeds a FIXED batch in training mode many times so
the running mean/var must converge to the batch stats; then asserts:

  1. running_mean / running_var ≈ host-computed batch mean / var, and
  2. eval-mode output (running stats) ≈ train-mode output (batch stats),

with gamma=1, beta=0. If (1) fails the finalize-kernel EMA write isn't
persisting; if (1) holds but (2) fails the eval kernel reads stats wrong.

Run (Apple):  pixi run -e apple  mojo run -I . tests/nn/test_batch_norm_2d_running_stats_gpu.mojo
Run (NVIDIA): pixi run -e nvidia mojo run -I . tests/nn/test_batch_norm_2d_running_stats_gpu.mojo
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import child_refs
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.primitives.batch_norm_2d import BatchNorm2D


def main() raises:
    comptime C = 4
    comptime H = 4
    comptime W = 4
    comptime B = 8
    comptime SP = H * W
    comptime FLAT = C * SP
    var ctx = DeviceContext()
    var bn = BatchNorm2D[C, H, W].make["gpu", Kaiming](Optional(ctx))

    # Fixed batch — each channel has a distinct mean/scale.
    var inp = Tensor.alloc(B * FLAT)
    for b in range(B):
        for c in range(C):
            for s in range(SP):
                var idx = b * FLAT + c * SP + s
                var v = (
                    Float32((idx * 7 % 13) - 6) * Float32(c + 1) * 0.1
                    + Float32(c + 1)
                )
                inp.data[idx] = Scalar[DT](v)
    inp.upload(ctx)

    # Host (biased) batch mean/var per channel — the convergence target.
    var hmean = List[Float64](length=C, fill=0.0)
    var hvar = List[Float64](length=C, fill=0.0)
    for c in range(C):
        var m: Float64 = 0.0
        for b in range(B):
            for s in range(SP):
                m += Float64(inp.data[b * FLAT + c * SP + s])
        m /= Float64(B * SP)
        var v: Float64 = 0.0
        for b in range(B):
            for s in range(SP):
                var d = Float64(inp.data[b * FLAT + c * SP + s]) - m
                v += d * d
        v /= Float64(B * SP)
        hmean[c] = m
        hvar[c] = v

    # Drive training mode → running stats EMA toward the (fixed) batch stats.
    var out = Tensor()
    bn.set_attr["training"](Scalar[DT](1.0))
    for _ in range(800):
        bn.forward["gpu", B](child_refs[1, DT](inp), out, Optional(ctx))
    ctx.synchronize()
    out.download(ctx)
    var train_out = List[Scalar[DT]](length=B * FLAT, fill=0.0)
    for i in range(B * FLAT):
        train_out[i] = out.data[i]

    # (1) running stats must have converged to the batch stats.
    bn.running_mean.t.download(ctx)
    bn.running_var.t.download(ctx)
    var stats_ok = True
    for c in range(C):
        var dm = abs(Float64(bn.running_mean.t.data[c]) - hmean[c])
        var dv = abs(Float64(bn.running_var.t.data[c]) - hvar[c])
        print(
            "c", c,
            "| running_mean", Float64(bn.running_mean.t.data[c]),
            "(host", hmean[c], ") d=", dm,
            "| running_var", Float64(bn.running_var.t.data[c]),
            "(host", hvar[c], ") d=", dv,
        )
        if dm > 1e-2 or dv > 1e-2:
            stats_ok = False

    # (2) eval-mode output (running stats) must match train-mode (batch stats).
    bn.set_attr["training"](Scalar[DT](0.0))
    var out_eval = Tensor()
    bn.forward["gpu", B](child_refs[1, DT](inp), out_eval, Optional(ctx))
    ctx.synchronize()
    out_eval.download(ctx)
    var max_abs_diff: Float64 = 0.0
    for i in range(B * FLAT):
        var d = abs(Float64(train_out[i]) - Float64(out_eval.data[i]))
        if d > max_abs_diff:
            max_abs_diff = d
    print("\nmax |train_out - eval_out| =", max_abs_diff)

    assert_true(
        stats_ok,
        "BN running stats did NOT converge to batch stats (finalize-kernel EMA"
        " write not persisting on this backend).",
    )
    assert_true(
        max_abs_diff < 1e-2,
        "BN eval output diverges from train output despite converged running"
        " stats (eval kernel reads stats wrong on this backend).",
    )
    print("PASS")
