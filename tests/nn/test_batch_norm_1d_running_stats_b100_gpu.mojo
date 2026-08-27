"""BatchNorm1D running-stat test at BATCH=100 (large-batch EMA store path).

The 1D analog of test_batch_norm_2d_running_stats_b100_gpu.mojo. Feeds a fixed
batch in training mode many times so running_mean/running_var must converge to
the batch stats, then asserts the running stats match host-computed batch stats
and eval-mode output (running stats) matches train-mode output (batch stats).
Guards the same NVIDIA large-batch EMA store-drop that hit BatchNorm2D.

Run (Apple):  pixi run -e apple  mojo run -I . tests/nn/test_batch_norm_1d_running_stats_b100_gpu.mojo
Run (NVIDIA): pixi run -e nvidia mojo run -I . tests/nn/test_batch_norm_1d_running_stats_b100_gpu.mojo
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import child_refs
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.primitives.batch_norm_1d import BatchNorm1D


def main() raises:
    comptime DIM = 6
    comptime B = 100
    var ctx = DeviceContext()
    var bn = BatchNorm1D[DIM].make["gpu", Kaiming](Optional(ctx))

    var inp = Tensor.alloc(B * DIM)
    for b in range(B):
        for f in range(DIM):
            var idx = b * DIM + f
            var v = (
                Float32((idx * 7 % 13) - 6) * Float32(f + 1) * 0.1
                + Float32(f + 1)
            )
            inp.data[idx] = Scalar[DT](v)
    inp.upload(ctx)

    var hmean = List[Float64](length=DIM, fill=0.0)
    var hvar = List[Float64](length=DIM, fill=0.0)
    for f in range(DIM):
        var m: Float64 = 0.0
        for b in range(B):
            m += Float64(inp.data[b * DIM + f])
        m /= Float64(B)
        var v: Float64 = 0.0
        for b in range(B):
            var d = Float64(inp.data[b * DIM + f]) - m
            v += d * d
        v /= Float64(B)
        hmean[f] = m
        hvar[f] = v

    var out = Tensor()
    bn.set_attr["training"](Scalar[DT](1.0))
    for _ in range(800):
        bn.forward["gpu", B](child_refs[1, DT](inp), out, Optional(ctx))
    ctx.synchronize()
    out.download(ctx)
    var train_out = List[Scalar[DT]](length=B * DIM, fill=0.0)
    for i in range(B * DIM):
        train_out[i] = out.data[i]

    bn.running_mean.t.download(ctx)
    bn.running_var.t.download(ctx)
    var stats_ok = True
    for f in range(DIM):
        var dm = abs(Float64(bn.running_mean.t.data[f]) - hmean[f])
        var dv = abs(Float64(bn.running_var.t.data[f]) - hvar[f])
        print(
            "f", f,
            "| running_mean", Float64(bn.running_mean.t.data[f]),
            "(host", hmean[f], ") d=", dm,
            "| running_var", Float64(bn.running_var.t.data[f]),
            "(host", hvar[f], ") d=", dv,
        )
        if dm > 1e-2 or dv > 1e-2:
            stats_ok = False

    bn.set_attr["training"](Scalar[DT](0.0))
    var out_eval = Tensor()
    bn.forward["gpu", B](child_refs[1, DT](inp), out_eval, Optional(ctx))
    ctx.synchronize()
    out_eval.download(ctx)
    var max_abs_diff: Float64 = 0.0
    for i in range(B * DIM):
        var d = abs(Float64(train_out[i]) - Float64(out_eval.data[i]))
        if d > max_abs_diff:
            max_abs_diff = d
    print("\nmax |train_out - eval_out| =", max_abs_diff)

    assert_true(
        stats_ok,
        "BN1D running stats did NOT converge to batch stats at B=100 (large-batch"
        " EMA store path).",
    )
    assert_true(
        max_abs_diff < 1e-2,
        "BN1D eval output diverges from train output at B=100.",
    )
    print("PASS")
