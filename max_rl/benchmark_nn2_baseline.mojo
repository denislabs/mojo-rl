"""nn2 native-Mojo MLP inference baseline — the "why incorporate MAX?" comparison.

Runs the SAME MLP shapes as `benchmark_interop.mojo` through nn2's native GPU forward
pass, so MAX's "MAX device compute" line can be compared directly against nn2's on-device
compute. Same device, same dims, same steady-state (input already on device) regime.

The honest framing for a Mojo caller:
  * MAX path  : compute + H2D + D2H + Python glue (+ ~0 interop floor)   [see interop bench]
  * nn2 path  : compute only — data is already in Mojo GPU buffers, no Python, no transfer.
So nn2's *delivered* latency to a Mojo caller is just the number below; MAX must overcome
its transfer + Python-glue tax (~hundreds of us, see benchmark_interop.mojo) to win.

This file is pure nn2 (no Python / no max.engine), so plain `mojo run` is fine:
  pixi run -e apple  mojo run -I . max_rl/benchmark_nn2_baseline.mojo
  pixi run -e nvidia mojo run -I . max_rl/benchmark_nn2_baseline.mojo
"""

from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators import Sequential
from mojo_rl.nn2.initializer import Kaiming


def f2(x: Float64) -> String:
    var neg = x < 0.0
    var v = -x if neg else x
    var scaled = Int(v * 100.0 + 0.5)
    var whole = scaled // 100
    var frac = scaled % 100
    var fs = String(frac)
    if frac < 10:
        fs = "0" + fs
    var s = String(whole) + "." + fs
    return "-" + s if neg else s


def bench_nn2[
    IN: Int, H1: Int, H2: Int, OUT: Int, BATCH: Int
](ctx: DeviceContext, name: String, iters: Int) raises:
    # MLP: Linear+ReLU(IN->H1) -> Linear+ReLU(H1->H2) -> Linear(H2->OUT).
    var net = Sequential(
        Linear[IN, H1].make["gpu", INIT=Kaiming](ctx),
        ReLU[H1].make["gpu", INIT=Kaiming](ctx),
        Linear[H1, H2].make["gpu", INIT=Kaiming](ctx),
        ReLU[H2].make["gpu", INIT=Kaiming](ctx),
        Linear[H2, OUT].make["gpu", INIT=Kaiming](ctx),
        ctx=ctx,
    )

    var x_dev = ctx.enqueue_create_buffer[DT](BATCH * IN)
    var y_dev = ctx.enqueue_create_buffer[DT](BATCH * OUT)
    x_dev.enqueue_fill(Scalar[DT](0.1))
    y_dev.enqueue_fill(Scalar[DT](0.0))
    ctx.synchronize()

    var x_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = x_dev.unsafe_ptr()
    var y_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = y_dev.unsafe_ptr()
    var x_t = TileTensor(x_p, row_major[BATCH, IN]())
    var y_t = TileTensor(y_p, row_major[BATCH, OUT]())

    # warmup
    for _ in range(50):
        net.forward["gpu", BATCH](x_t, output=y_t)
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(iters):
        net.forward["gpu", BATCH](x_t, output=y_t)
    ctx.synchronize()
    var per = Float64(perf_counter_ns() - t0) / Float64(iters) / 1000.0  # us

    var params = (IN * H1 + H1) + (H1 * H2 + H2) + (H2 * OUT + OUT)
    print(
        "  "
        + name
        + ":  "
        + String(IN)
        + "->"
        + String(H1)
        + "->"
        + String(H2)
        + "->"
        + String(OUT)
        + " batch="
        + String(BATCH)
        + " params="
        + String(params)
        + "   nn2 forward = "
        + f2(per)
        + " us/call"
    )


def main() raises:
    var ctx = DeviceContext()
    var iters = 2000
    print("nn2 native GPU MLP inference baseline (on-device compute, no Python/transfer)")
    print("  (compare 'nn2 forward us/call' vs MAX 'device compute' in benchmark_interop)")
    print("")
    bench_nn2[17, 256, 256, 6, 1](ctx, "actor-b1", iters)
    bench_nn2[17, 256, 256, 6, 64](ctx, "actor-b64", iters)
    bench_nn2[17, 256, 256, 6, 1024](ctx, "actor-b1024", iters)
    bench_nn2[256, 512, 512, 64, 1](ctx, "wide-b1", iters)
    bench_nn2[256, 512, 512, 64, 1024](ctx, "wide-b1024", iters)
