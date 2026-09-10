"""Per-step cost of Adam's per-param walk on an MLP: CPU, and the un-adopted
GPU path (one kernel per Param). Times `step` only. Run before and after the
runtime-visitor change; the arena path is not touched by it."""
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.core.module import Module

comptime NET = Sequential[Linear[784, 512], Linear[512, 512], Linear[512, 512], Linear[512, 10]]
comptime SMALL = Sequential[Linear[16, 32], Linear[32, 32], Linear[32, 32], Linear[32, 4]]
comptime STEPS = 200


def bench[N: Module, target: StaticString](label: String, ctx: Optional[DeviceContext]) raises:
    var net = N.make[target, Deterministic](ctx)
    var opt = Adam(lr=Scalar[DT](1e-3))
    net.zero_grad[target](ctx)
    # warm-up (allocates the moments on the first step)
    for _ in range(5):
        opt.step[target](net, ctx)
    if ctx:
        ctx.value().synchronize()
    var t0 = perf_counter_ns()
    for _ in range(STEPS):
        opt.step[target](net, ctx)
    if ctx:
        ctx.value().synchronize()
    var dt = Float64(perf_counter_ns() - t0) / Float64(STEPS)
    print(label, target, "us/step =", dt / 1000.0)


def main() raises:
    bench[NET, "cpu"]("mlp 784-512-512-512-10 ", None)
    bench[SMALL, "cpu"]("mlp 16-32-32-32-4      ", None)
    try:
        var ctx = DeviceContext()
        bench[NET, "gpu"]("mlp 784-512-512-512-10 ", Optional(ctx))
        bench[SMALL, "gpu"]("mlp 16-32-32-32-4      ", Optional(ctx))
    except e:
        print("gpu: skipped:", e)
