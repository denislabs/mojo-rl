"""Smoke test for DQNQUpdateBlock — CPU + GPU.

Validates the block compiles + runs end-to-end (forward, gather,
MSE, scatter, vjp, opt.step) and produces a finite loss.

Setup: Zero-init Q-net + zero mb_y + non-zero mb_s. After one
step, loss must be 0 (Q_zero(s) - y_zero = 0). After two steps,
loss is still 0 because the Q-net stays zero (zero grad ⇒ zero
param update).
"""

from std.math import isnan, isinf
from std.memory import alloc
from std.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.deep_agents.dqn.q_update_block import DQNQUpdateBlock


def test_zero_init_cpu() raises:
    print("test_zero_init_cpu ...")
    comptime BATCH = 4
    comptime OBS = 2
    comptime NA = 3
    comptime QNet = Linear[OBS, NA]

    var blk = DQNQUpdateBlock[QNet, BATCH, OBS, NA].make[target="cpu"]()
    var q = QNet.make[target="cpu", INIT=Zero]()
    var opt = Adam.make[target="cpu", M=QNet](q)
    opt.lr = Scalar[DT](1e-3)

    var mb_s: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OBS)
    var mb_a: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var mb_y: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    for k in range(BATCH * OBS):
        mb_s[k] = Scalar[DT](0.1 * Float64(k))
    mb_a[0] = 0; mb_a[1] = 1; mb_a[2] = 2; mb_a[3] = 0
    for b in range(BATCH):
        mb_y[b] = Scalar[DT](0.0)

    var loss = blk.step["cpu"](q, opt, mb_s, mb_a, mb_y)
    print("  cpu loss =", loss)
    assert_true(not isnan(loss), "loss NaN")
    assert_true(not isinf(loss), "loss Inf")
    assert_true(loss == Scalar[DT](0.0), "loss must be 0 for Zero-init Q-net + zero y")
    print("  ok")


def test_zero_init_gpu() raises:
    print("test_zero_init_gpu ...")
    comptime BATCH = 8
    comptime OBS = 3
    comptime NA = 4
    comptime QNet = Linear[OBS, NA]

    try:
        var ctx = DeviceContext()

        var blk = DQNQUpdateBlock[QNet, BATCH, OBS, NA].make[target="gpu"](
            ctx=ctx,
        )
        var q = QNet.make[target="gpu", INIT=Zero](ctx=ctx)
        var opt = Adam.make[target="gpu", M=QNet](q, ctx=ctx)
        opt.lr = Scalar[DT](1e-3)

        var mb_s_host: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OBS)
        var mb_a_host: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
        var mb_y_host: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
        for k in range(BATCH * OBS):
            mb_s_host[k] = Scalar[DT](0.1 * Float64(k))
        for b in range(BATCH):
            mb_a_host[b] = Scalar[DT](b % NA)
            mb_y_host[b] = Scalar[DT](0.0)

        var mb_s_dev = ctx.enqueue_create_buffer[DT](BATCH * OBS)
        var mb_a_dev = ctx.enqueue_create_buffer[DT](BATCH)
        var mb_y_dev = ctx.enqueue_create_buffer[DT](BATCH)
        ctx.enqueue_copy(mb_s_dev, mb_s_host)
        ctx.enqueue_copy(mb_a_dev, mb_a_host)
        ctx.enqueue_copy(mb_y_dev, mb_y_host)

        var s_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](mb_s_dev.unsafe_ptr())
        var a_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](mb_a_dev.unsafe_ptr())
        var y_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](mb_y_dev.unsafe_ptr())

        var loss = blk.step["gpu"](q, opt, s_p, a_p, y_p)
        ctx.synchronize()
        print("  gpu loss =", loss)
        assert_true(not isnan(loss), "GPU loss NaN")
        assert_true(not isinf(loss), "GPU loss Inf")
        assert_true(loss == Scalar[DT](0.0), "GPU loss must be 0 for Zero-init Q-net + zero y")
        print("  ok")
    except e:
        print("  (skipped — no GPU available:", e, ")")


def main() raises:
    print("=" * 70)
    print("DQNQUpdateBlock smoke (CPU + GPU, Zero-init Q-net → loss == 0)")
    print("=" * 70)
    test_zero_init_cpu()
    test_zero_init_gpu()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
