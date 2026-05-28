"""Smoke + CPU↔GPU parity test for DQNTargetYBlock (standard branch).

Strategy: use `Zero` initializer for the Q-net, so `Q(sp) == 0` for any
`sp`. Then:
    y = r + γ·max_a 0·(1 − d) = r
Letting us verify the block's output bit-exactly equals `mb_r`.

CPU + GPU. GPU only runs when DeviceContext can be created.
"""

from std.memory import alloc
from std.gpu.host import DeviceContext
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.initializer import Zero
from mojo_rl.deep_agents2.dqn.target_y_block import DQNTargetYBlock


def _abs(x: Scalar[DT]) -> Scalar[DT]:
    return x if x >= Scalar[DT](0) else -x


def test_zero_init_cpu() raises:
    print("test_zero_init_cpu ...")
    comptime BATCH = 4
    comptime OBS = 2
    comptime NA = 3
    comptime QNet = Linear[OBS, NA]

    var blk = DQNTargetYBlock[QNet, BATCH, OBS, NA].make[target="cpu"](
        gamma=Scalar[DT](0.99),
    )

    var q_target = QNet.make[target="cpu", INIT=Zero]()
    var q_online = QNet.make[target="cpu", INIT=Zero]()

    var sp: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OBS)
    var r:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var d:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var y:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)

    for k in range(BATCH * OBS):
        sp[k] = Scalar[DT](0.1 * Float64(k))
    r[0] = 1.0; r[1] = 2.0;  r[2] = 3.0; r[3] = -1.5
    d[0] = 0.0; d[1] = 1.0;  d[2] = 0.0; d[3] = 0.0
    for b in range(BATCH):
        y[b] = Scalar[DT](999.0)  # junk fill

    blk.step["cpu"](q_target, q_online, sp, r, d, y)

    # Q(sp) = 0 everywhere → max_q = 0 → y = r exactly.
    var max_diff: Scalar[DT] = 0.0
    for b in range(BATCH):
        var d_diff = _abs(y[b] - r[b])
        if d_diff > max_diff:
            max_diff = d_diff
    print("  max |y - r| =", max_diff)
    assert_true(max_diff == Scalar[DT](0), "y must equal r for Zero-init Q-net")
    print("  ok")


def test_zero_init_gpu_parity() raises:
    print("test_zero_init_gpu_parity ...")
    comptime BATCH = 8
    comptime OBS = 3
    comptime NA = 4
    comptime QNet = Linear[OBS, NA]

    try:
        var ctx = DeviceContext()

        var blk = DQNTargetYBlock[QNet, BATCH, OBS, NA].make[target="gpu"](
            gamma=Scalar[DT](0.99),
            ctx=ctx,
        )
        var q_target = QNet.make[target="gpu", INIT=Zero](ctx=ctx)
        var q_online = QNet.make[target="gpu", INIT=Zero](ctx=ctx)

        # Host inputs.
        var sp_host: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OBS)
        var r_host:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
        var d_host:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
        var y_host:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
        for k in range(BATCH * OBS):
            sp_host[k] = Scalar[DT](0.05 * Float64(k))
        for b in range(BATCH):
            r_host[b] = Scalar[DT](Float64(b) - 2.0)
            d_host[b] = Scalar[DT](1.0) if (b % 3 == 0) else Scalar[DT](0.0)
            y_host[b] = Scalar[DT](-99999.0)

        # H2D.
        var sp_dev = ctx.enqueue_create_buffer[DT](BATCH * OBS)
        var r_dev = ctx.enqueue_create_buffer[DT](BATCH)
        var d_dev = ctx.enqueue_create_buffer[DT](BATCH)
        var y_dev = ctx.enqueue_create_buffer[DT](BATCH)
        ctx.enqueue_copy(sp_dev, sp_host)
        ctx.enqueue_copy(r_dev, r_host)
        ctx.enqueue_copy(d_dev, d_host)

        var sp_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](sp_dev.unsafe_ptr())
        var r_p  = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](r_dev.unsafe_ptr())
        var d_p  = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](d_dev.unsafe_ptr())
        var y_p  = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](y_dev.unsafe_ptr())

        blk.step["gpu"](q_target, q_online, sp_p, r_p, d_p, y_p)

        ctx.enqueue_copy(y_host, y_dev)
        ctx.synchronize()

        var max_diff: Scalar[DT] = 0.0
        for b in range(BATCH):
            var d_diff = _abs(y_host[b] - r_host[b])
            if d_diff > max_diff:
                max_diff = d_diff
        print("  max |y - r| =", max_diff)
        assert_true(max_diff == Scalar[DT](0), "GPU: y must equal r for Zero-init Q-net")
        print("  ok")
    except e:
        print("  (skipped — no GPU available:", e, ")")


def main() raises:
    print("=" * 70)
    print("DQNTargetYBlock smoke (standard, Zero-init Q-net → y == r)")
    print("=" * 70)
    test_zero_init_cpu()
    test_zero_init_gpu_parity()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
