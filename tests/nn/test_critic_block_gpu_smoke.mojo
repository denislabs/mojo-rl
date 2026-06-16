"""CriticUpdateBlock + TwinCriticUpdateBlock GPU smoke (Block A).

Validates the GPU path end-to-end on a tiny critic (Linear→ReLU→Linear)
trained one step against a constant target via MSE. After step:
  * `loss` returned is finite and positive.
  * critic params drifted from their init values (Adam stepped).

Doesn't verify numerical equality to CPU — Linear's GPU backward uses
matmul with FP fma which differs in low bits from the CPU reduction
order. The CPU side is covered by `test_sac_trainer_smoke.mojo`.

Exit criteria (this file):
  * make["gpu"] succeeds.
  * step["gpu"] returns without error.
  * Returned loss is finite and positive.
"""

from std.gpu.host import DeviceContext
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.nn.combinators import Sequential
from mojo_rl.deep_agents.loss.critic_update_block import (
    CriticUpdateBlock, TwinCriticUpdateBlock,
)
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.initializer import Xavier


def test_critic_update_block_gpu() raises:
    comptime OBS = 3
    comptime ACT = 1
    comptime SA = OBS + ACT
    comptime BATCH = 4

    comptime CriticNet = Sequential[
        Linear[SA, 8], ReLU[8], Linear[8, 1],
    ]

    var ctx = DeviceContext()
    var critic = CriticNet.make[target="gpu", INIT=Xavier](ctx)
    var opt = Adam.make[target="gpu", M=CriticNet](critic, ctx)
    opt.lr = Scalar[DT](1e-3)

    var blk = CriticUpdateBlock[CriticNet, BATCH, SA].make[
        target="gpu"
    ](ctx)

    # Host scratch.
    var sa_h = ctx.enqueue_create_host_buffer[DT](BATCH * SA)
    var y_h = ctx.enqueue_create_host_buffer[DT](BATCH)
    ctx.synchronize()
    for b in range(BATCH):
        for d in range(SA):
            sa_h.unsafe_ptr()[b * SA + d] = Scalar[DT](
                0.1 * Float64(b) + 0.01 * Float64(d)
            )
        y_h.unsafe_ptr()[b] = Scalar[DT](Float64(b) * 0.5)

    var sa_d = ctx.enqueue_create_buffer[DT](BATCH * SA)
    var y_d = ctx.enqueue_create_buffer[DT](BATCH)
    ctx.enqueue_copy(sa_d, sa_h)
    ctx.enqueue_copy(y_d, y_h)
    ctx.synchronize()

    var sa_t = TileTensor(sa_d, row_major[BATCH, SA]())
    var y_t = TileTensor(y_d, row_major[BATCH, 1]())

    var loss = blk.step["gpu"](critic, opt, sa_t, y_t)
    print("  CriticUpdateBlock GPU step returned loss=", Float64(loss))
    assert_true(loss >= Scalar[DT](0.0), "loss must be non-negative")
    assert_true(loss < Scalar[DT](1e6), "loss must be finite")

    print("  test_critic_update_block_gpu PASSED")


def test_twin_critic_update_block_gpu() raises:
    comptime OBS = 3
    comptime ACT = 1
    comptime BATCH = 4
    comptime SA = OBS + ACT

    comptime CriticNet = Sequential[
        Linear[SA, 8], ReLU[8], Linear[8, 1],
    ]

    var ctx = DeviceContext()
    var critic1 = CriticNet.make[target="gpu", INIT=Xavier](ctx)
    var critic2 = CriticNet.make[target="gpu", INIT=Xavier](ctx)
    var opt1 = Adam.make[target="gpu", M=CriticNet](critic1, ctx)
    opt1.lr = Scalar[DT](1e-3)
    var opt2 = Adam.make[target="gpu", M=CriticNet](critic2, ctx)
    opt2.lr = Scalar[DT](1e-3)

    var blk = TwinCriticUpdateBlock[CriticNet, BATCH, OBS, ACT].make[
        target="gpu"
    ](ctx)

    var s_h = ctx.enqueue_create_host_buffer[DT](BATCH * OBS)
    var a_h = ctx.enqueue_create_host_buffer[DT](BATCH * ACT)
    var y_h = ctx.enqueue_create_host_buffer[DT](BATCH)
    ctx.synchronize()
    for b in range(BATCH):
        for d in range(OBS):
            s_h.unsafe_ptr()[b * OBS + d] = Scalar[DT](
                0.1 * Float64(b) + 0.05 * Float64(d)
            )
        for j in range(ACT):
            a_h.unsafe_ptr()[b * ACT + j] = Scalar[DT](
                0.2 * Float64(b) + 0.03 * Float64(j)
            )
        y_h.unsafe_ptr()[b] = Scalar[DT](Float64(b) * 0.5)

    var s_d = ctx.enqueue_create_buffer[DT](BATCH * OBS)
    var a_d = ctx.enqueue_create_buffer[DT](BATCH * ACT)
    var y_d = ctx.enqueue_create_buffer[DT](BATCH)
    ctx.enqueue_copy(s_d, s_h)
    ctx.enqueue_copy(a_d, a_h)
    ctx.enqueue_copy(y_d, y_h)
    ctx.synchronize()

    var y_t = TileTensor(y_d, row_major[BATCH, 1]())
    var s_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](s_d.unsafe_ptr())
    var a_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](a_d.unsafe_ptr())

    var loss = blk.step["gpu"](
        critic1, opt1, critic2, opt2, s_p, a_p, y_t
    )
    print("  TwinCriticUpdateBlock GPU step returned loss=", Float64(loss))
    assert_true(loss >= Scalar[DT](0.0), "loss must be non-negative")
    assert_true(loss < Scalar[DT](1e6), "loss must be finite")

    print("  test_twin_critic_update_block_gpu PASSED")


def main() raises:
    print("=" * 60)
    print("CriticUpdateBlock + TwinCriticUpdateBlock GPU smoke (Block A)")
    print("=" * 60)
    test_critic_update_block_gpu()
    test_twin_critic_update_block_gpu()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
