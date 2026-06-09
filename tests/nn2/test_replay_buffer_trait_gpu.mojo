"""ReplayBuffer trait — GPU conformance smoke (Apple/NVIDIA).

Same generic `drive[R: ReplayBuffer]` as the CPU test, with GPUReplay.
Proves GPUReplay conforms and the device add/sample path works through
the trait surface.
"""

from std.random import seed
from std.testing import assert_equal, assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.training.replay_buffer import ReplayBuffer
from mojo_rl.deep_agents2.training.trainer_block import TrainerState
from mojo_rl.deep_agents2.data.gpu_replay import GPUReplay
from mojo_rl.deep_agents2.data.per_replay import GPUPrioritizedReplay


comptime OBS = 3
comptime ACT = 1
comptime CAP = 64
comptime BATCH = 8


def drive[R: ReplayBuffer](mut buf: R, mut state: TrainerState[R.OBS, R.ACT, BATCH], ctx: DeviceContext) raises:
    var s = List[Scalar[DT]](length=R.OBS, fill=Scalar[DT](0.0))
    var a = List[Scalar[DT]](length=R.ACT, fill=Scalar[DT](0.0))
    var sp = List[Scalar[DT]](length=R.OBS, fill=Scalar[DT](0.0))
    for t in range(20):
        for d in range(R.OBS):
            s[d] = Scalar[DT](t)
            sp[d] = Scalar[DT](t + 1)
        a[0] = Scalar[DT](t % 2)
        buf.add(s, a, Scalar[DT](t), sp, Scalar[DT](0.0), ctx=ctx)
    buf.sample_into[BATCH](state)


def main() raises:
    seed(42)
    var ctx = DeviceContext()
    var buf = GPUReplay[OBS, ACT, CAP].make(ctx=ctx, batch_capacity=BATCH)
    var state = TrainerState[OBS, ACT, BATCH].make["gpu"](ctx=ctx)
    drive(buf, state, ctx)
    ctx.synchronize()
    assert_equal(buf.count(), 20)
    print("  GPUReplay conforms: count=", buf.count())

    # GPUPrioritizedReplay conforms through the SAME generic drive[R].
    var per = GPUPrioritizedReplay[OBS, ACT, CAP].make(
        ctx=ctx, batch_capacity=BATCH
    )
    per.configure_per(alpha=0.6, beta=0.4, epsilon=1e-6)
    var pstate = TrainerState[OBS, ACT, BATCH].make["gpu"](ctx=ctx)
    drive(per, pstate, ctx)
    ctx.synchronize()
    assert_equal(per.count(), 20)
    assert_true(pstate.has_per, "GPU PER sample_into must flip has_per")
    print("  GPUPrioritizedReplay conforms: count=", per.count())
    print("ALL PASSED")
