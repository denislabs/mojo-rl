"""ReplayBuffer trait — CPU conformance smoke.

Exercises the trait surface (make / add / sample_into / count) through a
generic `def drive[R: ReplayBuffer]`, proving CPUReplay conforms and the
generic path works end-to-end. GPU conformance is covered by
test_replay_buffer_trait_gpu.mojo + the existing C51 GPU smokes.
"""

from std.random import seed
from std.testing import assert_equal, assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.training.replay_buffer import ReplayBuffer
from mojo_rl.deep_agents.training.trainer_block import TrainerState
from mojo_rl.deep_agents.data.cpu_replay import CPUReplay
from mojo_rl.deep_agents.data.cpu_per_replay import CPUPrioritizedReplay


comptime OBS = 3
comptime ACT = 1
comptime CAP = 64
comptime BATCH = 8


def drive[R: ReplayBuffer](mut buf: R, mut state: TrainerState[R.OBS, R.ACT, BATCH]) raises:
    var s = List[Scalar[DT]](length=R.OBS, fill=Scalar[DT](0.0))
    var a = List[Scalar[DT]](length=R.ACT, fill=Scalar[DT](0.0))
    var sp = List[Scalar[DT]](length=R.OBS, fill=Scalar[DT](0.0))
    for t in range(20):
        for d in range(R.OBS):
            s[d] = Scalar[DT](t)
            sp[d] = Scalar[DT](t + 1)
        a[0] = Scalar[DT](t % 2)
        buf.add(s, a, Scalar[DT](t), sp, Scalar[DT](0.0))
    buf.sample_into[BATCH](state)


def main() raises:
    seed(42)
    var buf = CPUReplay[OBS, ACT, CAP].make()
    var state = TrainerState[OBS, ACT, BATCH].make["cpu"]()
    drive(buf, state)
    assert_equal(buf.count(), 20)
    # mb_s should hold a sampled obs (each row's 3 dims equal, value in [0,19]).
    var v0 = state.mb_s.data[0]
    assert_true(v0 >= 0.0 and v0 <= 19.0, "sampled obs out of range")
    print("  CPUReplay conforms: count=", buf.count(), " mb_s[0]=", v0)

    # CPUPrioritizedReplay conforms through the SAME generic drive[R].
    var per = CPUPrioritizedReplay[OBS, ACT, CAP].make()
    per.configure_per(alpha=0.6, beta=0.4, epsilon=1e-6)
    var pstate = TrainerState[OBS, ACT, BATCH].make["cpu"]()
    drive(per, pstate)
    assert_equal(per.count(), 20)
    # PER sample_into must have filled IS weights (normalised, max==1).
    var w0 = pstate.mb_w.data[0]
    assert_true(pstate.has_per, "PER sample_into must flip has_per")
    assert_true(w0 > 0.0 and w0 <= 1.0001, "IS weight out of range")
    print("  CPUPrioritizedReplay conforms: count=", per.count(), " w0=", w0)
    print("ALL PASSED")
