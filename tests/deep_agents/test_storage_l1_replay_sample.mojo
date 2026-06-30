"""L1 gate — replay buffers sample into the storage TrainerState data bus.

The seam: `replay.sample_into[BATCH](state)` writes a minibatch into the storage
`TrainerState.mb_*` (owned Tensors). Verifies (CPU + GPU):
  - CPUReplay: add transitions -> sample_into fills mb_s/a/r/sp/d host lists.
  - CPUPrioritizedReplay: sample_into fills mb_* + mb_w (IS weights) + flips
    has_per; update_priorities consumes td_residuals (storage List path).
  - GPUReplay: add (host-staged) -> sample_into gathers into the device mirrors;
    download mb_s and confirm the values came from the buffer.

Run:
  pixi run mojo run -I . tests/deep_agents/test_storage_l1_replay_sample.mojo
  pixi run -e apple mojo run -I . tests/deep_agents/test_storage_l1_replay_sample.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.training.trainer_block import TrainerState
from mojo_rl.deep_agents.data.cpu_replay import CPUReplay
from mojo_rl.deep_agents.data.cpu_per_replay import CPUPrioritizedReplay
from mojo_rl.deep_agents.data.gpu_replay import GPUReplay


comptime OBS = 3
comptime ACT = 1
comptime CAP = 256
comptime BATCH = 8


def _mk_obs(base: Scalar[DT]) -> List[Scalar[DT]]:
    var l = List[Scalar[DT]](capacity=OBS)
    for i in range(OBS):
        l.append(base + Scalar[DT](i) * 0.1)
    return l^


def _mk_act(base: Scalar[DT]) -> List[Scalar[DT]]:
    var l = List[Scalar[DT]](capacity=ACT)
    for j in range(ACT):
        l.append(base + Scalar[DT](j) * 0.01)
    return l^


def test_cpu_uniform() raises:
    print("CPUReplay -> storage TrainerState ...")
    var buf = CPUReplay[OBS, ACT, CAP].make()
    for k in range(64):
        var s = _mk_obs(Scalar[DT](k))
        var a = _mk_act(Scalar[DT](k))
        var sp = _mk_obs(Scalar[DT](k) + 100.0)
        buf.add(s, a, Scalar[DT](k) * 0.5, sp, Scalar[DT](0.0))
    var st = TrainerState[OBS, ACT, BATCH].make["cpu"]()
    buf.sample_into[BATCH](st)
    # every sampled obs row must be one of the stored rows (s[0] is an integer)
    var ok = True
    for b in range(BATCH):
        var s0 = st.mb_s.data[b * OBS]
        if s0 < Scalar[DT](0.0) or s0 > Scalar[DT](63.0):
            ok = False
    assert_true(ok, "mb_s filled from buffer")
    assert_true(not st.has_per, "uniform leaves has_per false")
    print("  ok")


def test_cpu_per() raises:
    print("CPUPrioritizedReplay -> storage TrainerState ...")
    var buf = CPUPrioritizedReplay[OBS, ACT, CAP].make(batch_capacity=BATCH)
    buf.configure_per(alpha=0.6, beta=0.4, epsilon=1e-6)
    for k in range(64):
        var s = _mk_obs(Scalar[DT](k))
        var a = _mk_act(Scalar[DT](k))
        var sp = _mk_obs(Scalar[DT](k) + 100.0)
        buf.add(s, a, Scalar[DT](k) * 0.5, sp, Scalar[DT](0.0))
    var st = TrainerState[OBS, ACT, BATCH].make["cpu"]()
    buf.sample_into[BATCH](st)
    assert_true(st.has_per, "PER flips has_per")
    var w_ok = True
    for b in range(BATCH):
        var w = st.mb_w.data[b]
        if w <= Scalar[DT](0.0) or w > Scalar[DT](1.0001):
            w_ok = False
    assert_true(w_ok, "mb_w normalized IS weights in (0,1]")
    # feed back synthetic TD residuals -> update_priorities must not raise
    for b in range(BATCH):
        st.td_residuals.data[b] = Scalar[DT](b) * 0.1 + 0.05
    buf.update_priorities[BATCH](st)
    print("  ok")


def test_gpu_uniform() raises:
    print("GPUReplay -> storage TrainerState (device gather) ...")
    var c = DeviceContext()
    var buf = GPUReplay[OBS, ACT, CAP].make(ctx=Optional(c), batch_capacity=BATCH)
    for k in range(64):
        var s = _mk_obs(Scalar[DT](k))
        var a = _mk_act(Scalar[DT](k))
        var sp = _mk_obs(Scalar[DT](k) + 100.0)
        buf.add(s, a, Scalar[DT](k) * 0.5, sp, Scalar[DT](0.0), ctx=Optional(c))
    var st = TrainerState[OBS, ACT, BATCH].make["gpu"](Optional(c))
    buf.sample_into[BATCH](st)
    st.mb_s.download(c)
    var ok = True
    for b in range(BATCH):
        var s0 = st.mb_s.data[b * OBS]
        if s0 < Scalar[DT](0.0) or s0 > Scalar[DT](63.0):
            ok = False
    assert_true(ok, "gpu mb_s gathered from device buffer")
    print("  ok")


def main() raises:
    print("=" * 60)
    print("L1 storage replay -> TrainerState data bus gate")
    print("=" * 60)
    test_cpu_uniform()
    test_cpu_per()
    test_gpu_uniform()
    print("ALL PASSED")
