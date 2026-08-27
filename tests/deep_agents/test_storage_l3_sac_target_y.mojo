"""L3 gate — SAC TargetYBlock on storage ComputeGraph + ExternalRef (CPU + GPU).

Runs the target-y graph (online actor + 2 target critics threaded as externals →
min_q; log_prob via node_output) + sac_target_y, then verifies the written mb_y
equals  r + γ·(1−d)·(min_q − α·logp)  recomputed on host from the graph's OWN
min_q (node 7) and logp (node 3) — so the stochastic rsample draw doesn't matter;
this checks the graph→sac_target_y wiring + ExternalRef dispatch end-to-end.

Run:
  pixi run mojo run -I . tests/deep_agents/test_storage_l3_sac_target_y.mojo
  pixi run -e apple mojo run -I . tests/deep_agents/test_storage_l3_sac_target_y.mojo
"""

from std.math import isnan, isinf
from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.linear_relu import LinearReLU
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.core.initializer import Xavier
from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents.training.trainer_block import TrainerState
from mojo_rl.deep_agents.sac.target_y_block import TargetYBlock


comptime OBS = 3
comptime ACT = 1
comptime SA = OBS + ACT
comptime H = 32
comptime BATCH = 16
comptime GAMMA = Scalar[DT](0.99)
comptime ALPHA = Scalar[DT](0.2)
comptime ACTOR = StochasticActor[OBS, ACT, LinearReLU[OBS, H], LinearReLU[H, H]]
comptime CRITIC = Sequential[LinearReLU[SA, H], Linear[H, 1]]
comptime BLK = TargetYBlock[ACTOR, CRITIC, BATCH, OBS, ACT]


def _check(mut blk: BLK, mut st: TrainerState[OBS, ACT, BATCH]) raises -> Bool:
    # graph outputs (host-readable copies already on CPU path; caller downloads
    # on GPU before calling).
    var ok = True
    for b in range(BATCH):
        var mq = blk.graph.node_output["min_q"]().data[b]
        var lp = blk.graph.node_output["logp"]().data[b]
        var soft = mq - ALPHA * lp
        var y_ref = st.mb_r.data[b] + GAMMA * (Scalar[DT](1.0) - st.mb_d.data[b]) * soft
        if isnan(st.mb_y.data[b]) or isinf(st.mb_y.data[b]):
            ok = False
        if abs(st.mb_y.data[b] - y_ref) > 1e-4:
            ok = False
    return ok


def test_cpu() raises:
    print("SAC TargetYBlock CPU ...")
    var blk = BLK.make["cpu"](action_scale=1.0, gamma=GAMMA)
    var actor = ACTOR.make["cpu", Xavier]()
    var t1 = CRITIC.make["cpu", Xavier]()
    var t2 = CRITIC.make["cpu", Xavier]()
    var st = TrainerState[OBS, ACT, BATCH].make["cpu"]()
    for i in range(BATCH * OBS):
        st.mb_sp.data[i] = Scalar[DT]((i % 7) - 3) * 0.2
    for b in range(BATCH):
        st.mb_r.data[b] = Scalar[DT]((b % 4) - 2) * 0.3
        st.mb_d.data[b] = Scalar[DT](1.0) if (b % 8 == 7) else Scalar[DT](0.0)
    st.alpha = ALPHA
    blk.step["cpu"](st, actor, t1, t2)
    assert_true(_check(blk, st), "mb_y == r + γ(1-d)(min_q - α·logp)")
    print("  ok")


def test_gpu() raises:
    print("SAC TargetYBlock GPU ...")
    var c = DeviceContext()
    var blk = BLK.make["gpu"](action_scale=1.0, gamma=GAMMA, ctx=Optional(c))
    var actor = ACTOR.make["gpu", Xavier](Optional(c))
    var t1 = CRITIC.make["gpu", Xavier](Optional(c))
    var t2 = CRITIC.make["gpu", Xavier](Optional(c))
    var st = TrainerState[OBS, ACT, BATCH].make["gpu"](Optional(c))
    st.mb_sp.data = List[Scalar[DT]](length=BATCH * OBS, fill=Scalar[DT](0))
    st.mb_r.data = List[Scalar[DT]](length=BATCH, fill=Scalar[DT](0))
    st.mb_d.data = List[Scalar[DT]](length=BATCH, fill=Scalar[DT](0))
    st.mb_sp.n = BATCH * OBS; st.mb_r.n = BATCH; st.mb_d.n = BATCH
    for i in range(BATCH * OBS):
        st.mb_sp.data[i] = Scalar[DT]((i % 7) - 3) * 0.2
    for b in range(BATCH):
        st.mb_r.data[b] = Scalar[DT]((b % 4) - 2) * 0.3
        st.mb_d.data[b] = Scalar[DT](1.0) if (b % 8 == 7) else Scalar[DT](0.0)
    st.mb_sp.upload(c); st.mb_r.upload(c); st.mb_d.upload(c)
    st.alpha = ALPHA
    blk.step["gpu"](st, actor, t1, t2)
    # download outputs the host check reads.
    st.mb_y.download(c)
    blk.graph.node_output["min_q"]().download(c)
    blk.graph.node_output["logp"]().download(c)
    assert_true(_check(blk, st), "gpu mb_y == r + γ(1-d)(min_q - α·logp)")
    print("  ok")


def main() raises:
    print("=" * 60)
    print("L3 SAC TargetYBlock (ComputeGraph + ExternalRef) gate")
    print("=" * 60)
    test_cpu()
    test_gpu()
    print("ALL PASSED")
