"""L2 gate — SingleCriticStep + ActionSamplingBlock + LossBlockBundle (storage).

- SingleCriticStep: single Sequential critic fits a fixed target via the
  migrated CriticUpdateBlock (concat_sa + storage MSELoss + Adam). Loss falls.
- ActionSamplingBlock: deterministic-with-noise + warmup produce actions clamped
  to [-scale, scale]; the policy path runs the actor through the storage surface.
- LossBlockBundle: constructs (compile/lifecycle check).

CPU + GPU.

Run:
  pixi run mojo run -I . tests/deep_agents/test_storage_l2_single_critic_action.mojo
  pixi run -e apple mojo run -I . tests/deep_agents/test_storage_l2_single_critic_action.mojo
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.linear_relu import LinearReLU
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.core.initializer import Xavier
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.deep_agents.training.trainer_block import TrainerState
from mojo_rl.deep_agents.training.blocks.single_critic_step import SingleCriticStep
from mojo_rl.deep_agents.training.action_sampling_block import ActionSamplingBlock
from mojo_rl.deep_agents.loss.loss_block_bundle import LossBlockBundle
from mojo_rl.deep_agents.loss.critic_update_block import CriticUpdateBlock


comptime OBS = 3
comptime ACT = 1
comptime SA = OBS + ACT
comptime H = 32
comptime BATCH = 16
comptime CRITIC = Sequential[LinearReLU[SA, H], Linear[H, 1]]
comptime DET_ACTOR = Sequential[LinearReLU[OBS, H], Linear[H, ACT]]  # OUT_DIM==ACT


def _fill(mut st: TrainerState[OBS, ACT, BATCH]):
    for i in range(BATCH * OBS):
        st.mb_s.data[i] = Scalar[DT]((i % 7) - 3) * 0.25
    for i in range(BATCH * ACT):
        st.mb_a.data[i] = Scalar[DT]((i % 5) - 2) * 0.3
    for b in range(BATCH):
        st.mb_y.data[b] = Scalar[DT]((b % 4) - 2) * 0.5


def test_single_critic_cpu() raises:
    print("SingleCriticStep CPU fit ...")
    var blk = SingleCriticStep[OBS, ACT, BATCH, CRITIC].make["cpu"]()
    var c1 = CRITIC.make["cpu", Xavier]()
    var o1 = Adam(lr=1e-2)
    var st = TrainerState[OBS, ACT, BATCH].make["cpu"]()
    _fill(st)
    var first: Scalar[DT] = 0
    var last: Scalar[DT] = 0
    for step in range(150):
        blk.step["cpu"](st, c1, o1)
        if step == 0: first = st.critic_loss
        last = st.critic_loss
    print("  critic_loss", first, "->", last)
    assert_true(last < first * 0.25, "single critic CPU fits")
    print("  ok")


def test_single_critic_gpu() raises:
    print("SingleCriticStep GPU fit ...")
    var c = DeviceContext()
    var blk = SingleCriticStep[OBS, ACT, BATCH, CRITIC].make["gpu"](Optional(c))
    var c1 = CRITIC.make["gpu", Xavier](Optional(c))
    var o1 = Adam(lr=1e-2)
    var st = TrainerState[OBS, ACT, BATCH].make["gpu"](Optional(c))
    st.mb_s.data = List[Scalar[DT]](length=BATCH * OBS, fill=Scalar[DT](0))
    st.mb_a.data = List[Scalar[DT]](length=BATCH * ACT, fill=Scalar[DT](0))
    st.mb_y.data = List[Scalar[DT]](length=BATCH, fill=Scalar[DT](0))
    st.mb_s.n = BATCH * OBS; st.mb_a.n = BATCH * ACT; st.mb_y.n = BATCH
    _fill(st)
    st.mb_s.upload(c); st.mb_a.upload(c); st.mb_y.upload(c)
    var first: Scalar[DT] = 0
    var last: Scalar[DT] = 0
    for step in range(150):
        blk.step["gpu"](st, c1, o1)
        if step == 0: first = st.critic_loss
        last = st.critic_loss
    print("  critic_loss", first, "->", last)
    assert_true(last < first * 0.35, "single critic GPU fits")
    print("  ok")


def _check_clamped(ref a: List[Scalar[DT]], scale: Scalar[DT]) -> Bool:
    for j in range(ACT):
        if a[j] > scale or a[j] < -scale:
            return False
    return True


def test_action_sampling[target: StaticString](ctx: Optional[DeviceContext]) raises:
    print("ActionSamplingBlock", target, "...")
    var blk = ActionSamplingBlock[DET_ACTOR, OBS, ACT, ACT].make[target](ctx)
    var actor = DET_ACTOR.make[target, Xavier](ctx)
    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0))
    for d in range(OBS): obs[d] = Scalar[DT](d) * 0.3 - 0.4
    var act = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0))
    var scale = Scalar[DT](1.0)

    # warmup (step_idx < learning_starts) -> uniform in [-scale, scale]
    blk.select_deterministic_with_noise[target](
        actor, obs, act, 0, 1000, scale, Scalar[DT](0.1)
    )
    assert_true(_check_clamped(act, scale), "warmup action clamped")
    # policy path (past warmup) -> actor + noise, clamped
    blk.select_deterministic_with_noise[target](
        actor, obs, act, 5000, 1000, scale, Scalar[DT](0.1)
    )
    assert_true(_check_clamped(act, scale), "noisy policy action clamped")
    # deterministic (no noise)
    blk.select_deterministic[target](actor, obs, act, 5000, 1000, scale)
    assert_true(_check_clamped(act, scale), "deterministic action clamped")
    print("  ok")


def test_bundle() raises:
    print("LossBlockBundle construct ...")
    var bundle = LossBlockBundle[
        CriticUpdateBlock[CRITIC, BATCH, SA],
        CriticUpdateBlock[CRITIC, BATCH, SA],
    ].make_default["cpu"]()
    _ = bundle
    print("  ok")


def main() raises:
    print("=" * 60)
    print("L2 storage SingleCritic + ActionSampling + LossBundle gate")
    print("=" * 60)
    test_single_critic_cpu()
    test_single_critic_gpu()
    test_action_sampling["cpu"](None)
    var c = DeviceContext()
    test_action_sampling["gpu"](Optional(c))
    test_bundle()
    print("ALL PASSED")
