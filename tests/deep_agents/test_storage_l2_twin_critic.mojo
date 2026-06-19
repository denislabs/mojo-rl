"""L2 gate — storage TwinCriticStep fits a fixed target (CPU + GPU).

Exercises the migrated nn-coupled critic path end-to-end: TwinCriticStep ->
TwinCriticUpdateBlock -> (concat_sa + Sequential critic forward + storage MSELoss
+ critic.vjp + Adam.step) over the storage TrainerState data bus. The critic is a
plain storage Sequential taking the pre-concatenated [B, OBS+ACT] input (the SAC
config shape). Both critics' losses must fall as they fit the fixed mb_y.

Run:
  pixi run mojo run -I . tests/deep_agents/test_storage_l2_twin_critic.mojo
  pixi run -e apple mojo run -I . tests/deep_agents/test_storage_l2_twin_critic.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.linear_relu import LinearReLU
from mojo_rl.nn.storage.combinators.sequential import Sequential
from mojo_rl.nn.storage.core.initializer import Xavier
from mojo_rl.nn.storage.optimizer.adam import Adam
from mojo_rl.deep_agents.training.trainer_block import TrainerState
from mojo_rl.deep_agents.training.blocks.twin_critic_step import TwinCriticStep


comptime OBS = 3
comptime ACT = 1
comptime SA = OBS + ACT
comptime H = 32
comptime BATCH = 16
comptime CRITIC = Sequential[LinearReLU[SA, H], Linear[H, 1]]


def _fill_inputs(mut st: TrainerState[OBS, ACT, BATCH]):
    for i in range(BATCH * OBS):
        st.mb_s.data[i] = Scalar[DT]((i % 7) - 3) * 0.25
    for i in range(BATCH * ACT):
        st.mb_a.data[i] = Scalar[DT]((i % 5) - 2) * 0.3
    for b in range(BATCH):
        st.mb_y.data[b] = Scalar[DT]((b % 4) - 2) * 0.5


def test_cpu() raises:
    print("TwinCriticStep CPU fit ...")
    var twin = TwinCriticStep[OBS, ACT, BATCH, CRITIC].make["cpu"]()
    var c1 = CRITIC.make["cpu", Xavier]()
    var c2 = CRITIC.make["cpu", Xavier]()
    var o1 = Adam(lr=1e-2)
    var o2 = Adam(lr=1e-2)
    var st = TrainerState[OBS, ACT, BATCH].make["cpu"]()
    _fill_inputs(st)

    var first: Scalar[DT] = 0
    var last: Scalar[DT] = 0
    for step in range(150):
        twin.step["cpu"](st, c1, o1, c2, o2)
        if step == 0:
            first = st.critic_loss
        last = st.critic_loss
    print("  critic_loss", first, "->", last)
    assert_true(last < first * 0.2, "twin critic CPU fits target")
    print("  ok")


def test_gpu() raises:
    print("TwinCriticStep GPU fit ...")
    var c = DeviceContext()
    var twin = TwinCriticStep[OBS, ACT, BATCH, CRITIC].make["gpu"](Optional(c))
    var c1 = CRITIC.make["gpu", Xavier](Optional(c))
    var c2 = CRITIC.make["gpu", Xavier](Optional(c))
    var o1 = Adam(lr=1e-2)
    var o2 = Adam(lr=1e-2)
    var st = TrainerState[OBS, ACT, BATCH].make["gpu"](Optional(c))
    # populate host data then upload the three input tensors
    st.mb_s.data = List[Scalar[DT]](length=BATCH * OBS, fill=Scalar[DT](0))
    st.mb_a.data = List[Scalar[DT]](length=BATCH * ACT, fill=Scalar[DT](0))
    st.mb_y.data = List[Scalar[DT]](length=BATCH, fill=Scalar[DT](0))
    st.mb_s.n = BATCH * OBS; st.mb_a.n = BATCH * ACT; st.mb_y.n = BATCH
    _fill_inputs(st)
    st.mb_s.upload(c); st.mb_a.upload(c); st.mb_y.upload(c)

    var first: Scalar[DT] = 0
    var last: Scalar[DT] = 0
    for step in range(150):
        twin.step["gpu"](st, c1, o1, c2, o2)
        if step == 0:
            first = st.critic_loss
        last = st.critic_loss
    print("  critic_loss", first, "->", last)
    assert_true(last < first * 0.3, "twin critic GPU fits target")
    print("  ok")


def main() raises:
    print("=" * 60)
    print("L2 storage TwinCriticStep gate")
    print("=" * 60)
    test_cpu()
    test_gpu()
    print("ALL PASSED")
