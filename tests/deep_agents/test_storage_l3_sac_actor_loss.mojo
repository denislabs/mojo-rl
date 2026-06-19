"""L3 gate — SAC SACActorLoss on storage ComputeGraph + ExternalRef (CPU + GPU).

forward_backward runs the actor-loss graph (actor + 2 critics as externals →
loss_per_b = α·logp − min_q), backprops the 1/BATCH seed THROUGH the critics into
the actor (the real ExternalRef.vjp path), and steps the actor. Over many steps
with FROZEN critics the windowed-mean actor loss must fall (actor raises min_q
while managing entropy) — the same gate spike_sac_actor uses.

Run:
  pixi run mojo run -I . tests/deep_agents/test_storage_l3_sac_actor_loss.mojo
  pixi run -e apple mojo run -I . tests/deep_agents/test_storage_l3_sac_actor_loss.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.primitives.linear import Linear
from mojo_rl.nn.storage.primitives.linear_relu import LinearReLU
from mojo_rl.nn.storage.combinators.sequential import Sequential
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.initializer import Xavier
from mojo_rl.nn.storage.optimizer.adam import Adam
from mojo_rl.deep_agents.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents.sac.actor_loss import SACActorLoss


comptime OBS = 3
comptime ACT = 1
comptime SA = OBS + ACT
comptime H = 32
comptime BATCH = 16
comptime ALPHA = Scalar[DT](0.2)
comptime ACTOR = StochasticActor[OBS, ACT, LinearReLU[OBS, H], LinearReLU[H, H]]
comptime CRITIC = Sequential[LinearReLU[SA, H], Linear[H, 1]]
comptime BLK = SACActorLoss[ACTOR, CRITIC, BATCH]


def _run[target: StaticString](ctx: Optional[DeviceContext]) raises:
    print("SAC SACActorLoss", target, "...")
    var blk = BLK.make[target](ctx, action_scale=1.0)
    var actor = ACTOR.make[target, Xavier](ctx)
    var c1 = CRITIC.make[target, Xavier](ctx)
    var c2 = CRITIC.make[target, Xavier](ctx)
    var opt = Adam(lr=3e-4)
    var mb_s = Tensor.alloc(BATCH * OBS)
    for i in range(BATCH * OBS):
        mb_s.data[i] = Scalar[DT]((i % 7) - 3) * 0.2
    comptime if target == "gpu":
        mb_s.upload(ctx.value())

    comptime STEPS = 200
    comptime WIN = 25
    var first_sum: Scalar[DT] = 0
    var last_sum: Scalar[DT] = 0
    for step in range(STEPS):
        c1.zero_grad[target](ctx)  # critics frozen — clear actor-loss pollution
        c2.zero_grad[target](ctx)
        var out = blk.forward_backward[target](
            actor, opt, c1, c2, mb_s, ALPHA, ctx
        )
        if step < WIN: first_sum += out.loss
        if step >= STEPS - WIN: last_sum += out.loss
    var first = first_sum / Scalar[DT](WIN)
    var last = last_sum / Scalar[DT](WIN)
    print("  windowed mean actor_loss:", first, "->", last)
    assert_true(last < first, "actor lowers α·logπ − min_q")
    print("  ok")


def main() raises:
    print("=" * 60)
    print("L3 SAC SACActorLoss (ComputeGraph + ExternalRef) gate")
    print("=" * 60)
    _run["cpu"](None)
    var c = DeviceContext()
    _run["gpu"](Optional(c))
    print("ALL PASSED")
