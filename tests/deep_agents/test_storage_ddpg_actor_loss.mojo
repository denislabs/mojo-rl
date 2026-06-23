"""DDPGActorLoss storage gate — loss parity vs an independent oracle (CPU+GPU).

`forward_backward` must return exactly `-mean_b critic(s_b, actor(s_b))` computed
with the CURRENT params (the optimizer step happens AFTER the loss is read, so an
oracle forward with the same actor/critic before the call must match). Also
exercises the GPU device-accumulator path: the on-device `read_loss_accum` window
mean must equal the CPU loss bit-closely.

Run:
  pixi run mojo run -I . tests/deep_agents/test_storage_ddpg_actor_loss.mojo
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Deterministic
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.activations import Tanh
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.deep_agents.ddpg.actor_loss import DDPGActorLoss


comptime OBS = 3
comptime ACT = 2
comptime SA = OBS + ACT
comptime B = 4
comptime ACTOR = Sequential[Linear[OBS, ACT], Tanh[ACT]]
comptime CRITIC = Linear[SA, 1]


def _fill_obs(mut t: Tensor) raises:
    t.ensure(B * OBS)
    for i in range(B * OBS):
        t.data[i] = Scalar[DT](((i * 5 + 1) % 11) - 5) * 0.2


def _oracle_loss[
    target: StaticString
](
    mut actor: ACTOR, mut critic: CRITIC, mut s: Tensor,
    ctx: Optional[DeviceContext],
) raises -> Scalar[DT]:
    """-mean_b critic(s_b, actor(s_b)) with the current params, CPU host math.
    (GPU runs the same forward then D2Hs q for the host mean — oracle only.)"""
    var a = Tensor.alloc(B * ACT)
    actor.forward[target, B](TensorRefs[ACTOR.ARITY](s), a, ctx)
    var sa = Tensor.alloc(B * SA)
    comptime if target == "gpu":
        a.download(ctx.value())
        s.download(ctx.value())
    for b in range(B):
        for d in range(OBS):
            sa.data[b * SA + d] = s.data[b * OBS + d]
        for j in range(ACT):
            sa.data[b * SA + OBS + j] = a.data[b * ACT + j]
    var q = Tensor.alloc(B)
    comptime if target == "gpu":
        sa.upload(ctx.value())
        var qd = Tensor.alloc_gpu(ctx.value(), B)
        critic.forward[target, B](TensorRefs[CRITIC.ARITY](sa), qd, ctx)
        qd.download(ctx.value())
        for b in range(B):
            q.data[b] = qd.data[b]
    else:
        critic.forward[target, B](TensorRefs[CRITIC.ARITY](sa), q, ctx)
    var s_sum: Scalar[DT] = 0.0
    for b in range(B):
        s_sum += q.data[b]
    return -s_sum / Scalar[DT](B)


def _check[target: StaticString](ctx: Optional[DeviceContext]) raises -> Bool:
    comptime TOL = Scalar[DT](1e-5)
    var actor = ACTOR.make[target, Deterministic](ctx)
    var critic = CRITIC.make[target, Deterministic](ctx)
    var actor_opt = Adam(lr=Scalar[DT](1e-3))
    comptime if target == "gpu":
        actor_opt.adopt[target, ACTOR](actor, ctx)
    var blk = DDPGActorLoss[ACTOR, CRITIC, B].make[target](ctx)

    var s = Tensor()
    _fill_obs(s)
    comptime if target == "gpu":
        s.upload(ctx.value())

    var expected = _oracle_loss[target](actor, critic, s, ctx)
    var got = blk.forward_backward[target](actor, actor_opt, critic, s, ctx)

    comptime if target == "cpu":
        var ok = abs(got - expected) < TOL
        print("  CPU loss: got =", got, " oracle =", expected, "=> ",
              "OK" if ok else "FAIL")
        return ok
    else:
        # GPU: forward_backward returns 0 sentinel; the real metric is the
        # device accumulator window mean.
        var acc = blk.read_loss_accum(ctx.value())
        var ok = abs(acc - expected) < Scalar[DT](2e-4)
        print("  GPU loss: acc =", acc, " oracle =", expected, "=> ",
              "OK" if ok else "FAIL")
        return ok


def main() raises:
    print("=" * 60)
    print("DDPGActorLoss storage gate — loss parity vs oracle (CPU+GPU)")
    print("=" * 60)
    var cpu_ok = _check["cpu"](None)
    with DeviceContext() as ctx:
        var gpu_ok = _check["gpu"](Optional(ctx))
        assert_true(cpu_ok and gpu_ok, "DDPGActorLoss loss parity")
    print("DDPG ACTOR LOSS OK")
