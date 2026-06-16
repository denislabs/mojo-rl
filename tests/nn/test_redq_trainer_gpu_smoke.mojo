"""R.5 GPU smoke for `REDQTrainer`. Mirrors R.3's CPU smoke at the
trainer level — synthetic data + 100 env steps + UTD inner accounting
+ actor/critic/target movement check. Apple Metal: gates the GPU
plumbing of every block + trainer end-to-end."""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.named_params import named_params
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.nn.combinators import Sequential

from mojo_rl.deep_agents.training.blocks import UniformSampleGpuStep
from mojo_rl.deep_agents.redq import REDQTrainer, REDQ_TARGET_MIN


comptime OBS = 3
comptime ACT = 1
comptime BATCH = 16
comptime CAP = 512

comptime N = 4
comptime N_MIN = 2
comptime UTD = 2
comptime POLICY_DELAY = 2
comptime Q_MODE = REDQ_TARGET_MIN

comptime ActorNet = Sequential[
    Linear[OBS, 32], ReLU[32], Linear[32, 2 * ACT],
]
comptime CriticNet = Sequential[
    Linear[OBS + ACT, 32], ReLU[32], Linear[32, 1],
]
comptime Sample = UniformSampleGpuStep[OBS, ACT, BATCH, CAP]
comptime Trainer = REDQTrainer[
    "gpu", Sample, ActorNet, CriticNet,
    N, N_MIN, UTD, POLICY_DELAY, Q_MODE,
]


def _gpu_param_sum[M: Module](
    ctx: DeviceContext, mut model: M,
) raises -> Float64:
    var ps = named_params["gpu", M](model)
    var acc: Float64 = 0.0
    for i in range(len(ps)):
        ref p = ps[i]
        var host = ctx.enqueue_create_host_buffer[DT](p.n_elems)
        ctx.synchronize()
        var src_view = DeviceBuffer[DT](
            ctx, p.param_ptr, p.n_elems, owning=False,
        )
        ctx.enqueue_copy(host, src_view)
        ctx.synchronize()
        var hp = host.unsafe_ptr()
        for k in range(p.n_elems):
            var v = Float64(hp[k])
            if v < 0.0:
                v = -v
            acc += v
    return acc


def test_redq_trainer_gpu_smoke() raises:
    print("--- REDQTrainer GPU N=4 M=2 UTD=2 POL_DELAY=2 MIN ---")
    var ctx = DeviceContext()

    comptime WARMUP = 32
    comptime TOTAL_STEPS = 100

    var trainer = Trainer.make(
        ctx=ctx,
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](3e-4),
        alpha_lr=Scalar[DT](3e-4),
        action_scale=Scalar[DT](1.0),
        learning_starts=WARMUP,
        initial_episode_fill=Scalar[DT](0.0),
        window_size=4,
        target_entropy=Scalar[DT](-1.0),
    )

    var actor_before = _gpu_param_sum[ActorNet](ctx, trainer.actor)
    var online_before = List[Float64](length=N, fill=0.0)
    var target_before = List[Float64](length=N, fill=0.0)
    for i in range(N):
        online_before[i] = _gpu_param_sum[CriticNet](
            ctx, trainer.ensemble.pairs[i].online
        )
        target_before[i] = _gpu_param_sum[CriticNet](
            ctx, trainer.ensemble.pairs[i].target_net
        )

    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var action = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    var next_obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))

    var n_pre = 0
    var n_post = 0

    for step in range(TOTAL_STEPS):
        for d in range(OBS):
            obs[d] = Scalar[DT](
                0.3 * Float64(d) - 0.1 + 0.005 * Float64(step % 25)
            )
        trainer.select_action(obs, action, step)
        var rew = Scalar[DT](
            -0.5 + 0.3 * Float64(action[0]) - 0.1 * Float64(step % 7)
        )
        for d in range(OBS):
            next_obs[d] = Scalar[DT](
                0.3 * Float64(d) + 0.005 * Float64((step + 1) % 25)
            )
        var done = Scalar[DT](0.0) if (step + 1) % 25 != 0 else Scalar[DT](1.0)
        trainer.record(obs, action, rew, next_obs, done)
        var did = trainer.train_step(step)
        if step < WARMUP:
            assert_true(not did, "no train during warmup")
            if did:
                n_pre += 1
        else:
            if did:
                n_post += 1
        if done == Scalar[DT](1.0):
            trainer.end_episode()

    print("  pre-warmup did_train: ", n_pre)
    print("  post-warmup did_train:", n_post)
    print("  total_train_steps:    ", trainer.total_train_steps())
    print("  actor updates:        ", trainer._actor_update_count)
    assert_true(n_pre == 0, "no training during warmup")
    assert_true(n_post > 0, "training must run after warmup")
    var expected_inner = n_post * UTD
    assert_true(
        trainer.total_train_steps() == expected_inner,
        "total_train_steps == n_post · UTD",
    )

    var actor_after = _gpu_param_sum[ActorNet](ctx, trainer.actor)
    var actor_d = actor_after - actor_before
    if actor_d < 0.0:
        actor_d = -actor_d
    print("  actor |Δ|sum =", actor_d)
    assert_true(actor_d > 0.0, "actor must change on GPU")
    for i in range(N):
        var on_after = _gpu_param_sum[CriticNet](
            ctx, trainer.ensemble.pairs[i].online
        )
        var tg_after = _gpu_param_sum[CriticNet](
            ctx, trainer.ensemble.pairs[i].target_net
        )
        var on_d = on_after - online_before[i]
        if on_d < 0.0:
            on_d = -on_d
        var tg_d = tg_after - target_before[i]
        if tg_d < 0.0:
            tg_d = -tg_d
        print(
            "  member", i,
            " online |Δ|sum =", on_d,
            " target |Δ|sum =", tg_d,
        )
        assert_true(on_d > 0.0, "online critic must change on GPU")
        assert_true(tg_d > 0.0, "target must move via polyak on GPU")

    var m = trainer.flush_metrics()
    print("  metrics.actor_loss  =", m.actor_loss.to_f64())
    print("  metrics.critic_loss =", m.critic_loss.to_f64())
    print("  metrics.alpha       =", m.alpha.to_f64())
    print("  metrics.train_steps =", m.train_steps.to_f64())
    assert_true(
        m.alpha.to_f64() > 0.0,
        "alpha must remain positive",
    )

    print("PASS — REDQTrainer GPU smoke green.")


def main() raises:
    test_redq_trainer_gpu_smoke()
