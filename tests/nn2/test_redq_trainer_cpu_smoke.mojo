"""Phase R.3 end-to-end CPU smoke for `REDQTrainer`.

Synthesizes a 1D "trivial reward" env: obs = small drifting state,
action ∈ [-1, +1]^ACT, reward = small function of (obs, action),
done every 25 env steps. Not a real env — just a deterministic data
generator that exercises the (select_action → record → train_step)
loop the off-policy driver implements.

Checks:
  (a) Warmup gate: `train_step` returns False before learning_starts.
  (b) Past warmup, every `train_step` returns True AND advances
      `total_train_steps` by exactly UTD.
  (c) Actor params CHANGED, all N online critics CHANGED, all N
      target nets CHANGED (polyak).
  (d) Per-actor-cadence accounting: actor only updates every
      POLICY_DELAY inner steps — verifies the `_actor_update_count`
      matches the expected `inner_steps / POLICY_DELAY` count.
  (e) `flush_metrics` returns a finite REDQMetrics bundle and resets
      per-chunk accumulators.
  (f) `select_greedy_action` returns finite actions in [-scale, scale].
"""

from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.core.named_params import named_params
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators import Sequential

from mojo_rl.deep_agents2.training.blocks import UniformSampleCpuStep
from mojo_rl.deep_agents2.redq import REDQTrainer, REDQ_TARGET_MIN


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
    Linear[OBS, 32],
    ReLU[32],
    Linear[32, 2 * ACT],
]
comptime CriticNet = Sequential[
    Linear[OBS + ACT, 32],
    ReLU[32],
    Linear[32, 1],
]

comptime Sample = UniformSampleCpuStep[OBS, ACT, BATCH, CAP]
comptime Trainer = REDQTrainer[
    "cpu", Sample, ActorNet, CriticNet,
    N, N_MIN, UTD, POLICY_DELAY, Q_MODE,
]


def _snapshot_param_sum[M: Module](mut model: M) raises -> Float64:
    var ps = named_params["cpu", M](model)
    var acc: Float64 = 0.0
    for i in range(len(ps)):
        ref p = ps[i]
        for k in range(p.n_elems):
            var v = Float64(p.param_ptr[k])
            if v < 0.0:
                v = -v
            acc += v
    return acc


def test_redq_trainer_cpu_smoke() raises:
    print(
        "--- REDQTrainer N=4 N_MIN=2 UTD=2 POL_DELAY=2 MODE=MIN CPU smoke ---"
    )

    comptime WARMUP = 64
    comptime TOTAL_STEPS = 200

    var trainer = Trainer.make(
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](3e-4),
        alpha_lr=Scalar[DT](3e-4),
        action_scale=Scalar[DT](1.0),
        learning_starts=WARMUP,
        initial_episode_fill=Scalar[DT](0.0),
        window_size=4,
        target_entropy=Scalar[DT](-1.0),
    )

    # Snapshot params before any training.
    var actor_before = _snapshot_param_sum[ActorNet](trainer.actor)
    var online_before = List[Float64](length=N, fill=0.0)
    var target_before = List[Float64](length=N, fill=0.0)
    for i in range(N):
        online_before[i] = _snapshot_param_sum[CriticNet](
            trainer.ensemble.pairs[i].online
        )
        target_before[i] = _snapshot_param_sum[CriticNet](
            trainer.ensemble.pairs[i].target_net
        )

    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var action = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    var next_obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))

    var n_pre_warmup_did_train = 0
    var n_post_warmup_did_train = 0
    var n_post_warmup_skipped = 0
    var ts_before_outer: Int = 0

    for step in range(TOTAL_STEPS):
        for d in range(OBS):
            obs[d] = Scalar[DT](
                0.3 * Float64(d) - 0.1 + 0.005 * Float64(step % 25)
            )
        trainer.select_action(obs, action, step)
        # Synthetic reward; bounded.
        var rew = Scalar[DT](
            -0.5 + 0.3 * Float64(action[0]) - 0.1 * Float64(step % 7)
        )
        for d in range(OBS):
            next_obs[d] = Scalar[DT](
                0.3 * Float64(d) + 0.005 * Float64((step + 1) % 25)
            )
        var done = Scalar[DT](0.0) if (step + 1) % 25 != 0 else Scalar[DT](1.0)
        trainer.record(obs, action, rew, next_obs, done)

        ts_before_outer = trainer.total_train_steps()
        var did_train = trainer.train_step(step)
        var ts_after_outer = trainer.total_train_steps()
        var delta = ts_after_outer - ts_before_outer

        if step < WARMUP:
            assert_true(
                not did_train,
                "train_step must return False during warmup",
            )
            assert_true(
                delta == 0,
                "no inner updates during warmup",
            )
            if did_train:
                n_pre_warmup_did_train += 1
        else:
            if did_train:
                # After warmup AND buffer >= BATCH: each train_step
                # contributes UTD inner updates.
                assert_true(
                    delta == UTD,
                    "post-warmup train_step must advance total_train_steps by UTD",
                )
                n_post_warmup_did_train += 1
            else:
                n_post_warmup_skipped += 1

        if done == Scalar[DT](1.0):
            trainer.end_episode()

    # (a) (b) — warmup vs post-warmup counts.
    print("  pre-warmup train_step ran:    ", n_pre_warmup_did_train)
    print("  post-warmup train_step ran:   ", n_post_warmup_did_train)
    print("  post-warmup train_step skipped:", n_post_warmup_skipped)
    print("  total_train_steps (cumulative):", trainer.total_train_steps())
    assert_true(n_pre_warmup_did_train == 0, "no training during warmup")
    assert_true(
        n_post_warmup_did_train > 0,
        "training must run after warmup",
    )
    # Buffer is well-warm by step WARMUP=64 (BATCH=16, so 16+ filled by then);
    # every post-warmup train_step should succeed.
    assert_true(
        n_post_warmup_skipped == 0,
        "post-warmup train_step should never skip (buffer is warm)",
    )

    # (d) — actor cadence accounting. Actor fires every POLICY_DELAY
    # inner steps → ~ total_inner_steps / POLICY_DELAY actor updates.
    var total_inner = trainer.total_train_steps()
    var expected_actor_updates = total_inner // POLICY_DELAY
    print(
        "  inner steps =", total_inner,
        " actor updates =", trainer._actor_update_count,
        " expected ~", expected_actor_updates,
    )
    assert_true(
        trainer._actor_update_count == expected_actor_updates,
        "actor updates must equal total_inner_steps // POLICY_DELAY",
    )

    # (c) — every learnable model moved.
    var actor_after = _snapshot_param_sum[ActorNet](trainer.actor)
    var actor_d = actor_after - actor_before
    if actor_d < 0.0:
        actor_d = -actor_d
    print("  actor |Δ|sum =", actor_d)
    assert_true(actor_d > 0.0, "actor must change")
    for i in range(N):
        var on_after = _snapshot_param_sum[CriticNet](
            trainer.ensemble.pairs[i].online
        )
        var tg_after = _snapshot_param_sum[CriticNet](
            trainer.ensemble.pairs[i].target_net
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
        assert_true(on_d > 0.0, "online critic must change")
        assert_true(tg_d > 0.0, "target net must move (polyak ran)")

    # (e) — flush_metrics + reset.
    var metrics = trainer.flush_metrics()
    print("  metrics.actor_loss   =", metrics.actor_loss.to_f64())
    print("  metrics.critic_loss  =", metrics.critic_loss.to_f64())
    print("  metrics.alpha        =", metrics.alpha.to_f64())
    print("  metrics.mean_q       =", metrics.mean_q.to_f64())
    print("  metrics.mean_target  =", metrics.mean_target.to_f64())
    print("  metrics.train_steps  =", metrics.train_steps.to_f64())
    print("  metrics.n_updates    =", metrics.n_updates.to_f64())
    assert_true(
        metrics.actor_loss.to_f64() == metrics.actor_loss.to_f64(),
        "actor_loss finite",
    )
    assert_true(
        metrics.critic_loss.to_f64() == metrics.critic_loss.to_f64(),
        "critic_loss finite",
    )
    assert_true(
        metrics.alpha.to_f64() > 0.0,
        "alpha must be positive (= exp(log_alpha))",
    )
    assert_true(
        Int(metrics.train_steps.to_f64()) == total_inner,
        "train_steps == total_inner",
    )
    # The bundle captures THIS chunk's count; the internal accumulators
    # reset AFTER the bundle is built.
    assert_true(
        Int(metrics.n_updates.to_f64()) == total_inner,
        "bundle.n_updates carries the just-flushed chunk's inner count",
    )
    assert_true(
        trainer._update_count == 0,
        "trainer's _update_count reset after flush",
    )
    assert_true(
        trainer._actor_update_count == 0,
        "trainer's _actor_update_count reset after flush",
    )

    # (f) — greedy eval.
    var greedy_obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    for d in range(OBS):
        greedy_obs[d] = Scalar[DT](0.2 * Float64(d) + 0.1)
    var greedy_act = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    trainer.select_greedy_action(greedy_obs, greedy_act)
    print("  greedy action[0] =", greedy_act[0])
    var ga = Float64(greedy_act[0])
    assert_true(ga == ga, "greedy action finite")
    assert_true(ga >= -1.0 and ga <= 1.0, "greedy action in [-1, 1]")

    print("PASS — REDQTrainer N=4 CPU smoke green.")


def main() raises:
    test_redq_trainer_cpu_smoke()
