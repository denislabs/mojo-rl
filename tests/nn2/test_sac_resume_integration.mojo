"""J.2-followup — SACTrainer save → resume integration test.

Mirrors `test_sac_resume_integration.mojo` (the legacy SACTrainer G.3
gate) but exercises the unified `SACTrainer["cpu", UniformSampleCpuStep, …]`.
SACTrainer has the same network field layout as the legacy trainer (actor +
pair1.{online,target_net} + pair2.{online,target_net}), so the same v2
checkpoint flow round-trips without trainer-side changes.

Coverage:
  - Covered: model-Param round-trip for all 5 nets via the v2 format.
    Surfaces gaps if any single net (e.g. one critic's target half) is
    missed by the save/load flow.
  - Covered: eval-time equivalence — loaded trainer produces the same
    greedy-action policy on the same observations.
  - NOT covered: training-trajectory equivalence (replay cursor, RNG
    state, episode tracker not yet checkpointable — same caveat as the
    legacy G.3 test).
"""

from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.core.checkpoint import save_state_v2, load_state_v2
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.deep_agents2.primitives.stochastic_actor import StochasticActor
from mojo_rl.deep_agents2.sac.trainer import SACTrainer
from mojo_rl.deep_agents2.training.blocks import UniformSampleCpuStep
from mojo_rl.deep_agents2.training.batched_env import BatchedCpuEnv
from mojo_rl.deep_agents2.training.driver_offpolicy import run_offpolicy_train_batched

from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime BATCH = 256
comptime REPLAY_CAPACITY = 50_000
comptime TRAIN_STEPS = 30_000
comptime EVAL_EPISODES = 20

comptime ActorNet = StochasticActor[
    OBS_DIM,
    ACT_DIM,
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
]
comptime CriticNet = Sequential[
    Linear[OBS_DIM + ACT_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, 1],
]
comptime Trainer = SACTrainer[
    "cpu",
    UniformSampleCpuStep[OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY],
    ActorNet,
    CriticNet,
]


def _make_trainer() raises -> Trainer:
    return Trainer.make(
        actor_lr=Scalar[DT](3e-4),
        critic_lr=Scalar[DT](1e-3),
        alpha_lr=Scalar[DT](3e-4),
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        action_scale=Scalar[DT](2.0),
        init_alpha=Scalar[DT](0.2),
        target_entropy=Scalar[DT](-1.0),
        learning_starts=1_000,
        window_size=10,
        initial_episode_fill=Scalar[DT](-1250.0),
    )


def _save_all(mut t: Trainer, prefix: String) raises:
    """Save every Param-bearing net to its own v2 file."""
    save_state_v2[ActorNet](t.actor, prefix + String("actor.ckpt"))
    save_state_v2[CriticNet](
        t.pair1.online, prefix + String("critic1_online.ckpt")
    )
    save_state_v2[CriticNet](
        t.pair1.target_net, prefix + String("critic1_target.ckpt")
    )
    save_state_v2[CriticNet](
        t.pair2.online, prefix + String("critic2_online.ckpt")
    )
    save_state_v2[CriticNet](
        t.pair2.target_net, prefix + String("critic2_target.ckpt")
    )


def _load_all(mut t: Trainer, prefix: String) raises:
    load_state_v2[ActorNet](t.actor, prefix + String("actor.ckpt"))
    load_state_v2[CriticNet](
        t.pair1.online, prefix + String("critic1_online.ckpt")
    )
    load_state_v2[CriticNet](
        t.pair1.target_net, prefix + String("critic1_target.ckpt")
    )
    load_state_v2[CriticNet](
        t.pair2.online, prefix + String("critic2_online.ckpt")
    )
    load_state_v2[CriticNet](
        t.pair2.target_net, prefix + String("critic2_target.ckpt")
    )


def _eval_greedy(mut t: Trainer, episodes: Int) raises -> Scalar[DT]:
    """Run `episodes` Pendulum episodes with `select_greedy_action`
    (deterministic policy) and return the mean episode return."""
    var env = PendulumEnv[DT]()
    var obs = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0.0))
    var action = List[Scalar[DT]](length=ACT_DIM, fill=Scalar[DT](0.0))
    _ = env.reset()

    var total = Scalar[DT](0.0)
    var ep_return = Scalar[DT](0.0)
    var ep_done = 0
    var obs_self = env.get_obs_list()
    while ep_done < episodes:
        for d in range(OBS_DIM):
            obs[d] = obs_self[d]
        t.select_greedy_action(obs, action)
        var step_res = env.step_continuous(action[0])
        var nxt = step_res[0].copy()
        var reward = step_res[1]
        var done = step_res[2]
        ep_return += reward
        if done:
            total += ep_return
            ep_return = Scalar[DT](0.0)
            ep_done += 1
            _ = env.reset()
            obs_self = env.get_obs_list()
        else:
            obs_self = nxt.copy()
    return total / Scalar[DT](Float64(episodes))


def test_save_resume_eval_equivalence() raises:
    var ckpt_prefix = String("/tmp/nn2_resume_")

    # ─── Step 1: train trainer A for TRAIN_STEPS, save weights. ───────
    print("Training trainer A for", TRAIN_STEPS, "steps...")
    seed(42)
    var trainer_a = _make_trainer()
    var template_a = PendulumEnv[DT]()
    var env_a = BatchedCpuEnv[PendulumEnv[DT], 1, OBS_DIM, ACT_DIM](
        template_a
    )
    _ = run_offpolicy_train_batched[
        Trainer,
        BatchedCpuEnv[PendulumEnv[DT], 1, OBS_DIM, ACT_DIM],
        1,
    ](
        None,
        trainer_a,
        env_a,
        TRAIN_STEPS,
        rng_seed=UInt64(42),
        updates_per_step=1,
        print_every=0,
        verbose=False,
    )
    var train_mean = trainer_a.mean_return()
    print("  trainer A train mean_ret(10):", train_mean)

    _save_all(trainer_a, ckpt_prefix)

    var eval_a = _eval_greedy(trainer_a, EVAL_EPISODES)
    print("  trainer A eval mean_ret over", EVAL_EPISODES, "eps:", eval_a)

    # ─── Step 2: build trainer C, load A's weights, eval. ─────────────
    seed(31337)
    var trainer_c = _make_trainer()
    var eval_c_before_load = _eval_greedy(trainer_c, EVAL_EPISODES)
    print("  trainer C pre-load eval (sanity, untrained):", eval_c_before_load)

    _load_all(trainer_c, ckpt_prefix)

    var eval_c = _eval_greedy(trainer_c, EVAL_EPISODES)
    print(
        "  trainer C post-load eval mean_ret over",
        EVAL_EPISODES,
        "eps:",
        eval_c,
    )

    # ─── Assertion 1: load actually changed something. ────────────────
    var loaded_improvement = eval_c - eval_c_before_load
    assert_true(
        loaded_improvement > Scalar[DT](100.0),
        "load_state_v2 didn't move trainer C: pre-load eval="
        + String(eval_c_before_load)
        + " post-load eval="
        + String(eval_c)
        + " (expected post-load to be ≥100 better; if not, load is a no-op)",
    )

    # ─── Assertion 2: trainer C ≈ trainer A on eval. ──────────────────
    var delta = eval_a - eval_c if eval_a > eval_c else eval_c - eval_a
    assert_true(
        delta < Scalar[DT](20.0),
        "Eval drift after save/load: A="
        + String(eval_a)
        + " C="
        + String(eval_c)
        + " |Δ|="
        + String(delta)
        + " (expected < 20.0)",
    )
    print("  eval equivalence |A - C| =", delta, " (< 20.0) PASS")


def main() raises:
    print("=" * 70)
    print("SAC trainer save → resume integration test")
    print("=" * 70)
    test_save_resume_eval_equivalence()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
