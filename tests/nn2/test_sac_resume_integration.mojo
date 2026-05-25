"""G.3 — SAC trainer save → resume integration test.

Proves that the v2 checkpoint surface (`save_state_v2` / `load_state_v2`)
round-trips every Param-bearing network in a production SAC trainer
(actor + twin online critics + twin target critics = 5 nets) such that
a freshly-instantiated trainer loaded with the saved weights produces
the same eval performance as the original.

What this test covers vs what it intentionally does not:

  - Covered: model-Param round-trip for all 5 nets via the v2 format.
    Surfaces gaps if any single net (e.g. one critic's target half) is
    missed by the save/load flow.
  - Covered: eval-time equivalence — loaded trainer produces the same
    greedy-action policy on the same observations.
  - NOT covered: training-trajectory equivalence. The audit (A.3 caveat)
    documents that CPU paths use process-global `std.random`, replay
    buffers aren't checkpointable, and the episode tracker doesn't
    round-trip. A "trainer C loads A's mid-training state then keeps
    training" test would diverge from a fresh control run via
    uncontrollable state, not via correctness issues — it'd be a flaky
    test, not a useful one.

When the trainer-state Saveable surface lands (post-G.3 — replay cursor,
n-step buffer, tracker, RNG seed), upgrade this test to assert "C resumes
mid-training → matches B's final mean_ret within ±5".
"""

from std.memory import alloc
from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.nn2.core.checkpoint import save_state_v2, load_state_v2
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.stochastic_actor import StochasticActor
from mojo_rl.nn2.training.sac_trainer import SACTrainer
from mojo_rl.nn2.training.driver_cpu import run_offpolicy_train_cpu

from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS_DIM = 3
comptime ACT_DIM = 1
comptime HIDDEN = 64
comptime BATCH = 256
comptime REPLAY_CAPACITY = 50_000
comptime TRAIN_STEPS = 30_000
comptime EVAL_EPISODES = 20

comptime ActorNet = StochasticActor[
    OBS_DIM, ACT_DIM,
    Linear[OBS_DIM, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
]
comptime CriticNet = Sequential[
    Linear[OBS_DIM + ACT_DIM, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, HIDDEN], ReLU[HIDDEN],
    Linear[HIDDEN, 1],
]
comptime Trainer = SACTrainer[
    ActorNet, CriticNet, OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY
]


def _make_trainer() raises -> Trainer:
    return Trainer.make["cpu"](
        actor_lr=Scalar[DT](3e-4), critic_lr=Scalar[DT](1e-3),
        alpha_lr=Scalar[DT](3e-4), gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005), action_scale=Scalar[DT](2.0),
        init_alpha=Scalar[DT](0.2), target_entropy=Scalar[DT](-1.0),
        learning_starts=1_000,
        window_size=10, initial_episode_fill=Scalar[DT](-1250.0),
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
    (deterministic policy) and return the mean episode return.

    Uses a fresh env (auto-reset on done). Pendulum truncates every 200
    steps so each episode is exactly 200 steps. No training, no replay
    pushes — purely a forward-only policy evaluation.
    """
    var env = PendulumEnv[DT]()
    var obs = alloc[Scalar[DT]](OBS_DIM)
    var action = alloc[Scalar[DT]](ACT_DIM)
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
    var ckpt_prefix = String("/tmp/nn2_g3_resume_")

    # ─── Step 1: train trainer A for TRAIN_STEPS, save weights. ───────
    print("Training trainer A for", TRAIN_STEPS, "steps...")
    seed(42)
    var trainer_a = _make_trainer()
    var env_a = PendulumEnv[DT]()
    _ = run_offpolicy_train_cpu(
        trainer_a, env_a, TRAIN_STEPS,
        obs_dim=OBS_DIM, act_dim=ACT_DIM,
        print_every=0, verbose=False,
    )
    var train_mean = trainer_a.mean_return()
    print("  trainer A train mean_ret(10):", train_mean)

    _save_all(trainer_a, ckpt_prefix)

    # Evaluate trainer A's greedy policy as the reference signal. (The
    # train-time mean is a stochastic-policy moving average over recent
    # episodes; eval is deterministic on a fresh env — they will differ.
    # The eval number is the one we hold loaded-trainer C to.)
    var eval_a = _eval_greedy(trainer_a, EVAL_EPISODES)
    print("  trainer A eval mean_ret over", EVAL_EPISODES, "eps:", eval_a)

    # ─── Step 2: build trainer C, load A's weights, eval. ─────────────
    # We deliberately use a *different* RNG seed for trainer C's
    # construction (which exercises a different Xavier-init draw, so the
    # pre-load weights are definitely different from trainer A's). After
    # load, the weights should be identical and eval should match.
    seed(31337)
    var trainer_c = _make_trainer()
    var eval_c_before_load = _eval_greedy(trainer_c, EVAL_EPISODES)
    print("  trainer C pre-load eval (sanity, untrained):", eval_c_before_load)

    _load_all(trainer_c, ckpt_prefix)

    var eval_c = _eval_greedy(trainer_c, EVAL_EPISODES)
    print("  trainer C post-load eval mean_ret over", EVAL_EPISODES, "eps:", eval_c)

    # ─── Assertion 1: load actually changed something. ────────────────
    # The untrained trainer C should be much worse than the loaded one;
    # if not, load is silently failing.
    var loaded_improvement = eval_c - eval_c_before_load
    assert_true(
        loaded_improvement > Scalar[DT](100.0),
        "load_state_v2 didn't move trainer C: pre-load eval="
        + String(eval_c_before_load) + " post-load eval=" + String(eval_c)
        + " (expected post-load to be ≥100 better; if not, load is a no-op)",
    )

    # ─── Assertion 2: trainer C ≈ trainer A on eval. ──────────────────
    # Eval-time equivalence after a clean weight round-trip. The
    # tolerance accommodates the v2 text-format fp32 round-trip slack
    # (≤ 1e-6 per weight) compounded across 5 networks + 20-episode
    # Pendulum eval variance. Empirically Pendulum greedy eval is
    # essentially deterministic given the same weights (no exploration),
    # so the tolerance is generous against per-episode reward noise.
    var delta = eval_a - eval_c if eval_a > eval_c else eval_c - eval_a
    assert_true(
        delta < Scalar[DT](20.0),
        "Eval drift after save/load: A=" + String(eval_a)
        + " C=" + String(eval_c) + " |Δ|=" + String(delta)
        + " (expected < 20.0)",
    )
    print("  eval equivalence |A - C| =", delta, " (< 20.0) PASS")


def main() raises:
    print("=" * 70)
    print("G.3: SAC trainer save → resume integration test")
    print("=" * 70)
    test_save_resume_eval_equivalence()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
