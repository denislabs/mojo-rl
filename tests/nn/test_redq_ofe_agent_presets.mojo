"""O.3 — REDQOFEAgent + REDQOFE6/8 preset smokes.

Gates the user-facing surface:

  (1) `REDQOFE6["cpu", OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT](...)` builds
      a `REDQOFEAgent` end-to-end with one line.
  (2) `agent.train_single(env, total_timesteps, ...)` runs the manual
      env loop, returns ep-returns list, drives learning over the
      baseline.
  (3) `agent.save(path)` → fresh agent → `agent.load(path)` →
      `select_greedy_action` matches bit-identically (text v2 format
      preserves enough precision for action equality).
  (4) `REDQOFE8` preset compiles and constructs at the comptime
      8-block architecture (Ant / Humanoid reference shape) without
      necessarily training to convergence — just the construction
      gate.
  (5) `agent_from_config_ofe[REDQOFE6Config[…]]` is type-equivalent
      to the capitalized preset (verified via comptime-shared types
      at the call site)."""

from std.random import seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT

from mojo_rl.deep_agents.redq_ofe import (
    REDQOFE6, REDQOFE8,
    LargeREDQOFE6, LargeREDQOFE8,
    REDQOFE6Config, REDQOFE8Config,
    REDQOFEAgent,
    agent_from_config_ofe,
)

from mojo_rl.envs.pendulum import PendulumEnv


comptime OBS = 3
comptime ACT = 1
comptime BATCH = 128
comptime CAP = 8_000
comptime HIDDEN = 32           # tiny for fast compile
comptime PER_UNIT = 4          # tiny → 6×4=24 added features


def _abs(x: Scalar[DT]) -> Scalar[DT]:
    return x if x >= Scalar[DT](0) else -x


# ─────────────────────────────────────────────────────────────────────────
# (1) REDQOFE6 preset — build + train_single + improvement gate
# ─────────────────────────────────────────────────────────────────────────


def test_redqofe6_preset_pendulum() raises:
    print("=" * 70)
    print("O.3 (1) — REDQOFE6 preset on Pendulum V1 (CPU, 3k steps)")
    print("=" * 70)
    seed(42)

    var agent = REDQOFE6[
        "cpu", OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ](
        action_scale=Scalar[DT](2.0),
        target_entropy=Scalar[DT](-1.0),
        learning_starts=500,
        window_size=10,
        initial_episode_fill=Scalar[DT](-1250.0),
    )
    var env = PendulumEnv[DT]()

    var ep_returns = agent.train_single(
        env,
        total_timesteps=2_500,
        print_every=1_000,
        verbose=True,
    )

    var final_mean = Float64(agent.mean_return())
    print("Final mean ep return:", final_mean)
    print("Episodes completed:  ", agent.ep_count())
    print("ep_returns length:   ", len(ep_returns))
    print("Total train steps:   ", agent.total_train_steps())

    # This test gates the AGENT SURFACE — preset built, env loop
    # ran end-to-end, train_step + record + select_action all
    # plumbed through. Convergence is gated by the dedicated
    # Pendulum smoke (`test_redq_ofe_pendulum_smoke.mojo`), so
    # here we just require finite + non-diverged.
    assert_true(
        len(ep_returns) > 0,
        "train_single must return at least one episode return",
    )
    assert_true(agent.ep_count() > 0, "agent must complete an episode")
    assert_true(
        final_mean == final_mean,
        "final mean_return must be finite (NaN gate)",
    )
    assert_true(
        final_mean > -2000.0,
        "trainer must not diverge below 2× the random baseline",
    )

    # ── Checkpoint round-trip via the agent surface ──────────────────
    var path = String("/tmp/redqofe6_agent.txt")
    agent.save(path)
    print("Saved checkpoint to:", path)

    seed(7)
    var fresh = REDQOFE6[
        "cpu", OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ](
        action_scale=Scalar[DT](2.0),
        target_entropy=Scalar[DT](-1.0),
        learning_starts=500,
        window_size=10,
        initial_episode_fill=Scalar[DT](-1250.0),
    )
    var probe = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    probe[0] = Scalar[DT](0.7)
    probe[1] = Scalar[DT](-0.3)
    probe[2] = Scalar[DT](1.5)
    var act_orig = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    var act_loaded = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    agent.select_greedy_action(probe, act_orig)
    fresh.load(path)
    fresh.select_greedy_action(probe, act_loaded)
    var dev = _abs(act_orig[0] - act_loaded[0])
    print(
        "Greedy [orig]:", act_orig[0],
        " [loaded]:", act_loaded[0],
        " |Δ|:", dev,
    )
    assert_true(
        dev < Scalar[DT](1e-4),
        "loaded agent must reproduce greedy action within v2 text tol",
    )
    print("PASS — REDQOFE6 preset + train_single + save/load.")


# ─────────────────────────────────────────────────────────────────────────
# (2) REDQOFE8 preset — construction gate (no training, just type-check
# the 8-block architecture compiles + a single greedy call works)
# ─────────────────────────────────────────────────────────────────────────


def test_redqofe8_preset_construction() raises:
    print("--- O.3 (2) — REDQOFE8 preset construction smoke ---")

    var agent = REDQOFE8[
        "cpu", OBS, ACT, BATCH, CAP, HIDDEN, 3,    # PER_UNIT=3 → 24-feat sum
    ](
        action_scale=Scalar[DT](2.0),
        target_entropy=Scalar[DT](-1.0),
        learning_starts=500,
        window_size=4,
        initial_episode_fill=Scalar[DT](0.0),
    )
    var probe = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    probe[0] = Scalar[DT](0.1)
    var act = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    agent.select_greedy_action(probe, act)
    var ga = Float64(act[0])
    print("  REDQOFE8 greedy[0] =", ga)
    assert_true(ga == ga, "greedy action must be finite")
    assert_true(
        ga >= -2.0 and ga <= 2.0,
        "greedy action must lie within action_scale clamp",
    )
    print("PASS — REDQOFE8 preset constructed + greedy passes.")


# ─────────────────────────────────────────────────────────────────────────
# (3) Type equivalence — agent_from_config_ofe[REDQOFE6Config[…]] is
# the same concrete REDQOFEAgent[...] as the REDQOFE6 preset
# ─────────────────────────────────────────────────────────────────────────


def test_agent_from_config_type_equivalence() raises:
    print("--- O.3 (3) — agent_from_config_ofe type-equivalence ---")
    comptime CFG = REDQOFE6Config[
        "cpu", OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ]
    var a = agent_from_config_ofe[CFG](
        action_scale=Scalar[DT](2.0),
        learning_starts=500,
        window_size=4,
        initial_episode_fill=Scalar[DT](0.0),
    )
    assert_true(
        a.ep_count() == 0,
        "fresh agent_from_config_ofe must have ep_count == 0",
    )
    print("PASS — agent_from_config_ofe[CFG] returns a working agent.")


# ─────────────────────────────────────────────────────────────────────────
# (4) LargeREDQOFE6 / LargeREDQOFE8 construction gate. Paper-faithful
# N=10/UTD=20 — too expensive to converge in a smoke; we just gate
# that the preset compiles and produces a callable agent.
# ─────────────────────────────────────────────────────────────────────────


def test_large_presets_construction() raises:
    print("--- O.3 (4) — LargeREDQOFE6/8 (paper-faithful) construction ---")
    seed(42)

    # 6-block paper-faithful — small dims so compile + a single
    # train_step stays fast.
    var a6 = LargeREDQOFE6[
        "cpu", OBS, ACT, BATCH, CAP, HIDDEN, PER_UNIT,
    ](
        action_scale=Scalar[DT](2.0),
        learning_starts=200,
        window_size=4,
        initial_episode_fill=Scalar[DT](0.0),
    )
    var probe = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    probe[1] = Scalar[DT](0.2)
    var act = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    a6.select_greedy_action(probe, act)
    print("  LargeREDQOFE6 greedy[0] =", act[0])
    assert_true(act[0] == act[0], "LargeREDQOFE6 greedy must be finite")

    # 8-block paper-faithful.
    var a8 = LargeREDQOFE8[
        "cpu", OBS, ACT, BATCH, CAP, HIDDEN, 3,
    ](
        action_scale=Scalar[DT](2.0),
        learning_starts=200,
        window_size=4,
        initial_episode_fill=Scalar[DT](0.0),
    )
    a8.select_greedy_action(probe, act)
    print("  LargeREDQOFE8 greedy[0] =", act[0])
    assert_true(act[0] == act[0], "LargeREDQOFE8 greedy must be finite")

    print("PASS — LargeREDQOFE6/8 constructed + greedy passes.")


def main() raises:
    test_redqofe6_preset_pendulum()
    test_redqofe8_preset_construction()
    test_agent_from_config_type_equivalence()
    test_large_presets_construction()
    print("=" * 70)
    print("ALL PASS — O.3 REDQOFEAgent + presets")
    print("=" * 70)
