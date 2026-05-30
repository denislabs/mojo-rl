"""Phase 0 — DQN config-driven preset bit-identity gate.

Asserts the new `DQNConfig` / `DoubleDQNConfig` "Design F" presets produce
training runs bit-identical to constructing `DQNAgent` directly with the
equivalent loose params. This locks the config layer as pure sugar — it
must not perturb numerics.

Mirrors the structure of `c51/config.mojo`'s presets. Run:
    pixi run mojo run -I . tests/nn2/test_dqn_config.mojo
"""

from std.random import seed
from std.testing import assert_equal, assert_true

from mojo_rl.nn2.constants import DT

from mojo_rl.deep_agents2.dqn import (
    DQNAgent,
    DQNConfig,
    DoubleDQNConfig,
    DQNNet,
    DuelingDQNNet,
    NoisyDQNNet,
    agent_from_config,
    DQN,
    DoubleDQN,
    DuelingDQN,
    NoisyDQN,
    DQNPER,
    RainbowDQN,
)
from mojo_rl.deep_agents2.training.blocks import ReplaySampleStep
from mojo_rl.deep_agents2.data.any_replay import AnyReplay

from mojo_rl.envs.cartpole import CartPoleEnv


comptime OBS_DIM = 4
comptime NUM_ACTIONS = 2
comptime HIDDEN = 64
comptime BATCH = 32
comptime CAP = 5_000
comptime WARMUP = 200
comptime TOTAL_STEPS = 1_500


def _run_manual_double() raises -> List[Scalar[DT]]:
    """Reference path: construct DQNAgent directly with DOUBLE=True."""
    seed(42)
    var env = CartPoleEnv[DT]()
    var agent = DQNAgent[
        "cpu",
        ReplaySampleStep[AnyReplay["cpu", OBS_DIM, 1, CAP], BATCH],
        DQNNet[OBS_DIM, NUM_ACTIONS, HIDDEN],
        True,
    ](
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        learning_starts=WARMUP,
    )
    return agent.train(env, total_timesteps=TOTAL_STEPS, verbose=False)


def _run_preset_double() raises -> List[Scalar[DT]]:
    """Config path: build via the capitalized `DoubleDQN` preset."""
    seed(42)
    var env = CartPoleEnv[DT]()
    var agent = DoubleDQN["cpu", OBS_DIM, NUM_ACTIONS, BATCH, CAP, HIDDEN](
        lr=1e-3,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        learning_starts=WARMUP,
    )
    return agent.train(env, total_timesteps=TOTAL_STEPS, verbose=False)


def _run_from_config_plain() raises -> List[Scalar[DT]]:
    """Config path: build via `agent_from_config[DQNConfig]` (DOUBLE=False)."""
    seed(42)
    var env = CartPoleEnv[DT]()
    var agent = agent_from_config[
        DQNConfig["cpu", OBS_DIM, NUM_ACTIONS, BATCH, CAP, HIDDEN]
    ](
        lr=1e-3,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        learning_starts=WARMUP,
    )
    return agent.train(env, total_timesteps=TOTAL_STEPS, verbose=False)


def _run_manual_plain() raises -> List[Scalar[DT]]:
    """Reference path for plain DQN (DOUBLE=False)."""
    seed(42)
    var env = CartPoleEnv[DT]()
    var agent = DQNAgent[
        "cpu",
        ReplaySampleStep[AnyReplay["cpu", OBS_DIM, 1, CAP], BATCH],
        DQNNet[OBS_DIM, NUM_ACTIONS, HIDDEN],
        False,
    ](
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        learning_starts=WARMUP,
    )
    return agent.train(env, total_timesteps=TOTAL_STEPS, verbose=False)


def _run_manual_dueling() raises -> List[Scalar[DT]]:
    """Reference: DQNAgent with DuelingDQNNet (uniform, DOUBLE=False)."""
    seed(42)
    var env = CartPoleEnv[DT]()
    var agent = DQNAgent[
        "cpu",
        ReplaySampleStep[AnyReplay["cpu", OBS_DIM, 1, CAP], BATCH],
        DuelingDQNNet[OBS_DIM, NUM_ACTIONS, HIDDEN],
        False,
    ](
        lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995,
        epsilon_min=0.01, learning_starts=WARMUP,
    )
    return agent.train(env, total_timesteps=TOTAL_STEPS, verbose=False)


def _run_preset_dueling() raises -> List[Scalar[DT]]:
    seed(42)
    var env = CartPoleEnv[DT]()
    var agent = DuelingDQN["cpu", OBS_DIM, NUM_ACTIONS, BATCH, CAP, HIDDEN](
        lr=1e-3, epsilon=1.0, epsilon_decay=0.995,
        epsilon_min=0.01, learning_starts=WARMUP,
    )
    return agent.train(env, total_timesteps=TOTAL_STEPS, verbose=False)


def _run_manual_noisy() raises -> List[Scalar[DT]]:
    """Reference: DQNAgent with NoisyDQNNet (uniform, ε=0, DOUBLE=False)."""
    seed(42)
    var env = CartPoleEnv[DT]()
    var agent = DQNAgent[
        "cpu",
        ReplaySampleStep[AnyReplay["cpu", OBS_DIM, 1, CAP], BATCH],
        NoisyDQNNet[OBS_DIM, NUM_ACTIONS, HIDDEN],
        False,
    ](
        lr=1e-3, gamma=0.99, epsilon=0.0, epsilon_decay=1.0,
        epsilon_min=0.0, learning_starts=WARMUP,
    )
    return agent.train(env, total_timesteps=TOTAL_STEPS, verbose=False)


def _run_preset_noisy() raises -> List[Scalar[DT]]:
    seed(42)
    var env = CartPoleEnv[DT]()
    var agent = NoisyDQN["cpu", OBS_DIM, NUM_ACTIONS, BATCH, CAP, HIDDEN](
        lr=1e-3, learning_starts=WARMUP,
    )
    return agent.train(env, total_timesteps=TOTAL_STEPS, verbose=False)


def _run_preset_per() raises -> List[Scalar[DT]]:
    """PER preset — build + train smoke (manual AnyPerReplay setup is
    heavier; the trainer-level PER path has its own smoke test)."""
    seed(42)
    var env = CartPoleEnv[DT]()
    var agent = DQNPER["cpu", OBS_DIM, NUM_ACTIONS, BATCH, CAP, HIDDEN](
        lr=1e-3, learning_starts=WARMUP,
    )
    return agent.train(env, total_timesteps=TOTAL_STEPS, verbose=False)


def _run_preset_rainbow() raises -> List[Scalar[DT]]:
    """RainbowDQN preset — build + train smoke (Double+Dueling+Noisy+
    PER+N-step over one trainer)."""
    seed(42)
    var env = CartPoleEnv[DT]()
    var agent = RainbowDQN["cpu", OBS_DIM, NUM_ACTIONS, BATCH, CAP, HIDDEN, 3](
        lr=1e-3, learning_starts=WARMUP,
    )
    return agent.train(env, total_timesteps=TOTAL_STEPS, verbose=False)


def _assert_finite_nonempty(rets: List[Scalar[DT]], label: String) raises:
    from std.math import isnan, isinf
    assert_true(len(rets) > 0, label + ": no episodes completed")
    for i in range(len(rets)):
        assert_true(not isnan(rets[i]), label + ": NaN return")
        assert_true(not isinf(rets[i]), label + ": Inf return")


def _assert_identical(
    a: List[Scalar[DT]], b: List[Scalar[DT]], label: String
) raises:
    assert_equal(len(a), len(b), label + ": length mismatch")
    for i in range(len(a)):
        assert_true(
            a[i] == b[i],
            label + ": return[" + String(i) + "] differs "
            + String(a[i]) + " vs " + String(b[i]),
        )


def main() raises:
    print("=== DQN config-driven preset bit-identity gate ===")

    # Double DQN: manual vs DoubleDQN preset.
    var manual_d = _run_manual_double()
    var preset_d = _run_preset_double()
    _assert_identical(manual_d, preset_d, "DoubleDQN preset")
    print(
        "  DoubleDQN preset  == manual DQNAgent[..., True]   OK (",
        len(manual_d), "episodes )",
    )

    # Plain DQN: manual vs agent_from_config[DQNConfig].
    var manual_p = _run_manual_plain()
    var cfg_p = _run_from_config_plain()
    _assert_identical(manual_p, cfg_p, "DQNConfig factory")
    print(
        "  agent_from_config == manual DQNAgent[..., False]  OK (",
        len(manual_p), "episodes )",
    )

    # Dueling: preset == manual.
    var manual_du = _run_manual_dueling()
    var preset_du = _run_preset_dueling()
    _assert_identical(manual_du, preset_du, "DuelingDQN preset")
    print(
        "  DuelingDQN preset == manual DuelingDQNNet          OK (",
        len(manual_du), "episodes )",
    )

    # Noisy: preset == manual.
    var manual_no = _run_manual_noisy()
    var preset_no = _run_preset_noisy()
    _assert_identical(manual_no, preset_no, "NoisyDQN preset")
    print(
        "  NoisyDQN preset   == manual NoisyDQNNet            OK (",
        len(manual_no), "episodes )",
    )

    # PER + Rainbow: build + train smoke (finite, non-empty).
    var per = _run_preset_per()
    _assert_finite_nonempty(per, "DQNPER preset")
    print("  DQNPER preset     builds + trains (finite)        OK (",
          len(per), "episodes )")

    var rainbow = _run_preset_rainbow()
    _assert_finite_nonempty(rainbow, "RainbowDQN preset")
    print("  RainbowDQN preset builds + trains (finite)        OK (",
          len(rainbow), "episodes )")

    print("PASS")
