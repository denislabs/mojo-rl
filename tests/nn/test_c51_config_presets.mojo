"""C51/Rainbow config-preset prototype — Design F smoke (CPU).

Asserts the capitalized presets build the primitive `C51Agent` with the
right type wiring (net out-dim, DOUBLE flag) AND tuned scalar defaults
flowing from the config, then runs a short CPU train/eval. The same
`C51["cpu", …]` / `Rainbow["cpu", …]` calls work with `"gpu"` — see
test_c51_config_presets_gpu.mojo.
"""

from std.random import seed
from std.testing import assert_equal, assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.c51 import (
    C51,
    DoubleC51,
    Rainbow,
    C51Config,
    RainbowConfig,
)
from mojo_rl.envs.cartpole import CartPoleEnv


comptime OBS = 4
comptime ACT = 2
comptime NA = 51
comptime HIDDEN = 64
comptime BATCH = 32
comptime CAP = 10_000


def test_preset_defaults_are_comptime() raises:
    assert_true(not C51Config["cpu", OBS, ACT, BATCH, CAP, NA, HIDDEN].DOUBLE)
    assert_true(RainbowConfig["cpu", OBS, ACT, BATCH, CAP, NA, HIDDEN, 3].DOUBLE)
    assert_equal(
        Int(RainbowConfig["cpu", OBS, ACT, BATCH, CAP, NA, HIDDEN, 3].DEF_NSTEP), 3
    )
    assert_equal(
        Float64(RainbowConfig["cpu", OBS, ACT, BATCH, CAP, NA, HIDDEN, 3].DEF_EPS), 0.0
    )
    assert_equal(
        Float64(C51Config["cpu", OBS, ACT, BATCH, CAP, NA, HIDDEN].DEF_EPS), 1.0
    )
    assert_equal(
        C51Config["cpu", OBS, ACT, BATCH, CAP, NA, HIDDEN].Q_NET.OUT_DIM, ACT * NA
    )
    print("  comptime config members OK")


def test_c51_cpu_builds_and_runs() raises:
    seed(42)
    var agent = C51["cpu", OBS, ACT, BATCH, CAP, NA, HIDDEN](
        v_min=Scalar[DT](0.0), v_max=Scalar[DT](100.0)
    )
    var env = CartPoleEnv[DT]()
    _ = agent.train(env, total_timesteps=400, print_every=0, verbose=False)
    var eval_env = CartPoleEnv[DT]()
    var ret = agent.eval(eval_env, num_episodes=2, max_steps_per_episode=200)
    print("  C51[cpu] eval_mean=", ret)


def test_double_c51_cpu_builds() raises:
    seed(42)
    var agent = DoubleC51["cpu", OBS, ACT, BATCH, CAP, NA, HIDDEN](
        lr=Scalar[DT](2.5e-4), v_min=Scalar[DT](0.0), v_max=Scalar[DT](100.0)
    )
    var env = CartPoleEnv[DT]()
    _ = agent.train(env, total_timesteps=300, print_every=0, verbose=False)
    print("  DoubleC51[cpu] built + ran")


def test_rainbow_cpu_builds_and_runs() raises:
    seed(42)
    var agent = Rainbow["cpu", OBS, ACT, BATCH, CAP, NA, HIDDEN, 3](
        v_min=Scalar[DT](0.0), v_max=Scalar[DT](100.0)
    )
    var env = CartPoleEnv[DT]()
    _ = agent.train(env, total_timesteps=500, print_every=0, verbose=False)
    var eval_env = CartPoleEnv[DT]()
    var ret = agent.eval(eval_env, num_episodes=2, max_steps_per_episode=200)
    print("  Rainbow[cpu] eval_mean=", ret)
    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var a = agent.select_greedy_action(obs)
    assert_true(a >= 0 and a < ACT, "action out of range")


def main() raises:
    print("=== C51/Rainbow config-preset prototype (CPU) ===")
    test_preset_defaults_are_comptime()
    test_c51_cpu_builds_and_runs()
    test_double_c51_cpu_builds()
    test_rainbow_cpu_builds_and_runs()
    print("ALL PASSED")
