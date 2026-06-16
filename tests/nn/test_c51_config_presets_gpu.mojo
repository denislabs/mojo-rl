"""C51/Rainbow config-preset prototype — GPU smoke (Apple/NVIDIA).

Same `C51[...]` / `Rainbow[...]` factory functions as the CPU test, with
`target="gpu"` — proves the target-generic sample block lets one preset
cover both paths with no per-target duplication.
"""

from std.random import seed
from std.testing import assert_true
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.c51 import C51, Rainbow
from mojo_rl.envs.cartpole import CartPoleEnv


comptime OBS = 4
comptime ACT = 2
comptime NA = 51
comptime HIDDEN = 64
comptime BATCH = 32
comptime CAP = 10_000


def test_c51_gpu_builds_and_runs() raises:
    seed(42)
    var ctx = DeviceContext()
    var agent = C51["gpu", OBS, ACT, BATCH, CAP, NA, HIDDEN](
        ctx=ctx, learning_starts=200, v_min=Scalar[DT](0.0),
        v_max=Scalar[DT](100.0),
    )
    var env = CartPoleEnv[DT]()
    _ = agent.train(env, total_timesteps=400, print_every=0, verbose=False)
    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var a = agent.select_greedy_action(obs)
    assert_true(a >= 0 and a < ACT, "C51[gpu] action out of range")
    print("  C51[gpu] built + ran")


def test_rainbow_gpu_builds_and_runs() raises:
    seed(42)
    var ctx = DeviceContext()
    var agent = Rainbow["gpu", OBS, ACT, BATCH, CAP, NA, HIDDEN, 3](
        ctx=ctx, learning_starts=200, v_min=Scalar[DT](0.0),
        v_max=Scalar[DT](100.0),
    )
    var env = CartPoleEnv[DT]()
    _ = agent.train(env, total_timesteps=500, print_every=0, verbose=False)
    var obs = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    var a = agent.select_greedy_action(obs)
    assert_true(a >= 0 and a < ACT, "Rainbow[gpu] action out of range")
    print("  Rainbow[gpu] built + ran")


def main() raises:
    print("=== C51/Rainbow config-preset prototype (GPU) ===")
    test_c51_gpu_builds_and_runs()
    test_rainbow_gpu_builds_and_runs()
    print("ALL PASSED")
