"""C51Agent facade smoke — compile + short CPU train/eval/save/load."""

from std.random import seed
from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.combinators.sequential import Sequential
from mojo_rl.deep_agents2.c51 import C51Agent
from mojo_rl.deep_agents2.training.blocks import UniformSampleCpuStep
from mojo_rl.envs.cartpole import CartPoleEnv


comptime OBS_DIM = 4
comptime NUM_ACTIONS = 2
comptime N_ATOMS = 51
comptime HIDDEN = 64
comptime BATCH = 32
comptime CAP = 10_000

comptime C51QNet = Sequential[
    Linear[OBS_DIM, HIDDEN],
    ReLU[HIDDEN],
    Linear[HIDDEN, NUM_ACTIONS * N_ATOMS],
]


def main() raises:
    seed(42)
    var agent = C51Agent[
        "cpu",
        UniformSampleCpuStep[OBS_DIM, 1, BATCH, CAP],
        C51QNet,
        N_ATOMS=N_ATOMS,
        NUM_ACTIONS=NUM_ACTIONS,
    ](
        lr=Scalar[DT](1e-4),
        learning_starts=200,
        target_update_freq=100,
        v_min=Scalar[DT](0.0),
        v_max=Scalar[DT](100.0),
    )
    var env = CartPoleEnv[DT]()
    _ = agent.train(env, total_timesteps=600, print_every=0, verbose=False)

    var eval_env = CartPoleEnv[DT]()
    var ret = agent.eval(eval_env, num_episodes=2, max_steps_per_episode=200)
    print("eval_mean=", ret)

    var m = agent.flush_metrics()
    print("loss=", m.loss.v, " train_steps=", m.train_steps.v)

    agent.save("/tmp/c51_agent_smoke.ckpt")
    agent.load("/tmp/c51_agent_smoke.ckpt")

    var obs = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0.0))
    var a = agent.select_greedy_action(obs)
    assert_true(a >= 0 and a < NUM_ACTIONS, "action index out of range")
    print("PASSED")
