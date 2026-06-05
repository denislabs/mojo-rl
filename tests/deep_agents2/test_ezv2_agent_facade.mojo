"""EZv2DiscreteAgent facade — train + greedy eval + checkpoint round-trip (CPU).

Exercises the full agent surface: short self-play train (loss finite), greedy
eval (in CartPole's [1, 500] return band), and save → fresh-agent load → save
byte-identity (CartPole reset is stochastic, so compare serialized bytes, not
rollout returns — same gotcha as the MuZero facade test).

Run (no GPU):
    pixi run mojo run -I . tests/deep_agents2/test_ezv2_agent_facade.mojo
"""

from std.testing import assert_true

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.efficient_zero_v2.config import EZV2DiscreteMLPConfig
from mojo_rl.deep_agents2.efficient_zero_v2.agent import EZv2DiscreteAgent
from mojo_rl.envs.cartpole import CartPoleEnv


def _read(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def main() raises:
    comptime Env = CartPoleEnv[DType.float64]
    comptime Cfg = EZV2DiscreteMLPConfig[
        OBS=4, ACT=2, LATENT=16, HIDDEN=32, BINS=51,
        PROJ=32, PROJ_HID=32, BOTTLENECK=16,
    ]
    comptime Agent = EZv2DiscreteAgent[
        Env, Cfg.Rep, Cfg.Dyn, Cfg.Pred, Cfg.Proj, Cfg.Predh,
        Cfg.OBS, Cfg.ACT, Cfg.LATENT, Cfg.BINS,
        NUM_SIMS=12, MAX_NODES=48, CAP=20000, B=8, K=3, N=3,
    ]

    var env = Env()
    var agent = Agent(lr=0.01)

    var loss = agent.train(
        env, iterations=400, learning_starts=200, seed=7, verbose=False
    )
    print("train loss:", loss)
    assert_true(loss == loss and loss < 1e30 and loss > -1e30,
        "train loss not finite")

    var g = agent.eval_greedy(env, episodes=3)
    print("greedy eval:", g)
    assert_true(g >= 1.0 and g <= 500.0, "greedy eval out of CartPole band")

    # checkpoint byte-identity: save → fresh load → save.
    var path = String("/tmp/ezv2_agent_ckpt.txt")
    agent.save(path)
    var bytes1 = _read(path)

    var agent2 = Agent(lr=0.01)
    agent2.load(path)
    var path2 = String("/tmp/ezv2_agent_ckpt2.txt")
    agent2.save(path2)
    var bytes2 = _read(path2)

    print("ckpt bytes:", bytes1.byte_length(), "vs", bytes2.byte_length())
    assert_true(bytes1 == bytes2, "checkpoint round-trip not byte-identical")
    print("EZv2DiscreteAgent facade: OK")
