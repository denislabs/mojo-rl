"""Gumbel MuZero CartPole convergence run (v2, GPU) — fully on-device.

The Gumbel-planner sibling of `muzero_cartpole_v2_gpu`: same nets, same train
step, same fix stack (±20 h-space support, value_coef 1.0, temperature
schedule, reanalyze, truncation-aware replay), but the search is
`GumbelGPUMCTS` — Gumbel-Top-k root action sampling + sequential halving
(`MAX_K=2` root candidates for CartPole's 2 actions). The stored policy target
is the planner's **improved policy** rather than raw visit counts; greedy eval
is its argmax. Gumbel MCTS gives policy improvement guarantees at low
simulation counts, so it is the interesting variant for sim-budget-constrained
runs — compare against the vanilla example at the same NUM_SIMS.

Driven through the `MuZeroAgent` facade: on the ``"gpu"`` target its `train`
wires exactly this fully on-device Gumbel driver (`run_muzero_gumbel_selfplay_gpu`),
so the run is identical to the hand-rolled driver call while reusing the
agent's optimizer setup, eval, and checkpoint surface. ``max_grad_norm=10.0``
reproduces the "clip 10" the convergence stack needs.

Run (GPU env required):
    pixi run -e apple mojo run -I . examples/cartpole/muzero_cartpole_v2_gpu_gumbel.mojo
"""

from std.memory import UnsafePointer
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents2.muzero import MuZeroMLPConfig, MuZeroAgent
from mojo_rl.envs.cartpole import CartPoleEnv


def main() raises:
    comptime Env = CartPoleEnv[DType.float64]
    comptime Cfg = MuZeroMLPConfig[OBS=4, ACT=2, LATENT=128, HIDDEN=128, BINS=51]
    comptime NUM_SIMS = 24
    comptime MAX_K = 2       # Gumbel root candidates (= ACT for CartPole)
    comptime Agent = MuZeroAgent[
        "gpu", Env,
        Cfg.Rep, Cfg.Dyn, Cfg.Pred,
        Cfg.OBS, Cfg.ACT, Cfg.LATENT, Cfg.BINS,
        NUM_SIMS=NUM_SIMS, MAX_NODES=128, CAP=50000, B=128, K=5, N=10,
        MAX_K=MAX_K,
    ]

    var ctx = DeviceContext()
    var env = Env()
    var agent = Agent(
        ctx=ctx,
        lr=Scalar[DT](3e-4),
        gamma=Scalar[DT](0.997),
        v_min=Scalar[DT](-20.0),
        v_max=Scalar[DT](20.0),
        value_coef=Scalar[DT](1.0),
        max_grad_norm=Scalar[DT](10.0),
    )

    # ── metrics logger (silent no-op without RL_MONITOR_URL in env/.env) ──
    var env_vars = load_dotenv()
    var logger = RemoteLogger(
        server_url=env_vars.get("RL_MONITOR_URL", ""),
        run_name="Gumbel MuZero CartPole (GPU)",
        buffer_size=64,
        api_key=env_vars.get("RL_MONITOR_API_KEY", ""),
    )
    logger.set_config("agent", "GumbelMuZero")
    logger.set_config("env", "CartPole")
    logger.set_config("framework", "deep_agents2/nn2")

    print("Gumbel MuZero CartPole convergence (v2, GPU — fully on-device)")
    print("  LATENT", Cfg.LATENT, "H", Cfg.HIDDEN, "BINS", Cfg.BINS,
          "sims", NUM_SIMS, "MAX_K", MAX_K, "K", 5, "N", 10, "B", 128,
          "lr 3e-4 clip 10")

    var loss = agent.train[L=RemoteLogger](
        env,
        iterations=60000,
        learning_starts=500,
        train_per_iter=1,
        temperature_decay_steps=60000,
        reanalyze_every=1,
        eval_every=2000,
        eval_episodes=5,
        diag_every=200,
        report_every=500,
        logger=UnsafePointer(to=logger),
        seed=42,
        verbose=True,
    )
    logger.close()

    print("final loss:", loss)
