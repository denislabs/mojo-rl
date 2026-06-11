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

Run (GPU env required):
    pixi run -e apple mojo run -I . examples/cartpole/muzero_cartpole_v2_gpu_gumbel.mojo
"""

from std.memory import UnsafePointer
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents2.muzero.nets import MZRepNet, MZDynNet, MZPredNet
from mojo_rl.deep_agents2.muzero.selfplay_gpu_device import (
    run_muzero_gumbel_selfplay_gpu,
)
from mojo_rl.envs.cartpole import CartPoleEnv


def main() raises:
    comptime OBS = 4
    comptime ACT = 2
    comptime LATENT = 128
    comptime BINS = 51
    comptime H = 128
    comptime NUM_SIMS = 24
    comptime MAX_NODES = 128
    comptime MAX_K = 2       # Gumbel root candidates (= ACT for CartPole)
    comptime CAP = 50000
    comptime B = 128
    comptime K = 5
    comptime N = 10

    comptime Rep = MZRepNet[OBS, LATENT, H]
    comptime Dyn = MZDynNet[LATENT, ACT, BINS, H]
    comptime Pred = MZPredNet[LATENT, ACT, BINS, H]

    var ctx = DeviceContext()
    var env = CartPoleEnv[DType.float64]()
    var rep = Rep.make["gpu", INIT=Kaiming](ctx)
    var dyn = Dyn.make["gpu", INIT=Kaiming](ctx)
    var pred = Pred.make["gpu", INIT=Kaiming](ctx)
    var orep = Adam.make["gpu", M=Rep](rep, ctx)
    var odyn = Adam.make["gpu", M=Dyn](dyn, ctx)
    var opred = Adam.make["gpu", M=Pred](pred, ctx)
    orep.lr = Scalar[DT](3e-4)
    odyn.lr = Scalar[DT](3e-4)
    opred.lr = Scalar[DT](3e-4)
    orep.max_grad_norm = Scalar[DT](10.0)
    odyn.max_grad_norm = Scalar[DT](10.0)
    opred.max_grad_norm = Scalar[DT](10.0)

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
    print("  LATENT", LATENT, "H", H, "BINS", BINS, "sims", NUM_SIMS,
          "MAX_K", MAX_K, "K", K, "N", N, "B", B, "lr 3e-4 clip 10")

    var loss = run_muzero_gumbel_selfplay_gpu[
        CartPoleEnv[DType.float64], Rep, Dyn, Pred,
        OBS, ACT, LATENT, BINS, NUM_SIMS, MAX_NODES, MAX_K, CAP, B, K, N,
        L=RemoteLogger,
    ](
        ctx, env, rep, dyn, pred, orep, odyn, opred,
        iterations=60000,
        learning_starts=500,
        train_per_iter=1,
        gamma=Scalar[DT](0.997),
        v_min=Scalar[DT](-20.0),
        v_max=Scalar[DT](20.0),
        value_coef=Scalar[DT](1.0),
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
