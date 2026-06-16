"""Gumbel MuZero CartPole convergence run (v2, GPU) — batched + device replay.

The batched, device-obs-replay variant of the CartPole-500 Gumbel lighthouse.
Same nets and fix stack (±20 h-space support, value_coef 1.0, temperature
schedule, reanalyze, truncation-aware replay, clip 10) and the same
`GumbelGPUMCTS` search (Gumbel-Top-k + sequential halving, ``MAX_K=2`` for
CartPole's 2 actions), but driven through
``run_muzero_gumbel_selfplay_gpu_batched_devreplay``:

  * ``N_ENVS`` CartPole envs step in parallel on the GPU
    (`BatchedGpuDiscreteEnv`) and are searched in ONE batched Gumbel launch
    (the rep net runs at batch=``N_ENVS`` at the root).
  * the trajectory replay (`GPUMCTSSequenceReplay`) keeps its obs ring on the
    **device** — obs are stored device→device from ``env.obs_ptr()`` and the
    training obs slab is gathered device→device into the train step, so no
    observation crosses the bus on the collection path.

CartPole obs are physical state values (not ``[0,1]`` pixels), so the obs ring
is stored as ``DT`` (``OBS_STORE_DT = DT`` — a lossless rebind, NOT the uint8
pixel quantization). The device ring needs ``CAP % N_ENVS == 0`` and
``CAP ≥ N_ENVS · max_ep_steps`` so no in-flight episode self-overwrites.

Driven through the `MuZeroBatchedAgent` facade (the batched sibling of
`MuZeroAgent`): its ``train`` wires this batched device-replay driver, recreating
session-local Adam optimizers with the convergence config (lr 3e-4, clip 10).

Run (GPU env required):
    pixi run -e apple mojo run -I . examples/cartpole/muzero_cartpole_v2_gpu_gumbel.mojo
"""

from std.memory import UnsafePointer
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.muzero import MuZeroMLPConfig, MuZeroBatchedAgent
from mojo_rl.deep_agents.training import BatchedGpuDiscreteEnv
from mojo_rl.envs.cartpole import CartPoleEnv


def main() raises:
    comptime Cfg = MuZeroMLPConfig[OBS=4, ACT=2, LATENT=128, HIDDEN=128, BINS=51]
    comptime OBS = Cfg.OBS          # 4
    comptime ACT = Cfg.ACT          # 2
    comptime LATENT = Cfg.LATENT
    comptime BINS = Cfg.BINS

    comptime N_ENVS = 8             # parallel GPU CartPole envs (batched search)
    comptime NUM_SIMS = 24
    comptime MAX_NODES = 128
    comptime MAX_K = 2              # Gumbel root candidates (= ACT for CartPole)
    comptime MAX_EP = 500
    # Device obs ring: CAP % N_ENVS == 0 AND CAP >= N_ENVS·MAX_EP (= 4000).
    comptime CAP = 50_000
    comptime B = 128
    comptime K = 5
    comptime N = 10

    comptime BatchedEnvT = BatchedGpuDiscreteEnv[CartPoleEnv[DT], N_ENVS, OBS, 1]
    comptime Agent = MuZeroBatchedAgent[
        BatchedEnvT, Cfg.Rep, Cfg.Dyn, Cfg.Pred,
        N_ENVS, OBS, ACT, LATENT, BINS,
        NUM_SIMS, MAX_NODES, MAX_K, CAP, B, K, N,
        OBS_STORE_DT = DT,          # CartPole obs are state values, NOT pixels
    ]

    var ctx = DeviceContext()

    var agent = Agent(
        ctx=ctx,
        lr=Scalar[DT](3e-4),
        gamma=Scalar[DT](0.997),
        v_min=Scalar[DT](-20.0),
        v_max=Scalar[DT](20.0),
        value_coef=Scalar[DT](1.0),
        max_grad_norm=Scalar[DT](10.0),   # CartPole v2 convergence stack
    )

    var env = BatchedEnvT(ctx)
    var eval_env = BatchedEnvT(ctx)

    # ── metrics logger (silent no-op without RL_MONITOR_URL in env/.env) ──
    var env_vars = load_dotenv()
    var logger = RemoteLogger(
        server_url=env_vars.get("RL_MONITOR_URL", ""),
        run_name="Gumbel MuZero CartPole (GPU, device replay)",
        buffer_size=64,
        api_key=env_vars.get("RL_MONITOR_API_KEY", ""),
    )
    logger.set_config("agent", "GumbelMuZero")
    logger.set_config("env", "CartPole")
    logger.set_config("framework", "deep_agents/nn")
    logger.set_config("replay", "device (GPUMCTSSequenceReplay)")
    logger.set_config("n_envs", String(N_ENVS))

    print("Gumbel MuZero CartPole convergence (v2, GPU — batched device replay)")
    print("  LATENT", Cfg.LATENT, "H", Cfg.HIDDEN, "BINS", Cfg.BINS,
          "sims", NUM_SIMS, "MAX_K", MAX_K, "K", K, "N", N, "B", B,
          "N_ENVS", N_ENVS, "lr 3e-4 clip 10")

    var loss = agent.train[L=RemoteLogger](
        env,
        iterations=15000,           # ·N_ENVS = 120k env steps
        learning_starts=500,        # stored steps before training
        max_ep_steps=MAX_EP,
        temperature_decay_steps=15000,
        reanalyze_every=1,
        eval_every=1000,
        eval_episodes=10,           # mean of 10 complete greedy games
        eval_env=UnsafePointer(to=eval_env),
        diag_every=100,
        report_every=200,
        logger=UnsafePointer(to=logger),
        seed=42,
        verbose=True,
    )
    logger.close()

    print("final loss:", loss)
