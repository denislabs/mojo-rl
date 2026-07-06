"""Gumbel MuZero CartPole — reanalyze A/B (live-net vs target-net), GPU.

A validation harness for the gated **target-net reanalyze** added to the batched
MuZero drivers. It clones the converged CartPole-500 lighthouse config
(`muzero_cartpole_v2_gpu_gumbel.mojo`) EXACTLY — same nets, lr 3e-4, clip 10,
±20 h-space support, value_coef 1.0, temperature schedule, N_ENVS/sims/B/K/N,
seed — and changes only two things:

  * ``reanalyze_batch = B`` — HIGH coverage (vs the lighthouse's default
    ``N_ENVS``). This is the regime where target-net reanalyze is expected to
    matter: ~a full training batch worth of stored targets is re-searched per
    iteration, so most of each batch is freshly self-generated.
  * ``TARGET_SYNC_INTERVAL`` — the A/B knob (below). ``0`` reanalyzes with the
    LIVE nets (bit-identical to the pre-target-net path); ``> 0`` reanalyzes
    through lagging copies refreshed every that-many grad steps (the standard
    target-net stabiliser; matches EZv2 / official MuZero's delayed reanalyze).

How to run the A/B (GPU env required): run twice, flipping the knob, and compare
the two logged runs. Expect the target-net arm to converge **at least as well**
(CartPole is easy, so both should reach 500) with a **steadier value-loss** /
fewer target-chasing wobbles. If so, target-net reanalyze is de-risked for the
expensive pixel-Pong run.

    # arm A — live-net reanalyze (TARGET_SYNC_INTERVAL = 0)
    pixi run -e nvidia mojo run -I . examples/cartpole/muzero_cartpole_v2_gpu_gumbel_reanalyze_ab.mojo
    # arm B — target-net reanalyze: set TARGET_SYNC_INTERVAL = 200 below, rerun.
"""

from std.memory import UnsafePointer
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.muzero import MuZeroMLPConfig, MuZeroBatchedAgent
from mojo_rl.deep_agents.training import BatchedGpuDiscreteEnv
from mojo_rl.envs.cartpole import CartPoleEnv


# ── A/B knob: 0 → live-net reanalyze, > 0 → target-net synced every N grad steps.
comptime TARGET_SYNC_INTERVAL = 200


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

    # arm label for the run name / config so the two CSVs are distinguishable.
    var arm: String
    comptime if TARGET_SYNC_INTERVAL > 0:
        arm = String("target-net(sync ") + String(TARGET_SYNC_INTERVAL) + ")"
    else:
        arm = String("live-net")

    # ── metrics logger (silent no-op without RL_MONITOR_URL in env/.env) ──
    var env_vars = load_dotenv()
    var logger = RemoteLogger(
        server_url=env_vars.get("RL_MONITOR_URL", ""),
        run_name=String("Gumbel MuZero CartPole reanalyze A/B [") + arm + "]",
        buffer_size=64,
        api_key=env_vars.get("RL_MONITOR_API_KEY", ""),
    )
    logger.set_config("agent", "GumbelMuZero")
    logger.set_config("env", "CartPole")
    logger.set_config("framework", "deep_agents/nn")
    logger.set_config("replay", "device (GPUMCTSSequenceReplay)")
    logger.set_config("n_envs", String(N_ENVS))
    logger.set_config("reanalyze_batch", String(B))
    logger.set_config("target_sync_interval", String(TARGET_SYNC_INTERVAL))
    logger.set_config("reanalyze_arm", arm)

    print("Gumbel MuZero CartPole reanalyze A/B —", arm)
    print("  LATENT", Cfg.LATENT, "H", Cfg.HIDDEN, "BINS", Cfg.BINS,
          "sims", NUM_SIMS, "MAX_K", MAX_K, "K", K, "N", N, "B", B,
          "N_ENVS", N_ENVS, "lr 3e-4 clip 10")
    print("  reanalyze: every 1, batch", B, "(high coverage),",
          "target_sync_interval", TARGET_SYNC_INTERVAL)

    var loss = agent.train[L=RemoteLogger](
        env,
        iterations=15000,           # ·N_ENVS = 120k env steps
        learning_starts=500,        # stored steps before training
        max_ep_steps=MAX_EP,
        temperature_decay_steps=15000,
        reanalyze_every=1,
        reanalyze_batch=B,          # HIGH coverage — the target-net regime
        target_sync_interval=TARGET_SYNC_INTERVAL,
        eval_every=1000,
        eval_episodes=10,           # mean of 10 complete greedy games
        eval_env=UnsafePointer(to=eval_env).as_unsafe_any_origin(),
        diag_every=100,
        report_every=200,
        logger=UnsafePointer(to=logger).as_unsafe_any_origin(),
        seed=42,
        verbose=True,
    )
    logger.close()

    print("final loss:", loss)
