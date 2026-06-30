"""EfficientZeroV2 continuous Pendulum convergence run (v2, GPU sampled-Gumbel).

The continuous lighthouse for the deep_agents EZv2 port — the continuous twin of
`examples/cartpole/ezv2_cartpole_v2_gpu.mojo`. Search runs on the
`SampledGumbelGPUMCTS` planner (Gumbel-Top-k + sequential halving over *sampled*
continuous action vectors via the squashed-Gaussian head `MZContPredGPU`), and
the update is `ezv2_unroll_train_step_continuous_gpu` (MuZero BPTT + SimSiam
consistency + squashed-Gaussian policy NLL) on the resident GPU nets. Watch the
greedy ``[eval]`` return — random Pendulum ≈ −1200..−1600, a good policy
≈ −150..−250 (swung up and held).

Run (GPU env required):
    pixi run -e apple mojo run -I . examples/pendulum/ezv2_pendulum_v2_gpu.mojo
"""

from std.memory import UnsafePointer
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.efficient_zero_v2.nets import (
    MZRepNet, MZDynNet, EZProjectorNet, EZPredictorNet,
)
from mojo_rl.deep_agents.efficient_zero_v2.nets_continuous import EZContPredNet
from mojo_rl.deep_agents.efficient_zero_v2.selfplay_gpu_continuous import (
    run_ezv2_sampled_selfplay_gpu,
)
from mojo_rl.envs.pendulum import PendulumEnv


def main() raises:
    comptime OBS = 3
    comptime ACT_DIM = 1
    comptime LATENT = 64
    comptime BINS = 51
    comptime H = 64
    comptime PROJ = 128
    comptime PROJ_HID = 128
    comptime BOTTLENECK = 64
    # NOTE: the 64-sim / K_ROOT=8 "tuned" variant REGRESSED (greedy flat ~-1450
    # vs reanalyze-only's descent to -920) — halving root candidates starved the
    # continuous sampler of action diversity and amplified the policy collapse.
    # Reverted to the known-good reanalyze config; root diversity (K_ROOT=16)
    # matters more here than visits-per-candidate.
    comptime NUM_SIMS = 32
    comptime MAX_NODES = 128
    comptime K_ROOT = 16
    comptime K_NON_ROOT = 8
    comptime CAP = 50000
    comptime B = 128
    comptime K = 5
    comptime N = 5

    comptime Rep = MZRepNet[OBS, LATENT, H]
    comptime Dyn = MZDynNet[LATENT, ACT_DIM, BINS, H]
    comptime Pred = EZContPredNet[LATENT, ACT_DIM, BINS, H]
    comptime Proj = EZProjectorNet[LATENT, PROJ, PROJ_HID]
    comptime Predh = EZPredictorNet[PROJ, BOTTLENECK]

    var ctx = DeviceContext()
    var env = PendulumEnv[DType.float32]()
    var rep = Rep.make["gpu", Kaiming](Optional(ctx))
    var dyn = Dyn.make["gpu", Kaiming](Optional(ctx))
    var pred = Pred.make["gpu", Kaiming](Optional(ctx))
    var proj = Proj.make["gpu", Kaiming](Optional(ctx))
    var predh = Predh.make["gpu", Kaiming](Optional(ctx))
    var orep = Adam(lr=Scalar[DT](3e-4))
    var odyn = Adam(lr=Scalar[DT](3e-4))
    var opred = Adam(lr=Scalar[DT](3e-4))
    var oproj = Adam(lr=Scalar[DT](3e-4))
    var opredh = Adam(lr=Scalar[DT](3e-4))

    # ── metrics logger (silent no-op without RL_MONITOR_URL in env/.env) ──
    var env_vars = load_dotenv()
    var logger = RemoteLogger(
        server_url=env_vars.get("RL_MONITOR_URL", ""),
        run_name="EZv2 Pendulum (GPU sampled-Gumbel)",
        buffer_size=64,
        api_key=env_vars.get("RL_MONITOR_API_KEY", ""),
    )
    logger.set_config("agent", "EZv2")
    logger.set_config("env", "Pendulum")
    logger.set_config("framework", "deep_agents/nn")

    print("EZv2 Pendulum convergence (v2, GPU sampled-Gumbel — MuZero BPTT + SimSiam)")
    print("  LATENT", LATENT, "H", H, "PROJ", PROJ, "BINS", BINS,
          "sims", NUM_SIMS, "K_root", K_ROOT, "K", K, "N", N, "B", B)

    var loss = run_ezv2_sampled_selfplay_gpu[
        PendulumEnv[DType.float32], Rep, Dyn, Pred, Proj, Predh,
        OBS, ACT_DIM, LATENT, BINS, NUM_SIMS, MAX_NODES, K_ROOT, K_NON_ROOT,
        CAP, B, K, N,
        L=RemoteLogger,
    ](
        ctx, env, rep, dyn, pred, proj, predh,
        orep, odyn, opred, oproj, opredh,
        # the -920 run was still descending at 30k → give it room to converge.
        iterations=60000,
        learning_starts=2000,
        train_per_iter=1,
        gamma=Scalar[DT](0.99),
        v_min=Scalar[DT](-50.0),
        v_max=Scalar[DT](2.0),
        max_action=Scalar[DT](2.0),
        min_std=Scalar[DT](0.5),
        std_magnification=Scalar[DT](3.0),
        ent_scale=Scalar[DT](0.05),
        max_ep_steps=200,
        value_coef=Scalar[DT](0.25),
        consistency_coef=Scalar[DT](2.0),
        # stale-target fix: lagging target net + reanalyze refresh (~2 stored
        # positions re-searched per train step once warmed up).
        target_sync_interval=200,
        reanalyze_interval=1,
        reanalyze_warmup=500,
        reanalyze_batch=2,
        eval_every=2000,
        eval_episodes=5,
        diag_every=200,
        report_every=500,
        logger=UnsafePointer(to=logger).as_unsafe_any_origin(),
        seed=42,
        verbose=True,
    )
    logger.close()

    print("final loss:", loss)
