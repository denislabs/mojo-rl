"""EfficientZeroV2 discrete CartPole convergence run (v2, GPU Gumbel) — lighthouse.

The GPU twin of `ezv2_cartpole_v2_cpu`: identical learned-model hyperparameters,
but **search runs on the `GumbelGPUMCTS` planner** (Gumbel-Top-k + sequential
halving over the on-device h/g/f nets) and the update is `ezv2_unroll_train_step_gpu`
(MuZero BPTT + SimSiam consistency) on the resident GPU nets. Watch the greedy
``[eval]`` return — random CartPole ~22, "solving" ~195+. The training
``avg_return(10)`` understates the policy (∝-improved-policy sampling); judge by
the greedy eval.

Run (GPU env required):
    pixi run -e apple mojo run -I . examples/cartpole/ezv2_cartpole_v2_gpu.mojo
"""

from std.memory import UnsafePointer
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents2.efficient_zero_v2.nets import (
    MZRepNet, MZDynNet, MZPredNet, EZProjectorNet, EZPredictorNet,
)
from mojo_rl.deep_agents2.efficient_zero_v2.selfplay_gpu import (
    run_ezv2_gumbel_selfplay_gpu,
)
from mojo_rl.envs.cartpole import CartPoleEnv


def main() raises:
    comptime OBS = 4
    comptime ACT = 2
    comptime LATENT = 64
    comptime BINS = 51
    comptime H = 128
    comptime PROJ = 128
    comptime PROJ_HID = 128
    comptime BOTTLENECK = 64
    comptime NUM_SIMS = 25
    comptime MAX_NODES = 128
    comptime MAX_K = 2
    comptime CAP = 50000
    comptime B = 64
    comptime K = 5
    comptime N = 10

    comptime Rep = MZRepNet[OBS, LATENT, H]
    comptime Dyn = MZDynNet[LATENT, ACT, BINS, H]
    comptime Pred = MZPredNet[LATENT, ACT, BINS, H]
    comptime Proj = EZProjectorNet[LATENT, PROJ, PROJ_HID]
    comptime Predh = EZPredictorNet[PROJ, BOTTLENECK]

    var ctx = DeviceContext()
    var env = CartPoleEnv[DType.float32]()
    var rep = Rep.make["gpu", INIT=Kaiming](ctx)
    var dyn = Dyn.make["gpu", INIT=Kaiming](ctx)
    var pred = Pred.make["gpu", INIT=Kaiming](ctx)
    var proj = Proj.make["gpu", INIT=Kaiming](ctx)
    var predh = Predh.make["gpu", INIT=Kaiming](ctx)
    var orep = Adam.make["gpu", M=Rep](rep, ctx)
    var odyn = Adam.make["gpu", M=Dyn](dyn, ctx)
    var opred = Adam.make["gpu", M=Pred](pred, ctx)
    var oproj = Adam.make["gpu", M=Proj](proj, ctx)
    var opredh = Adam.make["gpu", M=Predh](predh, ctx)
    orep.lr = Scalar[DT](3e-4)
    odyn.lr = Scalar[DT](3e-4)
    opred.lr = Scalar[DT](3e-4)
    oproj.lr = Scalar[DT](3e-4)
    opredh.lr = Scalar[DT](3e-4)
    # Gradient clipping (reference uses 5; MuZero example uses 10). Without it
    # the dominant SimSiam consistency loss (coef 2.0) makes the update noisy.
    orep.max_grad_norm = Scalar[DT](5.0)
    odyn.max_grad_norm = Scalar[DT](5.0)
    opred.max_grad_norm = Scalar[DT](5.0)
    oproj.max_grad_norm = Scalar[DT](5.0)
    opredh.max_grad_norm = Scalar[DT](5.0)

    # ── metrics logger (silent no-op without RL_MONITOR_URL in env/.env) ──
    var env_vars = load_dotenv()
    var logger = RemoteLogger(
        server_url=env_vars.get("RL_MONITOR_URL", ""),
        run_name="EZv2 CartPole (GPU Gumbel)",
        buffer_size=64,
        api_key=env_vars.get("RL_MONITOR_API_KEY", ""),
    )
    logger.set_config("agent", "EZv2")
    logger.set_config("env", "CartPole")
    logger.set_config("framework", "deep_agents2/nn2")

    print("EZv2 CartPole convergence (v2, GPU Gumbel — MuZero BPTT + SimSiam)")
    print("  LATENT", LATENT, "H", H, "PROJ", PROJ, "BINS", BINS,
          "sims", NUM_SIMS, "K_gumbel", MAX_K, "K", K, "N", N, "B", B,
          "v±20 vcoef 0.5 clip 5 reanalyze temp")

    var loss = run_ezv2_gumbel_selfplay_gpu[
        CartPoleEnv[DType.float32], Rep, Dyn, Pred, Proj, Predh,
        OBS, ACT, LATENT, BINS, NUM_SIMS, MAX_NODES, MAX_K, CAP, B, K, N,
        L=RemoteLogger,
    ](
        ctx, env, rep, dyn, pred, proj, predh,
        orep, odyn, opred, oproj, opredh,
        iterations=30000,
        learning_starts=500,
        train_per_iter=1,
        gamma=Scalar[DT](0.997),
        v_min=Scalar[DT](-20.0),
        v_max=Scalar[DT](20.0),
        value_coef=Scalar[DT](0.5),
        consistency_coef=Scalar[DT](2.0),
        temperature_decay_steps=30000,
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
