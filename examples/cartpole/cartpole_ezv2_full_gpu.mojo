"""EZ-V2 CartPole — final full-GPU demo.

End-of-plan deliverable: a clean call into
`run_ezv2_train_gpu[Env, Config, N_ENVS, NUM_ENV_STEPS](agent, ctx, ...)`
that packages Steps 1-4 of `docs/EZV2_FULL_GPU_PLAN.md`. Compare with
the legacy demos under `cartpole_ezv2_full_gpu_step{1..5}.mojo` to see
how the inline driver collapses into a single function call.

Plan:    docs/EZV2_FULL_GPU_PLAN.md
Gate:    final 100-ep mean return within ±5% of the Step 0 baseline
         (`logs/ezv2_full_gpu_baseline_step0.log`).

Run:
    pixi run mojo run -I . examples/cartpole/cartpole_ezv2_full_gpu.mojo
"""

from std.random import seed
from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.efficient_zero_v2 import (
    EZV2DiscreteMLPConfig,
    GenericEfficientZeroV2Agent,
)
from mojo_rl.deep_agents.efficient_zero_v2.gpu_train import (
    run_ezv2_train_gpu,
)
from mojo_rl.envs.cartpole import CartPoleEnv


def _mean(xs: List[Float64]) -> Float64:
    if len(xs) == 0:
        return 0.0
    var s = Float64(0.0)
    for i in range(len(xs)):
        s += xs[i]
    return s / Float64(len(xs))


def main() raises:
    print("=== EZ-V2 CartPole — full-GPU demo ===")

    comptime NUM_ENV_STEPS = 50_000
    comptime EVAL_WINDOW = 100
    comptime CONVERGENCE_TARGET = 450.0
    comptime N_ENVS = 4

    comptime Config = EZV2DiscreteMLPConfig[
        OBS=4,
        ACT=2,
        LATENT=128,
        HIDDEN=128,
        PROJ=256,
        PRED_BOTTLENECK=128,
        BINS=21,
        BS=64,
        K_UNROLL=3,
        N_TD=5,
        SIMS=16,
        NODES=64,
        K_GUMBEL=2,
        LR=Float64(5e-4),
        LAMBDA_G=Float64(1.0),
    ]

    seed(2026)
    var agent = GenericEfficientZeroV2Agent[Config](
        gamma=0.99,
        v_min=-15.0,
        v_max=15.0,
        temperature=1.0,
        temperature_decay_steps=10_000_000,
        n_envs=N_ENVS,
    )
    var ctx = DeviceContext()

    var stats = run_ezv2_train_gpu[
        CartPoleEnv[DType.float32],
        Config,
        N_ENVS,
        NUM_ENV_STEPS,
    ](
        agent,
        ctx,
        train_interval=4,
        sync_interval=50,
        target_sync_interval=200,
        reanalyze_interval=200,
        reanalyze_samples=32,
        reanalyze_warmup=1000,
        log_every=2_000,
        rng_seed_base=UInt64(2026),
        verbose=True,
    )

    var n_eps = stats.num_episodes
    if n_eps < EVAL_WINDOW:
        print()
        print("FAIL: only", n_eps, "episodes — need ≥", EVAL_WINDOW)
        return

    var last_window = List[Float64]()
    for i in range(n_eps - EVAL_WINDOW, n_eps):
        last_window.append(stats.ep_returns[i])
    var final_mean = _mean(last_window)
    var first_window = List[Float64]()
    var first_n = EVAL_WINDOW if EVAL_WINDOW < n_eps else n_eps
    for i in range(first_n):
        first_window.append(stats.ep_returns[i])
    var initial_mean = _mean(first_window)

    print(
        "    first ", EVAL_WINDOW, " ep mean return =", initial_mean,
    )
    print(
        "    last ", EVAL_WINDOW, " ep mean return =", final_mean,
    )
    print("    convergence target    =", CONVERGENCE_TARGET)

    print()
    if stats.any_nan_loss:
        print("FAIL: NaN/Inf loss during training")
    elif final_mean >= CONVERGENCE_TARGET:
        print("PASS: CartPole converged ≥", CONVERGENCE_TARGET,
              "(got", final_mean, ")")
    else:
        print("INCONCLUSIVE: did not hit", CONVERGENCE_TARGET,
              "— got", final_mean,
              "(improvement", initial_mean, "→", final_mean,
              "= ", final_mean - initial_mean, ")")
