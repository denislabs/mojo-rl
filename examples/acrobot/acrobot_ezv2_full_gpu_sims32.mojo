"""EZ-V2 Acrobot — full-GPU (Phase B) with SIMS=32.

Same setup as `acrobot_ezv2_full_gpu.mojo` (GPU env + GPU MCTS + GPU
train via `run_ezv2_train_gpu`) but SIMS doubled from 16 to 32 to
compensate for the higher Gumbel-Top-k noise floor of GPU MCTS
(Float32 + Philox vs CPU's Float64 + Mojo `random`).

Background — see `docs/EZV2_FULL_GPU_PLAN.md` "GPU MCTS variance":

  Bisect (CPU env + GPU MCTS, SIMS=16, NVIDIA): mean=-322 best=-98
  Bisect (CPU env + GPU MCTS, SIMS=32, NVIDIA): mean=-202 best=-98
  Phase A (CPU env + CPU MCTS, SIMS=16, NVIDIA): mean=-252 best=-112

Doubling SIMS narrowed L_P (policy CE) from 1.023 → 0.955 vs CPU's
0.850, recovering most of the convergence gap. With CPU env held
fixed, SIMS=32 GPU MCTS *beats* CPU MCTS at SIMS=16 by ~50 mean
return points in similar wall time.

This file is the matching variant for the GPU env path. If it lands
near −202 (matching the bisect), GPU env physics drift is small and
SIMS=32 is the right production default. If it stays near −363
(matching the original SIMS=16 Phase B), the env path also contributes
meaningfully and warrants a Hopper-style trajectory-compare test.

Run:
    pixi run -e nvidia mojo run -I . examples/acrobot/acrobot_ezv2_full_gpu_sims32.mojo
    pixi run -e apple  mojo run -I . examples/acrobot/acrobot_ezv2_full_gpu_sims32.mojo
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
from mojo_rl.envs.acrobot import AcrobotEnv


def _mean(xs: List[Float64]) -> Float64:
    if len(xs) == 0:
        return 0.0
    var s = Float64(0.0)
    for i in range(len(xs)):
        s += xs[i]
    return s / Float64(len(xs))


def main() raises:
    print("=== EZ-V2 Acrobot — full-GPU (Phase B), SIMS=32 ===")

    # Same total-transitions budget as the Phase A multi-env demo so
    # results are directly comparable.
    comptime NUM_ENV_STEPS = 100_000
    comptime EVAL_WINDOW = 100
    comptime CONVERGENCE_TARGET = -100.0
    comptime N_ENVS = 4
    # Random-action warmup, paper `start_transitions: 2000`.
    comptime WARMUP_RANDOM_STEPS = 2_000

    # Identical to the Phase A multi-env config so loss / convergence
    # diffs isolate the env→GPU + MCTS→GPU port effect, not hyper-
    # parameter noise.
    comptime Config = EZV2DiscreteMLPConfig[
        OBS=6,
        ACT=3,
        LATENT=128,
        HIDDEN=128,
        PROJ=256,
        PRED_BOTTLENECK=128,
        BINS=51,
        BS=128,
        K_UNROLL=5,
        N_TD=5,
        SIMS=32,
        NODES=64,
        K_GUMBEL=2,
        LR=Float64(3e-4),
        LAMBDA_V=Float64(0.5),
        LAMBDA_G=Float64(2.0),
    ]

    seed(2026)
    var agent = GenericEfficientZeroV2Agent[Config](
        gamma=0.997,
        v_min=-400.0,
        v_max=0.0,
        temperature=1.0,
        temperature_decay_steps=10_000_000,
        max_grad_norm=5.0,
        n_envs=N_ENVS,
    )
    var ctx = DeviceContext()

    # `train_interval=1`: train every env-batch. With N_ENVS=4 that's
    # one train_step per 4 transitions — matches the multi-env demo's
    # `BATCH_TRAIN_INTERVAL=1`.
    var stats = run_ezv2_train_gpu[
        AcrobotEnv[DType.float32],
        Config,
        N_ENVS,
        NUM_ENV_STEPS,
    ](
        agent,
        ctx,
        train_interval=1,
        sync_interval=50,
        target_sync_interval=200,
        reanalyze_interval=200,
        reanalyze_samples=32,
        reanalyze_warmup=1000,
        warmup_random_steps=WARMUP_RANDOM_STEPS,
        log_every=2_000,
        rng_seed_base=UInt64(2026),
        verbose=True,
    )

    var n_eps = stats.num_episodes
    if n_eps < EVAL_WINDOW:
        print()
        print(
            "FAIL: only", n_eps,
            "episodes finished — need ≥", EVAL_WINDOW,
            "to evaluate. Try increasing NUM_ENV_STEPS.",
        )
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
        print(
            "PASS: Acrobot solved (mean ≥",
            CONVERGENCE_TARGET, ", got", final_mean, ")",
        )
    else:
        print(
            "INCONCLUSIVE: Acrobot did not hit", CONVERGENCE_TARGET,
            "— got", final_mean,
            "(improvement", initial_mean, "→", final_mean,
            "= ", final_mean - initial_mean, ")",
        )
        print("    Compare against Phase A: mean −252 / best −112")
        print("    (memory: project_ezv2_multienv_phase_a.md)")
