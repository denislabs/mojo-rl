"""Driver-level smoke test for `run_ezv2_train_gpu(use_gpu_sampling=True)`.

Runs the unified full-GPU training driver on CartPole for a short
window with GPU-side priority sampling enabled (`train_step_gpu_with_replay`
under the hood). Verifies:

  • run completes without NaN losses,
  • some episodes finish,
  • the four loss components are all finite at the end,
  • `stats.num_buffer_uploads` reflects the pre-train-step gpu_replay
    sync schedule (one upload at train_step=0 + one per `sync_interval`
    boundary).

Not a convergence test — the env-step budget is too small to expect
learning. We only verify the wiring is healthy.
"""

from std.random import seed
from std.gpu.host import DeviceContext
from mojo_rl.envs.cartpole import CartPoleEnv
from mojo_rl.deep_agents.efficient_zero_v2 import (
    EZV2DiscreteMLPConfig,
    GenericEfficientZeroV2Agent,
)
from mojo_rl.deep_agents.efficient_zero_v2.gpu_train import (
    run_ezv2_train_gpu,
)


def _is_finite(x: Float64) -> Bool:
    if x != x:
        return False
    if x > 1.0e300 or x < -1.0e300:
        return False
    return True


def _expect(
    cond: Bool,
    label: String,
    mut passed: Int,
    mut total: Int,
):
    total += 1
    if cond:
        print("PASS:", label)
        passed += 1
    else:
        print("FAIL:", label)


def main() raises:
    print("=== EZ-V2 run_ezv2_train_gpu(use_gpu_sampling=True) smoke ===")
    var passed = 0
    var total = 0

    comptime Config = EZV2DiscreteMLPConfig[
        OBS=4,
        ACT=2,
        LATENT=32,
        HIDDEN=32,
        PROJ=64,
        PRED_BOTTLENECK=32,
        BINS=21,
        BS=8,
        K_UNROLL=3,
        N_TD=5,
        SIMS=8,
        NODES=32,
        K_GUMBEL=2,
    ]
    comptime N_ENVS = 4
    comptime NUM_ENV_STEPS = 4_000

    seed(2026)
    var agent = GenericEfficientZeroV2Agent[Config](
        gamma=0.99, v_min=-10.0, v_max=10.0, temperature=1.0,
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
        sync_interval=20,
        target_sync_interval=200,
        reanalyze_interval=200,
        reanalyze_samples=8,
        reanalyze_warmup=10_000,  # > NUM_ENV_STEPS — disabled here
        log_every=1_000,
        rng_seed_base=UInt64(2026),
        use_gpu_sampling=True,
        verbose=True,
    )

    print()
    print("--- Summary ---")
    print("    num_train_calls    =", stats.num_train_calls)
    print("    num_buffer_uploads =", stats.num_buffer_uploads)
    print("    num_gpu_syncs      =", stats.num_gpu_syncs)
    print("    num_episodes       =", stats.num_episodes)
    print("    any_nan_loss       =", stats.any_nan_loss)
    print("    L_R / L_P / L_V / L_G =",
          stats.last_L_R, stats.last_L_P,
          stats.last_L_V, stats.last_L_G)

    _expect(
        not stats.any_nan_loss,
        "no NaN/Inf loss across the run",
        passed, total,
    )
    _expect(
        stats.num_train_calls > 0,
        "at least one train_step_gpu_with_replay call fired",
        passed, total,
    )
    _expect(
        stats.num_episodes > 0,
        "at least one episode finished",
        passed, total,
    )
    _expect(
        _is_finite(stats.last_L_R)
        and _is_finite(stats.last_L_P)
        and _is_finite(stats.last_L_V)
        and _is_finite(stats.last_L_G),
        "all four final loss components are finite",
        passed, total,
    )
    # GPU-sampling sync schedule: 1 upload at train_step=0 + 1 per
    # sync_interval=20 train calls thereafter. With `num_train_calls`
    # known we can bound `num_buffer_uploads` from below.
    var expected_uploads = (
        1 + (stats.num_train_calls // 20)
        if stats.num_train_calls >= 20
        else 1
    )
    _expect(
        stats.num_buffer_uploads >= expected_uploads,
        "gpu_replay synced ≥ expected (first sync + sync_interval boundaries)",
        passed, total,
    )

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
