"""Phase 3 EZv2 agent rewiring smoke: `NEW_MCTS=True`.

Sibling of `test_ezv2_run_gpu_with_sampling.mojo` but with the
`GumbelGPUMCTS` orchestrator routed in. Instantiates
`EZV2DiscreteMLPConfig` with the `NEW_MCTS=True` parameter so the
comptime-if branch in `run_ezv2_train_gpu` dispatches through the
orchestrator + mirrors outputs back into the legacy `mcts_gpu` buffer
for the downstream host-side SVE read.

Asserts: training run completes without NaN losses, fires at least one
train call, all four final loss components finite. Not a convergence
test — env-step budget is too small to expect learning.

Usage:
    pixi run -e apple mojo run -I . \\
        tests/deep_agents/test_ezv2_gpu_new_mcts.mojo
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


def main() raises:
    print("=== EZ-V2 NEW_MCTS=True smoke ===")

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
        NEW_MCTS=True,   # ← Route through GumbelGPUMCTS
    ]
    comptime N_ENVS = 4
    comptime NUM_ENV_STEPS = 2_000

    seed(2026)
    var agent = GenericEfficientZeroV2Agent[Config](
        gamma=0.99, v_min=-10.0, v_max=10.0, temperature=1.0,
        n_envs=N_ENVS,
    )
    var ctx = DeviceContext()

    print("Config.USE_NEW_MCTS =", Config.USE_NEW_MCTS)

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
        reanalyze_samples=4,
        reanalyze_warmup=10_000,
        log_every=500,
        rng_seed_base=UInt64(2026),
        use_gpu_sampling=False,
        verbose=True,
    )

    print()
    print("--- Summary ---")
    print("    num_train_calls    =", stats.num_train_calls)
    print("    num_episodes       =", stats.num_episodes)
    print("    any_nan_loss       =", stats.any_nan_loss)
    print(
        "    L_R / L_P / L_V / L_G =",
        stats.last_L_R, stats.last_L_P, stats.last_L_V, stats.last_L_G,
    )

    if stats.any_nan_loss:
        raise Error("FAIL: NaN/Inf in losses")
    if stats.num_train_calls <= 0:
        raise Error("FAIL: no train_step call fired")
    if not _is_finite(stats.last_L_R):
        raise Error("FAIL: last_L_R not finite")
    if not _is_finite(stats.last_L_P):
        raise Error("FAIL: last_L_P not finite")
    if not _is_finite(stats.last_L_V):
        raise Error("FAIL: last_L_V not finite")
    if not _is_finite(stats.last_L_G):
        raise Error("FAIL: last_L_G not finite")

    print()
    print("PASS: EZv2 training completed via GumbelGPUMCTS")
