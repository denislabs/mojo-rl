"""Phase 3 EZv2 continuous-agent rewiring smoke: `NEW_MCTS=True`.

Sibling of `test_ezv2_gpu_new_mcts.mojo` for the sampled-Gumbel
(continuous-action) variant. Instantiates `EZV2ContinuousMLPShallowConfig`
with `NEW_MCTS=True` so the comptime-if branch in
`run_ezv2_continuous_train_gpu` dispatches through `SampledGumbelGPUMCTS`
and mirrors outputs back into the legacy `mcts_gpu` buffer for the
downstream host-side SVE / policy-target code.

Asserts: training run completes without NaN losses, fires at least one
train call, all four final loss components finite. Not a convergence
test — env-step budget is too small.

Usage:
    pixi run -e apple mojo run -I . \\
        tests/deep_agents/test_ezv2_gpu_new_mcts_continuous.mojo
"""

from std.random import seed
from std.gpu.host import DeviceContext

from mojo_rl.envs.pendulum import PendulumV2
from mojo_rl.deep_agents.efficient_zero_v2 import (
    EZV2ContinuousMLPShallowConfig,
    GenericEZV2ContinuousAgent,
    run_ezv2_continuous_train_gpu,
)
from mojo_rl.nn.constants import dtype


def _is_finite(x: Float64) -> Bool:
    if x != x:
        return False
    if x > 1.0e300 or x < -1.0e300:
        return False
    return True


def main() raises:
    print("=== EZ-V2 continuous NEW_MCTS=True smoke ===")

    comptime Config = EZV2ContinuousMLPShallowConfig[
        OBS=3,
        ACT_DIM=1,
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
        K_ROOT=4,
        K_NON_ROOT=2,
        MAX_ACTION=2.0,
        MIN_STD=0.5,
        NEW_MCTS=True,  # ← Route through SampledGumbelGPUMCTS
    ]
    comptime N_ENVS = 4
    comptime NUM_ENV_STEPS = 2_000

    seed(2026)
    var agent = GenericEZV2ContinuousAgent[Config](
        gamma=0.99, v_min=-50.0, v_max=2.0, temperature=1.0,
        n_envs=N_ENVS,
    )
    var ctx = DeviceContext()

    print("Config.USE_NEW_MCTS =", Config.USE_NEW_MCTS)

    var stats = run_ezv2_continuous_train_gpu[
        PendulumV2[DType.float32],
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
        reanalyze_warmup=10_000,
        log_every=500,
        rng_seed_base=UInt64(2026),
        use_gpu_sampling=False,
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
    print("PASS: EZv2 continuous training completed via SampledGumbelGPUMCTS")
