"""EZ-V2 Pendulum — 15-min Apple budget A/B test: HYBRID path baseline.

Companion to `ezv2_pendulum_15min_full_gpu_mcts.mojo` — identical config
except `use_gpu_mcts=False`. Together they form an A/B test for whether
the GPU MCTS bug (open-issues item 2) still reproduces post-fixes, or
whether the search engine bug was incidentally resolved by recent
patches (CPU↔GPU parity test passes on both Apple and NVIDIA — see
`tests/deep_agents/test_ezv2_cpu_gpu_mcts_parity.mojo`).

Config is the converging-baseline Pendulum config from
`ezv2_pendulum_training_gpu.mojo` (5-bug-fix postmortem): MIN_STD=0.5,
ENT_WEIGHT=0.05, MAX_ACTION=2.0, VALUE_TARGET_SARSA, K_ROOT=16,
K_NON_ROOT=8 (so full-π policy loss fires with ACT_DIM=1).

Budget: 20k env-steps. Established convergence point on this config is
~20k env-steps (mean10 ≈ -212 ≈ swing-up territory). On Apple with
N_ENVS=8 the hybrid path should finish in ~10-13 min.
"""

from std.random import seed
from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.efficient_zero_v2 import (
    EZV2ContinuousMLPConfig,
    GenericEZV2ContinuousAgent,
    VALUE_TARGET_SARSA,
    run_ezv2_continuous_train_gpu,
)
from mojo_rl.envs.pendulum import PendulumV2
from mojo_rl.nn.constants import dtype


def main() raises:
    print("=" * 72)
    print("    EZ-V2 Pendulum 15-min — HYBRID (CPU MCTS + GPU env/train)")
    print("=" * 72)

    comptime NUM_ENV_STEPS = 20_000
    comptime N_ENVS = 8

    comptime Config = EZV2ContinuousMLPConfig[
        OBS=3,
        ACT_DIM=1,
        LATENT=64,
        HIDDEN=64,
        PROJ=128,
        PRED_BOTTLENECK=64,
        BINS=51,
        BS=128,
        K_UNROLL=5,
        N_TD=5,
        SIMS=32,
        NODES=128,
        K_ROOT=16,
        K_NON_ROOT=8,
        MAX_ACTION=2.0,
        MIN_STD=0.5,
        STD_MAGNIFICATION=3.0,
        ENT_WEIGHT=0.05,
        VALUE_TARGET_MODE=VALUE_TARGET_SARSA,
    ]

    seed(2026)
    var agent = GenericEZV2ContinuousAgent[Config](
        gamma=0.99,
        v_min=-50.0,
        v_max=2.0,
        temperature=1.0,
        temperature_decay_steps=10_000_000,
        max_grad_norm=5.0,
        n_envs=N_ENVS,
    )

    var ctx = DeviceContext()

    var _stats = run_ezv2_continuous_train_gpu[
        PendulumV2[dtype],
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
        warmup_random_steps=2_000,
        max_steps_per_episode=200,
        log_every=2_000,
        rng_seed_base=UInt64(2026),
        use_gpu_sampling=False,
        use_gpu_mcts=False,  # ← HYBRID
        verbose=True,
    )
