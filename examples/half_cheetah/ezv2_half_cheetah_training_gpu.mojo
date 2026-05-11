"""EZ-V2 HalfCheetah training — full GPU path, N_ENVS=8.

Phase 4 validation of continuous EZ-V2 on a multi-dim action env after
the 5-bug Pendulum fix session (see
`docs/EZV2_CONTINUOUS_PHASE3_POSTMORTEM.md`).

Uses the unified continuous GPU driver `run_ezv2_continuous_train_gpu`:
  • Batched env stepping via `Phyics3dEnv.step_kernel_gpu[N_ENVS]`
    (one env struct, N parallel physics sims on a strided state buffer
    — same pattern SAC's `agent.train_gpu` already uses).
  • GPU MCTS via `run_sampled_gumbel_search_gpu` (sampled-Gumbel over
    K_ROOT candidates per env).
  • CPU replay buffer ground truth (the agent's host state). GPU
    sampling for continuous is SEARCH-only today; this script uses
    `VALUE_TARGET_SARSA`, so the driver dispatches `train_step_gpu`
    (CPU-sample, GPU-train).

ACT_DIM=6 path through the policy-loss kernels:
  • `USE_FULLPI = (IS_CONTINUOUS and action_dim==1)` evaluates False, so
    the simple-best loss path fires (`ezv2_policy_loss_grad_continuous_kernel`,
    paper Eq. 7). The K-candidate replay fields (`mcts_sampled_actions`,
    `mcts_improved_policy`) are stored but unused — same data flow as
    Pendulum, just without the full-π branch.

Original Phase 3 success criterion (`EFFICIENTZERO_V2_PLAN.md`):
≥ 400 mean return in 100k env steps (paper: 677). First attempts here
run 30k env steps as fast pulse-checks; extend once showing progress.

Reference: `references/EfficientZeroV2-main/ez/config/exp/dmc_state.yaml`
for hyperparameter defaults. SAC baseline on this env reaches ~5000
mean return in 600k env steps — EZ-V2 should match at much lower
budget per its sample-efficiency claim.
"""

from std.random import seed
from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.efficient_zero_v2 import (
    EZV2ContinuousMLPConfig,
    GenericEZV2ContinuousAgent,
    VALUE_TARGET_SARSA,
    run_ezv2_continuous_train_gpu,
)
from mojo_rl.envs.half_cheetah import HalfCheetah
from mojo_rl.nn.constants import dtype


def main() raises:
    print("=" * 72)
    print("    EZ-V2 HalfCheetah — full GPU path (Phase 4, N_ENVS=8)")
    print("=" * 72)

    comptime NUM_ENV_STEPS = 30_000
    comptime N_ENVS = 8

    comptime Config = EZV2ContinuousMLPConfig[
        OBS=17,
        ACT_DIM=6,
        LATENT=128,
        HIDDEN=128,
        PROJ=256,
        PRED_BOTTLENECK=128,
        BINS=51,
        BS=128,
        K_UNROLL=5,
        N_TD=5,
        SIMS=32,
        NODES=128,
        K_ROOT=16,
        K_NON_ROOT=8,
        MAX_ACTION=1.0,            # ← DMC convention: actions ∈ [-1, 1]
        MIN_STD=0.1,               # ← paper default (Pendulum used 0.5)
        STD_MAGNIFICATION=3.0,
        ENT_WEIGHT=5e-3,           # ← paper default
        # Bootstrapped TD value target — the keystone fix from Pendulum.
        # Reference uses this for DMC state envs. See
        # `docs/EZV2_CONTINUOUS_PHASE3_POSTMORTEM.md`.
        VALUE_TARGET_MODE=VALUE_TARGET_SARSA,
    ]

    seed(2026)
    var agent = GenericEZV2ContinuousAgent[Config](
        gamma=0.99,
        # HalfCheetah V(s) range: random ≈ -50 to 0, trained ≈ +500-1000.
        # Transformed (h(x) = sign(x)·(√(|x|+1)-1) + 0.001·x):
        #   h(-50)  = -6.05
        #   h(1000) = +30.7
        # [-50, +100] in transformed space comfortably covers actual V.
        v_min=-50.0,
        v_max=100.0,
        temperature=1.0,
        temperature_decay_steps=10_000_000,
        max_grad_norm=5.0,
        n_envs=N_ENVS,
    )

    var ctx = DeviceContext()

    # `TERMINATE_ON_UNHEALTHY=False` — match SAC's setting. Lets episodes
    # run the full 1000 steps even if the cheetah falls; poor postures
    # are penalized via reward, not by truncation. Default True is good
    # for deployment but bad for early-training exploration.
    var _stats = run_ezv2_continuous_train_gpu[
        HalfCheetah[dtype, TERMINATE_ON_UNHEALTHY=False],
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
        max_steps_per_episode=1_000,
        log_every=2_000,
        rng_seed_base=UInt64(2026),
        use_gpu_sampling=False,  # ← SARSA path requires CPU sampling
        verbose=True,
    )
