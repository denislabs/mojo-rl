"""EZ-V2 HalfCheetah profiling script — minimal-runtime variant of
`ezv2_half_cheetah_training_gpu.mojo` for `nsys`/profiling.

Identical Config to the training script — same architecture (ResBlocks
in Rep+Dyn, deeper BN heads, PROJ_HID=512, ref-parity loss weights,
UTD=1.0, MIXED value target, etc.) so the profile reflects actual
training-time workload.

Differences from the training script:
  • `NUM_ENV_STEPS` drastically shortened (≈ a few minutes vs 8 h).
  • `warmup_random_steps` reduced to the minimum needed to fill one
    sample batch (BS=256 with N_ENVS=4 → 64 iterations × 4 env-steps =
    256 env-steps). Anything below this triggers `is_ready()=False` and
    training never starts.
  • `log_every` tightened so we still see structured output during a
    short run.
  • `reanalyze_warmup` and `reanalyze_interval` left as configured so
    reanalyze fires at least once in the profile window.

Build + profile on NVIDIA:

    pixi run -e nvidia mojo build -I . \\
        examples/half_cheetah/ezv2_half_cheetah_profile.mojo \\
        -o /tmp/ezv2_hc_profile

    nsys profile --trace=cuda,nvtx,osrt \\
        --output=/tmp/ezv2_hc_nsys \\
        /tmp/ezv2_hc_profile

Open the resulting `.nsys-rep` in Nsight Systems for kernel-level
timing analysis. Look for:
  • Train-step hot path: rep / dyn / pred forward + backward kernels
    inside `train_step_core`.
  • Projector / predictor GPU MM kernels (now with BN — extra
    BatchNorm1D kernels added).
  • ResBlock kernels — 2× LN→Linear→ReLU→Linear→add in both Rep and Dyn.
  • CPU MCTS in `agent.select_action` (host-only — should show up as
    long CPU stretches between GPU bursts).
  • Reanalyze CPU search (also host-only).
"""

from std.random import seed
from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.efficient_zero_v2 import (
    EZV2ContinuousMLPConfig,
    GenericEZV2ContinuousAgent,
    VALUE_TARGET_MIXED,
    run_ezv2_continuous_train_gpu,
)
from mojo_rl.envs.half_cheetah import HalfCheetah
from mojo_rl.nn.constants import dtype


def main() raises:
    print("=" * 72)
    print("    EZ-V2 HalfCheetah — PROFILE BUILD (short run, nsys-ready)")
    print("=" * 72)

    # Profile budget: ~30-60s of steady-state training kernels on most
    # hardware. With BS=256 the buffer needs ~64 iters to fill before
    # training fires (gated by `agent.state.is_ready()`); after that
    # every iteration runs `train_steps_per_iter=N_ENVS` train calls.
    #   total       = 3000 env-steps (750 iterations × N_ENVS)
    #   buffer-fill = 64 iters (no training)
    #   train-iters = 686 × train_steps_per_iter=4 = 2744 train calls
    comptime NUM_ENV_STEPS = 3000
    comptime N_ENVS = 4

    # Identical to the training script — keep config in sync so profile
    # measures the actual training workload.
    comptime Config = EZV2ContinuousMLPConfig[
        OBS=17,
        ACT_DIM=6,
        LATENT=128,
        HIDDEN=256,
        HEAD_HIDDEN=256,
        PROJ=128,
        PROJ_HID=512,
        PRED_BOTTLENECK=512,
        BINS=51,
        BS=256,
        CAP=100000,
        K_UNROLL=5,
        N_TD=5,
        SIMS=32,
        NODES=128,
        K_ROOT=16,
        K_NON_ROOT=2,
        MAX_ACTION=1.0,
        MIN_STD=0.1,
        STD_MAGNIFICATION=3.0,
        N_POLICY_AT_ROOT=4,
        ENT_WEIGHT=5e-2,
        LAMBDA_G=2.0,
        LAMBDA_V=0.5,
        VALUE_TARGET_MODE=VALUE_TARGET_MIXED,
    ]

    seed(2026)
    var agent = GenericEZV2ContinuousAgent[Config](
        gamma=0.997,
        v_min=-50.0,
        v_max=100.0,
        temperature=1.0,
        temperature_decay_steps=10_000_000,
        max_grad_norm=5.0,
        n_envs=N_ENVS,
    )

    var ctx = DeviceContext()

    var _stats = run_ezv2_continuous_train_gpu[
        HalfCheetah[dtype, TERMINATE_ON_UNHEALTHY=False],
        Config,
        N_ENVS,
        NUM_ENV_STEPS,
    ](
        agent,
        ctx,
        train_interval=1,
        # UTD=1.0 — same as training script. The hot loop fires this many
        # train_step_gpu calls per iteration; profile captures the cost.
        train_steps_per_iter=N_ENVS,
        sync_interval=50,
        target_sync_interval=200,
        reanalyze_interval=200,
        reanalyze_samples=32,
        reanalyze_warmup=1000,
        # No artificial warmup — training fires as soon as buffer fills.
        warmup_random_steps=0,
        # Profile-only: shorten episodes from 1000 to 100 so episodes
        # complete within the budget and the replay buffer flushes (the
        # buffer only flushes at `done`; with the production
        # `max_steps=1000` no episode would finish in 3000 env-steps and
        # training would never fire — leaving the profile capturing only
        # env-step kernels). With `max_steps=100`, each of N_ENVS=4 envs
        # completes a fresh episode every 25 iterations → first flush at
        # iter 25 (≈100 env-steps); BS=256-buffer ready at iter ~64;
        # steady-state training from iter ~64 onward.
        max_steps_per_episode=100,
        log_every=500,
        rng_seed_base=UInt64(2026),
        use_gpu_sampling=True,
        use_gpu_mcts=True,
        obs_norm=True,
        verbose=True,
    )
