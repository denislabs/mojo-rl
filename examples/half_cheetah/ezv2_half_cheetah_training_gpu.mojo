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

    # Paper budget for HalfCheetah first-convergence on DMC state.
    comptime NUM_ENV_STEPS = 100_000
    comptime N_ENVS = 8

    # Paper-spec network sizing (`references/EfficientZeroV2-main/ez/config/exp/dmc_state.yaml`).
    # Previous 128/128/256 pulse-check was undersized for HalfCheetah's
    # 17D obs × 6D action representational demand.
    #
    # PROJ=128 (was 1024) — 2026-05-13 audit. Reference's
    # `proj_shape=128` is 8× narrower than what we had. The previous
    # PROJ=1024 caused SimSiam encoder collapse: `L_G` raced to -0.999
    # within 10k env-steps (cosine ≈ +0.999 = trivial all-same-direction
    # solution), `L_V`/`L_R` then pinned at log(2)=0.69 (heads predicting
    # the marginal of the two-hot target since latents carry no state
    # info), and `L_P` went deeply negative as σ collapsed to MIN_STD.
    # Narrow PROJ constrains the projection manifold so the network must
    # build state-discriminative latents to drive consistency down.
    # `PRED_BOTTLENECK=512` is kept — combined with `PROJ=128` this gives
    # the 128→512→128 predictor shape that matches reference exactly.
    comptime Config = EZV2ContinuousMLPConfig[
        OBS=17,
        ACT_DIM=6,
        LATENT=256,
        HIDDEN=256,
        PROJ=128,
        PRED_BOTTLENECK=512,
        BINS=51,
        BS=128,
        K_UNROLL=5,
        N_TD=5,
        SIMS=32,
        NODES=128,
        K_ROOT=16,
        K_NON_ROOT=8,
        MAX_ACTION=1.0,  # ← DMC convention: actions ∈ [-1, 1]
        MIN_STD=0.1,  # ← paper default (Pendulum used 0.5)
        STD_MAGNIFICATION=3.0,
        # Reference (`ez/config/exp/dmc_state.yaml:67`): entropy_coeff = 5e-2
        # for ALL DMC envs (not Pendulum-specific). Previous reductions
        # (1e-3, 5e-3) chased the wrong direction — with ACT_DIM=6 and
        # simple-best NLL the policy collapses σ → MIN_STD=0.1 and
        # L_P → very negative; strong entropy gradient (-ENT/σ per dim)
        # is the load-bearing fix to prevent collapse. Mismatch found
        # 2026-05-13 audit of EfficientZeroV2-main/.
        ENT_WEIGHT=5e-2,
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

    # ── Paper init_zero on output heads ──────────────────────────────────
    # Reference (`dmc_state.yaml:120`) sets init_zero=True. With W=b=0 the
    # head's pre-activation is exactly 0 → softmax over BINS is uniform →
    # expected V/reward = mid-bin in transformed space, collapsing the
    # multi-thousand-batch overestimation-correction window that would
    # otherwise land the policy in a bad local mode.
    # See `docs/EZV2_CONTINUOUS_PHASE3_POSTMORTEM.md` (Phase 4 blocker #1).
    #
    # CONTINUOUS-only carve-out (reference `base_model.py:181`):
    #     `init_zero=False if is_continuous else init_zero`
    # for the value head. For continuous envs the value head is NOT
    # zeroed — only the policy head and the dynamics reward head are.
    # Why this matters: with W=0 on both pred heads, gradient through
    # `Linear` w.r.t. its input is `grad_out @ W^T = 0`, so the encoder
    # receives ZERO gradient from L_V and L_P at training start. The only
    # signal feeding the encoder is then SimSiam consistency — which
    # collapses to its trivial all-same-direction solution in ~250 train
    # steps (cos → +0.999, L_V/L_R pinned at log(2)=0.69). Keeping the
    # value head random-initialized lets L_V immediately pull the
    # encoder toward state-discriminative latents, defending against the
    # collapse attractor. Found 2026-05-13 audit.
    #
    # PredModel = Sequential[LinearMish, Parallel[PolicyHead, ValueHead]]
    #   → zero only branch 0 (policy head) of the trailing Parallel.
    # DynModel  = Sequential[SplitApply, LinearMish, LinearMish,
    #                        Parallel[NextLatent, RewardHead]]
    #   → zero only branch 1 (reward head) of the trailing Parallel.
    #
    # We can't bury this in a method on `GenericEZV2ContinuousAgent`
    # because the agent's `Config` is the `EZV2DiscreteConfig` trait,
    # which types PredModel/DynModel as the generic `Model` trait — its
    # Sequential/Parallel-specific `_param_offset` and `model_types`
    # accessors are only visible when we hold the concrete struct type.
    comptime PRED_N = Config.PredModel.N
    comptime PredLast = Config.PredModel.model_types[PRED_N - 1]
    comptime PRED_PARALLEL_OFF = Config.PredModel._param_offset[PRED_N - 1]()
    comptime PRED_POLICY_OFF_IN_PARALLEL = PredLast._param_offset[0]()
    comptime PRED_POLICY_PS = PredLast.branch_types[0].PARAM_SIZE
    var _pred_policy_start = PRED_PARALLEL_OFF + PRED_POLICY_OFF_IN_PARALLEL
    for i in range(_pred_policy_start, _pred_policy_start + PRED_POLICY_PS):
        agent.state.prediction.params[i] = Scalar[dtype](0.0)

    comptime DYN_N = Config.DynModel.N
    comptime DynLast = Config.DynModel.model_types[DYN_N - 1]
    comptime DYN_PARALLEL_OFF = Config.DynModel._param_offset[DYN_N - 1]()
    comptime DYN_REWARD_OFF_IN_PARALLEL = DynLast._param_offset[1]()
    comptime DYN_REWARD_PS = DynLast.branch_types[1].PARAM_SIZE
    var _dyn_reward_start = DYN_PARALLEL_OFF + DYN_REWARD_OFF_IN_PARALLEL
    for i in range(_dyn_reward_start, _dyn_reward_start + DYN_REWARD_PS):
        agent.state.dynamics.params[i] = Scalar[dtype](0.0)

    # Mirror the zeroing into the target nets so the boot-v decode at
    # train step 0 sees the same init as the online nets.
    agent.update_target_networks(tau=1.0)

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
        # 4× reanalyze: with CAP=50k and the previous (interval=200,
        # samples=32) cadence, only ~1k buffer slots were refreshed
        # over 60k env-steps — most stored MCTS targets were collected
        # under stale online networks. Tighten so the policy fits
        # targets that track the current target-net's Q.
        reanalyze_interval=50,
        reanalyze_samples=128,
        reanalyze_warmup=1000,
        warmup_random_steps=2_000,
        max_steps_per_episode=1_000,
        log_every=2_000,
        rng_seed_base=UInt64(2026),
        use_gpu_sampling=False,  # ← SARSA path requires CPU sampling
        # Hybrid path: GPU env stepping + GPU training + CPU MCTS via
        # `agent.select_action()`. The GPU MCTS itself has an unresolved
        # bug — Pendulum regression test 2026-05-13 confirmed GPU MCTS
        # doesn't learn while CPU MCTS does, same agent/training/env.
        # See docs/EZV2_CONTINUOUS_PHASE3_POSTMORTEM.md (GPU MCTS audit).
        # Hybrid is ~2× slower per env-step than full GPU MCTS but
        # actually trains; remaining diagnostic work is to instrument
        # gpu_mcts_sampled.mojo at runtime to find the tree-state
        # divergence from `mcts_sampled.mojo`.
        use_gpu_mcts=False,
        # Running obs-normalization (CleanRL VecNormalize semantics). Disabled for now to match the CPU baseline.
        # Reference (`EfficientZeroV2-main/ez/agents/ez_dmc_state.py:173-182`)
        # makes obs-norm part of the representation network — load-bearing
        # for DMC-state HalfCheetah where raw obs spans 17 dims with mixed
        # zero-/non-zero-mean and wide scales. We do the equivalent at the
        # env boundary so replay stores normalized obs and every consumer
        # (CPU MCTS, GPU MCTS, training, reanalyze) sees a consistent
        # distribution.
        obs_norm=False,
        verbose=True,
    )
