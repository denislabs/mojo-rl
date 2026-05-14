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
from std.memory import UnsafePointer
from std.gpu.host import DeviceContext
from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.efficient_zero_v2 import (
    EZV2ContinuousMLPConfig,
    GenericEZV2ContinuousAgent,
    VALUE_TARGET_MIXED,
    run_ezv2_continuous_train_gpu,
)
from mojo_rl.envs.half_cheetah import HalfCheetah
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.training.scheduler import LinearWarmupSchedule


def main() raises:
    print("=" * 72)
    print("    EZ-V2 HalfCheetah — full GPU path (Phase 4, N_ENVS=4)")
    print("=" * 72)

    # Paper budget for HalfCheetah first-convergence on DMC state.
    comptime NUM_ENV_STEPS = 100_000
    comptime N_ENVS = 4

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
    # 2026-05-13: sweeping audit aligned everything to reference
    # `dmc_state.yaml`. See doc-update commit for the full diff table.
    comptime Config = EZV2ContinuousMLPConfig[
        OBS=17,
        ACT_DIM=6,
        # LATENT = reference `hidden_shape: 128`. Was 256 (2× over-wide).
        LATENT=128,
        # HIDDEN = reference `rep_net_shape: 256` and `dyn_shape: 256`.
        HIDDEN=256,
        # HEAD_HIDDEN = reference `pi_net_shape=val_net_shape=
        # rew_net_shape=[256, 256]`.
        HEAD_HIDDEN=256,
        # Projector: 128 → 512 → 512 → 128 (ref `proj_hid_shape=512,
        # proj_shape=128`). Predictor: 128 → 512 → 128 (ref
        # `pred_hid_shape=512, pred_shape=128`).
        PROJ=128,
        PROJ_HID=512,
        PRED_BOTTLENECK=512,
        BINS=51,
        # Reference `batch_size: 256` and `buffer_size: 100000` (these
        # are also the new struct defaults, so the explicit overrides
        # below are redundant — kept for clarity).
        BS=256,
        CAP=100000,
        # Reference `dmc_state.yaml: lr=3e-4, weight_decay=2e-5`. The
        # `EZV2ContinuousMLPConfig` defaults are `LR=1e-3, WD=1e-4` —
        # ~3.3× / 5× more aggressive than reference. With Adam + the
        # `softplus(σ_raw + INIT_STD) + MIN_STD` policy parameterisation
        # the higher LR collapses σ toward MIN_STD before the encoder
        # has built state-discriminative features, locking the policy
        # into a moderate-action attractor (see 2026-05-14 HC analysis).
        # Pair with the 1000-step `LinearWarmupSchedule` below.
        LR=3e-4,
        WD=2e-5,
        K_UNROLL=5,
        N_TD=5,
        SIMS=32,
        NODES=128,
        K_ROOT=16,
        # Reference `cy_mcts.py: leaf_num=2`. Was 8.
        K_NON_ROOT=2,
        MAX_ACTION=1.0,  # ← DMC convention: actions ∈ [-1, 1]
        # Reference `dmc_state.yaml` policy MIN_STD = 0.1. Was raised to
        # 0.5 during action-saturation debugging; now safe to revert with
        # uniform-random root sampling providing exploration.
        MIN_STD=0.1,
        STD_MAGNIFICATION=3.0,
        # SOFT_CLAMP=5.0 and INIT_STD=1.0 are the reference Dreamer-v3
        # parameterization (`ez_dmc_state.py:421-422`); they're the
        # `ContinuousActionSpace` defaults so they don't strictly need
        # to be set here, but listing them makes the action-policy
        # parameterization (μ = SOFT_CLAMP·tanh(μ_raw/SOFT_CLAMP); σ =
        # softplus(σ_raw + INIT_STD) + MIN_STD) explicit. Before the
        # 2026-05-14 fix the soft-clamp was tied to MAX_ACTION=1.0,
        # capping the pre-squash mean at ±1 and the post-squash action
        # mean at ±0.76 — HC could not learn to drive actions to
        # saturation. See `docs/EZV2_CONTINUOUS_OPEN_ISSUES.md`.
        SOFT_CLAMP=5.0,
        INIT_STD=1.0,
        # Reference DMC root sampling: 4 from policy, 12 uniform random.
        N_POLICY_AT_ROOT=4,
        # Reference `dmc_state.yaml: entropy_coeff: 5e-2`.
        ENT_WEIGHT=5e-2,
        # Reference `dmc_state.yaml: consistency_coeff: 2.0`.
        LAMBDA_G=2.0,
        # Reference `dmc_state.yaml: value_loss_coeff: 0.5` (also matches
        # the new struct default after the 2026-05-13 audit).
        LAMBDA_V=0.5,
        # Reference `dmc_state.yaml: value_target: 'mixed'`. Pure SARSA
        # used target-net V from step 0 when target net is undertrained,
        # producing biased targets. MIXED uses SVE (search-derived) for
        # the first ~20k training steps, then linearly transitions to
        # SARSA — same as reference's
        # `start_use_mix_training_steps=4e4, mixed_value_threshold=2e4`.
        VALUE_TARGET_MODE=VALUE_TARGET_MIXED,
    ]

    seed(2026)
    var agent = GenericEZV2ContinuousAgent[Config](
        # Reference `dmc_state.yaml: discount: 0.997`. Was 0.99 — effective
        # horizon 100 steps vs ref 333 steps for 1000-step HC episodes.
        gamma=0.997,
        # Value support — reference `dmc_state.yaml: value_support: range=
        # [-299, 299], bins=51, type=support`. The reference applies
        # `transform_one` (= our `scalar_transform`) to the raw range, so
        # the bins live at `[h(-299), h(299)] ≈ [-16.62, +16.62]` in
        # transformed space (step ≈ 0.665).
        # SYMMETRY MATTERS: prior `[-50, +100]` mid-bin in transformed
        # space was +25 → `h⁻¹(+25) ≈ +640` raw → random-init V decoded to
        # ~+640, and TD self-bootstrap kept value targets stuck at
        # +2000–3000 throughout training while real returns were ≈ -200.
        v_min=-16.620_185_174_601_966,  # h(-299)
        v_max=16.620_185_174_601_966,   # h(+299)
        # Reward support — reference `reward_support: range=[-2, 2]` was
        # sized for DMC HC where per-step reward ∈ [0, 1] (dm_control's
        # tolerance-normalized reward). Gymnasium HC's per-step reward is
        # `1.0·v_x − 0.1·||a||²` and a fast cheetah hits v_x ≈ 8–10 → raw
        # rewards in roughly [−0.6, +10]. With the old `±h(2) = ±0.732`
        # support, every per-step reward > 2 (or < −2) was silently
        # clipped by `two_hot_encode_kernel` → reward head structurally
        # underpredicts → MCTS Q ≈ R+γV biased low at fast states → no
        # gradient toward saturating actuators. Widened to ±h(15) ≈ ±3.0
        # in transformed space (3× headroom over max expected raw 10).
        reward_min=-3.015,   # h(-15)
        reward_max=3.015,    # h(+15)
        temperature=1.0,
        temperature_decay_steps=10_000_000,
        # Reference `dmc_state.yaml: max_grad_norm: 5`.
        max_grad_norm=5.0,
        n_envs=N_ENVS,
    )

    # init_zero on policy + reward heads (paper default, continuous
    # carve-out skips the value head) is applied automatically inside
    # `GenericEZV2ContinuousAgent.__init__`. Pass `init_zero_heads=False`
    # to the agent constructor to opt out.

    var ctx = DeviceContext()

    # Remote metrics logger — pulls server URL + API key from .env.
    # No-ops if RL_MONITOR_URL is empty (RemoteLogger.is_active() returns
    # False), so safe to keep enabled by default.
    var env_vars = load_dotenv()
    var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
    var url = env_vars.get("RL_MONITOR_URL", "")

    var logger = RemoteLogger(
        server_url=url,
        run_name="EZ-V2 HalfCheetah GPU (MIXED, paper-spec)",
        buffer_size=64,
        api_key=api_key,
    )
    logger.set_config("agent", "EZV2 Continuous")
    logger.set_config("env", "HalfCheetah")
    logger.set_config("obs_dim", String(17))
    logger.set_config("action_dim", String(6))
    logger.set_config("latent_dim", String(128))
    logger.set_config("hidden_dim", String(256))
    logger.set_config("head_hidden", String(256))
    logger.set_config("proj_dim", String(128))
    logger.set_config("proj_hidden", String(512))
    logger.set_config("pred_bottleneck", String(512))
    logger.set_config("batch_size", String(256))
    logger.set_config("buffer_capacity", String(100000))
    logger.set_config("num_simulations", String(32))
    logger.set_config("k_root", String(16))
    logger.set_config("k_non_root", String(2))
    logger.set_config("n_envs", String(N_ENVS))
    logger.set_config("value_target", "MIXED")
    logger.set_config("gamma", "0.997")
    logger.set_config("max_action", "1.0")
    logger.set_config("min_std", "0.1")
    logger.set_config("entropy_coeff", "5e-2")
    logger.set_config("consistency_coeff", "2.0")
    logger.set_config("value_coeff", "0.5")
    logger.set_config("max_grad_norm", "5.0")
    logger.set_config("num_env_steps", String(NUM_ENV_STEPS))
    logger.set_config("obs_norm", "True")

    # `TERMINATE_ON_UNHEALTHY=False` — match SAC's setting. Lets episodes
    # run the full 1000 steps even if the cheetah falls; poor postures
    # are penalized via reward, not by truncation. Default True is good
    # for deployment but bad for early-training exploration.
    var _stats = run_ezv2_continuous_train_gpu[
        HalfCheetah[dtype, TERMINATE_ON_UNHEALTHY=False],
        Config,
        N_ENVS,
        NUM_ENV_STEPS,
        L=RemoteLogger,
        # Reference `dmc_state.yaml: lr_warm_up: 0.01` = 1% of
        # `training_steps=100000` = 1000 train-step linear warmup, then
        # flat at LR. `LinearWarmupSchedule[1000]` does exactly this —
        # the scheduler returns `(epoch+1)/WARMUP_EPOCHS` for the first
        # 1000 train calls, then 1.0 thereafter. Total train steps with
        # `train_steps_per_iter=N_ENVS=4` and `NUM_ENV_STEPS=100000`
        # works out to 100000, so warmup is the first 1% of training —
        # exactly the reference recipe.
        SCHEDULER=LinearWarmupSchedule[WARMUP_EPOCHS=1000],
    ](
        agent,
        ctx,
        train_interval=1,
        # UTD=1.0 (match reference `dmc_state.yaml: training_steps=100000,
        # total_transitions=100000, num_envs=4` → 1 train per env-step =
        # 4 trains per iteration of N_ENVS=4 envs). Default 1 = UTD=0.25
        # which under-trains by 4× vs reference.
        train_steps_per_iter=N_ENVS,
        sync_interval=50,
        target_sync_interval=200,
        # Paper defaults (`dmc_state.yaml:55`). Reverted from the 16×
        # aggressive `(50, 128)` cadence 2026-05-13 — diagnostic for
        # whether aggressive reanalyze was propagating the SimSiam
        # collapse into stored MCTS targets faster than fresh
        # exploration could dilute it.
        reanalyze_interval=200,
        reanalyze_warmup=1000,
        warmup_random_steps=2_000,
        max_steps_per_episode=1_000,
        log_every=2_000,
        rng_seed_base=UInt64(2026),
        use_gpu_sampling=True,  # ← SARSA path requires CPU sampling
        # Hybrid path: GPU env stepping + GPU training + CPU MCTS via
        # `agent.select_action()`. The GPU MCTS itself has an unresolved
        # bug — Pendulum regression test 2026-05-13 confirmed GPU MCTS
        # doesn't learn while CPU MCTS does, same agent/training/env.
        # See docs/EZV2_CONTINUOUS_PHASE3_POSTMORTEM.md (GPU MCTS audit).
        # Hybrid is ~2× slower per env-step than full GPU MCTS but
        # actually trains; remaining diagnostic work is to instrument
        # gpu_mcts_sampled.mojo at runtime to find the tree-state
        # divergence from `mcts_sampled.mojo`.
        use_gpu_mcts=True,
        # Running obs-normalization (CleanRL VecNormalize semantics).
        # Reference (`EfficientZeroV2-main/ez/agents/ez_dmc_state.py:173-182`)
        # makes obs-norm part of the representation network — load-bearing
        # for DMC-state HalfCheetah where raw obs spans 17 dims with mixed
        # zero-/non-zero-mean and wide scales. We do the equivalent at the
        # env boundary so replay stores normalized obs and every consumer
        # (CPU MCTS, GPU MCTS, training, reanalyze) sees a consistent
        # distribution. Re-enabled 2026-05-13 after Bugs 1+2 alone weren't
        # enough — the Xavier-initialized first encoder linear assumes
        # unit-variance inputs and a few high-scale obs dims (qvel angular
        # velocity ~ ±100) were dominating the pre-activation, leaving the
        # encoder partially-collapsed and easy prey for the SimSiam pull.
        obs_norm=True,
        logger=UnsafePointer(to=logger),
        verbose=True,
    )

    logger.close()
