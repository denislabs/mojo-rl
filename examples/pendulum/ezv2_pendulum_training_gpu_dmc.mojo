"""EZ-V2 Pendulum — full-reference DMC-config diagnostic (GPU MCTS).

Same `EZV2ContinuousMLPConfig` (deep / BN-equipped / paper-spec) that
HalfCheetah uses, just with Pendulum dims + Pendulum-appropriate value
and reward ranges. The Pendulum-shallow example
(`ezv2_pendulum_training_gpu.mojo`) is the known-converging anchor at
-212 with the simpler `EZV2ContinuousMLPShallowConfig`.

This script is the **#2 diagnostic** in the 2026-05-16 GPU-MCTS
investigation. After multiple single-knob tweaks on the shallow config
(c_scale, N_POLICY_AT_ROOT=4) failed to fix the GPU-MCTS non-convergence,
the plan was to establish whether the reference EZ-V2 setup converges
on Pendulum at all. The original plan ran CPU MCTS for a clean
bisection, but at deep-config + UTD=1.0 + N_ENVS=8 + 100k env_steps
that was projecting to ~25-50h wall, so we switched to **Option B**
(2026-05-16): run on GPU MCTS to get tractable wall time (~2-3h),
accepting that the bisection is now two-variable (config bundle AND
GPU-MCTS path together).

Reanalyze stays disabled (`reanalyze_warmup=10_000_000`) so we don't
fold a THIRD variable in. So vs the shallow CPU-MCTS converging run,
this flips exactly two switches: config family + acting-time MCTS
path. The inspect_root_gpu instrumentation will tell us whether any
failure mode matches the shallow-GPU saturation pattern.

Three possible outcomes and what each tells us:

  • Converges to good policy (≤ -200): deep config + GPU MCTS works.
    The shallow-GPU failure was specific to the shallow config's
    candidate-saturation regime (MIN_STD=0.5 + bang-bang μ → identical
    samples → MCTS can't differentiate). Deep config's MIN_STD=0.1 +
    bigger nets sidestep it. We adopt the DMC config as the production
    Pendulum config and the original investigation closes.

  • Doesn't converge AND inspect_root_gpu shows the same saturation
    pattern (log_prior clamped, mean_v indistinguishable, visit_H
    near log K): same root cause as shallow-GPU. The deeper arch
    didn't rescue it. Investigation pivots to fundamental algo/kernel
    issues in GPU MCTS at fp32.

  • Doesn't converge but the failure mode is DIFFERENT (e.g. NaN, BN
    train/eval mismatch, gradient explosion): we've at least ruled out
    the shallow-saturation hypothesis; new failure-mode-specific
    investigation begins.

The shallow Pendulum example is kept untouched as the regression
anchor.

(Original docstring framed this as an HC-diagnostic; that question is
still answerable from the same run.)

Knob alignment with HC (so the comparison is clean):
  • LATENT=128, HIDDEN=256, HEAD_HIDDEN=256, PROJ=128, PROJ_HID=512,
    PRED_BOTTLENECK=512 (paper-spec network sizes from `dmc_state.yaml`).
  • BS=256, K_UNROLL=5, N_TD=5, SIMS=32, K_ROOT=16, K_NON_ROOT=2.
  • LR=3e-4, WD=2e-5 (reference DMC) + LinearWarmupSchedule[1000]
    (= `lr_warm_up: 0.01` × 100k training_steps).
  • N_POLICY_AT_ROOT=4 (4 policy + 12 random root candidates).
  • ENT_WEIGHT=5e-2, LAMBDA_G=2.0, LAMBDA_V=0.5.
  • VALUE_TARGET_SARSA from step 0 (keystone fix per
    `[project_ezv2_continuous_pendulum_bugs]`; MIXED's SVE warmup was
    poisoning the value head in the previous DMC-Pendulum attempt).
  • CAP=30000 (overrides the previously-ignored config; bug fix
    2026-05-16 — replay capacity was hardcoded to 50k regardless).
  • init_zero on policy + reward heads (applied automatically).
  • UTD=1.0 via `train_steps_per_iter=N_ENVS`.

Pendulum-specific overrides:
  • OBS=3, ACT_DIM=1 → triggers the full-π policy loss path (paper
    Eq. 6), so this run also tests the deep-config × full-π combo.
  • MAX_ACTION=2.0 (Pendulum torque ∈ [-2, 2]).
  • gamma=0.99 (Pendulum standard; HC uses 0.997 for 1000-step ep).
  • v_min=-50, v_max=2 in transformed space (Pendulum returns sit in
    raw [-1600, 0], well inside h⁻¹([-50, 2])).
  • reward_min=-3.5, reward_max=3.5 (= ±h(20), covers Pendulum's
    raw per-step reward range [-16, 0] with safety margin).
  • max_steps_per_episode=200.
  • warmup_random_steps=2_000 (Pendulum converges fast; long warmup
    not needed here).
"""

from std.random import seed
from std.memory import UnsafePointer
from std.gpu.host import DeviceContext
from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.efficient_zero_v2 import (
    EZV2ContinuousMLPConfig,
    GenericEZV2ContinuousAgent,
    VALUE_TARGET_SARSA,
    run_ezv2_continuous_train_gpu,
)
from mojo_rl.envs.pendulum import PendulumV2
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.training.scheduler import LinearWarmupSchedule


def main() raises:
    print("=" * 72)
    print("    EZ-V2 Pendulum — DMC-config diagnostic (GPU MCTS, N_ENVS=8)")
    print("=" * 72)

    comptime NUM_ENV_STEPS = 100_000
    comptime N_ENVS = 8

    # Paper-spec network sizing (HC parity). Identical to
    # examples/half_cheetah/ezv2_half_cheetah_training_gpu.mojo
    # except for Pendulum dims (OBS, ACT_DIM) and Pendulum's action
    # range (MAX_ACTION).
    comptime Config = EZV2ContinuousMLPConfig[
        OBS=3,
        ACT_DIM=1,
        LATENT=128,
        HIDDEN=256,
        HEAD_HIDDEN=256,
        PROJ=128,
        PROJ_HID=512,
        PRED_BOTTLENECK=512,
        BINS=51,
        BS=256,
        # CAP threading bug fixed 2026-05-16 (this Config.CAP was previously
        # ignored; the agent + GPU replay state used a hardcoded _CAP=50000).
        # Pendulum converges in ~20-40K env_steps with the shallow config, so
        # 30k is plenty. Smaller buffer also forces eviction of early-training
        # transitions (including any SVE-poisoned ones if MIXED is re-enabled)
        # which helps freshness for off-policy learning.
        CAP=30000,
        LR=3e-4,
        WD=2e-5,
        K_UNROLL=5,
        N_TD=5,
        SIMS=32,
        NODES=128,
        K_ROOT=16,
        K_NON_ROOT=2,
        MAX_ACTION=2.0,         # ← Pendulum torque ∈ [-2, 2]
        MIN_STD=0.1,            # DMC default. Was 0.5 in shallow.
        STD_MAGNIFICATION=3.0,
        SOFT_CLAMP=5.0,
        INIT_STD=1.0,
        N_POLICY_AT_ROOT=4,     # 4 policy + 12 random.
        ENT_WEIGHT=5e-2,
        LAMBDA_G=2.0,
        LAMBDA_V=0.5,
        # SARSA from step 0 (vs MIXED which uses SVE for first T_FRESH=20K
        # train steps then switches). Memory [project_ezv2_continuous_
        # pendulum_bugs] identified VALUE_TARGET_SARSA as the keystone fix
        # for Pendulum — SVE overestimates V → critic collapse → world
        # model hallucination, exactly what we observed in the previous
        # MIXED run (v_pred_var collapsed 9187 → 289 at step 12K, then
        # SARSA transition recovered it at step 22K but the world model
        # was already poisoned and predicted -26 returns while reality
        # gave -855).
        VALUE_TARGET_MODE=VALUE_TARGET_SARSA,
    ]

    seed(2026)
    var agent = GenericEZV2ContinuousAgent[Config](
        # Pendulum standard discount. HC uses 0.997 for 1000-step ep;
        # Pendulum's 200-step ep doesn't need that horizon.
        gamma=0.99,
        # Pendulum value range — h-transformed. Raw returns ≈ [-1600, 0]
        # → h⁻¹([-50, 2]) ≈ [-2500, +3]. Same asymmetric range as the
        # shallow Pendulum baseline.
        v_min=-50.0,
        v_max=2.0,
        # Pendulum reward formula: -(θ² + 0.1·θ_dot² + 0.001·u²). Raw
        # per-step range [-16, 0]. h(±16) ≈ ±3.14 in transformed space;
        # ±3.5 covers it with safety margin. The shallow Pendulum config
        # leaves this at the default ±h(2) = ±0.732 — which silently
        # clips most per-step rewards. The shallow config converges
        # *despite* that clip because Pendulum's short horizon makes
        # value bootstrap dominate; for the DMC-config diagnostic we
        # set this correctly to remove one variable.
        reward_min=-3.5,
        reward_max=3.5,
        temperature=1.0,
        temperature_decay_steps=10_000_000,
        max_grad_norm=5.0,
        n_envs=N_ENVS,
    )

    var ctx = DeviceContext()

    var env_vars = load_dotenv()
    var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
    var url = env_vars.get("RL_MONITOR_URL", "")

    var logger = RemoteLogger(
        server_url=url,
        run_name="EZ-V2 Pendulum (DMC config, GPU MCTS, reanalyze off)",
        buffer_size=64,
        api_key=api_key,
    )
    logger.set_config("agent", "EZV2 Continuous (DMC config)")
    logger.set_config("env", "Pendulum")
    logger.set_config("obs_dim", String(3))
    logger.set_config("action_dim", String(1))
    logger.set_config("latent_dim", String(128))
    logger.set_config("hidden_dim", String(256))
    logger.set_config("batch_size", String(256))
    logger.set_config("num_simulations", String(32))
    logger.set_config("k_root", String(16))
    logger.set_config("n_envs", String(N_ENVS))
    logger.set_config("value_target", "MIXED")
    logger.set_config("lr", "3e-4")
    logger.set_config("weight_decay", "2e-5")
    logger.set_config("num_env_steps", String(NUM_ENV_STEPS))

    var _stats = run_ezv2_continuous_train_gpu[
        PendulumV2[dtype],
        Config,
        N_ENVS,
        NUM_ENV_STEPS,
        L=RemoteLogger,
        # Reference DMC `lr_warm_up: 0.01` × training_steps. With
        # N_ENVS=8 and train_steps_per_iter=8, total train calls =
        # NUM_ENV_STEPS = 100k, so 1000 warmup steps = 1% as in HC.
        SCHEDULER=LinearWarmupSchedule[WARMUP_EPOCHS=1000],
    ](
        agent,
        ctx,
        train_interval=1,
        # UTD=1.0 (match HC and DMC reference).
        train_steps_per_iter=N_ENVS,
        sync_interval=50,
        target_sync_interval=200,
        reanalyze_interval=200,
        # Reanalyze disabled: it runs `run_sampled_gumbel_search_gpu`
        # (GPU MCTS) regardless of `use_gpu_mcts`, which would
        # contaminate replay-buffer targets via the very kernel we're
        # trying to bisect out of the picture. Keep at the same sentinel
        # value the shallow CPU-MCTS converging run used.
        reanalyze_warmup=10_000_000,
        # Pendulum converges fast — no need for a long random warmup.
        # The exploration problem that drove HC to 20k doesn't apply here:
        # the agent's initial policy already produces full-range torques
        # via the squashed-Gaussian σ=1.4 init, and Pendulum's dense
        # reward gives signal even from random play.
        warmup_random_steps=2_000,
        max_steps_per_episode=200,
        log_every=2_000,
        rng_seed_base=UInt64(2026),
        # GPU MCTS path: CPU MCTS at deep-config + UTD=1.0 + N_ENVS=8 was
        # taking 10-20× the wall time of the shallow CPU-MCTS run. Option B
        # (2026-05-16) accepts the confounded bisection (config bundle +
        # GPU MCTS change together) in exchange for tractable wall time
        # (~2-3h). Reanalyze stays disabled below so only TWO variables
        # are flipped vs the shallow CPU-MCTS converging run: config
        # family and acting-time MCTS path. Reading the inspect_root_gpu
        # dumps will tell us whether the failure mode (if any) matches
        # the shallow-GPU saturation pattern.
        use_gpu_sampling=True,
        use_gpu_mcts=True,
        # Pendulum's 3D obs is already well-scaled; obs_norm not needed
        # (and could hurt — the shallow config doesn't use it).
        obs_norm=False,
        logger=UnsafePointer(to=logger),
        verbose=True,
    )

    logger.close()
