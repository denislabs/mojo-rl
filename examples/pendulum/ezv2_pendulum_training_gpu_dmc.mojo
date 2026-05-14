"""EZ-V2 Pendulum — DMC-config diagnostic.

Same `EZV2ContinuousMLPConfig` (deep / BN-equipped / paper-spec) that
HalfCheetah uses, just with Pendulum dims + Pendulum-appropriate value
and reward ranges. The Pendulum-shallow example
(`ezv2_pendulum_training_gpu.mojo`) is the known-converging anchor at
-212 with the simpler `EZV2ContinuousMLPShallowConfig`.

This script exists to settle a forked hypothesis on the HC training
plateau:

  • If Pendulum-DMC converges → the deep config (BN projector, deep
    BN heads, ImproveResBlock × 2 rep+dyn, init_zero heads, LR warmup,
    LR=3e-4/WD=2e-5) is healthy across the algorithm. HC's failure is
    HC-specific — exploration / reward sparsity / chicken-and-egg
    bootstrap, not a code regression.

  • If Pendulum-DMC fails → one of our recent changes (BN swap in
    projector+predictor, SimSiam target branch BN training-mode fix,
    init_zero on heads, LR scheduler integration, etc.) regressed the
    algorithm itself. We then have a real bug to find rather than
    knobs to tune.

Either result is decisive. The shallow Pendulum example is kept
untouched as the regression anchor.

Knob alignment with HC (so the comparison is clean):
  • LATENT=128, HIDDEN=256, HEAD_HIDDEN=256, PROJ=128, PROJ_HID=512,
    PRED_BOTTLENECK=512 (paper-spec network sizes from `dmc_state.yaml`).
  • BS=256, K_UNROLL=5, N_TD=5, SIMS=32, K_ROOT=16, K_NON_ROOT=2.
  • LR=3e-4, WD=2e-5 (reference DMC) + LinearWarmupSchedule[1000]
    (= `lr_warm_up: 0.01` × 100k training_steps).
  • N_POLICY_AT_ROOT=4 (4 policy + 12 random root candidates).
  • ENT_WEIGHT=5e-2, LAMBDA_G=2.0, LAMBDA_V=0.5.
  • VALUE_TARGET_MIXED (reference behavior — SVE early, SARSA after).
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
    VALUE_TARGET_MIXED,
    run_ezv2_continuous_train_gpu,
)
from mojo_rl.envs.pendulum import PendulumV2
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.training.scheduler import LinearWarmupSchedule


def main() raises:
    print("=" * 72)
    print("    EZ-V2 Pendulum — DMC-config diagnostic (N_ENVS=8)")
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
        CAP=100000,
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
        VALUE_TARGET_MODE=VALUE_TARGET_MIXED,
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
        run_name="EZ-V2 Pendulum GPU (DMC config diagnostic)",
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
        reanalyze_warmup=1000,
        # Pendulum converges fast — no need for a long random warmup.
        # The exploration problem that drove HC to 20k doesn't apply here:
        # the agent's initial policy already produces full-range torques
        # via the squashed-Gaussian σ=1.4 init, and Pendulum's dense
        # reward gives signal even from random play.
        warmup_random_steps=2_000,
        max_steps_per_episode=200,
        log_every=2_000,
        rng_seed_base=UInt64(2026),
        use_gpu_sampling=True,
        use_gpu_mcts=True,
        # Pendulum's 3D obs is already well-scaled; obs_norm not needed
        # (and could hurt — the shallow config doesn't use it).
        obs_norm=False,
        logger=UnsafePointer(to=logger),
        verbose=True,
    )

    logger.close()
