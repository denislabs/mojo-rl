"""EZ-V2 Pendulum — full GPU path regression test.

Same config as the known-converging CPU-stepping baseline
(`ezv2_pendulum_training_multienv_kroot16.mojo`, mean10=-212 at step
20400) but routed through the new `run_ezv2_continuous_train_gpu`
driver. Purpose: validate that the GPU driver itself is correct.

This is the strongest diagnostic for the HalfCheetah regression — if
Pendulum reaches mean10 ≤ -300 (well into swing-up) within 30k env-
steps, the driver is healthy and HalfCheetah's failure is environment-
hard. If Pendulum stays in random-policy territory (mean10 around
-1000), there's a bug in the driver / GPU MCTS plumbing.

Config matches the 5-bug-fix converging baseline:
  • v_min=-50 (was -20 in pre-fix scripts; required for Pendulum V(s)
    range to fit in the BINS support).
  • MIN_STD=0.5 + ENT_WEIGHT=0.05 (exploration knobs that paired with
    full-π NLL).
  • MAX_ACTION=2.0 (Pendulum torque range).
  • VALUE_TARGET_SARSA — the keystone fix.
  • K_ROOT=16, K_NON_ROOT=8 → ACT_DIM=1 → full-π policy loss fires.

`init_zero_output_heads` not used here — the converging baseline didn't
have it. Keeping the diff to baseline minimal so we isolate driver-vs-
script differences. Can be added back if the GPU driver path is shown
to be correct.
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


def main() raises:
    print("=" * 72)
    print("    EZ-V2 Pendulum — GPU driver regression test (N_ENVS=8)")
    print("=" * 72)

    comptime NUM_ENV_STEPS = 100_000
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
        MAX_ACTION=2.0,  # ← Pendulum torque ∈ [-2, 2]
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

    # Remote metrics logger — pulls server URL + API key from .env.
    # No-ops if RL_MONITOR_URL is empty (RemoteLogger.is_active() returns
    # False), so safe to keep enabled by default.
    var env_vars = load_dotenv()
    var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
    var url = env_vars.get("RL_MONITOR_URL", "")

    var logger = RemoteLogger(
        server_url=url,
        run_name="EZ-V2 Pendulum GPU (SARSA)",
        buffer_size=64,
        api_key=api_key,
    )
    logger.set_config("agent", "EZV2 Continuous")
    logger.set_config("env", "Pendulum")
    logger.set_config("obs_dim", String(3))
    logger.set_config("action_dim", String(1))
    logger.set_config("latent_dim", String(64))
    logger.set_config("hidden_dim", String(64))
    logger.set_config("batch_size", String(128))
    logger.set_config("num_simulations", String(32))
    logger.set_config("k_root", String(16))
    logger.set_config("n_envs", String(N_ENVS))
    logger.set_config("value_target", "SARSA")
    logger.set_config("num_env_steps", String(NUM_ENV_STEPS))

    var _stats = run_ezv2_continuous_train_gpu[
        PendulumV2[dtype],
        Config,
        N_ENVS,
        NUM_ENV_STEPS,
        L=RemoteLogger,
    ](
        agent,
        ctx,
        train_interval=1,
        # Pinned to preserve the postmortem -212 reference run, which
        # predates the UTD=1.0 default (was effective UTD=0.125 here).
        train_steps_per_iter=1,
        sync_interval=50,
        target_sync_interval=200,
        # Keep reanalyze cadence matching the converging CPU baseline.
        # The HalfCheetah experiment's 4× aggressive reanalyze isn't
        # required for Pendulum and would muddy the regression test.
        reanalyze_interval=200,
        reanalyze_warmup=1000,
        warmup_random_steps=2_000,
        max_steps_per_episode=200,  # ← Pendulum episode length
        log_every=2_000,
        rng_seed_base=UInt64(2026),
        use_gpu_sampling=True,
        use_gpu_mcts=True,
        logger=UnsafePointer(to=logger),
        verbose=True,
    )

    logger.close()
