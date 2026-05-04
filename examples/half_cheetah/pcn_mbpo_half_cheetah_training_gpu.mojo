"""PCN-MBPO Agent GPU Training on HalfCheetah — Phase B3 control.

Mirror of `mbpo_half_cheetah_training_gpu.mojo` but uses the **PCN
dynamics ensemble** (deterministic, per-block PC weight rule + SGLD
inference) in place of vanilla MBPO's probabilistic Swish ensemble.

Same SAC side, same env, same rough hyperparameter shape; the only thing
that varies for the comparison is the world-model training procedure.

Run with:
    pixi run -e nvidia mojo run -I . examples/half_cheetah/pcn_mbpo_half_cheetah_training_gpu.mojo

For benchmarking against vanilla MBPO, run both:
    pixi run -e nvidia mojo run -I . examples/half_cheetah/mbpo_half_cheetah_training_gpu.mojo
    pixi run -e nvidia mojo run -I . examples/half_cheetah/pcn_mbpo_half_cheetah_training_gpu.mojo

with the same NUM_STEPS budget; final-100-episode reward + sample
efficiency curves are the comparable signals.
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import UnsafePointer

from std.gpu.host import DeviceContext

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.mbpo_pcn import (
    DefaultPCNMBPOConfig,
    PCNMBPOAgent,
)
from mojo_rl.deep_agents.core.strategies.termination import NeverTerminate
from mojo_rl.envs.half_cheetah import (
    HalfCheetah,
    HalfCheetahConfig,
)


# =============================================================================
# Constants — match vanilla MBPO HalfCheetah for a fair comparison.
# =============================================================================

# HalfCheetah: 17D observation, 6D continuous action.
comptime OBS_DIM = HalfCheetahConfig.OBS_DIM  # 17
comptime ACTION_DIM = HalfCheetahConfig.ACTION_DIM  # 6

# SAC network architecture — same as vanilla.
comptime HIDDEN_DIM = 256

# Buffer sizes — same as vanilla.
comptime BUFFER_CAPACITY = 1_000_000
comptime SYNTH_CAPACITY = 400_000
comptime BATCH_SIZE = 128

# Dynamics ensemble — match vanilla MBPO's NUM_ENSEMBLE/NUM_ELITES, with
# PCN's HIDDEN_DIM swapped in for the PCN architecture.
comptime NUM_ENSEMBLE = 7
comptime NUM_ELITES = 5
comptime DYN_HIDDEN = 200          # PCN's internal latent z width.
comptime DYN_BATCH = 256
comptime ROLLOUT_BATCH = 400       # match vanilla's `num_rollouts_per_step` shape.

# PCN training hyperparameters.
comptime T_INFER = 10              # SGLD inference iterations per minibatch.
comptime LR_X = 0.01               # SGLD step on latent z.
comptime DYN_GRAD_CLIP = 1.0

# Training duration — match vanilla.
comptime NUM_STEPS = 300_000
comptime WARMUP_STEPS = 5_000

comptime dtype = DType.float32

# PCN-MBPO config.
comptime PCNMBPOHalfCheetahConfig = DefaultPCNMBPOConfig[
    OBS_DIM,
    ACTION_DIM,
    HIDDEN_DIM,
    BUFFER_CAPACITY,
    SYNTH_CAPACITY,
    BATCH_SIZE,
    NUM_ENSEMBLE,
    NUM_ELITES,
    DYN_HIDDEN,
    DYN_BATCH,
    ROLLOUT_BATCH,
    0.0003,             # actor_lr
    0.0003,             # critic_lr
    0.001,              # model_lr (Adam on PC weights)
    T_INFER,
    LR_X,
    DYN_GRAD_CLIP,
    NeverTerminate,     # HalfCheetah has no termination
    1.0,                # action_scale
]


# =============================================================================
# Main
# =============================================================================


def main() raises:
    seed(42)
    print("=" * 70)
    print("PCN-MBPO Agent GPU Training on HalfCheetah — Phase B3")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        var agent = PCNMBPOAgent[
            PCNMBPOHalfCheetahConfig,
            RemoteLogger,
        ](
            gamma=0.99,
            tau=0.005,
            action_scale=1.0,
            alpha=0.2,
            auto_alpha=True,
            alpha_lr=0.0003,
            target_entropy=-3.0,
            model_train_freq=250,
            rollout_min_length=1,
            rollout_max_length=1,
            rollout_min_epoch=20,
            rollout_max_epoch=150,
            num_rollouts_per_step=100_000,
            real_ratio=0.05,
            sac_updates_per_step=40,
            # Option 2: bump per-call dynamics training budget to close the
            # ~3× per-obs-dim MSE gap to vanilla MBPO. Default is 50; vanilla
            # MBPO does ~50–500 effective per-member epochs per train call.
            # 200 ≈ 4× the previous PCN budget. Cost on RTX 5090: ~1.5s extra
            # per train call (every 250 env steps) ≈ ~6 min over a 60K-step run.
            dyn_train_minibatches_per_call=200,
            # Bump the warmup pretrain proportionally so dynamics is close to
            # vanilla quality before the first SAC update sees synth data.
            dyn_warmup_minibatches=2000,
            checkpoint_every=50_000,
            checkpoint_path="pcn_mbpo_half_cheetah.ckpt",
            diag_every=500,
        )

        print("Environment: HalfCheetah Continuous (GPU)")
        print("Agent: PCN-MBPO (PCN dynamics ensemble + SAC policy)")
        print("  Observation dim     : " + String(OBS_DIM))
        print("  Action dim          : " + String(ACTION_DIM))
        print("  SAC hidden dim      : " + String(HIDDEN_DIM))
        print("  PCN dynamics hidden : " + String(DYN_HIDDEN))
        print("  Real buffer cap     : " + String(BUFFER_CAPACITY))
        print("  Synth buffer cap    : " + String(SYNTH_CAPACITY))
        print("  Batch size (SAC)    : " + String(BATCH_SIZE))
        print("  Batch size (PCN)    : " + String(DYN_BATCH))
        print("  Rollout batch       : " + String(ROLLOUT_BATCH))
        print("  Ensemble / elites   : "
              + String(NUM_ENSEMBLE) + " / " + String(NUM_ELITES))
        print("  PCN T_INFER         : " + String(T_INFER))
        print("  PCN lr_x            : " + String(LR_X))
        print("  Total env steps     : " + String(NUM_STEPS))
        print("  Warmup steps        : " + String(WARMUP_STEPS))
        print()

        # Logger setup (same as vanilla example).
        var env_vars = load_dotenv()
        var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
        var url = env_vars.get("RL_MONITOR_URL", "")

        var logger = RemoteLogger(
            server_url=url,
            run_name="PCN-MBPO HalfCheetah GPU",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("agent", "PCN-MBPO")
        logger.set_config("env", "HalfCheetah")
        logger.set_config("hidden_dim", String(HIDDEN_DIM))
        logger.set_config("dyn_hidden", String(DYN_HIDDEN))
        logger.set_config("ensemble_size", String(NUM_ENSEMBLE))
        logger.set_config("actor_lr", "3e-4")
        logger.set_config("critic_lr", "3e-4")
        logger.set_config("model_lr", "1e-3")
        logger.set_config("batch_size", String(BATCH_SIZE))
        logger.set_config("dyn_batch", String(DYN_BATCH))
        logger.set_config("rollout_batch", String(ROLLOUT_BATCH))
        logger.set_config("model_train_freq", "250")
        logger.set_config("rollout_length", "1")
        logger.set_config("real_ratio", "0.05")
        logger.set_config("sac_updates_per_step", "40")
        logger.set_config("t_infer", String(T_INFER))
        logger.set_config("lr_x", String(LR_X))

        print("Starting GPU training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        try:
            agent.logger = UnsafePointer(to=logger)
            var metrics = agent.train_gpu[
                HalfCheetah[dtype, TERMINATE_ON_UNHEALTHY=False],
            ](
                ctx,
                num_steps=NUM_STEPS,
                warmup_steps=WARMUP_STEPS,
                verbose=True,
                print_every=10_000,
                environment_name="HalfCheetah",
                logger=UnsafePointer(to=logger),
            )

            var end_time = perf_counter_ns()
            var elapsed_s = Float64(end_time - start_time) / 1e9
            logger.close()

            print("-" * 70)
            print()
            print("=" * 70)
            print("PCN-MBPO GPU Training Complete")
            print("=" * 70)
            print()
            print("Total steps    : " + String(NUM_STEPS))
            print("Training time  : " + String(elapsed_s)[byte=:6] + " seconds")
            print()
            print(
                "Final avg reward (last 100 episodes): "
                + String(metrics.mean_reward_last_n(100))[byte=:8]
            )
            print(
                "Best episode reward                 : "
                + String(metrics.max_reward())[byte=:8]
            )
            print()

            var final_avg = metrics.mean_reward_last_n(100)
            if final_avg > 1000.0:
                print("EXCELLENT: agent is running fast (avg > 1000)")
            elif final_avg > 500.0:
                print("SUCCESS: agent learned to run (avg > 500)")
            elif final_avg > 100.0:
                print("GOOD PROGRESS: agent learning locomotion (avg > 100)")
            elif final_avg > 0.0:
                print("LEARNING: agent improving (avg > 0)")
            else:
                print("EARLY STAGE: still exploring (avg < 0)")

            print()
            print("=" * 70)

        except e:
            print("!!! EXCEPTION !!!")
            print("Error:", e)
            print("!!! END EXCEPTION !!!")

    print(">>> main() completed normally <<<")
