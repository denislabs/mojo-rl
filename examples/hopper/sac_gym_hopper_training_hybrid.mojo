"""SAC GPU agent trained on Gymnasium MuJoCo Hopper-v5 (CPU envs) — diagnostic.

This is the env-vs-algorithm attribution test. We swap our native physics3d
Hopper for the Gymnasium MuJoCo Hopper-v5 (CPU only, via Python). The SAC
agent's networks, replay buffer, and gradient updates still run on GPU; only
the env step happens on CPU. Each iteration marshals obs/actions/rewards
across the H↔D boundary.

If this run produces a stable converging policy on Gymnasium Hopper, we know
the algorithm + infrastructure are sound and the issue is in our env. If this
run plateaus the same way, the problem is upstream of the env.

Run with:
    pixi run -e nvidia mojo run -I . examples/hopper/sac_gym_hopper_training_hybrid.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import UnsafePointer, alloc

from std.gpu.host import DeviceContext

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.envs.gymnasium import GymMuJoCoEnv


# =============================================================================
# Constants — match sac_hopper_training_gpu.mojo so the comparison is clean.
# =============================================================================

comptime OBS_DIM = 11  # Gymnasium Hopper-v5 obs is 11D
comptime ACTION_DIM = 3
comptime HIDDEN_DIM = 256

comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 256

# Number of CPU envs stepped sequentially per iteration. Match the native
# Hopper run so total transitions / training-time per env matches.
comptime N_ENVS = 4

# Training duration. Hybrid is slower than full GPU due to per-step CPU env
# stepping + H↔D marshalling, so 1M is a reasonable diagnostic budget.
comptime NUM_STEPS = 1_000_000
comptime WARMUP_STEPS = 10_000


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC (GPU) on Gymnasium MuJoCo Hopper-v5 (CPU envs) — Hybrid")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        var agent = DeepSACAgent[
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=HIDDEN_DIM,
            buffer_capacity=BUFFER_CAPACITY,
            batch_size=BATCH_SIZE,
            actor_lr=0.001,
            critic_lr=0.001,
            L=RemoteLogger,
            max_n_envs=N_ENVS,
        ](
            gamma=0.99,
            tau=0.005,
            action_scale=1.0,
            alpha=0.2,
            auto_alpha=True,
            alpha_lr=0.001,
            target_entropy=-3.0,
            max_grad_norm=0.0,
            checkpoint_every=100_000,
            checkpoint_path="sac_gym_hopper_hybrid.ckpt",
        )

        # Build N CPU envs on the heap. Each is an independent Python/MuJoCo
        # instance — `gym.make` creates a fresh `gym.Env` per call.
        var envs = List[UnsafePointer[GymMuJoCoEnv, MutAnyOrigin]]()
        for _ in range(N_ENVS):
            var p = alloc[GymMuJoCoEnv](1)
            p.init_pointee_move(GymMuJoCoEnv("Hopper-v5"))
            envs.append(p)

        # Separate env for deterministic eval — must not share state with
        # any training env so eval doesn't perturb the rollout.
        var eval_env_ptr = alloc[GymMuJoCoEnv](1)
        eval_env_ptr.init_pointee_move(GymMuJoCoEnv("Hopper-v5"))

        print("Env: Gymnasium MuJoCo Hopper-v5 (CPU)")
        print("Agent: SAC (GPU networks, replay, training)")
        print("  Obs dim: " + String(OBS_DIM))
        print("  Action dim: " + String(ACTION_DIM))
        print("  Hidden dim: " + String(HIDDEN_DIM))
        print("  Parallel envs: " + String(N_ENVS))
        print("  Warmup steps: " + String(WARMUP_STEPS))
        print()

        # ---------------------------------------------------------------
        # Logger
        # ---------------------------------------------------------------
        var env_vars = load_dotenv()
        var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
        var url = env_vars.get("RL_MONITOR_URL", "")

        var logger = RemoteLogger(
            server_url=url,
            run_name="SAC GymHopper Hybrid",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("agent", "SAC")
        logger.set_config("env", "Gymnasium-Hopper-v5")
        logger.set_config("hidden_dim", String(HIDDEN_DIM))
        logger.set_config("actor_lr", "1e-3")
        logger.set_config("critic_lr", "1e-3")
        logger.set_config("alpha_lr", "1e-3")
        logger.set_config("batch_size", String(BATCH_SIZE))
        logger.set_config("buffer_capacity", String(BUFFER_CAPACITY))
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("warmup_steps", String(WARMUP_STEPS))

        print("Starting hybrid training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        try:
            var metrics = agent.train_hybrid[GymMuJoCoEnv](
                ctx,
                envs,
                num_steps=NUM_STEPS,
                warmup_steps=WARMUP_STEPS,
                verbose=True,
                print_every=50_000,
                logger=UnsafePointer(to=logger),
                reward_scale=1.0,
                eval_env=eval_env_ptr,
                eval_every=50_000,
                eval_episodes=5,
                eval_max_steps=1000,
                diag_every=1000,
            )

            var end_time = perf_counter_ns()
            var elapsed_s = Float64(end_time - start_time) / 1e9

            logger.close()

            print("-" * 70)
            print()
            print("Hybrid training complete")
            print("Total steps: " + String(NUM_STEPS))
            print(
                "Training time: "
                + String(elapsed_s)[byte=:6]
                + " seconds"
            )
            print()
            print(
                "Final avg reward (last 100): "
                + String(metrics.mean_reward_last_n(100))[byte=:8]
            )
            print(
                "Best episode reward: "
                + String(metrics.max_reward())[byte=:8]
            )

            var final_avg = metrics.mean_reward_last_n(100)
            print()
            if final_avg > 3000.0:
                print(
                    "EXCELLENT — sustained Hopper-expert level on MuJoCo."
                    " Algorithm is fine; the gap is in our physics."
                )
            elif final_avg > 1500.0:
                print(
                    "GOOD — agent learns to hop. Compare convergence speed"
                    " against native run to attribute the residual gap."
                )
            else:
                print(
                    "PLATEAU — same failure on MuJoCo as on our physics."
                    " The issue is upstream of the env (algorithm or"
                    " infrastructure)."
                )

        except e:
            print("!!! EXCEPTION !!!")
            print("Error:", e)
            print("!!! END EXCEPTION !!!")

        # Close all envs and free the heap-allocated storage.
        for i in range(N_ENVS):
            envs[i][].close()
            envs[i].free()
        eval_env_ptr[].close()
        eval_env_ptr.free()

    print(">>> main() completed <<<")
