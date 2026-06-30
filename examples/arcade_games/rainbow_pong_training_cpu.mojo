"""Rainbow DQN CPU Training on Pong — clean obs (deep_agents).

Single-env CPU variant of `rainbow_pong_training_gpu.mojo` for FAST local
iteration on Apple. On a tiny 6→128→128→head MLP, single-env CPU stepping +
training runs thousands of env-steps AND train-steps per second — far faster
than the Apple-Metal GPU-batched path, whose per-step kernel-launch latency
dominates at this scale. Use this to sweep the convergence levers (C51
support, shaping reward, n-step, lr) without NVIDIA hardware.

Same agent as the GPU script (C51 + Double + PER + Dueling + Noisy + N-step)
built on the validated CPU path: host n-step ring + CPU prioritized replay +
`run_offpolicy_discrete_train` (one env step + one train step per iteration,
replay ratio 1.0). The driver runs a deterministic noise-off greedy eval on a
separate env and logs `eval/mean_return` — the true policy quality, not the
noisy training rollout.

Run with:
    pixi run mojo run -I . examples/arcade_games/rainbow_pong_training_cpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import UnsafePointer

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.nn.constants import DT

from mojo_rl.deep_agents.c51.trainer import C51Trainer
from mojo_rl.deep_agents.c51.config import RainbowNet
from mojo_rl.deep_agents.training.blocks import NStepSampleStep
from mojo_rl.deep_agents.data.any_per_replay import AnyPerReplay
from mojo_rl.deep_agents.training import run_offpolicy_discrete_train
from mojo_rl.envs.arcade_games.pong import PongEnv


# =============================================================================
# Constants
# =============================================================================

comptime OBS_DIM = PongEnv[DType.float64].OBS_DIM  # 6
comptime NUM_ACTIONS = PongEnv[DType.float64].NUM_ACTIONS  # 3

comptime HIDDEN_DIM = 128
comptime NUM_ATOMS = 51
# N_STEP=1 routes the host n-step ring through a plain 1-step add (the
# isolation setting). Bump to 3 for full Rainbow once the value config works.
comptime N_STEP = 1
comptime BUFFER_CAPACITY = 100_000  # host memory — cheap for 6-D obs
comptime BATCH_SIZE = 64

comptime WARMUP = 10_000
comptime NUM_STEPS = 1_000_000  # ~minutes on Apple CPU; lower for faster loops

comptime LR = Scalar[DT](6.25e-5)

# Distributional support — must bracket the DISCOUNTED return (≈ ±0.3..±6 with
# γ=0.99 + sparse rewards), NOT the raw ±21 episode score. [-2, 2] → atom
# spacing 0.08 (vs 0.84 at [-21, 21], which couldn't separate the 3 actions).
comptime V_MIN = Scalar[DT](-2.0)
comptime V_MAX = Scalar[DT](2.0)

# Dense ball-return shaping (env `HIT_REWARD`): 0.0 = clean sparse ±1 rewards;
# 0.1 = original shaping (distorts the value scale). Disabled here.
comptime HIT_REWARD = 0.0

comptime SAMPLE = NStepSampleStep[
    N_STEP, AnyPerReplay["cpu", OBS_DIM, 1, BUFFER_CAPACITY], BATCH_SIZE
]
comptime QNET = RainbowNet[OBS_DIM, NUM_ACTIONS, NUM_ATOMS, HIDDEN_DIM]
comptime RainbowTrainer = C51Trainer[
    "cpu", SAMPLE, QNET, NUM_ATOMS, NUM_ACTIONS, True
]
comptime PongCPU = PongEnv[DT, HIT_REWARD]


# =============================================================================
# Main
# =============================================================================


def main() raises:
    seed(42)
    print("=" * 70)
    print("Rainbow DQN CPU Training on Pong (deep_agents, single-env)")
    print("=" * 70)
    print()

    var trainer = RainbowTrainer.make(
        lr=LR,
        gamma=Scalar[DT](0.99),
        tau=Scalar[DT](0.005),
        epsilon=Scalar[DT](0.0),  # Noisy nets supply exploration
        learning_starts=WARMUP,
        target_update_freq=500,
        max_grad_norm=Scalar[DT](10.0),
        per_alpha=Scalar[DT](0.5),
        per_beta=Scalar[DT](0.4),
        per_epsilon=Scalar[DT](1e-6),
        nstep=N_STEP,
        v_min=V_MIN,
        v_max=V_MAX,
    )

    var env = PongCPU()
    var eval_env = PongCPU()  # separate env for the noise-off greedy eval

    print("Environment: Pong (CPU, single env)")
    print("Agent: Rainbow DQN (deep_agents C51, CPU)")
    print("  Components: C51 + Double + PER + Dueling + Noisy +", N_STEP, "-step")
    print("  Observation dim:", OBS_DIM)
    print("  Actions:", NUM_ACTIONS, "(NOOP, UP, DOWN)")
    print("  Hidden dim:", HIDDEN_DIM)
    print("  Atoms:", NUM_ATOMS, "support [", V_MIN, ",", V_MAX, "]")
    print("  Hit-reward shaping:", HIT_REWARD)
    print("  Buffer capacity:", BUFFER_CAPACITY)
    print("  Batch size:", BATCH_SIZE)
    print("  Learning rate:", LR)
    print("  Warmup:", WARMUP)
    print("  Total transitions:", NUM_STEPS)
    print()

    # =========================================================================
    # Logger
    # =========================================================================

    var env_vars = load_dotenv()
    var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
    var url = env_vars.get("RL_MONITOR_URL", "")

    var logger = RemoteLogger(
        server_url=url,
        run_name="Rainbow Pong CPU (deep_agents)",
        buffer_size=64,
        api_key=api_key,
    )
    logger.set_config("agent", "Rainbow DQN (deep_agents, CPU)")
    logger.set_config("env", "Pong")
    logger.set_config("hidden_dim", String(HIDDEN_DIM))
    logger.set_config("lr", String(LR))
    logger.set_config("gamma", "0.99")
    logger.set_config("batch_size", String(BATCH_SIZE))
    logger.set_config("buffer_capacity", String(BUFFER_CAPACITY))
    logger.set_config("n_step", String(N_STEP))
    logger.set_config("num_atoms", String(NUM_ATOMS))
    logger.set_config("v_min", String(V_MIN))
    logger.set_config("v_max", String(V_MAX))
    logger.set_config("hit_reward", String(HIT_REWARD))

    # =========================================================================
    # Train
    # =========================================================================

    print("Starting CPU training...")
    print("-" * 70)

    var start_time = perf_counter_ns()

    try:
        var _ep_returns = run_offpolicy_discrete_train[
            RainbowTrainer, PongCPU, RemoteLogger
        ](
            trainer,
            env,
            NUM_STEPS,
            print_every=20_000,
            verbose=True,
            logger=UnsafePointer(to=logger).as_unsafe_any_origin(),
            diag_every=5_000,
            eval_env=UnsafePointer(to=eval_env),
            eval_every=50_000,
            eval_episodes=3,  # each is a full episode → keep small
        )

        var elapsed_s = Float64(perf_counter_ns() - start_time) / 1e9
        logger.close()

        print("-" * 70)
        print()
        print("=" * 70)
        print("Rainbow CPU Training Complete")
        print("=" * 70)
        print("Total transitions:", NUM_STEPS)
        print("Training time:", String(elapsed_s)[byte=:6], "seconds")
        print(
            "Transitions/second:",
            String(Float64(NUM_STEPS) / elapsed_s)[byte=:9],
        )
        print("Final mean return (last 10):", trainer.mean_return())
        print("Episodes completed:", trainer.ep_count())
        print("=" * 70)

    except e:
        print("!!! EXCEPTION CAUGHT !!!")
        print("Error:", e)
        print("!!! END EXCEPTION !!!")

    print(">>> main() completed normally <<<")
