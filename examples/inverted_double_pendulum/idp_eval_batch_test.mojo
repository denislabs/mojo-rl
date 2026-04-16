"""IDP: Test evaluate_gpu with different N_EVAL_ENVS to diagnose batch-size issue.

Run with:
    pixi run -e apple mojo run -I . examples/inverted_double_pendulum/idp_eval_batch_test.mojo
    pixi run -e nvidia mojo run -I . examples/inverted_double_pendulum/idp_eval_batch_test.mojo
"""

from std.random import seed
from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.envs.inverted_double_pendulum import InvertedDoublePendulum
from mojo_rl.core.logger import NoOpLogger


comptime OBS_DIM = 9
comptime ACTION_DIM = 1
comptime HIDDEN_DIM = 128
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 256

comptime GPU_DTYPE = DType.float32
comptime TERMINATE_ON_UNHEALTHY = True
comptime EVAL_EPISODES = 128
comptime MAX_STEPS = 100000  # Large enough to not be the bottleneck

# Agent type needs max_n_envs >= largest N_EVAL_ENVS we'll test
comptime AgentType = DeepSACAgent[
    OBS_DIM, ACTION_DIM, HIDDEN_DIM, BUFFER_CAPACITY, BATCH_SIZE,
    0.0003, 0.001, 0, NoOpLogger, 64,
]


def main() raises:
    seed(42)
    print("=" * 80)
    print("IDP: evaluate_gpu batch size comparison")
    print("=" * 80)
    print()

    var ctx = DeviceContext()

    var agent = AgentType(
        gamma=0.99, tau=0.005, action_scale=1.0,
        alpha=0.2, auto_alpha=False, target_entropy=-1.0,
    )
    agent.load_checkpoint("sac_inverted_double_pendulum.ckpt")
    print("Loaded checkpoint")
    print()

    # Also do a CPU eval for reference
    var cpu_env = InvertedDoublePendulum[
        DType.float64, TERMINATE_ON_UNHEALTHY=TERMINATE_ON_UNHEALTHY
    ]()
    print("CPU eval (f64, " + String(EVAL_EPISODES) + " episodes)...")
    var cpu_reward = agent.evaluate(
        cpu_env,
        num_episodes=EVAL_EPISODES,
        max_steps_per_episode=1000,
    )
    print("  CPU avg reward: " + String(cpu_reward)[byte=:12])
    print()

    # Test with N_EVAL_ENVS = 1
    print("GPU eval N_EVAL_ENVS=1 (" + String(EVAL_EPISODES) + " episodes, max_steps=" + String(MAX_STEPS) + ")...")
    var gpu1 = agent.evaluate_gpu[
        InvertedDoublePendulum[GPU_DTYPE, TERMINATE_ON_UNHEALTHY=TERMINATE_ON_UNHEALTHY],
        N_EVAL_ENVS=1,
    ](ctx, num_episodes=EVAL_EPISODES, max_steps=MAX_STEPS, stochastic=False)
    print("  GPU N=1 avg reward: " + String(gpu1)[byte=:12])
    print()

    # Test with N_EVAL_ENVS = 4
    print("GPU eval N_EVAL_ENVS=4 (" + String(EVAL_EPISODES) + " episodes, max_steps=" + String(MAX_STEPS) + ")...")
    var gpu4 = agent.evaluate_gpu[
        InvertedDoublePendulum[GPU_DTYPE, TERMINATE_ON_UNHEALTHY=TERMINATE_ON_UNHEALTHY],
        N_EVAL_ENVS=4,
    ](ctx, num_episodes=EVAL_EPISODES, max_steps=MAX_STEPS, stochastic=False)
    print("  GPU N=4 avg reward: " + String(gpu4)[byte=:12])
    print()

    # Test with N_EVAL_ENVS = 32
    print("GPU eval N_EVAL_ENVS=32 (" + String(EVAL_EPISODES) + " episodes, max_steps=" + String(MAX_STEPS) + ")...")
    var gpu32 = agent.evaluate_gpu[
        InvertedDoublePendulum[GPU_DTYPE, TERMINATE_ON_UNHEALTHY=TERMINATE_ON_UNHEALTHY],
        N_EVAL_ENVS=32,
    ](ctx, num_episodes=EVAL_EPISODES, max_steps=MAX_STEPS, stochastic=False)
    print("  GPU N=32 avg reward: " + String(gpu32)[byte=:12])
    print()

    print("-" * 80)
    print("Summary:")
    print("  CPU f64:      " + String(cpu_reward)[byte=:12])
    print("  GPU f32 N=1:  " + String(gpu1)[byte=:12])
    print("  GPU f32 N=4:  " + String(gpu4)[byte=:12])
    print("  GPU f32 N=32: " + String(gpu32)[byte=:12])
    print("-" * 80)

    if gpu1 > 0.0 and gpu32 / gpu1 < 0.1:
        print("BUG CONFIRMED: N_EVAL_ENVS scaling dramatically affects reward")
    elif gpu1 < cpu_reward * 0.5:
        print("GPU f32 significantly worse than CPU f64 (chaotic system precision issue)")
    else:
        print("All looks reasonable")

    cpu_env.close()
