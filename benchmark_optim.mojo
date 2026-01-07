"""Benchmark CartPoleEnv and QLearningAgent performance."""

from time import perf_counter_ns
from random import seed, random_float64
from envs import CartPoleEnv
from agents import QLearningAgent


fn benchmark_cartpole_step(num_steps: Int) raises -> Float64:
    """Benchmark CartPoleEnv.step_raw performance."""
    var env = CartPoleEnv()
    _ = env.reset()
    
    var start = perf_counter_ns()
    for _ in range(num_steps):
        var result = env.step_raw(Int(random_float64() * 2.0))
        if result[2]:  # done
            _ = env.reset()
    var end = perf_counter_ns()
    
    return Float64(num_steps) / (Float64(end - start) / 1_000_000_000.0)


fn benchmark_qtable_ops(num_ops: Int) raises -> Tuple[Float64, Float64, Float64]:
    """Benchmark QTable get/set/get_best_action operations."""
    var agent = QLearningAgent(num_states=10000, num_actions=2)
    
    # Benchmark get
    var start = perf_counter_ns()
    var dummy: Float64 = 0.0
    for i in range(num_ops):
        dummy += agent.q_table.get(i % 10000, i % 2)
    var end = perf_counter_ns()
    var get_rate = Float64(num_ops) / (Float64(end - start) / 1_000_000_000.0)
    
    # Benchmark set
    start = perf_counter_ns()
    for i in range(num_ops):
        agent.q_table.set(i % 10000, i % 2, Float64(i))
    end = perf_counter_ns()
    var set_rate = Float64(num_ops) / (Float64(end - start) / 1_000_000_000.0)
    
    # Benchmark get_best_action
    start = perf_counter_ns()
    var dummy_action = 0
    for i in range(num_ops):
        dummy_action += agent.q_table.get_best_action(i % 10000)
    end = perf_counter_ns()
    var best_rate = Float64(num_ops) / (Float64(end - start) / 1_000_000_000.0)
    
    # Use dummy values to prevent optimization
    if dummy < -1e10:
        print(dummy)
    if dummy_action < -1000000:
        print(dummy_action)
    
    return (get_rate, set_rate, best_rate)


fn benchmark_training(num_episodes: Int) raises -> Float64:
    """Benchmark full training loop."""
    var env = CartPoleEnv()
    var agent = QLearningAgent(
        num_states=env.num_states(),
        num_actions=2,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
    )
    
    var start = perf_counter_ns()
    _ = agent.train(env, num_episodes=num_episodes, max_steps_per_episode=500)
    var end = perf_counter_ns()
    
    return Float64(end - start) / 1_000_000_000.0


fn main() raises:
    seed()
    
    print("=" * 60)
    print("Performance Benchmark (BEFORE optimization)")
    print("=" * 60)
    print()
    
    # Warmup
    _ = benchmark_cartpole_step(1000)
    _ = benchmark_qtable_ops(1000)
    
    # CartPole step benchmark
    print("CartPoleEnv.step_raw:")
    var step_rate = benchmark_cartpole_step(100000)
    print("  ", Int(step_rate), "steps/sec")
    print()
    
    # QTable operations benchmark
    print("QTable operations (10000 states, 2 actions):")
    var qtable_rates = benchmark_qtable_ops(1000000)
    print("  get():           ", Int(qtable_rates[0]), "ops/sec")
    print("  set():           ", Int(qtable_rates[1]), "ops/sec")
    print("  get_best_action():", Int(qtable_rates[2]), "ops/sec")
    print()
    
    # Full training benchmark
    print("Full training (100 episodes, CartPole):")
    var train_time = benchmark_training(100)
    print("  Time:", train_time, "seconds")
    print("  Rate:", Int(100.0 / train_time), "episodes/sec")
