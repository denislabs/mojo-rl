"""Composable GPU RL Example.

This example demonstrates the composable GPU RL framework where:
1. Environments implement the GPUEnv trait
2. Algorithms are generic over any GPUEnv
3. Training is done by passing the environment type to the algorithm

Usage:
    pixi run -e apple mojo run examples/composable_gpu_rl.mojo
"""

from random import seed

from gpu.host import DeviceContext

from envs.cartpole_gpu import GPUCartPole
from deep_agents.gpu import train_reinforce, TrainResult


def main():
    seed(42)

    print("=" * 70)
    print("Composable GPU RL Framework Demo")
    print("=" * 70)
    print()
    print("This demonstrates the composable design:")
    print("  - GPUCartPole implements GPUEnv trait")
    print("  - train_reinforce[Env: GPUEnv] works with any GPUEnv")
    print("  - Just pass the environment type to train!")
    print()

    with DeviceContext() as ctx:
        # Train CartPole with REINFORCE - just pass the type!
        print("Training CartPole with REINFORCE...")
        print("-" * 50)

        var result = train_reinforce[
            GPUCartPole,  # Environment type
            HIDDEN_DIM=32,  # Policy hidden layer size
            NUM_ENVS=1024,  # Parallel environments
            STEPS_PER_KERNEL=200,  # Steps per kernel launch
        ](
            ctx,
            num_updates=100,
            lr=Float32(0.01),
            gamma=Float32(0.99),
            verbose=True,
        )

        print()
        print("=" * 70)
        print("Summary")
        print("=" * 70)
        print()
        print("Composable GPU RL Framework:")
        print("  1. Define environment: struct MyEnv(GPUEnvDims)")
        print("  2. Implement step/reset/get_obs methods")
        print("  3. Train: train_reinforce[MyEnv](...)")
        print()
        print("Adding a new environment is simple:")
        print("  - Create a new struct implementing GPUEnvDims")
        print("  - Provide OBS_DIM, NUM_ACTIONS, STATE_SIZE comptime values")
        print("  - Implement step, reset, get_obs static methods")
        print("  - The same REINFORCE code works automatically!")
        print()
        print("Final Results:")
        print("  Total steps: " + String(result.total_steps))
        print("  Total episodes: " + String(result.total_episodes))
        print("  Avg episode length: " + String(result.final_avg_ep_length)[:6])
        print(
            "  Throughput: " + String(Int(result.steps_per_sec)) + " steps/sec"
        )
        print("=" * 70)
