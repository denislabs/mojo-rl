"""Train Deep DQN on LunarLander using GPU V4 with Domain Randomization.

Run with: pixi run -e apple mojo run examples/lunar_lander_dqn_gpu_v4.mojo

This trains a Deep Q-Network using the GPU V4 LunarLander environment which
includes domain randomization for better sim-to-sim transfer:
- Gravity variation (±10%)
- Engine power variation (±10%)
- Mass/inertia variation (±5%)
- Observation noise (±2%)
- Contact threshold variation (±20%)

The domain randomization helps policies trained on GPU transfer better to
the CPU Box2D environment by learning to handle physics variations.

Expected performance:
- Episode 100: ~-200 avg reward (still learning)
- Episode 300: ~-50 avg reward (improving)
- Episode 500: ~100+ avg reward (solving)

After GPU training, the policy is evaluated on the CPU environment
to measure transfer quality.
"""

from random import seed

from gpu.host import DeviceContext

from envs.lunar_lander import LunarLanderEnv
from envs.lunar_lander_gpu_v4 import LunarLanderGPUv4
from deep_agents import DQNAgent


fn main() raises:
    print("=" * 70)
    print("Deep DQN on LunarLander GPU V4 (Domain Randomization)")
    print("=" * 70)
    print()
    print("GPU V4 Features:")
    print("  - Gravity variation: ±10%")
    print("  - Engine power variation: ±10%")
    print("  - Mass/inertia variation: ±5%")
    print("  - Observation noise: ±2%")
    print("  - Contact threshold variation: ±20%")
    print()
    print("Training on GPU with domain randomization...")
    print("Policies should transfer better to CPU environment.")
    print()

    # Seed for reproducibility
    seed(42)

    # Create GPU context
    var ctx = DeviceContext()

    # Create GPU V4 environment with domain randomization
    var gpu_env = LunarLanderGPUv4()

    # Create CPU environment for transfer evaluation
    # Using float64 to match DQN agent's internal dtype for evaluation
    var cpu_env = LunarLanderEnv[DType.float64](continuous=False, enable_wind=False)

    # Create DQN agent with tuned hyperparameters
    # LunarLander: 8D observations, 4 discrete actions
    #
    # Hyperparameters tuned for GPU training with domain randomization:
    # - hidden_dim=128: Good capacity for LunarLander
    # - lr=5e-4: Stable learning rate
    # - gamma=0.99: Standard discount factor
    # - epsilon_decay=0.995: Slightly faster decay (domain randomization helps explore)
    # - tau=0.005: Standard target update rate
    # - Double DQN enabled (default) for reduced overestimation
    var agent = DQNAgent[
        obs_dim=8,
        num_actions=4,
        hidden_dim=128,
        buffer_capacity=20000,
        batch_size=64,
    ](
        gamma=0.99,  # Standard discount
        tau=0.005,  # Standard target update
        lr=0.0005,  # Stable learning rate (5e-4)
        epsilon=1.0,  # Start with full exploration
        epsilon_min=0.01,  # Low minimum
        epsilon_decay=0.995,  # Slightly faster decay with domain randomization
    )

    # Train on GPU with domain randomization
    print("-" * 70)
    print("PHASE 1: GPU Training with Domain Randomization")
    print("-" * 70)
    print()

    var metrics = agent.train_gpu(
        ctx,
        gpu_env,
        num_episodes=600,  # Sufficient for convergence
        max_steps_per_episode=1000,
        warmup_steps=5000,  # Warmup for diverse experiences
        train_every=4,  # Standard DQN training frequency
        verbose=True,
        print_every=25,
        environment_name="LunarLander-GPU-V4",
    )

    # Evaluate on CPU environment to test transfer
    # This is the key metric - how well does GPU-trained policy work on CPU?
    print()
    print("-" * 70)
    print("PHASE 2: CPU Transfer Evaluation")
    print("-" * 70)
    print()
    print("Testing policy transfer from GPU (simplified physics) to CPU (Box2D)...")
    print()

    var cpu_eval_reward = agent.evaluate_greedy(
        cpu_env,
        num_episodes=20,
        max_steps=1000,
    )
    print()
    print("Average CPU evaluation reward: " + String(cpu_eval_reward)[:10])

    # Interpretation
    print()
    print("=" * 70)
    print("TRANSFER QUALITY ASSESSMENT")
    print("=" * 70)
    print()
    print("CPU evaluation reward: " + String(cpu_eval_reward)[:10])
    print()

    if cpu_eval_reward > 200:
        print("EXCELLENT: Policy solves CPU environment (reward > 200)!")
        print("Domain randomization successfully enabled transfer.")
    elif cpu_eval_reward > 100:
        print("GOOD: Policy performs well on CPU environment.")
        print("Transfer is working - policy learned robust behaviors.")
    elif cpu_eval_reward > 0:
        print("MODERATE: Policy shows some transfer capability.")
        print("Consider more training or tuning randomization parameters.")
    else:
        print("POOR: Policy struggles on CPU environment.")
        print("May need more diverse training or different approach.")

    print()
    print("Domain randomization helps by training the policy to handle:")
    print("  - Physics parameter variations (gravity, engine power, mass)")
    print("  - Observation noise (sensor uncertainty)")
    print("  - Contact detection variations")
    print()

    cpu_env.close()
    print()
    print("Training complete!")
