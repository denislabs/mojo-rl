"""Test DDPG Agent from deeprl package.

Run with:
    pixi run mojo run examples/test_ddpg.mojo
"""

from random import random_float64
from deeprl.ddpg import DDPGAgent


fn test_ddpg_basic():
    """Test basic DDPG agent operations."""
    print("=" * 60)
    print("Testing DDPG Agent")
    print("=" * 60)

    # Create DDPG agent for simple environment
    # obs_dim=3, action_dim=1 (like simplified Pendulum)
    comptime obs_dim = 3
    comptime action_dim = 1
    comptime hidden_dim = 64
    comptime buffer_capacity = 1000
    comptime batch_size = 32

    var agent = DDPGAgent[obs_dim, action_dim, hidden_dim, buffer_capacity, batch_size](
        gamma=0.99,
        tau=0.005,
        actor_lr=0.001,
        critic_lr=0.001,
        noise_std=0.1,
        action_scale=2.0,  # Actions in [-2, 2]
    )

    agent.print_info()

    # Test action selection
    print("\n--- Testing Action Selection ---")
    var obs = InlineArray[Float64, obs_dim](fill=0.0)
    obs[0] = 0.5   # cos(theta)
    obs[1] = 0.866 # sin(theta)
    obs[2] = 0.1   # angular velocity

    var action = agent.select_action(obs, add_noise=True)
    print("Observation: [" + String(obs[0])[:6] + ", " + String(obs[1])[:6] + ", " + String(obs[2])[:6] + "]")
    print("Action (with noise): " + String(action[0])[:8])

    var action_no_noise = agent.select_action(obs, add_noise=False)
    print("Action (no noise): " + String(action_no_noise[0])[:8])

    # Verify action is bounded
    if action[0] >= -2.0 and action[0] <= 2.0:
        print("Action bounds check PASSED!")
    else:
        print("Action bounds check FAILED!")

    # Test storing transitions
    print("\n--- Testing Transition Storage ---")

    # Generate fake transitions
    for i in range(100):
        var obs_i = InlineArray[Float64, obs_dim](fill=0.0)
        var next_obs_i = InlineArray[Float64, obs_dim](fill=0.0)
        var action_i = InlineArray[Float64, action_dim](fill=0.0)

        # Random observations
        for j in range(obs_dim):
            obs_i[j] = random_float64() * 2.0 - 1.0
            next_obs_i[j] = random_float64() * 2.0 - 1.0

        # Random action
        action_i[0] = random_float64() * 4.0 - 2.0

        # Random reward
        var reward = random_float64() * 2.0 - 1.0

        # Random done
        var done = random_float64() < 0.1

        agent.store_transition(obs_i, action_i, reward, next_obs_i, done)

    print("Stored 100 transitions")
    print("Total steps: " + String(agent.total_steps))
    print("Buffer ready: " + String(agent.buffer.is_ready[batch_size]()))

    # Test training step
    print("\n--- Testing Training Step ---")

    var critic_loss = agent.train_step()
    print("Training step 1 - Critic loss: " + String(critic_loss)[:10])

    # Do a few more training steps
    for step in range(10):
        critic_loss = agent.train_step()

    print("After 10 more steps - Critic loss: " + String(critic_loss)[:10])

    print("\nDDPG basic test PASSED!")


fn test_ddpg_learning():
    """Test that DDPG can learn on simple fake data."""
    print("\n" + "=" * 60)
    print("Testing DDPG Learning Dynamics")
    print("=" * 60)

    comptime obs_dim = 2
    comptime action_dim = 1
    comptime hidden_dim = 32
    comptime buffer_capacity = 500
    comptime batch_size = 16

    var agent = DDPGAgent[obs_dim, action_dim, hidden_dim, buffer_capacity, batch_size](
        gamma=0.99,
        tau=0.01,  # Faster target updates for test
        actor_lr=0.001,
        critic_lr=0.001,
        noise_std=0.2,
        action_scale=1.0,
    )

    # Fill buffer with simple pattern:
    # Observation = [x, 0], optimal action = x (to reach goal at x=0)
    # Reward = -x^2 (maximize when at center)
    print("Filling buffer with structured transitions...")

    for i in range(200):
        var x = random_float64() * 2.0 - 1.0  # x in [-1, 1]

        var obs = InlineArray[Float64, obs_dim](fill=0.0)
        obs[0] = x
        obs[1] = 0.0

        var action = InlineArray[Float64, action_dim](fill=0.0)
        action[0] = -x * 0.5  # Move towards center

        var next_x = x + action[0] * 0.1
        var next_obs = InlineArray[Float64, obs_dim](fill=0.0)
        next_obs[0] = next_x
        next_obs[1] = 0.0

        # Reward is higher when closer to center
        var reward = -x * x

        var done = (next_x > -0.1 and next_x < 0.1)

        agent.store_transition(obs, action, reward, next_obs, done)

    print("Buffer size: " + String(agent.buffer.len()))

    # Train for several steps
    print("\nTraining...")
    var total_loss: Float64 = 0.0
    for step in range(50):
        var loss = agent.train_step()
        total_loss += loss
        if (step + 1) % 10 == 0:
            print("  Step " + String(step + 1) + " - Avg loss: " + String(total_loss / 10.0)[:10])
            total_loss = 0.0

    # Test policy at different states
    print("\nTesting learned policy:")
    var test_xs = List[Float64]()
    test_xs.append(-0.8)
    test_xs.append(-0.4)
    test_xs.append(0.0)
    test_xs.append(0.4)
    test_xs.append(0.8)

    for i in range(len(test_xs)):
        var test_obs = InlineArray[Float64, obs_dim](fill=0.0)
        test_obs[0] = test_xs[i]
        test_obs[1] = 0.0

        var test_action = agent.select_action(test_obs, add_noise=False)
        print("  x=" + String(test_xs[i])[:5] + " -> action=" + String(test_action[0])[:8])

    print("\nDDPG learning dynamics test completed!")


fn main() raises:
    print("DDPG Agent Tests")
    print("=" * 60)
    print("")

    test_ddpg_basic()
    test_ddpg_learning()

    print("\n" + "=" * 60)
    print("All DDPG tests completed!")
    print("=" * 60)
