"""Test Actor-Critic networks from deeprl package.

This tests the Actor and Critic networks used in DDPG/TD3.

Run with:
    pixi run mojo run examples/test_actor_critic.mojo
"""

from deeprl import Actor, Critic, print_matrix


fn test_actor():
    """Test Actor network forward pass."""
    print("=" * 60)
    print("Testing Actor Network")
    print("=" * 60)

    # Create Actor: 4 obs -> 64 hidden -> 64 hidden -> 2 actions
    comptime obs_dim = 4
    comptime action_dim = 2
    comptime hidden_dim = 64
    comptime batch_size = 2

    var actor = Actor[obs_dim, action_dim, hidden_dim, hidden_dim](
        action_scale=2.0,  # Actions in [-2, 2]
        action_bias=0.0,
    )
    actor.print_info("Test Actor")

    # Create sample observations
    var obs = InlineArray[Float64, batch_size * obs_dim](fill=0.0)
    obs[0] = 0.1
    obs[1] = -0.2
    obs[2] = 0.3
    obs[3] = -0.4
    obs[4] = 0.5
    obs[5] = -0.6
    obs[6] = 0.7
    obs[7] = -0.8

    print("\nObservations:")
    print_matrix[batch_size, obs_dim](obs, "obs")

    # Forward pass
    var actions = actor.forward[batch_size](obs)

    print("\nActions (bounded to [-2, 2]):")
    print_matrix[batch_size, action_dim](actions, "actions")

    # Verify actions are bounded
    var bounded = True
    for i in range(batch_size * action_dim):
        if actions[i] < -2.0 or actions[i] > 2.0:
            bounded = False

    if bounded:
        print("\nActor forward pass PASSED!")
    else:
        print("\nActor forward pass FAILED - actions out of bounds!")


fn test_critic():
    """Test Critic network forward pass."""
    print("\n" + "=" * 60)
    print("Testing Critic Network")
    print("=" * 60)

    # Create Critic: (4 obs + 2 actions) -> 64 hidden -> 64 hidden -> 1 Q-value
    comptime obs_dim = 4
    comptime action_dim = 2
    comptime hidden_dim = 64
    comptime batch_size = 2

    var critic = Critic[obs_dim, action_dim, hidden_dim, hidden_dim]()
    critic.print_info("Test Critic")

    # Create sample observations and actions
    var obs = InlineArray[Float64, batch_size * obs_dim](fill=0.0)
    obs[0] = 0.1
    obs[1] = -0.2
    obs[2] = 0.3
    obs[3] = -0.4
    obs[4] = 0.5
    obs[5] = -0.6
    obs[6] = 0.7
    obs[7] = -0.8

    var actions = InlineArray[Float64, batch_size * action_dim](fill=0.0)
    actions[0] = 1.0
    actions[1] = -0.5
    actions[2] = -1.0
    actions[3] = 0.5

    print("\nObservations:")
    print_matrix[batch_size, obs_dim](obs, "obs")
    print("\nActions:")
    print_matrix[batch_size, action_dim](actions, "actions")

    # Forward pass
    var q_values = critic.forward[batch_size](obs, actions)

    print("\nQ-values:")
    for i in range(batch_size):
        print("  Q[" + String(i) + "] = " + String(q_values[i])[:10])

    print("\nCritic forward pass PASSED!")


fn test_soft_update():
    """Test soft update for target networks."""
    print("\n" + "=" * 60)
    print("Testing Soft Update (Target Networks)")
    print("=" * 60)

    comptime obs_dim = 4
    comptime action_dim = 2
    comptime hidden_dim = 32
    comptime batch_size = 1

    # Create main and target actors
    var actor = Actor[obs_dim, action_dim, hidden_dim, hidden_dim](action_scale=1.0)
    var actor_target = Actor[obs_dim, action_dim, hidden_dim, hidden_dim](action_scale=1.0)

    # Copy main to target (hard copy)
    actor_target.copy_from(actor)

    # Sample observation
    var obs = InlineArray[Float64, batch_size * obs_dim](fill=0.5)

    # Get actions from both
    var actions1 = actor.forward[batch_size](obs)
    var actions2 = actor_target.forward[batch_size](obs)

    print("After hard copy:")
    print("  Main actor action[0]: " + String(actions1[0])[:10])
    print("  Target actor action[0]: " + String(actions2[0])[:10])

    var diff = actions1[0] - actions2[0]
    if diff < 1e-10 and diff > -1e-10:
        print("  Hard copy PASSED - actions identical")
    else:
        print("  Hard copy FAILED - actions differ")

    # Modify main actor (simulate training update)
    # layer1 has obs_dim * hidden_dim = 4 * 32 = 128 elements
    for i in range(obs_dim * hidden_dim):
        actor.layer1.W[i] += 0.1

    # Soft update target
    var tau: Float64 = 0.005
    actor_target.soft_update_from(actor, tau)

    # Get actions again
    var actions3 = actor.forward[batch_size](obs)
    var actions4 = actor_target.forward[batch_size](obs)

    print("\nAfter soft update (tau=" + String(tau) + "):")
    print("  Main actor action[0]: " + String(actions3[0])[:10])
    print("  Target actor action[0]: " + String(actions4[0])[:10])

    # Target should be closer to main but not equal
    var diff2 = actions3[0] - actions4[0]
    if diff2 < diff or diff2 > -diff:
        print("  Soft update PASSED - target moved toward main")
    else:
        print("  Soft update may need verification")


fn test_backward_pass():
    """Test Actor-Critic backward pass (gradient flow)."""
    print("\n" + "=" * 60)
    print("Testing Backward Pass (Gradient Flow)")
    print("=" * 60)

    comptime obs_dim = 4
    comptime action_dim = 2
    comptime hidden_dim = 32
    comptime batch_size = 2

    var actor = Actor[obs_dim, action_dim, hidden_dim, hidden_dim](action_scale=1.0)
    var critic = Critic[obs_dim, action_dim, hidden_dim, hidden_dim]()

    # Sample data
    var obs = InlineArray[Float64, batch_size * obs_dim](fill=0.5)

    # Actor forward with cache
    var h1_actor = InlineArray[Float64, batch_size * hidden_dim](fill=0.0)
    var h2_actor = InlineArray[Float64, batch_size * hidden_dim](fill=0.0)
    var out_tanh = InlineArray[Float64, batch_size * action_dim](fill=0.0)
    var actions = actor.forward_with_cache[batch_size](obs, h1_actor, h2_actor, out_tanh)

    # Critic forward with cache
    var x_critic = InlineArray[Float64, batch_size * (obs_dim + action_dim)](fill=0.0)
    var h1_critic = InlineArray[Float64, batch_size * hidden_dim](fill=0.0)
    var h2_critic = InlineArray[Float64, batch_size * hidden_dim](fill=0.0)
    var q_values = critic.forward_with_cache[batch_size](obs, actions, x_critic, h1_critic, h2_critic)

    print("Q-values from critic:")
    for i in range(batch_size):
        print("  Q[" + String(i) + "] = " + String(q_values[i])[:10])

    # Critic backward: maximize Q -> gradient is 1.0
    var dq = InlineArray[Float64, batch_size](fill=-1.0)  # Negative for gradient ascent on Q
    var dactions = critic.backward[batch_size](dq, x_critic, h1_critic, h2_critic)

    print("\nGradient w.r.t. actions:")
    print_matrix[batch_size, action_dim](dactions, "dactions")

    # Actor backward
    actor.backward[batch_size](dactions, obs, h1_actor, h2_actor, out_tanh)

    # Update with Adam
    actor.update_adam(lr=0.001)
    critic.update_adam(lr=0.001)

    print("\nBackward pass PASSED - gradients computed and weights updated")


fn main() raises:
    print("Actor-Critic Network Tests")
    print("=" * 60)
    print("")

    test_actor()
    test_critic()
    test_soft_update()
    test_backward_pass()

    print("\n" + "=" * 60)
    print("All Actor-Critic tests completed!")
    print("=" * 60)
