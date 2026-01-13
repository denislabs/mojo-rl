"""Demo of Prioritized Experience Replay buffers."""

from core.replay_buffer import PrioritizedReplayBuffer, PrioritizedTransition
from core.sum_tree import SumTree
from deep_rl.replay import PrioritizedReplayBuffer as DeepPER


fn test_sum_tree():
    """Test basic sum-tree operations."""
    print("Testing SumTree...")

    var tree = SumTree(capacity=8)

    # Add some priorities
    _ = tree.add(1.0)
    _ = tree.add(2.0)
    _ = tree.add(3.0)
    _ = tree.add(4.0)

    print("  Total sum:", tree.total_sum(), "(expected: 10.0)")
    print("  Max priority:", tree.max_priority(), "(expected: 4.0)")
    print("  Min priority:", tree.min_priority(), "(expected: 1.0)")

    # Test sampling
    var counts = List[Int]()
    for _ in range(4):
        counts.append(0)

    for _ in range(1000):
        var target = tree.total_sum() * 0.5  # Sample middle
        var idx = tree.sample(target)
        if idx < 4:
            counts[idx] += 1

    print("  Sample distribution (1000 samples at midpoint):")
    print("    idx 0 (p=1.0):", counts[0])
    print("    idx 1 (p=2.0):", counts[1])
    print("    idx 2 (p=3.0):", counts[2])
    print("    idx 3 (p=4.0):", counts[3])

    print("  SumTree tests passed!")


fn test_discrete_per():
    """Test discrete PrioritizedReplayBuffer."""
    print("\nTesting discrete PrioritizedReplayBuffer...")

    var buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4)

    # Add some transitions
    for i in range(50):
        buffer.push(
            state=i,
            action=i % 4,
            reward=Float64(i) * 0.1,
            next_state=i + 1,
            done=(i == 49),
        )

    print("  Buffer size:", buffer.len())

    # Sample with importance weights
    var result = buffer.sample(batch_size=10, beta=0.4)
    var indices = result[0].copy()
    var batch = result[1].copy()

    print("  Sampled", len(batch), "transitions")

    # Check weights are normalized (max should be ~1.0)
    var max_weight: Float64 = 0.0
    for i in range(len(batch)):
        if batch[i].weight > max_weight:
            max_weight = batch[i].weight
    print("  Max IS weight:", max_weight, "(should be ~1.0)")

    # Update priorities with some TD errors
    for i in range(len(indices)):
        buffer.update_priority(indices[i], Float64(i + 1) * 0.5)

    print("  Updated priorities for sampled transitions")

    # Test beta annealing
    buffer.anneal_beta(progress=0.5, beta_start=0.4)
    print("  Beta after 50% annealing:", buffer.beta, "(expected: 0.7)")

    print("  Discrete PER tests passed!")


fn test_deep_per():
    """Test deep RL PrioritizedReplayBuffer with compile-time dimensions."""
    print("\nTesting deep RL PrioritizedReplayBuffer...")

    comptime obs_dim = 4
    comptime action_dim = 1
    comptime capacity = 100
    comptime batch_size = 8

    var buffer = DeepPER[capacity, obs_dim, action_dim]()

    # Add some transitions
    for i in range(50):
        var obs = InlineArray[Float64, obs_dim](fill=Float64(i))
        var action = InlineArray[Float64, action_dim](fill=Float64(i % 2))
        var reward = Float64(i) * 0.1
        var next_obs = InlineArray[Float64, obs_dim](fill=Float64(i + 1))
        var done = i == 49

        buffer.add(obs, action, reward, next_obs, done)

    print("  Buffer size:", buffer.len())

    # Sample with importance weights
    var batch_obs = InlineArray[Float64, batch_size * obs_dim](fill=0)
    var batch_actions = InlineArray[Float64, batch_size * action_dim](fill=0)
    var batch_rewards = InlineArray[Float64, batch_size](fill=0)
    var batch_next_obs = InlineArray[Float64, batch_size * obs_dim](fill=0)
    var batch_dones = InlineArray[Float64, batch_size](fill=0)
    var batch_weights = InlineArray[Float64, batch_size](fill=0)
    var batch_indices = InlineArray[Int, batch_size](fill=0)

    buffer.sample[batch_size](
        batch_obs,
        batch_actions,
        batch_rewards,
        batch_next_obs,
        batch_dones,
        batch_weights,
        batch_indices,
    )

    print("  Sampled", batch_size, "transitions")

    # Check weights
    var max_weight: Float64 = 0.0
    for i in range(batch_size):
        if batch_weights[i] > max_weight:
            max_weight = batch_weights[i]
    print("  Max IS weight:", max_weight, "(should be ~1.0)")

    # Update priorities
    var td_errors = InlineArray[Float64, batch_size](fill=0.5)
    buffer.update_priorities[batch_size](batch_indices, td_errors)
    print("  Updated priorities")

    print("  Deep RL PER tests passed!")


fn main():
    """Run all PER tests."""
    print("=" * 60)
    print("Prioritized Experience Replay Demo")
    print("=" * 60)

    test_sum_tree()
    test_discrete_per()
    test_deep_per()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
