"""Checkpoint Demo - Save and load neural network and agent state.

This example demonstrates:
1. Saving/loading Network parameters
2. Saving/loading DQN agent state (online network, target network, hyperparameters)

Usage:
    pixi run mojo run examples/checkpoint_demo.mojo
"""

from deep_rl import (
    seq,
    Linear,
    ReLU,
    Adam,
    Kaiming,
    Network,
)
from deep_agents.dqn import DQNAgent


fn demo_network_checkpoint() raises:
    """Demonstrate Network checkpoint save/load."""
    print("=== Network Checkpoint Demo ===\n")

    # Create a simple neural network: 4 -> 8 (ReLU) -> 2
    var model = seq(
        Linear[4, 8](),
        ReLU[8](),
        Linear[8, 2](),
    )
    var network = Network(model, Adam(lr=0.001), Kaiming())

    print("Original network params (first 5):")
    for i in range(5):
        print("  params[" + String(i) + "] = " + String(Float64(network.params[i])))

    # Save checkpoint
    network.save_checkpoint("network_checkpoint.ckpt")
    print("\nCheckpoint saved to network_checkpoint.ckpt")

    # Create a new network (will have different random initialization)
    var loaded_network = Network(model, Adam(lr=0.001), Kaiming())
    print("\nNew network params before load (first 5):")
    for i in range(5):
        print("  params[" + String(i) + "] = " + String(Float64(loaded_network.params[i])))

    # Load checkpoint
    loaded_network.load_checkpoint("network_checkpoint.ckpt")
    print("\nNew network params after load (first 5):")
    for i in range(5):
        print("  params[" + String(i) + "] = " + String(Float64(loaded_network.params[i])))

    # Verify
    var all_match = True
    for i in range(network.PARAM_SIZE):
        if network.params[i] != loaded_network.params[i]:
            all_match = False
            break

    if all_match:
        print("\nVerification: All " + String(network.PARAM_SIZE) + " parameters match!")
    else:
        print("\nVerification FAILED: Parameters don't match!")


fn demo_dqn_checkpoint() raises:
    """Demonstrate DQN agent checkpoint save/load."""
    print("\n\n=== DQN Agent Checkpoint Demo ===\n")

    # Create DQN agent (obs_dim=4, num_actions=2, hidden=32, buffer=1000, batch=32)
    var agent = DQNAgent[4, 2, 32, 1000, 32](
        gamma=0.99,
        tau=0.005,
        lr=0.001,
        epsilon=0.5,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    )

    print("Original agent state:")
    print("  gamma = " + String(agent.gamma))
    print("  epsilon = " + String(agent.epsilon))
    print("  train_step_count = " + String(agent.train_step_count))
    print("  online_params[0] = " + String(Float64(agent.online_model.params[0])))
    print("  target_params[0] = " + String(Float64(agent.target_model.params[0])))

    # Save checkpoint (single file)
    agent.save_checkpoint("dqn_checkpoint.ckpt")
    print("\nCheckpoint saved to dqn_checkpoint.ckpt")

    # Create a new agent with different hyperparameters
    var loaded_agent = DQNAgent[4, 2, 32, 1000, 32](
        gamma=0.95,      # Different
        epsilon=1.0,     # Different
    )

    print("\nNew agent state before load:")
    print("  gamma = " + String(loaded_agent.gamma))
    print("  epsilon = " + String(loaded_agent.epsilon))
    print("  online_params[0] = " + String(Float64(loaded_agent.online_model.params[0])))

    # Load checkpoint
    loaded_agent.load_checkpoint("dqn_checkpoint.ckpt")

    print("\nNew agent state after load:")
    print("  gamma = " + String(loaded_agent.gamma))
    print("  epsilon = " + String(loaded_agent.epsilon))
    print("  train_step_count = " + String(loaded_agent.train_step_count))
    print("  online_params[0] = " + String(Float64(loaded_agent.online_model.params[0])))
    print("  target_params[0] = " + String(Float64(loaded_agent.target_model.params[0])))

    # Verify
    var params_match = True
    for i in range(agent.NETWORK_PARAM_SIZE):
        if agent.online_model.params[i] != loaded_agent.online_model.params[i]:
            params_match = False
            break
        if agent.target_model.params[i] != loaded_agent.target_model.params[i]:
            params_match = False
            break

    var hyperparams_match = (
        agent.gamma == loaded_agent.gamma
        and agent.epsilon == loaded_agent.epsilon
        and agent.tau == loaded_agent.tau
    )

    if params_match and hyperparams_match:
        print("\nVerification: All network params and hyperparameters match!")
    else:
        if not params_match:
            print("\nVerification FAILED: Network parameters don't match!")
        if not hyperparams_match:
            print("\nVerification FAILED: Hyperparameters don't match!")


fn main() raises:
    print("Checkpoint Demo - mojo-rl\n")
    print("This demo shows how to save and load model checkpoints.\n")

    demo_network_checkpoint()
    demo_dqn_checkpoint()

    print("\n\n=== Demo Complete ===")
    print("\nCheckpoint files created:")
    print("  - network_checkpoint.ckpt (Network params + optimizer state)")
    print("  - dqn_checkpoint.ckpt (DQN: both networks + hyperparameters)")
