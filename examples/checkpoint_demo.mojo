"""Checkpoint Demo - Save and load neural network and agent state.

This example demonstrates:
1. Saving/loading NetworkState parameters (new trait-based API)
2. Saving/loading DQN agent state (online network, target network, hyperparameters)

Usage:
    pixi run mojo run examples/checkpoint_demo.mojo
"""

from nn import Linear, ReLU, Adam, Kaiming
from nn.model import Sequential
from nn.training import NetworkState
from deep_agents.dqn import DQNAgent


fn demo_network_checkpoint() raises:
    """Demonstrate NetworkState checkpoint save/load."""
    print("=== NetworkState Checkpoint Demo ===\n")

    # Build model type: 4 -> 8 (ReLU) -> 2
    alias DemoModel = Sequential[Linear[4, 8], ReLU[8], Linear[8, 2]]
    alias DemoOpt = Adam[0.001]
    alias DemoState = NetworkState[DemoModel, DemoOpt]

    # Create and initialize a NetworkState
    var state = DemoState()
    state.initialize[Kaiming[]]()

    print("Original params (first 5):")
    for i in range(5):
        print("  params[" + String(i) + "] = " + String(Float64((state.params + i)[])))

    # Save checkpoint
    state.save_checkpoint("network_checkpoint.ckpt")
    print("\nCheckpoint saved to network_checkpoint.ckpt")

    # Create a new state (different random init)
    var loaded = DemoState()
    loaded.initialize[Kaiming[]]()
    print("\nNew state params before load (first 5):")
    for i in range(5):
        print("  params[" + String(i) + "] = " + String(Float64((loaded.params + i)[])))

    # Load checkpoint
    loaded.load_checkpoint("network_checkpoint.ckpt")
    print("\nNew state params after load (first 5):")
    for i in range(5):
        print("  params[" + String(i) + "] = " + String(Float64((loaded.params + i)[])))

    # Verify
    var all_match = True
    for i in range(DemoState.PARAM_SIZE):
        if (state.params + i)[] != (loaded.params + i)[]:
            all_match = False
            break

    if all_match:
        print(
            "\nVerification: All "
            + String(DemoState.PARAM_SIZE)
            + " parameters match!"
        )
    else:
        print("\nVerification FAILED: Parameters don't match!")


fn demo_dqn_checkpoint() raises:
    """Demonstrate DQN agent checkpoint save/load."""
    print("\n\n=== DQN Agent Checkpoint Demo ===\n")

    # lr is now compile-time; obs_dim=4, num_actions=2, hidden=32, buffer=1000, batch=32
    var agent = DQNAgent[4, 2, 32, 1000, 32, lr=0.001](
        gamma=0.99,
        tau=0.005,
        epsilon=0.5,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    )

    print("Original agent state:")
    print("  gamma = " + String(agent.gamma))
    print("  epsilon = " + String(agent.epsilon))
    print("  train_step_count = " + String(agent.train_step_count))
    print("  online.params[0] = " + String(Float64(agent.online.params[])))
    print("  target.params[0] = " + String(Float64(agent.target.params[])))

    # Save checkpoint (single file)
    agent.save_checkpoint("dqn_checkpoint.ckpt")
    print("\nCheckpoint saved to dqn_checkpoint.ckpt")

    # Create a new agent with different hyperparameters
    var loaded_agent = DQNAgent[4, 2, 32, 1000, 32, lr=0.001](
        gamma=0.95,  # Different
        epsilon=1.0,  # Different
    )

    print("\nNew agent state before load:")
    print("  gamma = " + String(loaded_agent.gamma))
    print("  epsilon = " + String(loaded_agent.epsilon))
    print("  online.params[0] = " + String(Float64(loaded_agent.online.params[])))

    # Load checkpoint
    loaded_agent.load_checkpoint("dqn_checkpoint.ckpt")

    print("\nNew agent state after load:")
    print("  gamma = " + String(loaded_agent.gamma))
    print("  epsilon = " + String(loaded_agent.epsilon))
    print("  train_step_count = " + String(loaded_agent.train_step_count))
    print("  online.params[0] = " + String(Float64(loaded_agent.online.params[])))
    print("  target.params[0] = " + String(Float64(loaded_agent.target.params[])))

    # Verify
    alias PSIZE = DQNAgent[4, 2, 32, 1000, 32, lr=0.001].Q_Model.PARAM_SIZE
    var params_match = True
    for i in range(PSIZE):
        if (agent.online.params + i)[] != loaded_(agent.online.params + i)[]:
            params_match = False
            break
        if (agent.target.params + i)[] != loaded_(agent.target.params + i)[]:
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
    print("  - network_checkpoint.ckpt (NetworkState params + optimizer state)")
    print("  - dqn_checkpoint.ckpt (DQN: both networks + hyperparameters)")
