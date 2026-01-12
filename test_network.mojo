"""Test Network wrapper for RL agents.

This tests the basic functionality of the Network struct:
- Creation with model, optimizer, initializer
- Forward pass (inference)
- Forward pass with cache (training)
- Backward pass
- Optimizer update
- Target network operations (copy_params, soft_update)

Run with:
    pixi run mojo run test_network.mojo
"""

from random import seed, random_float64

from deep_rl.constants import dtype
from deep_rl.model import Linear, ReLU, seq
from deep_rl.optimizer import Adam
from deep_rl.initializer import Kaiming
from deep_rl.training import Network


# =============================================================================
# Constants
# =============================================================================

comptime OBS_DIM = 4
comptime HIDDEN_DIM = 32
comptime NUM_ACTIONS = 2
comptime BATCH_SIZE = 8


# =============================================================================
# Main
# =============================================================================


def main():
    seed(42)
    print("=" * 70)
    print("Network Wrapper Test")
    print("=" * 70)
    print()

    # =========================================================================
    # Create Q-network model: obs -> hidden (ReLU) -> hidden (ReLU) -> actions
    # =========================================================================

    var q_model = seq(
        Linear[OBS_DIM, HIDDEN_DIM](),
        ReLU[HIDDEN_DIM](),
        Linear[HIDDEN_DIM, HIDDEN_DIM](),
        ReLU[HIDDEN_DIM](),
        Linear[HIDDEN_DIM, NUM_ACTIONS](),
    )

    print("Q-Network Architecture:")
    print("  Input dim: " + String(q_model.IN_DIM))
    print("  Output dim: " + String(q_model.OUT_DIM))
    print("  Param size: " + String(q_model.PARAM_SIZE))
    print("  Cache size: " + String(q_model.CACHE_SIZE))
    print()

    # =========================================================================
    # Create online and target networks
    # =========================================================================

    var online = Network(q_model, Adam(lr=0.001), Kaiming())
    var target = Network(q_model, Adam(lr=0.001), Kaiming())

    print("Created online and target networks")
    print()

    # =========================================================================
    # Initialize target with online's weights
    # =========================================================================

    target.copy_params_from(online)
    print("Copied online params to target")

    # Verify params are equal
    var params_match = True
    for i in range(online.PARAM_SIZE):
        if online.params[i] != target.params[i]:
            params_match = False
            break
    print("Params match after copy: " + String(params_match))
    print()

    # =========================================================================
    # Test forward pass (inference)
    # =========================================================================

    print("Testing forward pass (inference)...")

    # Create dummy input
    var obs = InlineArray[Scalar[dtype], BATCH_SIZE * OBS_DIM](
        uninitialized=True
    )
    for i in range(BATCH_SIZE * OBS_DIM):
        obs[i] = Scalar[dtype](random_float64() * 2 - 1)

    # Forward pass
    var q_values = InlineArray[Scalar[dtype], BATCH_SIZE * NUM_ACTIONS](
        uninitialized=True
    )
    online.forward[BATCH_SIZE](obs, q_values)

    print("Q-values for first sample:")
    print(
        "  Action 0: "
        + String(Float64(q_values[0]))[:8]
        + ", Action 1: "
        + String(Float64(q_values[1]))[:8]
    )
    print()

    # =========================================================================
    # Test forward with cache (training)
    # =========================================================================

    print("Testing forward pass with cache (training)...")

    var cache = InlineArray[Scalar[dtype], BATCH_SIZE * online.CACHE_SIZE](
        uninitialized=True
    )
    online.forward_with_cache[BATCH_SIZE](obs, q_values, cache)

    print("Forward with cache completed")
    print()

    # =========================================================================
    # Test backward pass
    # =========================================================================

    print("Testing backward pass...")

    # Create dummy gradient (as if from loss)
    var grad_output = InlineArray[Scalar[dtype], BATCH_SIZE * NUM_ACTIONS](
        uninitialized=True
    )
    for i in range(BATCH_SIZE * NUM_ACTIONS):
        grad_output[i] = Scalar[dtype](0.1)

    var grad_input = InlineArray[Scalar[dtype], BATCH_SIZE * OBS_DIM](
        uninitialized=True
    )

    # Zero grads and backward
    online.zero_grads()
    online.backward[BATCH_SIZE](grad_output, grad_input, cache)

    # Check that some gradients are non-zero
    var has_grads = False
    for i in range(online.PARAM_SIZE):
        if online.grads[i] != 0:
            has_grads = True
            break
    print("Gradients computed: " + String(has_grads))
    print()

    # =========================================================================
    # Test optimizer update
    # =========================================================================

    print("Testing optimizer update...")

    # Save original param
    var original_param = online.params[0]

    # Update
    online.update()

    # Check param changed
    var param_changed = online.params[0] != original_param
    print("Parameters updated: " + String(param_changed))
    print()

    # =========================================================================
    # Test soft update
    # =========================================================================

    print("Testing soft update (tau=0.1)...")

    # Save target param before soft update
    var target_param_before = target.params[0]

    # Soft update: target = 0.1 * online + 0.9 * target
    target.soft_update_from(online, tau=0.1)

    # Expected: target_param_after = 0.1 * online.params[0] + 0.9 * target_param_before
    var expected = Scalar[dtype](0.1) * online.params[0] + Scalar[dtype](
        0.9
    ) * target_param_before
    var actual = target.params[0]
    var soft_update_correct = (
        Float64(actual) - Float64(expected)
    ).__abs__() < 1e-6

    print("Soft update correct: " + String(soft_update_correct))
    print(
        "  Expected: "
        + String(Float64(expected))[:10]
        + ", Actual: "
        + String(Float64(actual))[:10]
    )
    print()

    # =========================================================================
    # Summary
    # =========================================================================

    print("=" * 70)
    print("All Network wrapper tests completed!")
    print("=" * 70)
