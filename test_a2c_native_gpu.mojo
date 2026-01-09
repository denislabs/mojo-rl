"""Test file for GPU-Native A2C implementation.

Run with:
    pixi run -e apple mojo run test_a2c_native_gpu.mojo
"""

from deep_agents.gpu.a2c_native import (
    train_cartpole_native,
)


fn main() raises:
    print()
    print("GPU-Native A2C - Phase 5: CartPole Training")
    print()

    # Phase 5: Train on CartPole
    print("=" * 60)
    print("Training GPU-Native A2C on CartPole")
    print("=" * 60)
    train_cartpole_native()
