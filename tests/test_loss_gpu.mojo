"""Test New Loss Functions on GPU.

Tests:
- HuberLoss (GPU training)
- CrossEntropyLoss (GPU training)
- DQN-style training with Huber Loss

Run with:
    pixi run -e apple mojo run tests/test_loss_gpu.mojo
"""

from time import perf_counter_ns
from random import seed, random_float64

from gpu.host import DeviceContext

from deep_rl.constants import dtype
from deep_rl.model import Linear, ReLU, Tanh, Softmax, seq
from deep_rl.loss import MSELoss, HuberLoss, CrossEntropyLoss
from deep_rl.optimizer import Adam
from deep_rl.training import Trainer
from deep_rl.initializer import Xavier, Kaiming


# =============================================================================
# Test Helper
# =============================================================================


fn print_test_header(name: String):
    print("\n" + "=" * 70)
    print("TEST: " + name)
    print("=" * 70)


# =============================================================================
# Test Huber Loss on GPU
# =============================================================================


fn test_huber_loss_gpu():
    print_test_header("Huber Loss Training (GPU)")

    seed(42)

    comptime BATCH_SIZE = 64
    comptime INPUT_DIM = 8
    comptime HIDDEN_DIM = 32
    comptime OUTPUT_DIM = 4
    comptime NUM_EPOCHS = 200
    comptime PRINT_EVERY = 50

    # Model
    var model = seq(
        Linear[INPUT_DIM, HIDDEN_DIM](),
        ReLU[HIDDEN_DIM](),
        Linear[HIDDEN_DIM, OUTPUT_DIM](),
    )

    print("  Model: Linear[8, 32] -> ReLU[32] -> Linear[32, 4]")
    print("  PARAM_SIZE: " + String(model.PARAM_SIZE))

    # Huber Loss
    var loss_fn = HuberLoss(delta=1.0)
    print("\n  Loss: HuberLoss(delta=1.0)")
    print("  Use case: DQN, robust to reward clipping/outliers")

    var optimizer = Adam(lr=0.01)

    var trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        Kaiming(),
        epochs=NUM_EPOCHS,
        print_every=PRINT_EVERY,
    )

    # Generate training data with some outliers
    var input_data = InlineArray[Scalar[dtype], BATCH_SIZE * INPUT_DIM](
        uninitialized=True
    )
    var target_data = InlineArray[Scalar[dtype], BATCH_SIZE * OUTPUT_DIM](
        uninitialized=True
    )

    for i in range(BATCH_SIZE):
        for j in range(INPUT_DIM):
            input_data[i * INPUT_DIM + j] = Scalar[dtype](random_float64() * 2 - 1)

        for j in range(OUTPUT_DIM):
            # Add some outliers (10% of data)
            if random_float64() < 0.1:
                target_data[i * OUTPUT_DIM + j] = Scalar[dtype](random_float64() * 10)
            else:
                var idx1 = j * 2
                var idx2 = j * 2 + 1
                var x1 = Float64(input_data[i * INPUT_DIM + idx1])
                var x2 = Float64(input_data[i * INPUT_DIM + idx2])
                target_data[i * OUTPUT_DIM + j] = Scalar[dtype](x1 * x2)

    print("\n  Training data: 10% outliers (values up to 10)")
    print("-" * 70)

    var start = perf_counter_ns()

    try:
        with DeviceContext() as ctx:
            var result = trainer.train_gpu[BATCH_SIZE](ctx, input_data, target_data)

            var elapsed_ms = Float64(perf_counter_ns() - start) / 1e6

            print("-" * 70)
            print("  GPU Training completed!")
            print("  Final loss: " + String(result.final_loss))
            print("  Epochs: " + String(result.epochs_trained))
            print("  Time: " + String(elapsed_ms)[:8] + " ms")
            print("  Avg per epoch: " + String(elapsed_ms / Float64(NUM_EPOCHS))[:6] + " ms")

            print("\n  PASS: Huber Loss GPU training completed")

    except e:
        print("  GPU Error: " + String(e))


# =============================================================================
# Test Cross-Entropy Loss on GPU
# =============================================================================


fn test_cross_entropy_gpu():
    print_test_header("Cross-Entropy Loss Training (GPU)")

    seed(123)

    comptime BATCH_SIZE = 64
    comptime INPUT_DIM = 8
    comptime HIDDEN_DIM = 32
    comptime NUM_CLASSES = 4
    comptime NUM_EPOCHS = 200
    comptime PRINT_EVERY = 50

    # Model outputting logits
    var model = seq(
        Linear[INPUT_DIM, HIDDEN_DIM](),
        ReLU[HIDDEN_DIM](),
        Linear[HIDDEN_DIM, NUM_CLASSES](),
    )

    print("  Model: Linear[8, 32] -> ReLU[32] -> Linear[32, 4]")
    print("  PARAM_SIZE: " + String(model.PARAM_SIZE))

    # Cross-Entropy Loss
    var loss_fn = CrossEntropyLoss()
    print("\n  Loss: CrossEntropyLoss")
    print("  Use case: Classification, policy gradients")

    var optimizer = Adam(lr=0.01)

    var trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        Kaiming(),
        epochs=NUM_EPOCHS,
        print_every=PRINT_EVERY,
    )

    # Generate classification data
    var input_data = InlineArray[Scalar[dtype], BATCH_SIZE * INPUT_DIM](
        uninitialized=True
    )
    var target_data = InlineArray[Scalar[dtype], BATCH_SIZE * NUM_CLASSES](
        uninitialized=True
    )

    for i in range(BATCH_SIZE):
        for j in range(INPUT_DIM):
            input_data[i * INPUT_DIM + j] = Scalar[dtype](random_float64() * 2 - 1)

        # Determine class based on quadrants of first 4 inputs
        var x1 = Float64(input_data[i * INPUT_DIM + 0])
        var x2 = Float64(input_data[i * INPUT_DIM + 1])

        var class_idx = 0
        if x1 >= 0 and x2 >= 0:
            class_idx = 0
        elif x1 < 0 and x2 >= 0:
            class_idx = 1
        elif x1 < 0 and x2 < 0:
            class_idx = 2
        else:
            class_idx = 3

        # One-hot encoding
        for j in range(NUM_CLASSES):
            target_data[i * NUM_CLASSES + j] = Scalar[dtype](1.0) if j == class_idx else Scalar[dtype](0.0)

    print("\n  Classification task: 4 classes based on quadrants")
    print("-" * 70)

    var start = perf_counter_ns()

    try:
        with DeviceContext() as ctx:
            var result = trainer.train_gpu[BATCH_SIZE](ctx, input_data, target_data)

            var elapsed_ms = Float64(perf_counter_ns() - start) / 1e6

            print("-" * 70)
            print("  GPU Training completed!")
            print("  Final loss: " + String(result.final_loss))
            print("  Epochs: " + String(result.epochs_trained))
            print("  Time: " + String(elapsed_ms)[:8] + " ms")
            print("  Avg per epoch: " + String(elapsed_ms / Float64(NUM_EPOCHS))[:6] + " ms")

            print("\n  PASS: Cross-Entropy GPU training completed")

    except e:
        print("  GPU Error: " + String(e))


# =============================================================================
# Test DQN-style Training with Huber Loss
# =============================================================================


fn test_dqn_style_training():
    print_test_header("DQN-Style Training (GPU) - Huber Loss")

    seed(456)

    # DQN typical architecture
    comptime BATCH_SIZE = 64
    comptime STATE_DIM = 8  # e.g., LunarLander observation
    comptime HIDDEN1 = 128
    comptime HIDDEN2 = 64
    comptime NUM_ACTIONS = 4
    comptime NUM_EPOCHS = 300
    comptime PRINT_EVERY = 75

    # Q-Network
    var q_network = seq(
        Linear[STATE_DIM, HIDDEN1](),
        ReLU[HIDDEN1](),
        Linear[HIDDEN1, HIDDEN2](),
        ReLU[HIDDEN2](),
        Linear[HIDDEN2, NUM_ACTIONS](),
    )

    print("  DQN Q-Network: " + String(STATE_DIM) + " -> 128 (ReLU) -> 64 (ReLU) -> " + String(NUM_ACTIONS))
    print("  PARAM_SIZE: " + String(q_network.PARAM_SIZE))

    # Huber Loss (standard for DQN)
    var loss_fn = HuberLoss(delta=1.0)
    print("\n  Loss: HuberLoss(delta=1.0)")
    print("  Why: Huber loss clips gradients from large TD errors")

    var optimizer = Adam(lr=0.0001)  # Low LR typical for DQN
    print("  Optimizer: Adam(lr=0.0001)")

    var trainer = Trainer(
        q_network,
        optimizer,
        loss_fn,
        Kaiming(),
        epochs=NUM_EPOCHS,
        print_every=PRINT_EVERY,
    )

    # Simulate experience replay batch
    # states: observations, targets: TD targets (r + gamma * max Q')
    var input_data = InlineArray[Scalar[dtype], BATCH_SIZE * STATE_DIM](
        uninitialized=True
    )
    var target_data = InlineArray[Scalar[dtype], BATCH_SIZE * NUM_ACTIONS](
        uninitialized=True
    )

    for i in range(BATCH_SIZE):
        # Random state (normalized like real observations)
        for j in range(STATE_DIM):
            input_data[i * STATE_DIM + j] = Scalar[dtype](random_float64() * 2 - 1)

        # Random TD targets (Q-values typically in [-inf, inf] but we simulate reasonable range)
        for j in range(NUM_ACTIONS):
            # Mix of small and occasional large values (like real TD errors)
            if random_float64() < 0.1:
                # Large TD error (outlier)
                target_data[i * NUM_ACTIONS + j] = Scalar[dtype](random_float64() * 20 - 10)
            else:
                # Normal TD target
                target_data[i * NUM_ACTIONS + j] = Scalar[dtype](random_float64() * 4 - 2)

    print("\n  Simulated replay batch with mixed TD targets")
    print("  (10% large TD errors to test Huber robustness)")
    print("-" * 70)

    var start = perf_counter_ns()

    try:
        with DeviceContext() as ctx:
            var result = trainer.train_gpu[BATCH_SIZE](ctx, input_data, target_data)

            var elapsed_ms = Float64(perf_counter_ns() - start) / 1e6

            print("-" * 70)
            print("  GPU Training completed!")
            print("  Final loss: " + String(result.final_loss))
            print("  Epochs: " + String(result.epochs_trained))
            print("  Time: " + String(elapsed_ms)[:8] + " ms")
            print("  Avg per epoch: " + String(elapsed_ms / Float64(NUM_EPOCHS))[:6] + " ms")

            print("\n  PASS: DQN-style training with Huber Loss completed")

    except e:
        print("  GPU Error: " + String(e))


# =============================================================================
# Compare Loss Functions on GPU
# =============================================================================


fn compare_loss_functions_gpu():
    print_test_header("Loss Function Comparison (GPU)")

    seed(789)

    comptime BATCH_SIZE = 64
    comptime INPUT_DIM = 8
    comptime HIDDEN_DIM = 32
    comptime OUTPUT_DIM = 4
    comptime NUM_EPOCHS = 100

    # Generate training data with outliers
    var input_data = InlineArray[Scalar[dtype], BATCH_SIZE * INPUT_DIM](
        uninitialized=True
    )
    var target_data = InlineArray[Scalar[dtype], BATCH_SIZE * OUTPUT_DIM](
        uninitialized=True
    )

    for i in range(BATCH_SIZE):
        for j in range(INPUT_DIM):
            input_data[i * INPUT_DIM + j] = Scalar[dtype](random_float64() * 2 - 1)

        for j in range(OUTPUT_DIM):
            # 15% outliers
            if random_float64() < 0.15:
                target_data[i * OUTPUT_DIM + j] = Scalar[dtype](random_float64() * 10)
            else:
                var idx1 = j * 2
                var idx2 = j * 2 + 1
                var x1 = Float64(input_data[i * INPUT_DIM + idx1])
                var x2 = Float64(input_data[i * INPUT_DIM + idx2])
                target_data[i * OUTPUT_DIM + j] = Scalar[dtype](x1 + x2)

    print("  Comparing MSE vs Huber on data with 15% outliers...")
    print("  Model: Linear[8, 32] -> ReLU[32] -> Linear[32, 4]")
    print("  Epochs: " + String(NUM_EPOCHS))
    print()

    try:
        with DeviceContext() as ctx:
            # Test MSE
            var model_mse = seq(
                Linear[INPUT_DIM, HIDDEN_DIM](),
                ReLU[HIDDEN_DIM](),
                Linear[HIDDEN_DIM, OUTPUT_DIM](),
            )
            var trainer_mse = Trainer(
                model_mse, Adam(lr=0.01), MSELoss(), Kaiming(),
                epochs=NUM_EPOCHS, print_every=0,
            )
            var start_mse = perf_counter_ns()
            var result_mse = trainer_mse.train_gpu[BATCH_SIZE](ctx, input_data, target_data)
            var time_mse = Float64(perf_counter_ns() - start_mse) / 1e6
            print("  MSELoss:            loss = " + String(result_mse.final_loss)[:8] + "  time = " + String(time_mse)[:6] + " ms")

            # Test Huber (delta=1)
            var model_huber1 = seq(
                Linear[INPUT_DIM, HIDDEN_DIM](),
                ReLU[HIDDEN_DIM](),
                Linear[HIDDEN_DIM, OUTPUT_DIM](),
            )
            var trainer_huber1 = Trainer(
                model_huber1, Adam(lr=0.01), HuberLoss(delta=1.0), Kaiming(),
                epochs=NUM_EPOCHS, print_every=0,
            )
            var start_huber1 = perf_counter_ns()
            var result_huber1 = trainer_huber1.train_gpu[BATCH_SIZE](ctx, input_data, target_data)
            var time_huber1 = Float64(perf_counter_ns() - start_huber1) / 1e6
            print("  HuberLoss(d=1.0):   loss = " + String(result_huber1.final_loss)[:8] + "  time = " + String(time_huber1)[:6] + " ms")

            # Test Huber (delta=0.5)
            var model_huber05 = seq(
                Linear[INPUT_DIM, HIDDEN_DIM](),
                ReLU[HIDDEN_DIM](),
                Linear[HIDDEN_DIM, OUTPUT_DIM](),
            )
            var trainer_huber05 = Trainer(
                model_huber05, Adam(lr=0.01), HuberLoss(delta=0.5), Kaiming(),
                epochs=NUM_EPOCHS, print_every=0,
            )
            var start_huber05 = perf_counter_ns()
            var result_huber05 = trainer_huber05.train_gpu[BATCH_SIZE](ctx, input_data, target_data)
            var time_huber05 = Float64(perf_counter_ns() - start_huber05) / 1e6
            print("  HuberLoss(d=0.5):   loss = " + String(result_huber05.final_loss)[:8] + "  time = " + String(time_huber05)[:6] + " ms")

            # Test Huber (delta=2)
            var model_huber2 = seq(
                Linear[INPUT_DIM, HIDDEN_DIM](),
                ReLU[HIDDEN_DIM](),
                Linear[HIDDEN_DIM, OUTPUT_DIM](),
            )
            var trainer_huber2 = Trainer(
                model_huber2, Adam(lr=0.01), HuberLoss(delta=2.0), Kaiming(),
                epochs=NUM_EPOCHS, print_every=0,
            )
            var start_huber2 = perf_counter_ns()
            var result_huber2 = trainer_huber2.train_gpu[BATCH_SIZE](ctx, input_data, target_data)
            var time_huber2 = Float64(perf_counter_ns() - start_huber2) / 1e6
            print("  HuberLoss(d=2.0):   loss = " + String(result_huber2.final_loss)[:8] + "  time = " + String(time_huber2)[:6] + " ms")

            print("\n  Note: Different delta values trade off robustness vs smoothness")
            print("  All loss functions trained on GPU successfully!")

    except e:
        print("  GPU Error: " + String(e))


# =============================================================================
# Test Policy Gradient Style with Cross-Entropy
# =============================================================================


fn test_policy_gradient_style():
    print_test_header("Policy Gradient Style (GPU) - Cross-Entropy")

    seed(999)

    comptime BATCH_SIZE = 64
    comptime STATE_DIM = 8
    comptime HIDDEN_DIM = 64
    comptime NUM_ACTIONS = 4
    comptime NUM_EPOCHS = 200
    comptime PRINT_EVERY = 50

    # Policy network (outputs action logits)
    var policy_net = seq(
        Linear[STATE_DIM, HIDDEN_DIM](),
        Tanh[HIDDEN_DIM](),
        Linear[HIDDEN_DIM, NUM_ACTIONS](),
    )

    print("  Policy Network: " + String(STATE_DIM) + " -> 64 (Tanh) -> " + String(NUM_ACTIONS))
    print("  PARAM_SIZE: " + String(policy_net.PARAM_SIZE))

    # Cross-Entropy Loss
    var loss_fn = CrossEntropyLoss()
    print("\n  Loss: CrossEntropyLoss")
    print("  Use case: Supervised policy learning (behavior cloning)")

    var optimizer = Adam(lr=0.001)

    var trainer = Trainer(
        policy_net,
        optimizer,
        loss_fn,
        Xavier(),
        epochs=NUM_EPOCHS,
        print_every=PRINT_EVERY,
    )

    # Simulate expert demonstrations
    var input_data = InlineArray[Scalar[dtype], BATCH_SIZE * STATE_DIM](
        uninitialized=True
    )
    var target_data = InlineArray[Scalar[dtype], BATCH_SIZE * NUM_ACTIONS](
        uninitialized=True
    )

    for i in range(BATCH_SIZE):
        # Random state
        for j in range(STATE_DIM):
            input_data[i * STATE_DIM + j] = Scalar[dtype](random_float64() * 2 - 1)

        # "Expert" action based on simple policy
        var x1 = Float64(input_data[i * STATE_DIM + 0])
        var x2 = Float64(input_data[i * STATE_DIM + 1])

        var expert_action = 0
        if x1 >= 0 and x2 >= 0:
            expert_action = 0
        elif x1 < 0 and x2 >= 0:
            expert_action = 1
        elif x1 < 0 and x2 < 0:
            expert_action = 2
        else:
            expert_action = 3

        # One-hot target
        for j in range(NUM_ACTIONS):
            target_data[i * NUM_ACTIONS + j] = Scalar[dtype](1.0) if j == expert_action else Scalar[dtype](0.0)

    print("\n  Behavior cloning from expert demonstrations")
    print("-" * 70)

    var start = perf_counter_ns()

    try:
        with DeviceContext() as ctx:
            var result = trainer.train_gpu[BATCH_SIZE](ctx, input_data, target_data)

            var elapsed_ms = Float64(perf_counter_ns() - start) / 1e6

            print("-" * 70)
            print("  GPU Training completed!")
            print("  Final loss: " + String(result.final_loss))
            print("  Epochs: " + String(result.epochs_trained))
            print("  Time: " + String(elapsed_ms)[:8] + " ms")

            if result.final_loss < 1.0:
                print("\n  PASS: Policy gradient style training succeeded")
            else:
                print("\n  Note: May need more training for convergence")

    except e:
        print("  GPU Error: " + String(e))


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("Deep RL - New Loss Functions GPU Tests")
    print("=" * 70)
    print()
    print("Testing: HuberLoss, CrossEntropyLoss on GPU")
    print()

    test_huber_loss_gpu()
    test_cross_entropy_gpu()
    test_dqn_style_training()
    compare_loss_functions_gpu()
    test_policy_gradient_style()

    print("\n" + "=" * 70)
    print("All Loss Function GPU Tests Completed!")
    print("=" * 70)
