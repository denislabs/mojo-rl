"""Test for nn.mojo - Neural Network Module with Sequential composition."""

from deep_rl import (
    Linear,
    ReLU,
    Tanh,
    seq,
    Adam,
    SGD,
    MSELoss,
    dtype,
    Trainer,
    Xavier,
    Kaiming,
)
from layout import Layout, LayoutTensor


fn main():
    print("=" * 60)
    print("Testing Neural Network Module with Sequential")
    print("=" * 60)
    print()

    # ==========================================================================
    # Test 1: Linear layer
    # ==========================================================================
    print("Test 1: Linear layer forward/backward")
    print("-" * 40)

    comptime BATCH = 4
    comptime IN = 2
    comptime OUT = 3

    var linear = Linear[IN, OUT]()

    # Create input - flattened [batch, in_dim]
    var input_storage = InlineArray[Scalar[dtype], BATCH * IN](
        uninitialized=True
    )
    # Sample 0: [1, 2]
    input_storage[0] = 1.0
    input_storage[1] = 2.0
    # Sample 1: [3, 4]
    input_storage[2] = 3.0
    input_storage[3] = 4.0
    # Sample 2: [5, 6]
    input_storage[4] = 5.0
    input_storage[5] = 6.0
    # Sample 3: [7, 8]
    input_storage[6] = 7.0
    input_storage[7] = 8.0

    var output_storage = InlineArray[Scalar[dtype], BATCH * OUT](
        uninitialized=True
    )
    var cache_storage = InlineArray[Scalar[dtype], BATCH * linear.CACHE_SIZE](
        uninitialized=True
    )

    # Initialize params for linear layer
    var xavier = Xavier()
    var params = xavier.init[linear.PARAM_SIZE, IN, OUT]()

    # Create LayoutTensor views
    var input_tensor = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
    ](input_storage.unsafe_ptr())
    var output_tensor = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
    ](output_storage.unsafe_ptr())
    var params_tensor = LayoutTensor[
        dtype, Layout.row_major(linear.PARAM_SIZE), MutAnyOrigin
    ](params.unsafe_ptr())
    var cache = LayoutTensor[
        dtype, Layout.row_major(BATCH, linear.CACHE_SIZE), MutAnyOrigin
    ](cache_storage.unsafe_ptr())

    linear.forward[BATCH](input_tensor, output_tensor, params_tensor, cache)

    print("Input shape: [4, 2]")
    print("Output shape: [4, 3]")
    print(
        "Output sample 0:",
        Float64(output_storage[0]),
        Float64(output_storage[1]),
        Float64(output_storage[2]),
    )
    print("Linear test passed!")
    print()

    # ==========================================================================
    # Test 2: Seq2 composition (Linear -> ReLU)
    # ==========================================================================
    print("Test 2: Seq2 (Linear -> ReLU)")
    print("-" * 40)

    # Using type inference with seq() helper
    var model2 = seq(Linear[IN, OUT](), ReLU[OUT]())

    print("Model: Linear[2, 3] -> ReLU[3]")
    print("PARAM_SIZE:", model2.PARAM_SIZE)  # Should be 2*3 + 3 = 9

    var out2 = InlineArray[Scalar[dtype], BATCH * OUT](uninitialized=True)
    var cache2_storage = InlineArray[Scalar[dtype], BATCH * model2.CACHE_SIZE](
        uninitialized=True
    )

    # Initialize params for model2
    var params2 = xavier.init[model2.PARAM_SIZE, IN, OUT]()

    # Create LayoutTensor views
    var input2_tensor = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
    ](input_storage.unsafe_ptr())
    var out2_tensor = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT), MutAnyOrigin
    ](out2.unsafe_ptr())
    var params2_tensor = LayoutTensor[
        dtype, Layout.row_major(model2.PARAM_SIZE), MutAnyOrigin
    ](params2.unsafe_ptr())
    var cache2 = LayoutTensor[
        dtype, Layout.row_major(BATCH, model2.CACHE_SIZE), MutAnyOrigin
    ](cache2_storage.unsafe_ptr())

    model2.forward[BATCH](input2_tensor, out2_tensor, params2_tensor, cache2)

    print("After ReLU, all negative values should be 0")
    print(
        "Output sample 0:", Float64(out2[0]), Float64(out2[1]), Float64(out2[2])
    )
    print("Seq2 test passed!")
    print()

    # ==========================================================================
    # Test 3: Seq3 composition (Linear -> ReLU -> Linear)
    # ==========================================================================
    print("Test 3: Seq3 (Linear -> ReLU -> Linear)")
    print("-" * 40)

    comptime HIDDEN = 4
    comptime OUT_FINAL = 1

    # Using type inference with seq() helper
    var model3 = seq(
        Linear[IN, HIDDEN](),
        ReLU[HIDDEN](),
        Linear[HIDDEN, OUT_FINAL](),
    )

    print("Model: Linear[2, 4] -> ReLU[4] -> Linear[4, 1]")
    print(
        "PARAM_SIZE:", model3.PARAM_SIZE
    )  # (2*4 + 4) + 0 + (4*1 + 1) = 12 + 5 = 17

    var out3 = InlineArray[Scalar[dtype], BATCH * OUT_FINAL](uninitialized=True)
    var cache3_storage = InlineArray[Scalar[dtype], BATCH * model3.CACHE_SIZE](
        uninitialized=True
    )

    # Initialize params for model3
    var params3 = xavier.init[model3.PARAM_SIZE, IN, OUT_FINAL]()

    # Create LayoutTensor views
    var input3_tensor = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
    ](input_storage.unsafe_ptr())
    var out3_tensor = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_FINAL), MutAnyOrigin
    ](out3.unsafe_ptr())
    var params3_tensor = LayoutTensor[
        dtype, Layout.row_major(model3.PARAM_SIZE), MutAnyOrigin
    ](params3.unsafe_ptr())
    var cache3 = LayoutTensor[
        dtype, Layout.row_major(BATCH, model3.CACHE_SIZE), MutAnyOrigin
    ](cache3_storage.unsafe_ptr())

    model3.forward[BATCH](input3_tensor, out3_tensor, params3_tensor, cache3)

    print(
        "Outputs:",
        Float64(out3[0]),
        Float64(out3[1]),
        Float64(out3[2]),
        Float64(out3[3]),
    )
    print("Seq3 test passed!")
    print()

    # ==========================================================================
    # Test 4: Seq4 composition (Linear -> Tanh -> Linear -> ReLU)
    # ==========================================================================
    print("Test 4: Seq4 (Linear -> Tanh -> Linear -> ReLU)")
    print("-" * 40)

    var model4 = seq(
        Linear[IN, HIDDEN](),
        Tanh[HIDDEN](),
        Linear[HIDDEN, OUT_FINAL](),
        ReLU[OUT_FINAL](),
    )

    print("Model: Linear[2, 4] -> Tanh[4] -> Linear[4, 1] -> ReLU[1]")
    print("PARAM_SIZE:", model4.PARAM_SIZE)  # 12 + 0 + 5 + 0 = 17

    var out4 = InlineArray[Scalar[dtype], BATCH * OUT_FINAL](uninitialized=True)
    var cache4_storage = InlineArray[Scalar[dtype], BATCH * model4.CACHE_SIZE](
        uninitialized=True
    )

    # Initialize params for model4
    var params4 = xavier.init[model4.PARAM_SIZE, IN, OUT_FINAL]()

    # Create LayoutTensor views
    var input4_tensor = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
    ](input_storage.unsafe_ptr())
    var out4_tensor = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_FINAL), MutAnyOrigin
    ](out4.unsafe_ptr())
    var params4_tensor = LayoutTensor[
        dtype, Layout.row_major(model4.PARAM_SIZE), MutAnyOrigin
    ](params4.unsafe_ptr())
    var cache4 = LayoutTensor[
        dtype, Layout.row_major(BATCH, model4.CACHE_SIZE), MutAnyOrigin
    ](cache4_storage.unsafe_ptr())

    model4.forward[BATCH](input4_tensor, out4_tensor, params4_tensor, cache4)

    print(
        "Outputs:",
        Float64(out4[0]),
        Float64(out4[1]),
        Float64(out4[2]),
        Float64(out4[3]),
    )
    print("Seq4 test passed!")
    print()

    # ==========================================================================
    # Test 5: Training with Seq3 (Learning y = 2*x1 + 3*x2)
    # ==========================================================================
    print("Test 5: Training Seq3 on regression task")
    print("-" * 40)

    # Create model: Linear[2, 8] -> ReLU -> Linear[8, 1]
    comptime TRAIN_IN = 2
    comptime TRAIN_HIDDEN = 8
    comptime TRAIN_OUT = 1
    comptime TRAIN_BATCH = 4
    comptime TRAIN_PARAM_SIZE = (TRAIN_IN * TRAIN_HIDDEN + TRAIN_HIDDEN) + 0 + (
        TRAIN_HIDDEN * TRAIN_OUT + TRAIN_OUT
    )

    var train_model = seq(
        Linear[TRAIN_IN, TRAIN_HIDDEN](),
        ReLU[TRAIN_HIDDEN](),
        Linear[TRAIN_HIDDEN, TRAIN_OUT](),
    )

    var optimizer = Adam[TRAIN_PARAM_SIZE](lr=0.1)

    var loss_function = MSELoss()

    # Training data: y = 2*x1 + 3*x2
    var train_input = InlineArray[Scalar[dtype], TRAIN_BATCH * TRAIN_IN](
        uninitialized=True
    )
    var train_target = InlineArray[Scalar[dtype], TRAIN_BATCH * TRAIN_OUT](
        uninitialized=True
    )

    # [1, 1] -> 5, [2, 1] -> 7, [1, 2] -> 8, [2, 2] -> 10
    train_input[0] = 1.0
    train_input[1] = 1.0
    train_target[0] = 5.0
    train_input[2] = 2.0
    train_input[3] = 1.0
    train_target[1] = 7.0
    train_input[4] = 1.0
    train_input[5] = 2.0
    train_target[2] = 8.0
    train_input[6] = 2.0
    train_input[7] = 2.0
    train_target[3] = 10.0

    print("Training y = 2*x1 + 3*x2 for 100 epochs...")

    var trainer = Trainer(
        model=train_model,
        optimizer=optimizer,
        loss_function=loss_function,
        initializer=Xavier(),
        epochs=100,
        print_every=10,
    )
    var result = trainer.train[TRAIN_BATCH](train_input, train_target)
    print("Final loss:", result.final_loss)
    print("Epochs trained:", result.epochs_trained)

    # Final prediction

    var final_input = InlineArray[Scalar[dtype], 4](uninitialized=True)
    final_input[0] = 1.0
    final_input[1] = 1.0
    final_input[2] = 2.0
    final_input[3] = 1.0

    var final_target = InlineArray[Scalar[dtype], 2](uninitialized=True)
    final_target[0] = 5.0
    final_target[1] = 7.0

    var final_loss = trainer.evaluate[2](final_input, final_target)
    print("Final loss:", final_loss)

    print("Training test passed!")
    print()

    # ==========================================================================
    # Test 6: Training with Kaiming initialization (better for ReLU networks)
    # ==========================================================================
    print("Test 6: Training with Kaiming initialization")
    print("-" * 40)

    var train_model2 = seq(
        Linear[TRAIN_IN, TRAIN_HIDDEN](),
        ReLU[TRAIN_HIDDEN](),
        Linear[TRAIN_HIDDEN, TRAIN_OUT](),
    )

    var optimizer2 = Adam[TRAIN_PARAM_SIZE](lr=0.1)

    print("Using Kaiming initialization for ReLU network...")

    # Use Kaiming initializer (better for ReLU networks)
    comptime TrainModel2Type = type_of(train_model2)
    var trainer2 = Trainer[TrainModel2Type, Adam[TRAIN_PARAM_SIZE], MSELoss, Kaiming](
        model=train_model2,
        optimizer=optimizer2,
        loss_function=loss_function,
        initializer=Kaiming(),
        epochs=100,
        print_every=20,
    )
    var result2 = trainer2.train[TRAIN_BATCH](train_input, train_target)
    print("Final loss with Kaiming init:", result2.final_loss)

    print("Kaiming initialization test passed!")
    print()

    # ==========================================================================
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
