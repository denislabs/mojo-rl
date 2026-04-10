"""GPU Neural Network Training Example.

Trains a small MLP to learn a nonlinear function on GPU using the Trainer API.

Run with:
    pixi run -e apple mojo run -I . examples/nn_gpu_training.mojo
"""

from std.random import seed, random_float64
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model.linear import Linear
from mojo_rl.nn.model.relu import ReLU
from mojo_rl.nn.model.sequential import Sequential
from mojo_rl.nn.loss.mse import MSELoss
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.training.trainer import Trainer
from mojo_rl.nn.initializer.initializers import Kaiming
from std.math import sin, cos


def main() raises:
    seed(42)

    # Network: 4 -> 64 (ReLU) -> 64 (ReLU) -> 1
    comptime IN_DIM = 4
    comptime HIDDEN = 64
    comptime OUT_DIM = 1
    comptime BATCH = 128
    comptime EPOCHS = 500

    comptime MLP = Sequential[
        Linear[IN_DIM, HIDDEN],
        ReLU[HIDDEN],
        Linear[HIDDEN, HIDDEN],
        ReLU[HIDDEN],
        Linear[HIDDEN, OUT_DIM],
    ]

    print("GPU Neural Network Training")
    print("=" * 50)
    print("  Architecture: Linear[4,64] -> ReLU -> Linear[64,64] -> ReLU -> Linear[64,1]")
    print("  Parameters: " + String(MLP.PARAM_SIZE))
    print("  Batch size: " + String(BATCH))
    print("  Epochs: " + String(EPOCHS))

    # Generate training data: y = sin(x0) + cos(x1) + 0.5*x2 - x3
    var inputs = InlineArray[Scalar[dtype], BATCH * IN_DIM](uninitialized=True)
    var targets = InlineArray[Scalar[dtype], BATCH * OUT_DIM](uninitialized=True)

    for b in range(BATCH):
        var x0 = random_float64(-2.0, 2.0)
        var x1 = random_float64(-2.0, 2.0)
        var x2 = random_float64(-2.0, 2.0)
        var x3 = random_float64(-2.0, 2.0)
        inputs[b * IN_DIM + 0] = Scalar[dtype](x0)
        inputs[b * IN_DIM + 1] = Scalar[dtype](x1)
        inputs[b * IN_DIM + 2] = Scalar[dtype](x2)
        inputs[b * IN_DIM + 3] = Scalar[dtype](x3)

        targets[b] = Scalar[dtype](sin(x0) + cos(x1) + 0.5 * x2 - x3)

    var input_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN_DIM), MutAnyOrigin
    ](inputs.unsafe_ptr())
    var target_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OUT_DIM), MutAnyOrigin
    ](targets.unsafe_ptr())

    # Train on GPU
    comptime TRAINER = Trainer[MLP, Adam[], MSELoss]

    var ctx = DeviceContext()
    var state = TRAINER.init_state_gpu[Kaiming[]](ctx)

    print("\n  Training on GPU...")
    var start = perf_counter_ns()
    var result = TRAINER.train_gpu[BATCH](
        state, ctx, input_t, target_t, epochs=EPOCHS, print_every=100,
    )
    var end = perf_counter_ns()

    var time_ms = Float64(end - start) / 1e6
    print("\n  Final loss: " + String(result.final_loss))
    print("  Epochs: " + String(result.epochs_trained))
    print("  Time: " + String(time_ms)[byte=:8] + " ms")
    print(
        "  Avg per epoch: " + String(time_ms / Float64(EPOCHS))[byte=:6] + " ms"
    )

    if result.final_loss < 0.05:
        print("\n  Training converged successfully!")
    else:
        print("\n  Training ran but could use more epochs.")
