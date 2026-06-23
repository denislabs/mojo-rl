"""DIAGNOSTIC (throwaway): minimal conv+BN net through the REAL Trainer.

Reproduces the failing structure of the CIFAR BN nets in seconds: one
BatchNorm2D, trained via Trainer.train_gpu (so it exercises Adam.adopt + the
contiguous arena + the eval sub-buffer-view path), on a tiny synthetic task
where the class is the high-energy quadrant (trivially learnable).

After training it prints test top-1 in eval-mode (BN running stats) AND
train-mode (BN per-batch stats). They should BOTH be high and roughly equal. If
eval-mode collapses toward chance (25%) while train-mode stays high, the BN
running-stat eval path is broken in the full Trainer/arena context.

Run (Apple):  pixi run -e apple  mojo run -I . examples/nn/conv2d/_diag_tiny_bn_arena_gpu.mojo
Run (NVIDIA): pixi run -e nvidia mojo run -I . examples/nn/conv2d/_diag_tiny_bn_arena_gpu.mojo

Delete after debugging.
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.models.conv import Conv2DBatchNormReLU
from mojo_rl.nn.primitives.flatten import Flatten
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.training.trainer import Trainer


def main() raises:
    comptime C = 3
    comptime H = 8
    comptime W = 8
    comptime IN = C * H * W       # 192
    comptime NC = 4               # 4 quadrants
    comptime BATCH = 20
    comptime N_TRAIN = 400
    comptime N_TEST = 100
    comptime N_EPOCHS = 40

    var ctx = DeviceContext()

    comptime Net = Sequential[
        Conv2DBatchNormReLU[C, 8, 3, 1, 1, H, W],   # → 8×8×8 = 512
        Flatten[8 * H * W],
        Linear[8 * H * W, NC],
    ]
    var trainer = Trainer[Net, NC, IN, BATCH, "gpu"].make[Kaiming](
        Optional(ctx), lr=2e-3
    )

    # Synthetic data: class = high-energy quadrant (channel 0), + small noise.
    def make_x(n: Int) -> List[Scalar[DT]]:
        var x = List[Scalar[DT]](length=n * IN, fill=0.0)
        for i in range(n):
            var cls = i % NC
            for ch in range(C):
                for y in range(H):
                    for xx in range(W):
                        var idx = i * IN + ch * (H * W) + y * W + xx
                        var noise = (
                            Float32(((i * 13 + ch * 7 + y * 3 + xx) % 11) - 5)
                            * 0.1
                        )
                        x[idx] = Scalar[DT](noise)
            var qy = (cls // 2) * 4
            var qx = (cls % 2) * 4
            for yy in range(qy, qy + 4):
                for xx in range(qx, qx + 4):
                    x[i * IN + yy * W + xx] += Scalar[DT](5.0)
        return x^

    var train_x = make_x(N_TRAIN)
    var test_x = make_x(N_TEST)
    var train_y = List[Scalar[DT]](length=N_TRAIN * NC, fill=0.0)
    for i in range(N_TRAIN):
        train_y[i * NC + (i % NC)] = 1.0
    var test_labels = List[Int32](length=N_TEST, fill=0)
    for i in range(N_TEST):
        test_labels[i] = Int32(i % NC)

    _ = trainer.train_gpu[N_TRAIN, N_TEST](
        train_x, train_y, test_x, test_labels,
        Optional(ctx), epochs=N_EPOCHS, shuffle=True, print_progress=True,
    )

    print("\n=== post-train eval comparison ===")
    var acc_eval = trainer.eval_top1[N_TEST](test_x, test_labels, Optional(ctx))
    print("eval-mode  (running stats) top1 = " + String(acc_eval * 100.0) + "%")
    trainer.model.set_attr["training"](Scalar[DT](1.0))
    var acc_train = trainer.eval_top1[N_TEST](test_x, test_labels, Optional(ctx))
    print("train-mode (batch stats)   top1 = " + String(acc_train * 100.0) + "%")
    print("DONE")
