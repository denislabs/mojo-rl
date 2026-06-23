"""DIAGNOSTIC (throwaway): DEEP conv+BN net (CIFAR-like structure, tiny spatial).

Bisects the NVIDIA BN eval collapse on the depth/MaxPool/multi-BN axis while
keeping spatial dims small (fast). 4 BatchNorm2D layers + 2 MaxPool, through the
real Trainer (Adam.adopt + arena + eval views), on the trivially-learnable
synthetic quadrant task.

If the tiny (1-BN) repro PASSES on NVIDIA but THIS (4-BN + MaxPool) FAILS
(eval-mode ≪ train-mode), the trigger is depth / multiple BN layers / MaxPool —
not big channel/spatial sizes. If this also passes, the trigger is the large
channel/spatial sizes of the real net.

Run (NVIDIA): pixi run -e nvidia mojo run -I . examples/nn/conv2d/_diag_deep_bn_arena_gpu.mojo

Delete after debugging.
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.models.conv import Conv2DBatchNormReLU
from mojo_rl.nn.primitives.max_pool_2d import MaxPool2D
from mojo_rl.nn.primitives.flatten import Flatten
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.combinators.sequential import Sequential
from mojo_rl.nn.training.trainer import Trainer


def main() raises:
    comptime C = 3
    comptime H = 8
    comptime W = 8
    comptime IN = C * H * W       # 192
    comptime NC = 4
    comptime BATCH = 20
    comptime N_TRAIN = 400
    comptime N_TEST = 100
    comptime N_EPOCHS = 60

    var ctx = DeviceContext()

    comptime Net = Sequential[
        Conv2DBatchNormReLU[C, 16, 3, 1, 1, 8, 8],    # 16×8×8
        Conv2DBatchNormReLU[16, 16, 3, 1, 1, 8, 8],   # 16×8×8
        MaxPool2D[16, 2, 2, 0, 8, 8],                 # 16×4×4
        Conv2DBatchNormReLU[16, 32, 3, 1, 1, 4, 4],   # 32×4×4
        Conv2DBatchNormReLU[32, 32, 3, 1, 1, 4, 4],   # 32×4×4
        MaxPool2D[32, 2, 2, 0, 4, 4],                 # 32×2×2
        Flatten[32 * 2 * 2],                          # 128
        Linear[32 * 2 * 2, NC],
    ]
    var trainer = Trainer[Net, NC, IN, BATCH, "gpu"].make[Kaiming](
        Optional(ctx), lr=2e-3
    )

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
