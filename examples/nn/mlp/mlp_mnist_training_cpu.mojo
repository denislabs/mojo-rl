"""End-to-end MLP training on MNIST — CPU baseline.

Mirror of `mlp_mnist_training_gpu.mojo` for CPU. Trains the same 2-hidden-layer
MLP (784 → 256 → 128 → 10) on real MNIST and checks that test accuracy exceeds
97%. Uses `LinearReLU` (auto-fused MatMul+BiasAdd+ReLU) which on CPU now routes
matmul through `linalg.matmul[target="cpu"]` (vendor BLAS / Modular CPU GEMM).

There is no `Trainer.train_cpu_minibatch_full`, so this example assembles the
minibatch loop directly from the same primitives the GPU helper composes:
`MODEL.forward/backward` + `LOSS_FUNCTION.forward/backward` + `OPTIMIZER.step`.

Run:
    pixi run mojo run -I . examples/nn/mlp/mlp_mnist_training_cpu.mojo
"""

from std.random import seed, random_ui64
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model.linear_act import LinearReLU
from mojo_rl.nn.model.linear import Linear
from mojo_rl.nn.model.sequential import Sequential
from mojo_rl.nn.loss.cross_entropy import CrossEntropyLoss
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.training import Trainer
from mojo_rl.nn.initializer.initializers import Kaiming
from mojo_rl.nn.datasets import MNIST


comptime BATCH = 128
comptime EPOCHS = 5

comptime MLP = Sequential[
    LinearReLU[28 * 28, 256],
    LinearReLU[256, 128],
    Linear[128, 10],
]
comptime TRAINER = Trainer[MLP, Adam[LR=0.001], CrossEntropyLoss]


@always_inline
def _fisher_yates(mut idx: List[Int], n: Int):
    """In-place Fisher–Yates shuffle using the global RNG (seeded in main)."""
    for i in range(n - 1, 0, -1):
        var j = Int(random_ui64(0, UInt64(i)))
        var tmp = idx[i]
        idx[i] = idx[j]
        idx[j] = tmp


def _argmax_row[N: Int](
    logits: LayoutTensor[dtype, Layout.row_major(BATCH, N), MutAnyOrigin],
    row: Int,
) -> Int:
    var best_j = 0
    var best_v = Float64(rebind[Scalar[dtype]](logits[row, 0]))
    for j in range(1, N):
        var v = Float64(rebind[Scalar[dtype]](logits[row, j]))
        if v > best_v:
            best_v = v
            best_j = j
    return best_j


def _eval_accuracy(
    state_params: LayoutTensor[
        dtype, Layout.row_major(MLP.PARAM_SIZE), MutAnyOrigin
    ],
    mut model_state: LayoutTensor[
        dtype, Layout.row_major(MLP.STATE_SIZE), MutAnyOrigin
    ],
    test_images: List[Scalar[dtype]],
    test_labels: List[Int32],
    n_test: Int,
) -> Float64:
    """Forward-only sweep over the test set, return top-1 accuracy."""
    # One reusable minibatch worth of buffers.
    var x_buf = List[Scalar[dtype]](capacity=BATCH * MNIST.IMG_SIZE)
    var y_buf = List[Scalar[dtype]](capacity=BATCH * MNIST.NUM_CLASSES)
    for _ in range(BATCH * MNIST.IMG_SIZE):
        x_buf.append(0)
    for _ in range(BATCH * MNIST.NUM_CLASSES):
        y_buf.append(0)

    var x_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, MNIST.IMG_SIZE), MutAnyOrigin
    ](x_buf.unsafe_ptr())
    var y_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, MNIST.NUM_CLASSES), MutAnyOrigin
    ](y_buf.unsafe_ptr())

    var correct = 0
    var seen = 0
    var n_full = n_test // BATCH  # drop the trailing partial batch
    for b in range(n_full):
        var base = b * BATCH
        for s in range(BATCH):
            var src = (base + s) * MNIST.IMG_SIZE
            for k in range(MNIST.IMG_SIZE):
                x_t[s, k] = test_images[src + k]

        MLP.forward[BATCH](x_t, y_t, state_params, model_state)

        for s in range(BATCH):
            var pred = _argmax_row[MNIST.NUM_CLASSES](y_t, s)
            if pred == Int(test_labels[base + s]):
                correct += 1
        seen += BATCH

    return Float64(correct) / Float64(seen)


def main() raises:
    seed(42)

    print("=" * 65)
    print("MNIST MLP training — CPU baseline (linalg.matmul[target='cpu'])")
    print("=" * 65)
    print(
        "  architecture: LinearReLU(784→256) → LinearReLU(256→128) →"
        " Linear(128→10)"
    )
    print("  params: " + String(MLP.PARAM_SIZE))
    print("  batch: " + String(BATCH) + " | epochs: " + String(EPOCHS))

    var ds = MNIST()

    var state = TRAINER.init_state[Kaiming[]]()

    # ── Per-minibatch reusable buffers (heap-allocated, alive for whole run) ──
    var x_buf = List[Scalar[dtype]](capacity=BATCH * MNIST.IMG_SIZE)
    var y_buf = List[Scalar[dtype]](capacity=BATCH * MNIST.NUM_CLASSES)
    var out_buf = List[Scalar[dtype]](capacity=BATCH * MLP.OUT_DIM)
    var go_buf = List[Scalar[dtype]](capacity=BATCH * MLP.OUT_DIM)
    var gi_buf = List[Scalar[dtype]](capacity=BATCH * MLP.IN_DIM)
    var cache_buf = List[Scalar[dtype]](capacity=BATCH * MLP.CACHE_SIZE)

    for _ in range(BATCH * MNIST.IMG_SIZE):
        x_buf.append(0)
    for _ in range(BATCH * MNIST.NUM_CLASSES):
        y_buf.append(0)
    for _ in range(BATCH * MLP.OUT_DIM):
        out_buf.append(0)
        go_buf.append(0)
    for _ in range(BATCH * MLP.IN_DIM):
        gi_buf.append(0)
    for _ in range(BATCH * MLP.CACHE_SIZE):
        cache_buf.append(0)

    var x_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, MNIST.IMG_SIZE), MutAnyOrigin
    ](x_buf.unsafe_ptr())
    var y_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, MNIST.NUM_CLASSES), MutAnyOrigin
    ](y_buf.unsafe_ptr())
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, MLP.OUT_DIM), MutAnyOrigin
    ](out_buf.unsafe_ptr())
    var go_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, MLP.OUT_DIM), MutAnyOrigin
    ](go_buf.unsafe_ptr())
    var gi_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, MLP.IN_DIM), MutAnyOrigin
    ](gi_buf.unsafe_ptr())
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, MLP.CACHE_SIZE), MutAnyOrigin
    ](cache_buf.unsafe_ptr())

    var params = state.params_view()
    var grads = state.grads_view()
    var opt_state = state.opt_state_view()
    var model_state = state.model_state_view()
    var opt_global = state.opt_global_state_view()

    # Shuffle indices, re-permuted each epoch.
    var perm = List[Int](capacity=MNIST.N_TRAIN)
    for i in range(MNIST.N_TRAIN):
        perm.append(i)

    var batches_per_epoch = MNIST.N_TRAIN // BATCH
    var last_acc: Float64 = 0.0
    var last_test_loss: Float64 = 0.0

    print("\n── Training ──")
    var t0 = perf_counter_ns()
    for epoch in range(EPOCHS):
        _fisher_yates(perm, MNIST.N_TRAIN)

        var epoch_loss: Float64 = 0.0
        for b in range(batches_per_epoch):
            var base = b * BATCH
            # Assemble minibatch from shuffled indices.
            for s in range(BATCH):
                var sample = perm[base + s]
                var src = sample * MNIST.IMG_SIZE
                for k in range(MNIST.IMG_SIZE):
                    x_t[s, k] = ds.train_images[src + k]
                # One-hot target (zero then set the active class).
                for c in range(MNIST.NUM_CLASSES):
                    y_t[s, c] = 0
                y_t[s, Int(ds.train_labels[sample])] = 1

            MLP.forward[BATCH](x_t, out_t, params, model_state, cache_t)

            var loss = CrossEntropyLoss.forward[BATCH, MLP.OUT_DIM](out_t, y_t)
            CrossEntropyLoss.backward[BATCH, MLP.OUT_DIM](out_t, y_t, go_t)

            state.zero_grads()
            MLP.backward[BATCH](
                go_t, gi_t, params, model_state, cache_t, grads
            )

            state.step_num += 1
            Adam[LR=0.001].step[MLP.PARAM_SIZE](
                params, grads, opt_state, opt_global, state.step_num
            )

            epoch_loss = loss

        # ── Per-epoch eval on the test set ──
        var acc = _eval_accuracy(
            params,
            model_state,
            ds.test_images,
            ds.test_labels,
            MNIST.N_TEST,
        )
        last_acc = acc

        # Optional: also recompute test loss on a single batch so the user has
        # a comparable signal vs the GPU example (not the full-set CE).
        var tl_x_off = 0
        for s in range(BATCH):
            var src = (tl_x_off + s) * MNIST.IMG_SIZE
            for k in range(MNIST.IMG_SIZE):
                x_t[s, k] = ds.test_images[src + k]
            for c in range(MNIST.NUM_CLASSES):
                y_t[s, c] = 0
            y_t[s, Int(ds.test_labels[tl_x_off + s])] = 1
        MLP.forward[BATCH](x_t, out_t, params, model_state)
        last_test_loss = CrossEntropyLoss.forward[BATCH, MLP.OUT_DIM](
            out_t, y_t
        )

        print(
            "  epoch "
            + String(epoch + 1)
            + "/"
            + String(EPOCHS)
            + "  loss="
            + String(epoch_loss)[byte=:7]
            + "  test_loss="
            + String(last_test_loss)[byte=:7]
            + "  top1="
            + String(last_acc * 100.0)[byte=:6]
            + "%"
        )

    var t1 = perf_counter_ns()
    print(
        "  training time: " + String(Float64(t1 - t0) / 1e9)[byte=:6] + " s"
    )

    # ── Final report ──
    print("\n── Final evaluation (full test set) ──")
    print(
        "  test_loss="
        + String(last_test_loss)
        + "  top1="
        + String(last_acc * 100.0)[byte=:6]
        + "%"
    )

    print("=" * 65)
    if last_acc >= 0.97:
        print("PASS — MLP converges on MNIST (>=97%)")
    else:
        print(
            "FAIL — expected >=97% test accuracy, got " + String(last_acc)
        )
        raise Error("accuracy below threshold")
