"""Backprop MLP baseline on CIFAR-10 — apples-to-apples comparison vs PCN.

Same architecture as the arxiv 2506.06332 PCN setup:
    Linear[3072, 1000] + ReLU
    Linear[1000, 500]  + ReLU
    Linear[500,  10]   + ReLU
    Linear[10,   10]                # classifier (readout)

3,578,620 trainable params (3.58M; same OOM as PCN's 3.58M, plus
~1520 bias terms — PCN has no biases).

Training: standard SGD with mini-batches, Adam, cross-entropy.
Reports test top-1 / top-3 accuracy. Honest generalization metric (no label
leakage).

Comparison target — PCN reference numbers on the same architecture
(per arxiv 2506.06332 + GitHub-issue replications + our own runs):
    Supervised inference (paper headline, label-leak protocol): 99.92% / 99.99%
    Free inference (honest):                                    11.6 – 19% / ~33%

This BP baseline tells us what's actually achievable with this architecture.
PCN's free-inference number is the fair comparison.

Run:
    pixi run -e nvidia mojo run -I . tests/nn_pc/test_bp_mlp_cifar10_baseline.mojo
    pixi run -e apple  mojo run -I . tests/nn_pc/test_bp_mlp_cifar10_baseline.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model.linear import Linear
from mojo_rl.nn.model.linear_act import LinearReLU
from mojo_rl.nn.model.sequential import Sequential
from mojo_rl.nn.loss.cross_entropy import CrossEntropyLoss
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.training.trainer import Trainer
from mojo_rl.nn.initializer.initializers import Kaiming
from mojo_rl.nn.datasets.cifar10 import CIFAR10


comptime BATCH = 128
comptime EPOCHS = 10
comptime LR: Float64 = 0.001

# Architecture mirrors the PCN paper's MLP exactly (3 hidden + readout).
comptime MLP = Sequential[
    LinearReLU[3072, 1000],
    LinearReLU[1000, 500],
    LinearReLU[500, 10],
    Linear[10, 10],   # classifier — produces 10-way logits
]


def main() raises:
    seed(42)
    print("=" * 65)
    print("CIFAR-10 backprop MLP baseline — same arch as PCN paper")
    print("=" * 65)
    print("  arch       : 3072 → 1000 → 500 → 10 (ReLU) → 10 (logits)")
    print("  params     :", MLP.PARAM_SIZE)
    print(
        "  hyperparams: BATCH=", BATCH, " EPOCHS=", EPOCHS,
        " LR=", LR, " (Adam, Kaiming init, cross-entropy)",
    )

    var ds = CIFAR10()
    var ctx = DeviceContext()

    comptime TRAINER = Trainer[MLP, Adam[LR=LR], CrossEntropyLoss]
    var state = TRAINER.init_state_gpu[Kaiming[]](ctx)

    # ── Upload full training set + one-hot labels to GPU once ──
    var train_img_host = ctx.enqueue_create_host_buffer[dtype](
        CIFAR10.N_TRAIN * CIFAR10.IMG_SIZE
    )
    var train_tgt_host = ctx.enqueue_create_host_buffer[dtype](
        CIFAR10.N_TRAIN * CIFAR10.NUM_CLASSES
    )
    for i in range(CIFAR10.N_TRAIN * CIFAR10.IMG_SIZE):
        train_img_host.unsafe_ptr()[i] = ds.train_images[i]
    for i in range(CIFAR10.N_TRAIN * CIFAR10.NUM_CLASSES):
        train_tgt_host.unsafe_ptr()[i] = Scalar[dtype](0)
    for i in range(CIFAR10.N_TRAIN):
        train_tgt_host.unsafe_ptr()[
            i * CIFAR10.NUM_CLASSES + Int(ds.train_labels[i])
        ] = Scalar[dtype](1.0)

    var train_img_buf = ctx.enqueue_create_buffer[dtype](
        CIFAR10.N_TRAIN * CIFAR10.IMG_SIZE
    )
    var train_tgt_buf = ctx.enqueue_create_buffer[dtype](
        CIFAR10.N_TRAIN * CIFAR10.NUM_CLASSES
    )
    ctx.enqueue_copy(train_img_buf, train_img_host)
    ctx.enqueue_copy(train_tgt_buf, train_tgt_host)

    var train_img_lt = LayoutTensor[
        dtype, Layout.row_major(CIFAR10.N_TRAIN, CIFAR10.IMG_SIZE), MutAnyOrigin
    ](train_img_buf)
    var train_tgt_lt = LayoutTensor[
        dtype,
        Layout.row_major(CIFAR10.N_TRAIN, CIFAR10.NUM_CLASSES),
        MutAnyOrigin,
    ](train_tgt_buf)

    # ── Train via the existing minibatch trainer (handles shuffling) ──
    print("\n── Training ──")
    var t0 = perf_counter_ns()
    var result = TRAINER.train_gpu_minibatch[BATCH, CIFAR10.N_TRAIN](
        state,
        ctx,
        train_img_lt,
        train_tgt_lt,
        epochs=EPOCHS,
        print_every_batches=100,
        shuffle=True,
        rng_seed=UInt64(42),
    )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var train_time_s = Float64(t1 - t0) / 1e9
    print("  total training time: " + String(train_time_s)[byte=:6] + " s")
    print("  final batch loss   : " + String(result.final_loss)[byte=:8])

    # ── Evaluate on test set ──
    print("\n── Evaluating ──")
    var test_img_host = ctx.enqueue_create_host_buffer[dtype](
        CIFAR10.N_TEST * CIFAR10.IMG_SIZE
    )
    for i in range(CIFAR10.N_TEST * CIFAR10.IMG_SIZE):
        test_img_host.unsafe_ptr()[i] = ds.test_images[i]
    var test_img_buf = ctx.enqueue_create_buffer[dtype](
        CIFAR10.N_TEST * CIFAR10.IMG_SIZE
    )
    ctx.enqueue_copy(test_img_buf, test_img_host)

    comptime num_test_batches = CIFAR10.N_TEST // BATCH
    var output_buf = ctx.enqueue_create_buffer[dtype](BATCH * MLP.OUT_DIM)
    var workspace_buf = ctx.enqueue_create_buffer[dtype](
        BATCH * MLP.WORKSPACE_SIZE_PER_SAMPLE
    )
    var output_host = ctx.enqueue_create_host_buffer[dtype](BATCH * MLP.OUT_DIM)
    var output_lt = LayoutTensor[
        dtype, Layout.row_major(BATCH, MLP.OUT_DIM), MutAnyOrigin
    ](output_buf)

    var top1: Int = 0
    var top3: Int = 0
    var total: Int = 0
    for batch_idx in range(num_test_batches):
        var batch_input = LayoutTensor[
            dtype, Layout.row_major(BATCH, MLP.IN_DIM), MutAnyOrigin
        ](test_img_buf.unsafe_ptr() + batch_idx * BATCH * MLP.IN_DIM)

        var params_eval = state.params_view()
        MLP.forward_gpu_no_cache[BATCH](
            ctx,
            output_lt,
            batch_input,
            params_eval,
            state.model_state_view(),
            workspace_buf,
        )
        ctx.enqueue_copy(output_host, output_buf)
        ctx.synchronize()

        for b in range(BATCH):
            var true_label = Int(ds.test_labels[batch_idx * BATCH + b])
            # one-pass top-1 and top-3
            var t1_idx = -1; var t2_idx = -1; var t3_idx = -1
            var t1v = Float64(-1e30); var t2v = Float64(-1e30); var t3v = Float64(-1e30)
            for c in range(CIFAR10.NUM_CLASSES):
                var v = Float64(output_host.unsafe_ptr()[b * MLP.OUT_DIM + c])
                if v > t1v:
                    t3v = t2v; t3_idx = t2_idx
                    t2v = t1v; t2_idx = t1_idx
                    t1v = v; t1_idx = c
                elif v > t2v:
                    t3v = t2v; t3_idx = t2_idx
                    t2v = v; t2_idx = c
                elif v > t3v:
                    t3v = v; t3_idx = c
            if true_label == t1_idx:
                top1 += 1
            if true_label == t1_idx or true_label == t2_idx or true_label == t3_idx:
                top3 += 1
            total += 1

    var acc1 = Float64(top1) / Float64(total)
    var acc3 = Float64(top3) / Float64(total)
    print("  test top-1: " + String(top1) + " / " + String(total)
          + " = " + String(acc1 * 100.0)[byte=:6] + "%")
    print("  test top-3: " + String(top3) + " / " + String(total)
          + " = " + String(acc3 * 100.0)[byte=:6] + "%")

    print("=" * 65)
    print("Reference numbers on the SAME architecture:")
    print("  PCN supervised inference (paper headline): 99.92% top-1  (label leak)")
    print("  PCN free inference (honest):                ~12 – 19% top-1")
    print("  Backprop MLP (this run):                    "
          + String(acc1 * 100.0)[byte=:6] + "% top-1")
    print("=" * 65)
    if acc1 >= 0.40:
        print(
            "PASS — backprop MLP reaches >=40% on CIFAR-10 (typical for"
            " plain MLP with no convs / augmentation)."
        )
    else:
        print(
            "WARN — backprop MLP below 40%. Try more epochs or LR tuning."
        )
