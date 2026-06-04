"""ViT learning lighthouse (Wave D) — overfit a synthetic patch task.

Proves the full ViT stack (PatchEmbed → pos BiasAdd → TransformerBlocks →
LayerNorm → TokenMean → head) trains end-to-end through the nn2
Trainer + Adam + CrossEntropy. The task: a 1×8×8 image whose class
(0/1/2) is encoded by which 4×4 quadrant (= which patch) is bright. This
is linearly separable in patch space, so a correct ViT should drive train
accuracy to ~100% in a few dozen full-batch steps.

Run: pixi run mojo run -I . tests/nn2/test_vit_learns.mojo
Docs: docs/NN2_TRANSFORMER_PORT.md Phase 1 Wave D.
"""

from std.memory import alloc
from std.random import seed
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.composites import ViT
from mojo_rl.nn2.loss import CrossEntropyLoss
from mojo_rl.nn2.optimizer import Adam
from mojo_rl.nn2.training import Trainer
from mojo_rl.nn2.initializer import Kaiming


comptime IC = 1
comptime H = 8
comptime W = 8
comptime PATCH = 4
comptime EMBED = 16
comptime HEADS = 2
comptime LAYERS = 2
comptime NPATCH = (H // PATCH) * (W // PATCH)   # 4
comptime CLASSES = 3
comptime BATCH = 48
comptime STEPS = 80
comptime LR: Scalar[DT] = 0.01
comptime IN_DIM = IC * H * W


def _noise(i: Int, s: Int) -> Float64:
    var x = 0.31 * Float64(i) + 0.123 * Float64(s)
    var t = x - 6.2831853 * Float64(Int(x / 6.2831853))
    return 0.05 * (t - (t * t * t) / 6.0)


def _make_dataset(
    in_buf: UnsafePointer[Scalar[DT], MutAnyOrigin],
    tgt_buf: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    """Sample s → class s%3; brighten the s%3-th 4×4 quadrant."""
    for s in range(BATCH):
        var c = s % CLASSES
        var pr = c // 2          # patch row (0/1)
        var pc = c % 2           # patch col (0/1)
        var r0 = pr * PATCH
        var c0 = pc * PATCH
        for px in range(IN_DIM):
            var row = px // W
            var col = px % W
            var base = _noise(px, s)
            var bright = (
                1.0 if (row >= r0 and row < r0 + PATCH
                        and col >= c0 and col < c0 + PATCH) else 0.0
            )
            in_buf[s * IN_DIM + px] = Scalar[DT](bright + base)
        for k in range(CLASSES):
            tgt_buf[s * CLASSES + k] = 0.0
        tgt_buf[s * CLASSES + c] = 1.0


def main() raises:
    seed(7)
    print("=" * 70)
    print("ViT learning lighthouse (Wave D) — synthetic patch task")
    print("=" * 70)

    var net = ViT[
        IC, H, W, PATCH, EMBED, HEADS, LAYERS, NPATCH, CLASSES
    ].make[target="cpu", INIT=Kaiming]()
    var loss_fn = CrossEntropyLoss[CLASSES].make["cpu"]()
    var optim = Adam.make["cpu", M = type_of(net)](net)
    optim.lr = LR
    var trainer = Trainer[
        type_of(net), type_of(optim), type_of(loss_fn), BATCH, target="cpu",
    ].make_from(net^, optim^, loss_fn^)

    var in_buf = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        alloc[Scalar[DT]](BATCH * IN_DIM)
    )
    var tgt_buf = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        alloc[Scalar[DT]](BATCH * CLASSES)
    )
    var out_buf = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        alloc[Scalar[DT]](BATCH * CLASSES)
    )
    _make_dataset(in_buf, tgt_buf)

    var first_loss: Float64 = 0.0
    var last_loss: Float64 = 0.0
    for step in range(STEPS):
        var l = Float64(trainer.train_step(in_buf, tgt_buf))
        if step == 0:
            first_loss = l
        last_loss = l
        if step % 10 == 0 or step == STEPS - 1:
            print("  step " + String(step) + "  loss=" + String(l)[byte=:7])

    # Train accuracy.
    trainer.predict(in_buf, out_buf)
    var correct = 0
    for s in range(BATCH):
        var best_c = 0
        var best_v = out_buf[s * CLASSES + 0]
        for k in range(1, CLASSES):
            if out_buf[s * CLASSES + k] > best_v:
                best_v = out_buf[s * CLASSES + k]
                best_c = k
        if best_c == s % CLASSES:
            correct += 1
    var acc = Float64(correct) / Float64(BATCH)

    print("-" * 70)
    print(
        "  first_loss=" + String(first_loss)[byte=:7]
        + "  last_loss=" + String(last_loss)[byte=:7]
        + "  train_acc=" + String(acc * 100.0)[byte=:6] + "%"
    )
    in_buf.free()
    tgt_buf.free()
    out_buf.free()

    print("=" * 70)
    if last_loss < first_loss * 0.5 and acc >= 0.95:
        print("PASS — ViT learns the synthetic patch task")
    else:
        raise Error(
            "ViT failed to learn: first=" + String(first_loss)
            + " last=" + String(last_loss) + " acc=" + String(acc)
        )
