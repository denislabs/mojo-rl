"""GPT + ViT instantiation / forward / backward smoke (Wave D).

The deeply-nested generic specialization (Sequential→Repeat→
TransformerBlock→Residual→Sequential→Tokenwise→…) is exactly where
compile or wiring issues surface. This test builds tiny GPT and ViT
stacks, runs forward + vjp on CPU, and asserts finite outputs and that
gradients flow (nonzero grad_input). Docs: NN2_TRANSFORMER_PORT.md.
"""

from std.memory import alloc
from std.math import abs
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.models.gpt import GPT
from mojo_rl.nn.models.vit import ViT


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        alloc[Scalar[DT]](n)
    )


def _all_finite(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int) -> Bool:
    for i in range(n):
        var v = Float64(p[i])
        if not (v == v) or abs(v) > 1e30:
            return False
    return True


def _sum_abs(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int) -> Float64:
    var s: Float64 = 0.0
    for i in range(n):
        s += abs(Float64(p[i]))
    return s


def test_gpt_smoke() raises:
    print("test_gpt_smoke ...")
    comptime VOCAB = 8
    comptime SEQ = 4
    comptime EMBED = 8
    comptime HEADS = 2
    comptime LAYERS = 2
    comptime BATCH = 2
    comptime IN_N = BATCH * SEQ * VOCAB
    comptime OUT_N = BATCH * SEQ * VOCAB

    var net = GPT[VOCAB, SEQ, EMBED, HEADS, LAYERS].make[
        target="cpu", INIT=Kaiming
    ]()

    var x = _alloc(IN_N)
    var y = _alloc(OUT_N)
    var go = _alloc(OUT_N)
    var gi = _alloc(IN_N)
    # One-hot tokens: position t of sample b → token (b + t) % VOCAB.
    for i in range(IN_N):
        x[i] = 0.0
    for b in range(BATCH):
        for t in range(SEQ):
            x[b * SEQ * VOCAB + t * VOCAB + ((b + t) % VOCAB)] = 1.0
    for i in range(OUT_N):
        go[i] = Scalar[DT](0.01 * Float64((i % 7) - 3))

    var x_t = TileTensor(x, row_major[BATCH, SEQ * VOCAB]())
    var y_t = TileTensor(y, row_major[BATCH, SEQ * VOCAB]())
    net.forward["cpu", BATCH](x_t, output=y_t)
    assert_true(_all_finite(y, OUT_N), "GPT forward finite")

    net.zero_grad["cpu"]()
    var go_t = TileTensor(go, row_major[BATCH, SEQ * VOCAB]())
    var gi_t = TileTensor(gi, row_major[BATCH, SEQ * VOCAB]())
    net.vjp["cpu", BATCH](go_t, gi_t)
    assert_true(_all_finite(gi, IN_N), "GPT grad_input finite")
    var gsum = _sum_abs(gi, IN_N)
    print("   GPT grad_input |·|sum =", gsum)
    assert_true(gsum > 0.0, "GPT gradients flow")
    print("  ok")


def test_vit_smoke() raises:
    print("test_vit_smoke ...")
    comptime IC = 1
    comptime H = 8
    comptime W = 8
    comptime PATCH = 4
    comptime EMBED = 16
    comptime HEADS = 2
    comptime LAYERS = 2
    comptime NPATCH = (H // PATCH) * (W // PATCH)  # 4
    comptime CLASSES = 3
    comptime BATCH = 2
    comptime IN_N = BATCH * IC * H * W
    comptime OUT_N = BATCH * CLASSES

    var net = ViT[
        IC, H, W, PATCH, EMBED, HEADS, LAYERS, NPATCH, CLASSES
    ].make[target="cpu", INIT=Kaiming]()

    var x = _alloc(IN_N)
    var y = _alloc(OUT_N)
    var go = _alloc(OUT_N)
    var gi = _alloc(IN_N)
    for i in range(IN_N):
        var t = 0.7 * Float64(i)
        x[i] = Scalar[DT](0.3 * (t - 6.2831853 * Float64(Int(t / 6.2831853))))
    for i in range(OUT_N):
        go[i] = Scalar[DT](0.1 * Float64((i % 3) - 1))

    var x_t = TileTensor(x, row_major[BATCH, IC * H * W]())
    var y_t = TileTensor(y, row_major[BATCH, CLASSES]())
    net.forward["cpu", BATCH](x_t, output=y_t)
    assert_true(_all_finite(y, OUT_N), "ViT forward finite")

    net.zero_grad["cpu"]()
    var go_t = TileTensor(go, row_major[BATCH, CLASSES]())
    var gi_t = TileTensor(gi, row_major[BATCH, IC * H * W]())
    net.vjp["cpu", BATCH](go_t, gi_t)
    assert_true(_all_finite(gi, IN_N), "ViT grad_input finite")
    var gsum = _sum_abs(gi, IN_N)
    print("   ViT grad_input |·|sum =", gsum)
    assert_true(gsum > 0.0, "ViT gradients flow")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("GPT + ViT instantiation/forward/backward smoke (Wave D)")
    print("=" * 70)
    test_gpt_smoke()
    test_vit_smoke()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
