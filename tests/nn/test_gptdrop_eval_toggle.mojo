"""GPTDrop dropout eval/train toggle — regression (generation-quality port).

Validates that `net.set_attr["training"](v)` propagates through the whole
GPTDrop composite (Sequential→Repeat→TransformerBlockDrop→Residual→…→Dropout)
and that dropout is correctly active in train mode / disabled in eval mode:

  - training=False  → two forwards over the same input are IDENTICAL
                      (dropout is identity, deterministic).
  - training=True   → two forwards DIFFER (fresh Bernoulli masks per forward
                      via the host-side counter bump).

If set_attr didn't reach the Dropout leaves, eval-mode forwards would still
differ (dropout stuck on) — so the identical-eval assertion is the real test.

    pixi run -e apple  mojo run -I . tests/nn/test_gptdrop_eval_toggle.mojo
    pixi run -e nvidia mojo run -I . tests/nn/test_gptdrop_eval_toggle.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import abs
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.models.gpt import GPTDrop
from mojo_rl.nn.initializer import Normal


comptime VOCAB = 8
comptime SEQ = 4
comptime EMBED = 8
comptime HEADS = 2
comptime LAYERS = 2
comptime BATCH = 2
comptime IN_DIM = SEQ * VOCAB
comptime OUT_DIM = SEQ * VOCAB
# High p makes the train-mode difference obvious. use_max=False keeps the
# compile light (the toggle is independent of the attention path).
comptime MODEL = GPTDrop[
    VOCAB, SEQ, EMBED, HEADS, LAYERS, 4, True, 0.5, UInt64(123), False
]


def _mao(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def _maxdiff(
    a: UnsafePointer[Scalar[DT], MutAnyOrigin],
    b: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n: Int,
) -> Float64:
    var m: Float64 = 0.0
    for i in range(n):
        var d = abs(Float64(a[i]) - Float64(b[i]))
        if d > m:
            m = d
    return m


def main() raises:
    print("=" * 70)
    print("GPTDrop dropout eval/train toggle")
    print("=" * 70)
    var ctx = DeviceContext()
    var net = MODEL.make["gpu", INIT = Normal[0.0, 0.02]](ctx)

    # One-hot input (token (b+t)%VOCAB at position t).
    var xh = ctx.enqueue_create_host_buffer[DT](BATCH * IN_DIM)
    ctx.synchronize()
    for i in range(BATCH * IN_DIM):
        xh.unsafe_ptr()[i] = 0.0
    for b in range(BATCH):
        for t in range(SEQ):
            xh.unsafe_ptr()[b * IN_DIM + t * VOCAB + ((b + t) % VOCAB)] = 1.0
    var xd = ctx.enqueue_create_buffer[DT](BATCH * IN_DIM)
    ctx.enqueue_copy(xd, xh)
    ctx.synchronize()
    var x_tt = TileTensor(_mao(xd), row_major[BATCH, IN_DIM]())

    var o1 = ctx.enqueue_create_buffer[DT](BATCH * OUT_DIM)
    var o2 = ctx.enqueue_create_buffer[DT](BATCH * OUT_DIM)
    var h1 = ctx.enqueue_create_host_buffer[DT](BATCH * OUT_DIM)
    var h2 = ctx.enqueue_create_host_buffer[DT](BATCH * OUT_DIM)

    # ── Eval mode: two forwards must be identical ──
    net.set_attr["training"](Scalar[DT](0.0))
    var o1t = TileTensor(_mao(o1), row_major[BATCH, OUT_DIM]())
    var o2t = TileTensor(_mao(o2), row_major[BATCH, OUT_DIM]())
    net.forward["gpu", BATCH](x_tt, output=o1t)
    net.forward["gpu", BATCH](x_tt, output=o2t)
    ctx.enqueue_copy(h1, o1)
    ctx.enqueue_copy(h2, o2)
    ctx.synchronize()
    var eval_diff = _maxdiff(h1.unsafe_ptr(), h2.unsafe_ptr(), BATCH * OUT_DIM)
    print("   eval-mode diff (must be 0) =", eval_diff)
    assert_true(eval_diff == 0.0, "eval-mode forwards must be deterministic")

    # ── Train mode: two forwards must DIFFER (fresh masks) ──
    net.set_attr["training"](Scalar[DT](1.0))
    var o3t = TileTensor(_mao(o1), row_major[BATCH, OUT_DIM]())
    var o4t = TileTensor(_mao(o2), row_major[BATCH, OUT_DIM]())
    net.forward["gpu", BATCH](x_tt, output=o3t)
    net.forward["gpu", BATCH](x_tt, output=o4t)
    ctx.enqueue_copy(h1, o1)
    ctx.enqueue_copy(h2, o2)
    ctx.synchronize()
    var train_diff = _maxdiff(h1.unsafe_ptr(), h2.unsafe_ptr(), BATCH * OUT_DIM)
    print("   train-mode diff (must be > 0) =", train_diff)
    assert_true(train_diff > 1e-6, "train-mode forwards must differ (dropout on)")

    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
