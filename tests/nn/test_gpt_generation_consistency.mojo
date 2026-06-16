"""GPT generation-feeding vs teacher-forcing consistency.

The generation harness feeds a front-anchored window: real tokens at positions
0..k, pad (token 0 = '\\n') at k+1..SEQ-1, and reads logits at position k.
Causality guarantees output[k] depends only on tokens 0..k, so these logits
MUST equal the teacher-forced logits at position k from the full sequence.

If they differ, the generation harness is the bug (explains: teacher-forced
eval good, free generation garbage). If they match, the harness is correct and
the generation problem is the model (exposure bias), not the feeding.

Tests the real GPTDrop end-to-end (Embedding+pos+blocks+LN+head), BATCH=1,
eval mode, both for an interior position and the boundary.
"""

from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.math import abs
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.models.gpt import GPTDrop
from mojo_rl.nn.initializer import Normal


comptime VOCAB = 8
comptime SEQ = 6
comptime EMBED = 16
comptime HEADS = 2
comptime LAYERS = 2
comptime IN_DIM = SEQ * VOCAB
comptime OUT_DIM = SEQ * VOCAB
comptime MODEL = GPTDrop[VOCAB, SEQ, EMBED, HEADS, LAYERS, 4, True]


def _mao(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def _fwd(
    mut net: MODEL,
    ctx: DeviceContext,
    toks: List[Int],
    in_h: HostBuffer[DT],  # reused host buffer
    in_d: DeviceBuffer[DT],
    out_d: DeviceBuffer[DT],
    out_h: HostBuffer[DT],
) raises:
    """One-hot `toks` (front-anchored, pad=0 for t>=len) → forward → out_h."""
    for i in range(IN_DIM):
        in_h.unsafe_ptr()[i] = 0.0
    for t in range(SEQ):
        var tid = toks[t] if t < len(toks) else 0
        in_h.unsafe_ptr()[t * VOCAB + tid] = 1.0
    ctx.enqueue_copy(in_d, in_h)
    var in_t = TileTensor(_mao(in_d), row_major[1, IN_DIM]())
    var out_t = TileTensor(_mao(out_d), row_major[1, OUT_DIM]())
    net.forward["gpu", 1](in_t, output=out_t)
    ctx.enqueue_copy(out_h, out_d)
    ctx.synchronize()


def main() raises:
    print("=" * 70)
    print("GPT generation-feeding vs teacher-forcing consistency")
    print("=" * 70)
    var ctx = DeviceContext()
    var net = MODEL.make["gpu", INIT = Normal[0.0, 0.02]](ctx)
    net.set_attr["training"](Scalar[DT](0.0))  # eval: dropout off

    var in_h = ctx.enqueue_create_host_buffer[DT](IN_DIM)
    var in_d = ctx.enqueue_create_buffer[DT](IN_DIM)
    var out_d = ctx.enqueue_create_buffer[DT](OUT_DIM)
    var out_h = ctx.enqueue_create_host_buffer[DT](OUT_DIM)
    var full_h = ctx.enqueue_create_host_buffer[DT](OUT_DIM)
    ctx.synchronize()

    # Full sequence (teacher-forcing reference).
    var full = List[Int]()
    full.append(3); full.append(1); full.append(5); full.append(2)
    full.append(6); full.append(0)
    _fwd(net, ctx, full, in_h, in_d, out_d, full_h)
    # snapshot full logits
    var ref_logits = List[Scalar[DT]]()
    for i in range(OUT_DIM):
        ref_logits.append(full_h.unsafe_ptr()[i])

    var max_diff: Float64 = 0.0
    # For each read position k, feed only tokens 0..k (pad rest) and compare
    # logits at position k against the teacher-forced reference.
    for k in range(SEQ):
        var prefix = List[Int]()
        for t in range(k + 1):
            prefix.append(full[t])
        _fwd(net, ctx, prefix, in_h, in_d, out_d, out_h)
        var d_k: Float64 = 0.0
        for v in range(VOCAB):
            var diff = abs(
                Float64(out_h.unsafe_ptr()[k * VOCAB + v])
                - Float64(ref_logits[k * VOCAB + v])
            )
            if diff > d_k:
                d_k = diff
        print("   read_pos k=", k, " max|gen - teacher| =", d_k)
        if d_k > max_diff:
            max_diff = d_k

    print("-" * 70)
    print("   overall max diff =", max_diff)
    assert_true(
        max_diff < 1e-4,
        "generation feeding must match teacher-forcing at each read position",
    )
    print("  ok — generation harness is consistent with teacher-forcing")
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
