"""Two-hot value encode/decode round-trip (run on NVIDIA AND apple; compare).

Gumbel MuZero's POLICY target is derived from the value/Q estimates, so a broken
value path (two-hot encode or decode) corrupts the policy targets → dead policy.
This checks the reference math: for a sweep of raw scalars s, encode → two-hot
distribution q, feed log(q) as logits to the decoder, and require decode(s) ≈ s.

CPU-only host functions (same on every backend) — but run it on NVIDIA too: if
it ever differs from apple, the toolchain miscompiles the host transform. The
GPU MCTS kernel's INLINE decode must stay bit-identical to this; if this passes
but value still won't learn on NVIDIA, suspect the GPU kernel/transfer, not the
math. Run: pixi run -e nvidia mojo run -I . tests/deep_agents/test_mz_twohot_roundtrip.mojo
"""

from std.math import log
from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.zero.twohot_targets import (
    mz_two_hot_target_batch, mz_decode_value_batch,
)


def main() raises:
    comptime BINS = 51
    var v_min = Scalar[DT](-1.0)
    var v_max = Scalar[DT](1.0)
    # h-space support for a raw value in roughly [-1, 1] is small; use a support
    # that covers h([-1,1]) — h(1)=~0.414+eps. Keep [-1,1] (matches the C4 run's
    # BINS=51 over [-1,1]); test raw scalars inside the decodable range.
    var samples = List[Scalar[DT]]()
    samples.append(Scalar[DT](-0.9))
    samples.append(Scalar[DT](-0.5))
    samples.append(Scalar[DT](-0.1))
    samples.append(Scalar[DT](0.0))
    samples.append(Scalar[DT](0.1))
    samples.append(Scalar[DT](0.5))
    samples.append(Scalar[DT](0.9))

    var worst = Scalar[DT](0.0)
    for si in range(len(samples)):
        var s = samples[si]
        # encode s -> two-hot distribution q [BINS]
        var sc = List[Scalar[DT]](length=1, fill=s)
        var q = List[Scalar[DT]](length=BINS, fill=0)
        mz_two_hot_target_batch[1, BINS](sc, 0, v_min, v_max, q, 0)
        # logits = log(q) (so softmax(logits) == q); guard log(0)
        var logits = List[Scalar[DT]](length=BINS, fill=0)
        for i in range(BINS):
            var qi = Float64(q[i])
            logits[i] = Scalar[DT](log(qi)) if qi > 1e-12 else Scalar[DT](-50.0)
        var dec = List[Scalar[DT]](length=1, fill=0)
        mz_decode_value_batch[1, BINS](logits, 0, v_min, v_max, dec, 0)
        var err = abs(dec[0] - s)
        if err > worst:
            worst = err
        print("  s =", s, " -> decoded =", dec[0], " | err =", err)

    print("worst round-trip err =", worst, " (want < ~1e-3)")
    if worst > Scalar[DT](1e-2):
        raise Error("two-hot round-trip BROKEN — value encode/decode mismatch")
    print("TWO-HOT ROUND-TRIP OK")
