"""MuZero two-hot target encode/decode round-trip — pure CPU, no GPU.

Validates `zero/twohot_targets.mojo`: (1) h / h⁻¹ are inverses, (2) two-hot
targets are a valid distribution (non-negative, sum to 1, ≤2 nonzero bins),
(3) encode→decode recovers the raw scalar within bin resolution. The decode
mirrors the GPU MCTS kernel's inline categorical decode, so this also pins the
train/search numeric contract.

Run:
    pixi run mojo run -I . tests/deep_agents2/test_mz_twohot_targets.mojo
"""

from std.memory import alloc
from std.math import log
from std.testing import assert_true, assert_almost_equal

from mojo_rl.nn2.constants import DT
from mojo_rl.deep_agents2.zero.twohot_targets import (
    mz_scalar_transform,
    mz_inverse_scalar_transform,
    mz_two_hot_target_batch,
    mz_decode_value_batch,
)


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](alloc[Scalar[DT]](n))


def main() raises:
    comptime BATCH = 6
    comptime NUM_BINS = 51
    var v_min = Scalar[DT](-10.0)
    var v_max = Scalar[DT](10.0)

    # ── (1) h / h⁻¹ inverse round-trip across a wide scalar range ──
    var xs = [
        Scalar[DT](0.0), Scalar[DT](0.5), Scalar[DT](-0.5),
        Scalar[DT](1.0), Scalar[DT](-7.3), Scalar[DT](42.0),
        Scalar[DT](-500.0), Scalar[DT](500.0),
    ]
    for i in range(len(xs)):
        var y = mz_scalar_transform(xs[i])
        var x_rt = mz_inverse_scalar_transform(y)
        assert_almost_equal(
            x_rt, xs[i], atol=1e-3, rtol=1e-4,
            msg=String("h/h^-1 not inverse for ") + String(xs[i]),
        )
    # h compresses: h(500) should be far smaller than 500.
    assert_true(
        mz_scalar_transform(Scalar[DT](500.0)) < Scalar[DT](25.0),
        "scalar transform did not compress large value",
    )
    print("(1) h / h^-1 inverse: OK")

    # ── (2) two-hot targets are a valid distribution ──
    # Raw values whose h(x) lands inside [v_min, v_max].
    var raw = _alloc(BATCH)
    raw[0] = Scalar[DT](0.0)
    raw[1] = Scalar[DT](1.7)
    raw[2] = Scalar[DT](-2.3)
    raw[3] = Scalar[DT](5.0)
    raw[4] = Scalar[DT](-8.0)
    raw[5] = Scalar[DT](3.14159)

    var tgt = _alloc(BATCH * NUM_BINS)
    mz_two_hot_target_batch[BATCH, NUM_BINS](raw, v_min, v_max, tgt)
    for b in range(BATCH):
        var s = Scalar[DT](0.0)
        var nnz = 0
        for i in range(NUM_BINS):
            var p = tgt[b * NUM_BINS + i]
            # nn2 two_hot can emit a ~1e-7 negative at a bin boundary (fp).
            assert_true(p >= Scalar[DT](-1e-5), "negative two-hot weight")
            s += p
            if p > Scalar[DT](1e-6):
                nnz += 1
        assert_almost_equal(
            s, Scalar[DT](1.0), atol=1e-5, rtol=1e-5,
            msg=String("two-hot row ") + String(b) + " does not sum to 1",
        )
        assert_true(nnz <= 2, "two-hot row has >2 nonzero bins")
    print("(2) two-hot distribution validity: OK")

    # ── (3) encode → (treat target as softmax probs) decode round-trip ──
    # Convert the two-hot probs to logits via log so softmax recovers them,
    # then decode and compare to the raw value (within bin resolution).
    var logits = _alloc(BATCH * NUM_BINS)
    for b in range(BATCH):
        for i in range(NUM_BINS):
            var p = tgt[b * NUM_BINS + i]
            logits[b * NUM_BINS + i] = (
                log(p) if p > Scalar[DT](1e-12) else Scalar[DT](-50.0)
            )
    var decoded = _alloc(BATCH)
    mz_decode_value_batch[BATCH, NUM_BINS](logits, v_min, v_max, decoded)
    # bin resolution in h-space = 20/50 = 0.4; h⁻¹ expands it near the edges,
    # so allow a modest absolute tolerance.
    for b in range(BATCH):
        assert_almost_equal(
            decoded[b], raw[b], atol=2e-2, rtol=1e-2,
            msg=String("decode round-trip failed for row ") + String(b),
        )
    print("(3) encode->decode round-trip: OK")

    raw.free()
    tgt.free()
    logits.free()
    decoded.free()
    print("MuZero two-hot targets (h-transform + linear two-hot): OK")
