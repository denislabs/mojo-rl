"""Test MuZero utility functions."""

from mojo_rl.deep_agents.muzero.utils import (
    scalar_transform,
    inverse_scalar_transform,
    MinMaxStats,
    compute_support,
    encode_categorical,
    decode_categorical,
    cross_entropy_with_softmax,
)
from std.memory import alloc, memset


fn main():
    print("=== MuZero Utils Tests ===")

    # Test scalar transform roundtrip
    var x = Float64(10.0)
    var h = scalar_transform(x)
    var x_recovered = inverse_scalar_transform(h)
    print("transform(10.0) =", h)
    print("inverse(transform(10.0)) =", x_recovered)

    var err = x_recovered - x
    if err < 0.0:
        err = -err
    if err < 0.1:
        print("PASS: scalar_transform roundtrip")
    else:
        print("FAIL: scalar_transform roundtrip, error =", err)

    # Test negative
    var h_neg = scalar_transform(-5.0)
    var neg_recovered = inverse_scalar_transform(h_neg)
    print("transform(-5.0) =", h_neg, "inverse =", neg_recovered)
    var err_neg = neg_recovered - (-5.0)
    if err_neg < 0.0:
        err_neg = -err_neg
    if err_neg < 0.1:
        print("PASS: negative scalar_transform roundtrip")
    else:
        print("FAIL: negative scalar_transform roundtrip, error =", err_neg)

    # Test MinMaxStats
    var mm = MinMaxStats()
    mm.update(1.0)
    mm.update(5.0)
    var norm_3 = mm.normalize(3.0)
    print("MinMax normalize(3.0) in [1, 5] =", norm_3)
    if norm_3 > 0.49 and norm_3 < 0.51:
        print("PASS: MinMaxStats normalize")
    else:
        print("FAIL: MinMaxStats normalize")

    # Test categorical encoding/decoding
    comptime BINS = 11
    var target = alloc[Float64](BINS)
    encode_categorical[BINS](0.5, -1.0, 1.0, target)
    print("two-hot encode(0.5) in [-1, 1] with 11 bins:")
    for i in range(BINS):
        if target[i] > 0.001:
            print("  bin[", i, "] =", target[i])

    # Test decode
    var logits = alloc[Float64](BINS)
    for i in range(BINS):
        logits[i] = Float64(0.0)
    logits[7] = Float64(10.0)  # Strongly peaked at bin 7 = -1 + 7*0.2 = 0.4
    logits[8] = Float64(10.0)  # and bin 8 = 0.6
    var decoded = decode_categorical[BINS](logits, -1.0, 1.0)
    print("decode peaked at bins 7,8 =", decoded, "(expect ~0.5)")

    target.free()
    logits.free()

    print("=== Done ===")
