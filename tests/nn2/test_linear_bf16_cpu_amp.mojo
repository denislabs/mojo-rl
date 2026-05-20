"""Phase 8.3 — positive validation of Bf16Compute on CPU Linear.

Three tests:

  1. test_forward_differs_under_bf16  — assert that calling
       `Linear.forward[POLICY=Bf16Compute]` produces a *different* output
       from `Linear.forward[POLICY=NoAMP]` on the same inputs and weights.
       This is the positive validation: if POLICY were ignored on CPU,
       both calls would route through fp32 and the outputs would be
       bit-identical. They aren't — bf16 quantizes the inner product.

  2. test_forward_bf16_within_quantization_tolerance — assert that the
       bf16 output is still close enough to fp32 to be a valid AMP
       result (max-rel-err < 5% per the probe).

  3. test_backward_input_bf16_routes_through_amp_path  — same shape
       check for `backward_input[POLICY=Bf16Compute]` — output differs
       from fp32 baseline (proves the AMP branch is taken in backward
       too), but stays within bf16 tolerance.
"""

from std.math import abs as fabs
from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core import NoAMP, Bf16Compute
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.initializer import Zero


def _set_weights(mut lin: Linear[4, 3]) raises:
    """Set deterministic non-trivial weights + biases."""
    var w = TileTensor(lin.weight, row_major[4, 3]())
    var b = TileTensor(lin.bias,   row_major[3]())
    var k = 0
    for i in range(4):
        for j in range(3):
            w[i, j] = Scalar[DT](Float32(k) * 0.13 - 0.5)
            k += 1
    for j in range(3):
        b[j] = Scalar[DT](Float32(j) * 0.07 + 0.1)


def test_forward_differs_under_bf16() raises:
    """The headline positive-validation test: NoAMP vs Bf16Compute must
    produce DIFFERENT outputs on the same inputs. If they're equal, the
    POLICY parameter is being silently dropped — exactly what Phase 8.3
    must prove isn't happening."""
    comptime IN = 4
    comptime OUT = 3
    comptime BATCH = 2

    var lin_fp32 = Linear[IN, OUT].make[target="cpu", INIT=Zero]()
    var lin_bf16 = Linear[IN, OUT].make[target="cpu", INIT=Zero]()
    _set_weights(lin_fp32)
    _set_weights(lin_bf16)

    # Build a non-trivial input. Use values that don't trivially round to
    # the same bf16 representation (so bf16 truncation actually matters).
    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var out_a:   UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var out_b:   UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    for i in range(BATCH * IN):
        in_buf[i] = Scalar[DT](Float32(i) * 0.317 - 0.4)

    var input_tt  = TileTensor(in_buf, row_major[BATCH, IN]())
    var out_a_tt  = TileTensor(out_a,  row_major[BATCH, OUT]())
    var out_b_tt  = TileTensor(out_b,  row_major[BATCH, OUT]())

    lin_fp32.forward["cpu", BATCH, POLICY=NoAMP](input_tt, out_a_tt)
    lin_bf16.forward["cpu", BATCH, POLICY=Bf16Compute](input_tt, out_b_tt)

    # Compute max-abs diff. With weights * inputs of order ~1, the bf16
    # 7-bit mantissa should give a per-element diff in the [1e-4, 1e-2]
    # range. We need at least *one* element to differ to prove POLICY
    # actually took effect.
    var max_diff: Scalar[DT] = 0.0
    var any_differ = False
    for i in range(BATCH * OUT):
        var d = fabs(out_a[i] - out_b[i])
        if d > max_diff:
            max_diff = d
        if d > 1e-6:
            any_differ = True

    print("forward NoAMP vs Bf16Compute max-abs diff = " + String(max_diff))
    assert_true(
        any_differ,
        "Bf16Compute and NoAMP produced bit-identical output — POLICY is "
        "being IGNORED on CPU forward (Phase 8.3 regression).",
    )

    in_buf.free()
    out_a.free()
    out_b.free()


def test_forward_bf16_within_quantization_tolerance() raises:
    """Confirm the bf16 path is still a *useful* approximation: every
    element matches fp32 within ~5% rel-err (matches the CPU bf16 matmul
    probe tolerance)."""
    comptime IN = 4
    comptime OUT = 3
    comptime BATCH = 2

    var lin_fp32 = Linear[IN, OUT].make[target="cpu", INIT=Zero]()
    var lin_bf16 = Linear[IN, OUT].make[target="cpu", INIT=Zero]()
    _set_weights(lin_fp32)
    _set_weights(lin_bf16)

    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var out_a:   UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var out_b:   UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    for i in range(BATCH * IN):
        in_buf[i] = Scalar[DT](Float32(i) * 0.317 - 0.4)

    var input_tt = TileTensor(in_buf, row_major[BATCH, IN]())
    var out_a_tt = TileTensor(out_a,  row_major[BATCH, OUT]())
    var out_b_tt = TileTensor(out_b,  row_major[BATCH, OUT]())

    lin_fp32.forward["cpu", BATCH, POLICY=NoAMP](input_tt, out_a_tt)
    lin_bf16.forward["cpu", BATCH, POLICY=Bf16Compute](input_tt, out_b_tt)

    var max_rel: Scalar[DT] = 0.0
    for i in range(BATCH * OUT):
        var ref_v = out_a[i]
        var got = out_b[i]
        var d = fabs(got - ref_v)
        if fabs(ref_v) > 1e-6:
            var r = d / fabs(ref_v)
            if r > max_rel:
                max_rel = r

    print("forward Bf16Compute max-rel-err vs fp32 = " + String(max_rel))
    assert_true(
        max_rel < 0.05,
        "Bf16Compute output max-rel-err " + String(max_rel)
            + " > 5% — bf16 cast-around-matmul broken?",
    )

    in_buf.free()
    out_a.free()
    out_b.free()


def test_backward_input_bf16_routes_through_amp_path() raises:
    """Backward_input[POLICY=Bf16Compute] must differ from the fp32
    baseline. This is the analog of test #1 for the backward direction —
    proves POLICY isn't dropped in backward_input either."""
    comptime IN = 4
    comptime OUT = 3
    comptime BATCH = 2

    var lin_fp32 = Linear[IN, OUT].make[target="cpu", INIT=Zero]()
    var lin_bf16 = Linear[IN, OUT].make[target="cpu", INIT=Zero]()
    _set_weights(lin_fp32)
    _set_weights(lin_bf16)

    # backward_input doesn't need cache — set up grad_output, run.
    var go_buf:   UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT)
    var gi_a:     UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    var gi_b:     UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN)
    for i in range(BATCH * OUT):
        go_buf[i] = Scalar[DT](Float32(i) * 0.21 - 0.3)
    for i in range(BATCH * IN):
        gi_a[i] = 0.0
        gi_b[i] = 0.0

    var go_tt  = TileTensor(go_buf, row_major[BATCH, OUT]())
    var gi_a_tt = TileTensor(gi_a,  row_major[BATCH, IN]())
    var gi_b_tt = TileTensor(gi_b,  row_major[BATCH, IN]())

    lin_fp32.backward_input["cpu", BATCH, POLICY=NoAMP](go_tt, gi_a_tt)
    lin_bf16.backward_input["cpu", BATCH, POLICY=Bf16Compute](go_tt, gi_b_tt)

    var any_differ = False
    var max_rel: Scalar[DT] = 0.0
    for i in range(BATCH * IN):
        var d = fabs(gi_a[i] - gi_b[i])
        if d > 1e-6:
            any_differ = True
        if fabs(gi_a[i]) > 1e-6:
            var r = d / fabs(gi_a[i])
            if r > max_rel:
                max_rel = r

    print("backward_input Bf16Compute max-rel-err vs fp32 = " + String(max_rel))
    assert_true(
        any_differ,
        "Bf16Compute and NoAMP backward_input produced bit-identical "
        "grad_input — POLICY is IGNORED in backward_input on CPU.",
    )
    assert_true(
        max_rel < 0.05,
        "backward_input Bf16Compute max-rel-err " + String(max_rel)
            + " > 5% — bf16 grad_input matmul broken?",
    )

    go_buf.free()
    gi_a.free()
    gi_b.free()


def main() raises:
    print("=" * 60)
    print("nn2 Phase 8.3 — Bf16Compute CPU validation (Linear)")
    print("=" * 60)
    test_forward_differs_under_bf16()
    print("  test_forward_differs_under_bf16 PASSED")
    test_forward_bf16_within_quantization_tolerance()
    print("  test_forward_bf16_within_quantization_tolerance PASSED")
    test_backward_input_bf16_routes_through_amp_path()
    print("  test_backward_input_bf16_routes_through_amp_path PASSED")
    print("=" * 60)
    print("ALL PASSED — Bf16Compute is wired through on CPU Linear")
    print("=" * 60)
