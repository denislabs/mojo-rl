"""Two-hot storage loss — CPU correctness (golden).

Standalone storage test (no legacy oracle — converted from the former
`_storage_parity` test in legacy-removal Phase 0b). It exercises the storage
two-hot surface (bins / scalar + batched encode / symexp + linear decode) over
the same scalar range, bin count and ranges the parity test fed the storage
side, and asserts golden fingerprints captured from that bit-identical
legacy↔storage run.

This is a LOSS module (no Module forward/vjp): it uses the storage loss
helpers directly. Tensor fingerprints use S = Σ vᵢ, W = Σ vᵢ·(i+1) (the weight
catches sign/position errors a plain sum would cancel). Scalars are asserted
directly. Run:
  pixi run -e apple mojo run -I . tests/nn/test_two_hot_storage.mojo
"""

from std.math import abs as math_abs
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor

from mojo_rl.nn.loss.two_hot import (
    compute_bins as st_compute_bins,
    compute_symlog_bins as st_compute_symlog_bins,
    fill_bins as st_fill_bins,
    fill_symlog_bins as st_fill_symlog_bins,
    two_hot_encode as st_two_hot_encode,
    two_hot_encode_batch as st_two_hot_encode_batch,
    two_hot_encode_symlog_batch as st_two_hot_encode_symlog_batch,
    decode_value as st_decode_value,
    decode_value_batch as st_decode_value_batch,
    decode_value_batch_linear as st_decode_value_batch_linear,
)


comptime NUM_BINS = 9
comptime BATCH = 7


def _check(name: String, data: Tensor, n: Int,
           es: Scalar[DT], ew: Scalar[DT], tol: Scalar[DT]) -> Bool:
    """Assert tensor fingerprint (Σ vᵢ, Σ vᵢ·(i+1)) matches golden (es, ew)."""
    var s: Scalar[DT] = 0
    var w: Scalar[DT] = 0
    for i in range(n):
        s += data.data[i]
        w += data.data[i] * Scalar[DT](i + 1)
    var ok = math_abs(s - es) < tol and math_abs(w - ew) < tol
    print("  ", name, "S", s, "(exp", es, ") W", w, "(exp", ew, ")", "OK" if ok else "FAIL")
    return ok


def _check_scalar(name: String, v: Scalar[DT], ev: Scalar[DT], tol: Scalar[DT]) -> Bool:
    var ok = math_abs(v - ev) < tol
    print("  ", name, "=", v, "(exp", ev, ")", "OK" if ok else "FAIL")
    return ok


def test_two_hot_cpu_golden() raises:
    print("test_two_hot_cpu_golden (storage CPU vs golden) ...")
    comptime TOL = Scalar[DT](5e-3)
    var ok = True

    # ── bins (Tensor form) ──────────────────────────────────────────────
    var st_bins = st_compute_bins[NUM_BINS](Scalar[DT](-3.0), Scalar[DT](5.0))
    var st_slog = st_compute_symlog_bins[NUM_BINS]()

    var st_bins_t = Tensor.alloc(NUM_BINS)
    st_fill_bins[NUM_BINS](Scalar[DT](-3.0), Scalar[DT](5.0), st_bins_t)
    ok = _check("bins", st_bins_t, NUM_BINS, 9.0, 105.0, TOL) and ok

    var st_slog_t = Tensor.alloc(NUM_BINS)
    st_fill_symlog_bins[NUM_BINS](st_slog_t)
    ok = _check("symlog_bins", st_slog_t, NUM_BINS, 0.0, 300.0, TOL) and ok

    # ── range of scalars: scalar encode + decode round-trip ─────────────
    var xs = List[Scalar[DT]]()
    xs.append(Scalar[DT](-10.0))
    xs.append(Scalar[DT](-3.5))
    xs.append(Scalar[DT](-3.0))
    xs.append(Scalar[DT](-2.1))
    xs.append(Scalar[DT](-0.7))
    xs.append(Scalar[DT](0.0))
    xs.append(Scalar[DT](1.3))
    xs.append(Scalar[DT](2.999))
    xs.append(Scalar[DT](4.0))
    xs.append(Scalar[DT](5.0))
    xs.append(Scalar[DT](9.0))

    # Accumulate the scalar encode fingerprint across all xs (flattened over
    # NUM_BINS per sample), plus the sum of decode_value round-trips.
    var enc_flat = Tensor.alloc(len(xs) * NUM_BINS)
    var dec_sum = Scalar[DT](0.0)
    for xi in range(len(xs)):
        var x = xs[xi]
        var st_t = InlineArray[Scalar[DT], NUM_BINS](fill=0)
        st_two_hot_encode[NUM_BINS](x, st_bins, st_t)
        for i in range(NUM_BINS):
            enc_flat.data[xi * NUM_BINS + i] = st_t[i]
        # decode (treat the two-hot target as logits — exercises softmax path)
        dec_sum += st_decode_value[NUM_BINS](st_t, st_bins)
    ok = _check("scalar_encode", enc_flat, len(xs) * NUM_BINS, 11.0, 545.499, TOL) and ok
    ok = _check_scalar("scalar_decode_sum", dec_sum, 20.77351, TOL) and ok

    # ── batched encode ──────────────────────────────────────────────────
    var values = Tensor.alloc(BATCH)
    for b in range(BATCH):
        values.data[b] = xs[b]

    var st_bins_b = Tensor.alloc(NUM_BINS)
    for i in range(NUM_BINS):
        st_bins_b.data[i] = st_bins[i]

    var st_targ = Tensor.alloc(BATCH * NUM_BINS)
    st_two_hot_encode_batch[BATCH, NUM_BINS](values, st_bins_b, st_targ)
    ok = _check("encode_batch", st_targ, BATCH * NUM_BINS, 7.0, 206.5, TOL) and ok

    # symlog batched encode (against symlog bins)
    var st_slog_b = Tensor.alloc(NUM_BINS)
    for i in range(NUM_BINS):
        st_slog_b.data[i] = st_slog[i]
    var st_targ_s = Tensor.alloc(BATCH * NUM_BINS)
    st_two_hot_encode_symlog_batch[BATCH, NUM_BINS](values, st_slog_b, st_targ_s)
    ok = _check("encode_symlog_batch", st_targ_s, BATCH * NUM_BINS, 7.0, 222.77653, TOL) and ok

    # ── batched decode (symexp + linear) using targets as logits ────────
    var st_vals = Tensor.alloc(BATCH)
    st_decode_value_batch[BATCH, NUM_BINS](st_targ_s, st_slog_b, st_vals)
    ok = _check("decode_symexp", st_vals, BATCH, -0.79283917, -1.3748049, TOL) and ok

    var st_vals_l = Tensor.alloc(BATCH)
    st_decode_value_batch_linear[BATCH, NUM_BINS](st_targ, st_bins_b, st_vals_l)
    ok = _check("decode_linear", st_vals_l, BATCH, 4.266, 20.462769, TOL) and ok

    assert_true(ok, "two_hot CPU golden")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("two_hot storage loss (CPU golden)")
    print("=" * 70)
    test_two_hot_cpu_golden()
    print("ALL PASSED")
