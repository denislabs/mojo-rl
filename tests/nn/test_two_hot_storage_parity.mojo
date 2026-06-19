"""Parity: storage two_hot vs legacy nn two_hot (CPU, bit-identical).

Compares the storage-surface helpers against the legacy ones over a range
of scalars: bins, scalar two_hot_encode, batched encode, decode_value
round-trip, and the symlog variants.
"""

from std.math import abs as math_abs

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.tensor import Tensor

# legacy
from mojo_rl.nn.loss.two_hot import (
    compute_bins as legacy_compute_bins,
    compute_symlog_bins as legacy_compute_symlog_bins,
    two_hot_encode as legacy_two_hot_encode,
    two_hot_encode_batch_ptr as legacy_two_hot_encode_batch_ptr,
    two_hot_encode_symlog_batch_ptr as legacy_two_hot_encode_symlog_batch_ptr,
    decode_value as legacy_decode_value,
    decode_value_batch_ptr as legacy_decode_value_batch_ptr,
    decode_value_batch_linear_ptr as legacy_decode_value_batch_linear_ptr,
)

# storage
from mojo_rl.nn.storage.loss.two_hot import (
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


def main() raises:
    comptime NUM_BINS = 9
    comptime BATCH = 7
    comptime TOL = Scalar[DT](1e-6)
    var max_d = Scalar[DT](0.0)

    # ── bins parity ─────────────────────────────────────────────────────
    var leg_bins = legacy_compute_bins[NUM_BINS](Scalar[DT](-3.0), Scalar[DT](5.0))
    var st_bins = st_compute_bins[NUM_BINS](Scalar[DT](-3.0), Scalar[DT](5.0))
    for i in range(NUM_BINS):
        var d = math_abs(leg_bins[i] - st_bins[i])
        if d > max_d:
            max_d = d

    var leg_slog = legacy_compute_symlog_bins[NUM_BINS]()
    var st_slog = st_compute_symlog_bins[NUM_BINS]()
    for i in range(NUM_BINS):
        var d = math_abs(leg_slog[i] - st_slog[i])
        if d > max_d:
            max_d = d

    # fill_bins (Tensor) vs compute_bins
    var st_bins_t = Tensor.alloc(NUM_BINS)
    st_fill_bins[NUM_BINS](Scalar[DT](-3.0), Scalar[DT](5.0), st_bins_t)
    for i in range(NUM_BINS):
        var d = math_abs(leg_bins[i] - st_bins_t.data[i])
        if d > max_d:
            max_d = d

    var st_slog_t = Tensor.alloc(NUM_BINS)
    st_fill_symlog_bins[NUM_BINS](st_slog_t)
    for i in range(NUM_BINS):
        var d = math_abs(leg_slog[i] - st_slog_t.data[i])
        if d > max_d:
            max_d = d

    # ── range of scalars ────────────────────────────────────────────────
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

    # scalar two_hot_encode + decode round-trip parity
    for xi in range(len(xs)):
        var x = xs[xi]
        var leg_t = InlineArray[Scalar[DT], NUM_BINS](fill=0)
        legacy_two_hot_encode[NUM_BINS](x, leg_bins, leg_t)
        var st_t = InlineArray[Scalar[DT], NUM_BINS](fill=0)
        st_two_hot_encode[NUM_BINS](x, st_bins, st_t)
        for i in range(NUM_BINS):
            var d = math_abs(leg_t[i] - st_t[i])
            if d > max_d:
                max_d = d

        # decode (treat the two-hot target as logits — exercises softmax path)
        var leg_v = legacy_decode_value[NUM_BINS](leg_t, leg_bins)
        var st_v = st_decode_value[NUM_BINS](st_t, st_bins)
        var dv = math_abs(leg_v - st_v)
        if dv > max_d:
            max_d = dv

    # ── batched encode parity (ptr vs Tensor) ───────────────────────────
    var values = Tensor.alloc(BATCH)
    for b in range(BATCH):
        values.data[b] = xs[b]

    var leg_targ = List[Scalar[DT]](length=BATCH * NUM_BINS, fill=0)
    var leg_bins_buf = List[Scalar[DT]](length=NUM_BINS, fill=0)
    for i in range(NUM_BINS):
        leg_bins_buf[i] = leg_bins[i]

    var st_bins_b = Tensor.alloc(NUM_BINS)
    for i in range(NUM_BINS):
        st_bins_b.data[i] = st_bins[i]

    var leg_targ_ptr = leg_targ.unsafe_ptr().as_unsafe_any_origin()
    legacy_two_hot_encode_batch_ptr[BATCH, NUM_BINS](
        values.data.unsafe_ptr(),
        leg_bins_buf.unsafe_ptr(),
        leg_targ_ptr,
    )
    var st_targ = Tensor.alloc(BATCH * NUM_BINS)
    st_two_hot_encode_batch[BATCH, NUM_BINS](values, st_bins_b, st_targ)
    for i in range(BATCH * NUM_BINS):
        var d = math_abs(leg_targ[i] - st_targ.data[i])
        if d > max_d:
            max_d = d

    # symlog batched encode parity (against symlog bins)
    var leg_slog_buf = List[Scalar[DT]](length=NUM_BINS, fill=0)
    for i in range(NUM_BINS):
        leg_slog_buf[i] = leg_slog[i]
    var st_slog_b = Tensor.alloc(NUM_BINS)
    for i in range(NUM_BINS):
        st_slog_b.data[i] = st_slog[i]

    var leg_targ_s = List[Scalar[DT]](length=BATCH * NUM_BINS, fill=0)
    var leg_targ_s_ptr = leg_targ_s.unsafe_ptr().as_unsafe_any_origin()
    legacy_two_hot_encode_symlog_batch_ptr[BATCH, NUM_BINS](
        values.data.unsafe_ptr(),
        leg_slog_buf.unsafe_ptr(),
        leg_targ_s_ptr,
    )
    var st_targ_s = Tensor.alloc(BATCH * NUM_BINS)
    st_two_hot_encode_symlog_batch[BATCH, NUM_BINS](values, st_slog_b, st_targ_s)
    for i in range(BATCH * NUM_BINS):
        var d = math_abs(leg_targ_s[i] - st_targ_s.data[i])
        if d > max_d:
            max_d = d

    # ── batched decode parity (symexp + linear) using symlog targets as logits ─
    var leg_vals = List[Scalar[DT]](length=BATCH, fill=0)
    var leg_vals_ptr = leg_vals.unsafe_ptr().as_unsafe_any_origin()
    legacy_decode_value_batch_ptr[BATCH, NUM_BINS](
        leg_targ_s.unsafe_ptr(),
        leg_slog_buf.unsafe_ptr(),
        leg_vals_ptr,
    )
    var st_vals = Tensor.alloc(BATCH)
    st_decode_value_batch[BATCH, NUM_BINS](st_targ_s, st_slog_b, st_vals)
    for b in range(BATCH):
        var d = math_abs(leg_vals[b] - st_vals.data[b])
        if d > max_d:
            max_d = d

    var leg_vals_l = List[Scalar[DT]](length=BATCH, fill=0)
    var leg_vals_l_ptr = leg_vals_l.unsafe_ptr().as_unsafe_any_origin()
    legacy_decode_value_batch_linear_ptr[BATCH, NUM_BINS](
        leg_targ.unsafe_ptr(),
        leg_bins_buf.unsafe_ptr(),
        leg_vals_l_ptr,
    )
    var st_vals_l = Tensor.alloc(BATCH)
    st_decode_value_batch_linear[BATCH, NUM_BINS](st_targ, st_bins_b, st_vals_l)
    for b in range(BATCH):
        var d = math_abs(leg_vals_l[b] - st_vals_l.data[b])
        if d > max_d:
            max_d = d

    print("two_hot storage parity max|delta| =", max_d)
    if max_d <= TOL:
        print("PASS")
    else:
        print("FAIL")
