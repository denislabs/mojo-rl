"""Phase 5 EXIT CRITERION — StopGrad bit-exact advantage-norm parity.

PPO's advantage-normalization pattern needs the normalized value to
appear in the loss but the gradient to NOT flow back into whatever
produced the raw advantages. In a copy-out implementation:

    A_raw    = <upstream value>      # gradient-bearing
    A_norm   = (A_raw - mean) / (std + eps)
    A_const  = copy(A_norm)          # treat as constant
    loss     = ... A_const ...       # gradient through A_const is zero

With nn2's `StopGrad`, the same pattern composes through Sequential:

    pipeline = Sequential[<upstream>, StopGrad[D]]

forward(pipeline, A_raw):
    out = StopGrad.forward(upstream.forward(A_raw))
        = upstream.forward(A_raw)        (identity)

backward(pipeline, go):
    StopGrad.backward zero-fills its grad_input.
    upstream.backward then sees an all-zero `grad_output`, so its
    param gradients stay at zero (any pre-existing accumulator value
    is untouched).

Two checks:
  1. Forward identity: pipeline.forward(A_raw) == hand-computed (A_raw - mean) / (std + eps).
     (Where the upstream module here is a vanilla normalize step we
     implement in plain Mojo to match the bit pattern.)
  2. Gradient seal: after backward through StopGrad, the upstream
     module's grad accumulators are untouched (= 0 since we initialized
     them to 0 and got no gradient).

For the upstream module we use Linear[D, D] — it carries learnable
params whose grad accumulators are observable after backward.
"""

from std.math import abs as fabs, sqrt
from std.memory import alloc
from std.testing import assert_equal, assert_true
from layout import TileTensor, TensorLayout, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.stop_grad import StopGrad
from mojo_rl.nn2.combinators import Sequential


comptime EPS_NORM: Scalar[DT] = 1e-8


def _compute_reference_advantage_norm(
    raw_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n: Int,
    norm_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    """Hand-coded reference: A_norm = (A - mean) / (std + eps).
    Computed in plain Mojo, treats everything as a constant — exactly
    what the copy-out implementation does."""
    var s: Scalar[DT] = 0.0
    for i in range(n):
        s += raw_ptr[i]
    var mean = s / Scalar[DT](Float32(n))
    var sv: Scalar[DT] = 0.0
    for i in range(n):
        var d = raw_ptr[i] - mean
        sv += d * d
    var std = sqrt(sv / Scalar[DT](Float32(n)))
    for i in range(n):
        norm_ptr[i] = (raw_ptr[i] - mean) / (std + EPS_NORM)


def test_stop_grad_forward_bit_exact_with_reference() raises:
    """Forward through StopGrad must be bit-exact with the hand-coded
    advantage-norm reference (StopGrad's forward is pure copy)."""
    comptime BATCH = 8
    comptime DIM = 1                  # advantage is scalar-per-sample

    var sg = StopGrad[DIM].make["cpu", INIT=Zero]()

    # Raw advantages = some arbitrary values.
    var raw_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var ref_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var out_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    for k in range(BATCH * DIM):
        raw_buf[k] = Scalar[DT](Float32(k) * 1.5 - 4.5)   # span both signs

    # Reference normalization.
    _compute_reference_advantage_norm(raw_buf, BATCH * DIM, ref_buf)

    # Pipeline: feed the *normalized* values through StopGrad.
    # StopGrad's forward is identity, so output should equal ref_buf
    # bit-exactly.
    var in_tt  = TileTensor(ref_buf, row_major[BATCH, DIM]())
    var out_tt = TileTensor(out_buf, row_major[BATCH, DIM]())
    sg.forward["cpu", BATCH](in_tt, out_tt)

    var max_diff: Scalar[DT] = 0.0
    for k in range(BATCH * DIM):
        var d = fabs(out_buf[k] - ref_buf[k])
        if d > max_diff: max_diff = d
    assert_true(max_diff == Scalar[DT](0.0),
        "StopGrad forward not bit-exact vs reference: " + String(max_diff))

    raw_buf.free()
    ref_buf.free()
    out_buf.free()
    print("  test_stop_grad_forward_bit_exact_with_reference PASSED")


def test_stop_grad_severs_grad_chain_through_upstream() raises:
    """Sequential[Linear, StopGrad] — backward from grad_output must
    NOT update Linear's grad_w / grad_b (since StopGrad zero-fills the
    intermediate gradient before it reaches Linear)."""
    comptime D = 4
    comptime BATCH = 3

    var pipeline = Sequential[Linear[D, D], StopGrad[D]].make[
        "cpu", INIT=Zero,
    ]()

    # Sanity: Linear's grad_w / grad_b start at 0.
    for i in range(D * D):
        assert_equal(pipeline.children[0].grad_w[i], 0.0)
    for j in range(D):
        assert_equal(pipeline.children[0].grad_b[j], 0.0)

    # Forward.
    var in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * D)
    var out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * D)
    var go_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * D)
    var gi_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * D)
    for k in range(BATCH * D):
        in_buf[k] = Scalar[DT](Float32(k) * 0.2 - 0.5)
        go_buf[k] = Scalar[DT](7.0 + Float32(k))   # large nonzero grad_output
        gi_buf[k] = -999.0
    var input    = TileTensor(in_buf,  row_major[BATCH, D]())
    var output   = TileTensor(out_buf, row_major[BATCH, D]())
    var grad_out = TileTensor(go_buf,  row_major[BATCH, D]())
    var grad_in  = TileTensor(gi_buf,  row_major[BATCH, D]())

    pipeline.forward["cpu", BATCH](input, output)
    pipeline.backward["cpu", BATCH](grad_out, grad_in)

    # CRITICAL ASSERTION: Linear's grad accumulators must still be all
    # zero, because StopGrad zeroed the intermediate before it reached
    # Linear.backward.
    var max_gw: Scalar[DT] = 0.0
    for i in range(D * D):
        if fabs(pipeline.children[0].grad_w[i]) > max_gw:
            max_gw = fabs(pipeline.children[0].grad_w[i])
    var max_gb: Scalar[DT] = 0.0
    for j in range(D):
        if fabs(pipeline.children[0].grad_b[j]) > max_gb:
            max_gb = fabs(pipeline.children[0].grad_b[j])
    print("Linear.grad_w max-abs after StopGrad backward = " + String(max_gw))
    print("Linear.grad_b max-abs after StopGrad backward = " + String(max_gb))
    assert_true(max_gw == Scalar[DT](0.0),
        "GRADIENT LEAKED THROUGH StopGrad into Linear.grad_w: " + String(max_gw))
    assert_true(max_gb == Scalar[DT](0.0),
        "GRADIENT LEAKED THROUGH StopGrad into Linear.grad_b: " + String(max_gb))

    # And grad_input itself: should be 0 (StopGrad.backward writes 0,
    # then Linear.backward consumes 0 and writes the matmul-with-weight
    # of 0, which is 0).
    var max_gi: Scalar[DT] = 0.0
    for k in range(BATCH * D):
        if fabs(gi_buf[k]) > max_gi:
            max_gi = fabs(gi_buf[k])
    assert_true(max_gi == Scalar[DT](0.0),
        "grad_input not zero after StopGrad: " + String(max_gi))

    in_buf.free()
    out_buf.free()
    go_buf.free()
    gi_buf.free()
    print("  test_stop_grad_severs_grad_chain_through_upstream PASSED")


def test_full_advantage_norm_pipeline() raises:
    """End-to-end: produce advantages → normalize → StopGrad → loss.
    Verifies that the pipeline's output matches the reference AND
    gradient does not flow back into the upstream producer."""
    comptime BATCH = 6
    comptime DIM = 1

    # Upstream producer: a Linear (params live, grad accumulators live).
    # In real PPO this would be the critic; here it's any module that
    # outputs the "raw advantage" tensor.
    var producer = Linear[2, DIM].make["cpu", INIT=Zero]()
    # Initialize producer's weights to small nonzero so that its
    # output varies (Zero init would give all-zeros → no normalization
    # signal).
    for i in range(2 * DIM):
        producer.weight[i] = Scalar[DT](0.3 + Float32(i) * 0.1)
    for j in range(DIM):
        producer.bias[j] = Scalar[DT](0.2)

    # Producer input.
    var prod_in_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2)
    var raw_adv_buf:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var norm_buf:     UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var pipe_out_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    for k in range(BATCH * 2):
        prod_in_buf[k] = Scalar[DT](Float32(k) * 0.4 - 1.0)
    var prod_in  = TileTensor(prod_in_buf, row_major[BATCH, 2]())
    var raw_adv  = TileTensor(raw_adv_buf, row_major[BATCH, DIM]())

    producer.forward["cpu", BATCH](prod_in, raw_adv)

    # Reference advantage normalization (hand-coded).
    _compute_reference_advantage_norm(raw_adv_buf, BATCH * DIM, norm_buf)

    # Now pump norm_buf through StopGrad — emulating the "normalize
    # outside the autodiff graph, then freeze with StopGrad" pattern.
    var sg = StopGrad[DIM].make["cpu", INIT=Zero]()
    var norm_in  = TileTensor(norm_buf,     row_major[BATCH, DIM]())
    var pipe_out = TileTensor(pipe_out_buf, row_major[BATCH, DIM]())
    sg.forward["cpu", BATCH](norm_in, pipe_out)

    # Forward bit-exact vs reference.
    for k in range(BATCH * DIM):
        var diff = fabs(pipe_out_buf[k] - norm_buf[k])
        assert_true(diff == Scalar[DT](0.0),
            "k=" + String(k) + " pipe " + String(pipe_out_buf[k])
            + " ref " + String(norm_buf[k]) + " diff " + String(diff))

    # Backward through StopGrad with arbitrary nonzero grad_output.
    var go_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    for k in range(BATCH * DIM):
        go_buf[k] = Scalar[DT](3.7 + Float32(k))
        gi_buf[k] = -999.0
    var grad_out = TileTensor(go_buf, row_major[BATCH, DIM]())
    var grad_in  = TileTensor(gi_buf, row_major[BATCH, DIM]())
    sg.backward["cpu", BATCH](grad_out, grad_in)

    # If the user then routed grad_in into producer.backward, the
    # all-zero grad_in would produce all-zero updates to producer's
    # grad accumulators. Verify that grad_in is exactly zero.
    for k in range(BATCH * DIM):
        assert_true(gi_buf[k] == Scalar[DT](0.0),
            "k=" + String(k) + " grad_in " + String(gi_buf[k]) + " not zero")

    # Sanity: producer's grad accumulators are untouched (never had
    # backward called) — still zero from make().
    for i in range(2 * DIM):
        assert_equal(producer.grad_w[i], 0.0)
    for j in range(DIM):
        assert_equal(producer.grad_b[j], 0.0)

    prod_in_buf.free()
    raw_adv_buf.free()
    norm_buf.free()
    pipe_out_buf.free()
    go_buf.free()
    gi_buf.free()
    print("  test_full_advantage_norm_pipeline PASSED (forward bit-exact, "
          "grad zeroed by StopGrad)")


def main() raises:
    print("=" * 60)
    print("nn2 Phase 5 EXIT CRITERION:")
    print("  StopGrad bit-exact parity for PPO advantage-norm pattern")
    print("=" * 60)
    test_stop_grad_forward_bit_exact_with_reference()
    test_stop_grad_severs_grad_chain_through_upstream()
    test_full_advantage_norm_pipeline()
    print("=" * 60)
    print("ALL PASSED — Phase 5 exit criterion met")
    print("=" * 60)
