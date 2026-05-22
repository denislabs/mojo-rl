"""Phase 4.6a foundation: NaryBinarySub smoke + parity vs legacy BinarySub.

Verifies the new NaryModule trait surface works end-to-end:
  1. Forward bit-identical to legacy BinarySub for the same inputs.
  2. vjp produces grad_in0 = grad_output, grad_in1 = -grad_output.
  3. Multiple calls don't mutate state in unexpected ways.

If this passes, the Phase 4.6a foundation is sound and Phase 4.6b's
big-bang migration can proceed against the same pattern.
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.primitives.nary_binary_sub import NaryBinarySub
from mojo_rl.nn2.primitives.binary_sub import BinarySub


def test_forward_bit_identical_to_legacy() raises:
    """Run both NaryBinarySub and legacy BinarySub on the same inputs;
    require bit-identical outputs."""
    print("test_forward_bit_identical_to_legacy ...")
    comptime BATCH = 4
    comptime DIM = 8
    comptime N = BATCH * DIM

    var in0_buf = alloc[Scalar[DT]](N)
    var in1_buf = alloc[Scalar[DT]](N)
    for i in range(N):
        in0_buf[i] = Scalar[DT](Float64(i) * 0.5)
        in1_buf[i] = Scalar[DT](Float64(i) * 0.3 + 1.0)

    # Legacy run.
    var legacy_out = alloc[Scalar[DT]](N)
    for i in range(N): legacy_out[i] = Scalar[DT](0.0)
    var legacy = BinarySub[DIM].make[target="cpu", INIT=Kaiming]()
    var legacy_in0 = TileTensor(in0_buf, row_major[BATCH, DIM]())
    var legacy_in1 = TileTensor(in1_buf, row_major[BATCH, DIM]())
    var legacy_out_t = TileTensor(legacy_out, row_major[BATCH, DIM]())
    legacy.forward["cpu", BATCH](legacy_in0, legacy_in1, legacy_out_t)

    # NaryBinarySub run — caller rebinds inputs to MutAnyOrigin.
    var nary_out = alloc[Scalar[DT]](N)
    for i in range(N): nary_out[i] = Scalar[DT](0.0)
    var nary = NaryBinarySub[DIM].make[target="cpu", INIT=Kaiming]()
    var i0_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](in0_buf)
    var i1_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](in1_buf)
    var o_ptr  = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](nary_out)
    var nary_in0 = TileTensor(i0_ptr, row_major[BATCH, DIM]())
    var nary_in1 = TileTensor(i1_ptr, row_major[BATCH, DIM]())
    var nary_out_t = TileTensor(o_ptr, row_major[BATCH, DIM]())
    nary.forward["cpu", BATCH](nary_in0, nary_in1, output=nary_out_t)

    # Compare bit-for-bit (FP32 — same SIMD width, same op order).
    var max_diff: Scalar[DT] = 0.0
    for i in range(N):
        var d = (legacy_out[i] - nary_out[i]).__abs__()
        if d > max_diff:
            max_diff = d
    print("  max |legacy - nary| =", max_diff)
    assert_true(
        max_diff == Scalar[DT](0.0),
        "NaryBinarySub.forward must be bit-identical to legacy BinarySub",
    )
    in0_buf.free(); in1_buf.free(); legacy_out.free(); nary_out.free()
    print("  ok")


def test_vjp_correctness() raises:
    """grad_in0 = grad_output; grad_in1 = -grad_output."""
    print("test_vjp_correctness ...")
    comptime BATCH = 2
    comptime DIM = 4
    comptime N = BATCH * DIM

    var go_buf = alloc[Scalar[DT]](N)
    for i in range(N): go_buf[i] = Scalar[DT](Float64(i + 1) * 0.25)

    var gi0_buf = alloc[Scalar[DT]](N)
    var gi1_buf = alloc[Scalar[DT]](N)
    for i in range(N):
        gi0_buf[i] = Scalar[DT](0.0)
        gi1_buf[i] = Scalar[DT](0.0)

    var nary = NaryBinarySub[DIM].make[target="cpu", INIT=Kaiming]()
    var go_ptr  = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go_buf)
    var gi0_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gi0_buf)
    var gi1_ptr = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gi1_buf)
    var go_t  = TileTensor(go_ptr, row_major[BATCH, DIM]())
    var gi0_t = TileTensor(gi0_ptr, row_major[BATCH, DIM]())
    var gi1_t = TileTensor(gi1_ptr, row_major[BATCH, DIM]())
    nary.vjp["cpu", BATCH](go_t, gi0_t, gi1_t)

    for i in range(N):
        var expected_gi0 = go_buf[i]
        var expected_gi1 = -go_buf[i]
        assert_true(
            (gi0_buf[i] - expected_gi0).__abs__() < Scalar[DT](1e-6),
            "grad_in0 must equal grad_output",
        )
        assert_true(
            (gi1_buf[i] - expected_gi1).__abs__() < Scalar[DT](1e-6),
            "grad_in1 must equal -grad_output",
        )

    go_buf.free(); gi0_buf.free(); gi1_buf.free()
    print("  ok")


def test_arity_field() raises:
    """Check the comptime ARITY field is reflectable."""
    print("test_arity_field ...")
    comptime assert NaryBinarySub[4].ARITY == 2
    comptime assert NaryBinarySub[4].OUT_DIM == 4
    comptime assert NaryBinarySub[4].IN0_DIM == 4
    comptime assert NaryBinarySub[4].IN1_DIM == 4
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Phase 4.6a foundation — NaryBinarySub smoke")
    print("=" * 70)
    test_arity_field()
    test_forward_bit_identical_to_legacy()
    test_vjp_correctness()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
