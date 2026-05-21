"""Parity test: SumV2 + MeanV2 match v1 bit-for-bit on CPU."""

from std.memory import alloc
from layout import TileTensor, row_major

from mojo_rl.nn2.primitives.reduce import Sum, Mean
from mojo_rl.nn2.primitives.reduce_v2 import SumV2, MeanV2
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero


comptime BATCH = 4
comptime DIM = 9


def _seed(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int, salt: UInt64):
    var state: UInt64 = salt
    for k in range(n):
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var r = Scalar[DT]((Int(state >> 32) & 0xFFFF)) / Scalar[DT](65535.0)
        p[k] = (r - Scalar[DT](0.5))


def _check_parity[ARM_NAME: StaticString](
    s1: UnsafePointer[Scalar[DT], MutAnyOrigin],
    s2: UnsafePointer[Scalar[DT], MutAnyOrigin],
    n: Int,
) raises -> Bool:
    for k in range(n):
        if s1[k] != s2[k]:
            return False
    return True


def main() raises:
    # ── SumV2 ────────────────────────────────────────────────────────
    var s1 = Sum[DIM].make[target="cpu", INIT=Zero]()
    var s2 = SumV2[DIM].make[target="cpu", INIT=Zero]()

    var in_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    _seed(in_p, BATCH * DIM, UInt64(0xAABBCCDD))
    var o1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var o2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var in_t = TileTensor(in_p, row_major[BATCH, DIM]())
    var o1_t = TileTensor(o1,  row_major[BATCH, 1]())
    var o2_t = TileTensor(o2,  row_major[BATCH, 1]())
    s1.forward["cpu", BATCH](in_t, o1_t)
    s2.forward["cpu", BATCH](in_t, o2_t)
    var ok_sum_fwd = _check_parity["sum"](o1, o2, BATCH)

    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var gi1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    for k in range(BATCH):
        go[k] = Scalar[DT](k) + Scalar[DT](1.0)
    var go_t  = TileTensor(go,  row_major[BATCH, 1]())
    var gi1_t = TileTensor(gi1, row_major[BATCH, DIM]())
    var gi2_t = TileTensor(gi2, row_major[BATCH, DIM]())
    s1.backward["cpu", BATCH](go_t, gi1_t)
    s2.backward["cpu", BATCH, mode="all"](go_t, gi2_t)
    var ok_sum_bwd = _check_parity["sum_bwd"](gi1, gi2, BATCH * DIM)

    print(
        "Sum: forward=", "PASS" if ok_sum_fwd else "FAIL",
        " backward(all)=", "PASS" if ok_sum_bwd else "FAIL",
    )

    # ── MeanV2 ───────────────────────────────────────────────────────
    var m1 = Mean[DIM].make[target="cpu", INIT=Zero]()
    var m2 = MeanV2[DIM].make[target="cpu", INIT=Zero]()
    m1.forward["cpu", BATCH](in_t, o1_t)
    m2.forward["cpu", BATCH](in_t, o2_t)
    var ok_mean_fwd = _check_parity["mean"](o1, o2, BATCH)

    m1.backward["cpu", BATCH](go_t, gi1_t)
    m2.backward["cpu", BATCH, mode="all"](go_t, gi2_t)
    var ok_mean_bwd = _check_parity["mean_bwd"](gi1, gi2, BATCH * DIM)

    # backward(input_only) — same as backward for these parameterless leaves.
    var gi_ref: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi_test: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi_ref_t  = TileTensor(gi_ref,  row_major[BATCH, DIM]())
    var gi_test_t = TileTensor(gi_test, row_major[BATCH, DIM]())
    m1.backward_input["cpu", BATCH](go_t, gi_ref_t)
    m2.backward["cpu", BATCH, mode="input_only"](go_t, gi_test_t)
    var ok_mean_bwi = _check_parity["mean_bwi"](gi_ref, gi_test, BATCH * DIM)

    print(
        "Mean: forward=", "PASS" if ok_mean_fwd else "FAIL",
        " backward(all)=", "PASS" if ok_mean_bwd else "FAIL",
        " backward(input_only)=", "PASS" if ok_mean_bwi else "FAIL",
    )

    if ok_sum_fwd and ok_sum_bwd and ok_mean_fwd and ok_mean_bwd and ok_mean_bwi:
        print()
        print("PASS — SumV2 + MeanV2 are bit-identical to v1 on CPU.")
    else:
        raise Error("reduce_v2 parity test failed")

    in_p.free()
    o1.free()
    o2.free()
    go.free()
    gi1.free()
    gi2.free()
    gi_ref.free()
    gi_test.free()
