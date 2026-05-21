"""Parity tests: BinarySubV2, BinaryElemMinV2, BinaryConcatV2 vs v1 on CPU."""

from std.memory import alloc
from layout import TileTensor, row_major

from mojo_rl.nn2.primitives.binary_sub import BinarySub
from mojo_rl.nn2.primitives.binary_sub_v2 import BinarySubV2
from mojo_rl.nn2.primitives.binary_elem_min import BinaryElemMin
from mojo_rl.nn2.primitives.binary_elem_min_v2 import BinaryElemMinV2
from mojo_rl.nn2.primitives.binary_concat import BinaryConcat
from mojo_rl.nn2.primitives.binary_concat_v2 import BinaryConcatV2
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero


comptime BATCH = 4
comptime DIM = 6
comptime IN0 = 5
comptime IN1 = 3
comptime CONCAT_OUT = IN0 + IN1


def _seed(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int, salt: UInt64):
    var state: UInt64 = salt
    for k in range(n):
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var r = Scalar[DT]((Int(state >> 32) & 0xFFFF)) / Scalar[DT](65535.0)
        p[k] = (r - Scalar[DT](0.5))


def _eq(a: UnsafePointer[Scalar[DT], MutAnyOrigin],
        b: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int) raises -> Bool:
    for k in range(n):
        if a[k] != b[k]:
            return False
    return True


def _check_sub() raises -> Bool:
    var s1 = BinarySub[DIM].make[target="cpu", INIT=Zero]()
    var s2 = BinarySubV2[DIM].make[target="cpu", INIT=Zero]()

    var a: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var b: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    _seed(a, BATCH * DIM, UInt64(0x1111))
    _seed(b, BATCH * DIM, UInt64(0x2222))
    var o1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var o2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var a_t = TileTensor(a, row_major[BATCH, DIM]())
    var b_t = TileTensor(b, row_major[BATCH, DIM]())
    var o1_t = TileTensor(o1, row_major[BATCH, DIM]())
    var o2_t = TileTensor(o2, row_major[BATCH, DIM]())
    s1.forward["cpu", BATCH](a_t, b_t, o1_t)
    s2.forward["cpu", BATCH](a_t, b_t, o2_t)
    var ok_fwd = _eq(o1, o2, BATCH * DIM)

    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi0_1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi1_1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi0_2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi1_2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    _seed(go, BATCH * DIM, UInt64(0x3333))
    var go_t = TileTensor(go, row_major[BATCH, DIM]())
    var gi0_1t = TileTensor(gi0_1, row_major[BATCH, DIM]())
    var gi1_1t = TileTensor(gi1_1, row_major[BATCH, DIM]())
    var gi0_2t = TileTensor(gi0_2, row_major[BATCH, DIM]())
    var gi1_2t = TileTensor(gi1_2, row_major[BATCH, DIM]())
    s1.backward["cpu", BATCH](go_t, gi0_1t, gi1_1t)
    s2.backward["cpu", BATCH, mode="all"](go_t, gi0_2t, gi1_2t)
    var ok_bwd = _eq(gi0_1, gi0_2, BATCH * DIM) and _eq(gi1_1, gi1_2, BATCH * DIM)

    print("BinarySub: fwd=", "PASS" if ok_fwd else "FAIL",
          " bwd=", "PASS" if ok_bwd else "FAIL")

    a.free(); b.free(); o1.free(); o2.free()
    go.free(); gi0_1.free(); gi1_1.free(); gi0_2.free(); gi1_2.free()
    return ok_fwd and ok_bwd


def _check_min() raises -> Bool:
    var m1 = BinaryElemMin[DIM].make[target="cpu", INIT=Zero]()
    var m2 = BinaryElemMinV2[DIM].make[target="cpu", INIT=Zero]()

    var a: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var b: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    _seed(a, BATCH * DIM, UInt64(0x4444))
    _seed(b, BATCH * DIM, UInt64(0x5555))
    var o1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var o2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var a_t = TileTensor(a, row_major[BATCH, DIM]())
    var b_t = TileTensor(b, row_major[BATCH, DIM]())
    var o1_t = TileTensor(o1, row_major[BATCH, DIM]())
    var o2_t = TileTensor(o2, row_major[BATCH, DIM]())
    m1.forward["cpu", BATCH](a_t, b_t, o1_t)
    m2.forward["cpu", BATCH](a_t, b_t, o2_t)
    var ok_fwd = _eq(o1, o2, BATCH * DIM)

    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi0_1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi1_1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi0_2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi1_2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    _seed(go, BATCH * DIM, UInt64(0x6666))
    var go_t = TileTensor(go, row_major[BATCH, DIM]())
    var gi0_1t = TileTensor(gi0_1, row_major[BATCH, DIM]())
    var gi1_1t = TileTensor(gi1_1, row_major[BATCH, DIM]())
    var gi0_2t = TileTensor(gi0_2, row_major[BATCH, DIM]())
    var gi1_2t = TileTensor(gi1_2, row_major[BATCH, DIM]())
    m1.backward["cpu", BATCH](go_t, gi0_1t, gi1_1t)
    m2.backward["cpu", BATCH, mode="all"](go_t, gi0_2t, gi1_2t)
    var ok_bwd = _eq(gi0_1, gi0_2, BATCH * DIM) and _eq(gi1_1, gi1_2, BATCH * DIM)

    print("BinaryElemMin: fwd=", "PASS" if ok_fwd else "FAIL",
          " bwd=", "PASS" if ok_bwd else "FAIL")

    a.free(); b.free(); o1.free(); o2.free()
    go.free(); gi0_1.free(); gi1_1.free(); gi0_2.free(); gi1_2.free()
    return ok_fwd and ok_bwd


def _check_concat() raises -> Bool:
    var c1 = BinaryConcat[IN0, IN1].make[target="cpu", INIT=Zero]()
    var c2 = BinaryConcatV2[IN0, IN1].make[target="cpu", INIT=Zero]()

    var a: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN0)
    var b: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN1)
    _seed(a, BATCH * IN0, UInt64(0x7777))
    _seed(b, BATCH * IN1, UInt64(0x8888))
    var o1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * CONCAT_OUT)
    var o2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * CONCAT_OUT)
    var a_t = TileTensor(a, row_major[BATCH, IN0]())
    var b_t = TileTensor(b, row_major[BATCH, IN1]())
    var o1_t = TileTensor(o1, row_major[BATCH, CONCAT_OUT]())
    var o2_t = TileTensor(o2, row_major[BATCH, CONCAT_OUT]())
    c1.forward["cpu", BATCH](a_t, b_t, o1_t)
    c2.forward["cpu", BATCH](a_t, b_t, o2_t)
    var ok_fwd = _eq(o1, o2, BATCH * CONCAT_OUT)

    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * CONCAT_OUT)
    var gi0_1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN0)
    var gi1_1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN1)
    var gi0_2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN0)
    var gi1_2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN1)
    _seed(go, BATCH * CONCAT_OUT, UInt64(0x9999))
    var go_t = TileTensor(go, row_major[BATCH, CONCAT_OUT]())
    var gi0_1t = TileTensor(gi0_1, row_major[BATCH, IN0]())
    var gi1_1t = TileTensor(gi1_1, row_major[BATCH, IN1]())
    var gi0_2t = TileTensor(gi0_2, row_major[BATCH, IN0]())
    var gi1_2t = TileTensor(gi1_2, row_major[BATCH, IN1]())
    c1.backward["cpu", BATCH](go_t, gi0_1t, gi1_1t)
    c2.backward["cpu", BATCH, mode="all"](go_t, gi0_2t, gi1_2t)
    var ok_bwd = _eq(gi0_1, gi0_2, BATCH * IN0) and _eq(gi1_1, gi1_2, BATCH * IN1)

    print("BinaryConcat: fwd=", "PASS" if ok_fwd else "FAIL",
          " bwd=", "PASS" if ok_bwd else "FAIL")

    a.free(); b.free(); o1.free(); o2.free()
    go.free(); gi0_1.free(); gi1_1.free(); gi0_2.free(); gi1_2.free()
    return ok_fwd and ok_bwd


def main() raises:
    var ok = True
    ok = ok and _check_sub()
    ok = ok and _check_min()
    ok = ok and _check_concat()
    if ok:
        print()
        print("PASS — BinarySubV2 + BinaryElemMinV2 + BinaryConcatV2 are")
        print("       bit-identical to v1 on CPU.")
    else:
        raise Error("binaries_v2 parity test failed")
