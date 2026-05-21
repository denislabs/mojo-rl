"""Parity test: SliceV2 matches v1 Slice bit-for-bit on CPU."""

from std.memory import alloc
from layout import TileTensor, row_major

from mojo_rl.nn2.primitives.slice import Slice
from mojo_rl.nn2.primitives.slice_v2 import SliceV2
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero


comptime BATCH = 4
comptime IN_DIM = 8
comptime START = 2
comptime END = 6
comptime OUT_DIM = END - START


def main() raises:
    var in_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN_DIM)
    var state: UInt64 = UInt64(0x1A2B3C4D)
    for k in range(BATCH * IN_DIM):
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var r = Scalar[DT]((Int(state >> 32) & 0xFFFF)) / Scalar[DT](65535.0)
        in_p[k] = (r - Scalar[DT](0.5))

    var out_v1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT_DIM)
    var out_v2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT_DIM)

    var s1 = Slice[IN_DIM, START, END].make[target="cpu", INIT=Zero]()
    var s2 = SliceV2[IN_DIM, START, END].make[target="cpu", INIT=Zero]()

    var in_t  = TileTensor(in_p, row_major[BATCH, IN_DIM]())
    var ov1_t = TileTensor(out_v1, row_major[BATCH, OUT_DIM]())
    var ov2_t = TileTensor(out_v2, row_major[BATCH, OUT_DIM]())
    s1.forward["cpu", BATCH](in_t, ov1_t)
    s2.forward["cpu", BATCH](in_t, ov2_t)

    var ok_fwd = True
    for k in range(BATCH * OUT_DIM):
        if out_v1[k] != out_v2[k]:
            ok_fwd = False
    print("forward: PASS" if ok_fwd else "forward: FAIL")

    var go_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * OUT_DIM)
    var gi_v1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN_DIM)
    var gi_v2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN_DIM)
    for k in range(BATCH * OUT_DIM):
        go_p[k] = Scalar[DT](k) + Scalar[DT](1.0)
    # Pre-fill grad_in with non-zero so we can prove the zero-fill happens.
    for k in range(BATCH * IN_DIM):
        gi_v1[k] = Scalar[DT](99.0)
        gi_v2[k] = Scalar[DT](99.0)
    var go_t  = TileTensor(go_p,  row_major[BATCH, OUT_DIM]())
    var gv1_t = TileTensor(gi_v1, row_major[BATCH, IN_DIM]())
    var gv2_t = TileTensor(gi_v2, row_major[BATCH, IN_DIM]())
    s1.backward["cpu", BATCH](go_t, gv1_t)
    s2.backward["cpu", BATCH, mode="all"](go_t, gv2_t)

    var ok_bwd = True
    for k in range(BATCH * IN_DIM):
        if gi_v1[k] != gi_v2[k]:
            ok_bwd = False
    print("backward(mode=all): PASS" if ok_bwd else "backward(mode=all): FAIL")

    if ok_fwd and ok_bwd:
        print()
        print("PASS — SliceV2 is bit-identical to v1 Slice on CPU.")
    else:
        raise Error("slice_v2 parity test failed")

    in_p.free()
    out_v1.free()
    out_v2.free()
    go_p.free()
    gi_v1.free()
    gi_v2.free()
