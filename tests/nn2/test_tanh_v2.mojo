"""Parity test: TanhV2 matches v1 Tanh bit-for-bit on CPU."""

from std.memory import alloc
from layout import TileTensor, row_major

from mojo_rl.nn2.primitives.tanh import Tanh
from mojo_rl.nn2.primitives.tanh_v2 import TanhV2
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero


comptime BATCH = 4
comptime DIM = 6


def main() raises:
    var in_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var state: UInt64 = UInt64(0xCAFEBABE)
    for k in range(BATCH * DIM):
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var r = Scalar[DT]((Int(state >> 32) & 0xFFFF)) / Scalar[DT](65535.0)
        in_p[k] = (r - Scalar[DT](0.5)) * Scalar[DT](4.0)  # [-2, 2] — exercises tanh range

    var out_v1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var out_v2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)

    var t1 = Tanh[DIM].make[target="cpu", INIT=Zero]()
    var t2 = TanhV2[DIM].make[target="cpu", INIT=Zero]()

    var in_t  = TileTensor(in_p, row_major[BATCH, DIM]())
    var ov1_t = TileTensor(out_v1, row_major[BATCH, DIM]())
    var ov2_t = TileTensor(out_v2, row_major[BATCH, DIM]())
    t1.forward["cpu", BATCH](in_t, ov1_t)
    t2.forward["cpu", BATCH](in_t, ov2_t)

    var ok_fwd = True
    for k in range(BATCH * DIM):
        if out_v1[k] != out_v2[k]:
            print("forward mismatch k=", k, "  v1=", out_v1[k], "  v2=", out_v2[k])
            ok_fwd = False
    if ok_fwd:
        print("forward: PASS")

    # Backward(mode=all).
    var go_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi_v1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi_v2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    for k in range(BATCH * DIM):
        go_p[k] = Scalar[DT](1.0) + Scalar[DT](k) * Scalar[DT](0.1)
    var go_t  = TileTensor(go_p,  row_major[BATCH, DIM]())
    var gv1_t = TileTensor(gi_v1, row_major[BATCH, DIM]())
    var gv2_t = TileTensor(gi_v2, row_major[BATCH, DIM]())
    t1.backward["cpu", BATCH](go_t, gv1_t)
    t2.backward["cpu", BATCH, mode="all"](go_t, gv2_t)

    var ok_bwd = True
    for k in range(BATCH * DIM):
        if gi_v1[k] != gi_v2[k]:
            print("backward(all) mismatch k=", k, "  v1=", gi_v1[k], "  v2=", gi_v2[k])
            ok_bwd = False
    if ok_bwd:
        print("backward(mode=all): PASS")

    # Backward(mode=input_only) vs v1.backward_input.
    var gi_v1_in: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi_v2_in: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gv1_in_t = TileTensor(gi_v1_in, row_major[BATCH, DIM]())
    var gv2_in_t = TileTensor(gi_v2_in, row_major[BATCH, DIM]())
    t1.backward_input["cpu", BATCH](go_t, gv1_in_t)
    t2.backward["cpu", BATCH, mode="input_only"](go_t, gv2_in_t)

    var ok_in = True
    for k in range(BATCH * DIM):
        if gi_v1_in[k] != gi_v2_in[k]:
            print("backward(input_only) mismatch k=", k)
            ok_in = False
    if ok_in:
        print("backward(mode=input_only): PASS")

    if ok_fwd and ok_bwd and ok_in:
        print()
        print("PASS — TanhV2 is bit-identical to v1 Tanh on CPU.")
    else:
        raise Error("tanh_v2 parity test failed")

    in_p.free()
    out_v1.free()
    out_v2.free()
    go_p.free()
    gi_v1.free()
    gi_v2.free()
    gi_v1_in.free()
    gi_v2_in.free()
