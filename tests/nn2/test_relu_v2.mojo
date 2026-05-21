"""Parity test: ReLUV2 matches the v1 ReLU bit-for-bit on CPU.

Runs the same input through both, compares output buffers byte-for-byte.
Same for backward, both for `mode="all"` and `mode="input_only"` (the
latter on v1 is `backward_input`).

This is the lighthouse test that says "the retrofit's new patterns
produce the same numerics as the existing nn2."
"""

from std.memory import alloc
from layout import TileTensor, row_major

from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.relu_v2 import ReLUV2
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero


comptime BATCH = 4
comptime DIM = 6


def main() raises:
    # Mixed positive/negative/zero inputs to exercise the mask boundary.
    var in_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var state: UInt64 = UInt64(0xDEADBEEF)
    for k in range(BATCH * DIM):
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var r = Scalar[DT]((Int(state >> 32) & 0xFFFF)) / Scalar[DT](65535.0)
        in_p[k] = (r - Scalar[DT](0.5)) * Scalar[DT](2.0)
    # Pin a couple of cells to exactly 0 to test boundary.
    in_p[0] = 0.0
    in_p[5] = 0.0

    var out_v1:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var out_v2:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)

    # ── Forward parity. ──
    var r1 = ReLU[DIM].make[target="cpu", INIT=Zero]()
    var r2 = ReLUV2[DIM].make[target="cpu", INIT=Zero]()

    var in_t  = TileTensor(in_p, row_major[BATCH, DIM]())
    var ov1_t = TileTensor(out_v1, row_major[BATCH, DIM]())
    var ov2_t = TileTensor(out_v2, row_major[BATCH, DIM]())
    r1.forward["cpu", BATCH](in_t, ov1_t)
    r2.forward["cpu", BATCH](in_t, ov2_t)

    var ok_fwd = True
    for k in range(BATCH * DIM):
        if out_v1[k] != out_v2[k]:
            print("forward mismatch at k=", k,
                  "  v1=", out_v1[k], "  v2=", out_v2[k])
            ok_fwd = False
    if ok_fwd:
        print("forward: PASS (", BATCH * DIM, "elements bit-equal )")

    # ── Backward parity (mode=all). ──
    var go_p:    UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi_v1:   UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi_v2:   UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    for k in range(BATCH * DIM):
        go_p[k] = Scalar[DT](1.0) + Scalar[DT](k) * Scalar[DT](0.1)

    var go_t   = TileTensor(go_p, row_major[BATCH, DIM]())
    var gv1_t  = TileTensor(gi_v1, row_major[BATCH, DIM]())
    var gv2_t  = TileTensor(gi_v2, row_major[BATCH, DIM]())
    r1.backward["cpu", BATCH](go_t, gv1_t)
    r2.backward["cpu", BATCH, mode="all"](go_t, gv2_t)

    var ok_bwd = True
    for k in range(BATCH * DIM):
        if gi_v1[k] != gi_v2[k]:
            print("backward(all) mismatch at k=", k,
                  "  v1=", gi_v1[k], "  v2=", gi_v2[k])
            ok_bwd = False
    if ok_bwd:
        print("backward(mode=all): PASS")

    # ── Backward(mode=input_only) parity vs v1.backward_input. ──
    var gi_v1_in:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi_v2_in:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gv1_in_t   = TileTensor(gi_v1_in, row_major[BATCH, DIM]())
    var gv2_in_t   = TileTensor(gi_v2_in, row_major[BATCH, DIM]())
    r1.backward_input["cpu", BATCH](go_t, gv1_in_t)
    r2.backward["cpu", BATCH, mode="input_only"](go_t, gv2_in_t)

    var ok_in = True
    for k in range(BATCH * DIM):
        if gi_v1_in[k] != gi_v2_in[k]:
            print("backward(input_only) mismatch at k=", k,
                  "  v1=", gi_v1_in[k], "  v2=", gi_v2_in[k])
            ok_in = False
    if ok_in:
        print("backward(mode=input_only): PASS")

    if ok_fwd and ok_bwd and ok_in:
        print()
        print("PASS — ReLUV2 is bit-identical to v1 ReLU on CPU.")
    else:
        raise Error("relu_v2 parity test failed")

    in_p.free()
    out_v1.free()
    out_v2.free()
    go_p.free()
    gi_v1.free()
    gi_v2.free()
    gi_v1_in.free()
    gi_v2_in.free()


