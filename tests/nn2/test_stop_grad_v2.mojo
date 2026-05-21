"""Parity test: StopGradV2 matches v1 StopGrad bit-for-bit on CPU."""

from std.memory import alloc
from layout import TileTensor, row_major

from mojo_rl.nn2.primitives.stop_grad import StopGrad
from mojo_rl.nn2.primitives.stop_grad_v2 import StopGradV2
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero


comptime BATCH = 4
comptime DIM = 6


def main() raises:
    var in_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var state: UInt64 = UInt64(0xFEEDFACE)
    for k in range(BATCH * DIM):
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var r = Scalar[DT]((Int(state >> 32) & 0xFFFF)) / Scalar[DT](65535.0)
        in_p[k] = (r - Scalar[DT](0.5))

    var out_v1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var out_v2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)

    var s1 = StopGrad[DIM].make[target="cpu", INIT=Zero]()
    var s2 = StopGradV2[DIM].make[target="cpu", INIT=Zero]()

    var in_t  = TileTensor(in_p, row_major[BATCH, DIM]())
    var ov1_t = TileTensor(out_v1, row_major[BATCH, DIM]())
    var ov2_t = TileTensor(out_v2, row_major[BATCH, DIM]())
    s1.forward["cpu", BATCH](in_t, ov1_t)
    s2.forward["cpu", BATCH](in_t, ov2_t)

    var ok_fwd = True
    for k in range(BATCH * DIM):
        if out_v1[k] != out_v2[k]:
            ok_fwd = False
    print("forward: PASS" if ok_fwd else "forward: FAIL")

    # Backward: grad_input must be zeros regardless of grad_output.
    var go_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi_v1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    var gi_v2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * DIM)
    for k in range(BATCH * DIM):
        go_p[k] = Scalar[DT](k) + Scalar[DT](1.0)  # nonzero so we can prove zeroing happens
    var go_t  = TileTensor(go_p,  row_major[BATCH, DIM]())
    var gv1_t = TileTensor(gi_v1, row_major[BATCH, DIM]())
    var gv2_t = TileTensor(gi_v2, row_major[BATCH, DIM]())
    s1.backward["cpu", BATCH](go_t, gv1_t)
    s2.backward["cpu", BATCH, mode="all"](go_t, gv2_t)

    var ok_bwd = True
    for k in range(BATCH * DIM):
        if gi_v1[k] != gi_v2[k] or gi_v2[k] != Scalar[DT](0.0):
            ok_bwd = False
    print("backward(mode=all): PASS" if ok_bwd else "backward(mode=all): FAIL")

    if ok_fwd and ok_bwd:
        print()
        print("PASS — StopGradV2 is bit-identical to v1 StopGrad on CPU.")
    else:
        raise Error("stop_grad_v2 parity test failed")

    in_p.free()
    out_v1.free()
    out_v2.free()
    go_p.free()
    gi_v1.free()
    gi_v2.free()
