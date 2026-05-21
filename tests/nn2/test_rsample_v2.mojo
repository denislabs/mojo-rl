"""Parity test: RSampleV2 matches v1 RSample on CPU, including the
freshly-drawn z noise (seeded RNG)."""

from std.memory import alloc
from std.random import seed
from layout import TileTensor, row_major

from mojo_rl.nn2.primitives.rsample import RSample
from mojo_rl.nn2.primitives.rsample_v2 import RSampleV2
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero


comptime BATCH = 3
comptime ACT = 2


def main() raises:
    var ao_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var o1_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * (ACT + 1))
    var o2_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * (ACT + 1))

    # Hand-picked inputs (mu, log_std stacked).
    ao_p[0]  = Scalar[DT](0.3); ao_p[1]  = Scalar[DT](-0.6)
    ao_p[2]  = Scalar[DT](-0.5); ao_p[3] = Scalar[DT](0.1)
    ao_p[4]  = Scalar[DT](0.8); ao_p[5]  = Scalar[DT](0.0)
    ao_p[6]  = Scalar[DT](-1.0); ao_p[7] = Scalar[DT](-0.3)
    ao_p[8]  = Scalar[DT](-0.4); ao_p[9] = Scalar[DT](0.7)
    ao_p[10] = Scalar[DT](-0.2); ao_p[11]= Scalar[DT](-0.8)

    var ao_t = TileTensor(ao_p, row_major[BATCH, 2 * ACT]())
    var o1_t = TileTensor(o1_p, row_major[BATCH, ACT + 1]())
    var o2_t = TileTensor(o2_p, row_major[BATCH, ACT + 1]())

    var r1 = RSample[ACT].make[target="cpu", INIT=Zero]()
    var r2 = RSampleV2[ACT].make[target="cpu", INIT=Zero]()
    r1.action_scale = Scalar[DT](2.0)
    r2.action_scale = Scalar[DT](2.0)

    # Both use the same global RNG via box_muller_normal — seed before
    # each forward to match z.
    seed(1234)
    r1.forward["cpu", BATCH](ao_t, o1_t)
    seed(1234)
    r2.forward["cpu", BATCH](ao_t, o2_t)

    var ok_fwd = True
    for k in range(BATCH * (ACT + 1)):
        if o1_p[k] != o2_p[k]:
            ok_fwd = False
            print("  k=", k, " v1=", o1_p[k], " v2=", o2_p[k])
    print("forward: PASS" if ok_fwd else "forward: FAIL")

    # ── Backward parity ─────────────────────────────────────────────
    var go_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * (ACT + 1))
    var gi1_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var gi2_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    # Seed grad_output: grad_action positions = 1.0, grad_log_prob = 0.5.
    for b in range(BATCH):
        for j in range(ACT):
            go_p[b * (ACT + 1) + j] = Scalar[DT](1.0)
        go_p[b * (ACT + 1) + ACT] = Scalar[DT](0.5)
    for k in range(BATCH * 2 * ACT):
        gi1_p[k] = Scalar[DT](0.0)
        gi2_p[k] = Scalar[DT](0.0)
    var go_t  = TileTensor(go_p,  row_major[BATCH, ACT + 1]())
    var gi1_t = TileTensor(gi1_p, row_major[BATCH, 2 * ACT]())
    var gi2_t = TileTensor(gi2_p, row_major[BATCH, 2 * ACT]())
    r1.backward["cpu", BATCH](go_t, gi1_t)
    r2.backward["cpu", BATCH, mode="all"](go_t, gi2_t)

    var ok_bwd = True
    for k in range(BATCH * 2 * ACT):
        if gi1_p[k] != gi2_p[k]:
            ok_bwd = False
            print("  k=", k, " v1=", gi1_p[k], " v2=", gi2_p[k])
    print("backward: PASS" if ok_bwd else "backward: FAIL")

    if ok_fwd and ok_bwd:
        print()
        print("PASS — RSampleV2 is bit-identical to v1 RSample on CPU.")
        print("       Math now delegates to canonical squashed_gaussian pair.")
    else:
        raise Error("rsample_v2 parity test failed")

    ao_p.free()
    o1_p.free()
    o2_p.free()
    go_p.free()
    gi1_p.free()
    gi2_p.free()
