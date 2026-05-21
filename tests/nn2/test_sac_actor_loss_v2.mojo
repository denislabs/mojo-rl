"""Parity test: sac_actor_loss_v2 matches v1 sac_actor_loss bit-for-bit on CPU.

Same actor_output / z / grad_action / alpha / action_scale produces the
same grad_actor_output between v1's inline-α-fold-in `sac_actor_backward`
and v2's delegate-to-canonical wrapper.

Also verifies `squashed_gaussian_sample` alias and `sac_actor_loss_value`.
"""

from std.memory import alloc
from layout import TileTensor, row_major

from mojo_rl.nn2.loss.sac_actor_loss import (
    squashed_gaussian_sample as v1_sample,
    sac_actor_backward as v1_backward,
    sac_actor_loss_value as v1_loss_value,
)
from mojo_rl.nn2.loss.sac_actor_loss_v2 import (
    squashed_gaussian_sample as v2_sample,
    sac_actor_backward as v2_backward,
    sac_actor_loss_value as v2_loss_value,
)
from mojo_rl.nn2.constants import DT


comptime BATCH = 3
comptime ACT = 2


def _seed(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int, salt: UInt64):
    var state: UInt64 = salt
    for k in range(n):
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var r = Scalar[DT]((Int(state >> 32) & 0xFFFF)) / Scalar[DT](65535.0)
        p[k] = (r - Scalar[DT](0.5))


def main() raises:
    var ao_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var z_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * ACT)
    var a1:   UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * ACT)
    var a2:   UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * ACT)
    var lp1:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var lp2:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    _seed(ao_p, BATCH * 2 * ACT, UInt64(0x5AC0))
    _seed(z_p,  BATCH * ACT,     UInt64(0x5AC1))

    var ao_t = TileTensor(ao_p, row_major[BATCH, 2 * ACT]())
    var z_t  = TileTensor(z_p,  row_major[BATCH, ACT]())
    var a1_t = TileTensor(a1, row_major[BATCH, ACT]())
    var a2_t = TileTensor(a2, row_major[BATCH, ACT]())
    var lp1_t = TileTensor(lp1, row_major[BATCH]())
    var lp2_t = TileTensor(lp2, row_major[BATCH]())

    var action_scale = Scalar[DT](2.0)

    # ── squashed_gaussian_sample alias parity ────────────────────────
    v1_sample[ACT, BATCH](ao_t, z_t, action_scale, a1_t, lp1_t)
    v2_sample[ACT, BATCH](ao_t, z_t, action_scale, a2_t, lp2_t)
    var ok_sample = True
    for k in range(BATCH * ACT):
        if a1[k] != a2[k]:
            ok_sample = False
    for k in range(BATCH):
        if lp1[k] != lp2[k]:
            ok_sample = False
    print("squashed_gaussian_sample: PASS" if ok_sample else "FAIL")

    # ── sac_actor_backward parity ────────────────────────────────────
    var ga_p:   UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * ACT)
    var gao1_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var gao2_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    _seed(ga_p, BATCH * ACT, UInt64(0x5AC2))
    var ga_t   = TileTensor(ga_p,   row_major[BATCH, ACT]())
    var gao1_t = TileTensor(gao1_p, row_major[BATCH, 2 * ACT]())
    var gao2_t = TileTensor(gao2_p, row_major[BATCH, 2 * ACT]())

    var alpha = Scalar[DT](0.2)
    v1_backward[ACT, BATCH](ao_t, z_t, ga_t, alpha, action_scale, gao1_t)
    v2_backward[ACT, BATCH](ao_t, z_t, ga_t, alpha, action_scale, gao2_t)

    var ok_bwd = True
    for k in range(BATCH * 2 * ACT):
        if gao1_p[k] != gao2_p[k]:
            ok_bwd = False
            print("  k=", k, " v1=", gao1_p[k], " v2=", gao2_p[k])
    print("sac_actor_backward: PASS" if ok_bwd else "FAIL")

    # ── sac_actor_loss_value parity ──────────────────────────────────
    var minq: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    _seed(minq, BATCH, UInt64(0x5AC3))
    var minq_t = TileTensor(minq, row_major[BATCH]())
    var v1_loss = v1_loss_value[BATCH](lp1_t, minq_t, alpha)
    var v2_loss = v2_loss_value[BATCH](lp1_t, minq_t, alpha)
    var ok_loss = v1_loss == v2_loss
    print("sac_actor_loss_value: PASS" if ok_loss else "FAIL")

    if ok_sample and ok_bwd and ok_loss:
        print()
        print("PASS — sac_actor_loss_v2 is bit-identical to v1.")
        print("       Backward delegates α-fold-in to canonical pair via grad_log_prob.")
    else:
        raise Error("sac_actor_loss_v2 parity test failed")

    ao_p.free()
    z_p.free()
    a1.free()
    a2.free()
    lp1.free()
    lp2.free()
    ga_p.free()
    gao1_p.free()
    gao2_p.free()
    minq.free()
