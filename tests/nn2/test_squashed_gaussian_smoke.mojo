"""Smoke test: canonical squashed_gaussian forward/backward compute
sensible values for a single batch of known inputs."""

from std.math import abs as fabs
from std.memory import alloc
from layout import TileTensor, row_major

from mojo_rl.nn2.loss.squashed_gaussian import (
    squashed_gaussian_forward,
    squashed_gaussian_backward,
)
from mojo_rl.nn2.constants import DT


comptime BATCH = 2
comptime ACT = 3


def main() raises:
    var ao_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    var z_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * ACT)
    var a_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * ACT)
    var lp_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)

    # Batch 0: mu=0, log_std=0, z=0  → action=0, sensible log_prob.
    # Batch 1: mu=0.5, log_std=-1, z=1.
    for j in range(ACT):
        ao_p[0 * 2 * ACT + j]       = Scalar[DT](0.0)
        ao_p[0 * 2 * ACT + ACT + j] = Scalar[DT](0.0)
        ao_p[1 * 2 * ACT + j]       = Scalar[DT](0.5)
        ao_p[1 * 2 * ACT + ACT + j] = Scalar[DT](-1.0)
        z_p[0 * ACT + j] = Scalar[DT](0.0)
        z_p[1 * ACT + j] = Scalar[DT](1.0)

    var ao_t = TileTensor(ao_p, row_major[BATCH, 2 * ACT]())
    var z_t  = TileTensor(z_p,  row_major[BATCH, ACT]())
    var a_t  = TileTensor(a_p,  row_major[BATCH, ACT]())
    var lp_t = TileTensor(lp_p, row_major[BATCH]())

    squashed_gaussian_forward[ACT, BATCH](
        ao_t, z_t, Scalar[DT](2.0), a_t, lp_t,
    )

    # Batch 0: z=0 → pre=mu=0 → tanh(0)=0 → action=0.
    var ok_fwd = True
    for j in range(ACT):
        if fabs(a_p[0 * ACT + j]) > Scalar[DT](1e-6):
            ok_fwd = False
            print("batch0 action[", j, "]=", a_p[0 * ACT + j], "expected ~0")

    # Backward smoke: provide unit grad_action + unit grad_log_prob,
    # verify the chain math runs without NaN.
    var ga_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * ACT)
    var glp_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var gao_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * 2 * ACT)
    for k in range(BATCH * ACT):
        ga_p[k] = Scalar[DT](1.0)
    for k in range(BATCH):
        glp_p[k] = Scalar[DT](1.0)
    var ga_t  = TileTensor(ga_p,  row_major[BATCH, ACT]())
    var glp_t = TileTensor(glp_p, row_major[BATCH]())
    var gao_t = TileTensor(gao_p, row_major[BATCH, 2 * ACT]())
    squashed_gaussian_backward[ACT, BATCH](
        ao_t, z_t, ga_t, glp_t, Scalar[DT](2.0), gao_t,
    )

    # Smoke check: no NaN, no inf, all values finite.
    var ok_bwd = True
    for k in range(BATCH * 2 * ACT):
        var v = gao_p[k]
        if v != v:  # NaN check
            ok_bwd = False
        if fabs(v) > Scalar[DT](1e6):
            ok_bwd = False

    if ok_fwd and ok_bwd:
        print("PASS — canonical squashed_gaussian forward + backward run cleanly.")
    else:
        print("FAIL — ok_fwd=", ok_fwd, " ok_bwd=", ok_bwd)
        raise Error("squashed_gaussian smoke failed")

    ao_p.free()
    z_p.free()
    a_p.free()
    lp_p.free()
    ga_p.free()
    glp_p.free()
    gao_p.free()
