"""FD-gradcheck for the discrete (unimix categorical) imag_loss policy path.

Builds a synthetic imagination rollout (one-hot actions, random reward/cont/
value logits + policy logits), runs `imag_loss_cpu[..., DISCRETE=True]`, and
verifies `imag_loss_backward[..., DISCRETE=True]` produces logit gradients
matching central finite differences of  L = Σ_{b,t} policy_loss[b,t].

Since the policy logits affect ONLY the policy term (ret / adv / rscale are
functions of value & reward, not the actor), uses `PercentileNormalize("none")`
(rscale=1) so `adv` stays moderate and the FD signal isn't lost to
cancellation; the categorical grad is unaffected by the constant divisor.
`grad_pstd_raw` must stay 0 in discrete mode.

Run: pixi run mojo run -I . tests/nn/test_dreamerv3_imag_loss_discrete.mojo
"""

from std.memory import alloc
from std.math import abs
from std.random import random_float64, seed
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.deep_agents.dreamerv3.imag_loss import (
    imag_loss_cpu, imag_loss_backward,
)
from mojo_rl.deep_agents.dreamerv3.normalize import PercentileNormalize


comptime BK = 3
comptime T = 4
comptime ACT = 4
comptime BINS = 7
comptime TM1 = T - 1


def _sum_policy(
    logits: Pointer[Scalar[DT], MutAnyOrigin],
    act: Pointer[Scalar[DT], MutAnyOrigin],
    rew: Pointer[Scalar[DT], MutAnyOrigin],
    con: Pointer[Scalar[DT], MutAnyOrigin],
    vlog: Pointer[Scalar[DT], MutAnyOrigin],
    svlog: Pointer[Scalar[DT], MutAnyOrigin],
    pstd: Pointer[Scalar[DT], MutAnyOrigin],
    bins: Pointer[Scalar[DT], MutAnyOrigin],
    lam: Scalar[DT], actent: Scalar[DT], slowreg: Scalar[DT],
) raises -> Scalar[DT]:
    var pol = List[Scalar[DT]](length=BK * TM1, fill=Scalar[DT](0))
    var val = List[Scalar[DT]](length=BK * TM1, fill=Scalar[DT](0))
    var ret = List[Scalar[DT]](length=BK * TM1, fill=Scalar[DT](0))
    var rn = PercentileNormalize.make("none")
    imag_loss_cpu[BK, T, ACT, BINS, True](
        act, rew, con, vlog, svlog, logits, pstd, bins,
        Scalar[DT](0.1), Scalar[DT](1.0), lam, actent, slowreg,
        rn, pol, val, ret,
    )
    var s = Scalar[DT](0.0)
    for i in range(BK * TM1):
        s += pol[i]
    return s


def main() raises:
    print("=" * 70)
    print("DreamerV3 discrete imag_loss — FD gradcheck (policy logits)")
    print("=" * 70)
    seed(776655)
    var lam = Scalar[DT](0.95)
    var actent = Scalar[DT](3e-4)
    var slowreg = Scalar[DT](1.0)

    var act = alloc[Scalar[DT]](BK * T * ACT).as_unsafe_any_origin()
    var rew = alloc[Scalar[DT]](BK * T).as_unsafe_any_origin()
    var con = alloc[Scalar[DT]](BK * T).as_unsafe_any_origin()
    var vlog = alloc[Scalar[DT]](BK * T * BINS).as_unsafe_any_origin()
    var svlog = alloc[Scalar[DT]](BK * T * BINS).as_unsafe_any_origin()
    var logits = alloc[Scalar[DT]](BK * T * ACT).as_unsafe_any_origin()
    var pstd = alloc[Scalar[DT]](BK * T * ACT).as_unsafe_any_origin()   # unused in discrete
    var bins = alloc[Scalar[DT]](BINS).as_unsafe_any_origin()

    for c in range(BINS):
        bins[c] = Scalar[DT](-3.0) + Scalar[DT](6.0) * Scalar[DT](c) / Scalar[DT](BINS - 1)
    for b in range(BK):
        for t in range(T):
            rew[b * T + t] = Scalar[DT](random_float64() * 0.5 - 0.25)
            con[b * T + t] = Scalar[DT](0.92)
            # one-hot action: pick a class
            var k = Int(random_float64() * Scalar[DT](ACT).cast[DType.float64]())
            if k >= ACT:
                k = ACT - 1
            for a in range(ACT):
                act[(b * T + t) * ACT + a] = Scalar[DT](1.0) if a == k else Scalar[DT](0.0)
                logits[(b * T + t) * ACT + a] = Scalar[DT](random_float64() * 2.0 - 1.0)
                pstd[(b * T + t) * ACT + a] = Scalar[DT](0.0)
            for c in range(BINS):
                vlog[(b * T + t) * BINS + c] = Scalar[DT](random_float64() * 0.5)
                svlog[(b * T + t) * BINS + c] = Scalar[DT](random_float64() * 0.5)

    # forward (for rscale) + analytic backward
    var pol = List[Scalar[DT]](length=BK * TM1, fill=Scalar[DT](0))
    var val = List[Scalar[DT]](length=BK * TM1, fill=Scalar[DT](0))
    var ret = List[Scalar[DT]](length=BK * TM1, fill=Scalar[DT](0))
    var rn = PercentileNormalize.make("none")
    imag_loss_cpu[BK, T, ACT, BINS, True](
        act, rew, con, vlog, svlog, logits, pstd, bins,
        Scalar[DT](0.1), Scalar[DT](1.0), lam, actent, slowreg,
        rn, pol, val, ret,
    )
    var rscale = rn.stats()[1]
    var d_pol = alloc[Scalar[DT]](BK * TM1).as_unsafe_any_origin()
    var d_val = alloc[Scalar[DT]](BK * TM1).as_unsafe_any_origin()
    for i in range(BK * TM1):
        d_pol[i] = 1.0
        d_val[i] = 0.0
    var g_vlog = alloc[Scalar[DT]](BK * T * BINS).as_unsafe_any_origin()
    var g_logits = alloc[Scalar[DT]](BK * T * ACT).as_unsafe_any_origin()
    var g_pstd = alloc[Scalar[DT]](BK * T * ACT).as_unsafe_any_origin()
    imag_loss_backward[BK, T, ACT, BINS, True](
        act, rew, con, vlog, svlog, logits, pstd, bins,
        Scalar[DT](0.1), Scalar[DT](1.0), lam, actent, slowreg, rscale,
        d_pol, d_val, g_vlog, g_logits, g_pstd,
    )
    # grad_pstd_raw must be untouched (zeroed) in discrete mode
    var max_pstd: Scalar[DT] = 0.0
    for i in range(BK * T * ACT):
        if abs(g_pstd[i]) > max_pstd:
            max_pstd = abs(g_pstd[i])
    assert_true(max_pstd == Scalar[DT](0.0), "grad_pstd_raw stays 0 (discrete)")

    # central FD on each logit
    var eps = Scalar[DT](1e-3)
    var max_rel: Scalar[DT] = 0.0
    var worst_fd: Scalar[DT] = 0.0
    var worst_an: Scalar[DT] = 0.0
    var worst_i: Int = 0
    for i in range(BK * T * ACT):
        var orig = logits[i]
        logits[i] = orig + eps
        var lp = _sum_policy(logits, act, rew, con, vlog, svlog, pstd, bins, lam, actent, slowreg)
        logits[i] = orig - eps
        var lm = _sum_policy(logits, act, rew, con, vlog, svlog, pstd, bins, lam, actent, slowreg)
        logits[i] = orig
        var fd = (lp - lm) / (Scalar[DT](2.0) * eps)
        var an = g_logits[i]
        # floored denom: tiny gradients (|grad|≪1) hit the FD cancellation
        # noise floor, so an absolute floor keeps their rel error meaningful.
        var denom = abs(fd) + abs(an) + Scalar[DT](1e-2)
        var rel = abs(fd - an) / denom
        if rel > max_rel:
            max_rel = rel
            worst_fd = fd
            worst_an = an
            worst_i = i

    print("  worst i=", worst_i, " fd=", worst_fd, " analytic=", worst_an)
    print("  max relative grad error =", max_rel)
    print("  max |grad_pstd_raw| =", max_pstd, "(should be 0)")
    assert_true(max_rel < Scalar[DT](2e-2), "discrete imag_loss grad matches FD")

    act.free(); rew.free(); con.free(); vlog.free(); svlog.free()
    logits.free(); pstd.free(); bins.free()  # pol/val/ret are now owned Lists
    d_pol.free(); d_val.free(); g_vlog.free(); g_logits.free(); g_pstd.free()
    print("=" * 70)
    print("PASSED — discrete imag_loss forward + policy gradient verified")
    print("=" * 70)
