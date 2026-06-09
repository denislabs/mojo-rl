"""DreamerV3 replay value loss (forward).

Ports `dreamerv3/agent.py:repl_loss` for the v1 config (`slowtar=False`,
`valnorm=none`). Bootstraps the value head on REAL replay transitions
(vs imagined), using the imagined return `boot` as the terminal bootstrap.

  disc   = 1 - 1/horizon
  weight = f32(~last)                              # 1 where not episode-last
  ret    = λ-return(last, term, rew, boot, disc, lam)          [BK, T-1]
  repval = weight[:-1] · (twohot_ce(vlogits, ret) +
                          slowreg·twohot_ce(vlogits, slowval.pred()))

Gated on `repval_loss=True` (the reference default). Forward only (PR5a),
validated ≤1e-4 vs the actual reference.
"""

from std.memory import alloc

from mojo_rl.nn2.constants import DT
from .twohot import twohot_pred, twohot_loss, twohot_loss_backward


def repl_loss_cpu[
    BK: Int, T: Int, BINS: Int
](
    last: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [BK,T] (0/1)
    term: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [BK,T] (0/1)
    rew: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [BK,T]
    boot: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [BK,T]
    vlogits: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [BK,T,BINS]
    svlogits: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [BK,T,BINS]
    bins: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [BINS]
    horizon: Scalar[DT],
    lam: Scalar[DT],
    slowreg: Scalar[DT],
    out_repval: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [BK,T-1]
    out_ret: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [BK,T-1]
) raises:
    comptime assert T >= 2, "repl_loss needs T >= 2"
    comptime TM1 = T - 1
    var disc = Scalar[DT](1.0) - Scalar[DT](1.0) / horizon

    # ret = λ-return(last, term, rew, boot, disc, lam)
    for b in range(BK):
        var ret_next = boot[b * T + (T - 1)]
        var t = T - 2
        while t >= 0:
            var live = (Scalar[DT](1.0) - term[b * T + t + 1]) * disc
            var cont = (Scalar[DT](1.0) - last[b * T + t + 1]) * lam
            var interm = (
                rew[b * T + t + 1]
                + (Scalar[DT](1.0) - cont) * live * boot[b * T + t + 1]
            )
            var cur = interm + live * cont * ret_next
            out_ret[b * TM1 + t] = cur
            ret_next = cur
            t -= 1

    var slowval = alloc[Scalar[DT]](BK * T)
    for b in range(BK):
        for t in range(T):
            slowval[b * T + t] = twohot_pred[BINS](
                svlogits, (b * T + t) * BINS, bins
            )

    for b in range(BK):
        for t in range(TM1):
            var w = Scalar[DT](1.0) - last[b * T + t]  # f32(~last)
            var l1 = twohot_loss[BINS](
                vlogits, (b * T + t) * BINS, bins, out_ret[b * TM1 + t]
            )
            var l2 = twohot_loss[BINS](
                vlogits, (b * T + t) * BINS, bins, slowval[b * T + t]
            )
            out_repval[b * TM1 + t] = w * (l1 + slowreg * l2)

    slowval.free()


def repl_loss_backward[
    BK: Int, T: Int, BINS: Int
](
    last: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [BK,T]
    term: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [BK,T]
    rew: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [BK,T]
    boot: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [BK,T]
    vlogits: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [BK,T,BINS]
    svlogits: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [BK,T,BINS]
    bins: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [BINS]
    horizon: Scalar[DT],
    lam: Scalar[DT],
    slowreg: Scalar[DT],
    d_repval: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [BK,T-1] cotangent
    grad_vlogits: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [BK,T,BINS]
) raises:
    """Backward of `repl_loss_cpu` w.r.t. the value logits (targets sg'd).
    grad_vlogits ZEROED then accumulated."""
    comptime assert T >= 2, "repl_loss needs T >= 2"
    comptime TM1 = T - 1
    var disc = Scalar[DT](1.0) - Scalar[DT](1.0) / horizon
    for i in range(BK * T * BINS):
        grad_vlogits[i] = 0.0

    var ret = alloc[Scalar[DT]](BK * TM1)
    var slowval = alloc[Scalar[DT]](BK * T)
    for b in range(BK):
        var ret_next = boot[b * T + (T - 1)]
        var t = T - 2
        while t >= 0:
            var live = (Scalar[DT](1.0) - term[b * T + t + 1]) * disc
            var cont = (Scalar[DT](1.0) - last[b * T + t + 1]) * lam
            var interm = (
                rew[b * T + t + 1]
                + (Scalar[DT](1.0) - cont) * live * boot[b * T + t + 1]
            )
            ret[b * TM1 + t] = interm + live * cont * ret_next
            ret_next = ret[b * TM1 + t]
            t -= 1
        for t2 in range(T):
            slowval[b * T + t2] = twohot_pred[BINS](
                svlogits, (b * T + t2) * BINS, bins
            )

    for b in range(BK):
        for t in range(TM1):
            var up = d_repval[b * TM1 + t] * (Scalar[DT](1.0) - last[b * T + t])
            twohot_loss_backward[BINS](
                vlogits,
                (b * T + t) * BINS,
                bins,
                ret[b * TM1 + t],
                up,
                grad_vlogits,
            )
            twohot_loss_backward[BINS](
                vlogits,
                (b * T + t) * BINS,
                bins,
                slowval[b * T + t],
                up * slowreg,
                grad_vlogits,
            )

    ret.free()
    slowval.free()
