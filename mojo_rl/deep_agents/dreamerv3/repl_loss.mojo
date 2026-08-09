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

from mojo_rl.nn.constants import DT
from .twohot import twohot_pred, twohot_loss, twohot_loss_backward


def repl_loss_cpu[
    BK: Int, T: Int, BINS: Int
](
    last: Pointer[Scalar[DT], MutAnyOrigin],  # [BK,T] (0/1)
    term: Pointer[Scalar[DT], MutAnyOrigin],  # [BK,T] (0/1)
    rew: Pointer[Scalar[DT], MutAnyOrigin],  # [BK,T]
    boot: Pointer[Scalar[DT], MutAnyOrigin],  # [BK,T]
    vlogits: Pointer[Scalar[DT], MutAnyOrigin],  # [BK,T,BINS]
    svlogits: Pointer[Scalar[DT], MutAnyOrigin],  # [BK,T,BINS]
    bins: Pointer[Scalar[DT], MutAnyOrigin],  # [BINS]
    horizon: Scalar[DT],
    lam: Scalar[DT],
    slowreg: Scalar[DT],
    out_repval: Pointer[Scalar[DT], MutAnyOrigin],  # [BK,T-1]
    out_ret: Pointer[Scalar[DT], MutAnyOrigin],  # [BK,T-1]
) raises:
    comptime assert T >= 2, "repl_loss needs T >= 2"
    comptime TM1 = T - 1
    var disc = Scalar[DT](1.0) - Scalar[DT](1.0) / horizon

    # ret = λ-return(last, term, rew, boot, disc, lam)
    for b in range(BK):
        var ret_next = boot[unsafe_offset=b * T + (T - 1)]
        var t = T - 2
        while t >= 0:
            var live = (Scalar[DT](1.0) - term[unsafe_offset=b * T + t + 1]) * disc
            var cont = (Scalar[DT](1.0) - last[unsafe_offset=b * T + t + 1]) * lam
            var interm = (
                rew[unsafe_offset=b * T + t + 1]
                + (Scalar[DT](1.0) - cont) * live * boot[unsafe_offset=b * T + t + 1]
            )
            var cur = interm + live * cont * ret_next
            out_ret[unsafe_offset=b * TM1 + t] = cur
            ret_next = cur
            t -= 1

    var slowval = alloc[Scalar[DT]](BK * T)
    for b in range(BK):
        for t in range(T):
            slowval[unsafe_offset=b * T + t] = twohot_pred[BINS](
                svlogits, (b * T + t) * BINS, bins
            )

    for b in range(BK):
        for t in range(TM1):
            var w = Scalar[DT](1.0) - last[unsafe_offset=b * T + t]  # f32(~last)
            var l1 = twohot_loss[BINS](
                vlogits, (b * T + t) * BINS, bins, out_ret[unsafe_offset=b * TM1 + t]
            )
            var l2 = twohot_loss[BINS](
                vlogits, (b * T + t) * BINS, bins, slowval[unsafe_offset=b * T + t]
            )
            out_repval[unsafe_offset=b * TM1 + t] = w * (l1 + slowreg * l2)

    slowval.unsafe_free()


def repl_loss_backward[
    BK: Int, T: Int, BINS: Int
](
    last: Pointer[Scalar[DT], MutAnyOrigin],  # [BK,T]
    term: List[Scalar[DT]],  # [BK,T]
    rew: Pointer[Scalar[DT], MutAnyOrigin],  # [BK,T]
    boot: List[Scalar[DT]],  # [BK,T]
    vlogits: Pointer[Scalar[DT], MutAnyOrigin],  # [BK,T,BINS]
    svlogits: Pointer[Scalar[DT], MutAnyOrigin],  # [BK,T,BINS]
    bins: Pointer[Scalar[DT], MutAnyOrigin],  # [BINS]
    horizon: Scalar[DT],
    lam: Scalar[DT],
    slowreg: Scalar[DT],
    d_repval: List[Scalar[DT]],  # [BK,T-1] cotangent
    grad_vlogits: Pointer[Scalar[DT], MutAnyOrigin],  # [BK,T,BINS]
) raises:
    """Backward of `repl_loss_cpu` w.r.t. the value logits (targets sg'd).
    grad_vlogits ZEROED then accumulated."""
    comptime assert T >= 2, "repl_loss needs T >= 2"
    comptime TM1 = T - 1
    var disc = Scalar[DT](1.0) - Scalar[DT](1.0) / horizon
    for i in range(BK * T * BINS):
        grad_vlogits[unsafe_offset=i] = 0.0

    var ret = alloc[Scalar[DT]](BK * TM1)
    var slowval = alloc[Scalar[DT]](BK * T)
    for b in range(BK):
        var ret_next = boot[b * T + (T - 1)]
        var t = T - 2
        while t >= 0:
            var live = (Scalar[DT](1.0) - term[b * T + t + 1]) * disc
            var cont = (Scalar[DT](1.0) - last[unsafe_offset=b * T + t + 1]) * lam
            var interm = (
                rew[unsafe_offset=b * T + t + 1]
                + (Scalar[DT](1.0) - cont) * live * boot[b * T + t + 1]
            )
            ret[unsafe_offset=b * TM1 + t] = interm + live * cont * ret_next
            ret_next = ret[unsafe_offset=b * TM1 + t]
            t -= 1
        for t2 in range(T):
            slowval[unsafe_offset=b * T + t2] = twohot_pred[BINS](
                svlogits, (b * T + t2) * BINS, bins
            )

    for b in range(BK):
        for t in range(TM1):
            var up = d_repval[b * TM1 + t] * (Scalar[DT](1.0) - last[unsafe_offset=b * T + t])
            twohot_loss_backward[BINS](
                vlogits,
                (b * T + t) * BINS,
                bins,
                ret[unsafe_offset=b * TM1 + t],
                up,
                grad_vlogits,
            )
            twohot_loss_backward[BINS](
                vlogits,
                (b * T + t) * BINS,
                bins,
                slowval[unsafe_offset=b * T + t],
                up * slowreg,
                grad_vlogits,
            )

    ret.unsafe_free()
    slowval.unsafe_free()
