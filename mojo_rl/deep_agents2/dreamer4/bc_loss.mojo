"""Behavior-cloning loss (paper eq. 9) — multi-token-prediction NLL.

    L_BC = − Σ_n ln p(a_{t+n} | h_t) − Σ_n ln p(r_{t+n} | h_t)     n = 0..L

For each frame at window-position t (sequence b), the agent task-output
embedding h_t (from `Dreamer4Dynamics`) drives the MTP policy + reward heads
(`heads.mojo`), whose distance-`n` logit block predicts the action / reward at
window-position t+n. Predictions with t+n ≥ T fall off the window and are
masked. Policy uses the categorical (`dists_discrete`) loss; reward uses the
symexp-twohot (`twohot`) loss.

`bc_mtp_loss` runs both head forwards on h_t, accumulates the per-(frame,
distance) NLL (normalised by the number of valid predictions), backpropagates
through the heads (filling their param grads, mode="all") and SUMS the two
grad-wrt-h_t contributions into `grad_h` — which the caller then feeds to
`Dreamer4Dynamics.set_grad_h` before `dyn.vjp` (alongside the continued
shortcut-forcing video-prediction grad). CPU; the head logits are small so the
arithmetic stays host-side (same pattern as the shortcut loss).

Convention: `actions`/`rewards` are per (b, window-position) — flat [B·T], with
position p = b·T + j (j ∈ [0,T)). `actions` holds class indices as fp ints.
"""

from std.memory import alloc

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.module import Module
from layout import TileTensor, row_major

from mojo_rl.deep_agents2.dreamerv3.dists_discrete import cat_fwd, cat_bwd
from mojo_rl.deep_agents2.dreamerv3.twohot import twohot_loss, twohot_loss_backward


def bc_n_valid(B: Int, T: Int, NMTP: Int) -> Int:
    """Number of valid (frame, distance) predictions: Σ_{b,t} min(NMTP, T−t)."""
    var per_seq = 0
    for t in range(T):
        var c = NMTP
        if T - t < c:
            c = T - t
        per_seq += c
    return B * per_seq


def bc_mtp_loss[
    PH: Module,
    RH: Module,
    B: Int,
    T: Int,
    NMTP: Int,
    NACT: Int,
    NBINS: Int,
    D_IN: Int,
](
    mut ph: PH,
    mut rh: RH,
    h: UnsafePointer[Scalar[DT], MutAnyOrigin],          # [B*T, D_IN] = h_t
    actions: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [B*T] class ids (fp)
    rewards: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [B*T]
    bins: UnsafePointer[Scalar[DT], MutAnyOrigin],       # [NBINS]
    # scratch (caller-owned)
    plog: UnsafePointer[Scalar[DT], MutAnyOrigin],       # [B*T, NMTP*NACT]
    rlog: UnsafePointer[Scalar[DT], MutAnyOrigin],       # [B*T, NMTP*NBINS]
    gpl: UnsafePointer[Scalar[DT], MutAnyOrigin],        # [B*T, NMTP*NACT]
    grl: UnsafePointer[Scalar[DT], MutAnyOrigin],        # [B*T, NMTP*NBINS]
    grad_h: UnsafePointer[Scalar[DT], MutAnyOrigin],     # [B*T, D_IN]  (output)
    grad_h_tmp: UnsafePointer[Scalar[DT], MutAnyOrigin], # [B*T, D_IN]  (scratch)
    unimix: Scalar[DT] = Scalar[DT](0.0),
    policy_weight: Scalar[DT] = Scalar[DT](1.0),
    reward_weight: Scalar[DT] = Scalar[DT](1.0),
) raises -> Float64:
    comptime BT = B * T
    comptime PLOG = NMTP * NACT
    comptime RLOG = NMTP * NBINS
    comptime DIN = D_IN

    var n_valid = bc_n_valid(B, T, NMTP)
    var inv = Scalar[DT](1.0) / Scalar[DT](n_valid)

    # ── head forwards on h_t ────────────────────────────────────────────
    var ht = TileTensor(h, row_major[BT, DIN]())
    var plt = TileTensor(plog, row_major[BT, PLOG]())
    var rlt = TileTensor(rlog, row_major[BT, RLOG]())
    ph.forward["cpu", BT](ht, output=plt)
    rh.forward["cpu", BT](ht, output=rlt)

    # ── accumulate NLL + logit grads ────────────────────────────────────
    for i in range(BT * PLOG):
        gpl[i] = Scalar[DT](0.0)
    for i in range(BT * RLOG):
        grl[i] = Scalar[DT](0.0)

    var sm = alloc[Scalar[DT]](NACT)
    var pp = alloc[Scalar[DT]](NACT)
    var loss: Float64 = 0.0
    for b in range(B):
        for j in range(T):                       # window-position of the frame
            var bt = b * T + j
            for n in range(NMTP):
                var tgt = j + n
                if tgt >= T:
                    break                        # prediction falls off window
                var pos = b * T + tgt
                # policy NLL
                var pbase = bt * PLOG + n * NACT
                var k = Int(Float64(actions[pos]) + 0.5)
                var lp_ent = cat_fwd[NACT](plog, pbase, unimix, k, sm, pp)
                loss += -Float64(lp_ent[0]) * Float64(policy_weight * inv)
                cat_bwd[NACT](
                    sm, pp, unimix, k,
                    -policy_weight * inv, Scalar[DT](0.0), gpl, pbase,
                )
                # reward CE
                var rbase = bt * RLOG + n * NBINS
                var tr = rewards[pos]
                loss += Float64(
                    twohot_loss[NBINS](rlog, rbase, bins, tr)
                ) * Float64(reward_weight * inv)
                twohot_loss_backward[NBINS](
                    rlog, rbase, bins, tr, reward_weight * inv, grl
                )
    sm.free()
    pp.free()

    # ── backprop through both heads, sum grad wrt h_t ───────────────────
    var gph = TileTensor(gpl, row_major[BT, PLOG]())
    var grh = TileTensor(grl, row_major[BT, RLOG]())
    var ghv = TileTensor(grad_h, row_major[BT, DIN]())
    var ght = TileTensor(grad_h_tmp, row_major[BT, DIN]())
    ph.vjp["cpu", BT, mode="all"](gph, ghv)            # grad_h from policy
    rh.vjp["cpu", BT, mode="all"](grh, ght)            # grad_h from reward
    for i in range(BT * DIN):
        grad_h[i] = grad_h[i] + grad_h_tmp[i]
    return loss
