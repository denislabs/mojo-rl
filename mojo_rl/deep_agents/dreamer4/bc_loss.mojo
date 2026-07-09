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
through the heads (filling their param grads) and SUMS the two
grad-wrt-h_t contributions into `grad_h` — which the caller then feeds to
`Dreamer4Dynamics.set_grad_h` before `dyn.vjp` (alongside the continued
shortcut-forcing video-prediction grad). CPU; the head logits are small so the
arithmetic stays host-side (same pattern as the shortcut loss).

Convention: `actions`/`rewards` are per (b, window-position) — flat [B·T], with
position p = b·T + j (j ∈ [0,T)). `actions` holds class indices as fp ints.
"""

from std.memory import alloc

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.call import call_forward, call_vjp

from mojo_rl.deep_agents.dreamerv3.dists_discrete import cat_fwd, cat_bwd
from mojo_rl.deep_agents.dreamerv3.twohot import twohot_loss, twohot_loss_backward


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
    actions_o: Origin[mut=True],
    rewards_o: Origin[mut=True],
    bins_o: Origin[mut=True],
](
    mut ph: PH,
    mut rh: RH,
    # `h`/`plog`/`rlog`/`gpl`/`grl`/`grad_h`/`grad_h_tmp` are caller-owned raw
    # scratch/IO buffers — the sanctioned raw-pointer boundary. They are bridged
    # into boundary `Tensor`s around the storage head forward/vjp, and the
    # cat/twohot host helpers read/write them directly, so they stay MutAnyOrigin.
    h: UnsafePointer[Scalar[DT], MutAnyOrigin],          # [B*T, D_IN] = h_t
    actions: UnsafePointer[Scalar[DT], actions_o],    # [B*T] class ids (fp)
    rewards: UnsafePointer[Scalar[DT], rewards_o],    # [B*T]
    bins: UnsafePointer[Scalar[DT], bins_o],       # [NBINS]
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
    # Bridge the caller's raw `h` pointer into a boundary Tensor; run each head
    # through the storage Module surface; copy the head logits back to the raw
    # `plog`/`rlog` pointers so the cat/twohot host helpers (raw-pointer) work
    # unchanged.
    var h_t = Tensor.alloc(BT * DIN)
    for i in range(BT * DIN):
        h_t.data[i] = h[i]
    var plog_t = Tensor.alloc(BT * PLOG)
    var rlog_t = Tensor.alloc(BT * RLOG)
    call_forward["cpu", BT](ph, TensorRefs[PH.ARITY](h_t), plog_t, None)
    call_forward["cpu", BT](rh, TensorRefs[RH.ARITY](h_t), rlog_t, None)
    for i in range(BT * PLOG):
        plog[i] = plog_t.data[i]
    for i in range(BT * RLOG):
        rlog[i] = rlog_t.data[i]

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
    # The NLL loop filled the raw `gpl`/`grl` logit-grads; bridge those into
    # boundary grad-output Tensors and run each head's vjp on the SAME `h_t`
    # used for the forward (heads recompute their forward from it). Copy the
    # resulting grad-wrt-h back to the raw `grad_h`/`grad_h_tmp` pointers.
    var gpl_t = Tensor.alloc(BT * PLOG)
    for i in range(BT * PLOG):
        gpl_t.data[i] = gpl[i]
    var grl_t = Tensor.alloc(BT * RLOG)
    for i in range(BT * RLOG):
        grl_t.data[i] = grl[i]
    var grad_h_t = Tensor.alloc(BT * DIN)
    var grad_h_tmp_t = Tensor.alloc(BT * DIN)
    call_vjp["cpu", BT](
        ph, TensorRefs[PH.ARITY](h_t), gpl_t, TensorRefs[PH.ARITY](grad_h_t), None
    )  # grad_h from policy
    call_vjp["cpu", BT](
        rh, TensorRefs[RH.ARITY](h_t), grl_t, TensorRefs[RH.ARITY](grad_h_tmp_t),
        None,
    )  # grad_h from reward
    for i in range(BT * DIN):
        grad_h[i] = grad_h_t.data[i]
    for i in range(BT * DIN):
        grad_h_tmp[i] = grad_h_tmp_t.data[i]
    for i in range(BT * DIN):
        grad_h[i] = grad_h[i] + grad_h_tmp[i]
    return loss


def bc_policy_only_loss[
    PH: Module,
    B: Int,
    T: Int,
    NMTP: Int,
    NACT: Int,
    D_IN: Int,
    actions_o: Origin[mut=True],
](
    mut ph: PH,
    h: UnsafePointer[Scalar[DT], MutAnyOrigin],          # [B*T, D_IN] = h_t
    actions: UnsafePointer[Scalar[DT], actions_o],       # [B*T] class ids (fp)
    plog: UnsafePointer[Scalar[DT], MutAnyOrigin],       # scratch [B*T, NMTP*NACT]
    gpl: UnsafePointer[Scalar[DT], MutAnyOrigin],        # scratch [B*T, NMTP*NACT]
    grad_h: UnsafePointer[Scalar[DT], MutAnyOrigin],     # THROWAWAY [B*T, D_IN]
    unimix: Scalar[DT] = Scalar[DT](0.0),
    policy_weight: Scalar[DT] = Scalar[DT](1.0),
) raises -> Float64:
    """POLICY-ONLY BC (the policy half of `bc_mtp_loss`) — MTP action NLL through
    a single policy head. Accumulates `ph`'s PARAM grads and returns the loss.
    `grad_h` is a THROWAWAY: the caller does NOT feed it to the dynamics. Used to
    BC-train a frozen behavioral-prior head (`ph_prior`) that must stay a diverse
    anchor for the imagination reverse-KL WITHOUT perturbing the world-model
    gradient (vs the old self-snapshot prior, which collapsed with the policy)."""
    comptime BT = B * T
    comptime PLOG = NMTP * NACT
    comptime DIN = D_IN

    var n_valid = bc_n_valid(B, T, NMTP)
    var inv = Scalar[DT](1.0) / Scalar[DT](n_valid)

    var h_t = Tensor.alloc(BT * DIN)
    for i in range(BT * DIN):
        h_t.data[i] = h[i]
    var plog_t = Tensor.alloc(BT * PLOG)
    call_forward["cpu", BT](ph, TensorRefs[PH.ARITY](h_t), plog_t, None)
    for i in range(BT * PLOG):
        plog[i] = plog_t.data[i]

    for i in range(BT * PLOG):
        gpl[i] = Scalar[DT](0.0)

    var sm = alloc[Scalar[DT]](NACT)
    var pp = alloc[Scalar[DT]](NACT)
    var loss: Float64 = 0.0
    for b in range(B):
        for j in range(T):
            var bt = b * T + j
            for n in range(NMTP):
                var tgt = j + n
                if tgt >= T:
                    break
                var pos = b * T + tgt
                var pbase = bt * PLOG + n * NACT
                var k = Int(Float64(actions[pos]) + 0.5)
                var lp_ent = cat_fwd[NACT](plog, pbase, unimix, k, sm, pp)
                loss += -Float64(lp_ent[0]) * Float64(policy_weight * inv)
                cat_bwd[NACT](
                    sm, pp, unimix, k,
                    -policy_weight * inv, Scalar[DT](0.0), gpl, pbase,
                )
    sm.free()
    pp.free()

    var gpl_t = Tensor.alloc(BT * PLOG)
    for i in range(BT * PLOG):
        gpl_t.data[i] = gpl[i]
    var grad_h_t = Tensor.alloc(BT * DIN)
    call_vjp["cpu", BT](
        ph, TensorRefs[PH.ARITY](h_t), gpl_t, TensorRefs[PH.ARITY](grad_h_t), None
    )
    for i in range(BT * DIN):
        grad_h[i] = grad_h_t.data[i]
    return loss
