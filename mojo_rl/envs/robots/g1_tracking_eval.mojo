"""The G1 tracking protocol, natively — no Python, no `references/`.

`tools/g1/bfm_zero_tracking_oracle.py` is the ORACLE this is gated against
(`examples/g1/bfm_zero_eval_tracking.mojo` checks every metric of every
segment it scores, every run). It is not the supported path: the protocol
needs nothing but the store, so the trainer can score itself IN THE LOOP
rather than the run being scored by hand hours after it ended.

What used to be Python and is now here:

  `Protocol.n_segments`  -> `g1_n_segments`, pure store arithmetic. The oracle
                            prefers the released CSV's count when it has one
                            and falls back to exactly this; its own comment
                            records that the two agree on all 40 clips.
  `Episode.first_row`    -> `g1_segment_row`
  `Episode.init_state`   -> `G1RsiTable` already stores qpos verbatim and qvel
                            with the root angular velocity rotated into the
                            body frame, which IS `init_state`. Reused, not
                            reimplemented.
  `compute_metrics`      -> `g1_track_metrics`. `distance` = mean_t ‖q−q*‖₂,
                            `proximity` = the bound-2 / margin-2 ramp, `emd` =
                            `core/assignment.mojo` (the OT plan between equal
                            uniform clouds is a permutation, so the oracle's
                            `linear_sum_assignment` and ours must agree
                            EXACTLY, not approximately).

⚠ ONE definition of the dims and ONE rollout loop, used by both the standalone
eval and the trainer. They used to be two copies with a comment asking the
reader to keep `H`/`L`/`HB`/`D` in sync by hand, and a mismatch only showed up
as a checkpoint that would not load.
"""

from std.math import sqrt, abs

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.core.cont_action import ContAction
from mojo_rl.core.assignment import emd_uniform
from mojo_rl.deep_agents.fb.trainer import FBTrainer
from mojo_rl.deep_agents.fb.obs_norm import ObsNorm
from mojo_rl.envs.robots import UnitreeG1
from mojo_rl.envs.robots.unitree_g1_xml import (
    UnitreeG1Model, UNITREE_G1_STATE_DIM, UNITREE_G1_PRIV_DIM,
)
from mojo_rl.envs.robots.unitree_g1_rsi import G1RsiTable, G1_RSI_NQ, G1_RSI_NV


# ── the G1 network dims, in ONE place ─────────────────────────────────────
# The trainer and the standalone eval must agree on these EXACTLY or a
# checkpoint will not load into the eval — they used to be declared in both
# files with a comment asking the reader to keep them in sync by hand.
#
# BFM-Zero's own tower is `hidden_dim 2048, hidden_layers 6`
# (`released/new_model/config.yaml`), 299 M parameters across F x2, Q_D x2,
# actor, B and D. Measured budget on a 32 GiB card (docs §12.21):
#
#     weights + target + Adam   4.46 GiB
#     activations (rough)       2.62 GiB
#     replay ring, expert and tracking tables   11.22 GiB
#     ----------------------------------------------------
#     ~18.3 GiB of 32
#
# ⚠ The REPLAY RING is the big term, not the model: 9.98 GiB at CAP = 2 M,
# because every transition stores `obs` TWICE (2*527 + 29 + 256 + 1 floats =
# 5360 B). The reference's own CAP of 5.12 M would be 25.6 GiB of ring alone
# and does NOT fit beside this tower — that scale-down is forced by the card.
#
# Smaller alternatives, if compile time or throughput bites:
#     H = 1536, L = 4  -> 111 M params, ~14.4 GiB
#     H = 1024, L = 3  ->  40 M params, ~12.7 GiB   (what runs 1-5 used)
comptime G1_D: Int = 256          # z_dim
comptime G1_H: Int = 2048         # hidden_dim
comptime G1_L: Int = 6            # hidden_layers
comptime G1_HB: Int = 256         # backward map hidden
comptime G1_HD: Int = 1024        # discriminator hidden


# ── the protocol's constants (oracle lines 95-97) ─────────────────────────
comptime G1_SEG_ROWS: Int = 499      # ceil((300 - 1) / 30 / 0.02)
comptime G1_SEG_STRIDE: Int = 500    # 10 s at 50 Hz
comptime G1_PROX_BOUND: Float64 = 2.0
comptime G1_PROX_MARGIN: Float64 = 2.0


def g1_n_segments(ep_len: Int) -> Int:
    """Every 500-row start whose 499 rows fit inside the clip."""
    if ep_len < G1_SEG_ROWS:
        return 0
    return (ep_len - G1_SEG_ROWS) // G1_SEG_STRIDE + 1


def g1_segment_row(ep_offset: Int, seg: Int) -> Int:
    """The store row this segment starts at."""
    return ep_offset + seg * G1_SEG_STRIDE


@fieldwise_init
struct G1TrackScore(Copyable):
    """One segment's numbers, or a mean over several."""

    var distance: Float64
    var emd: Float64
    var proximity: Float64
    var n: Int


def g1_track_metrics[
    ACT: Int
](ref ach: List[Float64], ref tgt: List[Float64], T: Int) raises -> G1TrackScore:
    """`compute_metrics`, formula for formula, on `(T, ACT)` joint angles."""
    var sum_d = 0.0
    var sum_p = 0.0
    for j in range(T):
        var s = 0.0
        for k in range(ACT):
            var d = ach[j * ACT + k] - tgt[j * ACT + k]
            s += d * d
        var dist = sqrt(s)
        sum_d += dist
        # `inb + ((bound + margin - dist)/margin)·(~inb)·(~outb)`
        if dist <= G1_PROX_BOUND:
            sum_p += 1.0
        elif dist <= G1_PROX_BOUND + G1_PROX_MARGIN:
            sum_p += (G1_PROX_BOUND + G1_PROX_MARGIN - dist) / G1_PROX_MARGIN
    var tf = Float64(T)
    return G1TrackScore(
        sum_d / tf, emd_uniform(ach, tgt, T, ACT), sum_p / tf, 1
    )


def g1_project_z[Dz: Int](mut z: Tensor, row: Int):
    """Onto the radius-sqrt(d) sphere."""
    var s = 0.0
    for k in range(Dz):
        var v = Float64(z.data[row * Dz + k])
        s += v * v
    var scale = sqrt(Float64(Dz)) / sqrt(s + 1e-12)
    for k in range(Dz):
        z.data[row * Dz + k] = Scalar[DT](
            Float64(z.data[row * Dz + k]) * scale
        )


def g1_score_segment[
    FNET: Module, BNET: Module, ANET: Module,
    OBS: Int, ACT: Int, D: Int, BATCH: Int,
](
    mut t: FBTrainer[FNET, BNET, ANET, OBS, ACT, D, BATCH, "cpu"],
    mut env: UnitreeG1[DType.float64],
    ref rsi: G1RsiTable,
    ref st: List[Scalar[DType.float32]],
    ref pv: List[Scalar[DType.float32]],
    ref qpos_col: List[Scalar[DType.float32]],
    ref norm: Optional[ObsNorm[OBS]],
    r0: Int,
    mut ach: List[Float64],
    mut tgt: List[Float64],
    mut b_in: Tensor,
    mut b_out: Tensor,
    mut z_seg: Tensor,
    mut obs_t: Tensor,
    mut z1: Tensor,
    mut act_out: Tensor,
) raises -> G1TrackScore:
    """`tracking_inference` for ONE segment starting at store row `r0`.

    `z_t = project(B(row t+1))` — a SINGLE row, not the mean of eight the
    training rollouts use — then reset to row 0 and `T-1` mean-action steps.
    """
    comptime T = G1_SEG_ROWS
    comptime NQ = UnitreeG1Model.NQ
    comptime NV = UnitreeG1Model.NV

    for j in range(T):
        for k in range(UNITREE_G1_STATE_DIM):
            b_in.data[j * OBS + k] = Scalar[DT](
                st[(r0 + j) * UNITREE_G1_STATE_DIM + k]
            )
        for k in range(UNITREE_G1_PRIV_DIM):
            b_in.data[j * OBS + UNITREE_G1_STATE_DIM + k] = Scalar[DT](
                pv[(r0 + j) * UNITREE_G1_PRIV_DIM + k]
            )
    if norm:
        norm.value().apply_rows(b_in, T)
    t.backward_embed[T](b_in, b_out)
    for j in range(T - 1):
        for k in range(D):
            z_seg.data[j * D + k] = b_out.data[(j + 1) * D + k]
        g1_project_z[D](z_seg, j)

    # reset: the RSI row IS `init_state()` (qvel already body-frame)
    var qp = List[Float64](length=NQ, fill=0.0)
    var qv = List[Float64](length=NV, fill=0.0)
    var base = r0 * (G1_RSI_NQ + G1_RSI_NV)
    for i in range(NQ):
        qp[i] = Float64(rsi.rows.data[base + i])
    for i in range(NV):
        qv[i] = Float64(rsi.rows.data[base + G1_RSI_NQ + i])
    env.set_state(qp, qv)

    for i in range(NQ):
        qp[i] = Float64(env.d.qpos.data[i])
    var nrec = 0
    # ⚠ TWO RECORD SITES, AND THEY MUST STAY IN STEP — the `nrec != T` raise
    # below is the guard that caught exactly that bug once already.
    for k in range(ACT):
        ach[nrec * ACT + k] = qp[7 + k]
    nrec += 1
    for step in range(T - 1):
        var o = env.get_obs_list()
        for k in range(OBS):
            obs_t.data[k] = Scalar[DT](Float64(o[k]))
        if norm:
            norm.value().apply_row(obs_t)
        for k in range(D):
            z1.data[k] = z_seg.data[step * D + k]
        t.act[1](obs_t, z1, act_out)
        # the env's action dim is the model's, not the caller's `ACT` — they
        # are the same number, but only one of them is what `step` accepts
        var a = ContAction[UnitreeG1Model.ACTION_DIM]()
        for k in range(UnitreeG1Model.ACTION_DIM):
            var v = Float64(act_out.data[k])
            if v > 1.0:
                v = 1.0
            elif v < -1.0:
                v = -1.0
            a.data[k] = v
        _ = env.step(a)
        for i in range(NQ):
            qp[i] = Float64(env.d.qpos.data[i])
        for k in range(ACT):
            ach[nrec * ACT + k] = qp[7 + k]
        nrec += 1
    if nrec != T:
        raise Error(
            "achieved rows " + String(nrec) + " != T " + String(T)
            + " — a record site lost its accumulation"
        )

    for j in range(T):
        for k in range(ACT):
            tgt[j * ACT + k] = Float64(qpos_col[(r0 + j) * NQ + 7 + k])
    return g1_track_metrics[ACT](ach, tgt, T)
