"""MuZero / EfficientZeroV2 per-batch training diagnostics (remote-logger parity).

The categorical-value sibling of AlphaZero's `append_az_train_diagnostics`
(`deep_agents/alphazero/selfplay_arena.mojo`). Computes the same head-fit /
target-sharpness metrics on the host from the **root** prediction of the last
train batch, and appends them to the ``names``/``values`` lists the selfplay
driver hands to ``logger.log_scalars``.

The one real difference from AlphaZero: the MuZero/EZv2 value head is
**categorical** (``BINS`` two-hot bins in h-space), so ``value_mse`` decodes
``softmax(value_logits)·bins → h⁻¹`` to a raw scalar (matching the planner /
`zero/twohot_targets.mojo` decode) instead of AlphaZero's ``tanh`` squash. The
policy block is identical (soft-CE of the policy head vs the MCTS target).

Inputs (all host buffers; the GPU caller D2H-copies ``pred`` first):
  * ``pred``       — root net output ``[policy_logits(ACT) | value_logits(BINS)]``,
                     shape ``[BATCH·(ACT+BINS)]``.
  * ``policy_tgt`` — MCTS policy target ``[BATCH·ACT]`` (the stored visit/improved
                     policy at the root position).
  * ``value_tgt``  — raw scalar n-step value target ``[BATCH]``.

Metrics appended:
  * ``policy_ce`` / ``policy_entropy``                — policy-head fit + sharpness.
  * ``target_entropy`` / ``target_max_prob``          — MCTS target sharpness.
  * ``policy_ce_minus_target_entropy``                — the policy KL gap.
  * ``value_mse`` / ``value_mean`` / ``value_target_mean`` — value-head fit (raw).

NaN/inf entries are dropped by the logger's own guard, so no clamping here.
"""

from std.math import exp, log, sqrt

from mojo_rl.nn.constants import DT
from ..zero.twohot_targets import mz_inverse_scalar_transform


def append_mz_train_diagnostics[
    ACT: Int, BINS: Int, BATCH: Int,
](
    pred: UnsafePointer[Scalar[DT], MutAnyOrigin],        # [BATCH, ACT+BINS]
    policy_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [BATCH, ACT]
    value_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [BATCH] raw scalar
    v_min: Scalar[DT],
    v_max: Scalar[DT],
    mut names: List[String],
    mut values: List[Float64],
):
    comptime W = ACT + BINS
    var n = Float64(BATCH)
    var step = (
        Float64(v_max - v_min) / Float64(BINS - 1) if BINS > 1 else 0.0
    )

    var ce_sum = 0.0
    var ent_sum = 0.0
    var tent_sum = 0.0
    var tmax_sum = 0.0
    var vmse_sum = 0.0
    var vmean_sum = 0.0
    var vt_sum = 0.0

    for b in range(BATCH):
        var pbase = b * W
        var tbase = b * ACT

        # ── policy: softmax(logits[0:ACT]) → CE vs MCTS target π + entropy ──
        var maxl = Float64(pred[pbase])
        for a in range(1, ACT):
            var v = Float64(pred[pbase + a])
            if v > maxl:
                maxl = v
        var sume = 0.0
        for a in range(ACT):
            sume += exp(Float64(pred[pbase + a]) - maxl)
        var ce = 0.0
        var ent = 0.0
        for a in range(ACT):
            var prob = exp(Float64(pred[pbase + a]) - maxl) / sume
            var t = Float64(policy_tgt[tbase + a])
            if t > 1e-8:
                var p_cl = prob if prob > 1e-12 else 1e-12
                ce -= t * log(p_cl)
            if prob > 1e-8:
                ent -= prob * log(prob)
        ce_sum += ce
        ent_sum += ent

        # ── MCTS target distribution sharpness ──
        var tent = 0.0
        var tmax = 0.0
        for a in range(ACT):
            var tp = Float64(policy_tgt[tbase + a])
            if tp > 1e-8:
                tent -= tp * log(tp)
            if tp > tmax:
                tmax = tp
        tent_sum += tent
        tmax_sum += tmax

        # ── value: decode categorical head (softmax·bins → h-space → h⁻¹) ──
        var vmaxl = Float64(pred[pbase + ACT])
        for i in range(1, BINS):
            var v = Float64(pred[pbase + ACT + i])
            if v > vmaxl:
                vmaxl = v
        var vsum = 0.0
        for i in range(BINS):
            vsum += exp(Float64(pred[pbase + ACT + i]) - vmaxl)
        var hval = 0.0
        for i in range(BINS):
            var p = exp(Float64(pred[pbase + ACT + i]) - vmaxl) / vsum
            hval += p * (Float64(v_min) + step * Float64(i))
        var pv = Float64(mz_inverse_scalar_transform(Scalar[DT](hval)))
        var z = Float64(value_tgt[b])
        vmse_sum += (pv - z) * (pv - z)
        vmean_sum += pv
        vt_sum += z

    var policy_ce = ce_sum / n
    var target_entropy = tent_sum / n
    names.append(String("policy_ce"))
    values.append(policy_ce)
    names.append(String("policy_entropy"))
    values.append(ent_sum / n)
    names.append(String("target_entropy"))
    values.append(target_entropy)
    names.append(String("target_max_prob"))
    values.append(tmax_sum / n)
    names.append(String("policy_ce_minus_target_entropy"))
    values.append(policy_ce - target_entropy)
    names.append(String("value_mse"))
    values.append(vmse_sum / n)
    names.append(String("value_mean"))
    values.append(vmean_sum / n)
    names.append(String("value_target_mean"))
    values.append(vt_sum / n)


def append_value_diagnostics[
    ROW: Int, VOFF: Int, BINS: Int, B: Int,
](
    pred: UnsafePointer[Scalar[DT], MutAnyOrigin],       # [B, ROW]
    value_tgt: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B] raw scalar
    v_min: Scalar[DT],
    v_max: Scalar[DT],
    mut names: List[String],
    mut values: List[Float64],
):
    """Value-head-only diagnostics for heads whose policy is **not** categorical
    (continuous EZv2: squashed-Gaussian policy). Decodes the categorical value
    logits at ``pred[b, VOFF : VOFF+BINS]`` (softmax·bins → h⁻¹) and appends
    ``value_mse`` / ``value_mean`` / ``value_target_mean``. The policy signal for
    these heads is the ``loss_policy`` component emitted by the loss split."""
    var n = Float64(B)
    var step = (
        Float64(v_max - v_min) / Float64(BINS - 1) if BINS > 1 else 0.0
    )
    var vmse_sum = 0.0
    var vmean_sum = 0.0
    var vt_sum = 0.0
    for b in range(B):
        var base = b * ROW + VOFF
        var vmaxl = Float64(pred[base])
        for i in range(1, BINS):
            var v = Float64(pred[base + i])
            if v > vmaxl:
                vmaxl = v
        var vsum = 0.0
        for i in range(BINS):
            vsum += exp(Float64(pred[base + i]) - vmaxl)
        var hval = 0.0
        for i in range(BINS):
            var p = exp(Float64(pred[base + i]) - vmaxl) / vsum
            hval += p * (Float64(v_min) + step * Float64(i))
        var pv = Float64(mz_inverse_scalar_transform(Scalar[DT](hval)))
        var z = Float64(value_tgt[b])
        vmse_sum += (pv - z) * (pv - z)
        vmean_sum += pv
        vt_sum += z
    names.append(String("value_mse")); values.append(vmse_sum / n)
    names.append(String("value_mean")); values.append(vmean_sum / n)
    names.append(String("value_target_mean")); values.append(vt_sum / n)
