"""FD gradcheck for the canonical squashed-Gaussian.

Loss: L = sum_b sum_j (action[b,j] - target_action[b,j])^2
       + sum_b (log_prob[b] - target_lp[b])^2

  dL/d_action[b,j] = 2*(action[b,j] - target_action[b,j])
  dL/d_log_prob[b] = 2*(log_prob[b] - target_lp[b])

Pass when max_rel_err on grad_actor_output (mu + log_std) < 1e-3, except
on samples where log_std is at the clamp boundary (gradient = 0 by design).
"""

from std.math import abs as fabs
from std.memory import alloc
from layout import TileTensor, row_major

from mojo_rl.nn2_v2.spike_squashed_gaussian import (
    DT,
    squashed_gaussian_forward,
    squashed_gaussian_backward,
    LOG_STD_MIN,
    LOG_STD_MAX,
)


comptime BATCH = 4
comptime ACT = 3


def _loss(
    ao: UnsafePointer[Scalar[DT], MutAnyOrigin],
    z: UnsafePointer[Scalar[DT], MutAnyOrigin],
    action_scale: Scalar[DT],
    tgt_a: UnsafePointer[Scalar[DT], MutAnyOrigin],
    tgt_lp: UnsafePointer[Scalar[DT], MutAnyOrigin],
    a_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
    lp_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
) raises -> Scalar[DT]:
    var ao_t = TileTensor(ao, row_major[BATCH, 2 * ACT]())
    var z_t  = TileTensor(z,  row_major[BATCH, ACT]())
    var a_t  = TileTensor(a_p, row_major[BATCH, ACT]())
    var lp_t = TileTensor(lp_p, row_major[BATCH]())
    squashed_gaussian_forward[ACT, BATCH](ao_t, z_t, action_scale, a_t, lp_t)
    var L: Scalar[DT] = 0.0
    for b in range(BATCH):
        for j in range(ACT):
            var d = a_p[b * ACT + j] - tgt_a[b * ACT + j]
            L += d * d
        var dl = lp_p[b] - tgt_lp[b]
        L += dl * dl
    return L


def main() raises:
    # ── Allocate. ──
    var n_ao = BATCH * 2 * ACT
    var n_z  = BATCH * ACT
    var n_a  = BATCH * ACT
    var n_lp = BATCH

    var ao: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](n_ao)
    var z:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](n_z)
    var tgt_a: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](n_a)
    var tgt_lp: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](n_lp)
    var a_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](n_a)
    var lp_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](n_lp)

    # ── Inputs. Deterministic small values, all log_stds inside clamp window. ──
    var state: UInt64 = UInt64(0x42)
    for k in range(n_ao):
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var r = Scalar[DT]((Int(state >> 32) & 0xFFFF)) / Scalar[DT](65535.0)
        ao[k] = (r - Scalar[DT](0.5)) * Scalar[DT](1.0)  # mu in [-0.5, 0.5]
    # Force log_stds (last ACT cols per row) into [-2, 0.5] window (well inside [-5, 2]).
    for b in range(BATCH):
        for j in range(ACT):
            ao[b * 2 * ACT + ACT + j] = (
                ao[b * 2 * ACT + ACT + j] * Scalar[DT](0.6) - Scalar[DT](0.3)
            )
    for k in range(n_z):
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var r = Scalar[DT]((Int(state >> 32) & 0xFFFF)) / Scalar[DT](65535.0)
        z[k] = (r - Scalar[DT](0.5)) * Scalar[DT](2.0)  # z in [-1, 1]
    for k in range(n_a):
        tgt_a[k] = 0.1
    for k in range(n_lp):
        tgt_lp[k] = 0.0

    var action_scale: Scalar[DT] = 2.0

    # ── Analytical grad. ──
    var L_baseline = _loss(ao, z, action_scale, tgt_a, tgt_lp, a_p, lp_p)
    print("L_baseline =", L_baseline)

    var grad_a:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](n_a)
    var grad_lp: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](n_lp)
    for b in range(BATCH):
        for j in range(ACT):
            grad_a[b * ACT + j] = (
                Scalar[DT](2.0) * (a_p[b * ACT + j] - tgt_a[b * ACT + j])
            )
        grad_lp[b] = Scalar[DT](2.0) * (lp_p[b] - tgt_lp[b])

    var grad_ao: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](n_ao)
    var ao_t   = TileTensor(ao, row_major[BATCH, 2 * ACT]())
    var z_t    = TileTensor(z,  row_major[BATCH, ACT]())
    var ga_t   = TileTensor(grad_a, row_major[BATCH, ACT]())
    var glp_t  = TileTensor(grad_lp, row_major[BATCH]())
    var gao_t  = TileTensor(grad_ao, row_major[BATCH, 2 * ACT]())
    squashed_gaussian_backward[ACT, BATCH](
        ao_t, z_t, ga_t, glp_t, action_scale, gao_t,
    )

    # ── FD grad each entry of `ao`. ──
    comptime eps: Scalar[DT] = 1e-3
    var num_ao: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](n_ao)
    for k in range(n_ao):
        var orig = ao[k]
        ao[k] = orig + eps
        var Lp = _loss(ao, z, action_scale, tgt_a, tgt_lp, a_p, lp_p)
        ao[k] = orig - eps
        var Lm = _loss(ao, z, action_scale, tgt_a, tgt_lp, a_p, lp_p)
        ao[k] = orig
        num_ao[k] = (Lp - Lm) / (Scalar[DT](2.0) * eps)

    var max_rel: Scalar[DT] = 0.0
    var max_k = 0
    for k in range(n_ao):
        var a_g = grad_ao[k]
        var n_g = num_ao[k]
        var denom: Scalar[DT] = fabs(a_g) + fabs(n_g) + Scalar[DT](1e-8)
        var rel = fabs(a_g - n_g) / denom
        if rel > max_rel:
            max_rel = rel
            max_k = k

    print("max_rel_err grad_actor_output =", max_rel, "at idx", max_k)

    comptime tol: Scalar[DT] = 1e-3
    if max_rel < tol:
        print("PASS — canonical squashed-Gaussian gradcheck within tol", tol)
    else:
        print("FAIL — canonical squashed-Gaussian gradcheck out of tol", tol)
        raise Error("squashed-Gaussian gradcheck failed")

    ao.free()
    z.free()
    tgt_a.free()
    tgt_lp.free()
    a_p.free()
    lp_p.free()
    grad_a.free()
    grad_lp.free()
    grad_ao.free()
    num_ao.free()
