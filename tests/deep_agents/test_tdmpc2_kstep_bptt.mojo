"""TD-MPC2 — K-step BPTT gradient correctness (Test 2 of 5).

Goal: Verify the K-step world-model rollout backprop accumulates gradients
across all K steps, not just the last one. We've previously had the bug
class "GPU vjp kernels overwriting (not accumulating) into grad_params"
silently dropping all-but-last backward call (see
project_autodiff_multicall_accumulation.md). This is the analogous CPU
correctness check.

Setup:
  Frozen random encoder (used for s_0 and z_target — same instance).
  We avoid the encoder/target stop-grad ambiguity by testing dynamics
  gradients only — the dynamics is used K times in the rollout chain
  and never as a stop-grad target.

  Random batch of (s_0, a_0..a_{K-1}, s_1..s_K), tiny dims so FD is fast.

K-step BPTT loop, mirroring production tdmpc2.mojo `_wm_bptt_gpu`:
  Forward:
    z_carry = encoder(s_0)                       # cache_0
    for t in 0..K:
      z_target_t = encoder(s_{t+1})              # stop-grad
      z_pred_t   = dynamics(z_carry, a_t)        # cache_dyn_t
      z_carry    = z_pred_t

  Loss = (1/K) * sum_t mean((z_pred_t - z_target_t)^2)

  Backward (reverse order, ACCUMULATES dyn grads across steps):
    grad_z_carry = 0
    for t in K-1..0:
      grad_z_pred = (2/(B*L)) * (z_pred_t - z_target_t) + grad_z_carry
      dynamics.backward(grad_z_pred → grad_za_t, dyn_grads += dW_t)
      grad_z_carry = grad_za_t[:, :LATENT]

Sub-tests:
  2a — FD vs analytic for K=3 on a sample of dynamics params.
       The strongest test: if dynamics.backward overwrites instead of
       accumulating, FD will see contributions from all K steps but
       analytic will only show the last step's contribution.
  2b — Grad-norm at K=3 > grad-norm at K=1 (accumulation present).
       Cheap sanity check; complementary to 2a.
  2c — FD vs analytic for K=1 (single-step) — validates the test
       harness itself isolates the K-step issue from a generic backward
       bug.
"""

from std.math import sqrt
from std.random import seed, random_float64
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.training import Network, NetworkState
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Normal
from mojo_rl.deep_agents.tdmpc2.world_model import WorldModel


comptime OBS = 4
comptime ACT = 2
comptime LATENT = 16
comptime MLP = 32
comptime ENC = 16
comptime SIMPLEX = 4
comptime BATCH = 4
comptime ZA = LATENT + ACT
comptime KMAX = 3

comptime ENC_LR = 9e-5
comptime DYN_LR = 3e-4

comptime WM = WorldModel[
    OBS_DIM=OBS,
    ACTION_DIM=ACT,
    LATENT_DIM=LATENT,
    MLP_DIM=MLP,
    ENC_DIM=ENC,
    NUM_BINS=11,
    NUM_Q=2,
    SIMPLEX_DIM=SIMPLEX,
    ENC_LR=ENC_LR,
    WM_LR=DYN_LR,
]
comptime EncModel = WM.EncModel
comptime DynModel = WM.DynModel
comptime EncOpt = Adam[LR=ENC_LR]
comptime DynOpt = Adam[LR=DYN_LR]


def _expect(cond: Bool, label: String, mut passed: Int, mut total: Int):
    total += 1
    if cond:
        print("PASS:", label)
        passed += 1
    else:
        print("FAIL:", label)


from std.memory import alloc, memset


# Dataset is a simple bag of pointers. `Movable` so we can return it from
# `_build_dataset()`. Heap-backed so pointers stay valid across the FD
# inner loop and across helper calls.
struct Dataset(Movable):
    var obs0: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [B, OBS]
    var acts: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [KMAX, B, ACT]
    var obsK: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [KMAX, B, OBS]

    def __init__(
        out self,
        obs0: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        acts: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        obsK: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ):
        self.obs0 = obs0
        self.acts = acts
        self.obsK = obsK


def _build_dataset() -> Dataset:
    var obs0 = alloc[Scalar[dtype]](BATCH * OBS)
    var acts = alloc[Scalar[dtype]](KMAX * BATCH * ACT)
    var obsK = alloc[Scalar[dtype]](KMAX * BATCH * OBS)
    for i in range(BATCH * OBS):
        obs0[i] = Scalar[dtype](random_float64() * 2.0 - 1.0)
    for i in range(KMAX * BATCH * ACT):
        acts[i] = Scalar[dtype](random_float64() * 2.0 - 1.0)
    for i in range(KMAX * BATCH * OBS):
        obsK[i] = Scalar[dtype](random_float64() * 2.0 - 1.0)
    return Dataset(obs0, acts, obsK)


def _kstep_forward_loss[
    K: Int
](
    ds: Dataset,
    enc: NetworkState[EncModel, EncOpt],
    dyn: NetworkState[DynModel, DynOpt],
) -> Float64:
    """Forward-only K-step loss (no caches). Used for FD."""
    var obs0_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
    ](ds.obs0)
    var z_carry = InlineArray[Scalar[dtype], BATCH * LATENT](
        uninitialized=True
    )
    var z_carry_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
    ](z_carry.unsafe_ptr())
    Network[EncModel, EncOpt].forward[BATCH](
        obs0_t, z_carry_t, enc.params_view(), enc.model_state_view()
    )

    var loss: Float64 = 0.0
    for t in range(K):
        # ── Target ──
        var obs_t1 = LayoutTensor[
            dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
        ](ds.obsK + t * BATCH * OBS)
        var z_target = InlineArray[Scalar[dtype], BATCH * LATENT](
            uninitialized=True
        )
        var z_target_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
        ](z_target.unsafe_ptr())
        Network[EncModel, EncOpt].forward[BATCH](
            obs_t1,
            z_target_t,
            enc.params_view(),
            enc.model_state_view(),
        )
        # ── Build za = (z_carry, a_t) ──
        var act_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ACT), MutAnyOrigin
        ](ds.acts + t * BATCH * ACT)
        var za = InlineArray[Scalar[dtype], BATCH * ZA](uninitialized=True)
        var za_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
        ](za.unsafe_ptr())
        for b in range(BATCH):
            for k in range(LATENT):
                za_t[b, k] = z_carry_t[b, k]
            for k in range(ACT):
                za_t[b, LATENT + k] = act_t[b, k]
        # ── Dynamics ──
        var z_pred = InlineArray[Scalar[dtype], BATCH * LATENT](
            uninitialized=True
        )
        var z_pred_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
        ](z_pred.unsafe_ptr())
        Network[DynModel, DynOpt].forward[BATCH](
            za_t,
            z_pred_t,
            dyn.params_view(),
            dyn.model_state_view(),
        )
        # ── MSE term ──
        var loss_t: Float64 = 0.0
        for b in range(BATCH):
            for k in range(LATENT):
                var diff = (
                    Float64(z_pred_t[b, k][0])
                    - Float64(z_target_t[b, k][0])
                )
                loss_t += diff * diff
        loss_t /= Float64(BATCH * LATENT)
        loss += loss_t
        # ── Carry forward ──
        for b in range(BATCH):
            for k in range(LATENT):
                z_carry_t[b, k] = z_pred_t[b, k]
    return loss / Float64(K)


def _kstep_backward[
    K: Int
](
    ds: Dataset,
    enc: NetworkState[EncModel, EncOpt],
    mut dyn: NetworkState[DynModel, DynOpt],
) -> Float64:
    """K-step rollout + BPTT, accumulating dynamics grads across steps.

    ZEROS dyn.grads at start. After return, dyn.grads holds the analytic
    gradient w.r.t. dyn.params for the K-step loss = (1/K) * sum_t MSE_t.

    Returns the final loss for sanity-check vs forward-only.
    """
    dyn.zero_grads()

    # ── Forward, caching at each step ──
    var obs0_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
    ](ds.obs0)

    # Persistent caches across K steps (we need them all for backward).
    var z_chain = alloc[Scalar[dtype]]((K + 1) * BATCH * LATENT)
    memset(z_chain, 0, (K + 1) * BATCH * LATENT)
    var z_pred_chain = alloc[Scalar[dtype]](K * BATCH * LATENT)
    memset(z_pred_chain, 0, K * BATCH * LATENT)
    var z_target_chain = alloc[Scalar[dtype]](K * BATCH * LATENT)
    memset(z_target_chain, 0, K * BATCH * LATENT)
    var dyn_caches = alloc[Scalar[dtype]](K * BATCH * DynModel.CACHE_SIZE)
    memset(dyn_caches, 0, K * BATCH * DynModel.CACHE_SIZE)

    # Encode s_0 (no cache — encoder is FROZEN here, no encoder backward).
    var z0_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
    ](z_chain)
    Network[EncModel, EncOpt].forward[BATCH](
        obs0_t, z0_t, enc.params_view(), enc.model_state_view()
    )

    var loss: Float64 = 0.0
    for t in range(K):
        # Encoder for target — frozen, no cache.
        var obs_t1_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
        ](ds.obsK + t * BATCH * OBS)
        var z_target_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
        ](z_target_chain + t * BATCH * LATENT)
        Network[EncModel, EncOpt].forward[BATCH](
            obs_t1_t,
            z_target_t,
            enc.params_view(),
            enc.model_state_view(),
        )

        # Build za = (z_carry, a_t).
        var act_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ACT), MutAnyOrigin
        ](ds.acts + t * BATCH * ACT)
        var z_carry_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
        ](z_chain + t * BATCH * LATENT)
        var za = InlineArray[Scalar[dtype], BATCH * ZA](uninitialized=True)
        var za_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
        ](za.unsafe_ptr())
        for b in range(BATCH):
            for k in range(LATENT):
                za_t[b, k] = z_carry_t[b, k]
            for k in range(ACT):
                za_t[b, LATENT + k] = act_t[b, k]

        # Dynamics forward with cache.
        var z_pred_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
        ](z_pred_chain + t * BATCH * LATENT)
        var dyn_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, DynModel.CACHE_SIZE),
            MutAnyOrigin,
        ](dyn_caches + t * BATCH * DynModel.CACHE_SIZE)
        Network[DynModel, DynOpt].forward_with_cache[BATCH](
            za_t,
            z_pred_t,
            dyn.params_view(),
            dyn.model_state_view(),
            dyn_cache_t,
        )

        # MSE accumulation.
        var loss_t: Float64 = 0.0
        for b in range(BATCH):
            for k in range(LATENT):
                var diff = (
                    Float64(z_pred_t[b, k][0])
                    - Float64(z_target_t[b, k][0])
                )
                loss_t += diff * diff
        loss_t /= Float64(BATCH * LATENT)
        loss += loss_t

        # Carry: z_chain[t+1] = z_pred[t]
        var z_next = LayoutTensor[
            dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
        ](z_chain + (t + 1) * BATCH * LATENT)
        for b in range(BATCH):
            for k in range(LATENT):
                z_next[b, k] = z_pred_t[b, k]

    loss /= Float64(K)

    # ── Backward (reverse) ──
    var grad_z_carry = InlineArray[Scalar[dtype], BATCH * LATENT](
        uninitialized=True
    )
    for i in range(BATCH * LATENT):
        grad_z_carry[i] = Scalar[dtype](0)
    var grad_z_carry_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
    ](grad_z_carry.unsafe_ptr())

    var dyn_grads = dyn.grads_view()
    var loss_scale = 1.0 / Float64(K)
    var mse_scale = (
        loss_scale * 2.0 / Float64(BATCH * LATENT)
    )

    # Iterate t from K-1 down to 0.
    for t_rev in range(K):
        var t = K - 1 - t_rev
        var z_pred_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
        ](z_pred_chain + t * BATCH * LATENT)
        var z_target_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
        ](z_target_chain + t * BATCH * LATENT)

        # grad_z_pred = mse_scale * (z_pred - z_target) + carry
        var grad_z_pred = InlineArray[
            Scalar[dtype], BATCH * LATENT
        ](uninitialized=True)
        var grad_z_pred_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
        ](grad_z_pred.unsafe_ptr())
        for b in range(BATCH):
            for k in range(LATENT):
                var diff = (
                    Float64(z_pred_t[b, k][0])
                    - Float64(z_target_t[b, k][0])
                )
                grad_z_pred_t[b, k] = (
                    Scalar[dtype](mse_scale * diff)
                    + grad_z_carry_t[b, k]
                )

        # Dynamics backward — ACCUMULATES into dyn_grads.
        var grad_za = InlineArray[Scalar[dtype], BATCH * ZA](
            uninitialized=True
        )
        var grad_za_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
        ](grad_za.unsafe_ptr())
        var dyn_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, DynModel.CACHE_SIZE),
            MutAnyOrigin,
        ](dyn_caches + t * BATCH * DynModel.CACHE_SIZE)
        Network[DynModel, DynOpt].backward[BATCH](
            grad_z_pred_t,
            grad_za_t,
            dyn.params_view(),
            dyn.model_state_view(),
            dyn_cache_t,
            dyn_grads,
        )

        # Extract grad_z carry from grad_za[:LATENT].
        for b in range(BATCH):
            for k in range(LATENT):
                grad_z_carry_t[b, k] = grad_za_t[b, k]

    z_chain.free()
    z_pred_chain.free()
    z_target_chain.free()
    dyn_caches.free()
    return loss


def _l2_norm(buf: UnsafePointer[Scalar[dtype], MutAnyOrigin], n: Int) -> Float64:
    var s: Float64 = 0.0
    for i in range(n):
        var v = Float64(buf[i])
        s += v * v
    return sqrt(s)


def _abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def main() raises:
    seed(0xBEEF42)
    print("=" * 70)
    print("TD-MPC2 Test 2 — K-step BPTT gradient correctness")
    print("=" * 70)

    var passed = 0
    var total = 0

    var enc = NetworkState[EncModel, EncOpt]()
    enc.initialize[Normal[0.0, 0.02]]()
    var dyn = NetworkState[DynModel, DynOpt]()
    dyn.initialize[Normal[0.0, 0.02]]()

    var ds = _build_dataset()

    # ─── 2a — FD vs analytic for K=3 ────────────────────────────────────
    print()
    print("--- 2a. Finite-diff gradcheck on dynamics, K=3 ---")
    var loss_k3 = _kstep_backward[3](ds, enc, dyn)
    print("    forward+bwd loss (K=3):", loss_k3)

    # Snapshot analytic dyn grads.
    var ana_grads = InlineArray[Float64, DynModel.PARAM_SIZE](
        uninitialized=True
    )
    for i in range(DynModel.PARAM_SIZE):
        ana_grads[i] = Float64(dyn.grads[i])
    var grad_norm_k3 = _l2_norm(dyn.grads, DynModel.PARAM_SIZE)

    # FD on a sample of dynamics params.
    comptime N_SAMPLES = 30
    var eps = 1e-3
    var max_rel: Float64 = 0.0
    var max_abs: Float64 = 0.0
    var checked = 0
    for s in range(N_SAMPLES):
        # Pseudo-random param index that varies across runs of seed(0xBEEF42).
        var idx = (s * 7919 + 1009) % DynModel.PARAM_SIZE
        var orig = Float64(dyn.params[idx])
        dyn.params[idx] = Scalar[dtype](orig + eps)
        var Lp = _kstep_forward_loss[3](ds, enc, dyn)
        dyn.params[idx] = Scalar[dtype](orig - eps)
        var Lm = _kstep_forward_loss[3](ds, enc, dyn)
        dyn.params[idx] = Scalar[dtype](orig)
        var fd = (Lp - Lm) / (2.0 * eps)
        var ana = ana_grads[idx]
        var d = _abs(fd - ana)
        var denom = _abs(fd) + _abs(ana) + 1e-9
        var rel = d / denom
        if d > max_abs:
            max_abs = d
        if rel > max_rel:
            max_rel = rel
        checked += 1

    print("    sampled", checked, "params, max |Δ| =", max_abs,
          "  max rel err =", max_rel)
    _expect(
        max_rel < 0.05,
        "2a — analytic K=3 dyn grad matches FD to 5% relative tolerance",
        passed,
        total,
    )

    # ─── 2b — Grad norm scales with K (accumulation actually happens) ───
    print()
    print("--- 2b. Grad-norm grows with K (accumulation present) ---")
    _ = _kstep_backward[1](ds, enc, dyn)
    var gn_k1 = _l2_norm(dyn.grads, DynModel.PARAM_SIZE)
    _ = _kstep_backward[2](ds, enc, dyn)
    var gn_k2 = _l2_norm(dyn.grads, DynModel.PARAM_SIZE)
    _ = _kstep_backward[3](ds, enc, dyn)
    var gn_k3 = _l2_norm(dyn.grads, DynModel.PARAM_SIZE)
    print("    |grad| K=1:", gn_k1)
    print("    |grad| K=2:", gn_k2)
    print("    |grad| K=3:", gn_k3)
    # If dynamics.backward overwrites instead of accumulating, K=2 and K=3
    # grad norms would equal K=1 (only the last step's grad survives, and
    # the loss is averaged so per-step contribution shrinks). Correct
    # accumulation makes |grad K=3| differ measurably from |grad K=1|.
    _expect(
        _abs(gn_k3 - gn_k1) > 0.1 * gn_k1,
        "2b — |grad K=3| differs >10% from |grad K=1| (accumulation, not overwrite)",
        passed,
        total,
    )

    # ─── 2c — FD vs analytic for K=1 (sanity) ──────────────────────────
    print()
    print("--- 2c. Finite-diff gradcheck on dynamics, K=1 (sanity) ---")
    var loss_k1 = _kstep_backward[1](ds, enc, dyn)
    print("    forward+bwd loss (K=1):", loss_k1)
    var ana_grads_k1 = InlineArray[Float64, DynModel.PARAM_SIZE](
        uninitialized=True
    )
    for i in range(DynModel.PARAM_SIZE):
        ana_grads_k1[i] = Float64(dyn.grads[i])

    var max_rel_k1: Float64 = 0.0
    var checked_k1 = 0
    for s in range(N_SAMPLES):
        var idx = (s * 7919 + 1009) % DynModel.PARAM_SIZE
        var orig = Float64(dyn.params[idx])
        dyn.params[idx] = Scalar[dtype](orig + eps)
        var Lp = _kstep_forward_loss[1](ds, enc, dyn)
        dyn.params[idx] = Scalar[dtype](orig - eps)
        var Lm = _kstep_forward_loss[1](ds, enc, dyn)
        dyn.params[idx] = Scalar[dtype](orig)
        var fd = (Lp - Lm) / (2.0 * eps)
        var ana = ana_grads_k1[idx]
        var denom = _abs(fd) + _abs(ana) + 1e-9
        var rel = _abs(fd - ana) / denom
        if rel > max_rel_k1:
            max_rel_k1 = rel
        checked_k1 += 1
    print("    K=1 max rel err =", max_rel_k1)
    _expect(
        max_rel_k1 < 0.05,
        "2c — analytic K=1 dyn grad matches FD to 5% relative tolerance",
        passed,
        total,
    )

    # ─── Summary ────────────────────────────────────────────────────────
    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
