"""TD-MPC2 — encoder gradient reach (Test 3 of 5).

Goal: Verify each loss head (consistency / reward / Q / termination) sends
gradient back to the encoder, AND that the combined gradient is the sum
of the per-head gradients (linearity of backprop).

Why this catches bugs:
  - If reward.backward, q.backward, or term.backward has a broken path,
    that loss never reshapes the encoder. The encoder is then driven by
    only consistency loss, whose trivial-collapse minimum (constant z)
    has no counterforce.
  - Linearity check catches sign-flip / scaling bugs that would still
    pass per-head gradient-norm > 0 but produce non-additive grads.

Trivial pseudo-loss L = sum(head_output) (i.e. grad_output = 1 elementwise).
This isolates gradient FLOW from loss computation correctness — which is
what we want to test here, since Test 1 already covered consistency loss
math and Test 2 covered the K-step path.

Sub-tests (all run on a frozen random encoder + random heads, K=1 setup):
  3a — consistency-only loss reaches encoder
  3b — reward-only loss reaches encoder
  3c — Q1-only loss reaches encoder
  3d — termination-only loss reaches encoder
  3e — sum(per-head encoder grads) ≈ combined-loss encoder grad (linearity)
"""

from std.math import sqrt
from std.random import seed, random_float64
from std.memory import alloc, memset

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
comptime BINS = 11
comptime BATCH = 4
comptime ZA = LATENT + ACT

comptime ENC_LR = 9e-5
comptime WM_LR = 3e-4

comptime WM = WorldModel[
    OBS_DIM=OBS,
    ACTION_DIM=ACT,
    LATENT_DIM=LATENT,
    MLP_DIM=MLP,
    ENC_DIM=ENC,
    NUM_BINS=BINS,
    NUM_Q=2,
    SIMPLEX_DIM=SIMPLEX,
    ENC_LR=ENC_LR,
    WM_LR=WM_LR,
]
comptime EncModel = WM.EncModel
comptime DynModel = WM.DynModel
comptime RewModel = WM.RewModel
comptime QModel = WM.QModel
comptime TermModel = WM.TermModel
comptime EncOpt = Adam[LR=ENC_LR]
comptime WMOpt = Adam[LR=WM_LR]


def _expect(cond: Bool, label: String, mut passed: Int, mut total: Int):
    total += 1
    if cond:
        print("PASS:", label)
        passed += 1
    else:
        print("FAIL:", label)


def _l2_norm(buf: UnsafePointer[Scalar[dtype], MutAnyOrigin], n: Int) -> Float64:
    var s: Float64 = 0.0
    for i in range(n):
        var v = Float64(buf[i])
        s += v * v
    return sqrt(s)


def _abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


struct GradReachResult(Movable):
    """Result of running backward through one or more loss heads.

    grad_norm is the L2 norm of encoder grads.
    snap is a heap-allocated copy of encoder grads (caller must free).
    """
    var grad_norm: Float64
    var snap: UnsafePointer[Scalar[dtype], MutAnyOrigin]

    def __init__(
        out self,
        grad_norm: Float64,
        snap: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ):
        self.grad_norm = grad_norm
        self.snap = snap


# `which` selects which loss heads contribute encoder gradient:
#   0 = consistency only
#   1 = reward only
#   2 = q1 only
#   3 = termination only
#   4 = all four combined
# After return, encoder gradient = backprop through selected head(s) only.
def _backward_through(
    which: Int,
    obs0_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
    ],
    obs1_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
    ],
    act_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACT), MutAnyOrigin
    ],
    mut enc: NetworkState[EncModel, EncOpt],
    mut dyn: NetworkState[DynModel, WMOpt],
    mut rew: NetworkState[RewModel, WMOpt],
    mut q1: NetworkState[QModel, WMOpt],
    mut term: NetworkState[TermModel, WMOpt],
) -> GradReachResult:
    # Encode s_0 with cache.
    var z0 = alloc[Scalar[dtype]](BATCH * LATENT)
    memset(z0, 0, BATCH * LATENT)
    var z0_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
    ](z0)
    var enc_cache = alloc[Scalar[dtype]](BATCH * EncModel.CACHE_SIZE)
    memset(enc_cache, 0, BATCH * EncModel.CACHE_SIZE)
    var enc_cache_t = LayoutTensor[
        dtype,
        Layout.row_major(BATCH, EncModel.CACHE_SIZE),
        MutAnyOrigin,
    ](enc_cache)
    Network[EncModel, EncOpt].forward_with_cache[BATCH](
        obs0_t,
        z0_t,
        enc.params_view(),
        enc.model_state_view(),
        enc_cache_t,
    )

    # Build za = (z_0, a)
    var za = alloc[Scalar[dtype]](BATCH * ZA)
    var za_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
    ](za)
    for b in range(BATCH):
        for k in range(LATENT):
            za_t[b, k] = z0_t[b, k]
        for k in range(ACT):
            za_t[b, LATENT + k] = act_t[b, k]

    # Accumulator for grad_z (carry into encoder).
    var grad_z = alloc[Scalar[dtype]](BATCH * LATENT)
    memset(grad_z, 0, BATCH * LATENT)
    var grad_z_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
    ](grad_z)

    # ---- Consistency path ----
    if which == 0 or which == 4:
        var z_target = alloc[Scalar[dtype]](BATCH * LATENT)
        var z_target_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
        ](z_target)
        Network[EncModel, EncOpt].forward[BATCH](
            obs1_t,
            z_target_t,
            enc.params_view(),
            enc.model_state_view(),
        )
        var z_pred = alloc[Scalar[dtype]](BATCH * LATENT)
        var z_pred_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
        ](z_pred)
        var dyn_cache = alloc[Scalar[dtype]](BATCH * DynModel.CACHE_SIZE)
        var dyn_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, DynModel.CACHE_SIZE),
            MutAnyOrigin,
        ](dyn_cache)
        Network[DynModel, WMOpt].forward_with_cache[BATCH](
            za_t,
            z_pred_t,
            dyn.params_view(),
            dyn.model_state_view(),
            dyn_cache_t,
        )
        var grad_z_pred = alloc[Scalar[dtype]](BATCH * LATENT)
        var grad_z_pred_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
        ](grad_z_pred)
        var sc = 2.0 / Float64(BATCH * LATENT)
        for b in range(BATCH):
            for k in range(LATENT):
                var diff = (
                    Float64(z_pred_t[b, k][0])
                    - Float64(z_target_t[b, k][0])
                )
                grad_z_pred_t[b, k] = Scalar[dtype](sc * diff)
        dyn.zero_grads()
        var grad_za = alloc[Scalar[dtype]](BATCH * ZA)
        var grad_za_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
        ](grad_za)
        var dyn_grads_v = dyn.grads_view()
        Network[DynModel, WMOpt].backward[BATCH](
            grad_z_pred_t,
            grad_za_t,
            dyn.params_view(),
            dyn.model_state_view(),
            dyn_cache_t,
            dyn_grads_v,
        )
        for b in range(BATCH):
            for k in range(LATENT):
                grad_z_t[b, k] = grad_z_t[b, k] + grad_za_t[b, k]
        grad_z_pred.free()
        grad_za.free()
        dyn_cache.free()
        z_pred.free()
        z_target.free()

    # ---- Reward path ----
    if which == 1 or which == 4:
        var rew_logits = alloc[Scalar[dtype]](BATCH * BINS)
        var rew_logits_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, BINS), MutAnyOrigin
        ](rew_logits)
        var rew_cache = alloc[Scalar[dtype]](BATCH * RewModel.CACHE_SIZE)
        var rew_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, RewModel.CACHE_SIZE),
            MutAnyOrigin,
        ](rew_cache)
        Network[RewModel, WMOpt].forward_with_cache[BATCH](
            za_t,
            rew_logits_t,
            rew.params_view(),
            rew.model_state_view(),
            rew_cache_t,
        )
        var grad_logits = alloc[Scalar[dtype]](BATCH * BINS)
        var grad_logits_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, BINS), MutAnyOrigin
        ](grad_logits)
        for i in range(BATCH * BINS):
            grad_logits[i] = Scalar[dtype](1.0 / Float64(BATCH * BINS))
        var grad_za = alloc[Scalar[dtype]](BATCH * ZA)
        var grad_za_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
        ](grad_za)
        rew.zero_grads()
        var rew_grads_v = rew.grads_view()
        Network[RewModel, WMOpt].backward[BATCH](
            grad_logits_t,
            grad_za_t,
            rew.params_view(),
            rew.model_state_view(),
            rew_cache_t,
            rew_grads_v,
        )
        for b in range(BATCH):
            for k in range(LATENT):
                grad_z_t[b, k] = grad_z_t[b, k] + grad_za_t[b, k]
        rew_logits.free()
        rew_cache.free()
        grad_logits.free()
        grad_za.free()

    # ---- Q1 path ----
    if which == 2 or which == 4:
        var q_logits = alloc[Scalar[dtype]](BATCH * BINS)
        var q_logits_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, BINS), MutAnyOrigin
        ](q_logits)
        var q_cache = alloc[Scalar[dtype]](BATCH * QModel.CACHE_SIZE)
        var q_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, QModel.CACHE_SIZE),
            MutAnyOrigin,
        ](q_cache)
        Network[QModel, WMOpt].forward_with_cache[BATCH](
            za_t,
            q_logits_t,
            q1.params_view(),
            q1.model_state_view(),
            q_cache_t,
        )
        var grad_logits = alloc[Scalar[dtype]](BATCH * BINS)
        var grad_logits_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, BINS), MutAnyOrigin
        ](grad_logits)
        for i in range(BATCH * BINS):
            grad_logits[i] = Scalar[dtype](1.0 / Float64(BATCH * BINS))
        var grad_za = alloc[Scalar[dtype]](BATCH * ZA)
        var grad_za_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
        ](grad_za)
        q1.zero_grads()
        var q1_grads_v = q1.grads_view()
        Network[QModel, WMOpt].backward[BATCH](
            grad_logits_t,
            grad_za_t,
            q1.params_view(),
            q1.model_state_view(),
            q_cache_t,
            q1_grads_v,
        )
        for b in range(BATCH):
            for k in range(LATENT):
                grad_z_t[b, k] = grad_z_t[b, k] + grad_za_t[b, k]
        q_logits.free()
        q_cache.free()
        grad_logits.free()
        grad_za.free()

    # ---- Termination path (term takes z only) ----
    if which == 3 or which == 4:
        var term_out = alloc[Scalar[dtype]](BATCH * 1)
        var term_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](term_out)
        var term_cache = alloc[Scalar[dtype]](BATCH * TermModel.CACHE_SIZE)
        var term_cache_t = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, TermModel.CACHE_SIZE),
            MutAnyOrigin,
        ](term_cache)
        Network[TermModel, WMOpt].forward_with_cache[BATCH](
            z0_t,
            term_out_t,
            term.params_view(),
            term.model_state_view(),
            term_cache_t,
        )
        var grad_term = alloc[Scalar[dtype]](BATCH * 1)
        var grad_term_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
        ](grad_term)
        for i in range(BATCH):
            grad_term[i] = Scalar[dtype](1.0 / Float64(BATCH))
        var grad_z_local = alloc[Scalar[dtype]](BATCH * LATENT)
        var grad_z_local_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
        ](grad_z_local)
        term.zero_grads()
        var term_grads_v = term.grads_view()
        Network[TermModel, WMOpt].backward[BATCH](
            grad_term_t,
            grad_z_local_t,
            term.params_view(),
            term.model_state_view(),
            term_cache_t,
            term_grads_v,
        )
        for b in range(BATCH):
            for k in range(LATENT):
                grad_z_t[b, k] = grad_z_t[b, k] + grad_z_local_t[b, k]
        term_out.free()
        term_cache.free()
        grad_term.free()
        grad_z_local.free()

    # ---- Encoder backward ----
    enc.zero_grads()
    var grad_obs = alloc[Scalar[dtype]](BATCH * OBS)
    var grad_obs_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
    ](grad_obs)
    var enc_grads_v = enc.grads_view()
    Network[EncModel, EncOpt].backward[BATCH](
        grad_z_t,
        grad_obs_t,
        enc.params_view(),
        enc.model_state_view(),
        enc_cache_t,
        enc_grads_v,
    )
    var enc_gn = _l2_norm(enc.grads, EncModel.PARAM_SIZE)

    var snap = alloc[Scalar[dtype]](EncModel.PARAM_SIZE)
    for i in range(EncModel.PARAM_SIZE):
        snap[i] = enc.grads[i]

    z0.free()
    enc_cache.free()
    za.free()
    grad_z.free()
    grad_obs.free()
    return GradReachResult(grad_norm=enc_gn, snap=snap)


def main() raises:
    seed(0xCAFE99)
    print("=" * 70)
    print("TD-MPC2 Test 3 — Encoder gradient reach")
    print("=" * 70)

    var passed = 0
    var total = 0

    # ─── Build all networks ─────────────────────────────────────────────
    var enc = NetworkState[EncModel, EncOpt]()
    enc.initialize[Normal[0.0, 0.02]]()
    var dyn = NetworkState[DynModel, WMOpt]()
    dyn.initialize[Normal[0.0, 0.02]]()
    var rew = NetworkState[RewModel, WMOpt]()
    rew.initialize[Normal[0.0, 0.02]]()
    var q1 = NetworkState[QModel, WMOpt]()
    q1.initialize[Normal[0.0, 0.02]]()
    var term = NetworkState[TermModel, WMOpt]()
    term.initialize[Normal[0.0, 0.02]]()

    # ─── Build batch ────────────────────────────────────────────────────
    var obs0 = alloc[Scalar[dtype]](BATCH * OBS)
    for i in range(BATCH * OBS):
        obs0[i] = Scalar[dtype](random_float64() * 2.0 - 1.0)
    var obs1 = alloc[Scalar[dtype]](BATCH * OBS)
    for i in range(BATCH * OBS):
        obs1[i] = Scalar[dtype](random_float64() * 2.0 - 1.0)
    var act = alloc[Scalar[dtype]](BATCH * ACT)
    for i in range(BATCH * ACT):
        act[i] = Scalar[dtype](random_float64() * 2.0 - 1.0)

    var obs0_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
    ](obs0)
    var obs1_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
    ](obs1)
    var act_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, ACT), MutAnyOrigin
    ](act)


    # ─── 3a–3d: each loss alone reaches encoder ────────────────────────
    print()
    print("--- Per-head encoder gradient norms (loss in isolation) ---")
    var r_cons = _backward_through(
        0, obs0_t, obs1_t, act_t, enc, dyn, rew, q1, term
    )
    var r_rew = _backward_through(
        1, obs0_t, obs1_t, act_t, enc, dyn, rew, q1, term
    )
    var r_q = _backward_through(
        2, obs0_t, obs1_t, act_t, enc, dyn, rew, q1, term
    )
    var r_term = _backward_through(
        3, obs0_t, obs1_t, act_t, enc, dyn, rew, q1, term
    )
    var r_all = _backward_through(
        4, obs0_t, obs1_t, act_t, enc, dyn, rew, q1, term
    )
    print("    consistency-only |∇enc| =", r_cons.grad_norm)
    print("    reward-only      |∇enc| =", r_rew.grad_norm)
    print("    Q1-only          |∇enc| =", r_q.grad_norm)
    print("    termination-only |∇enc| =", r_term.grad_norm)
    print("    all combined     |∇enc| =", r_all.grad_norm)

    _expect(r_cons.grad_norm > 0.0, "3a — consistency loss reaches encoder", passed, total)
    _expect(r_rew.grad_norm > 0.0, "3b — reward loss reaches encoder", passed, total)
    _expect(r_q.grad_norm > 0.0, "3c — Q1 loss reaches encoder", passed, total)
    _expect(r_term.grad_norm > 0.0, "3d — termination loss reaches encoder", passed, total)

    # ─── 3e: linearity — sum of per-head encoder grads ≈ combined ──────
    var sum_buf = alloc[Scalar[dtype]](EncModel.PARAM_SIZE)
    for i in range(EncModel.PARAM_SIZE):
        sum_buf[i] = (
            r_cons.snap[i] + r_rew.snap[i] + r_q.snap[i] + r_term.snap[i]
        )
    # Compare sum_buf to r_all.snap element-wise (relative error).
    var max_rel: Float64 = 0.0
    var max_abs: Float64 = 0.0
    for i in range(EncModel.PARAM_SIZE):
        var s = Float64(sum_buf[i])
        var a = Float64(r_all.snap[i])
        var d = _abs(s - a)
        var denom = _abs(s) + _abs(a) + 1e-9
        var rel = d / denom
        if d > max_abs:
            max_abs = d
        if rel > max_rel:
            max_rel = rel
    print("    linearity: max |Δ| =", max_abs, "  max rel err =", max_rel)
    _expect(
        max_rel < 1e-3,
        "3e — sum(per-head ∇enc) ≈ combined ∇enc (backprop linearity)",
        passed,
        total,
    )

    r_cons.snap.free()
    r_rew.snap.free()
    r_q.snap.free()
    r_term.snap.free()
    r_all.snap.free()
    sum_buf.free()
    obs0.free()
    obs1.free()
    act.free()

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
