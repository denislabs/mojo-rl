"""TD-MPC2 — consistency loss in isolation (Test 1 of 5).

Goal: drive the encoder + dynamics with ONLY the consistency objective
and verify the chain learns a non-trivial representation.

Why this catches bugs:
  - Production TD-MPC2 mixes consistency_loss with reward_loss, value_loss,
    termination_loss, policy_loss. If consistency_loss has a degenerate
    fixed point at "encoder + dynamics output the same constant" (loss=0),
    the system can settle there whenever the other losses temporarily
    vanish or oppose each other. We've seen consistency_loss bouncing
    between ~0 and ~0.004 in production training, suggesting exactly this.
  - This test isolates the gradient signal so we can see whether:
      (a) loss decreases monotonically (no sign / scaling bug),
      (b) z_pred and z_target stay non-trivial (no collapse to constants),
      (c) gradient actually flows into both encoder and dynamics.

Setup:
  Tiny synthetic dataset of 64 transitions (s_t, a_t, s_{t+1}) with a
  fixed linear ground-truth dynamics in observation space. The world
  model architecture is the real TD-MPC2 stack (NormedLinear + Linear +
  LayerNorm + SimNorm), just at small dimensions so it runs fast on CPU.

Pass criteria:
  Test 1a — final loss < 0.5 * initial loss (loss decreases significantly)
  Test 1b — std(z_pred) > 0.05 throughout (encoder output not constant)
  Test 1c — std(z_target) > 0.05 throughout (encoder output not constant)
  Test 1d — encoder grad-norm > 0 every step (gradient reaches encoder)
  Test 1e — dynamics grad-norm > 0 every step (gradient reaches dynamics)
"""

from std.math import sqrt
from std.random import seed, random_float64
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.training import Network, NetworkState
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Normal
from mojo_rl.deep_agents.tdmpc2.world_model import WorldModel


# Small dimensions so we can run the full TD-MPC2 stack in a unit test.
# LATENT must be divisible by SIMPLEX_DIM. We pick (16, 4) → 4 groups of 4.
comptime OBS = 4
comptime ACT = 2
comptime LATENT = 16
comptime MLP = 32
comptime ENC = 16
comptime SIMPLEX = 4
comptime BATCH = 64
comptime ZA = LATENT + ACT
comptime NUM_STEPS = 1000
comptime LOG_EVERY = 100

comptime ENC_LR = 9e-5
comptime DYN_LR = 3e-4

# Re-use the WorldModel sub-types so this test exercises the EXACT same
# architecture and initializer path as production.
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


def _build_dataset(
    mut obs_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
    ],
    mut act_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, ACT), MutAnyOrigin
    ],
    mut obs_next_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
    ],
):
    """Build BATCH transitions with a smooth fixed dynamics: s' = s + 0.1*W*a.

    The actual dynamics doesn't matter — we just need the (s, a, s') tuples to
    have non-trivial structure (correlations) so a learned encoder has
    something to recover.
    """
    # Fixed projection matrix W ∈ R^{OBS x ACT} for s' = s + 0.1 * W @ a.
    var W = InlineArray[Float64, OBS * ACT](uninitialized=True)
    for i in range(OBS * ACT):
        W[i] = random_float64() * 2.0 - 1.0

    for b in range(BATCH):
        for d in range(OBS):
            obs_t[b, d] = Scalar[dtype](random_float64() * 2.0 - 1.0)
        for d in range(ACT):
            act_t[b, d] = Scalar[dtype](random_float64() * 2.0 - 1.0)
        for d in range(OBS):
            var s = Float64(obs_t[b, d][0])
            var s_next = s
            for k in range(ACT):
                s_next += (
                    0.1 * W[d * ACT + k] * Float64(act_t[b, k][0])
                )
            obs_next_t[b, d] = Scalar[dtype](s_next)


def _l2_norm(buf: UnsafePointer[Scalar[dtype], MutAnyOrigin], n: Int) -> Float64:
    var s: Float64 = 0.0
    for i in range(n):
        var v = Float64(buf[i])
        s += v * v
    return sqrt(s)


def _batch_std(
    buf: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    batch: Int,
    feat: Int,
) -> Float64:
    """Average per-feature batch std (std across batch, then mean over features).
    """
    var sum_std: Float64 = 0.0
    for k in range(feat):
        var mean: Float64 = 0.0
        var sumsq: Float64 = 0.0
        for b in range(batch):
            var v = Float64(buf[b * feat + k])
            mean += v
            sumsq += v * v
        mean /= Float64(batch)
        var v = (sumsq / Float64(batch)) - mean * mean
        if v < 0.0:
            v = 0.0
        sum_std += sqrt(v)
    return sum_std / Float64(feat)


def main() raises:
    seed(0xC0FFEE)
    print("=" * 70)
    print("TD-MPC2 Test 1 — Consistency loss in isolation")
    print("=" * 70)

    var passed = 0
    var total = 0

    # ─── Build encoder + dynamics state ─────────────────────────────────
    var enc_state = NetworkState[EncModel, EncOpt]()
    enc_state.initialize[Normal[0.0, 0.02]]()
    var dyn_state = NetworkState[DynModel, DynOpt]()
    dyn_state.initialize[Normal[0.0, 0.02]]()

    # ─── Build fixed batch ──────────────────────────────────────────────
    var obs_arr = InlineArray[Scalar[dtype], BATCH * OBS](uninitialized=True)
    var act_arr = InlineArray[Scalar[dtype], BATCH * ACT](uninitialized=True)
    var obs_next_arr = InlineArray[Scalar[dtype], BATCH * OBS](
        uninitialized=True
    )
    var obs_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
    ](obs_arr.unsafe_ptr())
    var act_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, ACT), MutAnyOrigin
    ](act_arr.unsafe_ptr())
    var obs_next_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
    ](obs_next_arr.unsafe_ptr())
    _build_dataset(obs_t, act_t, obs_next_t)

    # ─── Workspace buffers ──────────────────────────────────────────────
    # Encoder forward+cache for s_t (gradient flows back through here).
    var z_t_arr = InlineArray[Scalar[dtype], BATCH * LATENT](
        uninitialized=True
    )
    var z_t_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
    ](z_t_arr.unsafe_ptr())
    var enc_cache_t_arr = InlineArray[
        Scalar[dtype], BATCH * EncModel.CACHE_SIZE
    ](uninitialized=True)
    var enc_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, EncModel.CACHE_SIZE), MutAnyOrigin
    ](enc_cache_t_arr.unsafe_ptr())

    # Encoder forward (no cache) for s_{t+1} (target = stop-grad).
    var z_target_arr = InlineArray[Scalar[dtype], BATCH * LATENT](
        uninitialized=True
    )
    var z_target_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
    ](z_target_arr.unsafe_ptr())

    # Concat (z_t, a_t) into ZA buffer.
    var za_arr = InlineArray[Scalar[dtype], BATCH * ZA](uninitialized=True)
    var za_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
    ](za_arr.unsafe_ptr())

    # Dynamics forward+cache for z_pred.
    var z_pred_arr = InlineArray[Scalar[dtype], BATCH * LATENT](
        uninitialized=True
    )
    var z_pred_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
    ](z_pred_arr.unsafe_ptr())
    var dyn_cache_arr = InlineArray[
        Scalar[dtype], BATCH * DynModel.CACHE_SIZE
    ](uninitialized=True)
    var dyn_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DynModel.CACHE_SIZE), MutAnyOrigin
    ](dyn_cache_arr.unsafe_ptr())

    # Gradients: grad_z_pred [B, LATENT], grad_za [B, ZA], grad_obs [B, OBS].
    var grad_z_pred_arr = InlineArray[Scalar[dtype], BATCH * LATENT](
        uninitialized=True
    )
    var grad_z_pred_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
    ](grad_z_pred_arr.unsafe_ptr())
    var grad_za_arr = InlineArray[Scalar[dtype], BATCH * ZA](
        uninitialized=True
    )
    var grad_za_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, ZA), MutAnyOrigin
    ](grad_za_arr.unsafe_ptr())
    var grad_z_arr = InlineArray[Scalar[dtype], BATCH * LATENT](
        uninitialized=True
    )
    var grad_z_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
    ](grad_z_arr.unsafe_ptr())
    var grad_obs_arr = InlineArray[Scalar[dtype], BATCH * OBS](
        uninitialized=True
    )
    var grad_obs_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, OBS), MutAnyOrigin
    ](grad_obs_arr.unsafe_ptr())

    # ─── Track properties across training ────────────────────────────────
    var initial_loss: Float64 = -1.0
    var final_loss: Float64 = 0.0
    var min_std_pred: Float64 = 1e30
    var min_std_target: Float64 = 1e30
    var enc_zero_grad_steps = 0
    var dyn_zero_grad_steps = 0

    print("[step]  loss      std(z_pred) std(z_target) |∇enc|     |∇dyn|")
    for step in range(NUM_STEPS):
        # ── Encoder fwd s_t with cache ──
        Network[EncModel, EncOpt].forward_with_cache[BATCH](
            obs_t,
            z_t_t,
            enc_state.params_view(),
            enc_state.model_state_view(),
            enc_cache_t,
        )
        # ── Encoder fwd s_{t+1} stop-grad ──
        Network[EncModel, EncOpt].forward[BATCH](
            obs_next_t,
            z_target_t,
            enc_state.params_view(),
            enc_state.model_state_view(),
        )

        # ── Build (z_t, a_t) into za ──
        for b in range(BATCH):
            for k in range(LATENT):
                za_t[b, k] = z_t_t[b, k]
            for k in range(ACT):
                za_t[b, LATENT + k] = act_t[b, k]

        # ── Dynamics fwd with cache ──
        Network[DynModel, DynOpt].forward_with_cache[BATCH](
            za_t,
            z_pred_t,
            dyn_state.params_view(),
            dyn_state.model_state_view(),
            dyn_cache_t,
        )

        # ── Loss + grad_z_pred (analytic MSE gradient) ──
        var loss: Float64 = 0.0
        var scale = 2.0 / Float64(BATCH * LATENT)
        for b in range(BATCH):
            for k in range(LATENT):
                var diff = (
                    Float64(z_pred_t[b, k][0])
                    - Float64(z_target_t[b, k][0])
                )
                loss += diff * diff
                grad_z_pred_t[b, k] = Scalar[dtype](scale * diff)
        loss /= Float64(BATCH * LATENT)

        var std_pred = _batch_std(z_pred_arr.unsafe_ptr(), BATCH, LATENT)
        var std_target = _batch_std(z_target_arr.unsafe_ptr(), BATCH, LATENT)
        if std_pred < min_std_pred:
            min_std_pred = std_pred
        if std_target < min_std_target:
            min_std_target = std_target
        if step == 0:
            initial_loss = loss
        final_loss = loss

        # ── Dynamics backward → grad_za, dyn grads accumulated ──
        dyn_state.zero_grads()
        var dyn_grads = dyn_state.grads_view()
        Network[DynModel, DynOpt].backward[BATCH](
            grad_z_pred_t,
            grad_za_t,
            dyn_state.params_view(),
            dyn_state.model_state_view(),
            dyn_cache_t,
            dyn_grads,
        )

        # ── Extract grad_z = grad_za[:, :LATENT] ──
        for b in range(BATCH):
            for k in range(LATENT):
                grad_z_t[b, k] = grad_za_t[b, k]

        # ── Encoder backward (cache from forward on s_t) ──
        enc_state.zero_grads()
        var enc_grads = enc_state.grads_view()
        Network[EncModel, EncOpt].backward[BATCH](
            grad_z_t,
            grad_obs_t,
            enc_state.params_view(),
            enc_state.model_state_view(),
            enc_cache_t,
            enc_grads,
        )

        # ── Grad-norm bookkeeping (BEFORE optimizer_step zeros / updates) ──
        var enc_gn = _l2_norm(enc_state.grads, EncModel.PARAM_SIZE)
        var dyn_gn = _l2_norm(dyn_state.grads, DynModel.PARAM_SIZE)
        if enc_gn == 0.0:
            enc_zero_grad_steps += 1
        if dyn_gn == 0.0:
            dyn_zero_grad_steps += 1

        # ── Optimizer steps ──
        enc_state.optimizer_step()
        dyn_state.optimizer_step()

        if step % LOG_EVERY == 0 or step == NUM_STEPS - 1:
            print(
                "[" + String(step) + "]",
                String(loss)[byte=:8],
                "  ",
                String(std_pred)[byte=:8],
                "  ",
                String(std_target)[byte=:8],
                "  ",
                String(enc_gn)[byte=:8],
                "  ",
                String(dyn_gn)[byte=:8],
            )

    # ─── Summary + assertions ───────────────────────────────────────────
    print()
    print("Initial loss:  ", initial_loss)
    print("Final loss:    ", final_loss)
    print("Reduction:     ", final_loss / initial_loss)
    print("min std(z_pred)  =", min_std_pred)
    print("min std(z_target)=", min_std_target)
    print("encoder zero-grad steps  =", enc_zero_grad_steps)
    print("dynamics zero-grad steps =", dyn_zero_grad_steps)
    print()

    _expect(
        final_loss < 0.5 * initial_loss,
        "1a — final loss < 0.5 * initial loss (loss decreasing)",
        passed,
        total,
    )
    _expect(
        min_std_pred > 0.05,
        "1b — std(z_pred) stays > 0.05 (no constant collapse)",
        passed,
        total,
    )
    _expect(
        min_std_target > 0.05,
        "1c — std(z_target) stays > 0.05 (no constant collapse)",
        passed,
        total,
    )
    _expect(
        enc_zero_grad_steps == 0,
        "1d — encoder grad-norm > 0 every step (gradient reaches encoder)",
        passed,
        total,
    )
    _expect(
        dyn_zero_grad_steps == 0,
        "1e — dynamics grad-norm > 0 every step (gradient reaches dynamics)",
        passed,
        total,
    )

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
