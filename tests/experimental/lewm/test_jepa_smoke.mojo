"""JEPA pipeline smoke test (Phase 2 exit criterion).

End-to-end forward + backward through the lower-fidelity JEPA pipeline:

    pixels (B, T, C, H, W)
        └─> Encoder ─> emb (B, T, EMB)
    actions (B, T, ACT)
        └─> ActionEmbedder ─> act_emb (B, T, EMB)

    ctx_x   = emb[:, :H]                         (B, H, EMB)   (with grad)
    ctx_a   = act_emb[:, :H]                     (B, H, EMB)   (with grad)
    ctx     = ctx_x + ctx_a                       (B, H, EMB)
    pred    = pred_proj(predictor(ctx))           (B, H, EMB)
    target  = emb[:, n_preds:n_preds+H].detach()  (B, H, EMB)   (no grad)

    L_pred  = (1/(B*H*EMB)) * sum((pred - target)^2)

We DROP the SIGReg term and the AdaLN-zero conditioning for this validation:
  - Predictor = standard causal `TransformerBlock` over H tokens (no AdaLN).
  - Conditioning = additive `ctx_x + ctx_a` instead of AdaLN modulation.

This is a deliberate scope reduction (`docs/LEWM_PORT_PLAN.md` §Phase 2 b):
the AR predictor + SIGReg integration happens in Phase 3 once the conditional
block is wrapped as a single-input Model. This test validates the data path,
shape plumbing, and gradient routing through the four parameter groups
(encoder, action_embedder, predictor, pred_proj) on a toy config.

Pass conditions:
  - Forward produces no NaN at any stage.
  - Each of the four parameter groups receives non-zero gradients.
  - max|grad| stays finite (< 1e6 — sanity bound, not a stability check).

Toy config (Phase 2 exit budget — ~250k params):
  B=2, T=3, H=2, n_preds=1, IN_CH=3, IMG=32, EMB=32

Run:
    pixi run mojo run -I . tests/experimental/lewm/test_jepa_smoke.mojo
"""

from std.math import abs
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.training import NetworkState
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.model import (
    Sequential, Linear, BatchNorm1D, Tokenwise,
)
from mojo_rl.nn.model.autodiff_layers import GELU
from mojo_rl.nn.composites import TransformerBlock
from mojo_rl.experimental.lewm import LeWMEncoder, ActionEmbedder


# =============================================================================
# Comptime config
# =============================================================================

comptime BATCH = 2
comptime T = 3
comptime H = 2                  # history_size — predictor context length
comptime N_PREDS = 1            # target = emb[:, n_preds : n_preds + H]
comptime IN_CH = 3
comptime IMG = 32
comptime PATCH = 4
comptime N_PATCHES = (IMG // PATCH) * (IMG // PATCH)   # 64
comptime HIDDEN = 32            # encoder hidden_dim
comptime ENC_HEADS = 2
comptime ENC_LAYERS = 1
comptime EMB = 32               # shared embedding dim
comptime PROJ_H = 64            # encoder & pred_proj inner MLP dim
comptime ACT = 2
comptime SMOOTHED = 16          # action_embedder smoothed dim
comptime PRED_HEADS = 2
comptime PRED_FF = 64
comptime SIZE_PIX_PER_SAMPLE = IN_CH * IMG * IMG
comptime IMG_DIM = SIZE_PIX_PER_SAMPLE       # = ENCODER.IN_DIM
comptime BT = BATCH * T                       # effective encoder batch


# =============================================================================
# Model definitions
# =============================================================================

# Encoder: pixels (BT, 3*32*32) → (BT, EMB)
comptime ENC = LeWMEncoder[
    IN_CH, IMG, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, N_PATCHES,
    EMB, 4, PROJ_H,
]

# ActionEmbedder: (B, T*ACT) → (B, T*EMB)
comptime AE = ActionEmbedder[T, ACT, SMOOTHED, EMB]

# Predictor stand-in: causal TransformerBlock over H tokens.
#   IN_DIM = OUT_DIM = H * EMB.
comptime PRED = TransformerBlock[EMB, PRED_HEADS, H, PRED_FF, True]

# Pred-proj: per-token MLP [Linear → BN1D → GELU → Linear], lifted Tokenwise[H, ...].
comptime _PredProjPerToken = Sequential[
    Linear[EMB, PROJ_H],
    BatchNorm1D[PROJ_H],
    GELU[PROJ_H],
    Linear[PROJ_H, EMB],
]
comptime PROJ = Tokenwise[H, _PredProjPerToken]


def main() raises:
    print("=== JEPA pipeline smoke test (Phase 2 exit) ===")
    print()
    print(
        "  B=", BATCH,
        " T=", T,
        " H=", H,
        " n_preds=", N_PREDS,
        " IMG=", IMG, "x", IMG, "x", IN_CH,
        " EMB=", EMB,
    )
    print(
        "  ENC.PARAM_SIZE=", ENC.PARAM_SIZE,
        " AE.PARAM_SIZE=", AE.PARAM_SIZE,
        " PRED.PARAM_SIZE=", PRED.PARAM_SIZE,
        " PROJ.PARAM_SIZE=", PROJ.PARAM_SIZE,
    )

    # ------------------------------------------------------------------
    # Initialize four models
    # ------------------------------------------------------------------
    var enc_state = NetworkState[ENC, Adam[]]()
    var ae_state = NetworkState[AE, Adam[]]()
    var pred_state = NetworkState[PRED, Adam[]]()
    var proj_state = NetworkState[PROJ, Adam[]]()

    enc_state.initialize[Xavier[]]()
    ae_state.initialize[Xavier[]]()
    pred_state.initialize[Xavier[]]()
    proj_state.initialize[Xavier[]]()

    var enc_params = enc_state.params_view()
    var ae_params = ae_state.params_view()
    var pred_params = pred_state.params_view()
    var proj_params = proj_state.params_view()

    var enc_mstate = enc_state.model_state_view()
    var ae_mstate = ae_state.model_state_view()
    var pred_mstate = pred_state.model_state_view()
    var proj_mstate = proj_state.model_state_view()

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    # pixels: (BT, IMG_DIM) — effective encoder batch is B*T
    var pixels_arr = InlineArray[Scalar[dtype], BT * IMG_DIM](uninitialized=True)
    for i in range(BT * IMG_DIM):
        pixels_arr[i] = Scalar[dtype](0.27 * Float64(i % 19) / 19.0 - 0.4)

    # actions: (B, T*ACT)
    var actions_arr = InlineArray[Scalar[dtype], BATCH * T * ACT](uninitialized=True)
    for i in range(BATCH * T * ACT):
        actions_arr[i] = Scalar[dtype](0.13 * Float64(i % 11) - 0.3)

    # ------------------------------------------------------------------
    # Buffers
    # ------------------------------------------------------------------
    # emb: (BT, EMB) — also viewed as (B, T*EMB) for downstream
    var emb_arr = InlineArray[Scalar[dtype], BT * EMB](uninitialized=True)
    var enc_cache_arr = InlineArray[Scalar[dtype], BT * ENC.CACHE_SIZE](
        uninitialized=True
    )

    # act_emb: (B, T*EMB)
    var act_emb_arr = InlineArray[Scalar[dtype], BATCH * T * EMB](uninitialized=True)
    var ae_cache_arr = InlineArray[Scalar[dtype], BATCH * AE.CACHE_SIZE](
        uninitialized=True
    )

    # ctx_x, ctx_a, ctx (all shape (B, H*EMB))
    var ctx_arr = InlineArray[Scalar[dtype], BATCH * H * EMB](uninitialized=True)

    # pred_raw, pred (B, H*EMB)
    var pred_raw_arr = InlineArray[Scalar[dtype], BATCH * H * EMB](
        uninitialized=True
    )
    var pred_arr = InlineArray[Scalar[dtype], BATCH * H * EMB](uninitialized=True)
    var pred_cache_arr = InlineArray[Scalar[dtype], BATCH * PRED.CACHE_SIZE](
        uninitialized=True
    )
    var proj_cache_arr = InlineArray[Scalar[dtype], BATCH * PROJ.CACHE_SIZE](
        uninitialized=True
    )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    var pixels_t = LayoutTensor[
        dtype, Layout.row_major(BT, IMG_DIM), MutAnyOrigin
    ](pixels_arr.unsafe_ptr())
    var emb_t = LayoutTensor[
        dtype, Layout.row_major(BT, EMB), MutAnyOrigin
    ](emb_arr.unsafe_ptr())
    var enc_cache_t = LayoutTensor[
        dtype, Layout.row_major(BT, ENC.CACHE_SIZE), MutAnyOrigin
    ](enc_cache_arr.unsafe_ptr())

    ENC.forward[BT](pixels_t, emb_t, enc_params, enc_mstate, enc_cache_t)

    # View emb as (B, T*EMB) — same memory, different layout.
    var emb_bte_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * EMB), MutAnyOrigin
    ](emb_arr.unsafe_ptr())

    var actions_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * ACT), MutAnyOrigin
    ](actions_arr.unsafe_ptr())
    var act_emb_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * EMB), MutAnyOrigin
    ](act_emb_arr.unsafe_ptr())
    var ae_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, AE.CACHE_SIZE), MutAnyOrigin
    ](ae_cache_arr.unsafe_ptr())

    AE.forward[BATCH](actions_t, act_emb_t, ae_params, ae_mstate, ae_cache_t)

    # ctx = emb[:, :H*EMB] + act_emb[:, :H*EMB]
    for b in range(BATCH):
        for i in range(H * EMB):
            ctx_arr[b * H * EMB + i] = (
                rebind[Scalar[dtype]](emb_bte_t[b, i])
                + rebind[Scalar[dtype]](act_emb_t[b, i])
            )

    var ctx_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ](ctx_arr.unsafe_ptr())
    var pred_raw_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ](pred_raw_arr.unsafe_ptr())
    var pred_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, PRED.CACHE_SIZE), MutAnyOrigin
    ](pred_cache_arr.unsafe_ptr())

    PRED.forward[BATCH](
        ctx_t, pred_raw_t, pred_params, pred_mstate, pred_cache_t
    )

    var pred_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ](pred_arr.unsafe_ptr())
    var proj_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, PROJ.CACHE_SIZE), MutAnyOrigin
    ](proj_cache_arr.unsafe_ptr())

    PROJ.forward[BATCH](
        pred_raw_t, pred_t, proj_params, proj_mstate, proj_cache_t
    )

    # ------------------------------------------------------------------
    # Loss: L = (1/(B*H*EMB)) * sum((pred - target)^2)
    # target slice = emb_bte[:, n_preds*EMB : (n_preds + H)*EMB]
    # ------------------------------------------------------------------
    var scale = Float64(BATCH * H * EMB)
    var pred_loss = Float64(0.0)
    var any_nan = False
    var max_abs_pred = Float64(0.0)
    var max_abs_target = Float64(0.0)
    for b in range(BATCH):
        for i in range(H * EMB):
            var p = Float64(rebind[Scalar[dtype]](pred_t[b, i]))
            var tgt = Float64(
                rebind[Scalar[dtype]](emb_bte_t[b, N_PREDS * EMB + i])
            )
            if p != p or tgt != tgt:
                any_nan = True
            if abs(p) > max_abs_pred:
                max_abs_pred = abs(p)
            if abs(tgt) > max_abs_target:
                max_abs_target = abs(tgt)
            var diff = p - tgt
            pred_loss += diff * diff
    pred_loss /= scale

    print()
    print(
        "  forward: pred_loss =",
        pred_loss,
        " max|pred| =",
        max_abs_pred,
        " max|target| =",
        max_abs_target,
        " any_nan =",
        any_nan,
    )

    if any_nan:
        print("  [FAIL] forward produced NaN — pipeline broken")
        return

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------
    # grad_pred[b, i] = 2 * (pred[b, i] - target[b, i]) / (B*H*EMB)
    var grad_pred_arr = InlineArray[Scalar[dtype], BATCH * H * EMB](
        uninitialized=True
    )
    var inv_scale = Scalar[dtype](2.0 / scale)
    for b in range(BATCH):
        for i in range(H * EMB):
            var p = rebind[Scalar[dtype]](pred_t[b, i])
            var tgt = rebind[Scalar[dtype]](emb_bte_t[b, N_PREDS * EMB + i])
            grad_pred_arr[b * H * EMB + i] = inv_scale * (p - tgt)

    var grad_pred_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ](grad_pred_arr.unsafe_ptr())

    # Param-grad buffers (zero-init).
    var enc_grads_arr = InlineArray[Scalar[dtype], ENC.PARAM_SIZE](
        uninitialized=True
    )
    var ae_grads_arr = InlineArray[Scalar[dtype], AE.PARAM_SIZE](
        uninitialized=True
    )
    var pred_grads_arr = InlineArray[Scalar[dtype], PRED.PARAM_SIZE](
        uninitialized=True
    )
    var proj_grads_arr = InlineArray[Scalar[dtype], PROJ.PARAM_SIZE](
        uninitialized=True
    )
    for i in range(ENC.PARAM_SIZE):
        enc_grads_arr[i] = Scalar[dtype](0.0)
    for i in range(AE.PARAM_SIZE):
        ae_grads_arr[i] = Scalar[dtype](0.0)
    for i in range(PRED.PARAM_SIZE):
        pred_grads_arr[i] = Scalar[dtype](0.0)
    for i in range(PROJ.PARAM_SIZE):
        proj_grads_arr[i] = Scalar[dtype](0.0)

    var enc_grads_t = LayoutTensor[
        dtype, Layout.row_major(ENC.PARAM_SIZE), MutAnyOrigin
    ](enc_grads_arr.unsafe_ptr())
    var ae_grads_t = LayoutTensor[
        dtype, Layout.row_major(AE.PARAM_SIZE), MutAnyOrigin
    ](ae_grads_arr.unsafe_ptr())
    var pred_grads_t = LayoutTensor[
        dtype, Layout.row_major(PRED.PARAM_SIZE), MutAnyOrigin
    ](pred_grads_arr.unsafe_ptr())
    var proj_grads_t = LayoutTensor[
        dtype, Layout.row_major(PROJ.PARAM_SIZE), MutAnyOrigin
    ](proj_grads_arr.unsafe_ptr())

    # Backprop pred_proj: grad_pred → grad_pred_raw
    var grad_pred_raw_arr = InlineArray[Scalar[dtype], BATCH * H * EMB](
        uninitialized=True
    )
    var grad_pred_raw_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ](grad_pred_raw_arr.unsafe_ptr())

    PROJ.backward[BATCH](
        grad_pred_t,
        grad_pred_raw_t,
        proj_params,
        proj_mstate,
        proj_cache_t,
        proj_grads_t,
    )

    # Backprop predictor: grad_pred_raw → grad_ctx
    var grad_ctx_arr = InlineArray[Scalar[dtype], BATCH * H * EMB](
        uninitialized=True
    )
    var grad_ctx_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ](grad_ctx_arr.unsafe_ptr())

    PRED.backward[BATCH](
        grad_pred_raw_t,
        grad_ctx_t,
        pred_params,
        pred_mstate,
        pred_cache_t,
        pred_grads_t,
    )

    # Split grad_ctx into grad_emb_ctx_slice and grad_act_emb_ctx_slice.
    # Build full grad_emb (B, T*EMB): zero except ctx_x slice gets grad_ctx.
    # Build full grad_act_emb (B, T*EMB): zero except ctx_a slice gets grad_ctx.
    var grad_emb_arr = InlineArray[Scalar[dtype], BATCH * T * EMB](
        uninitialized=True
    )
    var grad_act_emb_arr = InlineArray[Scalar[dtype], BATCH * T * EMB](
        uninitialized=True
    )
    for b in range(BATCH):
        for i in range(T * EMB):
            grad_emb_arr[b * T * EMB + i] = Scalar[dtype](0)
            grad_act_emb_arr[b * T * EMB + i] = Scalar[dtype](0)
    for b in range(BATCH):
        for i in range(H * EMB):
            var g = rebind[Scalar[dtype]](grad_ctx_t[b, i])
            grad_emb_arr[b * T * EMB + i] = g
            grad_act_emb_arr[b * T * EMB + i] = g

    var grad_emb_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * EMB), MutAnyOrigin
    ](grad_emb_arr.unsafe_ptr())
    var grad_act_emb_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * EMB), MutAnyOrigin
    ](grad_act_emb_arr.unsafe_ptr())

    # Backprop action_embedder: grad_act_emb → grad_actions
    var grad_actions_arr = InlineArray[Scalar[dtype], BATCH * T * ACT](
        uninitialized=True
    )
    var grad_actions_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * ACT), MutAnyOrigin
    ](grad_actions_arr.unsafe_ptr())

    AE.backward[BATCH](
        grad_act_emb_t,
        grad_actions_t,
        ae_params,
        ae_mstate,
        ae_cache_t,
        ae_grads_t,
    )

    # Backprop encoder: view grad_emb as (BT, EMB), backward into grad_pixels.
    var grad_emb_bt_t = LayoutTensor[
        dtype, Layout.row_major(BT, EMB), MutAnyOrigin
    ](grad_emb_arr.unsafe_ptr())

    var grad_pixels_arr = InlineArray[Scalar[dtype], BT * IMG_DIM](
        uninitialized=True
    )
    var grad_pixels_t = LayoutTensor[
        dtype, Layout.row_major(BT, IMG_DIM), MutAnyOrigin
    ](grad_pixels_arr.unsafe_ptr())

    ENC.backward[BT](
        grad_emb_bt_t,
        grad_pixels_t,
        enc_params,
        enc_mstate,
        enc_cache_t,
        enc_grads_t,
    )

    # ------------------------------------------------------------------
    # Verify: every model has non-zero, finite gradients
    # ------------------------------------------------------------------
    def _summarize(
        name: String, grads: UnsafePointer[Scalar[dtype], MutAnyOrigin], n: Int
    ) raises -> Tuple[Int, Float64, Bool]:
        var nz = 0
        var max_abs = Float64(0.0)
        var nan_seen = False
        for i in range(n):
            var v = Float64(grads[i])
            if v != v:
                nan_seen = True
            var av = abs(v)
            if av > max_abs:
                max_abs = av
            if av > 1e-8:
                nz += 1
        print(
            "  ",
            name,
            ": nz =",
            nz,
            "/",
            n,
            " max|g| =",
            max_abs,
            " nan =",
            nan_seen,
        )
        return (nz, max_abs, nan_seen)

    var enc_stats = _summarize(
        "encoder        ", enc_grads_arr.unsafe_ptr(), ENC.PARAM_SIZE
    )
    var ae_stats = _summarize(
        "action_embedder", ae_grads_arr.unsafe_ptr(), AE.PARAM_SIZE
    )
    var pred_stats = _summarize(
        "predictor      ", pred_grads_arr.unsafe_ptr(), PRED.PARAM_SIZE
    )
    var proj_stats = _summarize(
        "pred_proj      ", proj_grads_arr.unsafe_ptr(), PROJ.PARAM_SIZE
    )

    var all_ok = True
    if enc_stats[2] or ae_stats[2] or pred_stats[2] or proj_stats[2]:
        all_ok = False
    if enc_stats[0] < ENC.PARAM_SIZE // 4:
        all_ok = False
    if ae_stats[0] < AE.PARAM_SIZE // 4:
        all_ok = False
    if pred_stats[0] < PRED.PARAM_SIZE // 4:
        all_ok = False
    if proj_stats[0] < PROJ.PARAM_SIZE // 4:
        all_ok = False
    if (
        enc_stats[1] > 1e6
        or ae_stats[1] > 1e6
        or pred_stats[1] > 1e6
        or proj_stats[1] > 1e6
    ):
        all_ok = False

    print()
    if all_ok:
        print("  [PASS] JEPA pipeline smoke: all 4 groups have non-zero finite grads")
    else:
        print("  [FAIL] JEPA pipeline smoke")
    print()
    print("=== Done ===")
