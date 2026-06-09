"""JEPA pipeline + SIGReg — extended Phase 2 exit.

Extends `test_jepa_smoke.mojo` with the SIGReg Gaussianity regularizer on
the projected embeddings. Now the encoder receives gradient from BOTH:

  - `pred_loss = MSE(pred, target_detached)` via the ctx_x slice
  - `sigreg_loss = SIGReg(emb_3d)` directly on every token

Loss:
    L = pred_loss + lambda_sigreg * sigreg_loss

This matches `references/le-wm-main/train.py:30-42` modulo the placeholder
predictor and additive (instead of AdaLN) action conditioning.

What this validates:
  - SIGReg's forward + backward integrate with the rest of the autodiff
    pipeline on a real-shape config (encoder output → projector output →
    SIGReg → loss).
  - Encoder gradients combine both loss-term contributions without NaN.
  - SIGReg's "scalar output replicated across BATCH" semantics correctly
    contribute the right total gradient seed.

Run:
    pixi run mojo run -I . tests/experimental/lewm/test_jepa_smoke_sigreg.mojo
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
from mojo_rl.nn.autodiff.primitives import SIGRegOp
from mojo_rl.experimental.lewm import LeWMEncoder, ActionEmbedder


# =============================================================================
# Comptime config — same as test_jepa_smoke.mojo
# =============================================================================

comptime BATCH = 2
comptime T = 3
comptime H = 2
comptime N_PREDS = 1
comptime IN_CH = 3
comptime IMG = 32
comptime PATCH = 4
comptime N_PATCHES = (IMG // PATCH) * (IMG // PATCH)
comptime HIDDEN = 32
comptime ENC_HEADS = 2
comptime ENC_LAYERS = 1
comptime EMB = 32
comptime PROJ_H = 64
comptime ACT = 2
comptime SMOOTHED = 16
comptime PRED_HEADS = 2
comptime PRED_FF = 64
comptime IMG_DIM = IN_CH * IMG * IMG
comptime BT = BATCH * T

# SIGReg knobs (toy config — matches the gradcheck test).
comptime SIG_NUM_PROJ = 8
comptime SIG_KNOTS = 5
comptime LAMBDA_SIGREG = 0.09         # paper default

# Models — identical to test_jepa_smoke.mojo.
comptime ENC = LeWMEncoder[
    IN_CH, IMG, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, N_PATCHES,
    EMB, 4, PROJ_H,
]
comptime AE = ActionEmbedder[T, ACT, SMOOTHED, EMB]
comptime PRED = TransformerBlock[EMB, PRED_HEADS, H, PRED_FF, True]
comptime _PredProjPerToken = Sequential[
    Linear[EMB, PROJ_H],
    BatchNorm1D[PROJ_H],
    GELU[PROJ_H],
    Linear[PROJ_H, EMB],
]
comptime PROJ = Tokenwise[H, _PredProjPerToken]

# SIGReg op operating on (B, T*EMB).
comptime SIG = SIGRegOp[EMB, T, SIG_NUM_PROJ, SIG_KNOTS]


def main() raises:
    print("=== JEPA + SIGReg smoke test ===")
    print()
    print(
        "  B=", BATCH, " T=", T, " EMB=", EMB,
        " num_proj=", SIG_NUM_PROJ, " knots=", SIG_KNOTS,
        " lambda=", LAMBDA_SIGREG,
    )

    # ------ Initialize models ------
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

    # ------ Inputs ------
    var pixels_arr = InlineArray[Scalar[dtype], BT * IMG_DIM](uninitialized=True)
    for i in range(BT * IMG_DIM):
        pixels_arr[i] = Scalar[dtype](0.27 * Float64(i % 19) / 19.0 - 0.4)
    var actions_arr = InlineArray[Scalar[dtype], BATCH * T * ACT](uninitialized=True)
    for i in range(BATCH * T * ACT):
        actions_arr[i] = Scalar[dtype](0.13 * Float64(i % 11) - 0.3)

    # ------ Forward buffers ------
    var emb_arr = InlineArray[Scalar[dtype], BT * EMB](uninitialized=True)
    var enc_cache_arr = InlineArray[Scalar[dtype], BT * ENC.CACHE_SIZE](uninitialized=True)
    var act_emb_arr = InlineArray[Scalar[dtype], BATCH * T * EMB](uninitialized=True)
    var ae_cache_arr = InlineArray[Scalar[dtype], BATCH * AE.CACHE_SIZE](uninitialized=True)
    var ctx_arr = InlineArray[Scalar[dtype], BATCH * H * EMB](uninitialized=True)
    var pred_raw_arr = InlineArray[Scalar[dtype], BATCH * H * EMB](uninitialized=True)
    var pred_arr = InlineArray[Scalar[dtype], BATCH * H * EMB](uninitialized=True)
    var pred_cache_arr = InlineArray[Scalar[dtype], BATCH * PRED.CACHE_SIZE](uninitialized=True)
    var proj_cache_arr = InlineArray[Scalar[dtype], BATCH * PROJ.CACHE_SIZE](uninitialized=True)

    # SIGReg buffers.
    var sigreg_out_arr = InlineArray[Scalar[dtype], BATCH](uninitialized=True)
    var sigreg_cache_arr = InlineArray[Scalar[dtype], BATCH * SIG.CACHE_SIZE](uninitialized=True)

    # ------ Forward ------
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

    # SIGReg forward on emb_bte (B, T*EMB). PARAM_SIZE=0 → empty params view.
    var empty_params = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=Int(0)))
    var sigreg_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
    ](sigreg_out_arr.unsafe_ptr())
    var sigreg_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, SIG.CACHE_SIZE), MutAnyOrigin
    ](sigreg_cache_arr.unsafe_ptr())

    SIG.eval[BATCH](emb_bte_t, sigreg_out_t, empty_params, sigreg_cache_t)

    # Compute losses.
    var scale = Float64(BATCH * H * EMB)
    var pred_loss = Float64(0.0)
    for b in range(BATCH):
        for i in range(H * EMB):
            var p = Float64(rebind[Scalar[dtype]](pred_t[b, i]))
            var tgt = Float64(rebind[Scalar[dtype]](emb_bte_t[b, N_PREDS * EMB + i]))
            var d = p - tgt
            pred_loss += d * d
    pred_loss /= scale

    # SIGReg output is the same statistic replicated across batch slots; take slot 0.
    var sigreg_loss = Float64(sigreg_out_arr[0])
    var total_loss = pred_loss + LAMBDA_SIGREG * sigreg_loss

    print()
    print(
        "  forward:",
        " pred_loss =", pred_loss,
        " sigreg_loss =", sigreg_loss,
        " total =", total_loss,
    )

    # ------ Backward ------
    # grad_pred[b, i] = 2/(B*H*EMB) * (pred - target)
    var grad_pred_arr = InlineArray[Scalar[dtype], BATCH * H * EMB](uninitialized=True)
    var inv_scale = Scalar[dtype](2.0 / scale)
    for b in range(BATCH):
        for i in range(H * EMB):
            var p = rebind[Scalar[dtype]](pred_t[b, i])
            var tgt = rebind[Scalar[dtype]](emb_bte_t[b, N_PREDS * EMB + i])
            grad_pred_arr[b * H * EMB + i] = inv_scale * (p - tgt)
    var grad_pred_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ](grad_pred_arr.unsafe_ptr())

    # grad on SIGReg output: lambda/BATCH per slot (sum = lambda → contributes lambda * stat to total).
    var grad_sigreg_arr = InlineArray[Scalar[dtype], BATCH](uninitialized=True)
    for b in range(BATCH):
        grad_sigreg_arr[b] = Scalar[dtype](LAMBDA_SIGREG / Float64(BATCH))
    var grad_sigreg_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
    ](grad_sigreg_arr.unsafe_ptr())

    # Param-grad buffers (zero-init).
    var enc_grads_arr = InlineArray[Scalar[dtype], ENC.PARAM_SIZE](uninitialized=True)
    var ae_grads_arr = InlineArray[Scalar[dtype], AE.PARAM_SIZE](uninitialized=True)
    var pred_grads_arr = InlineArray[Scalar[dtype], PRED.PARAM_SIZE](uninitialized=True)
    var proj_grads_arr = InlineArray[Scalar[dtype], PROJ.PARAM_SIZE](uninitialized=True)
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

    # Backprop pred_proj → grad_pred_raw
    var grad_pred_raw_arr = InlineArray[Scalar[dtype], BATCH * H * EMB](uninitialized=True)
    var grad_pred_raw_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ](grad_pred_raw_arr.unsafe_ptr())
    PROJ.backward[BATCH](
        grad_pred_t, grad_pred_raw_t, proj_params, proj_mstate, proj_cache_t, proj_grads_t
    )

    # Backprop predictor → grad_ctx
    var grad_ctx_arr = InlineArray[Scalar[dtype], BATCH * H * EMB](uninitialized=True)
    var grad_ctx_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ](grad_ctx_arr.unsafe_ptr())
    PRED.backward[BATCH](
        grad_pred_raw_t, grad_ctx_t, pred_params, pred_mstate, pred_cache_t, pred_grads_t
    )

    # Build grad_emb: ctx_x slice gets grad_ctx, rest zero. THEN add SIGReg grad.
    var grad_emb_arr = InlineArray[Scalar[dtype], BATCH * T * EMB](uninitialized=True)
    var grad_act_emb_arr = InlineArray[Scalar[dtype], BATCH * T * EMB](uninitialized=True)
    for b in range(BATCH):
        for i in range(T * EMB):
            grad_emb_arr[b * T * EMB + i] = Scalar[dtype](0)
            grad_act_emb_arr[b * T * EMB + i] = Scalar[dtype](0)
    for b in range(BATCH):
        for i in range(H * EMB):
            var g = rebind[Scalar[dtype]](grad_ctx_t[b, i])
            grad_emb_arr[b * T * EMB + i] = g                # ctx_x branch
            grad_act_emb_arr[b * T * EMB + i] = g            # ctx_a branch

    # SIGReg.vjp ACCUMULATES into grad_emb (we already populated it with pred-path grads).
    # The op overwrites grad_input — we need to call into a separate buffer and add.
    var sigreg_grad_emb_arr = InlineArray[Scalar[dtype], BATCH * T * EMB](uninitialized=True)
    var grad_emb_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * EMB), MutAnyOrigin
    ](grad_emb_arr.unsafe_ptr())
    var sigreg_grad_emb_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * EMB), MutAnyOrigin
    ](sigreg_grad_emb_arr.unsafe_ptr())
    var empty_grad_params = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=Int(0)))

    SIG.vjp[BATCH](
        grad_sigreg_t,
        sigreg_grad_emb_t,
        empty_params,
        sigreg_cache_t,
        empty_grad_params,
    )

    # Sum SIGReg's grad into grad_emb.
    for b in range(BATCH):
        for i in range(T * EMB):
            grad_emb_arr[b * T * EMB + i] = (
                grad_emb_arr[b * T * EMB + i] + sigreg_grad_emb_arr[b * T * EMB + i]
            )

    var grad_act_emb_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * EMB), MutAnyOrigin
    ](grad_act_emb_arr.unsafe_ptr())

    # Backprop action_embedder
    var grad_actions_arr = InlineArray[Scalar[dtype], BATCH * T * ACT](uninitialized=True)
    var grad_actions_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * ACT), MutAnyOrigin
    ](grad_actions_arr.unsafe_ptr())
    AE.backward[BATCH](
        grad_act_emb_t, grad_actions_t, ae_params, ae_mstate, ae_cache_t, ae_grads_t
    )

    # Backprop encoder (grad_emb viewed as (BT, EMB))
    var grad_emb_bt_t = LayoutTensor[
        dtype, Layout.row_major(BT, EMB), MutAnyOrigin
    ](grad_emb_arr.unsafe_ptr())
    var grad_pixels_arr = InlineArray[Scalar[dtype], BT * IMG_DIM](uninitialized=True)
    var grad_pixels_t = LayoutTensor[
        dtype, Layout.row_major(BT, IMG_DIM), MutAnyOrigin
    ](grad_pixels_arr.unsafe_ptr())
    ENC.backward[BT](
        grad_emb_bt_t, grad_pixels_t, enc_params, enc_mstate, enc_cache_t, enc_grads_t
    )

    # ------ Summarize ------
    def _summarize(
        name: String, grads: UnsafePointer[Scalar[dtype], MutAnyOrigin], n: Int
    ) raises -> Tuple[Int, Float64, Bool]:
        var nz = 0
        var mx = Float64(0.0)
        var nan = False
        for i in range(n):
            var v = Float64(grads[i])
            if v != v:
                nan = True
            var av = abs(v)
            if av > mx:
                mx = av
            if av > 1e-8:
                nz += 1
        print(
            "  ", name, ": nz =", nz, "/", n,
            "  max|g| =", mx, "  nan =", nan,
        )
        return (nz, mx, nan)

    var enc_s = _summarize(
        "encoder        ", enc_grads_arr.unsafe_ptr(), ENC.PARAM_SIZE
    )
    var ae_s = _summarize(
        "action_embedder", ae_grads_arr.unsafe_ptr(), AE.PARAM_SIZE
    )
    var pred_s = _summarize(
        "predictor      ", pred_grads_arr.unsafe_ptr(), PRED.PARAM_SIZE
    )
    var proj_s = _summarize(
        "pred_proj      ", proj_grads_arr.unsafe_ptr(), PROJ.PARAM_SIZE
    )

    # SIGReg contribution probe: max abs of the SIGReg-only grad on emb.
    var sig_max = Float64(0.0)
    for i in range(BATCH * T * EMB):
        var av = abs(Float64(sigreg_grad_emb_arr[i]))
        if av > sig_max:
            sig_max = av
    print()
    print("  SIGReg grad probe: max|d sigreg / d emb| =", sig_max)

    var ok = True
    if enc_s[2] or ae_s[2] or pred_s[2] or proj_s[2]:
        ok = False
    if enc_s[0] < ENC.PARAM_SIZE // 4:
        ok = False
    if ae_s[0] < AE.PARAM_SIZE // 4:
        ok = False
    if pred_s[0] < PRED.PARAM_SIZE // 4:
        ok = False
    if proj_s[0] < PROJ.PARAM_SIZE // 4:
        ok = False
    if sig_max < 1e-8:
        ok = False                              # SIGReg should contribute something
    if (
        enc_s[1] > 1e6 or ae_s[1] > 1e6
        or pred_s[1] > 1e6 or proj_s[1] > 1e6
    ):
        ok = False

    print()
    if ok:
        print("  [PASS] JEPA+SIGReg smoke: full pipeline runs with finite combined grads")
    else:
        print("  [FAIL] JEPA+SIGReg smoke")
    print()
    print("=== Done ===")
