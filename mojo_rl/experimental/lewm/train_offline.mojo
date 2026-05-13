"""LeWM offline training loop (Phase 3, CPU smoke).

Loads a PongBuffer from disk, runs N optimizer steps over the lower-fidelity
JEPA pipeline from `test_jepa_smoke.mojo`:

    pixels  ─Encoder─→ emb            (BT, EMB)
    actions ─AE─→      act_emb        (B, T·EMB)
    ctx     = emb[:, :H·EMB] + act_emb[:, :H·EMB]
    pred    = PredProj(Predictor(ctx))                 (B, H·EMB)
    target  = emb[:, n_preds·EMB : (n_preds+H)·EMB].detach()
    L_pred  = mean((pred - target)^2)

No SIGReg in this first smoke pass — the goal is to validate that pred_loss
moves under gradient descent on real Pong frames before pulling in the
regularizer. SIGReg is wired in a follow-up (see plan §Phase 3).

This file exposes `train_lewm_offline(...)` as a function that accepts the
configuration as comptime parameters and a runtime buffer path.

Usage from a top-level example:

    from mojo_rl.experimental.lewm.train_offline import train_lewm_offline

    train_lewm_offline[
        BATCH=4, T=4, H=3, N_PREDS=1,
        IN_CH=4, IMG=84, PATCH=14, N_PATCHES=36,
        HIDDEN=32, ENC_HEADS=2, ENC_LAYERS=1, EMB=32, PROJ_H=64,
        ACT=3, SMOOTHED=16,
        PRED_HEADS=2, PRED_FF=64,
    ](
        buffer_path="/tmp/lewm_pong_buffer.bin",
        num_steps=200,
        log_every=20,
        seed=0xCAFE,
    )
"""

from std.math import abs
from std.memory import alloc
from std.random import seed as _set_seed
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor

from ...nn.constants import dtype
from ...nn.training import NetworkState
from ...nn.optimizer import Adam
from ...nn.initializer import Xavier
from ...nn.model import (
    Sequential, Linear, BatchNorm1D, Tokenwise,
)
from ...nn.model.autodiff_layers import GELU
from ...nn.composites import TransformerBlock
from .encoder import LeWMEncoder
from .action_embedder import ActionEmbedder
from .pong_buffer import (
    PongBuffer,
    PONG_FRAME_BYTES,
    PONG_NUM_ACTIONS,
)


# ============================================================================
# Helpers
# ============================================================================


@always_inline
def _zero(p: UnsafePointer[Scalar[dtype], MutAnyOrigin], n: Int):
    for i in range(n):
        p[i] = Scalar[dtype](0)


@always_inline
def _accum_stats(
    p: UnsafePointer[Scalar[dtype], MutAnyOrigin], n: Int
) -> Tuple[Float64, Float64, Int, Bool]:
    """Return (mean, max_abs, nnz, any_nan)."""
    var s = Float64(0.0)
    var m = Float64(0.0)
    var nz = 0
    var nan = False
    for i in range(n):
        var v = Float64(p[i])
        if v != v:
            nan = True
        s += v
        var av = abs(v)
        if av > m:
            m = av
        if av > 1e-8:
            nz += 1
    return (s / Float64(n), m, nz, nan)


# ============================================================================
# Main trainer
# ============================================================================


def train_lewm_offline[
    BATCH: Int,
    T: Int,
    H: Int,
    N_PREDS: Int,
    IN_CH: Int,
    IMG: Int,
    PATCH: Int,
    N_PATCHES: Int,
    HIDDEN: Int,
    ENC_HEADS: Int,
    ENC_LAYERS: Int,
    EMB: Int,
    PROJ_H: Int,
    ACT: Int,
    SMOOTHED: Int,
    PRED_HEADS: Int,
    PRED_FF: Int,
](
    buffer_path: String,
    num_steps: Int,
    log_every: Int = 10,
    rng_seed: Int = 0xCAFE,
) raises:
    """Run the offline JEPA training loop. CPU-only smoke variant."""

    comptime IMG_DIM: Int = IN_CH * IMG * IMG  # = ENC.IN_DIM
    comptime BT: Int = BATCH * T

    comptime ENC = LeWMEncoder[
        IN_CH, IMG, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, N_PATCHES,
        EMB, 2, PROJ_H,
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

    _set_seed(rng_seed)

    # ------------------------------------------------------------------
    # Load buffer
    # ------------------------------------------------------------------
    var buf = PongBuffer.load(buffer_path)
    print("Loaded buffer:", buf.n_frames, "frames from", buffer_path)
    print(
        "Models — ENC.PARAM=",
        ENC.PARAM_SIZE,
        " AE.PARAM=",
        AE.PARAM_SIZE,
        " PRED.PARAM=",
        PRED.PARAM_SIZE,
        " PROJ.PARAM=",
        PROJ.PARAM_SIZE,
    )
    var total_params = (
        ENC.PARAM_SIZE + AE.PARAM_SIZE + PRED.PARAM_SIZE + PROJ.PARAM_SIZE
    )
    print("Total params:", total_params)

    # ------------------------------------------------------------------
    # Build four NetworkState[Model, Adam[]] groups
    # ------------------------------------------------------------------
    var enc_state = NetworkState[ENC, Adam[]]()
    var ae_state = NetworkState[AE, Adam[]]()
    var pred_state = NetworkState[PRED, Adam[]]()
    var proj_state = NetworkState[PROJ, Adam[]]()
    enc_state.initialize[Xavier[]]()
    ae_state.initialize[Xavier[]]()
    pred_state.initialize[Xavier[]]()
    proj_state.initialize[Xavier[]]()

    # ------------------------------------------------------------------
    # Heap-allocated activation/cache/gradient buffers (reused per step)
    # ------------------------------------------------------------------
    var pixels = alloc[Scalar[dtype]](BT * IMG_DIM)
    var actions = alloc[Scalar[dtype]](BATCH * T * ACT)

    var emb = alloc[Scalar[dtype]](BT * EMB)
    var enc_cache = alloc[Scalar[dtype]](BT * ENC.CACHE_SIZE)

    var act_emb = alloc[Scalar[dtype]](BATCH * T * EMB)
    var ae_cache = alloc[Scalar[dtype]](BATCH * AE.CACHE_SIZE)

    var ctx = alloc[Scalar[dtype]](BATCH * H * EMB)
    var pred_raw = alloc[Scalar[dtype]](BATCH * H * EMB)
    var pred_out = alloc[Scalar[dtype]](BATCH * H * EMB)
    var pred_cache = alloc[Scalar[dtype]](BATCH * PRED.CACHE_SIZE)
    var proj_cache = alloc[Scalar[dtype]](BATCH * PROJ.CACHE_SIZE)

    var grad_pred = alloc[Scalar[dtype]](BATCH * H * EMB)
    var grad_pred_raw = alloc[Scalar[dtype]](BATCH * H * EMB)
    var grad_ctx = alloc[Scalar[dtype]](BATCH * H * EMB)
    var grad_emb = alloc[Scalar[dtype]](BATCH * T * EMB)
    var grad_act_emb = alloc[Scalar[dtype]](BATCH * T * EMB)
    var grad_actions = alloc[Scalar[dtype]](BATCH * T * ACT)
    var grad_pixels = alloc[Scalar[dtype]](BT * IMG_DIM)

    # ------------------------------------------------------------------
    # Layout tensors over the heap buffers
    # ------------------------------------------------------------------
    var pixels_t = LayoutTensor[
        dtype, Layout.row_major(BT, IMG_DIM), MutAnyOrigin
    ](pixels)
    var actions_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * ACT), MutAnyOrigin
    ](actions)

    var emb_t = LayoutTensor[
        dtype, Layout.row_major(BT, EMB), MutAnyOrigin
    ](emb)
    var emb_bte_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * EMB), MutAnyOrigin
    ](emb)
    var enc_cache_t = LayoutTensor[
        dtype, Layout.row_major(BT, ENC.CACHE_SIZE), MutAnyOrigin
    ](enc_cache)

    var act_emb_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * EMB), MutAnyOrigin
    ](act_emb)
    var ae_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, AE.CACHE_SIZE), MutAnyOrigin
    ](ae_cache)

    var ctx_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ](ctx)
    var pred_raw_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ](pred_raw)
    var pred_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ](pred_out)
    var pred_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, PRED.CACHE_SIZE), MutAnyOrigin
    ](pred_cache)
    var proj_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, PROJ.CACHE_SIZE), MutAnyOrigin
    ](proj_cache)

    var grad_pred_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ](grad_pred)
    var grad_pred_raw_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ](grad_pred_raw)
    var grad_ctx_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ](grad_ctx)
    var grad_emb_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * EMB), MutAnyOrigin
    ](grad_emb)
    var grad_emb_bt_t = LayoutTensor[
        dtype, Layout.row_major(BT, EMB), MutAnyOrigin
    ](grad_emb)
    var grad_act_emb_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * EMB), MutAnyOrigin
    ](grad_act_emb)
    var grad_actions_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * ACT), MutAnyOrigin
    ](grad_actions)
    var grad_pixels_t = LayoutTensor[
        dtype, Layout.row_major(BT, IMG_DIM), MutAnyOrigin
    ](grad_pixels)

    # ------------------------------------------------------------------
    # Step loop
    # ------------------------------------------------------------------
    var loss_ema: Float64 = 0.0
    var loss_first: Float64 = -1.0
    var loss_last: Float64 = 0.0
    var t0 = perf_counter_ns()

    for step in range(num_steps):
        # Sample a fresh batch.
        buf.sample_batch_fp32(BATCH, T, pixels, actions)

        # Zero grads
        enc_state.zero_grads()
        ae_state.zero_grads()
        pred_state.zero_grads()
        proj_state.zero_grads()

        # Forward: encoder (BT, IMG_DIM) → (BT, EMB)
        ENC.forward[BT](
            pixels_t,
            emb_t,
            enc_state.params_view(),
            enc_state.model_state_view(),
            enc_cache_t,
        )
        # Forward: action embedder (B, T*ACT) → (B, T*EMB)
        AE.forward[BATCH](
            actions_t,
            act_emb_t,
            ae_state.params_view(),
            ae_state.model_state_view(),
            ae_cache_t,
        )

        # ctx = emb[:, :H*EMB] + act_emb[:, :H*EMB]
        for b in range(BATCH):
            for i in range(H * EMB):
                ctx[b * H * EMB + i] = (
                    rebind[Scalar[dtype]](emb_bte_t[b, i])
                    + rebind[Scalar[dtype]](act_emb_t[b, i])
                )

        # Predictor + pred-proj
        PRED.forward[BATCH](
            ctx_t,
            pred_raw_t,
            pred_state.params_view(),
            pred_state.model_state_view(),
            pred_cache_t,
        )
        PROJ.forward[BATCH](
            pred_raw_t,
            pred_t,
            proj_state.params_view(),
            proj_state.model_state_view(),
            proj_cache_t,
        )

        # Loss: L = mean((pred - emb[:, n_preds*EMB:(n_preds+H)*EMB])^2)
        # Target is detached — we DO NOT backprop into emb_bte through this slice.
        var loss_scale = Float64(BATCH * H * EMB)
        var pred_loss: Float64 = 0.0
        var inv_scale = Scalar[dtype](2.0 / loss_scale)
        for b in range(BATCH):
            for i in range(H * EMB):
                var p = rebind[Scalar[dtype]](pred_t[b, i])
                var tgt = rebind[Scalar[dtype]](
                    emb_bte_t[b, N_PREDS * EMB + i]
                )
                var diff = p - tgt
                pred_loss += Float64(diff * diff)
                grad_pred[b * H * EMB + i] = inv_scale * diff
        pred_loss /= loss_scale

        if loss_first < 0.0:
            loss_first = pred_loss
            loss_ema = pred_loss
        else:
            loss_ema = 0.95 * loss_ema + 0.05 * pred_loss
        loss_last = pred_loss

        # Bind mutable views for grads so backward() can take mut refs.
        var enc_grads_v = enc_state.grads_view()
        var ae_grads_v = ae_state.grads_view()
        var pred_grads_v = pred_state.grads_view()
        var proj_grads_v = proj_state.grads_view()

        # Backward: pred_proj (BATCH, H*EMB) → grad_pred_raw
        PROJ.backward[BATCH](
            grad_pred_t,
            grad_pred_raw_t,
            proj_state.params_view(),
            proj_state.model_state_view(),
            proj_cache_t,
            proj_grads_v,
        )
        # Backward: predictor → grad_ctx
        PRED.backward[BATCH](
            grad_pred_raw_t,
            grad_ctx_t,
            pred_state.params_view(),
            pred_state.model_state_view(),
            pred_cache_t,
            pred_grads_v,
        )

        # Route grad_ctx to grad_emb (ctx_x slice) and grad_act_emb (ctx_a).
        # Detached target slice contributes zero gradient.
        _zero(grad_emb, BATCH * T * EMB)
        _zero(grad_act_emb, BATCH * T * EMB)
        for b in range(BATCH):
            for i in range(H * EMB):
                var g = rebind[Scalar[dtype]](grad_ctx_t[b, i])
                grad_emb[b * T * EMB + i] = g
                grad_act_emb[b * T * EMB + i] = g

        # Backward: action_embedder
        AE.backward[BATCH](
            grad_act_emb_t,
            grad_actions_t,
            ae_state.params_view(),
            ae_state.model_state_view(),
            ae_cache_t,
            ae_grads_v,
        )
        # Backward: encoder (BT view)
        ENC.backward[BT](
            grad_emb_bt_t,
            grad_pixels_t,
            enc_state.params_view(),
            enc_state.model_state_view(),
            enc_cache_t,
            enc_grads_v,
        )

        # Optimizer step on all four models.
        enc_state.optimizer_step()
        ae_state.optimizer_step()
        pred_state.optimizer_step()
        proj_state.optimizer_step()

        # Periodic logging.
        if step % log_every == 0 or step == num_steps - 1:
            var t_now = perf_counter_ns()
            var sps = Float64(step + 1) / (Float64(t_now - t0) / 1e9)
            print(
                "  step",
                step,
                "  loss=",
                pred_loss,
                "  ema=",
                loss_ema,
                "  it/s=",
                sps,
            )

    var t1 = perf_counter_ns()
    var total_s = Float64(t1 - t0) / 1e9
    print()
    print("Trained", num_steps, "steps in", total_s, "s")
    print("  loss_first =", loss_first)
    print("  loss_last  =", loss_last)
    print("  loss_ema   =", loss_ema)
    print(
        "  rel_drop   =",
        (loss_first - loss_last) / (loss_first + 1e-12),
    )

    # Free heap buffers.
    pixels.free()
    actions.free()
    emb.free()
    enc_cache.free()
    act_emb.free()
    ae_cache.free()
    ctx.free()
    pred_raw.free()
    pred_out.free()
    pred_cache.free()
    proj_cache.free()
    grad_pred.free()
    grad_pred_raw.free()
    grad_ctx.free()
    grad_emb.free()
    grad_act_emb.free()
    grad_actions.free()
    grad_pixels.free()
