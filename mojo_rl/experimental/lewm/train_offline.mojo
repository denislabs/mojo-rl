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
    Sequential,
    Linear,
    BatchNorm1D,
    Tokenwise,
)
from ...nn.model.autodiff_layers import GELU
from ...nn.composites import TransformerBlock, MultiHeadAttention
from ...nn.autodiff import AutoDiffChain
from ...nn.autodiff.primitives import SIGRegOp, BiasAdd
from .encoder import LeWMEncoder
from .action_embedder import ActionEmbedder
from .cond_block import (
    AdaLNMod,
    cond_block_forward,
    cond_block_backward,
)
from .pong_buffer import (
    PongBuffer,
    PONG_FRAME_BYTES,
    PONG_NUM_ACTIONS,
)
from std.math import sqrt


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
    SIG_NUM_PROJ: Int = 8,
    SIG_KNOTS: Int = 5,
](
    buffer_path: String,
    num_steps: Int,
    log_every: Int = 10,
    rng_seed: Int = 0xCAFE,
    lambda_sigreg: Float64 = 0.09,
) raises:
    """Run the offline JEPA training loop. CPU-only smoke variant.

    Loss: L = pred_mse + lambda_sigreg * sigreg_stat.
    Collapse probes (variance + off-diagonal Gram) logged every `log_every`.
    """

    comptime IMG_DIM: Int = IN_CH * IMG * IMG  # = ENC.IN_DIM
    comptime BT: Int = BATCH * T

    comptime ENC = LeWMEncoder[
        IN_CH,
        IMG,
        IMG,
        PATCH,
        HIDDEN,
        ENC_HEADS,
        ENC_LAYERS,
        N_PATCHES,
        EMB,
        2,
        PROJ_H,
    ]
    comptime AE = ActionEmbedder[T, ACT, SMOOTHED, EMB]
    # Position embedding: per-token learnable bias on the H-token context.
    # Same pattern as the encoder's ViT pos embed.
    comptime POS = AutoDiffChain[BiasAdd[H * EMB]]
    # Predictor: AdaLN-zero block (Linear[D, 3D] modulator + causal MSA over H tokens).
    # Replaces the additive ctx_x+ctx_a + non-conditional TransformerBlock stand-in.
    comptime ADALN = AdaLNMod[EMB]
    comptime MSA = MultiHeadAttention[EMB, PRED_HEADS, H, True]
    comptime _PredProjPerToken = Sequential[
        Linear[EMB, PROJ_H],
        BatchNorm1D[PROJ_H],
        GELU[PROJ_H],
        Linear[PROJ_H, EMB],
    ]
    comptime PROJ = Tokenwise[H, _PredProjPerToken]
    comptime SIG = SIGRegOp[EMB, T, SIG_NUM_PROJ, SIG_KNOTS]

    comptime BTH: Int = BATCH * H  # per-token effective batch for AdaLN ops

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
        " POS.PARAM=",
        POS.PARAM_SIZE,
        " ADALN.PARAM=",
        ADALN.PARAM_SIZE,
        " MSA.PARAM=",
        MSA.PARAM_SIZE,
        " PROJ.PARAM=",
        PROJ.PARAM_SIZE,
    )
    var total_params = (
        ENC.PARAM_SIZE
        + AE.PARAM_SIZE
        + POS.PARAM_SIZE
        + ADALN.PARAM_SIZE
        + MSA.PARAM_SIZE
        + PROJ.PARAM_SIZE
    )
    print("Total params:", total_params)

    # ------------------------------------------------------------------
    # Build 5 NetworkState[Model, Adam[]] groups
    #   enc, ae, adaln (zero-init for identity), msa, proj.
    # ------------------------------------------------------------------
    var enc_state = NetworkState[ENC, Adam[]]()
    var ae_state = NetworkState[AE, Adam[]]()
    var pos_state = NetworkState[POS, Adam[]]()
    var adaln_state = NetworkState[ADALN, Adam[]]()
    var msa_state = NetworkState[MSA, Adam[]]()
    var proj_state = NetworkState[PROJ, Adam[]]()
    enc_state.initialize[Xavier[]]()
    ae_state.initialize[Xavier[]]()
    pos_state.initialize[Xavier[]]()
    adaln_state.initialize[Xavier[]]()
    msa_state.initialize[Xavier[]]()
    proj_state.initialize[Xavier[]]()

    # AdaLN-zero: zero W and b after Xavier init so the block starts as identity.
    var adaln_params_ptr = adaln_state.params
    for i in range(ADALN.PARAM_SIZE):
        adaln_params_ptr[i] = Scalar[dtype](0)

    # Position embedding: zero-init so the block is exactly identity at step 0
    # (the encoder + AdaLN-zero contract already guarantees this; pos embed
    # zero-init keeps it). The pos embed then learns from gradient.
    var pos_params_ptr = pos_state.params
    for i in range(POS.PARAM_SIZE):
        pos_params_ptr[i] = Scalar[dtype](0)

    # ------------------------------------------------------------------
    # Heap-allocated activation/cache/gradient buffers (reused per step)
    # ------------------------------------------------------------------
    var pixels = alloc[Scalar[dtype]](BT * IMG_DIM)
    var actions = alloc[Scalar[dtype]](BATCH * T * ACT)

    var emb = alloc[Scalar[dtype]](BT * EMB)
    var enc_cache = alloc[Scalar[dtype]](BT * ENC.CACHE_SIZE)

    var act_emb = alloc[Scalar[dtype]](BATCH * T * EMB)
    var ae_cache = alloc[Scalar[dtype]](BATCH * AE.CACHE_SIZE)

    # cond_block input/output (BTH = BATCH * H, EMB).
    var x_prev = alloc[Scalar[dtype]](BTH * EMB)
    var x_prev_pe = alloc[Scalar[dtype]](BTH * EMB)  # x_prev + pos_emb
    var pos_cache = alloc[Scalar[dtype]](BATCH * POS.CACHE_SIZE)
    var c_in = alloc[Scalar[dtype]](BTH * EMB)
    var pred_raw = alloc[Scalar[dtype]](BTH * EMB)
    var pred_out = alloc[Scalar[dtype]](BATCH * H * EMB)
    var proj_cache = alloc[Scalar[dtype]](BATCH * PROJ.CACHE_SIZE)

    # cond_block caches.
    var silu_cache = alloc[Scalar[dtype]](BTH * EMB)
    var adaln_cache = alloc[Scalar[dtype]](BTH * ADALN.CACHE_SIZE)
    var ln_cache = alloc[Scalar[dtype]](BTH * (EMB + 1))
    var mod_cache = alloc[Scalar[dtype]](BTH * 2 * EMB)
    var msa_cache = alloc[Scalar[dtype]](BATCH * MSA.CACHE_SIZE)
    var gate_cache = alloc[Scalar[dtype]](BTH * 2 * EMB)
    # cond_block intermediate (kept across forward/backward). 6D = full AdaLN
    # output (MSA scale/shift/gate + MLP scale/shift/gate). CPU trainer only
    # writes the MSA half ([0:3D]); the MLP half is left zero (see comment at
    # cond_block_backward call below).
    var raw_mod = alloc[Scalar[dtype]](BTH * 6 * EMB)
    # cond_block forward scratch.
    var silu_buf = alloc[Scalar[dtype]](BTH * EMB)
    var ln_out_buf = alloc[Scalar[dtype]](BTH * EMB)
    var mod_inp_buf = alloc[Scalar[dtype]](BTH * 3 * EMB)
    var mod_x_buf = alloc[Scalar[dtype]](BTH * EMB)
    var attn_out_buf = alloc[Scalar[dtype]](BTH * EMB)
    var gate_inp_buf = alloc[Scalar[dtype]](BTH * 3 * EMB)
    # cond_block backward scratch.
    var sgg = alloc[Scalar[dtype]](BTH * 3 * EMB)
    var sgao = alloc[Scalar[dtype]](BTH * EMB)
    var sgmx = alloc[Scalar[dtype]](BTH * EMB)
    var sgmi = alloc[Scalar[dtype]](BTH * 3 * EMB)
    var sglnout = alloc[Scalar[dtype]](BTH * EMB)
    var sglnin = alloc[Scalar[dtype]](BTH * EMB)
    # sgrm = grad of raw_mod (6D wide). MLP slots [3D:6D] must be zeroed each
    # step so the LayerNorm vjp inside cond_block_backward doesn't see stale
    # values from the previous iteration.
    var sgrm = alloc[Scalar[dtype]](BTH * 6 * EMB)
    var sgsc = alloc[Scalar[dtype]](BTH * EMB)

    var grad_pred = alloc[Scalar[dtype]](BATCH * H * EMB)
    var grad_pred_raw = alloc[Scalar[dtype]](BTH * EMB)
    var grad_x_prev = alloc[Scalar[dtype]](BTH * EMB)
    var grad_x_prev_pe = alloc[Scalar[dtype]](BTH * EMB)  # before POS.backward
    var grad_c = alloc[Scalar[dtype]](BTH * EMB)
    var grad_emb = alloc[Scalar[dtype]](BATCH * T * EMB)
    var grad_act_emb = alloc[Scalar[dtype]](BATCH * T * EMB)
    var grad_actions = alloc[Scalar[dtype]](BATCH * T * ACT)
    var grad_pixels = alloc[Scalar[dtype]](BT * IMG_DIM)

    # SIGReg-specific buffers.
    var sigreg_out = alloc[Scalar[dtype]](BATCH)
    var sigreg_cache = alloc[Scalar[dtype]](BATCH * SIG.CACHE_SIZE)
    var grad_sigreg_out = alloc[Scalar[dtype]](BATCH)
    var sigreg_grad_emb = alloc[Scalar[dtype]](BATCH * T * EMB)

    # ------------------------------------------------------------------
    # Layout tensors over the heap buffers
    # ------------------------------------------------------------------
    var pixels_t = LayoutTensor[
        dtype, Layout.row_major(BT, IMG_DIM), MutAnyOrigin
    ](pixels)
    var actions_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * ACT), MutAnyOrigin
    ](actions)

    var emb_t = LayoutTensor[dtype, Layout.row_major(BT, EMB), MutAnyOrigin](
        emb
    )
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

    # cond_block input views: (BTH, EMB) per-token, used by AdaLN modulator.
    var x_prev_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](x_prev)
    # POS view of x_prev as (BATCH, H*EMB) — same buffer, different layout.
    var x_prev_bh_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ](x_prev)
    var x_prev_pe_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](x_prev_pe)
    var x_prev_pe_bh_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ](x_prev_pe)
    var pos_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, POS.CACHE_SIZE), MutAnyOrigin
    ](pos_cache)
    var c_in_t = LayoutTensor[dtype, Layout.row_major(BTH, EMB), MutAnyOrigin](
        c_in
    )
    # pred_raw is the cond_block output; viewed both (BTH, EMB) and (B, H*EMB).
    var pred_raw_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](pred_raw)
    var pred_raw_bh_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ](pred_raw)
    var pred_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ](pred_out)
    var proj_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, PROJ.CACHE_SIZE), MutAnyOrigin
    ](proj_cache)

    # cond_block caches.
    var silu_cache_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](silu_cache)
    var adaln_cache_t = LayoutTensor[
        dtype, Layout.row_major(BTH, ADALN.CACHE_SIZE), MutAnyOrigin
    ](adaln_cache)
    var ln_cache_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB + 1), MutAnyOrigin
    ](ln_cache)
    var mod_cache_t = LayoutTensor[
        dtype, Layout.row_major(BTH, 2 * EMB), MutAnyOrigin
    ](mod_cache)
    var msa_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, MSA.CACHE_SIZE), MutAnyOrigin
    ](msa_cache)
    var gate_cache_t = LayoutTensor[
        dtype, Layout.row_major(BTH, 2 * EMB), MutAnyOrigin
    ](gate_cache)
    var raw_mod_t = LayoutTensor[
        dtype, Layout.row_major(BTH, 6 * EMB), MutAnyOrigin
    ](raw_mod)

    # cond_block forward scratch views.
    var silu_buf_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](silu_buf)
    var ln_out_buf_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](ln_out_buf)
    var mod_inp_buf_t = LayoutTensor[
        dtype, Layout.row_major(BTH, 3 * EMB), MutAnyOrigin
    ](mod_inp_buf)
    var mod_x_buf_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](mod_x_buf)
    var attn_out_buf_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](attn_out_buf)
    var gate_inp_buf_t = LayoutTensor[
        dtype, Layout.row_major(BTH, 3 * EMB), MutAnyOrigin
    ](gate_inp_buf)

    # cond_block backward scratch views.
    var sgg_t = LayoutTensor[
        dtype, Layout.row_major(BTH, 3 * EMB), MutAnyOrigin
    ](sgg)
    var sgao_t = LayoutTensor[dtype, Layout.row_major(BTH, EMB), MutAnyOrigin](
        sgao
    )
    var sgmx_t = LayoutTensor[dtype, Layout.row_major(BTH, EMB), MutAnyOrigin](
        sgmx
    )
    var sgmi_t = LayoutTensor[
        dtype, Layout.row_major(BTH, 3 * EMB), MutAnyOrigin
    ](sgmi)
    var sglnout_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](sglnout)
    var sglnin_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](sglnin)
    var sgrm_t = LayoutTensor[
        dtype, Layout.row_major(BTH, 6 * EMB), MutAnyOrigin
    ](sgrm)
    var sgsc_t = LayoutTensor[dtype, Layout.row_major(BTH, EMB), MutAnyOrigin](
        sgsc
    )

    var grad_pred_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ](grad_pred)
    var grad_pred_raw_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](grad_pred_raw)
    var grad_x_prev_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](grad_x_prev)
    var grad_x_prev_bh_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ](grad_x_prev)
    var grad_x_prev_pe_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](grad_x_prev_pe)
    var grad_x_prev_pe_bh_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ](grad_x_prev_pe)
    var grad_c_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](grad_c)
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

    var sigreg_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
    ](sigreg_out)
    var sigreg_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, SIG.CACHE_SIZE), MutAnyOrigin
    ](sigreg_cache)
    var grad_sigreg_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
    ](grad_sigreg_out)
    var sigreg_grad_emb_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * EMB), MutAnyOrigin
    ](sigreg_grad_emb)

    # Empty params view for SIGReg (PARAM_SIZE = 0).
    var empty_params = LayoutTensor[dtype, Layout.row_major(0), MutAnyOrigin](
        UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=0)
    )
    var empty_grad_params = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=0))

    # ------------------------------------------------------------------
    # Step loop
    # ------------------------------------------------------------------
    var loss_ema: Float64 = 0.0
    var pred_ema: Float64 = 0.0
    var sigreg_ema: Float64 = 0.0
    var var_min_ema: Float64 = 0.0
    var var_mean_ema: Float64 = 0.0
    var gram_ema: Float64 = 0.0
    var loss_first: Float64 = -1.0
    var loss_last: Float64 = 0.0
    var t0 = perf_counter_ns()
    var inv_lambda_per_b = Scalar[dtype](lambda_sigreg / Float64(BATCH))

    for step in range(num_steps):
        # Sample a fresh batch.
        buf.sample_batch_fp32(BATCH, T, pixels, actions)

        # Zero grads
        enc_state.zero_grads()
        ae_state.zero_grads()
        pos_state.zero_grads()
        adaln_state.zero_grads()
        msa_state.zero_grads()
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

        # Slice first H tokens of emb + act_emb into per-token (BTH, EMB) views.
        # emb storage layout is (BATCH, T, EMB) row-major.
        for b in range(BATCH):
            for t in range(H):
                for i in range(EMB):
                    x_prev[(b * H + t) * EMB + i] = rebind[Scalar[dtype]](
                        emb_bte_t[b, t * EMB + i]
                    )
                    c_in[(b * H + t) * EMB + i] = rebind[Scalar[dtype]](
                        act_emb_t[b, t * EMB + i]
                    )

        # Position embedding: x_prev_pe = x_prev + pos_emb (broadcast over BATCH).
        # POS is AutoDiffChain[BiasAdd[H*EMB]] operating on (BATCH, H*EMB).
        POS.forward[BATCH](
            x_prev_bh_t,
            x_prev_pe_bh_t,
            pos_state.params_view(),
            pos_state.model_state_view(),
            pos_cache_t,
        )

        # AdaLN-zero conditional block: pred_raw = block(x_prev_pe, c).
        cond_block_forward[BATCH, H, EMB, PRED_HEADS](
            x_prev_pe_t,
            c_in_t,
            adaln_state.params_view(),
            adaln_state.model_state_view(),
            msa_state.params_view(),
            msa_state.model_state_view(),
            pred_raw_t,
            silu_cache_t,
            adaln_cache_t,
            ln_cache_t,
            mod_cache_t,
            msa_cache_t,
            gate_cache_t,
            raw_mod_t,
            silu_buf_t,
            ln_out_buf_t,
            mod_inp_buf_t,
            mod_x_buf_t,
            attn_out_buf_t,
            gate_inp_buf_t,
        )

        # PredProj: Tokenwise[H, ...] on (BATCH, H*EMB) view.
        PROJ.forward[BATCH](
            pred_raw_bh_t,
            pred_t,
            proj_state.params_view(),
            proj_state.model_state_view(),
            proj_cache_t,
        )

        # SIGReg forward over emb_bte (B, T*EMB). Output is the same statistic
        # replicated across BATCH slots.
        SIG.eval[BATCH](emb_bte_t, sigreg_out_t, empty_params, sigreg_cache_t)
        var sigreg_loss = Float64(sigreg_out[0])

        # Loss: L_pred = mean((pred - emb[:, n_preds*EMB:(n_preds+H)*EMB])^2).
        # Target slice is DETACHED — we don't backprop into emb_bte through it.
        var loss_scale = Float64(BATCH * H * EMB)
        var pred_loss: Float64 = 0.0
        var inv_scale = Scalar[dtype](2.0 / loss_scale)
        for b in range(BATCH):
            for i in range(H * EMB):
                var p = rebind[Scalar[dtype]](pred_t[b, i])
                var tgt = rebind[Scalar[dtype]](emb_bte_t[b, N_PREDS * EMB + i])
                var diff = p - tgt
                pred_loss += Float64(diff * diff)
                grad_pred[b * H * EMB + i] = inv_scale * diff
        pred_loss /= loss_scale

        var total_loss = pred_loss + lambda_sigreg * sigreg_loss

        # ----- Collapse probes on emb (BT, EMB) -----
        # Per-dim variance across BT samples; min/mean over EMB dims.
        var var_min: Float64 = 1e30
        var var_mean: Float64 = 0.0
        for d in range(EMB):
            var s: Float64 = 0.0
            var ss: Float64 = 0.0
            for bt in range(BT):
                var v = Float64(rebind[Scalar[dtype]](emb_t[bt, d]))
                s += v
                ss += v * v
            var mean_d = s / Float64(BT)
            var var_d = (ss / Float64(BT)) - mean_d * mean_d
            if var_d < var_min:
                var_min = var_d
            var_mean += var_d
        var_mean /= Float64(EMB)
        # Off-diagonal mean |cosine similarity|: normalize each BT embedding
        # to unit norm, then average |<x_i, x_j>| over i != j.
        var gram_off: Float64 = 0.0
        var gram_n: Int = 0
        for i in range(BT):
            var ni: Float64 = 0.0
            for d in range(EMB):
                var v = Float64(rebind[Scalar[dtype]](emb_t[i, d]))
                ni += v * v
            ni = sqrt(ni + 1e-12)
            for j in range(i + 1, BT):
                var nj: Float64 = 0.0
                var dot: Float64 = 0.0
                for d in range(EMB):
                    var vi = Float64(rebind[Scalar[dtype]](emb_t[i, d]))
                    var vj = Float64(rebind[Scalar[dtype]](emb_t[j, d]))
                    nj += vj * vj
                    dot += vi * vj
                nj = sqrt(nj + 1e-12)
                var c = dot / (ni * nj)
                if c < 0.0:
                    c = -c
                gram_off += c
                gram_n += 1
        gram_off /= Float64(gram_n)

        if loss_first < 0.0:
            loss_first = total_loss
            loss_ema = total_loss
            pred_ema = pred_loss
            sigreg_ema = sigreg_loss
            var_min_ema = var_min
            var_mean_ema = var_mean
            gram_ema = gram_off
        else:
            loss_ema = 0.95 * loss_ema + 0.05 * total_loss
            pred_ema = 0.95 * pred_ema + 0.05 * pred_loss
            sigreg_ema = 0.95 * sigreg_ema + 0.05 * sigreg_loss
            var_min_ema = 0.95 * var_min_ema + 0.05 * var_min
            var_mean_ema = 0.95 * var_mean_ema + 0.05 * var_mean
            gram_ema = 0.95 * gram_ema + 0.05 * gram_off
        loss_last = total_loss

        # Bind mutable views for grads so backward() can take mut refs.
        var enc_grads_v = enc_state.grads_view()
        var ae_grads_v = ae_state.grads_view()
        var pos_grads_v = pos_state.grads_view()
        var adaln_grads_v = adaln_state.grads_view()
        var msa_grads_v = msa_state.grads_view()
        var proj_grads_v = proj_state.grads_view()

        # Backward: pred_proj (BATCH, H*EMB) → grad_pred_raw (BH, EMB).
        # PROJ writes (BATCH, H*EMB); we view that same buffer as (BTH, EMB)
        # below for cond_block_backward.
        var grad_pred_raw_bh_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
        ](grad_pred_raw)
        PROJ.backward[BATCH](
            grad_pred_t,
            grad_pred_raw_bh_t,
            proj_state.params_view(),
            proj_state.model_state_view(),
            proj_cache_t,
            proj_grads_v,
        )

        # AdaLN-zero block backward → grad_x_prev_pe (BTH, EMB).
        # Pre-zero MLP slots [3D:6D] of sgrm: CPU backward only writes MSA
        # slots, so the MLP half would otherwise carry stale grads from the
        # previous step into the LayerNorm vjp.
        for b in range(BTH):
            for i in range(3 * EMB):
                sgrm_t[b, 3 * EMB + i] = Scalar[dtype](0)
        cond_block_backward[BATCH, H, EMB, PRED_HEADS](
            grad_pred_raw_t,
            adaln_state.params_view(),
            adaln_state.model_state_view(),
            msa_state.params_view(),
            msa_state.model_state_view(),
            silu_cache_t,
            adaln_cache_t,
            ln_cache_t,
            mod_cache_t,
            msa_cache_t,
            gate_cache_t,
            grad_x_prev_pe_t,
            grad_c_t,
            adaln_grads_v,
            msa_grads_v,
            sgg_t,
            sgao_t,
            sgmx_t,
            sgmi_t,
            sglnout_t,
            sglnin_t,
            sgrm_t,
            sgsc_t,
        )

        # POS.backward: grad_x_prev_pe → grad_x_prev + pos_emb gradient accumulator.
        POS.backward[BATCH](
            grad_x_prev_pe_bh_t,
            grad_x_prev_bh_t,
            pos_state.params_view(),
            pos_state.model_state_view(),
            pos_cache_t,
            pos_grads_v,
        )

        # Route grad_x_prev → grad_emb's first H tokens, grad_c → grad_act_emb's.
        # Detached target slice contributes zero gradient (rest of the buffer).
        _zero(grad_emb, BATCH * T * EMB)
        _zero(grad_act_emb, BATCH * T * EMB)
        for b in range(BATCH):
            for t in range(H):
                for i in range(EMB):
                    grad_emb[b * T * EMB + t * EMB + i] = grad_x_prev[
                        (b * H + t) * EMB + i
                    ]
                    grad_act_emb[b * T * EMB + t * EMB + i] = grad_c[
                        (b * H + t) * EMB + i
                    ]

        # SIGReg vjp: grad_seed per slot = λ/B (sums to λ → matches λ·stat).
        for b in range(BATCH):
            grad_sigreg_out[b] = inv_lambda_per_b
        SIG.vjp[BATCH](
            grad_sigreg_out_t,
            sigreg_grad_emb_t,
            empty_params,
            sigreg_cache_t,
            empty_grad_params,
        )
        # Accumulate SIGReg's contribution into grad_emb (additive — pred-path
        # grads were just written above).
        for i in range(BATCH * T * EMB):
            grad_emb[i] = grad_emb[i] + sigreg_grad_emb[i]

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

        # Optimizer step on all six models.
        enc_state.optimizer_step()
        ae_state.optimizer_step()
        pos_state.optimizer_step()
        adaln_state.optimizer_step()
        msa_state.optimizer_step()
        proj_state.optimizer_step()

        # Periodic logging.
        if step % log_every == 0 or step == num_steps - 1:
            var t_now = perf_counter_ns()
            var sps = Float64(step + 1) / (Float64(t_now - t0) / 1e9)
            print(
                "  step",
                step,
                " L=",
                total_loss,
                " pred=",
                pred_ema,
                " sig=",
                sigreg_ema,
                " var_min=",
                var_min_ema,
                " var_mean=",
                var_mean_ema,
                " gram=",
                gram_ema,
                " it/s=",
                sps,
            )

    var t1 = perf_counter_ns()
    var total_s = Float64(t1 - t0) / 1e9
    print()
    print("Trained", num_steps, "steps in", total_s, "s")
    print("  loss_first =", loss_first)
    print("  loss_last  =", loss_last)
    print("  loss_ema   =", loss_ema)
    print("  pred_ema   =", pred_ema)
    print("  sigreg_ema =", sigreg_ema)
    print(
        "  rel_drop   =",
        (loss_first - loss_last) / (loss_first + 1e-12),
    )
    print()
    print("Collapse probes (EMA across the run):")
    print("  var_min  =", var_min_ema, " (want > 0.1 for healthy spread)")
    print("  var_mean =", var_mean_ema)
    print("  gram_off =", gram_ema, " (want close to 0, bounded < ~0.5)")

    # Free heap buffers.
    pixels.free()
    actions.free()
    emb.free()
    enc_cache.free()
    act_emb.free()
    ae_cache.free()
    x_prev.free()
    x_prev_pe.free()
    pos_cache.free()
    c_in.free()
    pred_raw.free()
    pred_out.free()
    proj_cache.free()
    silu_cache.free()
    adaln_cache.free()
    ln_cache.free()
    mod_cache.free()
    msa_cache.free()
    gate_cache.free()
    raw_mod.free()
    silu_buf.free()
    ln_out_buf.free()
    mod_inp_buf.free()
    mod_x_buf.free()
    attn_out_buf.free()
    gate_inp_buf.free()
    sgg.free()
    sgao.free()
    sgmx.free()
    sgmi.free()
    sglnout.free()
    sglnin.free()
    sgrm.free()
    sgsc.free()
    grad_pred.free()
    grad_pred_raw.free()
    grad_x_prev.free()
    grad_x_prev_pe.free()
    grad_c.free()
    grad_emb.free()
    grad_act_emb.free()
    grad_actions.free()
    grad_pixels.free()
    sigreg_out.free()
    sigreg_cache.free()
    grad_sigreg_out.free()
    sigreg_grad_emb.free()
