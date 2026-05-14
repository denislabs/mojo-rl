"""LeWM offline training loop — GPU (Phase 3 GPU port).

Mirrors `train_offline.mojo` but runs all 6 model groups on GPU:
  - ENC (LeWMEncoder)        — Conv2D-based ViT, full GPU path
  - AE  (ActionEmbedder)
  - POS (AutoDiffChain[BiasAdd[H*EMB]])
  - ADALN (Linear[EMB, 3*EMB], zero-init for AdaLN-zero identity)
  - MSA (MultiHeadAttention[EMB, PRED_HEADS, H, causal=True])
  - PROJ (Tokenwise[H, Sequential[Linear, BatchNorm1D, GELU, Linear]])

Differences from the CPU trainer:
  1. SIGReg is dropped (no GPU implementation; probes carry the
     regularizer signal).
  2. Loss + grad_pred computed on host (BATCH * H * EMB fp32 round-trip
     per step, ~ a few KB — negligible).
  3. Collapse probes downloaded each `log_every` steps.

Buffer sampling stays on host (PongBuffer is CPU-resident); each step
host-samples (pixels, actions) and `enqueue_copy`s them to device.
"""

from std.math import abs, sqrt, ceildiv
from std.memory import alloc
from std.random import seed as _set_seed
from std.time import perf_counter_ns
from std.gpu import global_idx
from layout import Layout, LayoutTensor
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from ...nn.constants import dtype
from ...nn.training import NetworkState, GPUNetworkState
from ...nn.optimizer import Adam
from ...nn.initializer import Xavier
from ...nn.model import (
    Sequential, Linear, BatchNorm1D, Tokenwise,
)
from ...nn.model.autodiff_layers import GELU
from ...nn.composites import TransformerBlock, MultiHeadAttention
from ...nn.autodiff import AutoDiffChain
from ...nn.autodiff.primitives import BiasAdd, SIGRegOp
from .encoder import LeWMEncoder
from .action_embedder import ActionEmbedder
from .cond_block import (
    AdaLNMod,
    CondMLP,
    cond_block_forward_gpu,
    cond_block_backward_gpu,
    cb_accum_kernel,
)
from .pong_buffer import (
    PongBuffer,
    PONG_FRAME_BYTES,
    PONG_NUM_ACTIONS,
)


@always_inline
def _max_int(a: Int, b: Int) -> Int:
    return a if a > b else b


comptime TPB_X = 4
comptime TPB_Y = 4
comptime TPB_Z = 16


# Slice first H tokens out of a (BATCH * T, EMB) source into a (BATCH * H, EMB)
# destination. Used to extract the predictor's H-token context from both the
# encoder embeddings and the action embeddings.
def slice_h_kernel[
    BATCH: Int, T: Int, H: Int, EMB: Int,
](
    src: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, EMB), MutAnyOrigin
    ],
    dst: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, EMB), MutAnyOrigin
    ],
):
    var b = Int(global_idx.x)
    var t_idx = Int(global_idx.y)
    var d_idx = Int(global_idx.z)
    if b < BATCH and t_idx < H and d_idx < EMB:
        dst[b * H + t_idx, d_idx] = src[b * T + t_idx, d_idx]


# Scatter a (BATCH * H, EMB) source into the first H tokens of a (BATCH * T, EMB)
# destination. Used to route gradient slices back to grad_emb / grad_act_emb.
def scatter_h_kernel[
    BATCH: Int, T: Int, H: Int, EMB: Int,
](
    src: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, EMB), MutAnyOrigin
    ],
    dst: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, EMB), MutAnyOrigin
    ],
):
    var b = Int(global_idx.x)
    var t_idx = Int(global_idx.y)
    var d_idx = Int(global_idx.z)
    if b < BATCH and t_idx < H and d_idx < EMB:
        dst[b * T + t_idx, d_idx] = src[b * H + t_idx, d_idx]


# Scatter -grad_pred into the target slice of grad_emb so the target path is
# no longer stop-grad. Math: pred_loss = (pred - tgt)^2/N → d/d tgt = -grad_pred.
# The H "target tokens" inside emb live at b * T*EMB + N_PREDS*EMB + i for
# i in [0, H*EMB).
def scatter_target_neg_kernel[
    BATCH: Int, T: Int, H: Int, N_PREDS: Int, EMB: Int,
](
    grad_pred_bh_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ],
    grad_emb_bt_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, T * EMB), MutAnyOrigin
    ],
):
    var b = Int(global_idx.x)
    var i = Int(global_idx.y)
    if b < BATCH and i < H * EMB:
        var v = grad_pred_bh_t[b, i]
        grad_emb_bt_t[b, N_PREDS * EMB + i] = -v


# Elementwise: dst[b, i] += src[b, i]. Used to fold SIGReg's vjp into grad_emb.
def accumulate_emb_kernel[
    BATCH: Int, T: Int, EMB: Int,
](
    src: LayoutTensor[
        dtype, Layout.row_major(BATCH, T * EMB), MutAnyOrigin
    ],
    dst: LayoutTensor[
        dtype, Layout.row_major(BATCH, T * EMB), MutAnyOrigin
    ],
):
    var b = Int(global_idx.x)
    var i = Int(global_idx.y)
    if b < BATCH and i < T * EMB:
        dst[b, i] = dst[b, i] + src[b, i]


def train_lewm_offline_gpu[
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
    DEPTH: Int = 1,
    SIG_NUM_PROJ: Int = 1024,
    SIG_KNOTS: Int = 17,
](
    buffer_path: String,
    num_steps: Int,
    log_every: Int = 100,
    rng_seed: Int = 0xCAFE,
    lambda_sigreg: Float64 = 0.09,
) raises:
    """GPU offline JEPA trainer with SIGReg + full end-to-end gradient flow.

    Defaults match the LeWM paper: SIG_NUM_PROJ=1024, SIG_KNOTS=17,
    lambda_sigreg=0.09, DEPTH=1 (paper uses 6 stacked dual-branch blocks
    via `predictor.depth: 6` in config/train/lewm.yaml). Set DEPTH=6 for
    full paper parity.

    With SIGReg + lambda > 0, target stop-grad is dropped
    (`scatter_target_neg_kernel`) so gradients flow through both sides of
    the MSE — SIGReg is the only thing preventing collapse.
    """
    comptime assert DEPTH >= 1, "DEPTH must be >= 1"

    comptime IMG_DIM: Int = IN_CH * IMG * IMG
    comptime BT: Int = BATCH * T
    comptime BTH: Int = BATCH * H

    comptime ENC = LeWMEncoder[
        IN_CH, IMG, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, N_PATCHES,
        EMB, 2, PROJ_H,
    ]
    comptime AE = ActionEmbedder[T, ACT, SMOOTHED, EMB]
    comptime POS = AutoDiffChain[BiasAdd[H * EMB]]
    comptime ADALN = AdaLNMod[EMB]
    comptime MSA = MultiHeadAttention[EMB, PRED_HEADS, H, True]
    comptime MLP = CondMLP[EMB, PRED_FF]
    comptime _PredProjPerToken = Sequential[
        Linear[EMB, PROJ_H],
        BatchNorm1D[PROJ_H],
        GELU[PROJ_H],
        Linear[PROJ_H, EMB],
    ]
    comptime PROJ = Tokenwise[H, _PredProjPerToken]
    comptime SIG = SIGRegOp[EMB, T, SIG_NUM_PROJ, SIG_KNOTS]
    comptime SIG_WS_SIZE = SIG.workspace_size_for[BATCH]()

    _set_seed(rng_seed)

    var buf = PongBuffer.load(buffer_path)
    print("Loaded buffer:", buf.n_frames, "frames from", buffer_path)
    print(
        "Models — ENC.PARAM=", ENC.PARAM_SIZE,
        " AE.PARAM=", AE.PARAM_SIZE,
        " POS.PARAM=", POS.PARAM_SIZE,
        " ADALN.PARAM=", ADALN.PARAM_SIZE,
        " MSA.PARAM=", MSA.PARAM_SIZE,
        " MLP.PARAM=", MLP.PARAM_SIZE,
        " PROJ.PARAM=", PROJ.PARAM_SIZE,
        " DEPTH=", DEPTH,
    )
    var total_params = (
        ENC.PARAM_SIZE + AE.PARAM_SIZE + POS.PARAM_SIZE + PROJ.PARAM_SIZE
        + DEPTH * (ADALN.PARAM_SIZE + MSA.PARAM_SIZE + MLP.PARAM_SIZE)
    )
    print("Total params (incl. DEPTH stack):", total_params)
    print(
        "Workspaces/sample — ENC=", ENC.WORKSPACE_SIZE_PER_SAMPLE,
        " AE=", AE.WORKSPACE_SIZE_PER_SAMPLE,
        " POS=", POS.WORKSPACE_SIZE_PER_SAMPLE,
        " ADALN=", ADALN.WORKSPACE_SIZE_PER_SAMPLE,
        " MSA=", MSA.WORKSPACE_SIZE_PER_SAMPLE,
        " MLP=", MLP.WORKSPACE_SIZE_PER_SAMPLE,
        " PROJ=", PROJ.WORKSPACE_SIZE_PER_SAMPLE,
    )

    var ctx = DeviceContext()

    # ------------------------------------------------------------------
    # Init on CPU, upload to GPU.
    # ------------------------------------------------------------------
    # Shared (single-instance) models.
    var cpu_enc = NetworkState[ENC, Adam[]]()
    var cpu_ae = NetworkState[AE, Adam[]]()
    var cpu_pos = NetworkState[POS, Adam[]]()
    var cpu_proj = NetworkState[PROJ, Adam[]]()
    cpu_enc.initialize[Xavier[]]()
    cpu_ae.initialize[Xavier[]]()
    cpu_pos.initialize[Xavier[]]()
    cpu_proj.initialize[Xavier[]]()
    for i in range(POS.PARAM_SIZE):
        cpu_pos.params[i] = Scalar[dtype](0)

    var enc_state = GPUNetworkState[ENC, Adam[]](ctx)
    var ae_state = GPUNetworkState[AE, Adam[]](ctx)
    var pos_state = GPUNetworkState[POS, Adam[]](ctx)
    var proj_state = GPUNetworkState[PROJ, Adam[]](ctx)
    enc_state.upload_from(cpu_enc, ctx)
    ae_state.upload_from(cpu_ae, ctx)
    pos_state.upload_from(cpu_pos, ctx)
    proj_state.upload_from(cpu_proj, ctx)

    # Per-layer cond_block models — DEPTH copies of ADALN, MSA, MLP each.
    var cpu_adalns = List[NetworkState[ADALN, Adam[]]](capacity=DEPTH)
    var cpu_msas = List[NetworkState[MSA, Adam[]]](capacity=DEPTH)
    var cpu_mlps = List[NetworkState[MLP, Adam[]]](capacity=DEPTH)
    for d in range(DEPTH):
        var ca = NetworkState[ADALN, Adam[]]()
        ca.initialize[Xavier[]]()
        # Zero-init AdaLN for AdaLN-zero identity at step 0.
        for i in range(ADALN.PARAM_SIZE):
            ca.params[i] = Scalar[dtype](0)
        cpu_adalns.append(ca^)

        var cm = NetworkState[MSA, Adam[]]()
        cm.initialize[Xavier[]]()
        cpu_msas.append(cm^)

        var cf = NetworkState[MLP, Adam[]]()
        cf.initialize[Xavier[]]()
        cpu_mlps.append(cf^)

    var adaln_states = List[GPUNetworkState[ADALN, Adam[]]](capacity=DEPTH)
    var msa_states = List[GPUNetworkState[MSA, Adam[]]](capacity=DEPTH)
    var mlp_states = List[GPUNetworkState[MLP, Adam[]]](capacity=DEPTH)
    for d in range(DEPTH):
        var ga = GPUNetworkState[ADALN, Adam[]](ctx)
        ga.upload_from(cpu_adalns[d], ctx)
        adaln_states.append(ga^)
        var gm = GPUNetworkState[MSA, Adam[]](ctx)
        gm.upload_from(cpu_msas[d], ctx)
        msa_states.append(gm^)
        var gf = GPUNetworkState[MLP, Adam[]](ctx)
        gf.upload_from(cpu_mlps[d], ctx)
        mlp_states.append(gf^)

    # ------------------------------------------------------------------
    # Allocate device buffers for activations / caches / grads / scratch.
    # All sizes are comptime; we allocate at least 1 element to keep
    # DeviceBuffer construction valid for zero-CACHE/WS ops.
    # ------------------------------------------------------------------
    var pixels_buf = ctx.enqueue_create_buffer[dtype](BT * IMG_DIM)
    var actions_buf = ctx.enqueue_create_buffer[dtype](BATCH * T * ACT)
    var emb_buf = ctx.enqueue_create_buffer[dtype](BT * EMB)
    var enc_cache_buf = ctx.enqueue_create_buffer[dtype](BT * ENC.CACHE_SIZE)
    var enc_ws_buf = ctx.enqueue_create_buffer[dtype](
        _max_int(1, BT * ENC.WORKSPACE_SIZE_PER_SAMPLE)
    )

    var act_emb_buf = ctx.enqueue_create_buffer[dtype](BATCH * T * EMB)
    var ae_cache_buf = ctx.enqueue_create_buffer[dtype](BATCH * AE.CACHE_SIZE)
    var ae_ws_buf = ctx.enqueue_create_buffer[dtype](
        _max_int(1, BATCH * AE.WORKSPACE_SIZE_PER_SAMPLE)
    )

    var x_prev_buf = ctx.enqueue_create_buffer[dtype](BTH * EMB)
    var x_prev_pe_buf = ctx.enqueue_create_buffer[dtype](BTH * EMB)
    var pos_cache_buf = ctx.enqueue_create_buffer[dtype](
        _max_int(1, BATCH * POS.CACHE_SIZE)
    )
    var pos_ws_buf = ctx.enqueue_create_buffer[dtype](
        _max_int(1, BATCH * POS.WORKSPACE_SIZE_PER_SAMPLE)
    )
    var c_in_buf = ctx.enqueue_create_buffer[dtype](BTH * EMB)

    var pred_raw_buf = ctx.enqueue_create_buffer[dtype](BTH * EMB)
    var pred_out_buf = ctx.enqueue_create_buffer[dtype](BATCH * H * EMB)
    var proj_cache_buf = ctx.enqueue_create_buffer[dtype](BATCH * PROJ.CACHE_SIZE)
    var proj_ws_buf = ctx.enqueue_create_buffer[dtype](
        _max_int(1, BATCH * PROJ.WORKSPACE_SIZE_PER_SAMPLE)
    )

    # cond_block caches — MSA branch (DEPTH-fold, sliced per layer).
    var silu_cache_buf = ctx.enqueue_create_buffer[dtype](BTH * EMB * DEPTH)
    var adaln_cache_buf = ctx.enqueue_create_buffer[dtype](
        BTH * ADALN.CACHE_SIZE * DEPTH
    )
    var ln1_cache_buf = ctx.enqueue_create_buffer[dtype](BTH * (EMB + 1) * DEPTH)
    var mod1_cache_buf = ctx.enqueue_create_buffer[dtype](BTH * 2 * EMB * DEPTH)
    var msa_cache_buf = ctx.enqueue_create_buffer[dtype](
        BATCH * MSA.CACHE_SIZE * DEPTH
    )
    var gate1_cache_buf = ctx.enqueue_create_buffer[dtype](BTH * 2 * EMB * DEPTH)
    # cond_block caches — MLP branch.
    var ln2_cache_buf = ctx.enqueue_create_buffer[dtype](BTH * (EMB + 1) * DEPTH)
    var mod2_cache_buf = ctx.enqueue_create_buffer[dtype](BTH * 2 * EMB * DEPTH)
    var mlp_cache_buf = ctx.enqueue_create_buffer[dtype](
        BTH * MLP.CACHE_SIZE * DEPTH
    )
    var gate2_cache_buf = ctx.enqueue_create_buffer[dtype](BTH * 2 * EMB * DEPTH)
    # AdaLN output (6D wide).
    var raw_mod_buf = ctx.enqueue_create_buffer[dtype](BTH * 6 * EMB * DEPTH)
    # x_mid buffer per layer (MSA-branch output, needed for backward).
    var x_mid_buf_d = ctx.enqueue_create_buffer[dtype](BTH * EMB * DEPTH)
    # Intermediate x flow between layers. Layer 0 reads x_prev_pe, layer DEPTH-1
    # writes pred_raw. Layers in between read/write into this buffer (one slot
    # per intermediate). For DEPTH=1 we still allocate >=1 to keep DeviceBuffer
    # construction valid.
    var x_inter_buf = ctx.enqueue_create_buffer[dtype](
        _max_int(1, BTH * EMB * (DEPTH - 1))
    )

    # cond_block forward scratch (reused across MSA and MLP branches).
    var silu_buf_d = ctx.enqueue_create_buffer[dtype](BTH * EMB)
    var ln_out_buf_d = ctx.enqueue_create_buffer[dtype](BTH * EMB)
    var mod_inp_buf_d = ctx.enqueue_create_buffer[dtype](BTH * 3 * EMB)
    var mod_x_buf_d = ctx.enqueue_create_buffer[dtype](BTH * EMB)
    var branch_out_buf_d = ctx.enqueue_create_buffer[dtype](BTH * EMB)
    var gate_inp_buf_d = ctx.enqueue_create_buffer[dtype](BTH * 3 * EMB)

    # cond_block workspaces (shared with model forward_gpu calls).
    var adaln_ws_buf = ctx.enqueue_create_buffer[dtype](
        _max_int(1, BTH * ADALN.WORKSPACE_SIZE_PER_SAMPLE)
    )
    var msa_ws_buf = ctx.enqueue_create_buffer[dtype](
        _max_int(1, BATCH * MSA.WORKSPACE_SIZE_PER_SAMPLE)
    )
    var mlp_ws_buf = ctx.enqueue_create_buffer[dtype](
        _max_int(1, BTH * MLP.WORKSPACE_SIZE_PER_SAMPLE)
    )

    # cond_block backward scratch (reused across all DEPTH layers).
    var sgg_buf = ctx.enqueue_create_buffer[dtype](BTH * 3 * EMB)
    var sgbo_buf = ctx.enqueue_create_buffer[dtype](BTH * EMB)
    var sgmx_buf = ctx.enqueue_create_buffer[dtype](BTH * EMB)
    var sgmi_buf = ctx.enqueue_create_buffer[dtype](BTH * 3 * EMB)
    var sglnout_buf = ctx.enqueue_create_buffer[dtype](BTH * EMB)
    var sglnin_buf = ctx.enqueue_create_buffer[dtype](BTH * EMB)
    var sgrm_buf = ctx.enqueue_create_buffer[dtype](BTH * 6 * EMB)
    var sgsc_buf = ctx.enqueue_create_buffer[dtype](BTH * EMB)
    var grad_x_mid_buf = ctx.enqueue_create_buffer[dtype](BTH * EMB)
    # Backward intermediate grad_x flow between layers. Same ping-pong story
    # as forward x_inter_buf: layer DEPTH-1 reads grad_pred_raw, layer 0 writes
    # grad_x_prev_pe; intermediates live here.
    var grad_x_inter_buf = ctx.enqueue_create_buffer[dtype](
        _max_int(1, BTH * EMB * (DEPTH - 1))
    )
    # Per-layer grad_c output (single buffer, reused each layer; accumulated
    # into the trainer's grad_c_buf via cb_accum_kernel).
    var grad_c_layer_buf = ctx.enqueue_create_buffer[dtype](BTH * EMB)

    # SIGReg buffers (forward + backward).
    var sigreg_out_buf = ctx.enqueue_create_buffer[dtype](BATCH * SIG.OUT_DIM)
    var sigreg_cache_buf = ctx.enqueue_create_buffer[dtype](
        BATCH * SIG.CACHE_SIZE
    )
    var sigreg_grad_out_buf = ctx.enqueue_create_buffer[dtype](
        BATCH * SIG.OUT_DIM
    )
    var sigreg_grad_emb_buf = ctx.enqueue_create_buffer[dtype](
        BATCH * T * EMB
    )
    var sigreg_ws_buf = ctx.enqueue_create_buffer[dtype](SIG_WS_SIZE)
    # Seed grad_output = λ/B (constant across all steps; chain rule produces
    # an effective G = λ at the SIGReg dLdz step). See CPU trainer line 735.
    var sigreg_grad_out_host = ctx.enqueue_create_host_buffer[dtype](
        BATCH * SIG.OUT_DIM
    )
    for i in range(BATCH * SIG.OUT_DIM):
        sigreg_grad_out_host[i] = Scalar[dtype](
            lambda_sigreg / Float64(BATCH)
        )
    ctx.enqueue_copy(sigreg_grad_out_buf, sigreg_grad_out_host)

    # Gradient buffers (device).
    var grad_pred_buf = ctx.enqueue_create_buffer[dtype](BATCH * H * EMB)
    var grad_pred_raw_buf = ctx.enqueue_create_buffer[dtype](BTH * EMB)
    var grad_x_prev_buf = ctx.enqueue_create_buffer[dtype](BTH * EMB)
    var grad_x_prev_pe_buf = ctx.enqueue_create_buffer[dtype](BTH * EMB)
    var grad_c_buf = ctx.enqueue_create_buffer[dtype](BTH * EMB)
    var grad_emb_buf = ctx.enqueue_create_buffer[dtype](BATCH * T * EMB)
    var grad_act_emb_buf = ctx.enqueue_create_buffer[dtype](BATCH * T * EMB)
    var grad_actions_buf = ctx.enqueue_create_buffer[dtype](BATCH * T * ACT)
    var grad_pixels_buf = ctx.enqueue_create_buffer[dtype](BT * IMG_DIM)

    # Pinned host buffers for sampled data + per-step loss compute.
    var pixels_host = ctx.enqueue_create_host_buffer[dtype](BT * IMG_DIM)
    var actions_host = ctx.enqueue_create_host_buffer[dtype](BATCH * T * ACT)
    var pred_host = ctx.enqueue_create_host_buffer[dtype](BATCH * H * EMB)
    var target_host = ctx.enqueue_create_host_buffer[dtype](BATCH * H * EMB)
    var grad_pred_host = ctx.enqueue_create_host_buffer[dtype](
        BATCH * H * EMB
    )
    var sigreg_out_host = ctx.enqueue_create_host_buffer[dtype](
        BATCH * SIG.OUT_DIM
    )
    # emb on device has shape (BT, EMB) — aliased as (BATCH, T*EMB) for the
    # target slice. Same memory, single host buffer.
    var emb_host = ctx.enqueue_create_host_buffer[dtype](BT * EMB)

    # ------------------------------------------------------------------
    # Pre-build LayoutTensor views over device buffers.
    # ------------------------------------------------------------------
    var pixels_t = LayoutTensor[
        dtype, Layout.row_major(BT, IMG_DIM), MutAnyOrigin
    ](pixels_buf)
    var actions_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * ACT), MutAnyOrigin
    ](actions_buf)
    var emb_t = LayoutTensor[
        dtype, Layout.row_major(BT, EMB), MutAnyOrigin
    ](emb_buf)
    var enc_cache_t = LayoutTensor[
        dtype, Layout.row_major(BT, ENC.CACHE_SIZE), MutAnyOrigin
    ](enc_cache_buf)

    var act_emb_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * EMB), MutAnyOrigin
    ](act_emb_buf)
    var ae_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, AE.CACHE_SIZE), MutAnyOrigin
    ](ae_cache_buf)

    var x_prev_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](x_prev_buf)
    var x_prev_bh_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ](x_prev_buf)
    var x_prev_pe_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](x_prev_pe_buf)
    var x_prev_pe_bh_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ](x_prev_pe_buf)
    var pos_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, POS.CACHE_SIZE), MutAnyOrigin
    ](pos_cache_buf)
    var c_in_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](c_in_buf)

    var pred_raw_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](pred_raw_buf)
    var pred_raw_bh_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ](pred_raw_buf)
    var pred_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ](pred_out_buf)
    var proj_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, PROJ.CACHE_SIZE), MutAnyOrigin
    ](proj_cache_buf)

    # Per-layer LayoutTensor views are constructed inside the forward/backward
    # loops (one slice per layer d). The base buffers above are sized for all
    # DEPTH layers; each iteration carves a (BTH × cache_per_sample) slot.

    var silu_buf_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](silu_buf_d)
    var ln_out_buf_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](ln_out_buf_d)
    var mod_inp_buf_t = LayoutTensor[
        dtype, Layout.row_major(BTH, 3 * EMB), MutAnyOrigin
    ](mod_inp_buf_d)
    var mod_x_buf_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](mod_x_buf_d)
    var branch_out_buf_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](branch_out_buf_d)
    var gate_inp_buf_t = LayoutTensor[
        dtype, Layout.row_major(BTH, 3 * EMB), MutAnyOrigin
    ](gate_inp_buf_d)

    var sgg_t = LayoutTensor[
        dtype, Layout.row_major(BTH, 3 * EMB), MutAnyOrigin
    ](sgg_buf)
    var sgbo_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](sgbo_buf)
    var sgmx_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](sgmx_buf)
    var sgmi_t = LayoutTensor[
        dtype, Layout.row_major(BTH, 3 * EMB), MutAnyOrigin
    ](sgmi_buf)
    var sglnout_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](sglnout_buf)
    var sglnin_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](sglnin_buf)
    var sgrm_t = LayoutTensor[
        dtype, Layout.row_major(BTH, 6 * EMB), MutAnyOrigin
    ](sgrm_buf)
    var sgsc_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](sgsc_buf)
    var grad_x_mid_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](grad_x_mid_buf)

    var grad_pred_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ](grad_pred_buf)
    var grad_pred_raw_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](grad_pred_raw_buf)
    var grad_pred_raw_bh_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ](grad_pred_raw_buf)
    var grad_x_prev_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](grad_x_prev_buf)
    var grad_x_prev_bh_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ](grad_x_prev_buf)
    var grad_x_prev_pe_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](grad_x_prev_pe_buf)
    var grad_x_prev_pe_bh_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ](grad_x_prev_pe_buf)
    var grad_c_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](grad_c_buf)
    var grad_emb_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * EMB), MutAnyOrigin
    ](grad_emb_buf)
    var grad_emb_bt_t = LayoutTensor[
        dtype, Layout.row_major(BT, EMB), MutAnyOrigin
    ](grad_emb_buf)
    var grad_act_emb_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * EMB), MutAnyOrigin
    ](grad_act_emb_buf)
    var grad_actions_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * ACT), MutAnyOrigin
    ](grad_actions_buf)
    var grad_pixels_t = LayoutTensor[
        dtype, Layout.row_major(BT, IMG_DIM), MutAnyOrigin
    ](grad_pixels_buf)

    # SIGReg views (treat emb / grad_emb as (BATCH, T*EMB) — same memory).
    var emb_bte_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * EMB), MutAnyOrigin
    ](emb_buf)
    var sigreg_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, SIG.OUT_DIM), MutAnyOrigin
    ](sigreg_out_buf)
    var sigreg_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, SIG.CACHE_SIZE), MutAnyOrigin
    ](sigreg_cache_buf)
    var sigreg_grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, SIG.OUT_DIM), MutAnyOrigin
    ](sigreg_grad_out_buf)
    var sigreg_grad_emb_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, T * EMB), MutAnyOrigin
    ](sigreg_grad_emb_buf)
    var empty_params = LayoutTensor[
        dtype, Layout.row_major(SIG.PARAM_SIZE), MutAnyOrigin
    ](UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=0))
    var empty_grad_params = LayoutTensor[
        dtype, Layout.row_major(SIG.PARAM_SIZE), MutAnyOrigin
    ](UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=0))

    # ------------------------------------------------------------------
    # Host-side scratch for batch sampling (reused per step).
    # ------------------------------------------------------------------
    var pixels_sample = alloc[Scalar[dtype]](BT * IMG_DIM)
    var actions_sample = alloc[Scalar[dtype]](BATCH * T * ACT)

    var loss_ema: Float64 = 0.0
    var pred_ema: Float64 = 0.0
    var sigreg_ema: Float64 = 0.0
    var var_min_ema: Float64 = 0.0
    var var_mean_ema: Float64 = 0.0
    var gram_ema: Float64 = 0.0
    var loss_first: Float64 = -1.0
    var loss_last: Float64 = 0.0
    var t0 = perf_counter_ns()
    var loss_scale = Float64(BATCH * H * EMB)
    var inv_scale = Scalar[dtype](2.0 / loss_scale)

    # ------------------------------------------------------------------
    # Step loop
    # ------------------------------------------------------------------
    for step in range(num_steps):
        # Sample batch on host, copy to device.
        buf.sample_batch_fp32(BATCH, T, pixels_sample, actions_sample)
        for i in range(BT * IMG_DIM):
            pixels_host[i] = pixels_sample[i]
        for i in range(BATCH * T * ACT):
            actions_host[i] = actions_sample[i]
        ctx.enqueue_copy(pixels_buf, pixels_host)
        ctx.enqueue_copy(actions_buf, actions_host)

        # Zero grads on all 6 groups.
        enc_state.zero_grads(ctx)
        ae_state.zero_grads(ctx)
        pos_state.zero_grads(ctx)
        proj_state.zero_grads(ctx)
        for d in range(DEPTH):
            adaln_states[d].zero_grads(ctx)
            msa_states[d].zero_grads(ctx)
            mlp_states[d].zero_grads(ctx)

        # Encoder forward.
        ENC.forward_gpu[BT, dtype](
            ctx, emb_t, pixels_t,
            enc_state.params_view(), enc_state.model_state_view(),
            enc_cache_t, enc_ws_buf,
        )
        # Action embedder forward.
        AE.forward_gpu[BATCH, dtype](
            ctx, act_emb_t, actions_t,
            ae_state.params_view(), ae_state.model_state_view(),
            ae_cache_t, ae_ws_buf,
        )

        # Slice first H tokens of emb + act_emb into x_prev_buf + c_in_buf.
        var act_emb_bt_t = LayoutTensor[
            dtype, Layout.row_major(BT, EMB), MutAnyOrigin
        ](act_emb_buf)
        ctx.enqueue_function[
            slice_h_kernel[BATCH, T, H, EMB],
        ](
            emb_t, x_prev_t,
            grid_dim=(
                ceildiv(BATCH, TPB_X),
                ceildiv(H, TPB_Y),
                ceildiv(EMB, TPB_Z),
            ),
            block_dim=(TPB_X, TPB_Y, TPB_Z),
        )
        ctx.enqueue_function[
            slice_h_kernel[BATCH, T, H, EMB],
        ](
            act_emb_bt_t, c_in_t,
            grid_dim=(
                ceildiv(BATCH, TPB_X),
                ceildiv(H, TPB_Y),
                ceildiv(EMB, TPB_Z),
            ),
            block_dim=(TPB_X, TPB_Y, TPB_Z),
        )

        # Pos embed: x_prev_pe = x_prev + pos_bias (broadcast over BATCH).
        POS.forward_gpu[BATCH, dtype](
            ctx, x_prev_pe_bh_t, x_prev_bh_t,
            pos_state.params_view(), pos_state.model_state_view(),
            pos_cache_t, pos_ws_buf,
        )

        # cond_block stack: DEPTH dual-branch (MSA + MLP) layers.
        # Layer 0 input = x_prev_pe_t; layer DEPTH-1 output = pred_raw_t.
        # Intermediate x flow uses slices of x_inter_buf.
        for d in range(DEPTH):
            # Choose input/output tensor views for this layer.
            var x_in_t = LayoutTensor[
                dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
            ](
                x_prev_pe_buf.unsafe_ptr() if d == 0
                else x_inter_buf.unsafe_ptr() + (d - 1) * BTH * EMB
            )
            var x_out_t = LayoutTensor[
                dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
            ](
                pred_raw_buf.unsafe_ptr() if d == DEPTH - 1
                else x_inter_buf.unsafe_ptr() + d * BTH * EMB
            )
            # Per-layer cache slices.
            var silu_cache_d = LayoutTensor[
                dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
            ](silu_cache_buf.unsafe_ptr() + d * BTH * EMB)
            var adaln_cache_d = LayoutTensor[
                dtype, Layout.row_major(BTH, ADALN.CACHE_SIZE), MutAnyOrigin
            ](adaln_cache_buf.unsafe_ptr() + d * BTH * ADALN.CACHE_SIZE)
            var ln1_cache_d = LayoutTensor[
                dtype, Layout.row_major(BTH, EMB + 1), MutAnyOrigin
            ](ln1_cache_buf.unsafe_ptr() + d * BTH * (EMB + 1))
            var mod1_cache_d = LayoutTensor[
                dtype, Layout.row_major(BTH, 2 * EMB), MutAnyOrigin
            ](mod1_cache_buf.unsafe_ptr() + d * BTH * 2 * EMB)
            var msa_cache_d = LayoutTensor[
                dtype, Layout.row_major(BATCH, MSA.CACHE_SIZE), MutAnyOrigin
            ](msa_cache_buf.unsafe_ptr() + d * BATCH * MSA.CACHE_SIZE)
            var gate1_cache_d = LayoutTensor[
                dtype, Layout.row_major(BTH, 2 * EMB), MutAnyOrigin
            ](gate1_cache_buf.unsafe_ptr() + d * BTH * 2 * EMB)
            var ln2_cache_d = LayoutTensor[
                dtype, Layout.row_major(BTH, EMB + 1), MutAnyOrigin
            ](ln2_cache_buf.unsafe_ptr() + d * BTH * (EMB + 1))
            var mod2_cache_d = LayoutTensor[
                dtype, Layout.row_major(BTH, 2 * EMB), MutAnyOrigin
            ](mod2_cache_buf.unsafe_ptr() + d * BTH * 2 * EMB)
            var mlp_cache_d = LayoutTensor[
                dtype, Layout.row_major(BTH, MLP.CACHE_SIZE), MutAnyOrigin
            ](mlp_cache_buf.unsafe_ptr() + d * BTH * MLP.CACHE_SIZE)
            var gate2_cache_d = LayoutTensor[
                dtype, Layout.row_major(BTH, 2 * EMB), MutAnyOrigin
            ](gate2_cache_buf.unsafe_ptr() + d * BTH * 2 * EMB)
            var raw_mod_d = LayoutTensor[
                dtype, Layout.row_major(BTH, 6 * EMB), MutAnyOrigin
            ](raw_mod_buf.unsafe_ptr() + d * BTH * 6 * EMB)
            var x_mid_d = LayoutTensor[
                dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
            ](x_mid_buf_d.unsafe_ptr() + d * BTH * EMB)

            cond_block_forward_gpu[BATCH, H, EMB, PRED_HEADS, PRED_FF](
                ctx, x_in_t, c_in_t,
                adaln_states[d].params_view(),
                adaln_states[d].model_state_view(),
                msa_states[d].params_view(),
                msa_states[d].model_state_view(),
                mlp_states[d].params_view(),
                mlp_states[d].model_state_view(),
                x_out_t,
                silu_cache_d, adaln_cache_d,
                ln1_cache_d, mod1_cache_d, msa_cache_d, gate1_cache_d,
                ln2_cache_d, mod2_cache_d, mlp_cache_d, gate2_cache_d,
                raw_mod_d, x_mid_d,
                silu_buf_t, ln_out_buf_t, mod_inp_buf_t,
                mod_x_buf_t, branch_out_buf_t, gate_inp_buf_t,
                adaln_ws_buf, msa_ws_buf, mlp_ws_buf,
            )

        # PredProj: (BATCH, H*EMB) → (BATCH, H*EMB).
        PROJ.forward_gpu[BATCH, dtype](
            ctx, pred_t, pred_raw_bh_t,
            proj_state.params_view(), proj_state.model_state_view(),
            proj_cache_t, proj_ws_buf,
        )

        # SIGReg forward over emb viewed as (BATCH, T*EMB). Output is the
        # statistic replicated across BATCH slots (we read [0] for logging).
        SIG.eval_gpu[BATCH, dtype](
            ctx, sigreg_out_t, emb_bte_t,
            empty_params, sigreg_cache_t, sigreg_ws_buf.unsafe_ptr(),
        )

        # --------------------------------------------------------------
        # Loss + grad_pred on host (small round-trip).
        # --------------------------------------------------------------
        ctx.enqueue_copy(pred_host, pred_out_buf)
        # Download all of emb (BT, EMB) — used for both target slice and probes.
        ctx.enqueue_copy(emb_host, emb_buf)
        # Download SIGReg stat (tiny — BATCH floats) for logging.
        ctx.enqueue_copy(sigreg_out_host, sigreg_out_buf)
        ctx.synchronize()

        var pred_loss: Float64 = 0.0
        for b in range(BATCH):
            for i in range(H * EMB):
                var p = Float64(pred_host[b * H * EMB + i])
                # Target = emb[b, N_PREDS .. N_PREDS+H, :], flat index:
                #   b * T * EMB + N_PREDS * EMB + i
                var tgt = Float64(
                    emb_host[b * T * EMB + N_PREDS * EMB + i]
                )
                var diff = p - tgt
                pred_loss += diff * diff
                grad_pred_host[b * H * EMB + i] = inv_scale * (
                    Scalar[dtype](p) - Scalar[dtype](tgt)
                )
        pred_loss /= loss_scale

        # Read SIGReg stat (replicated across BATCH, take [0]).
        var sigreg_stat = Float64(sigreg_out_host[0])

        if loss_first < 0.0:
            loss_first = pred_loss
            loss_ema = pred_loss
            pred_ema = pred_loss
            sigreg_ema = sigreg_stat
        else:
            loss_ema = 0.95 * loss_ema + 0.05 * pred_loss
            pred_ema = 0.95 * pred_ema + 0.05 * pred_loss
            sigreg_ema = 0.95 * sigreg_ema + 0.05 * sigreg_stat
        loss_last = pred_loss

        # Upload grad_pred back to device.
        ctx.enqueue_copy(grad_pred_buf, grad_pred_host)

        # --------------------------------------------------------------
        # Collapse probes (cheap — emb already on host).
        # --------------------------------------------------------------
        var var_min: Float64 = 1e30
        var var_mean: Float64 = 0.0
        for d in range(EMB):
            var s: Float64 = 0.0
            var ss: Float64 = 0.0
            for bt in range(BT):
                var v = Float64(emb_host[bt * EMB + d])  # uses emb_bte_host? no, separate.
                s += v
                ss += v * v
            var mean_d = s / Float64(BT)
            var var_d = (ss / Float64(BT)) - mean_d * mean_d
            if var_d < var_min:
                var_min = var_d
            var_mean += var_d
        var_mean /= Float64(EMB)
        var gram_off: Float64 = 0.0
        var gram_n: Int = 0
        for i in range(BT):
            var ni: Float64 = 0.0
            for d in range(EMB):
                var v = Float64(emb_host[i * EMB + d])
                ni += v * v
            ni = sqrt(ni + 1e-12)
            for j in range(i + 1, BT):
                var nj: Float64 = 0.0
                var dot_v: Float64 = 0.0
                for d in range(EMB):
                    var vi = Float64(emb_host[i * EMB + d])
                    var vj = Float64(emb_host[j * EMB + d])
                    nj += vj * vj
                    dot_v += vi * vj
                nj = sqrt(nj + 1e-12)
                var c_v = dot_v / (ni * nj)
                if c_v < 0.0:
                    c_v = -c_v
                gram_off += c_v
                gram_n += 1
        gram_off /= Float64(gram_n)

        if step == 0:
            var_min_ema = var_min
            var_mean_ema = var_mean
            gram_ema = gram_off
        else:
            var_min_ema = 0.95 * var_min_ema + 0.05 * var_min
            var_mean_ema = 0.95 * var_mean_ema + 0.05 * var_mean
            gram_ema = 0.95 * gram_ema + 0.05 * gram_off

        # --------------------------------------------------------------
        # Backward
        # --------------------------------------------------------------
        var enc_g = enc_state.grads_view()
        var ae_g = ae_state.grads_view()
        var pos_g = pos_state.grads_view()
        var proj_g = proj_state.grads_view()

        # PROJ.backward
        PROJ.backward_gpu[BATCH, dtype](
            ctx, grad_pred_raw_bh_t, grad_pred_t,
            proj_state.params_view(), proj_state.model_state_view(),
            proj_cache_t, proj_g, proj_ws_buf,
        )

        # cond_block stack backward — reverse loop over DEPTH layers.
        # Per-layer grad_c is accumulated into trainer's grad_c_buf.
        ctx.enqueue_memset(grad_c_buf, 0)
        var grad_c_layer_t = LayoutTensor[
            dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
        ](grad_c_layer_buf)

        for d_rev in range(DEPTH):
            var d = DEPTH - 1 - d_rev
            # Choose grad_x_next (input) and grad_x_prev (output) tensor views.
            var grad_x_next_t = LayoutTensor[
                dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
            ](
                grad_pred_raw_buf.unsafe_ptr() if d == DEPTH - 1
                else grad_x_inter_buf.unsafe_ptr() + d * BTH * EMB
            )
            var grad_x_prev_layer_t = LayoutTensor[
                dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
            ](
                grad_x_prev_pe_buf.unsafe_ptr() if d == 0
                else grad_x_inter_buf.unsafe_ptr() + (d - 1) * BTH * EMB
            )
            # Per-layer cache slices (same as forward).
            var silu_cache_d = LayoutTensor[
                dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
            ](silu_cache_buf.unsafe_ptr() + d * BTH * EMB)
            var adaln_cache_d = LayoutTensor[
                dtype, Layout.row_major(BTH, ADALN.CACHE_SIZE), MutAnyOrigin
            ](adaln_cache_buf.unsafe_ptr() + d * BTH * ADALN.CACHE_SIZE)
            var ln1_cache_d = LayoutTensor[
                dtype, Layout.row_major(BTH, EMB + 1), MutAnyOrigin
            ](ln1_cache_buf.unsafe_ptr() + d * BTH * (EMB + 1))
            var mod1_cache_d = LayoutTensor[
                dtype, Layout.row_major(BTH, 2 * EMB), MutAnyOrigin
            ](mod1_cache_buf.unsafe_ptr() + d * BTH * 2 * EMB)
            var msa_cache_d = LayoutTensor[
                dtype, Layout.row_major(BATCH, MSA.CACHE_SIZE), MutAnyOrigin
            ](msa_cache_buf.unsafe_ptr() + d * BATCH * MSA.CACHE_SIZE)
            var gate1_cache_d = LayoutTensor[
                dtype, Layout.row_major(BTH, 2 * EMB), MutAnyOrigin
            ](gate1_cache_buf.unsafe_ptr() + d * BTH * 2 * EMB)
            var ln2_cache_d = LayoutTensor[
                dtype, Layout.row_major(BTH, EMB + 1), MutAnyOrigin
            ](ln2_cache_buf.unsafe_ptr() + d * BTH * (EMB + 1))
            var mod2_cache_d = LayoutTensor[
                dtype, Layout.row_major(BTH, 2 * EMB), MutAnyOrigin
            ](mod2_cache_buf.unsafe_ptr() + d * BTH * 2 * EMB)
            var mlp_cache_d = LayoutTensor[
                dtype, Layout.row_major(BTH, MLP.CACHE_SIZE), MutAnyOrigin
            ](mlp_cache_buf.unsafe_ptr() + d * BTH * MLP.CACHE_SIZE)
            var gate2_cache_d = LayoutTensor[
                dtype, Layout.row_major(BTH, 2 * EMB), MutAnyOrigin
            ](gate2_cache_buf.unsafe_ptr() + d * BTH * 2 * EMB)
            var x_mid_d = LayoutTensor[
                dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
            ](x_mid_buf_d.unsafe_ptr() + d * BTH * EMB)

            var adaln_g_d = adaln_states[d].grads_view()
            var msa_g_d = msa_states[d].grads_view()
            var mlp_g_d = mlp_states[d].grads_view()

            cond_block_backward_gpu[BATCH, H, EMB, PRED_HEADS, PRED_FF](
                ctx, grad_x_next_t,
                adaln_states[d].params_view(),
                adaln_states[d].model_state_view(),
                msa_states[d].params_view(),
                msa_states[d].model_state_view(),
                mlp_states[d].params_view(),
                mlp_states[d].model_state_view(),
                silu_cache_d, adaln_cache_d,
                ln1_cache_d, mod1_cache_d, msa_cache_d, gate1_cache_d,
                ln2_cache_d, mod2_cache_d, mlp_cache_d, gate2_cache_d,
                x_mid_d,
                grad_x_prev_layer_t, grad_c_layer_t,
                adaln_g_d, msa_g_d, mlp_g_d,
                sgg_t, sgbo_t, sgmx_t, sgmi_t,
                sglnout_t, sglnin_t, sgrm_t, sgsc_t,
                grad_x_mid_t,
                adaln_ws_buf, msa_ws_buf, mlp_ws_buf,
            )

            # Accumulate this layer's grad_c into grad_c_t (shared across layers).
            comptime TPB_GC_X = 16
            comptime TPB_GC_Y = 16
            ctx.enqueue_function[cb_accum_kernel[BTH, EMB]](
                grad_c_t, grad_c_layer_t,
                grid_dim=(ceildiv(BTH, TPB_GC_X), ceildiv(EMB, TPB_GC_Y)),
                block_dim=(TPB_GC_X, TPB_GC_Y),
            )

        # POS.backward
        POS.backward_gpu[BATCH, dtype](
            ctx, grad_x_prev_bh_t, grad_x_prev_pe_bh_t,
            pos_state.params_view(), pos_state.model_state_view(),
            pos_cache_t, pos_g, pos_ws_buf,
        )

        # Route grad_x_prev → grad_emb's first H tokens, grad_c → grad_act_emb's.
        # Target slice gradient is FILLED below (no stop-grad — paper recipe).
        ctx.enqueue_memset(grad_emb_buf, 0)
        ctx.enqueue_memset(grad_act_emb_buf, 0)

        var grad_emb_bte_to_bt = LayoutTensor[
            dtype, Layout.row_major(BT, EMB), MutAnyOrigin
        ](grad_emb_buf)
        var grad_act_emb_bte_to_bt = LayoutTensor[
            dtype, Layout.row_major(BT, EMB), MutAnyOrigin
        ](grad_act_emb_buf)
        ctx.enqueue_function[
            scatter_h_kernel[BATCH, T, H, EMB],
        ](
            grad_x_prev_t, grad_emb_bte_to_bt,
            grid_dim=(
                ceildiv(BATCH, TPB_X),
                ceildiv(H, TPB_Y),
                ceildiv(EMB, TPB_Z),
            ),
            block_dim=(TPB_X, TPB_Y, TPB_Z),
        )
        ctx.enqueue_function[
            scatter_h_kernel[BATCH, T, H, EMB],
        ](
            grad_c_t, grad_act_emb_bte_to_bt,
            grid_dim=(
                ceildiv(BATCH, TPB_X),
                ceildiv(H, TPB_Y),
                ceildiv(EMB, TPB_Z),
            ),
            block_dim=(TPB_X, TPB_Y, TPB_Z),
        )

        # Drop stop-grad: scatter -grad_pred into target slice of grad_emb.
        # Math: pred_loss = (pred - tgt)^2 / N → d/d tgt = -grad_pred.
        # Target tokens live at b * T*EMB + N_PREDS*EMB + [0..H*EMB).
        comptime TPB_TS_X = 4
        comptime TPB_TS_Y = 64
        ctx.enqueue_function[
            scatter_target_neg_kernel[BATCH, T, H, N_PREDS, EMB],
        ](
            grad_pred_t, grad_emb_t,
            grid_dim=(
                ceildiv(BATCH, TPB_TS_X),
                ceildiv(H * EMB, TPB_TS_Y),
            ),
            block_dim=(TPB_TS_X, TPB_TS_Y),
        )

        # SIGReg vjp: produces sigreg_grad_emb (BATCH, T*EMB) from
        # `sigreg_grad_out_t` seed = λ/B (set once at init).
        SIG.vjp_gpu[BATCH, dtype](
            ctx, sigreg_grad_out_t, sigreg_grad_emb_t,
            empty_params, sigreg_cache_t, empty_grad_params,
            sigreg_ws_buf.unsafe_ptr(),
        )
        # Accumulate sigreg's grad into grad_emb additively.
        comptime TPB_AC_X = 4
        comptime TPB_AC_Y = 64
        ctx.enqueue_function[
            accumulate_emb_kernel[BATCH, T, EMB],
        ](
            sigreg_grad_emb_t, grad_emb_t,
            grid_dim=(
                ceildiv(BATCH, TPB_AC_X),
                ceildiv(T * EMB, TPB_AC_Y),
            ),
            block_dim=(TPB_AC_X, TPB_AC_Y),
        )

        # AE.backward
        AE.backward_gpu[BATCH, dtype](
            ctx, grad_actions_t, grad_act_emb_t,
            ae_state.params_view(), ae_state.model_state_view(),
            ae_cache_t, ae_g, ae_ws_buf,
        )

        # ENC.backward
        ENC.backward_gpu[BT, dtype](
            ctx, grad_pixels_t, grad_emb_bt_t,
            enc_state.params_view(), enc_state.model_state_view(),
            enc_cache_t, enc_g, enc_ws_buf,
        )

        # Optimizer step — shared models + per-layer (ADALN/MSA/MLP × DEPTH).
        enc_state.optimizer_step(ctx)
        ae_state.optimizer_step(ctx)
        pos_state.optimizer_step(ctx)
        proj_state.optimizer_step(ctx)
        for d in range(DEPTH):
            adaln_states[d].optimizer_step(ctx)
            msa_states[d].optimizer_step(ctx)
            mlp_states[d].optimizer_step(ctx)

        # Periodic logging — download emb for probes was done inline above.
        if step % log_every == 0 or step == num_steps - 1:
            ctx.synchronize()
            var t_now = perf_counter_ns()
            var sps = Float64(step + 1) / (Float64(t_now - t0) / 1e9)
            print(
                "  step", step,
                " L=", pred_loss,
                " ema=", pred_ema,
                " sig=", sigreg_ema,
                " var_min=", var_min_ema,
                " var_mean=", var_mean_ema,
                " gram=", gram_ema,
                " it/s=", sps,
            )

    ctx.synchronize()
    var t1 = perf_counter_ns()
    var total_s = Float64(t1 - t0) / 1e9
    print()
    print("Trained", num_steps, "steps in", total_s, "s")
    print("  loss_first =", loss_first)
    print("  loss_last  =", loss_last)
    print("  pred_ema   =", pred_ema)
    print(
        "  rel_drop   =",
        (loss_first - loss_last) / (loss_first + 1e-12),
    )
    print()
    print("Collapse probes (EMA across the run):")
    print("  var_min  =", var_min_ema, " (want > 0.1)")
    print("  var_mean =", var_mean_ema)
    print("  gram_off =", gram_ema, " (want < ~0.5)")

    pixels_sample.free()
    actions_sample.free()
