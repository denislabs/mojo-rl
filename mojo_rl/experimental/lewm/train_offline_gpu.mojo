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
from std.random import seed as _set_seed, random_float64
from std.time import perf_counter_ns
from std.gpu import global_idx, thread_idx
from std.gpu.primitives import block, warp
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


# =============================================================================
# Phase 4 GPU-side rollout kernels — keep emb_seq, action_plan, score on device.
# Eliminates per-step host->device of BATCH*T*EMB + the corresponding pred
# download. Each kernel works on plain (BATCH, ROLL_T, EMB/ACT) layouts.
# =============================================================================


# Initialize emb_seq[b, 0..H-1, :] = emb_start[b, :], zero rest.
def replicate_start_emb_kernel[
    BATCH: Int, H: Int, EMB: Int, ROLL_T: Int,
](
    emb_start: LayoutTensor[
        dtype, Layout.row_major(BATCH, EMB), MutAnyOrigin
    ],
    emb_seq: LayoutTensor[
        dtype, Layout.row_major(BATCH, ROLL_T * EMB), MutAnyOrigin
    ],
):
    var b = Int(global_idx.x)
    var p = Int(global_idx.y)
    var d = Int(global_idx.z)
    if b < BATCH and p < ROLL_T and d < EMB:
        if p < H:
            emb_seq[b, p * EMB + d] = emb_start[b, d]
        else:
            emb_seq[b, p * EMB + d] = Scalar[dtype](0.0)


# Slide H-position window from emb_seq[k..k+H-1] -> emb_buf[0..H-1].
# Pad emb_buf[H..T-1] with zeros (slice_h_kernel only reads first H).
def slide_emb_window_kernel[
    BATCH: Int, T: Int, H: Int, EMB: Int, ROLL_T: Int,
](
    emb_seq: LayoutTensor[
        dtype, Layout.row_major(BATCH, ROLL_T * EMB), MutAnyOrigin
    ],
    emb_buf: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, EMB), MutAnyOrigin
    ],
    k: Int,
):
    var b = Int(global_idx.x)
    var p = Int(global_idx.y)
    var d = Int(global_idx.z)
    if b < BATCH and p < T and d < EMB:
        if p < H:
            emb_buf[b * T + p, d] = emb_seq[b, (k + p) * EMB + d]
        else:
            emb_buf[b * T + p, d] = Scalar[dtype](0.0)


# Slide H-position window from action_plan[k..k+H-1] -> actions_buf[0..H-1].
# Pad actions_buf[H..T-1] with zeros.
def slide_actions_window_kernel[
    BATCH: Int, T: Int, H: Int, ACT: Int, NEEDED_ACTIONS: Int,
](
    action_plan: LayoutTensor[
        dtype, Layout.row_major(BATCH, NEEDED_ACTIONS * ACT), MutAnyOrigin
    ],
    actions_buf: LayoutTensor[
        dtype, Layout.row_major(BATCH, T * ACT), MutAnyOrigin
    ],
    k: Int,
):
    var b = Int(global_idx.x)
    var p = Int(global_idx.y)
    var a = Int(global_idx.z)
    if b < BATCH and p < T and a < ACT:
        if p < H:
            actions_buf[b, p * ACT + a] = (
                action_plan[b, (k + p) * ACT + a]
            )
        else:
            actions_buf[b, p * ACT + a] = Scalar[dtype](0.0)


# Store pred[:, H-1, :] into emb_seq[:, k+H, :].
def store_pred_last_kernel[
    BATCH: Int, H: Int, EMB: Int, ROLL_T: Int,
](
    pred: LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ],
    emb_seq: LayoutTensor[
        dtype, Layout.row_major(BATCH, ROLL_T * EMB), MutAnyOrigin
    ],
    k: Int,
):
    var b = Int(global_idx.x)
    var d = Int(global_idx.y)
    if b < BATCH and d < EMB:
        emb_seq[b, (k + H) * EMB + d] = pred[b, (H - 1) * EMB + d]


# Compute MSE(emb_seq[:, GOAL_POS, :], emb_goal[:, :]) summed across
# BATCH × EMB; thread 0 writes the sum to score_out[0].
# Caller divides by BATCH * EMB on host (or just inspects raw sum).
# Single block of BATCH threads (BATCH ≤ 256 in our configs).
def mpc_score_kernel[
    BATCH: Int, EMB: Int, ROLL_T: Int,
](
    emb_seq: LayoutTensor[
        dtype, Layout.row_major(BATCH, ROLL_T * EMB), MutAnyOrigin
    ],
    emb_goal: LayoutTensor[
        dtype, Layout.row_major(BATCH, EMB), MutAnyOrigin
    ],
    score_out: LayoutTensor[
        dtype, Layout.row_major(1), MutAnyOrigin
    ],
    goal_pos: Int,
):
    # BATCH ≤ warp_size (32 NVIDIA / Apple) in all our configs, so a single
    # warp.sum suffices. Block launched with block_dim=32 (warp width);
    # threads with idx ≥ BATCH contribute 0.
    var b = Int(thread_idx.x)
    var local_sum: Scalar[dtype] = 0.0
    if b < BATCH:
        for d in range(EMB):
            var ev = rebind[Scalar[dtype]](
                emb_seq[b, goal_pos * EMB + d]
            )
            var gv = rebind[Scalar[dtype]](emb_goal[b, d])
            var diff = ev - gv
            local_sum += diff * diff
    var warp_sum = warp.sum(local_sum)
    if thread_idx.x == 0:
        score_out[0] = rebind[score_out.element_type](warp_sum)


# =============================================================================
# Per-layer cond_block orchestration (DEPTH-stacked predictor).
#
# These helpers exist to isolate compilation: putting the per-layer cache
# slicing + cond_block_forward/backward call directly in the trainer's
# main function body triggered a Mojo compile-time explosion (5GB+ RAM,
# 3+ min). Each helper is its own function with comptime params, so it
# compiles once. The trainer's for-d loop body is then just one helper
# call per iteration.
# =============================================================================


def run_cond_layer_forward[
    BATCH: Int, T: Int, D: Int, HEADS: Int, FF: Int,
](
    ctx: DeviceContext,
    d: Int,
    DEPTH_VAL: Int,
    # Base x buffers (helper picks x_in/x_out based on d, DEPTH_VAL).
    x_prev_pe_buf: DeviceBuffer[dtype],
    x_inter_buf: DeviceBuffer[dtype],
    pred_raw_buf: DeviceBuffer[dtype],
    # Shared c input (same across all layers).
    c_in_t: LayoutTensor[dtype, Layout.row_major((BATCH * T), D), MutAnyOrigin],
    # Per-layer params + state (caller passes adaln_states[d].params_view() etc.).
    adaln_params: LayoutTensor[
        dtype, Layout.row_major(AdaLNMod[D].PARAM_SIZE), MutAnyOrigin
    ],
    adaln_state: LayoutTensor[
        dtype, Layout.row_major(AdaLNMod[D].STATE_SIZE), MutAnyOrigin
    ],
    msa_params: LayoutTensor[
        dtype,
        Layout.row_major(MultiHeadAttention[D, HEADS, T, True].PARAM_SIZE),
        MutAnyOrigin,
    ],
    msa_state: LayoutTensor[
        dtype,
        Layout.row_major(
            MultiHeadAttention[D, HEADS, T, True].STATE_SIZE
        ),
        MutAnyOrigin,
    ],
    mlp_params: LayoutTensor[
        dtype, Layout.row_major(CondMLP[D, FF].PARAM_SIZE), MutAnyOrigin
    ],
    mlp_state: LayoutTensor[
        dtype, Layout.row_major(CondMLP[D, FF].STATE_SIZE), MutAnyOrigin
    ],
    # Base cache buffers (helper slices to layer d).
    silu_cache_buf: DeviceBuffer[dtype],
    adaln_cache_buf: DeviceBuffer[dtype],
    ln1_cache_buf: DeviceBuffer[dtype],
    mod1_cache_buf: DeviceBuffer[dtype],
    msa_cache_buf: DeviceBuffer[dtype],
    gate1_cache_buf: DeviceBuffer[dtype],
    ln2_cache_buf: DeviceBuffer[dtype],
    mod2_cache_buf: DeviceBuffer[dtype],
    mlp_cache_buf: DeviceBuffer[dtype],
    gate2_cache_buf: DeviceBuffer[dtype],
    raw_mod_buf: DeviceBuffer[dtype],
    x_mid_buf_d: DeviceBuffer[dtype],
    # Scratch (reused across layers).
    mut silu_buf_t: LayoutTensor[
        dtype, Layout.row_major((BATCH * T), D), MutAnyOrigin
    ],
    mut ln_out_buf_t: LayoutTensor[
        dtype, Layout.row_major((BATCH * T), D), MutAnyOrigin
    ],
    mut mod_inp_buf_t: LayoutTensor[
        dtype, Layout.row_major((BATCH * T), 3 * D), MutAnyOrigin
    ],
    mut mod_x_buf_t: LayoutTensor[
        dtype, Layout.row_major((BATCH * T), D), MutAnyOrigin
    ],
    mut branch_out_buf_t: LayoutTensor[
        dtype, Layout.row_major((BATCH * T), D), MutAnyOrigin
    ],
    mut gate_inp_buf_t: LayoutTensor[
        dtype, Layout.row_major((BATCH * T), 3 * D), MutAnyOrigin
    ],
    # Workspaces.
    adaln_workspace: DeviceBuffer[dtype],
    msa_workspace: DeviceBuffer[dtype],
    mlp_workspace: DeviceBuffer[dtype],
) raises:
    # x_in: x_prev_pe at layer 0; intermediate slice otherwise.
    var x_in_t = LayoutTensor[
        dtype, Layout.row_major((BATCH * T), D), MutAnyOrigin
    ](
        x_prev_pe_buf.unsafe_ptr() if d == 0
        else x_inter_buf.unsafe_ptr() + (d - 1) * (BATCH * T) * D
    )
    # x_out: pred_raw at layer DEPTH-1; intermediate slice otherwise.
    var x_out_t = LayoutTensor[
        dtype, Layout.row_major((BATCH * T), D), MutAnyOrigin
    ](
        pred_raw_buf.unsafe_ptr() if d == DEPTH_VAL - 1
        else x_inter_buf.unsafe_ptr() + d * (BATCH * T) * D
    )
    # Per-layer cache slices.
    var silu_cache_d = LayoutTensor[
        dtype, Layout.row_major((BATCH * T), D), MutAnyOrigin
    ](silu_cache_buf.unsafe_ptr() + d * (BATCH * T) * D)
    var adaln_cache_d = LayoutTensor[
        dtype, Layout.row_major((BATCH * T), AdaLNMod[D].CACHE_SIZE), MutAnyOrigin
    ](adaln_cache_buf.unsafe_ptr() + d * (BATCH * T) * AdaLNMod[D].CACHE_SIZE)
    var ln1_cache_d = LayoutTensor[
        dtype, Layout.row_major((BATCH * T), D + 1), MutAnyOrigin
    ](ln1_cache_buf.unsafe_ptr() + d * (BATCH * T) * (D + 1))
    var mod1_cache_d = LayoutTensor[
        dtype, Layout.row_major((BATCH * T), 2 * D), MutAnyOrigin
    ](mod1_cache_buf.unsafe_ptr() + d * (BATCH * T) * 2 * D)
    var msa_cache_d = LayoutTensor[
        dtype,
        Layout.row_major(BATCH, MultiHeadAttention[D, HEADS, T, True].CACHE_SIZE),
        MutAnyOrigin,
    ](
        msa_cache_buf.unsafe_ptr()
        + d * BATCH * MultiHeadAttention[D, HEADS, T, True].CACHE_SIZE
    )
    var gate1_cache_d = LayoutTensor[
        dtype, Layout.row_major((BATCH * T), 2 * D), MutAnyOrigin
    ](gate1_cache_buf.unsafe_ptr() + d * (BATCH * T) * 2 * D)
    var ln2_cache_d = LayoutTensor[
        dtype, Layout.row_major((BATCH * T), D + 1), MutAnyOrigin
    ](ln2_cache_buf.unsafe_ptr() + d * (BATCH * T) * (D + 1))
    var mod2_cache_d = LayoutTensor[
        dtype, Layout.row_major((BATCH * T), 2 * D), MutAnyOrigin
    ](mod2_cache_buf.unsafe_ptr() + d * (BATCH * T) * 2 * D)
    var mlp_cache_d = LayoutTensor[
        dtype, Layout.row_major((BATCH * T), CondMLP[D, FF].CACHE_SIZE), MutAnyOrigin
    ](mlp_cache_buf.unsafe_ptr() + d * (BATCH * T) * CondMLP[D, FF].CACHE_SIZE)
    var gate2_cache_d = LayoutTensor[
        dtype, Layout.row_major((BATCH * T), 2 * D), MutAnyOrigin
    ](gate2_cache_buf.unsafe_ptr() + d * (BATCH * T) * 2 * D)
    var raw_mod_d = LayoutTensor[
        dtype, Layout.row_major((BATCH * T), 6 * D), MutAnyOrigin
    ](raw_mod_buf.unsafe_ptr() + d * (BATCH * T) * 6 * D)
    var x_mid_d = LayoutTensor[
        dtype, Layout.row_major((BATCH * T), D), MutAnyOrigin
    ](x_mid_buf_d.unsafe_ptr() + d * (BATCH * T) * D)

    cond_block_forward_gpu[BATCH, T, D, HEADS, FF](
        ctx, x_in_t, c_in_t,
        adaln_params, adaln_state,
        msa_params, msa_state,
        mlp_params, mlp_state,
        x_out_t,
        silu_cache_d, adaln_cache_d,
        ln1_cache_d, mod1_cache_d, msa_cache_d, gate1_cache_d,
        ln2_cache_d, mod2_cache_d, mlp_cache_d, gate2_cache_d,
        raw_mod_d, x_mid_d,
        silu_buf_t, ln_out_buf_t, mod_inp_buf_t,
        mod_x_buf_t, branch_out_buf_t, gate_inp_buf_t,
        adaln_workspace, msa_workspace, mlp_workspace,
    )


def run_cond_layer_backward[
    BATCH: Int, T: Int, D: Int, HEADS: Int, FF: Int,
](
    ctx: DeviceContext,
    d: Int,
    DEPTH_VAL: Int,
    # Base grad_x buffers (helper picks grad_x_next/grad_x_prev based on d).
    grad_pred_raw_buf: DeviceBuffer[dtype],
    grad_x_inter_buf: DeviceBuffer[dtype],
    grad_x_prev_pe_buf: DeviceBuffer[dtype],
    # Per-layer params + state.
    adaln_params: LayoutTensor[
        dtype, Layout.row_major(AdaLNMod[D].PARAM_SIZE), MutAnyOrigin
    ],
    adaln_state: LayoutTensor[
        dtype, Layout.row_major(AdaLNMod[D].STATE_SIZE), MutAnyOrigin
    ],
    msa_params: LayoutTensor[
        dtype,
        Layout.row_major(MultiHeadAttention[D, HEADS, T, True].PARAM_SIZE),
        MutAnyOrigin,
    ],
    msa_state: LayoutTensor[
        dtype,
        Layout.row_major(
            MultiHeadAttention[D, HEADS, T, True].STATE_SIZE
        ),
        MutAnyOrigin,
    ],
    mlp_params: LayoutTensor[
        dtype, Layout.row_major(CondMLP[D, FF].PARAM_SIZE), MutAnyOrigin
    ],
    mlp_state: LayoutTensor[
        dtype, Layout.row_major(CondMLP[D, FF].STATE_SIZE), MutAnyOrigin
    ],
    # Base cache buffers (sliced per layer).
    silu_cache_buf: DeviceBuffer[dtype],
    adaln_cache_buf: DeviceBuffer[dtype],
    ln1_cache_buf: DeviceBuffer[dtype],
    mod1_cache_buf: DeviceBuffer[dtype],
    msa_cache_buf: DeviceBuffer[dtype],
    gate1_cache_buf: DeviceBuffer[dtype],
    ln2_cache_buf: DeviceBuffer[dtype],
    mod2_cache_buf: DeviceBuffer[dtype],
    mlp_cache_buf: DeviceBuffer[dtype],
    gate2_cache_buf: DeviceBuffer[dtype],
    x_mid_buf_d: DeviceBuffer[dtype],
    # Per-layer grad outputs.
    mut grad_c_layer_t: LayoutTensor[
        dtype, Layout.row_major((BATCH * T), D), MutAnyOrigin
    ],
    # Per-layer param grads.
    mut g_adaln: LayoutTensor[
        dtype, Layout.row_major(AdaLNMod[D].PARAM_SIZE), MutAnyOrigin
    ],
    mut g_msa: LayoutTensor[
        dtype,
        Layout.row_major(MultiHeadAttention[D, HEADS, T, True].PARAM_SIZE),
        MutAnyOrigin,
    ],
    mut g_mlp: LayoutTensor[
        dtype, Layout.row_major(CondMLP[D, FF].PARAM_SIZE), MutAnyOrigin
    ],
    # Scratch grads (reused across layers).
    mut sgg_t: LayoutTensor[
        dtype, Layout.row_major((BATCH * T), 3 * D), MutAnyOrigin
    ],
    mut sgbo_t: LayoutTensor[dtype, Layout.row_major((BATCH * T), D), MutAnyOrigin],
    mut sgmx_t: LayoutTensor[dtype, Layout.row_major((BATCH * T), D), MutAnyOrigin],
    mut sgmi_t: LayoutTensor[
        dtype, Layout.row_major((BATCH * T), 3 * D), MutAnyOrigin
    ],
    mut sglnout_t: LayoutTensor[
        dtype, Layout.row_major((BATCH * T), D), MutAnyOrigin
    ],
    mut sglnin_t: LayoutTensor[
        dtype, Layout.row_major((BATCH * T), D), MutAnyOrigin
    ],
    mut sgrm_t: LayoutTensor[
        dtype, Layout.row_major((BATCH * T), 6 * D), MutAnyOrigin
    ],
    mut sgsc_t: LayoutTensor[dtype, Layout.row_major((BATCH * T), D), MutAnyOrigin],
    mut grad_x_mid_t: LayoutTensor[
        dtype, Layout.row_major((BATCH * T), D), MutAnyOrigin
    ],
    # Workspaces.
    adaln_workspace: DeviceBuffer[dtype],
    msa_workspace: DeviceBuffer[dtype],
    mlp_workspace: DeviceBuffer[dtype],
) raises:
    # grad_x_next: grad_pred_raw at layer DEPTH-1; intermediate slice otherwise.
    var grad_x_next_t = LayoutTensor[
        dtype, Layout.row_major((BATCH * T), D), MutAnyOrigin
    ](
        grad_pred_raw_buf.unsafe_ptr() if d == DEPTH_VAL - 1
        else grad_x_inter_buf.unsafe_ptr() + d * (BATCH * T) * D
    )
    # grad_x_prev: grad_x_prev_pe at layer 0; intermediate slice otherwise.
    var grad_x_prev_t = LayoutTensor[
        dtype, Layout.row_major((BATCH * T), D), MutAnyOrigin
    ](
        grad_x_prev_pe_buf.unsafe_ptr() if d == 0
        else grad_x_inter_buf.unsafe_ptr() + (d - 1) * (BATCH * T) * D
    )
    # Per-layer cache slices (same as forward).
    var silu_cache_d = LayoutTensor[
        dtype, Layout.row_major((BATCH * T), D), MutAnyOrigin
    ](silu_cache_buf.unsafe_ptr() + d * (BATCH * T) * D)
    var adaln_cache_d = LayoutTensor[
        dtype, Layout.row_major((BATCH * T), AdaLNMod[D].CACHE_SIZE), MutAnyOrigin
    ](adaln_cache_buf.unsafe_ptr() + d * (BATCH * T) * AdaLNMod[D].CACHE_SIZE)
    var ln1_cache_d = LayoutTensor[
        dtype, Layout.row_major((BATCH * T), D + 1), MutAnyOrigin
    ](ln1_cache_buf.unsafe_ptr() + d * (BATCH * T) * (D + 1))
    var mod1_cache_d = LayoutTensor[
        dtype, Layout.row_major((BATCH * T), 2 * D), MutAnyOrigin
    ](mod1_cache_buf.unsafe_ptr() + d * (BATCH * T) * 2 * D)
    var msa_cache_d = LayoutTensor[
        dtype,
        Layout.row_major(BATCH, MultiHeadAttention[D, HEADS, T, True].CACHE_SIZE),
        MutAnyOrigin,
    ](
        msa_cache_buf.unsafe_ptr()
        + d * BATCH * MultiHeadAttention[D, HEADS, T, True].CACHE_SIZE
    )
    var gate1_cache_d = LayoutTensor[
        dtype, Layout.row_major((BATCH * T), 2 * D), MutAnyOrigin
    ](gate1_cache_buf.unsafe_ptr() + d * (BATCH * T) * 2 * D)
    var ln2_cache_d = LayoutTensor[
        dtype, Layout.row_major((BATCH * T), D + 1), MutAnyOrigin
    ](ln2_cache_buf.unsafe_ptr() + d * (BATCH * T) * (D + 1))
    var mod2_cache_d = LayoutTensor[
        dtype, Layout.row_major((BATCH * T), 2 * D), MutAnyOrigin
    ](mod2_cache_buf.unsafe_ptr() + d * (BATCH * T) * 2 * D)
    var mlp_cache_d = LayoutTensor[
        dtype, Layout.row_major((BATCH * T), CondMLP[D, FF].CACHE_SIZE), MutAnyOrigin
    ](mlp_cache_buf.unsafe_ptr() + d * (BATCH * T) * CondMLP[D, FF].CACHE_SIZE)
    var gate2_cache_d = LayoutTensor[
        dtype, Layout.row_major((BATCH * T), 2 * D), MutAnyOrigin
    ](gate2_cache_buf.unsafe_ptr() + d * (BATCH * T) * 2 * D)
    var x_mid_d = LayoutTensor[
        dtype, Layout.row_major((BATCH * T), D), MutAnyOrigin
    ](x_mid_buf_d.unsafe_ptr() + d * (BATCH * T) * D)

    cond_block_backward_gpu[BATCH, T, D, HEADS, FF](
        ctx, grad_x_next_t,
        adaln_params, adaln_state,
        msa_params, msa_state,
        mlp_params, mlp_state,
        silu_cache_d, adaln_cache_d,
        ln1_cache_d, mod1_cache_d, msa_cache_d, gate1_cache_d,
        ln2_cache_d, mod2_cache_d, mlp_cache_d, gate2_cache_d,
        x_mid_d,
        grad_x_prev_t, grad_c_layer_t,
        g_adaln, g_msa, g_mlp,
        sgg_t, sgbo_t, sgmx_t, sgmi_t,
        sglnout_t, sglnin_t, sgrm_t, sgsc_t,
        grad_x_mid_t,
        adaln_workspace, msa_workspace, mlp_workspace,
    )


# =============================================================================
# Phase 4 eval helper — one shot of the action-conditioned forward pipeline.
#
# Does: AE.forward -> slice_h x2 -> POS.forward -> DEPTH × run_cond_layer_forward
#       -> PROJ.forward
#
# Extracted to module scope so the trainer body doesn't double the inline
# forward path (training already has one copy). Same pattern as
# `run_cond_layer_forward` — keeps Mojo's comptime inliner from blowing up.
#
# Caller is responsible for: encoder forward (runs once per eval iter),
# action upload, pred download + MSE scoring.
# =============================================================================
def _run_eval_shot_forward[
    BATCH: Int, T: Int, H: Int, EMB: Int, ACT: Int,
    SMOOTHED: Int, PROJ_H: Int,
    PRED_HEADS: Int, PRED_FF: Int, DEPTH: Int,
](
    ctx: DeviceContext,
    # AE.
    ae_params: LayoutTensor[
        dtype,
        Layout.row_major(ActionEmbedder[T, ACT, SMOOTHED, EMB].PARAM_SIZE),
        MutAnyOrigin,
    ],
    ae_state: LayoutTensor[
        dtype,
        Layout.row_major(ActionEmbedder[T, ACT, SMOOTHED, EMB].STATE_SIZE),
        MutAnyOrigin,
    ],
    actions_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, T * ACT), MutAnyOrigin
    ],
    mut act_emb_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, T * EMB), MutAnyOrigin
    ],
    mut ae_cache_t: LayoutTensor[
        dtype,
        Layout.row_major(
            BATCH, ActionEmbedder[T, ACT, SMOOTHED, EMB].CACHE_SIZE
        ),
        MutAnyOrigin,
    ],
    ae_ws_buf: DeviceBuffer[dtype],
    # Slice IO. emb_t supplied by caller (encoder output, unchanged across
    # shots); act_emb_buf is the same DeviceBuffer that backs act_emb_t —
    # we re-view it as (BT, EMB) inside.
    emb_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, EMB), MutAnyOrigin
    ],
    act_emb_buf: DeviceBuffer[dtype],
    mut x_prev_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, EMB), MutAnyOrigin
    ],
    mut c_in_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, EMB), MutAnyOrigin
    ],
    # POS.
    pos_params: LayoutTensor[
        dtype,
        Layout.row_major(AutoDiffChain[BiasAdd[H * EMB]].PARAM_SIZE),
        MutAnyOrigin,
    ],
    pos_state: LayoutTensor[
        dtype,
        Layout.row_major(AutoDiffChain[BiasAdd[H * EMB]].STATE_SIZE),
        MutAnyOrigin,
    ],
    x_prev_bh_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ],
    mut x_prev_pe_bh_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ],
    mut pos_cache_t: LayoutTensor[
        dtype,
        Layout.row_major(
            BATCH, AutoDiffChain[BiasAdd[H * EMB]].CACHE_SIZE
        ),
        MutAnyOrigin,
    ],
    pos_ws_buf: DeviceBuffer[dtype],
    # DEPTH stack — per-layer state lists (caller indexes inside).
    mut adaln_states: List[GPUNetworkState[AdaLNMod[EMB], Adam[]]],
    mut msa_states: List[
        GPUNetworkState[
            MultiHeadAttention[EMB, PRED_HEADS, H, True], Adam[]
        ]
    ],
    mut mlp_states: List[GPUNetworkState[CondMLP[EMB, PRED_FF], Adam[]]],
    # Base DeviceBuffers consumed by run_cond_layer_forward.
    x_prev_pe_buf: DeviceBuffer[dtype],
    x_inter_buf: DeviceBuffer[dtype],
    pred_raw_buf: DeviceBuffer[dtype],
    # Cache buffers — sliced per-layer inside run_cond_layer_forward.
    silu_cache_buf: DeviceBuffer[dtype],
    adaln_cache_buf: DeviceBuffer[dtype],
    ln1_cache_buf: DeviceBuffer[dtype],
    mod1_cache_buf: DeviceBuffer[dtype],
    msa_cache_buf: DeviceBuffer[dtype],
    gate1_cache_buf: DeviceBuffer[dtype],
    ln2_cache_buf: DeviceBuffer[dtype],
    mod2_cache_buf: DeviceBuffer[dtype],
    mlp_cache_buf: DeviceBuffer[dtype],
    gate2_cache_buf: DeviceBuffer[dtype],
    raw_mod_buf: DeviceBuffer[dtype],
    x_mid_buf_d: DeviceBuffer[dtype],
    # Scratch tensors shared across layers (mut for run_cond_layer_forward).
    mut silu_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, EMB), MutAnyOrigin
    ],
    mut ln_out_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, EMB), MutAnyOrigin
    ],
    mut mod_inp_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, 3 * EMB), MutAnyOrigin
    ],
    mut mod_x_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, EMB), MutAnyOrigin
    ],
    mut branch_out_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, EMB), MutAnyOrigin
    ],
    mut gate_inp_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, 3 * EMB), MutAnyOrigin
    ],
    adaln_ws_buf: DeviceBuffer[dtype],
    msa_ws_buf: DeviceBuffer[dtype],
    mlp_ws_buf: DeviceBuffer[dtype],
    # PROJ — Tokenwise[H, Sequential[Linear, BatchNorm1D, GELU, Linear]].
    proj_params: LayoutTensor[
        dtype,
        Layout.row_major(
            Tokenwise[
                H,
                Sequential[
                    Linear[EMB, PROJ_H],
                    BatchNorm1D[PROJ_H],
                    GELU[PROJ_H],
                    Linear[PROJ_H, EMB],
                ],
            ].PARAM_SIZE
        ),
        MutAnyOrigin,
    ],
    proj_state: LayoutTensor[
        dtype,
        Layout.row_major(
            Tokenwise[
                H,
                Sequential[
                    Linear[EMB, PROJ_H],
                    BatchNorm1D[PROJ_H],
                    GELU[PROJ_H],
                    Linear[PROJ_H, EMB],
                ],
            ].STATE_SIZE
        ),
        MutAnyOrigin,
    ],
    mut proj_cache_t: LayoutTensor[
        dtype,
        Layout.row_major(
            BATCH,
            Tokenwise[
                H,
                Sequential[
                    Linear[EMB, PROJ_H],
                    BatchNorm1D[PROJ_H],
                    GELU[PROJ_H],
                    Linear[PROJ_H, EMB],
                ],
            ].CACHE_SIZE,
        ),
        MutAnyOrigin,
    ],
    proj_ws_buf: DeviceBuffer[dtype],
    pred_raw_bh_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ],
    mut pred_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ],
) raises:
    comptime AE = ActionEmbedder[T, ACT, SMOOTHED, EMB]
    comptime POS = AutoDiffChain[BiasAdd[H * EMB]]
    comptime _PredProjPerToken = Sequential[
        Linear[EMB, PROJ_H],
        BatchNorm1D[PROJ_H],
        GELU[PROJ_H],
        Linear[PROJ_H, EMB],
    ]
    comptime PROJ = Tokenwise[H, _PredProjPerToken]

    # AE: actions (BATCH, T*ACT) -> act_emb (BATCH, T*EMB).
    AE.forward_gpu[BATCH, dtype](
        ctx, act_emb_t, actions_t,
        ae_params, ae_state,
        ae_cache_t, ae_ws_buf,
    )

    # Re-view act_emb_buf as (BT, EMB) for slicing.
    var act_emb_bt_t = LayoutTensor[
        dtype, Layout.row_major(BATCH * T, EMB), MutAnyOrigin
    ](act_emb_buf.unsafe_ptr())

    # Slice first H tokens of emb + act_emb.
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

    # POS: x_prev_pe = x_prev + pos_bias.
    POS.forward_gpu[BATCH, dtype](
        ctx, x_prev_pe_bh_t, x_prev_bh_t,
        pos_params, pos_state,
        pos_cache_t, pos_ws_buf,
    )

    # DEPTH × cond_block forward.
    for d in range(DEPTH):
        run_cond_layer_forward[BATCH, H, EMB, PRED_HEADS, PRED_FF](
            ctx, d, DEPTH,
            x_prev_pe_buf, x_inter_buf, pred_raw_buf,
            c_in_t,
            adaln_states[d].params_view(),
            adaln_states[d].model_state_view(),
            msa_states[d].params_view(),
            msa_states[d].model_state_view(),
            mlp_states[d].params_view(),
            mlp_states[d].model_state_view(),
            silu_cache_buf, adaln_cache_buf,
            ln1_cache_buf, mod1_cache_buf,
            msa_cache_buf, gate1_cache_buf,
            ln2_cache_buf, mod2_cache_buf,
            mlp_cache_buf, gate2_cache_buf,
            raw_mod_buf, x_mid_buf_d,
            silu_buf_t, ln_out_buf_t, mod_inp_buf_t,
            mod_x_buf_t, branch_out_buf_t, gate_inp_buf_t,
            adaln_ws_buf, msa_ws_buf, mlp_ws_buf,
        )

    # PROJ: per-token Linear+BN+GELU+Linear over (BATCH, H*EMB).
    PROJ.forward_gpu[BATCH, dtype](
        ctx, pred_t, pred_raw_bh_t,
        proj_params, proj_state,
        proj_cache_t, proj_ws_buf,
    )


# =============================================================================
# H6 — action-conditioning diagnostic.
#
# Hypothesis: if the model is action-blind (just smoothing temporally), then
# replacing real actions with a random *permutation across the batch dimension*
# should not change the teacher-forced MSE materially. The permutation
# preserves the action *marginal distribution* exactly but breaks every
# (state, action) correlation. Compare expert MSE to shuffled MSE; ratio
# close to 1.0 ⇒ action-blind, << 1.0 ⇒ action-aware.
#
# This helper runs ONE eval iteration: 1 expert pass + S shuffled passes.
# Caller is responsible for sampling pixels/actions, running ENC, downloading
# emb_host (the target). Helper handles action shuffling, _run_eval_shot_forward,
# pred download + scoring.
#
# Extracted to module scope to keep train_lewm_offline_gpu's body small
# (compile-time inliner explodes past ~90 def-raises in one function — see
# memory: feedback_lewm_eval_block_compile_explosion.md).
#
# Returns SIMD[DType.float64, 4]: (expert_mse, shuffled_mean, shuffled_min,
# better_frac) where better_frac is the fraction of shuffled passes with MSE
# > expert (want close to 1.0 if model is action-aware).
# =============================================================================
def _run_h6_diag_shots[
    BATCH: Int, T: Int, H: Int, N_PREDS: Int, EMB: Int, ACT: Int,
    SMOOTHED: Int, PROJ_H: Int,
    PRED_HEADS: Int, PRED_FF: Int, DEPTH: Int,
](
    ctx: DeviceContext,
    eval_samples: Int,
    # Action data: actions_sample = real one-hot actions on host (BATCH*T*ACT).
    # perm_buf = scratch for within-batch permutation (size BATCH).
    actions_sample: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    perm_buf: UnsafePointer[Int, MutAnyOrigin],
    # Host/device action staging (caller uploaded expert actions before
    # calling; helper overwrites on shuffled passes).
    actions_host: HostBuffer[dtype],
    actions_buf: DeviceBuffer[dtype],
    # Target embedding (caller downloaded it from emb_buf already).
    emb_host: HostBuffer[dtype],
    # Pred output staging.
    pred_host: HostBuffer[dtype],
    pred_out_buf: DeviceBuffer[dtype],
    # All `_run_eval_shot_forward` args, passed through.
    ae_params: LayoutTensor[
        dtype,
        Layout.row_major(ActionEmbedder[T, ACT, SMOOTHED, EMB].PARAM_SIZE),
        MutAnyOrigin,
    ],
    ae_state: LayoutTensor[
        dtype,
        Layout.row_major(ActionEmbedder[T, ACT, SMOOTHED, EMB].STATE_SIZE),
        MutAnyOrigin,
    ],
    actions_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, T * ACT), MutAnyOrigin
    ],
    mut act_emb_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, T * EMB), MutAnyOrigin
    ],
    mut ae_cache_t: LayoutTensor[
        dtype,
        Layout.row_major(
            BATCH, ActionEmbedder[T, ACT, SMOOTHED, EMB].CACHE_SIZE
        ),
        MutAnyOrigin,
    ],
    ae_ws_buf: DeviceBuffer[dtype],
    emb_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, EMB), MutAnyOrigin
    ],
    act_emb_buf: DeviceBuffer[dtype],
    mut x_prev_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, EMB), MutAnyOrigin
    ],
    mut c_in_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, EMB), MutAnyOrigin
    ],
    pos_params: LayoutTensor[
        dtype,
        Layout.row_major(AutoDiffChain[BiasAdd[H * EMB]].PARAM_SIZE),
        MutAnyOrigin,
    ],
    pos_state: LayoutTensor[
        dtype,
        Layout.row_major(AutoDiffChain[BiasAdd[H * EMB]].STATE_SIZE),
        MutAnyOrigin,
    ],
    x_prev_bh_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ],
    mut x_prev_pe_bh_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ],
    mut pos_cache_t: LayoutTensor[
        dtype,
        Layout.row_major(
            BATCH, AutoDiffChain[BiasAdd[H * EMB]].CACHE_SIZE
        ),
        MutAnyOrigin,
    ],
    pos_ws_buf: DeviceBuffer[dtype],
    mut adaln_states: List[GPUNetworkState[AdaLNMod[EMB], Adam[]]],
    mut msa_states: List[
        GPUNetworkState[
            MultiHeadAttention[EMB, PRED_HEADS, H, True], Adam[]
        ]
    ],
    mut mlp_states: List[GPUNetworkState[CondMLP[EMB, PRED_FF], Adam[]]],
    x_prev_pe_buf: DeviceBuffer[dtype],
    x_inter_buf: DeviceBuffer[dtype],
    pred_raw_buf: DeviceBuffer[dtype],
    silu_cache_buf: DeviceBuffer[dtype],
    adaln_cache_buf: DeviceBuffer[dtype],
    ln1_cache_buf: DeviceBuffer[dtype],
    mod1_cache_buf: DeviceBuffer[dtype],
    msa_cache_buf: DeviceBuffer[dtype],
    gate1_cache_buf: DeviceBuffer[dtype],
    ln2_cache_buf: DeviceBuffer[dtype],
    mod2_cache_buf: DeviceBuffer[dtype],
    mlp_cache_buf: DeviceBuffer[dtype],
    gate2_cache_buf: DeviceBuffer[dtype],
    raw_mod_buf: DeviceBuffer[dtype],
    x_mid_buf_d: DeviceBuffer[dtype],
    mut silu_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, EMB), MutAnyOrigin
    ],
    mut ln_out_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, EMB), MutAnyOrigin
    ],
    mut mod_inp_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, 3 * EMB), MutAnyOrigin
    ],
    mut mod_x_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, EMB), MutAnyOrigin
    ],
    mut branch_out_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, EMB), MutAnyOrigin
    ],
    mut gate_inp_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, 3 * EMB), MutAnyOrigin
    ],
    adaln_ws_buf: DeviceBuffer[dtype],
    msa_ws_buf: DeviceBuffer[dtype],
    mlp_ws_buf: DeviceBuffer[dtype],
    proj_params: LayoutTensor[
        dtype,
        Layout.row_major(
            Tokenwise[
                H,
                Sequential[
                    Linear[EMB, PROJ_H],
                    BatchNorm1D[PROJ_H],
                    GELU[PROJ_H],
                    Linear[PROJ_H, EMB],
                ],
            ].PARAM_SIZE
        ),
        MutAnyOrigin,
    ],
    proj_state: LayoutTensor[
        dtype,
        Layout.row_major(
            Tokenwise[
                H,
                Sequential[
                    Linear[EMB, PROJ_H],
                    BatchNorm1D[PROJ_H],
                    GELU[PROJ_H],
                    Linear[PROJ_H, EMB],
                ],
            ].STATE_SIZE
        ),
        MutAnyOrigin,
    ],
    mut proj_cache_t: LayoutTensor[
        dtype,
        Layout.row_major(
            BATCH,
            Tokenwise[
                H,
                Sequential[
                    Linear[EMB, PROJ_H],
                    BatchNorm1D[PROJ_H],
                    GELU[PROJ_H],
                    Linear[PROJ_H, EMB],
                ],
            ].CACHE_SIZE,
        ),
        MutAnyOrigin,
    ],
    proj_ws_buf: DeviceBuffer[dtype],
    pred_raw_bh_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ],
    mut pred_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ],
) raises -> SIMD[DType.float64, 4]:
    """Run 1 expert pass + eval_samples shuffled passes; return stats SIMD."""
    var mse_div: Float64 = Float64(BATCH * H * EMB)

    var expert_loss: Float64 = 0.0
    var shuffled_sum: Float64 = 0.0
    var shuffled_min: Float64 = 1.0e30
    var better_count: Int = 0

    for s in range(1 + eval_samples):
        if s > 0:
            # Within-batch row permutation: action[b][:] ← actions_sample[perm[b]][:].
            # Fisher-Yates on perm_buf.
            for b in range(BATCH):
                perm_buf[b] = b
            for b in range(BATCH - 1, 0, -1):
                var j = Int(random_float64() * Float64(b + 1))
                if j > b:
                    j = b
                var tmp = perm_buf[b]
                perm_buf[b] = perm_buf[j]
                perm_buf[j] = tmp
            for b in range(BATCH):
                var src = perm_buf[b]
                for tt in range(T):
                    for k in range(ACT):
                        actions_host[
                            b * T * ACT + tt * ACT + k
                        ] = actions_sample[
                            src * T * ACT + tt * ACT + k
                        ]
            ctx.enqueue_copy(actions_buf, actions_host)

        _run_eval_shot_forward[
            BATCH, T, H, EMB, ACT, SMOOTHED, PROJ_H,
            PRED_HEADS, PRED_FF, DEPTH,
        ](
            ctx,
            ae_params, ae_state,
            actions_t, act_emb_t,
            ae_cache_t, ae_ws_buf,
            emb_t, act_emb_buf,
            x_prev_t, c_in_t,
            pos_params, pos_state,
            x_prev_bh_t, x_prev_pe_bh_t,
            pos_cache_t, pos_ws_buf,
            adaln_states, msa_states, mlp_states,
            x_prev_pe_buf, x_inter_buf, pred_raw_buf,
            silu_cache_buf, adaln_cache_buf,
            ln1_cache_buf, mod1_cache_buf,
            msa_cache_buf, gate1_cache_buf,
            ln2_cache_buf, mod2_cache_buf,
            mlp_cache_buf, gate2_cache_buf,
            raw_mod_buf, x_mid_buf_d,
            silu_buf_t, ln_out_buf_t, mod_inp_buf_t,
            mod_x_buf_t, branch_out_buf_t, gate_inp_buf_t,
            adaln_ws_buf, msa_ws_buf, mlp_ws_buf,
            proj_params, proj_state,
            proj_cache_t, proj_ws_buf,
            pred_raw_bh_t, pred_t,
        )

        ctx.enqueue_copy(pred_host, pred_out_buf)
        ctx.synchronize()
        var l: Float64 = 0.0
        for b in range(BATCH):
            for i in range(H * EMB):
                var p = Float64(pred_host[b * H * EMB + i])
                var tgt = Float64(
                    emb_host[b * T * EMB + N_PREDS * EMB + i]
                )
                var diff = p - tgt
                l += diff * diff
        l /= mse_div

        if s == 0:
            expert_loss = l
        else:
            shuffled_sum += l
            if l < shuffled_min:
                shuffled_min = l
            if l > expert_loss:
                better_count += 1

    var shuffled_mean: Float64 = (
        shuffled_sum / Float64(eval_samples) if eval_samples > 0 else 0.0
    )
    var better_frac: Float64 = (
        Float64(better_count) / Float64(eval_samples)
        if eval_samples > 0
        else 0.0
    )

    return SIMD[DType.float64, 4](
        expert_loss, shuffled_mean, shuffled_min, better_frac
    )


# =============================================================================
# Phase 4b MPC shot — one rollout of mpc_horizon autoregressive steps.
#
# Extracted to module scope to keep train_lewm_offline_gpu's body small
# (otherwise inline kernel call count past ~90 def-raises explodes compile time).
#
# Per step k ∈ [0, mpc_horizon):
#   1. Build emb window from emb_seq_host[k..k+H] -> upload to emb_buf.
#   2. Build action window from action_plan_host[k..k+H] -> upload to actions_buf.
#   3. Call _run_eval_shot_forward.
#   4. Download pred, take pred[:, H-1, :] as new emb, store in emb_seq_host[k+H].
#
# Returns: MSE(emb_seq_host[H+mpc_horizon-1], emb_goal_host) over BATCH × EMB.
# =============================================================================
def _run_mpc_shot[
    BATCH: Int, T: Int, H: Int, EMB: Int, ACT: Int,
    SMOOTHED: Int, PROJ_H: Int,
    PRED_HEADS: Int, PRED_FF: Int, DEPTH: Int,
](
    ctx: DeviceContext,
    mpc_horizon: Int,
    needed_actions: Int,
    # GPU-resident rollout state (caller pre-allocates + uploads).
    emb_start_dev_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, EMB), MutAnyOrigin
    ],
    emb_goal_dev_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, EMB), MutAnyOrigin
    ],
    mut emb_seq_dev_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, (T + 1) * EMB), MutAnyOrigin
    ],
    action_plan_dev_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, T * ACT), MutAnyOrigin
    ],
    mut score_dev_t: LayoutTensor[
        dtype, Layout.row_major(1), MutAnyOrigin
    ],
    score_dev_buf: DeviceBuffer[dtype],
    mut score_host_buf: HostBuffer[dtype],
    # All args passed through to _run_eval_shot_forward.
    ae_params: LayoutTensor[
        dtype,
        Layout.row_major(ActionEmbedder[T, ACT, SMOOTHED, EMB].PARAM_SIZE),
        MutAnyOrigin,
    ],
    ae_state: LayoutTensor[
        dtype,
        Layout.row_major(ActionEmbedder[T, ACT, SMOOTHED, EMB].STATE_SIZE),
        MutAnyOrigin,
    ],
    actions_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, T * ACT), MutAnyOrigin
    ],
    mut act_emb_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, T * EMB), MutAnyOrigin
    ],
    mut ae_cache_t: LayoutTensor[
        dtype,
        Layout.row_major(
            BATCH, ActionEmbedder[T, ACT, SMOOTHED, EMB].CACHE_SIZE
        ),
        MutAnyOrigin,
    ],
    ae_ws_buf: DeviceBuffer[dtype],
    emb_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, EMB), MutAnyOrigin
    ],
    act_emb_buf: DeviceBuffer[dtype],
    mut x_prev_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, EMB), MutAnyOrigin
    ],
    mut c_in_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, EMB), MutAnyOrigin
    ],
    pos_params: LayoutTensor[
        dtype,
        Layout.row_major(AutoDiffChain[BiasAdd[H * EMB]].PARAM_SIZE),
        MutAnyOrigin,
    ],
    pos_state: LayoutTensor[
        dtype,
        Layout.row_major(AutoDiffChain[BiasAdd[H * EMB]].STATE_SIZE),
        MutAnyOrigin,
    ],
    x_prev_bh_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ],
    mut x_prev_pe_bh_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ],
    mut pos_cache_t: LayoutTensor[
        dtype,
        Layout.row_major(
            BATCH, AutoDiffChain[BiasAdd[H * EMB]].CACHE_SIZE
        ),
        MutAnyOrigin,
    ],
    pos_ws_buf: DeviceBuffer[dtype],
    mut adaln_states: List[GPUNetworkState[AdaLNMod[EMB], Adam[]]],
    mut msa_states: List[
        GPUNetworkState[
            MultiHeadAttention[EMB, PRED_HEADS, H, True], Adam[]
        ]
    ],
    mut mlp_states: List[GPUNetworkState[CondMLP[EMB, PRED_FF], Adam[]]],
    x_prev_pe_buf: DeviceBuffer[dtype],
    x_inter_buf: DeviceBuffer[dtype],
    pred_raw_buf: DeviceBuffer[dtype],
    silu_cache_buf: DeviceBuffer[dtype],
    adaln_cache_buf: DeviceBuffer[dtype],
    ln1_cache_buf: DeviceBuffer[dtype],
    mod1_cache_buf: DeviceBuffer[dtype],
    msa_cache_buf: DeviceBuffer[dtype],
    gate1_cache_buf: DeviceBuffer[dtype],
    ln2_cache_buf: DeviceBuffer[dtype],
    mod2_cache_buf: DeviceBuffer[dtype],
    mlp_cache_buf: DeviceBuffer[dtype],
    gate2_cache_buf: DeviceBuffer[dtype],
    raw_mod_buf: DeviceBuffer[dtype],
    x_mid_buf_d: DeviceBuffer[dtype],
    mut silu_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, EMB), MutAnyOrigin
    ],
    mut ln_out_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, EMB), MutAnyOrigin
    ],
    mut mod_inp_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, 3 * EMB), MutAnyOrigin
    ],
    mut mod_x_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, EMB), MutAnyOrigin
    ],
    mut branch_out_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, EMB), MutAnyOrigin
    ],
    mut gate_inp_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, 3 * EMB), MutAnyOrigin
    ],
    adaln_ws_buf: DeviceBuffer[dtype],
    msa_ws_buf: DeviceBuffer[dtype],
    mlp_ws_buf: DeviceBuffer[dtype],
    proj_params: LayoutTensor[
        dtype,
        Layout.row_major(
            Tokenwise[
                H,
                Sequential[
                    Linear[EMB, PROJ_H],
                    BatchNorm1D[PROJ_H],
                    GELU[PROJ_H],
                    Linear[PROJ_H, EMB],
                ],
            ].PARAM_SIZE
        ),
        MutAnyOrigin,
    ],
    proj_state: LayoutTensor[
        dtype,
        Layout.row_major(
            Tokenwise[
                H,
                Sequential[
                    Linear[EMB, PROJ_H],
                    BatchNorm1D[PROJ_H],
                    GELU[PROJ_H],
                    Linear[PROJ_H, EMB],
                ],
            ].STATE_SIZE
        ),
        MutAnyOrigin,
    ],
    mut proj_cache_t: LayoutTensor[
        dtype,
        Layout.row_major(
            BATCH,
            Tokenwise[
                H,
                Sequential[
                    Linear[EMB, PROJ_H],
                    BatchNorm1D[PROJ_H],
                    GELU[PROJ_H],
                    Linear[PROJ_H, EMB],
                ],
            ].CACHE_SIZE,
        ),
        MutAnyOrigin,
    ],
    proj_ws_buf: DeviceBuffer[dtype],
    pred_raw_bh_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ],
    mut pred_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ],
) raises -> Float64:
    # ROLL_T = T + 1 (max emb_seq positions). emb_seq_t / action_plan_t /
    # emb_start_t / emb_goal_t / score_t live entirely on device; this
    # function does no per-step host<->device copy.

    # Init: emb_seq[b, 0..H-1] = emb_start[b]; rest zero.
    var tpb_rep = (4, 4, 16)
    ctx.enqueue_function[
        replicate_start_emb_kernel[BATCH, H, EMB, T + 1],
    ](
        emb_start_dev_t, emb_seq_dev_t,
        grid_dim=(
            ceildiv(BATCH, tpb_rep[0]),
            ceildiv(T + 1, tpb_rep[1]),
            ceildiv(EMB, tpb_rep[2]),
        ),
        block_dim=tpb_rep,
    )

    # Rollout: mpc_horizon GPU forward passes.
    for k in range(mpc_horizon):
        # Build emb window (positions 0..H-1) + zero rest.
        ctx.enqueue_function[
            slide_emb_window_kernel[BATCH, T, H, EMB, T + 1],
        ](
            emb_seq_dev_t, emb_t, k,
            grid_dim=(
                ceildiv(BATCH, tpb_rep[0]),
                ceildiv(T, tpb_rep[1]),
                ceildiv(EMB, tpb_rep[2]),
            ),
            block_dim=tpb_rep,
        )
        # Build action window (positions 0..H-1) + zero rest.
        ctx.enqueue_function[
            slide_actions_window_kernel[BATCH, T, H, ACT, T],
        ](
            action_plan_dev_t, actions_t, k,
            grid_dim=(
                ceildiv(BATCH, tpb_rep[0]),
                ceildiv(T, tpb_rep[1]),
                ceildiv(ACT, tpb_rep[2]),
            ),
            block_dim=tpb_rep,
        )

        _run_eval_shot_forward[
            BATCH, T, H, EMB, ACT, SMOOTHED, PROJ_H,
            PRED_HEADS, PRED_FF, DEPTH,
        ](
            ctx,
            ae_params, ae_state,
            actions_t, act_emb_t,
            ae_cache_t, ae_ws_buf,
            emb_t, act_emb_buf,
            x_prev_t, c_in_t,
            pos_params, pos_state,
            x_prev_bh_t, x_prev_pe_bh_t,
            pos_cache_t, pos_ws_buf,
            adaln_states, msa_states, mlp_states,
            x_prev_pe_buf, x_inter_buf, pred_raw_buf,
            silu_cache_buf, adaln_cache_buf,
            ln1_cache_buf, mod1_cache_buf,
            msa_cache_buf, gate1_cache_buf,
            ln2_cache_buf, mod2_cache_buf,
            mlp_cache_buf, gate2_cache_buf,
            raw_mod_buf, x_mid_buf_d,
            silu_buf_t, ln_out_buf_t, mod_inp_buf_t,
            mod_x_buf_t, branch_out_buf_t, gate_inp_buf_t,
            adaln_ws_buf, msa_ws_buf, mlp_ws_buf,
            proj_params, proj_state,
            proj_cache_t, proj_ws_buf,
            pred_raw_bh_t, pred_t,
        )

        # Store pred[:, H-1, :] into emb_seq[:, k+H, :].
        ctx.enqueue_function[
            store_pred_last_kernel[BATCH, H, EMB, T + 1],
        ](
            pred_t, emb_seq_dev_t, k,
            grid_dim=(ceildiv(BATCH, 16), ceildiv(EMB, 16)),
            block_dim=(16, 16),
        )

    # Score on device: MSE summed across BATCH × EMB → score_dev_t[0].
    # `needed_actions` (runtime) used in eval block for buffer sizing, no
    # use inside the helper itself.
    _ = needed_actions
    # Block of 32 threads (warp width). BATCH ≤ 32 in all our configs;
    # threads idx ≥ BATCH read nothing and contribute 0 to warp.sum.
    ctx.enqueue_function[
        mpc_score_kernel[BATCH, EMB, T + 1],
    ](
        emb_seq_dev_t, emb_goal_dev_t, score_dev_t,
        H + mpc_horizon - 1,
        grid_dim=1,
        block_dim=32,
    )
    ctx.enqueue_copy(score_host_buf, score_dev_buf)
    ctx.synchronize()
    return Float64(score_host_buf[0]) / Float64(BATCH * EMB)


# =============================================================================
# Phase 4c CEM eval — cross-entropy method over discrete action sequences.
#
# For each eval iter:
#   1. Initialize per-step categorical distribution to uniform.
#   2. For cem_iters CEM iterations:
#      a. Sample cem_samples plans from current distribution.
#      b. Score each via _run_mpc_shot (full autoregressive rollout to goal).
#      c. Pick top-K elites (lowest MSE).
#      d. Refit per-step categorical to elite frequencies with smoothing.
#   3. Return final best score across all CEM iters.
#
# All host-side state management (distribution, samples, scores, elite
# selection, refit). Per-step _run_mpc_shot does the GPU forward.
#
# Extracted as a module helper from the start — adds many def-raises
# (cem_iters × cem_samples = 80+ helper calls per eval iter) which would
# blow up train_lewm_offline_gpu's inline body otherwise.
# =============================================================================
def _run_cem_eval_iter[
    BATCH: Int, T: Int, H: Int, EMB: Int, ACT: Int,
    SMOOTHED: Int, PROJ_H: Int,
    PRED_HEADS: Int, PRED_FF: Int, DEPTH: Int,
](
    ctx: DeviceContext,
    mpc_horizon: Int, needed_actions: Int,
    cem_iters: Int, cem_samples: Int, cem_topk: Int,
    cem_smoothing: Float64,
    # Host scratch — distribution + samples + scores + elites stay on host.
    action_dist_host_buf: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        # (BATCH, needed_actions, ACT) — per-step categorical probs.
    action_plan_host_buf: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        # (BATCH, needed_actions, ACT) — one-hot plan staged for upload.
    sample_actions_host_buf: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        # (cem_samples, BATCH, needed_actions, ACT) — sampled plans.
    sample_scores_host_buf: UnsafePointer[Float64, MutAnyOrigin],
        # (cem_samples,) — MSE for each sample.
    elite_indices_host_buf: UnsafePointer[Int, MutAnyOrigin],
        # (cem_topk,) — indices of top-K samples by lowest score.
    # Device-side rollout state (caller pre-allocates; emb_start_dev_t and
    # emb_goal_dev_t already filled by trainer's encode step).
    emb_start_dev_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, EMB), MutAnyOrigin
    ],
    emb_goal_dev_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, EMB), MutAnyOrigin
    ],
    mut emb_seq_dev_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, (T + 1) * EMB), MutAnyOrigin
    ],
    mut action_plan_dev_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, T * ACT), MutAnyOrigin
    ],
    mut action_plan_dev_buf: DeviceBuffer[dtype],
    mut score_dev_t: LayoutTensor[
        dtype, Layout.row_major(1), MutAnyOrigin
    ],
    score_dev_buf: DeviceBuffer[dtype],
    mut score_host_buf: HostBuffer[dtype],
    # Staging HostBuffer for action_plan upload (size BATCH * T * ACT).
    mut action_plan_stage_host: HostBuffer[dtype],
    # All args passed through to _run_mpc_shot -> _run_eval_shot_forward.
    ae_params: LayoutTensor[
        dtype,
        Layout.row_major(ActionEmbedder[T, ACT, SMOOTHED, EMB].PARAM_SIZE),
        MutAnyOrigin,
    ],
    ae_state: LayoutTensor[
        dtype,
        Layout.row_major(ActionEmbedder[T, ACT, SMOOTHED, EMB].STATE_SIZE),
        MutAnyOrigin,
    ],
    actions_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, T * ACT), MutAnyOrigin
    ],
    mut act_emb_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, T * EMB), MutAnyOrigin
    ],
    mut ae_cache_t: LayoutTensor[
        dtype,
        Layout.row_major(
            BATCH, ActionEmbedder[T, ACT, SMOOTHED, EMB].CACHE_SIZE
        ),
        MutAnyOrigin,
    ],
    ae_ws_buf: DeviceBuffer[dtype],
    emb_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * T, EMB), MutAnyOrigin
    ],
    act_emb_buf: DeviceBuffer[dtype],
    mut x_prev_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, EMB), MutAnyOrigin
    ],
    mut c_in_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, EMB), MutAnyOrigin
    ],
    pos_params: LayoutTensor[
        dtype,
        Layout.row_major(AutoDiffChain[BiasAdd[H * EMB]].PARAM_SIZE),
        MutAnyOrigin,
    ],
    pos_state: LayoutTensor[
        dtype,
        Layout.row_major(AutoDiffChain[BiasAdd[H * EMB]].STATE_SIZE),
        MutAnyOrigin,
    ],
    x_prev_bh_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ],
    mut x_prev_pe_bh_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ],
    mut pos_cache_t: LayoutTensor[
        dtype,
        Layout.row_major(
            BATCH, AutoDiffChain[BiasAdd[H * EMB]].CACHE_SIZE
        ),
        MutAnyOrigin,
    ],
    pos_ws_buf: DeviceBuffer[dtype],
    mut adaln_states: List[GPUNetworkState[AdaLNMod[EMB], Adam[]]],
    mut msa_states: List[
        GPUNetworkState[
            MultiHeadAttention[EMB, PRED_HEADS, H, True], Adam[]
        ]
    ],
    mut mlp_states: List[GPUNetworkState[CondMLP[EMB, PRED_FF], Adam[]]],
    x_prev_pe_buf: DeviceBuffer[dtype],
    x_inter_buf: DeviceBuffer[dtype],
    pred_raw_buf: DeviceBuffer[dtype],
    silu_cache_buf: DeviceBuffer[dtype],
    adaln_cache_buf: DeviceBuffer[dtype],
    ln1_cache_buf: DeviceBuffer[dtype],
    mod1_cache_buf: DeviceBuffer[dtype],
    msa_cache_buf: DeviceBuffer[dtype],
    gate1_cache_buf: DeviceBuffer[dtype],
    ln2_cache_buf: DeviceBuffer[dtype],
    mod2_cache_buf: DeviceBuffer[dtype],
    mlp_cache_buf: DeviceBuffer[dtype],
    gate2_cache_buf: DeviceBuffer[dtype],
    raw_mod_buf: DeviceBuffer[dtype],
    x_mid_buf_d: DeviceBuffer[dtype],
    mut silu_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, EMB), MutAnyOrigin
    ],
    mut ln_out_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, EMB), MutAnyOrigin
    ],
    mut mod_inp_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, 3 * EMB), MutAnyOrigin
    ],
    mut mod_x_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, EMB), MutAnyOrigin
    ],
    mut branch_out_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, EMB), MutAnyOrigin
    ],
    mut gate_inp_buf_t: LayoutTensor[
        dtype, Layout.row_major(BATCH * H, 3 * EMB), MutAnyOrigin
    ],
    adaln_ws_buf: DeviceBuffer[dtype],
    msa_ws_buf: DeviceBuffer[dtype],
    mlp_ws_buf: DeviceBuffer[dtype],
    proj_params: LayoutTensor[
        dtype,
        Layout.row_major(
            Tokenwise[
                H,
                Sequential[
                    Linear[EMB, PROJ_H],
                    BatchNorm1D[PROJ_H],
                    GELU[PROJ_H],
                    Linear[PROJ_H, EMB],
                ],
            ].PARAM_SIZE
        ),
        MutAnyOrigin,
    ],
    proj_state: LayoutTensor[
        dtype,
        Layout.row_major(
            Tokenwise[
                H,
                Sequential[
                    Linear[EMB, PROJ_H],
                    BatchNorm1D[PROJ_H],
                    GELU[PROJ_H],
                    Linear[PROJ_H, EMB],
                ],
            ].STATE_SIZE
        ),
        MutAnyOrigin,
    ],
    mut proj_cache_t: LayoutTensor[
        dtype,
        Layout.row_major(
            BATCH,
            Tokenwise[
                H,
                Sequential[
                    Linear[EMB, PROJ_H],
                    BatchNorm1D[PROJ_H],
                    GELU[PROJ_H],
                    Linear[PROJ_H, EMB],
                ],
            ].CACHE_SIZE,
        ),
        MutAnyOrigin,
    ],
    proj_ws_buf: DeviceBuffer[dtype],
    pred_raw_bh_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ],
    mut pred_t: LayoutTensor[
        dtype, Layout.row_major(BATCH, H * EMB), MutAnyOrigin
    ],
) raises -> Float64:
    # Initialize uniform per-step categorical.
    var inv_act = Scalar[dtype](1.0 / Float64(ACT))
    for b in range(BATCH):
        for t in range(needed_actions):
            for a in range(ACT):
                action_dist_host_buf[
                    b * needed_actions * ACT + t * ACT + a
                ] = inv_act

    var best_overall: Float64 = 1.0e30

    for cem_it in range(cem_iters):
        # ---- Sample cem_samples plans, score each. ----
        for s in range(cem_samples):
            # Sample plan for each batch row.
            for b in range(BATCH):
                for t in range(needed_actions):
                    var r = random_float64()
                    var cumul: Float64 = 0.0
                    var picked = ACT - 1
                    for a in range(ACT):
                        cumul += Float64(
                            action_dist_host_buf[
                                b * needed_actions * ACT + t * ACT + a
                            ]
                        )
                        if r < cumul:
                            picked = a
                            break
                    # Store one-hot into both sample storage and per-call plan.
                    for a in range(ACT):
                        var v = (
                            Scalar[dtype](1.0)
                            if a == picked
                            else Scalar[dtype](0.0)
                        )
                        sample_actions_host_buf[
                            (s * BATCH + b) * needed_actions * ACT
                            + t * ACT + a
                        ] = v
                        action_plan_host_buf[
                            b * needed_actions * ACT + t * ACT + a
                        ] = v

            # Upload this action plan to device (BATCH * T * ACT).
            # Pad positions [needed_actions..T-1] with zeros (slide kernel
            # ignores them).
            for b in range(BATCH):
                for t in range(needed_actions):
                    for a in range(ACT):
                        action_plan_stage_host[
                            b * T * ACT + t * ACT + a
                        ] = action_plan_host_buf[
                            b * needed_actions * ACT + t * ACT + a
                        ]
                for t_pad in range(T - needed_actions):
                    for a in range(ACT):
                        action_plan_stage_host[
                            b * T * ACT + (needed_actions + t_pad) * ACT
                            + a
                        ] = Scalar[dtype](0.0)
            ctx.enqueue_copy(action_plan_dev_buf, action_plan_stage_host)

            # Score this plan (fully GPU rollout + score, 1 scalar back).
            var score = _run_mpc_shot[
                BATCH, T, H, EMB, ACT, SMOOTHED, PROJ_H,
                PRED_HEADS, PRED_FF, DEPTH,
            ](
                ctx, mpc_horizon, needed_actions,
                emb_start_dev_t, emb_goal_dev_t,
                emb_seq_dev_t, action_plan_dev_t,
                score_dev_t, score_dev_buf, score_host_buf,
                ae_params, ae_state,
                actions_t, act_emb_t,
                ae_cache_t, ae_ws_buf,
                emb_t, act_emb_buf,
                x_prev_t, c_in_t,
                pos_params, pos_state,
                x_prev_bh_t, x_prev_pe_bh_t,
                pos_cache_t, pos_ws_buf,
                adaln_states, msa_states, mlp_states,
                x_prev_pe_buf, x_inter_buf, pred_raw_buf,
                silu_cache_buf, adaln_cache_buf,
                ln1_cache_buf, mod1_cache_buf,
                msa_cache_buf, gate1_cache_buf,
                ln2_cache_buf, mod2_cache_buf,
                mlp_cache_buf, gate2_cache_buf,
                raw_mod_buf, x_mid_buf_d,
                silu_buf_t, ln_out_buf_t, mod_inp_buf_t,
                mod_x_buf_t, branch_out_buf_t, gate_inp_buf_t,
                adaln_ws_buf, msa_ws_buf, mlp_ws_buf,
                proj_params, proj_state,
                proj_cache_t, proj_ws_buf,
                pred_raw_bh_t, pred_t,
            )
            sample_scores_host_buf[s] = score
            if score < best_overall:
                best_overall = score

        # ---- Pick top-K elites by lowest score. ----
        # Mark used; greedy pick min.
        for k in range(cem_topk):
            elite_indices_host_buf[k] = -1
        for k in range(cem_topk):
            var best_idx: Int = -1
            var best_score: Float64 = 1.0e30
            for s in range(cem_samples):
                # Skip already-picked.
                var already_picked = False
                for kk in range(k):
                    if elite_indices_host_buf[kk] == s:
                        already_picked = True
                        break
                if not already_picked and sample_scores_host_buf[s] < best_score:
                    best_score = sample_scores_host_buf[s]
                    best_idx = s
            elite_indices_host_buf[k] = best_idx

        # ---- Refit per-step categorical from elites with smoothing. ----
        # action_dist[b, t, a] = (count[b, t, a] + smoothing) / (topk + ACT * smoothing)
        var denom = Float64(cem_topk) + Float64(ACT) * cem_smoothing
        for b in range(BATCH):
            for t in range(needed_actions):
                # Reset counts.
                for a in range(ACT):
                    action_dist_host_buf[
                        b * needed_actions * ACT + t * ACT + a
                    ] = Scalar[dtype](cem_smoothing / denom)
                # Add elite counts.
                for k in range(cem_topk):
                    var e = elite_indices_host_buf[k]
                    for a in range(ACT):
                        var v = sample_actions_host_buf[
                            (e * BATCH + b) * needed_actions * ACT
                            + t * ACT + a
                        ]
                        if v > Scalar[dtype](0.5):
                            action_dist_host_buf[
                                b * needed_actions * ACT + t * ACT + a
                            ] += Scalar[dtype](1.0 / denom)
                            break

        # Iter best for logging.
        var iter_best: Float64 = 1.0e30
        for s in range(cem_samples):
            if sample_scores_host_buf[s] < iter_best:
                iter_best = sample_scores_host_buf[s]
        print("    cem iter", cem_it, " best=", iter_best)

    return best_overall


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
    eval_steps: Int = 0,
    eval_samples: Int = 32,
    eval_seed: Int = 0xBEEF,
    mpc_horizon: Int = 0,
    cem_iters: Int = 0,
    cem_samples: Int = 64,
    cem_topk: Int = 8,
    cem_smoothing: Float64 = 0.5,
    eval_shuffle_diag: Bool = True,
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

    # Per-layer cond_block models — DEPTH copies of ADALN, MSA, MLP.
    var cpu_adalns = List[NetworkState[ADALN, Adam[]]](capacity=DEPTH)
    var cpu_msas = List[NetworkState[MSA, Adam[]]](capacity=DEPTH)
    var cpu_mlps = List[NetworkState[MLP, Adam[]]](capacity=DEPTH)
    for _ in range(DEPTH):
        var ca = NetworkState[ADALN, Adam[]]()
        ca.initialize[Xavier[]]()
        for i in range(ADALN.PARAM_SIZE):
            ca.params[i] = Scalar[dtype](0)  # AdaLN-zero
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
    for layer_idx in range(DEPTH):
        var ga = GPUNetworkState[ADALN, Adam[]](ctx)
        ga.upload_from(cpu_adalns[layer_idx], ctx)
        adaln_states.append(ga^)
        var gm = GPUNetworkState[MSA, Adam[]](ctx)
        gm.upload_from(cpu_msas[layer_idx], ctx)
        msa_states.append(gm^)
        var gf = GPUNetworkState[MLP, Adam[]](ctx)
        gf.upload_from(cpu_mlps[layer_idx], ctx)
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

    # cond_block caches — DEPTH-fold (sliced per layer in helpers).
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
    var ln2_cache_buf = ctx.enqueue_create_buffer[dtype](BTH * (EMB + 1) * DEPTH)
    var mod2_cache_buf = ctx.enqueue_create_buffer[dtype](BTH * 2 * EMB * DEPTH)
    var mlp_cache_buf = ctx.enqueue_create_buffer[dtype](
        BTH * MLP.CACHE_SIZE * DEPTH
    )
    var gate2_cache_buf = ctx.enqueue_create_buffer[dtype](BTH * 2 * EMB * DEPTH)
    var raw_mod_buf = ctx.enqueue_create_buffer[dtype](BTH * 6 * EMB * DEPTH)
    var x_mid_buf_d = ctx.enqueue_create_buffer[dtype](BTH * EMB * DEPTH)
    # Intermediate x flow between layers. (DEPTH-1) slots since layer 0 reads
    # x_prev_pe and layer DEPTH-1 writes pred_raw directly.
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
    # Backward intermediate grad_x flow between layers (DEPTH-1 slots).
    var grad_x_inter_buf = ctx.enqueue_create_buffer[dtype](
        _max_int(1, BTH * EMB * (DEPTH - 1))
    )
    # Per-layer grad_c output (single buffer, reused per layer; accumulated
    # into grad_c_buf via cb_accum_kernel).
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

    # Per-layer cache LayoutTensor views are created inside the
    # run_cond_layer_forward/backward helpers (one slice per layer d).
    var grad_c_layer_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin
    ](grad_c_layer_buf)

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
        for layer_idx in range(DEPTH):
            adaln_states[layer_idx].zero_grads(ctx)
            msa_states[layer_idx].zero_grads(ctx)
            mlp_states[layer_idx].zero_grads(ctx)

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

        # cond_block stack: DEPTH dual-branch (MSA + MLP) layers via helper.
        for d in range(DEPTH):
            run_cond_layer_forward[BATCH, H, EMB, PRED_HEADS, PRED_FF](
                ctx, d, DEPTH,
                x_prev_pe_buf, x_inter_buf, pred_raw_buf,
                c_in_t,
                adaln_states[d].params_view(),
                adaln_states[d].model_state_view(),
                msa_states[d].params_view(),
                msa_states[d].model_state_view(),
                mlp_states[d].params_view(),
                mlp_states[d].model_state_view(),
                silu_cache_buf, adaln_cache_buf,
                ln1_cache_buf, mod1_cache_buf, msa_cache_buf, gate1_cache_buf,
                ln2_cache_buf, mod2_cache_buf, mlp_cache_buf, gate2_cache_buf,
                raw_mod_buf, x_mid_buf_d,
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

        # cond_block stack backward — reverse depth loop via helper.
        # grad_c is accumulated across layers (c is shared input).
        ctx.enqueue_memset(grad_c_buf, 0)
        for d_rev in range(DEPTH):
            var d = DEPTH - 1 - d_rev
            # Bind per-layer grad views to vars (mut args can't take temps).
            var adaln_g_d = adaln_states[d].grads_view()
            var msa_g_d = msa_states[d].grads_view()
            var mlp_g_d = mlp_states[d].grads_view()
            run_cond_layer_backward[BATCH, H, EMB, PRED_HEADS, PRED_FF](
                ctx, d, DEPTH,
                grad_pred_raw_buf, grad_x_inter_buf, grad_x_prev_pe_buf,
                adaln_states[d].params_view(),
                adaln_states[d].model_state_view(),
                msa_states[d].params_view(),
                msa_states[d].model_state_view(),
                mlp_states[d].params_view(),
                mlp_states[d].model_state_view(),
                silu_cache_buf, adaln_cache_buf,
                ln1_cache_buf, mod1_cache_buf, msa_cache_buf, gate1_cache_buf,
                ln2_cache_buf, mod2_cache_buf, mlp_cache_buf, gate2_cache_buf,
                x_mid_buf_d,
                grad_c_layer_t,
                adaln_g_d, msa_g_d, mlp_g_d,
                sgg_t, sgbo_t, sgmx_t, sgmi_t,
                sglnout_t, sglnin_t, sgrm_t, sgsc_t,
                grad_x_mid_t,
                adaln_ws_buf, msa_ws_buf, mlp_ws_buf,
            )
            # Accumulate this layer's grad_c into the shared grad_c_t.
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
        for layer_idx in range(DEPTH):
            adaln_states[layer_idx].optimizer_step(ctx)
            msa_states[layer_idx].optimizer_step(ctx)
            mlp_states[layer_idx].optimizer_step(ctx)

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

    # ------------------------------------------------------------------
    # H6 — action-conditioning diagnostic (action-blind sanity check).
    #
    # For each eval iter: encode a batch, then run 1 expert pass + S
    # shuffled passes where actions are permuted within the batch
    # dimension (same action marginal, broken state-action pairing).
    # If ratio expert/shuffled_mean ≈ 1.0 the model is action-blind.
    # Independent of mpc_horizon: always runs when eval_steps > 0 and
    # eval_shuffle_diag is True.
    # ------------------------------------------------------------------
    if eval_steps > 0 and eval_shuffle_diag:
        print()
        print("==== H6: action-shuffle diagnostic (teacher-forced) ====")
        _set_seed(eval_seed)

        var perm_buf = alloc[Int](BATCH)

        var h6_sum_expert: Float64 = 0.0
        var h6_sum_shuf_mean: Float64 = 0.0
        var h6_sum_shuf_min: Float64 = 0.0
        var h6_sum_better: Float64 = 0.0

        for h6_iter in range(eval_steps):
            buf.sample_batch_fp32(
                BATCH, T, pixels_sample, actions_sample
            )
            for i in range(BT * IMG_DIM):
                pixels_host[i] = pixels_sample[i]
            for i in range(BATCH * T * ACT):
                actions_host[i] = actions_sample[i]
            ctx.enqueue_copy(pixels_buf, pixels_host)
            ctx.enqueue_copy(actions_buf, actions_host)

            ENC.forward_gpu[BT, dtype](
                ctx, emb_t, pixels_t,
                enc_state.params_view(), enc_state.model_state_view(),
                enc_cache_t, enc_ws_buf,
            )

            ctx.enqueue_copy(emb_host, emb_buf)
            ctx.synchronize()

            var stats = _run_h6_diag_shots[
                BATCH, T, H, N_PREDS, EMB, ACT, SMOOTHED, PROJ_H,
                PRED_HEADS, PRED_FF, DEPTH,
            ](
                ctx,
                eval_samples,
                actions_sample,
                perm_buf,
                actions_host, actions_buf,
                emb_host,
                pred_host, pred_out_buf,
                ae_state.params_view(), ae_state.model_state_view(),
                actions_t, act_emb_t,
                ae_cache_t, ae_ws_buf,
                emb_t, act_emb_buf,
                x_prev_t, c_in_t,
                pos_state.params_view(), pos_state.model_state_view(),
                x_prev_bh_t, x_prev_pe_bh_t,
                pos_cache_t, pos_ws_buf,
                adaln_states, msa_states, mlp_states,
                x_prev_pe_buf, x_inter_buf, pred_raw_buf,
                silu_cache_buf, adaln_cache_buf,
                ln1_cache_buf, mod1_cache_buf,
                msa_cache_buf, gate1_cache_buf,
                ln2_cache_buf, mod2_cache_buf,
                mlp_cache_buf, gate2_cache_buf,
                raw_mod_buf, x_mid_buf_d,
                silu_buf_t, ln_out_buf_t, mod_inp_buf_t,
                mod_x_buf_t, branch_out_buf_t, gate_inp_buf_t,
                adaln_ws_buf, msa_ws_buf, mlp_ws_buf,
                proj_state.params_view(), proj_state.model_state_view(),
                proj_cache_t, proj_ws_buf,
                pred_raw_bh_t, pred_t,
            )

            var expert_loss = stats[0]
            var shuf_mean = stats[1]
            var shuf_min = stats[2]
            var better_frac = stats[3]
            h6_sum_expert += expert_loss
            h6_sum_shuf_mean += shuf_mean
            h6_sum_shuf_min += shuf_min
            h6_sum_better += better_frac

            print(
                "  h6", h6_iter,
                " expert=", expert_loss,
                " shuf_mean=", shuf_mean,
                " shuf_min=", shuf_min,
                " ratio=", expert_loss / (shuf_mean + 1e-12),
                " frac_shuf_worse=", better_frac,
            )

        var avg_expert = h6_sum_expert / Float64(eval_steps)
        var avg_shuf_mean = h6_sum_shuf_mean / Float64(eval_steps)
        var avg_shuf_min = h6_sum_shuf_min / Float64(eval_steps)
        var avg_better = h6_sum_better / Float64(eval_steps)
        print()
        print(
            "H6 summary (", eval_steps,
            "iters x ", eval_samples, "shuffled samples):",
        )
        print("  expert MSE           =", avg_expert)
        print("  shuffled MSE (mean)  =", avg_shuf_mean)
        print("  shuffled MSE (min)   =", avg_shuf_min)
        print(
            "  expert/shuffled_mean =",
            avg_expert / (avg_shuf_mean + 1e-12),
            " (want < 1.0 — model is action-aware)",
        )
        print(
            "  expert/shuffled_min  =",
            avg_expert / (avg_shuf_min + 1e-12),
        )
        print(
            "  frac_shuffled_worse  =", avg_better,
            " (want > 0.5 — most shuffles are worse than expert)",
        )

        perm_buf.free()

    # ------------------------------------------------------------------
    # Phase 4 eval — random action shooter (teacher-forced)
    #
    # For each eval iteration:
    #   1. Sample fresh batch.
    #   2. Forward with EXPERT actions -> expert_loss = MSE(pred, real_emb[1:H+1]).
    #   3. For S random samples, replace actions with random one-hot and
    #      re-run AE + POS + DEPTH + PROJ (encoder unchanged).
    #   4. Report ratio expert/random — if << 1, model is action-aware.
    #
    # This is a "teacher-forced" shooter — it scores action sequences against
    # the actual observed next-frame embeddings (NOT a goal frame). True
    # autoregressive MPC with a goal frame is Phase 4b.
    # ------------------------------------------------------------------
    if eval_steps > 0 and mpc_horizon == 0:
        print()
        print("==== Phase 4 eval: random action shooter (teacher-forced) ====")
        _set_seed(eval_seed)

        var mse_div = Float64(BATCH * H * EMB)
        var sum_expert: Float64 = 0.0
        var sum_random_mean: Float64 = 0.0
        var sum_random_min: Float64 = 0.0
        var sum_better_frac: Float64 = 0.0

        for eval_iter in range(eval_steps):
            # Sample a fresh batch on host, copy to device.
            buf.sample_batch_fp32(BATCH, T, pixels_sample, actions_sample)
            for i in range(BT * IMG_DIM):
                pixels_host[i] = pixels_sample[i]
            for i in range(BATCH * T * ACT):
                actions_host[i] = actions_sample[i]
            ctx.enqueue_copy(pixels_buf, pixels_host)
            ctx.enqueue_copy(actions_buf, actions_host)

            # Encoder runs once per eval iter (pixels unchanged across S).
            ENC.forward_gpu[BT, dtype](
                ctx, emb_t, pixels_t,
                enc_state.params_view(), enc_state.model_state_view(),
                enc_cache_t, enc_ws_buf,
            )

            # Download emb once — used as target for all S shots.
            ctx.enqueue_copy(emb_host, emb_buf)
            ctx.synchronize()

            var expert_loss: Float64 = 0.0
            var random_mean: Float64 = 0.0
            var random_min: Float64 = 1e30
            var better_count: Int = 0

            # s == 0 -> expert actions (already in actions_buf).
            # s >= 1 -> random one-hot actions.
            for s in range(1 + eval_samples):
                if s > 0:
                    # Generate random one-hot actions (BATCH, T, ACT).
                    for b in range(BATCH):
                        for tt in range(T):
                            var r_act = Int(random_float64() * Float64(ACT))
                            if r_act >= ACT:
                                r_act = ACT - 1
                            for k in range(ACT):
                                actions_host[b * T * ACT + tt * ACT + k] = (
                                    Scalar[dtype](1.0)
                                    if k == r_act
                                    else Scalar[dtype](0.0)
                                )
                    ctx.enqueue_copy(actions_buf, actions_host)

                # One shot through AE + slice + POS + DEPTH × cond_block + PROJ.
                _run_eval_shot_forward[
                    BATCH, T, H, EMB, ACT, SMOOTHED, PROJ_H,
                    PRED_HEADS, PRED_FF, DEPTH,
                ](
                    ctx,
                    ae_state.params_view(), ae_state.model_state_view(),
                    actions_t, act_emb_t,
                    ae_cache_t, ae_ws_buf,
                    emb_t, act_emb_buf,
                    x_prev_t, c_in_t,
                    pos_state.params_view(), pos_state.model_state_view(),
                    x_prev_bh_t, x_prev_pe_bh_t,
                    pos_cache_t, pos_ws_buf,
                    adaln_states, msa_states, mlp_states,
                    x_prev_pe_buf, x_inter_buf, pred_raw_buf,
                    silu_cache_buf, adaln_cache_buf,
                    ln1_cache_buf, mod1_cache_buf,
                    msa_cache_buf, gate1_cache_buf,
                    ln2_cache_buf, mod2_cache_buf,
                    mlp_cache_buf, gate2_cache_buf,
                    raw_mod_buf, x_mid_buf_d,
                    silu_buf_t, ln_out_buf_t, mod_inp_buf_t,
                    mod_x_buf_t, branch_out_buf_t, gate_inp_buf_t,
                    adaln_ws_buf, msa_ws_buf, mlp_ws_buf,
                    proj_state.params_view(), proj_state.model_state_view(),
                    proj_cache_t, proj_ws_buf,
                    pred_raw_bh_t, pred_t,
                )

                # Download pred, score MSE against emb[N_PREDS:N_PREDS+H].
                ctx.enqueue_copy(pred_host, pred_out_buf)
                ctx.synchronize()
                var l: Float64 = 0.0
                for b in range(BATCH):
                    for i in range(H * EMB):
                        var p = Float64(pred_host[b * H * EMB + i])
                        var tgt = Float64(
                            emb_host[b * T * EMB + N_PREDS * EMB + i]
                        )
                        var diff = p - tgt
                        l += diff * diff
                l /= mse_div

                if s == 0:
                    expert_loss = l
                else:
                    random_mean += l
                    if l < random_min:
                        random_min = l
                    if l > expert_loss:
                        better_count += 1

            random_mean /= Float64(eval_samples)
            var better_frac = (
                Float64(better_count) / Float64(eval_samples)
            )
            sum_expert += expert_loss
            sum_random_mean += random_mean
            sum_random_min += random_min
            sum_better_frac += better_frac

            print(
                "  eval", eval_iter,
                " expert=", expert_loss,
                " rand_mean=", random_mean,
                " rand_min=", random_min,
                " ratio=", expert_loss / (random_mean + 1e-12),
                " frac_random_worse=", better_frac,
            )

        var avg_expert = sum_expert / Float64(eval_steps)
        var avg_rand_mean = sum_random_mean / Float64(eval_steps)
        var avg_rand_min = sum_random_min / Float64(eval_steps)
        var avg_better = sum_better_frac / Float64(eval_steps)
        print()
        print("Phase 4 eval summary (",
            eval_steps, "iters x ", eval_samples, "random samples):"
        )
        print("  expert MSE         =", avg_expert)
        print("  random MSE (mean)  =", avg_rand_mean)
        print("  random MSE (min)   =", avg_rand_min)
        print(
            "  expert/random_mean =", avg_expert / (avg_rand_mean + 1e-12),
            " (want < 1.0 — model is action-aware)",
        )
        print(
            "  expert/random_min  =", avg_expert / (avg_rand_min + 1e-12),
        )
        print(
            "  frac_random_worse  =", avg_better,
            " (want > 0.5 — most random plans are worse than expert)",
        )

    # ------------------------------------------------------------------
    # Phase 4b eval — autoregressive MPC against a goal frame.
    #
    # For each eval iter:
    #   1. Sample BATCH windows of length T. Use frame 0 as start, frame
    #      T-1 as goal. Encode the full window once via ENC.
    #   2. For each shot (1 expert + S random):
    #        a. Build action plan of length mpc_horizon + H - 1. The first
    #           H actions form the initial window; subsequent actions
    #           advance the window by 1 per rollout step.
    #        b. Initialize emb_seq[b, 0..H-1] = emb_start replicated H
    #           times. (We have no real history; we pad with start.)
    #        c. For k = 0..mpc_horizon-1:
    #           - Upload emb_seq[b, k..k+H-1] -> emb_buf positions 0..H-1.
    #           - Upload action_plan[b, k..k+H-1] -> actions_buf positions 0..H-1.
    #           - Run _run_eval_shot_forward.
    #           - Download pred; take pred[:, H-1, :] as new emb.
    #           - Store at emb_seq[b, k+H].
    #        d. Score: MSE(emb_seq[b, H+mpc_horizon-1], emb_goal[b]).
    #   3. Aggregate expert vs random over shots.
    #
    # mpc_horizon ≤ T - H + 1 (limited by sampled action window length).
    # ------------------------------------------------------------------
    if eval_steps > 0 and mpc_horizon > 0:
        # mpc_horizon validation — need H + mpc_horizon - 1 ≤ T actions.
        var needed_actions = H + mpc_horizon - 1
        if needed_actions > T:
            raise Error(
                "mpc_horizon too large: H + mpc_horizon - 1 > T"
                " (need bigger T or smaller horizon)"
            )

        print()
        print(
            "==== Phase 4b eval: autoregressive MPC (horizon=",
            mpc_horizon, ") ===="
        )
        _set_seed(eval_seed)

        # Host scratch — start/goal/action_plan staged on host before
        # upload; sample storage for CEM elites stays on host.
        var emb_start_host_buf = alloc[Scalar[dtype]](BATCH * EMB)
        var emb_goal_host_buf = alloc[Scalar[dtype]](BATCH * EMB)
        var action_plan_host_buf = alloc[Scalar[dtype]](
            BATCH * needed_actions * ACT
        )
        # CEM-specific host scratch.
        var cem_active = cem_iters > 0
        var _cs = cem_samples if cem_active else 1
        var _ck = cem_topk if cem_active else 1
        var action_dist_host_buf = alloc[Scalar[dtype]](
            BATCH * needed_actions * ACT
        )
        var sample_actions_host_buf = alloc[Scalar[dtype]](
            _cs * BATCH * needed_actions * ACT
        )
        var sample_scores_host_buf = alloc[Float64](_cs)
        var elite_indices_host_buf = alloc[Int](_ck)

        # GPU-resident rollout state — emb_seq sized for ROLL_T_MAX = T + 1
        # positions (worst case H + mpc_horizon ≤ T + 1).
        var emb_start_dev_buf = ctx.enqueue_create_buffer[dtype](
            BATCH * EMB
        )
        var emb_goal_dev_buf = ctx.enqueue_create_buffer[dtype](BATCH * EMB)
        var emb_seq_dev_buf = ctx.enqueue_create_buffer[dtype](
            BATCH * (T + 1) * EMB
        )
        var action_plan_dev_buf = ctx.enqueue_create_buffer[dtype](
            BATCH * T * ACT
        )
        var score_dev_buf = ctx.enqueue_create_buffer[dtype](1)
        var score_host_buf = ctx.enqueue_create_host_buffer[dtype](1)
        var emb_start_stage_host = ctx.enqueue_create_host_buffer[dtype](
            BATCH * EMB
        )
        var emb_goal_stage_host = ctx.enqueue_create_host_buffer[dtype](
            BATCH * EMB
        )
        var action_plan_stage_host = ctx.enqueue_create_host_buffer[dtype](
            BATCH * T * ACT
        )

        var emb_start_dev_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, EMB), MutAnyOrigin
        ](emb_start_dev_buf.unsafe_ptr())
        var emb_goal_dev_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, EMB), MutAnyOrigin
        ](emb_goal_dev_buf.unsafe_ptr())
        var emb_seq_dev_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, (T + 1) * EMB), MutAnyOrigin
        ](emb_seq_dev_buf.unsafe_ptr())
        var action_plan_dev_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, T * ACT), MutAnyOrigin
        ](action_plan_dev_buf.unsafe_ptr())
        var score_dev_t = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ](score_dev_buf.unsafe_ptr())

        var sum_expert_mpc: Float64 = 0.0
        var sum_random_mean_mpc: Float64 = 0.0
        var sum_random_min_mpc: Float64 = 0.0
        var sum_better_frac_mpc: Float64 = 0.0
        var sum_cem: Float64 = 0.0
        var cem_better_expert: Int = 0
        var cem_better_random_min: Int = 0

        for eval_iter in range(eval_steps):
            # Sample fresh batch, copy to device, encode.
            buf.sample_batch_fp32(BATCH, T, pixels_sample, actions_sample)
            for i in range(BT * IMG_DIM):
                pixels_host[i] = pixels_sample[i]
            ctx.enqueue_copy(pixels_buf, pixels_host)

            ENC.forward_gpu[BT, dtype](
                ctx, emb_t, pixels_t,
                enc_state.params_view(), enc_state.model_state_view(),
                enc_cache_t, enc_ws_buf,
            )
            ctx.enqueue_copy(emb_host, emb_buf)
            ctx.synchronize()

            # Extract start (frame 0) + goal (frame T-1) per batch row,
            # upload both to device.
            for b in range(BATCH):
                for d in range(EMB):
                    emb_start_stage_host[b * EMB + d] = (
                        emb_host[b * T * EMB + d]
                    )
                    emb_goal_stage_host[b * EMB + d] = (
                        emb_host[b * T * EMB + (T - 1) * EMB + d]
                    )
            ctx.enqueue_copy(emb_start_dev_buf, emb_start_stage_host)
            ctx.enqueue_copy(emb_goal_dev_buf, emb_goal_stage_host)

            var expert_loss_mpc: Float64 = 0.0
            var random_mean_mpc: Float64 = 0.0
            var random_min_mpc: Float64 = 1e30
            var better_count_mpc: Int = 0

            for s in range(1 + eval_samples):
                # Build action plan (BATCH, needed_actions, ACT) on host.
                if s == 0:
                    for b in range(BATCH):
                        for ti in range(needed_actions):
                            for k in range(ACT):
                                action_plan_host_buf[
                                    b * needed_actions * ACT + ti * ACT + k
                                ] = actions_sample[
                                    b * T * ACT + ti * ACT + k
                                ]
                else:
                    for b in range(BATCH):
                        for ti in range(needed_actions):
                            var r_act = Int(
                                random_float64() * Float64(ACT)
                            )
                            if r_act >= ACT:
                                r_act = ACT - 1
                            for k in range(ACT):
                                action_plan_host_buf[
                                    b * needed_actions * ACT
                                    + ti * ACT + k
                                ] = (
                                    Scalar[dtype](1.0)
                                    if k == r_act
                                    else Scalar[dtype](0.0)
                                )

                # Stage action_plan to (BATCH, T, ACT) layout (positions
                # [needed_actions..T-1] zero-padded; slide_actions_window
                # only reads up to k+H-1 ≤ needed_actions-1).
                for b in range(BATCH):
                    for ti in range(needed_actions):
                        for k in range(ACT):
                            action_plan_stage_host[
                                b * T * ACT + ti * ACT + k
                            ] = action_plan_host_buf[
                                b * needed_actions * ACT + ti * ACT + k
                            ]
                    for t_pad in range(T - needed_actions):
                        for k in range(ACT):
                            action_plan_stage_host[
                                b * T * ACT
                                + (needed_actions + t_pad) * ACT + k
                            ] = Scalar[dtype](0.0)
                ctx.enqueue_copy(
                    action_plan_dev_buf, action_plan_stage_host
                )

                var l = _run_mpc_shot[
                    BATCH, T, H, EMB, ACT, SMOOTHED, PROJ_H,
                    PRED_HEADS, PRED_FF, DEPTH,
                ](
                    ctx,
                    mpc_horizon, needed_actions,
                    emb_start_dev_t, emb_goal_dev_t,
                    emb_seq_dev_t, action_plan_dev_t,
                    score_dev_t, score_dev_buf, score_host_buf,
                    ae_state.params_view(), ae_state.model_state_view(),
                    actions_t, act_emb_t,
                    ae_cache_t, ae_ws_buf,
                    emb_t, act_emb_buf,
                    x_prev_t, c_in_t,
                    pos_state.params_view(), pos_state.model_state_view(),
                    x_prev_bh_t, x_prev_pe_bh_t,
                    pos_cache_t, pos_ws_buf,
                    adaln_states, msa_states, mlp_states,
                    x_prev_pe_buf, x_inter_buf, pred_raw_buf,
                    silu_cache_buf, adaln_cache_buf,
                    ln1_cache_buf, mod1_cache_buf,
                    msa_cache_buf, gate1_cache_buf,
                    ln2_cache_buf, mod2_cache_buf,
                    mlp_cache_buf, gate2_cache_buf,
                    raw_mod_buf, x_mid_buf_d,
                    silu_buf_t, ln_out_buf_t, mod_inp_buf_t,
                    mod_x_buf_t, branch_out_buf_t, gate_inp_buf_t,
                    adaln_ws_buf, msa_ws_buf, mlp_ws_buf,
                    proj_state.params_view(), proj_state.model_state_view(),
                    proj_cache_t, proj_ws_buf,
                    pred_raw_bh_t, pred_t,
                )

                if s == 0:
                    expert_loss_mpc = l
                else:
                    random_mean_mpc += l
                    if l < random_min_mpc:
                        random_min_mpc = l
                    if l > expert_loss_mpc:
                        better_count_mpc += 1

            random_mean_mpc /= Float64(eval_samples)
            var better_frac_mpc = (
                Float64(better_count_mpc) / Float64(eval_samples)
            )
            sum_expert_mpc += expert_loss_mpc
            sum_random_mean_mpc += random_mean_mpc
            sum_random_min_mpc += random_min_mpc
            sum_better_frac_mpc += better_frac_mpc

            print(
                "  mpc eval", eval_iter,
                " expert=", expert_loss_mpc,
                " rand_mean=", random_mean_mpc,
                " rand_min=", random_min_mpc,
                " ratio=", expert_loss_mpc / (random_mean_mpc + 1e-12),
                " frac_random_worse=", better_frac_mpc,
            )

            # ---- CEM eval for this iter (optional). ----
            if cem_active:
                print("  -- CEM eval iter", eval_iter, "--")
                var cem_score = _run_cem_eval_iter[
                    BATCH, T, H, EMB, ACT, SMOOTHED, PROJ_H,
                    PRED_HEADS, PRED_FF, DEPTH,
                ](
                    ctx,
                    mpc_horizon, needed_actions,
                    cem_iters, cem_samples, cem_topk, cem_smoothing,
                    action_dist_host_buf, action_plan_host_buf,
                    sample_actions_host_buf, sample_scores_host_buf,
                    elite_indices_host_buf,
                    emb_start_dev_t, emb_goal_dev_t,
                    emb_seq_dev_t, action_plan_dev_t,
                    action_plan_dev_buf,
                    score_dev_t, score_dev_buf, score_host_buf,
                    action_plan_stage_host,
                    ae_state.params_view(), ae_state.model_state_view(),
                    actions_t, act_emb_t,
                    ae_cache_t, ae_ws_buf,
                    emb_t, act_emb_buf,
                    x_prev_t, c_in_t,
                    pos_state.params_view(), pos_state.model_state_view(),
                    x_prev_bh_t, x_prev_pe_bh_t,
                    pos_cache_t, pos_ws_buf,
                    adaln_states, msa_states, mlp_states,
                    x_prev_pe_buf, x_inter_buf, pred_raw_buf,
                    silu_cache_buf, adaln_cache_buf,
                    ln1_cache_buf, mod1_cache_buf,
                    msa_cache_buf, gate1_cache_buf,
                    ln2_cache_buf, mod2_cache_buf,
                    mlp_cache_buf, gate2_cache_buf,
                    raw_mod_buf, x_mid_buf_d,
                    silu_buf_t, ln_out_buf_t, mod_inp_buf_t,
                    mod_x_buf_t, branch_out_buf_t, gate_inp_buf_t,
                    adaln_ws_buf, msa_ws_buf, mlp_ws_buf,
                    proj_state.params_view(), proj_state.model_state_view(),
                    proj_cache_t, proj_ws_buf,
                    pred_raw_bh_t, pred_t,
                )
                sum_cem += cem_score
                if cem_score < expert_loss_mpc:
                    cem_better_expert += 1
                if cem_score < random_min_mpc:
                    cem_better_random_min += 1
                print(
                    "  cem eval", eval_iter, " best=", cem_score,
                    " vs expert=", expert_loss_mpc,
                    " vs rand_min=", random_min_mpc,
                    " cem/expert=", cem_score / (expert_loss_mpc + 1e-12),
                    " cem/rand_min=", cem_score / (random_min_mpc + 1e-12),
                )

        var avg_expert_mpc = sum_expert_mpc / Float64(eval_steps)
        var avg_rand_mean_mpc = (
            sum_random_mean_mpc / Float64(eval_steps)
        )
        var avg_rand_min_mpc = sum_random_min_mpc / Float64(eval_steps)
        var avg_better_mpc = sum_better_frac_mpc / Float64(eval_steps)
        print()
        print("Phase 4b MPC eval summary (",
            eval_steps, "iters x ", eval_samples, "shots, horizon=",
            mpc_horizon, "):"
        )
        print("  expert MSE         =", avg_expert_mpc)
        print("  random MSE (mean)  =", avg_rand_mean_mpc)
        print("  random MSE (min)   =", avg_rand_min_mpc)
        print(
            "  expert/random_mean =",
            avg_expert_mpc / (avg_rand_mean_mpc + 1e-12),
            " (want < 1.0)",
        )
        print(
            "  expert/random_min  =",
            avg_expert_mpc / (avg_rand_min_mpc + 1e-12),
            " (want < 1.0 — paper exit criterion 0.5)",
        )
        print(
            "  frac_random_worse  =", avg_better_mpc,
            " (want > 0.5)",
        )

        if cem_active:
            var avg_cem = sum_cem / Float64(eval_steps)
            var cem_vs_expert_frac = (
                Float64(cem_better_expert) / Float64(eval_steps)
            )
            var cem_vs_rmin_frac = (
                Float64(cem_better_random_min) / Float64(eval_steps)
            )
            print()
            print("Phase 4c CEM eval summary (",
                eval_steps, "iters x ", cem_iters, "CEM iters x ",
                cem_samples, "samples, top", cem_topk, "):"
            )
            print("  cem MSE (best)     =", avg_cem)
            print(
                "  cem/expert         =",
                avg_cem / (avg_expert_mpc + 1e-12),
                " (want < 1.0 — CEM beats expert in latent)",
            )
            print(
                "  cem/random_min     =",
                avg_cem / (avg_rand_min_mpc + 1e-12),
                " (want < 1.0 — CEM beats best random)",
            )
            print(
                "  cem_better_expert  =", cem_vs_expert_frac,
                " (want > 0.5)",
            )
            print(
                "  cem_better_rmin    =", cem_vs_rmin_frac,
                " (want > 0.5 — CEM finds better-than-random plans)",
            )

        emb_start_host_buf.free()
        emb_goal_host_buf.free()
        action_plan_host_buf.free()
        action_dist_host_buf.free()
        sample_actions_host_buf.free()
        sample_scores_host_buf.free()
        elite_indices_host_buf.free()

    pixels_sample.free()
    actions_sample.free()
