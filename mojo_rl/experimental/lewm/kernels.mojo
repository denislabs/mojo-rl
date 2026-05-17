"""LeWM GPU kernels + per-layer / per-shot orchestration helpers.

Module-level helpers shared by `LeWMTrainer` (`offline_trainer.mojo`).
Kept module-level on purpose: inlining them into the trainer's
`train_step` / eval methods explodes Mojo compile time
(see memory: `feedback_lewm_depth_loop_compile_explosion` and
`feedback_lewm_eval_block_compile_explosion`).

Contents (in declaration order):

  GPU kernels (low-level launch-shape primitives):
    - slice_h_kernel, scatter_h_kernel, scatter_target_neg_kernel
    - accumulate_emb_kernel, replicate_start_emb_kernel
    - slide_emb_window_kernel, slide_actions_window_kernel
    - store_pred_last_kernel, mpc_score_kernel

  Per-layer training orchestration (used inside `train_step`):
    - run_cond_layer_forward, run_cond_layer_backward

  Per-shot eval orchestration (used inside `eval_h6` / `eval_h7_*` /
  `eval_random_shots` / `eval_mpc_cem`):
    - _run_eval_shot_forward, _run_h6_diag_shots,
      _run_mpc_shot, _run_cem_eval_iter

The 6 model groups they operate over:
  - ENC (LeWMEncoder)        Conv2D-based ViT
  - AE  (ActionEmbedder)
  - POS (AutoDiffChain[BiasAdd[H*EMB]])
  - ADALN (Linear[EMB, 3*EMB], zero-init for AdaLN-zero identity)
  - MSA (MultiHeadAttention[EMB, PRED_HEADS, H, causal=True])
  - PROJ (Tokenwise[H, Sequential[Linear, BatchNorm1D, GELU, Linear]])
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


# =============================================================================
# Pixel conversion kernel: uint8 (HWC or CHW) -> fp32 CHW + /255 normalize.
#
# Replaces the host-side scalar HWC->CHW permute + uint8->fp32 conversion
# loops with one GPU kernel pass. One thread per output element. Source
# layout is selected at compile time via INPUT_LAYOUT_HWC (true for the
# PushT HDF5 path, false for the Pong replay buffer).
# =============================================================================


def pixels_uint8_to_fp32_kernel[
    BT: Int, IN_CH: Int, IMG: Int, INPUT_LAYOUT_HWC: Bool,
](
    src_u8: LayoutTensor[
        DType.uint8,
        Layout.row_major(BT, IN_CH * IMG * IMG),
        MutAnyOrigin,
    ],
    dst_fp32: LayoutTensor[
        dtype,
        Layout.row_major(BT, IN_CH * IMG * IMG),
        MutAnyOrigin,
    ],
):
    var bt = Int(global_idx.x)
    var c = Int(global_idx.y)
    var hw = Int(global_idx.z)
    if bt >= BT or c >= IN_CH or hw >= IMG * IMG:
        return

    var h = hw // IMG
    var w = hw - h * IMG

    var src_offset_in_frame: Int
    comptime if INPUT_LAYOUT_HWC:
        src_offset_in_frame = h * (IMG * IN_CH) + w * IN_CH + c
    else:
        src_offset_in_frame = c * (IMG * IMG) + h * IMG + w

    var dst_offset_in_frame = c * (IMG * IMG) + h * IMG + w
    var byte_val = src_u8[bt, src_offset_in_frame]
    dst_fp32[bt, dst_offset_in_frame] = (
        Scalar[dtype](Int(byte_val)) * Scalar[dtype](1.0 / 255.0)
    )


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


