"""LeWM offline trainer — struct-based GPU training loop.

Two structs:

  - `LeWMGPUState[...]` — owns all GPU device/host buffers + per-model
    GPUNetworkStates. Comptime aliases for every model type are hoisted
    to struct level so they get instantiated once per (model-shape)
    specialization rather than once per call-site of the trainer.

  - `LeWMTrainer[..., BUF]` — owns a clip/window buffer, hyperparams,
    and per-run scalar EMAs. `BUF` is a comptime type parameter that must
    conform to `LeWMBuffer` (see `lewm_buffer.LeWMBuffer`): expose
    `INPUT_LAYOUT_HWC: Bool` comptime field and
    `sample_batch_uint8(B, T, pixels_u8_out, actions_out) raises` method.
    Concrete buffers: `pong_buffer.PongBuffer` (in-RAM CHW uint8) and
    `pusht_sampler.LewmPushTSampler` (HDF5-backed HWC uint8). Each phase
    of training becomes its own method (`train_step`, `eval_h6`,
    `eval_random_shots`, `eval_mpc_cem`, `eval_h7_closed_loop_drift`,
    `run`, `run_eval`).

`train_lewm_offline_gpu` (Pong) and `train_lewm_offline_gpu_pusht`
(PushT HDF5) are the thin entry points that construct the appropriate
buffer + state + trainer and call `trainer.run(...)`. `eval_lewm_offline_gpu`
+ `eval_lewm_offline_gpu_pusht` are the checkpoint-load + eval-only
counterparts.

Module-level GPU kernels and per-layer / per-shot orchestration helpers
(run_cond_layer_forward/backward, _run_eval_shot_forward,
_run_h6_diag_shots, _run_mpc_shot, _run_cem_eval_iter, plus the 9
kernels) live in `kernels.mojo` — kept module-level on purpose
because inlining them explodes Mojo compile time.
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
from .pusht_sampler import LewmPushTSampler
from .lewm_buffer import LeWMBuffer
from .lewm_checkpoint import _write_gpu_net_sections, _read_gpu_net_sections

from ...nn.checkpoint import (
    write_checkpoint_header,
    write_metadata_section,
    parse_checkpoint_header,
    read_checkpoint_file,
    read_metadata_section,
    save_checkpoint_file,
)
from .kernels import (
    run_cond_layer_forward, run_cond_layer_backward,
    _run_eval_shot_forward, _run_h6_diag_shots, _run_mpc_shot, _run_cem_eval_iter,
    slice_h_kernel, scatter_h_kernel, scatter_target_neg_kernel,
    accumulate_emb_kernel, replicate_start_emb_kernel,
    slide_emb_window_kernel, slide_actions_window_kernel,
    store_pred_last_kernel, mpc_score_kernel,
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


struct LeWMGPUState[
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
]:
    """All GPU device + host buffers and per-model GPUNetworkStates.

    Comptime aliases for the model types are hoisted to struct level so
    they instantiate once per specialization (mirrors the TDMPC2Agent
    pattern of avoiding re-instantiation across `train_gpu[ENV, n_envs]`
    call sites).
    """

    # ── Derived dimension aliases ────────────────────────────────────
    comptime IMG_DIM: Int = Self.IN_CH * Self.IMG * Self.IMG
    comptime BT: Int = Self.BATCH * Self.T
    comptime BTH: Int = Self.BATCH * Self.H

    # ── Model type aliases (hoisted) ─────────────────────────────────
    comptime ENC = LeWMEncoder[
        Self.IN_CH, Self.IMG, Self.IMG, Self.PATCH, Self.HIDDEN, Self.ENC_HEADS, Self.ENC_LAYERS, Self.N_PATCHES,
        Self.EMB, 2, Self.PROJ_H,
    ]
    comptime AE = ActionEmbedder[Self.T, Self.ACT, Self.SMOOTHED, Self.EMB]
    comptime POS = AutoDiffChain[BiasAdd[Self.H * Self.EMB]]
    comptime ADALN = AdaLNMod[Self.EMB]
    comptime MSA = MultiHeadAttention[Self.EMB, Self.PRED_HEADS, Self.H, True]
    comptime MLP = CondMLP[Self.EMB, Self.PRED_FF]
    comptime _PredProjPerToken = Sequential[
        Linear[Self.EMB, Self.PROJ_H],
        BatchNorm1D[Self.PROJ_H],
        GELU[Self.PROJ_H],
        Linear[Self.PROJ_H, Self.EMB],
    ]
    comptime PROJ = Tokenwise[Self.H, Self._PredProjPerToken]
    comptime SIG = SIGRegOp[Self.EMB, Self.T, Self.SIG_NUM_PROJ, Self.SIG_KNOTS]
    comptime SIG_WS_SIZE = Self.SIG.workspace_size_for[Self.BATCH]()

    # ── Shared (single-instance) GPUNetworkStates ────────────────────
    var enc_state: GPUNetworkState[Self.ENC, Adam[]]
    var ae_state: GPUNetworkState[Self.AE, Adam[]]
    var pos_state: GPUNetworkState[Self.POS, Adam[]]
    var proj_state: GPUNetworkState[Self.PROJ, Adam[]]

    # ── Per-layer cond_block GPUNetworkStates (Self.DEPTH copies each) ────
    var adaln_states: List[GPUNetworkState[Self.ADALN, Adam[]]]
    var msa_states: List[GPUNetworkState[Self.MSA, Adam[]]]
    var mlp_states: List[GPUNetworkState[Self.MLP, Adam[]]]

    # ── Activations / caches / scratch (device) ──────────────────────
    var pixels_buf: DeviceBuffer[dtype]
    var actions_buf: DeviceBuffer[dtype]
    var emb_buf: DeviceBuffer[dtype]
    var enc_cache_buf: DeviceBuffer[dtype]
    var enc_ws_buf: DeviceBuffer[dtype]

    var act_emb_buf: DeviceBuffer[dtype]
    var ae_cache_buf: DeviceBuffer[dtype]
    var ae_ws_buf: DeviceBuffer[dtype]

    var x_prev_buf: DeviceBuffer[dtype]
    var x_prev_pe_buf: DeviceBuffer[dtype]
    var pos_cache_buf: DeviceBuffer[dtype]
    var pos_ws_buf: DeviceBuffer[dtype]
    var c_in_buf: DeviceBuffer[dtype]

    var pred_raw_buf: DeviceBuffer[dtype]
    var pred_out_buf: DeviceBuffer[dtype]
    var proj_cache_buf: DeviceBuffer[dtype]
    var proj_ws_buf: DeviceBuffer[dtype]

    # cond_block caches — Self.DEPTH-fold (sliced per layer in helpers).
    var silu_cache_buf: DeviceBuffer[dtype]
    var adaln_cache_buf: DeviceBuffer[dtype]
    var ln1_cache_buf: DeviceBuffer[dtype]
    var mod1_cache_buf: DeviceBuffer[dtype]
    var msa_cache_buf: DeviceBuffer[dtype]
    var gate1_cache_buf: DeviceBuffer[dtype]
    var ln2_cache_buf: DeviceBuffer[dtype]
    var mod2_cache_buf: DeviceBuffer[dtype]
    var mlp_cache_buf: DeviceBuffer[dtype]
    var gate2_cache_buf: DeviceBuffer[dtype]
    var raw_mod_buf: DeviceBuffer[dtype]
    var x_mid_buf_d: DeviceBuffer[dtype]
    # Intermediate x flow between layers. (Self.DEPTH-1) slots since layer 0 reads
    # x_prev_pe and layer Self.DEPTH-1 writes pred_raw directly.
    var x_inter_buf: DeviceBuffer[dtype]

    # cond_block forward scratch (reused across MSA and MLP branches).
    var silu_buf_d: DeviceBuffer[dtype]
    var ln_out_buf_d: DeviceBuffer[dtype]
    var mod_inp_buf_d: DeviceBuffer[dtype]
    var mod_x_buf_d: DeviceBuffer[dtype]
    var branch_out_buf_d: DeviceBuffer[dtype]
    var gate_inp_buf_d: DeviceBuffer[dtype]

    # cond_block workspaces (shared with model forward_gpu calls).
    var adaln_ws_buf: DeviceBuffer[dtype]
    var msa_ws_buf: DeviceBuffer[dtype]
    var mlp_ws_buf: DeviceBuffer[dtype]

    # cond_block backward scratch (reused across all Self.DEPTH layers).
    var sgg_buf: DeviceBuffer[dtype]
    var sgbo_buf: DeviceBuffer[dtype]
    var sgmx_buf: DeviceBuffer[dtype]
    var sgmi_buf: DeviceBuffer[dtype]
    var sglnout_buf: DeviceBuffer[dtype]
    var sglnin_buf: DeviceBuffer[dtype]
    var sgrm_buf: DeviceBuffer[dtype]
    var sgsc_buf: DeviceBuffer[dtype]
    var grad_x_mid_buf: DeviceBuffer[dtype]
    # Backward intermediate grad_x flow between layers (Self.DEPTH-1 slots).
    var grad_x_inter_buf: DeviceBuffer[dtype]
    # Per-layer grad_c output (single buffer, reused per layer; accumulated
    # into grad_c_buf via cb_accum_kernel).
    var grad_c_layer_buf: DeviceBuffer[dtype]

    # SIGReg buffers (forward + backward).
    var sigreg_out_buf: DeviceBuffer[dtype]
    var sigreg_cache_buf: DeviceBuffer[dtype]
    var sigreg_grad_out_buf: DeviceBuffer[dtype]
    var sigreg_grad_emb_buf: DeviceBuffer[dtype]
    var sigreg_ws_buf: DeviceBuffer[dtype]

    # Gradient buffers (device).
    var grad_pred_buf: DeviceBuffer[dtype]
    var grad_pred_raw_buf: DeviceBuffer[dtype]
    var grad_x_prev_buf: DeviceBuffer[dtype]
    var grad_x_prev_pe_buf: DeviceBuffer[dtype]
    var grad_c_buf: DeviceBuffer[dtype]
    var grad_emb_buf: DeviceBuffer[dtype]
    var grad_act_emb_buf: DeviceBuffer[dtype]
    var grad_actions_buf: DeviceBuffer[dtype]
    var grad_pixels_buf: DeviceBuffer[dtype]

    # Pinned host buffers for sampled data + per-step loss compute.
    # `pixels_u8_host` + `pixels_u8_buf` carry uint8 pixels (HWC or CHW per
    # buf.INPUT_LAYOUT_HWC); the GPU `pixels_uint8_to_fp32_kernel` does the
    # permute + /255 cast, writing the fp32 result to `pixels_buf`.
    var pixels_u8_host: HostBuffer[DType.uint8]
    var pixels_u8_buf: DeviceBuffer[DType.uint8]
    var actions_host: HostBuffer[dtype]
    var pred_host: HostBuffer[dtype]
    var target_host: HostBuffer[dtype]
    var grad_pred_host: HostBuffer[dtype]
    var sigreg_out_host: HostBuffer[dtype]
    var emb_host: HostBuffer[dtype]

    # Small read-only snapshot of the most recently sampled expert actions.
    # Used only by `eval_h6` (which permutes `actions_host` in-place and
    # needs the unshuffled originals as ground truth). Train_step + the
    # other eval phases skip this — they sample directly into the pinned
    # `actions_host` buffer.
    var actions_sample: UnsafePointer[Scalar[dtype], MutAnyOrigin]

    def __init__(out self, ctx: DeviceContext, lambda_sigreg: Float64) raises:
        # ------------------------------------------------------------------
        # Init on CPU, upload to GPU.
        # ------------------------------------------------------------------
        # Shared (single-instance) models.
        var cpu_enc = NetworkState[Self.ENC, Adam[]]()
        var cpu_ae = NetworkState[Self.AE, Adam[]]()
        var cpu_pos = NetworkState[Self.POS, Adam[]]()
        var cpu_proj = NetworkState[Self.PROJ, Adam[]]()
        cpu_enc.initialize[Xavier[]]()
        cpu_ae.initialize[Xavier[]]()
        cpu_pos.initialize[Xavier[]]()
        cpu_proj.initialize[Xavier[]]()
        for i in range(Self.POS.PARAM_SIZE):
            cpu_pos.params[i] = Scalar[dtype](0)

        self.enc_state = GPUNetworkState[Self.ENC, Adam[]](ctx)
        self.ae_state = GPUNetworkState[Self.AE, Adam[]](ctx)
        self.pos_state = GPUNetworkState[Self.POS, Adam[]](ctx)
        self.proj_state = GPUNetworkState[Self.PROJ, Adam[]](ctx)
        self.enc_state.upload_from(cpu_enc, ctx)
        self.ae_state.upload_from(cpu_ae, ctx)
        self.pos_state.upload_from(cpu_pos, ctx)
        self.proj_state.upload_from(cpu_proj, ctx)

        # Per-layer cond_block models — Self.DEPTH copies of ADALN, MSA, MLP.
        var cpu_adalns = List[NetworkState[Self.ADALN, Adam[]]](capacity=Self.DEPTH)
        var cpu_msas = List[NetworkState[Self.MSA, Adam[]]](capacity=Self.DEPTH)
        var cpu_mlps = List[NetworkState[Self.MLP, Adam[]]](capacity=Self.DEPTH)
        for _ in range(Self.DEPTH):
            var ca = NetworkState[Self.ADALN, Adam[]]()
            ca.initialize[Xavier[]]()
            for i in range(Self.ADALN.PARAM_SIZE):
                ca.params[i] = Scalar[dtype](0)  # AdaLN-zero
            cpu_adalns.append(ca^)
            var cm = NetworkState[Self.MSA, Adam[]]()
            cm.initialize[Xavier[]]()
            cpu_msas.append(cm^)
            var cf = NetworkState[Self.MLP, Adam[]]()
            cf.initialize[Xavier[]]()
            cpu_mlps.append(cf^)

        self.adaln_states = List[GPUNetworkState[Self.ADALN, Adam[]]](capacity=Self.DEPTH)
        self.msa_states = List[GPUNetworkState[Self.MSA, Adam[]]](capacity=Self.DEPTH)
        self.mlp_states = List[GPUNetworkState[Self.MLP, Adam[]]](capacity=Self.DEPTH)
        for layer_idx in range(Self.DEPTH):
            var ga = GPUNetworkState[Self.ADALN, Adam[]](ctx)
            ga.upload_from(cpu_adalns[layer_idx], ctx)
            self.adaln_states.append(ga^)
            var gm = GPUNetworkState[Self.MSA, Adam[]](ctx)
            gm.upload_from(cpu_msas[layer_idx], ctx)
            self.msa_states.append(gm^)
            var gf = GPUNetworkState[Self.MLP, Adam[]](ctx)
            gf.upload_from(cpu_mlps[layer_idx], ctx)
            self.mlp_states.append(gf^)

        # ------------------------------------------------------------------
        # Allocate device buffers for activations / caches / grads / scratch.
        # All sizes are comptime; we allocate at least 1 element to keep
        # DeviceBuffer construction valid for zero-CACHE/WS ops.
        # ------------------------------------------------------------------
        self.pixels_buf = ctx.enqueue_create_buffer[dtype](Self.BT * Self.IMG_DIM)
        self.actions_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.T * Self.ACT)
        self.emb_buf = ctx.enqueue_create_buffer[dtype](Self.BT * Self.EMB)
        self.enc_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BT * Self.ENC.CACHE_SIZE)
        self.enc_ws_buf = ctx.enqueue_create_buffer[dtype](
            _max_int(1, Self.BT * Self.ENC.WORKSPACE_SIZE_PER_SAMPLE)
        )

        self.act_emb_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.T * Self.EMB)
        self.ae_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.AE.CACHE_SIZE)
        self.ae_ws_buf = ctx.enqueue_create_buffer[dtype](
            _max_int(1, Self.BATCH * Self.AE.WORKSPACE_SIZE_PER_SAMPLE)
        )

        self.x_prev_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)
        self.x_prev_pe_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)
        self.pos_cache_buf = ctx.enqueue_create_buffer[dtype](
            _max_int(1, Self.BATCH * Self.POS.CACHE_SIZE)
        )
        self.pos_ws_buf = ctx.enqueue_create_buffer[dtype](
            _max_int(1, Self.BATCH * Self.POS.WORKSPACE_SIZE_PER_SAMPLE)
        )
        self.c_in_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)

        self.pred_raw_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)
        self.pred_out_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.H * Self.EMB)
        self.proj_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.PROJ.CACHE_SIZE)
        self.proj_ws_buf = ctx.enqueue_create_buffer[dtype](
            _max_int(1, Self.BATCH * Self.PROJ.WORKSPACE_SIZE_PER_SAMPLE)
        )

        # cond_block caches — Self.DEPTH-fold (sliced per layer in helpers).
        self.silu_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB * Self.DEPTH)
        self.adaln_cache_buf = ctx.enqueue_create_buffer[dtype](
            Self.BTH * Self.ADALN.CACHE_SIZE * Self.DEPTH
        )
        self.ln1_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * (Self.EMB + 1) * Self.DEPTH)
        self.mod1_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * 2 * Self.EMB * Self.DEPTH)
        self.msa_cache_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.MSA.CACHE_SIZE * Self.DEPTH
        )
        self.gate1_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * 2 * Self.EMB * Self.DEPTH)
        self.ln2_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * (Self.EMB + 1) * Self.DEPTH)
        self.mod2_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * 2 * Self.EMB * Self.DEPTH)
        self.mlp_cache_buf = ctx.enqueue_create_buffer[dtype](
            Self.BTH * Self.MLP.CACHE_SIZE * Self.DEPTH
        )
        self.gate2_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * 2 * Self.EMB * Self.DEPTH)
        self.raw_mod_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * 6 * Self.EMB * Self.DEPTH)
        self.x_mid_buf_d = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB * Self.DEPTH)
        # Intermediate x flow between layers. (Self.DEPTH-1) slots since layer 0 reads
        # x_prev_pe and layer Self.DEPTH-1 writes pred_raw directly.
        self.x_inter_buf = ctx.enqueue_create_buffer[dtype](
            _max_int(1, Self.BTH * Self.EMB * (Self.DEPTH - 1))
        )

        # cond_block forward scratch (reused across MSA and MLP branches).
        self.silu_buf_d = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)
        self.ln_out_buf_d = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)
        self.mod_inp_buf_d = ctx.enqueue_create_buffer[dtype](Self.BTH * 3 * Self.EMB)
        self.mod_x_buf_d = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)
        self.branch_out_buf_d = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)
        self.gate_inp_buf_d = ctx.enqueue_create_buffer[dtype](Self.BTH * 3 * Self.EMB)

        # cond_block workspaces (shared with model forward_gpu calls).
        self.adaln_ws_buf = ctx.enqueue_create_buffer[dtype](
            _max_int(1, Self.BTH * Self.ADALN.WORKSPACE_SIZE_PER_SAMPLE)
        )
        self.msa_ws_buf = ctx.enqueue_create_buffer[dtype](
            _max_int(1, Self.BATCH * Self.MSA.WORKSPACE_SIZE_PER_SAMPLE)
        )
        self.mlp_ws_buf = ctx.enqueue_create_buffer[dtype](
            _max_int(1, Self.BTH * Self.MLP.WORKSPACE_SIZE_PER_SAMPLE)
        )

        # cond_block backward scratch (reused across all Self.DEPTH layers).
        self.sgg_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * 3 * Self.EMB)
        self.sgbo_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)
        self.sgmx_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)
        self.sgmi_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * 3 * Self.EMB)
        self.sglnout_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)
        self.sglnin_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)
        self.sgrm_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * 6 * Self.EMB)
        self.sgsc_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)
        self.grad_x_mid_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)
        # Backward intermediate grad_x flow between layers (Self.DEPTH-1 slots).
        self.grad_x_inter_buf = ctx.enqueue_create_buffer[dtype](
            _max_int(1, Self.BTH * Self.EMB * (Self.DEPTH - 1))
        )
        # Per-layer grad_c output (single buffer, reused per layer; accumulated
        # into grad_c_buf via cb_accum_kernel).
        self.grad_c_layer_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)

        # SIGReg buffers (forward + backward).
        self.sigreg_out_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.SIG.OUT_DIM)
        self.sigreg_cache_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.SIG.CACHE_SIZE
        )
        self.sigreg_grad_out_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.SIG.OUT_DIM
        )
        self.sigreg_grad_emb_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.T * Self.EMB
        )
        self.sigreg_ws_buf = ctx.enqueue_create_buffer[dtype](Self.SIG_WS_SIZE)
        # Seed grad_output = λ/B (constant across all steps; chain rule produces
        # an effective G = λ at the SIGReg dLdz step). See CPU trainer line 735.
        var sigreg_grad_out_host = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH * Self.SIG.OUT_DIM
        )
        for i in range(Self.BATCH * Self.SIG.OUT_DIM):
            sigreg_grad_out_host[i] = Scalar[dtype](
                lambda_sigreg / Float64(Self.BATCH)
            )
        ctx.enqueue_copy(self.sigreg_grad_out_buf, sigreg_grad_out_host)

        # Gradient buffers (device).
        self.grad_pred_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.H * Self.EMB)
        self.grad_pred_raw_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)
        self.grad_x_prev_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)
        self.grad_x_prev_pe_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)
        self.grad_c_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)
        self.grad_emb_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.T * Self.EMB)
        self.grad_act_emb_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.T * Self.EMB)
        self.grad_actions_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.T * Self.ACT)
        self.grad_pixels_buf = ctx.enqueue_create_buffer[dtype](Self.BT * Self.IMG_DIM)

        # Pinned host buffers for sampled data + per-step loss compute.
        # Uint8 staging: pixels_u8_host (host) -> pixels_u8_buf (device, uint8)
        # -> pixels_uint8_to_fp32_kernel -> pixels_buf (device, fp32).
        self.pixels_u8_host = ctx.enqueue_create_host_buffer[DType.uint8](
            Self.BT * Self.IMG_DIM
        )
        self.pixels_u8_buf = ctx.enqueue_create_buffer[DType.uint8](
            Self.BT * Self.IMG_DIM
        )
        self.actions_host = ctx.enqueue_create_host_buffer[dtype](Self.BATCH * Self.T * Self.ACT)
        self.pred_host = ctx.enqueue_create_host_buffer[dtype](Self.BATCH * Self.H * Self.EMB)
        self.target_host = ctx.enqueue_create_host_buffer[dtype](Self.BATCH * Self.H * Self.EMB)
        self.grad_pred_host = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH * Self.H * Self.EMB
        )
        self.sigreg_out_host = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH * Self.SIG.OUT_DIM
        )
        # emb on device has shape (BT, Self.EMB) — aliased as (Self.BATCH, Self.T*Self.EMB) for the
        # target slice. Same memory, single host buffer.
        self.emb_host = ctx.enqueue_create_host_buffer[dtype](Self.BT * Self.EMB)

        # Small scratch (BATCH*T*ACT floats — a few KB) for H6's expert-action snapshot.
        self.actions_sample = alloc[Scalar[dtype]](Self.BATCH * Self.T * Self.ACT)

    def __del__(deinit self):
        self.actions_sample.free()

    # =========================================================================
    # Checkpoint save/load (orchestration over all 4 + 3*DEPTH GPU networks)
    # =========================================================================

    def save_checkpoint(
        self,
        ctx: DeviceContext,
        path: String,
        var metadata: List[String],
    ) raises:
        """Serialize all GPU networks to a single multi-section text file.

        Uses the per-network helpers from `lewm_checkpoint` for the
        section emission; this method just orchestrates the walk over
        `enc/ae/pos/proj` plus the `DEPTH`-fold `adaln/msa/mlp` stack.
        `metadata` becomes the file's metadata section — consumed.
        """
        var total_params = (
            self.enc_state.PARAM_SIZE + self.ae_state.PARAM_SIZE
            + self.pos_state.PARAM_SIZE + self.proj_state.PARAM_SIZE
            + Self.DEPTH * (
                self.adaln_states[0].PARAM_SIZE
                + self.msa_states[0].PARAM_SIZE
                + self.mlp_states[0].PARAM_SIZE
            )
        )
        var total_opt_state = (
            self.enc_state.OPT_STATE_SIZE + self.ae_state.OPT_STATE_SIZE
            + self.pos_state.OPT_STATE_SIZE + self.proj_state.OPT_STATE_SIZE
            + Self.DEPTH * (
                self.adaln_states[0].OPT_STATE_SIZE
                + self.msa_states[0].OPT_STATE_SIZE
                + self.mlp_states[0].OPT_STATE_SIZE
            )
        )
        var content = write_checkpoint_header(
            "lewm", total_params, total_opt_state
        )
        content += _write_gpu_net_sections(self.enc_state, ctx, String("enc_"))
        content += _write_gpu_net_sections(self.ae_state, ctx, String("ae_"))
        content += _write_gpu_net_sections(self.pos_state, ctx, String("pos_"))
        content += _write_gpu_net_sections(self.proj_state, ctx, String("proj_"))
        for i in range(Self.DEPTH):
            content += _write_gpu_net_sections(
                self.adaln_states[i], ctx, String("adaln") + String(i) + "_"
            )
            content += _write_gpu_net_sections(
                self.msa_states[i], ctx, String("msa") + String(i) + "_"
            )
            content += _write_gpu_net_sections(
                self.mlp_states[i], ctx, String("mlp") + String(i) + "_"
            )
        content += write_metadata_section(metadata^)
        save_checkpoint_file(path, content)

    def load_checkpoint(
        mut self,
        ctx: DeviceContext,
        path: String,
    ) raises -> List[String]:
        """Load all GPU networks from a single multi-section text file.

        Returns the metadata list so callers can verify the comptime
        shape stored at save time matches the binary's instantiation
        (otherwise PARAM_SIZE mismatches silently truncate or pad).
        Synchronizes the device queue once before returning.
        """
        var content = read_checkpoint_file(path)
        var header = parse_checkpoint_header(content)
        if header.checkpoint_type != "lewm":
            raise Error(
                "LeWMGPUState.load_checkpoint: expected checkpoint type"
                " 'lewm', got '" + header.checkpoint_type + "'"
            )
        _read_gpu_net_sections(self.enc_state, content, ctx, String("enc_"))
        _read_gpu_net_sections(self.ae_state, content, ctx, String("ae_"))
        _read_gpu_net_sections(self.pos_state, content, ctx, String("pos_"))
        _read_gpu_net_sections(self.proj_state, content, ctx, String("proj_"))
        for i in range(Self.DEPTH):
            _read_gpu_net_sections(
                self.adaln_states[i], content, ctx,
                String("adaln") + String(i) + "_",
            )
            _read_gpu_net_sections(
                self.msa_states[i], content, ctx,
                String("msa") + String(i) + "_",
            )
            _read_gpu_net_sections(
                self.mlp_states[i], content, ctx,
                String("mlp") + String(i) + "_",
            )
        ctx.synchronize()
        return read_metadata_section(content)


struct LeWMTrainer[
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
    BUF: LeWMBuffer = PongBuffer,
]:
    """Owns hyperparams + per-run EMAs + a clip buffer; methods consume a
    `LeWMGPUState` for the GPU-resident data.

    `BUF` is the buffer type — must implement
    `sample_batch_fp32(B, T, pixels_out, actions_out) raises` and expose
    `n_frames: Int`. Concrete instances: `PongBuffer` (Atari-style
    pixel-obs replay) and `LewmPushTSampler` (HDF5-backed expert clips
    for the LeWM paper recipe).
    """

    comptime GPUState = LeWMGPUState[
        Self.BATCH, Self.T, Self.H, Self.N_PREDS, Self.IN_CH, Self.IMG, Self.PATCH, Self.N_PATCHES, Self.HIDDEN,
        Self.ENC_HEADS, Self.ENC_LAYERS, Self.EMB, Self.PROJ_H, Self.ACT, Self.SMOOTHED, Self.PRED_HEADS,
        Self.PRED_FF, Self.DEPTH, Self.SIG_NUM_PROJ, Self.SIG_KNOTS,
    ]

    # Buffer + hyperparams
    var buf: Self.BUF
    var lambda_sigreg: Float64
    var log_every: Int
    var eval_steps: Int
    var eval_samples: Int
    var eval_seed: Int
    var mpc_horizon: Int
    var cem_iters: Int
    var cem_samples: Int
    var cem_topk: Int
    var cem_smoothing: Float64
    var eval_shuffle_diag: Bool
    var eval_h7_closed_loop: Bool

    # Per-run scalar tracking
    var loss_ema: Float64
    var pred_ema: Float64
    var sigreg_ema: Float64
    var var_min_ema: Float64
    var var_mean_ema: Float64
    var gram_ema: Float64
    var loss_first: Float64
    var loss_last: Float64

    # Timing (perf_counter_ns returns UInt; field matches to avoid casts).
    var t0_ns: UInt

    # Per-phase perf counters (cumulative ns, divided by `n_timed` at print).
    # `t_step_ns` is the wall time per step from sample-start to step-end.
    # `t_step_ns - t_sample_ns - t_h2d_ns` ≈ host time for kernel launches +
    # implicit stream stalls. With `time_phases=True`, an end-of-step
    # ctx.synchronize() forces accurate GPU-wall accounting at the cost of
    # losing CPU/GPU overlap.
    var t_sample_ns: UInt
    var t_h2d_ns: UInt
    var t_step_ns: UInt
    var n_timed: Int
    var time_phases: Bool

    # Constants
    var loss_scale: Float64
    var inv_scale: Scalar[dtype]

    def __init__(
        out self,
        var buf: Self.BUF,
        lambda_sigreg: Float64,
        log_every: Int,
        eval_steps: Int,
        eval_samples: Int,
        eval_seed: Int,
        mpc_horizon: Int,
        cem_iters: Int,
        cem_samples: Int,
        cem_topk: Int,
        cem_smoothing: Float64,
        eval_shuffle_diag: Bool,
        eval_h7_closed_loop: Bool = True,
        time_phases: Bool = False,
    ) raises:
        self.buf = buf^
        self.lambda_sigreg = lambda_sigreg
        self.log_every = log_every
        self.eval_steps = eval_steps
        self.eval_samples = eval_samples
        self.eval_seed = eval_seed
        self.mpc_horizon = mpc_horizon
        self.cem_iters = cem_iters
        self.cem_samples = cem_samples
        self.cem_topk = cem_topk
        self.cem_smoothing = cem_smoothing
        self.eval_shuffle_diag = eval_shuffle_diag
        self.eval_h7_closed_loop = eval_h7_closed_loop

        self.loss_ema = 0.0
        self.pred_ema = 0.0
        self.sigreg_ema = 0.0
        self.var_min_ema = 0.0
        self.var_mean_ema = 0.0
        self.gram_ema = 0.0
        self.loss_first = -1.0
        self.loss_last = 0.0

        self.t0_ns = UInt(0)
        self.t_sample_ns = UInt(0)
        self.t_h2d_ns = UInt(0)
        self.t_step_ns = UInt(0)
        self.n_timed = 0
        self.time_phases = time_phases

        self.loss_scale = Float64(Self.BATCH * Self.H * Self.EMB)
        self.inv_scale = Scalar[dtype](2.0 / self.loss_scale)

    def _sample_and_upload_pixels(
        mut self,
        mut state: Self.GPUState,
        ctx: DeviceContext,
    ) raises:
        """Sample a batch of pixels+actions into pinned host buffers,
        upload uint8 pixels to device, run the GPU conversion kernel to
        produce fp32 CHW pixels in `pixels_buf`.

        Actions land in `actions_host` (fp32) but are NOT uploaded to
        `actions_buf` — caller decides whether/when. This separation
        matters for MPC (overwrites actions with CEM samples) and H6
        (snapshots actions before shuffle).
        """
        self.buf.sample_batch_uint8(
            Self.BATCH,
            Self.T,
            state.pixels_u8_host.unsafe_ptr(),
            state.actions_host.unsafe_ptr(),
        )
        ctx.enqueue_copy(state.pixels_u8_buf, state.pixels_u8_host)

        comptime BT = Self.GPUState.BT
        comptime IMG_DIM = Self.GPUState.IMG_DIM
        var src_u8_t = LayoutTensor[
            DType.uint8, Layout.row_major(BT, IMG_DIM), MutAnyOrigin,
        ](state.pixels_u8_buf)
        var dst_fp32_t = LayoutTensor[
            dtype, Layout.row_major(BT, IMG_DIM), MutAnyOrigin,
        ](state.pixels_buf)
        ctx.enqueue_function[
            pixels_uint8_to_fp32_kernel[
                BT, Self.IN_CH, Self.IMG, Self.BUF.INPUT_LAYOUT_HWC,
            ],
        ](
            src_u8_t,
            dst_fp32_t,
            grid_dim=(
                ceildiv(BT, TPB_X),
                ceildiv(Self.IN_CH, TPB_Y),
                ceildiv(Self.IMG * Self.IMG, TPB_Z),
            ),
            block_dim=(TPB_X, TPB_Y, TPB_Z),
        )

    def train_step(
        mut self,
        mut state: Self.GPUState,
        ctx: DeviceContext,
        step: Int,
        rng_seed: Int,
    ) raises -> Float64:
        # Phase timer — wall time across the whole step (sample + h2d + GPU work).
        var ts_step_start = perf_counter_ns()

        comptime IMG_DIM = Self.GPUState.IMG_DIM
        comptime BT = Self.GPUState.BT
        comptime BTH = Self.GPUState.BTH
        comptime ENC = Self.GPUState.ENC
        comptime AE = Self.GPUState.AE
        comptime POS = Self.GPUState.POS
        comptime PROJ = Self.GPUState.PROJ
        comptime SIG = Self.GPUState.SIG

        # ── LayoutTensor views (recomputed on demand — zero-cost ptr wrappers) ──
        var pixels_t = LayoutTensor[
            dtype, Layout.row_major(BT, IMG_DIM), MutAnyOrigin
        ](state.pixels_buf)
        var actions_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.T * Self.ACT), MutAnyOrigin
        ](state.actions_buf)
        var emb_t = LayoutTensor[
            dtype, Layout.row_major(BT, Self.EMB), MutAnyOrigin
        ](state.emb_buf)
        var enc_cache_t = LayoutTensor[
            dtype, Layout.row_major(BT, ENC.CACHE_SIZE), MutAnyOrigin
        ](state.enc_cache_buf)

        var act_emb_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.T * Self.EMB), MutAnyOrigin
        ](state.act_emb_buf)
        var ae_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, AE.CACHE_SIZE), MutAnyOrigin
        ](state.ae_cache_buf)

        var x_prev_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.x_prev_buf)
        var x_prev_bh_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.H * Self.EMB), MutAnyOrigin
        ](state.x_prev_buf)
        var x_prev_pe_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.x_prev_pe_buf)
        var x_prev_pe_bh_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.H * Self.EMB), MutAnyOrigin
        ](state.x_prev_pe_buf)
        var pos_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, POS.CACHE_SIZE), MutAnyOrigin
        ](state.pos_cache_buf)
        var c_in_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.c_in_buf)

        var pred_raw_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.pred_raw_buf)
        var pred_raw_bh_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.H * Self.EMB), MutAnyOrigin
        ](state.pred_raw_buf)
        var pred_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.H * Self.EMB), MutAnyOrigin
        ](state.pred_out_buf)
        var proj_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, PROJ.CACHE_SIZE), MutAnyOrigin
        ](state.proj_cache_buf)

        # Per-layer cache LayoutTensor views are created inside the
        # run_cond_layer_forward/backward helpers (one slice per layer d).
        var grad_c_layer_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.grad_c_layer_buf)

        var silu_buf_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.silu_buf_d)
        var ln_out_buf_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.ln_out_buf_d)
        var mod_inp_buf_t = LayoutTensor[
            dtype, Layout.row_major(BTH, 3 * Self.EMB), MutAnyOrigin
        ](state.mod_inp_buf_d)
        var mod_x_buf_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.mod_x_buf_d)
        var branch_out_buf_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.branch_out_buf_d)
        var gate_inp_buf_t = LayoutTensor[
            dtype, Layout.row_major(BTH, 3 * Self.EMB), MutAnyOrigin
        ](state.gate_inp_buf_d)

        var sgg_t = LayoutTensor[
            dtype, Layout.row_major(BTH, 3 * Self.EMB), MutAnyOrigin
        ](state.sgg_buf)
        var sgbo_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.sgbo_buf)
        var sgmx_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.sgmx_buf)
        var sgmi_t = LayoutTensor[
            dtype, Layout.row_major(BTH, 3 * Self.EMB), MutAnyOrigin
        ](state.sgmi_buf)
        var sglnout_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.sglnout_buf)
        var sglnin_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.sglnin_buf)
        var sgrm_t = LayoutTensor[
            dtype, Layout.row_major(BTH, 6 * Self.EMB), MutAnyOrigin
        ](state.sgrm_buf)
        var sgsc_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.sgsc_buf)
        var grad_x_mid_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.grad_x_mid_buf)

        var grad_pred_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.H * Self.EMB), MutAnyOrigin
        ](state.grad_pred_buf)
        var grad_pred_raw_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.grad_pred_raw_buf)
        var grad_pred_raw_bh_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.H * Self.EMB), MutAnyOrigin
        ](state.grad_pred_raw_buf)
        var grad_x_prev_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.grad_x_prev_buf)
        var grad_x_prev_bh_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.H * Self.EMB), MutAnyOrigin
        ](state.grad_x_prev_buf)
        var grad_x_prev_pe_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.grad_x_prev_pe_buf)
        var grad_x_prev_pe_bh_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.H * Self.EMB), MutAnyOrigin
        ](state.grad_x_prev_pe_buf)
        var grad_c_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.grad_c_buf)
        var grad_emb_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.T * Self.EMB), MutAnyOrigin
        ](state.grad_emb_buf)
        var grad_emb_bt_t = LayoutTensor[
            dtype, Layout.row_major(BT, Self.EMB), MutAnyOrigin
        ](state.grad_emb_buf)
        var grad_act_emb_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.T * Self.EMB), MutAnyOrigin
        ](state.grad_act_emb_buf)
        var grad_actions_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.T * Self.ACT), MutAnyOrigin
        ](state.grad_actions_buf)
        var grad_pixels_t = LayoutTensor[
            dtype, Layout.row_major(BT, IMG_DIM), MutAnyOrigin
        ](state.grad_pixels_buf)

        # SIGReg views (treat emb / grad_emb as (Self.BATCH, Self.T*Self.EMB) — same memory).
        var emb_bte_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.T * Self.EMB), MutAnyOrigin
        ](state.emb_buf)
        var sigreg_out_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, SIG.OUT_DIM), MutAnyOrigin
        ](state.sigreg_out_buf)
        var sigreg_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, SIG.CACHE_SIZE), MutAnyOrigin
        ](state.sigreg_cache_buf)
        var sigreg_grad_out_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, SIG.OUT_DIM), MutAnyOrigin
        ](state.sigreg_grad_out_buf)
        var sigreg_grad_emb_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.T * Self.EMB), MutAnyOrigin
        ](state.sigreg_grad_emb_buf)
        var empty_params = LayoutTensor[
            dtype, Layout.row_major(SIG.PARAM_SIZE), MutAnyOrigin
        ](UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=0))
        var empty_grad_params = LayoutTensor[
            dtype, Layout.row_major(SIG.PARAM_SIZE), MutAnyOrigin
        ](UnsafePointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=0))

        # Sample uint8 pixels into pinned host buffer, upload + convert on GPU.
        var ts_sample_start = perf_counter_ns()
        self.buf.sample_batch_uint8(
            Self.BATCH,
            Self.T,
            state.pixels_u8_host.unsafe_ptr(),
            state.actions_host.unsafe_ptr(),
        )
        var ts_sample_end = perf_counter_ns()
        ctx.enqueue_copy(state.pixels_u8_buf, state.pixels_u8_host)
        ctx.enqueue_copy(state.actions_buf, state.actions_host)
        comptime BT_LOCAL = Self.GPUState.BT
        comptime IMG_DIM_LOCAL = Self.GPUState.IMG_DIM
        var src_u8_t = LayoutTensor[
            DType.uint8, Layout.row_major(BT_LOCAL, IMG_DIM_LOCAL), MutAnyOrigin,
        ](state.pixels_u8_buf)
        var dst_fp32_t = LayoutTensor[
            dtype, Layout.row_major(BT_LOCAL, IMG_DIM_LOCAL), MutAnyOrigin,
        ](state.pixels_buf)
        ctx.enqueue_function[
            pixels_uint8_to_fp32_kernel[
                BT_LOCAL, Self.IN_CH, Self.IMG, Self.BUF.INPUT_LAYOUT_HWC,
            ],
        ](
            src_u8_t,
            dst_fp32_t,
            grid_dim=(
                ceildiv(BT_LOCAL, TPB_X),
                ceildiv(Self.IN_CH, TPB_Y),
                ceildiv(Self.IMG * Self.IMG, TPB_Z),
            ),
            block_dim=(TPB_X, TPB_Y, TPB_Z),
        )
        var ts_h2d_end = perf_counter_ns()

        # Zero grads on all 6 groups.
        state.enc_state.zero_grads(ctx)
        state.ae_state.zero_grads(ctx)
        state.pos_state.zero_grads(ctx)
        state.proj_state.zero_grads(ctx)
        for layer_idx in range(Self.DEPTH):
            state.adaln_states[layer_idx].zero_grads(ctx)
            state.msa_states[layer_idx].zero_grads(ctx)
            state.mlp_states[layer_idx].zero_grads(ctx)

        # Encoder forward.
        ENC.forward_gpu[BT, dtype](
            ctx, emb_t, pixels_t,
            state.enc_state.params_view(), state.enc_state.model_state_view(),
            enc_cache_t, state.enc_ws_buf,
        )
        # Action embedder forward.
        AE.forward_gpu[Self.BATCH, dtype](
            ctx, act_emb_t, actions_t,
            state.ae_state.params_view(), state.ae_state.model_state_view(),
            ae_cache_t, state.ae_ws_buf,
        )

        # Slice first Self.H tokens of emb + act_emb into x_prev_buf + c_in_buf.
        var act_emb_bt_t = LayoutTensor[
            dtype, Layout.row_major(BT, Self.EMB), MutAnyOrigin
        ](state.act_emb_buf)
        ctx.enqueue_function[
            slice_h_kernel[Self.BATCH, Self.T, Self.H, Self.EMB],
        ](
            emb_t, x_prev_t,
            grid_dim=(
                ceildiv(Self.BATCH, TPB_X),
                ceildiv(Self.H, TPB_Y),
                ceildiv(Self.EMB, TPB_Z),
            ),
            block_dim=(TPB_X, TPB_Y, TPB_Z),
        )
        ctx.enqueue_function[
            slice_h_kernel[Self.BATCH, Self.T, Self.H, Self.EMB],
        ](
            act_emb_bt_t, c_in_t,
            grid_dim=(
                ceildiv(Self.BATCH, TPB_X),
                ceildiv(Self.H, TPB_Y),
                ceildiv(Self.EMB, TPB_Z),
            ),
            block_dim=(TPB_X, TPB_Y, TPB_Z),
        )

        # Pos embed: x_prev_pe = x_prev + pos_bias (broadcast over Self.BATCH).
        POS.forward_gpu[Self.BATCH, dtype](
            ctx, x_prev_pe_bh_t, x_prev_bh_t,
            state.pos_state.params_view(), state.pos_state.model_state_view(),
            pos_cache_t, state.pos_ws_buf,
        )

        # cond_block stack: Self.DEPTH dual-branch (MSA + MLP) layers via helper.
        for d in range(Self.DEPTH):
            run_cond_layer_forward[Self.BATCH, Self.H, Self.EMB, Self.PRED_HEADS, Self.PRED_FF](
                ctx, d, Self.DEPTH,
                state.x_prev_pe_buf, state.x_inter_buf, state.pred_raw_buf,
                c_in_t,
                state.adaln_states[d].params_view(),
                state.adaln_states[d].model_state_view(),
                state.msa_states[d].params_view(),
                state.msa_states[d].model_state_view(),
                state.mlp_states[d].params_view(),
                state.mlp_states[d].model_state_view(),
                state.silu_cache_buf, state.adaln_cache_buf,
                state.ln1_cache_buf, state.mod1_cache_buf, state.msa_cache_buf, state.gate1_cache_buf,
                state.ln2_cache_buf, state.mod2_cache_buf, state.mlp_cache_buf, state.gate2_cache_buf,
                state.raw_mod_buf, state.x_mid_buf_d,
                silu_buf_t, ln_out_buf_t, mod_inp_buf_t,
                mod_x_buf_t, branch_out_buf_t, gate_inp_buf_t,
                state.adaln_ws_buf, state.msa_ws_buf, state.mlp_ws_buf,
            )

        # PredProj: (Self.BATCH, Self.H*Self.EMB) → (Self.BATCH, Self.H*Self.EMB).
        PROJ.forward_gpu[Self.BATCH, dtype](
            ctx, pred_t, pred_raw_bh_t,
            state.proj_state.params_view(), state.proj_state.model_state_view(),
            proj_cache_t, state.proj_ws_buf,
        )

        # SIGReg forward over emb viewed as (Self.BATCH, Self.T*Self.EMB). Output is the
        # statistic replicated across Self.BATCH slots (we read [0] for logging).
        SIG.eval_gpu[Self.BATCH, dtype](
            ctx, sigreg_out_t, emb_bte_t,
            empty_params, sigreg_cache_t, state.sigreg_ws_buf.unsafe_ptr(),
        )

        # --------------------------------------------------------------
        # Loss + grad_pred on host (small round-trip).
        # --------------------------------------------------------------
        ctx.enqueue_copy(state.pred_host, state.pred_out_buf)
        # Download all of emb (BT, Self.EMB) — used for both target slice and probes.
        ctx.enqueue_copy(state.emb_host, state.emb_buf)
        # Download SIGReg stat (tiny — Self.BATCH floats) for logging.
        ctx.enqueue_copy(state.sigreg_out_host, state.sigreg_out_buf)
        ctx.synchronize()

        var pred_loss: Float64 = 0.0
        for b in range(Self.BATCH):
            for i in range(Self.H * Self.EMB):
                var p = Float64(state.pred_host[b * Self.H * Self.EMB + i])
                # Target = emb[b, Self.N_PREDS .. Self.N_PREDS+Self.H, :], flat index:
                #   b * Self.T * Self.EMB + Self.N_PREDS * Self.EMB + i
                var tgt = Float64(
                    state.emb_host[b * Self.T * Self.EMB + Self.N_PREDS * Self.EMB + i]
                )
                var diff = p - tgt
                pred_loss += diff * diff
                state.grad_pred_host[b * Self.H * Self.EMB + i] = self.inv_scale * (
                    Scalar[dtype](p) - Scalar[dtype](tgt)
                )
        pred_loss /= self.loss_scale

        # Read SIGReg stat (replicated across Self.BATCH, take [0]).
        var sigreg_stat = Float64(state.sigreg_out_host[0])

        if self.loss_first < 0.0:
            self.loss_first = pred_loss
            self.loss_ema = pred_loss
            self.pred_ema = pred_loss
            self.sigreg_ema = sigreg_stat
        else:
            self.loss_ema = 0.95 * self.loss_ema + 0.05 * pred_loss
            self.pred_ema = 0.95 * self.pred_ema + 0.05 * pred_loss
            self.sigreg_ema = 0.95 * self.sigreg_ema + 0.05 * sigreg_stat
        self.loss_last = pred_loss

        # Upload grad_pred back to device.
        ctx.enqueue_copy(state.grad_pred_buf, state.grad_pred_host)

        # --------------------------------------------------------------
        # Collapse probes (cheap — emb already on host).
        # --------------------------------------------------------------
        var var_min: Float64 = 1e30
        var var_mean: Float64 = 0.0
        for d in range(Self.EMB):
            var s: Float64 = 0.0
            var ss: Float64 = 0.0
            for bt in range(BT):
                var v = Float64(state.emb_host[bt * Self.EMB + d])  # uses emb_bte_host? no, separate.
                s += v
                ss += v * v
            var mean_d = s / Float64(BT)
            var var_d = (ss / Float64(BT)) - mean_d * mean_d
            if var_d < var_min:
                var_min = var_d
            var_mean += var_d
        var_mean /= Float64(Self.EMB)
        var gram_off: Float64 = 0.0
        var gram_n: Int = 0
        for i in range(BT):
            var ni: Float64 = 0.0
            for d in range(Self.EMB):
                var v = Float64(state.emb_host[i * Self.EMB + d])
                ni += v * v
            ni = sqrt(ni + 1e-12)
            for j in range(i + 1, BT):
                var nj: Float64 = 0.0
                var dot_v: Float64 = 0.0
                for d in range(Self.EMB):
                    var vi = Float64(state.emb_host[i * Self.EMB + d])
                    var vj = Float64(state.emb_host[j * Self.EMB + d])
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
            self.var_min_ema = var_min
            self.var_mean_ema = var_mean
            self.gram_ema = gram_off
        else:
            self.var_min_ema = 0.95 * self.var_min_ema + 0.05 * var_min
            self.var_mean_ema = 0.95 * self.var_mean_ema + 0.05 * var_mean
            self.gram_ema = 0.95 * self.gram_ema + 0.05 * gram_off

        # --------------------------------------------------------------
        # Backward
        # --------------------------------------------------------------
        var enc_g = state.enc_state.grads_view()
        var ae_g = state.ae_state.grads_view()
        var pos_g = state.pos_state.grads_view()
        var proj_g = state.proj_state.grads_view()

        # PROJ.backward
        PROJ.backward_gpu[Self.BATCH, dtype](
            ctx, grad_pred_raw_bh_t, grad_pred_t,
            state.proj_state.params_view(), state.proj_state.model_state_view(),
            proj_cache_t, proj_g, state.proj_ws_buf,
        )

        # cond_block stack backward — reverse depth loop via helper.
        # grad_c is accumulated across layers (c is shared input).
        ctx.enqueue_memset(state.grad_c_buf, 0)
        for d_rev in range(Self.DEPTH):
            var d = Self.DEPTH - 1 - d_rev
            # Bind per-layer grad views to vars (mut args can't take temps).
            var adaln_g_d = state.adaln_states[d].grads_view()
            var msa_g_d = state.msa_states[d].grads_view()
            var mlp_g_d = state.mlp_states[d].grads_view()
            run_cond_layer_backward[Self.BATCH, Self.H, Self.EMB, Self.PRED_HEADS, Self.PRED_FF](
                ctx, d, Self.DEPTH,
                state.grad_pred_raw_buf, state.grad_x_inter_buf, state.grad_x_prev_pe_buf,
                state.adaln_states[d].params_view(),
                state.adaln_states[d].model_state_view(),
                state.msa_states[d].params_view(),
                state.msa_states[d].model_state_view(),
                state.mlp_states[d].params_view(),
                state.mlp_states[d].model_state_view(),
                state.silu_cache_buf, state.adaln_cache_buf,
                state.ln1_cache_buf, state.mod1_cache_buf, state.msa_cache_buf, state.gate1_cache_buf,
                state.ln2_cache_buf, state.mod2_cache_buf, state.mlp_cache_buf, state.gate2_cache_buf,
                state.x_mid_buf_d,
                grad_c_layer_t,
                adaln_g_d, msa_g_d, mlp_g_d,
                sgg_t, sgbo_t, sgmx_t, sgmi_t,
                sglnout_t, sglnin_t, sgrm_t, sgsc_t,
                grad_x_mid_t,
                state.adaln_ws_buf, state.msa_ws_buf, state.mlp_ws_buf,
            )
            # Accumulate this layer's grad_c into the shared grad_c_t.
            comptime TPB_GC_X = 16
            comptime TPB_GC_Y = 16
            ctx.enqueue_function[cb_accum_kernel[BTH, Self.EMB]](
                grad_c_t, grad_c_layer_t,
                grid_dim=(ceildiv(BTH, TPB_GC_X), ceildiv(Self.EMB, TPB_GC_Y)),
                block_dim=(TPB_GC_X, TPB_GC_Y),
            )

        # POS.backward
        POS.backward_gpu[Self.BATCH, dtype](
            ctx, grad_x_prev_bh_t, grad_x_prev_pe_bh_t,
            state.pos_state.params_view(), state.pos_state.model_state_view(),
            pos_cache_t, pos_g, state.pos_ws_buf,
        )

        # Route grad_x_prev → grad_emb's first Self.H tokens, grad_c → grad_act_emb's.
        # Target slice gradient is FILLED below (no stop-grad — paper recipe).
        ctx.enqueue_memset(state.grad_emb_buf, 0)
        ctx.enqueue_memset(state.grad_act_emb_buf, 0)

        var grad_emb_bte_to_bt = LayoutTensor[
            dtype, Layout.row_major(BT, Self.EMB), MutAnyOrigin
        ](state.grad_emb_buf)
        var grad_act_emb_bte_to_bt = LayoutTensor[
            dtype, Layout.row_major(BT, Self.EMB), MutAnyOrigin
        ](state.grad_act_emb_buf)
        ctx.enqueue_function[
            scatter_h_kernel[Self.BATCH, Self.T, Self.H, Self.EMB],
        ](
            grad_x_prev_t, grad_emb_bte_to_bt,
            grid_dim=(
                ceildiv(Self.BATCH, TPB_X),
                ceildiv(Self.H, TPB_Y),
                ceildiv(Self.EMB, TPB_Z),
            ),
            block_dim=(TPB_X, TPB_Y, TPB_Z),
        )
        ctx.enqueue_function[
            scatter_h_kernel[Self.BATCH, Self.T, Self.H, Self.EMB],
        ](
            grad_c_t, grad_act_emb_bte_to_bt,
            grid_dim=(
                ceildiv(Self.BATCH, TPB_X),
                ceildiv(Self.H, TPB_Y),
                ceildiv(Self.EMB, TPB_Z),
            ),
            block_dim=(TPB_X, TPB_Y, TPB_Z),
        )

        # Drop stop-grad: scatter -grad_pred into target slice of grad_emb.
        # Math: pred_loss = (pred - tgt)^2 / N → d/d tgt = -grad_pred.
        # Target tokens live at b * Self.T*Self.EMB + Self.N_PREDS*Self.EMB + [0..H*Self.EMB).
        comptime TPB_TS_X = 4
        comptime TPB_TS_Y = 64
        ctx.enqueue_function[
            scatter_target_neg_kernel[Self.BATCH, Self.T, Self.H, Self.N_PREDS, Self.EMB],
        ](
            grad_pred_t, grad_emb_t,
            grid_dim=(
                ceildiv(Self.BATCH, TPB_TS_X),
                ceildiv(Self.H * Self.EMB, TPB_TS_Y),
            ),
            block_dim=(TPB_TS_X, TPB_TS_Y),
        )

        # SIGReg vjp: produces sigreg_grad_emb (Self.BATCH, Self.T*Self.EMB) from
        # `sigreg_grad_out_t` seed = λ/B (set once at init).
        SIG.vjp_gpu[Self.BATCH, dtype](
            ctx, sigreg_grad_out_t, sigreg_grad_emb_t,
            empty_params, sigreg_cache_t, empty_grad_params,
            state.sigreg_ws_buf.unsafe_ptr(),
        )
        # Accumulate sigreg's grad into grad_emb additively.
        comptime TPB_AC_X = 4
        comptime TPB_AC_Y = 64
        ctx.enqueue_function[
            accumulate_emb_kernel[Self.BATCH, Self.T, Self.EMB],
        ](
            sigreg_grad_emb_t, grad_emb_t,
            grid_dim=(
                ceildiv(Self.BATCH, TPB_AC_X),
                ceildiv(Self.T * Self.EMB, TPB_AC_Y),
            ),
            block_dim=(TPB_AC_X, TPB_AC_Y),
        )

        # AE.backward
        AE.backward_gpu[Self.BATCH, dtype](
            ctx, grad_actions_t, grad_act_emb_t,
            state.ae_state.params_view(), state.ae_state.model_state_view(),
            ae_cache_t, ae_g, state.ae_ws_buf,
        )

        # ENC.backward
        ENC.backward_gpu[BT, dtype](
            ctx, grad_pixels_t, grad_emb_bt_t,
            state.enc_state.params_view(), state.enc_state.model_state_view(),
            enc_cache_t, enc_g, state.enc_ws_buf,
        )

        # Optimizer step — shared models + per-layer (ADALN/MSA/MLP × Self.DEPTH).
        state.enc_state.optimizer_step(ctx)
        state.ae_state.optimizer_step(ctx)
        state.pos_state.optimizer_step(ctx)
        state.proj_state.optimizer_step(ctx)
        for layer_idx in range(Self.DEPTH):
            state.adaln_states[layer_idx].optimizer_step(ctx)
            state.msa_states[layer_idx].optimizer_step(ctx)
            state.mlp_states[layer_idx].optimizer_step(ctx)

        # End-of-step timing accumulation. With time_phases=True, sync first
        # so `t_step_ns` includes pure GPU wall time (CPU/GPU overlap is
        # disabled in that case — only use for measurement, not production).
        if self.time_phases:
            ctx.synchronize()
        var ts_step_end = perf_counter_ns()
        self.t_sample_ns += UInt(ts_sample_end - ts_sample_start)
        self.t_h2d_ns += UInt(ts_h2d_end - ts_sample_end)
        self.t_step_ns += UInt(ts_step_end - ts_step_start)
        self.n_timed += 1

        # Periodic logging — download emb for probes was done inline above.
        if step % self.log_every == 0:
            ctx.synchronize()
            var t_now = perf_counter_ns()
            var sps = Float64(step + 1) / (Float64(t_now - self.t0_ns) / 1e9)
            print(
                "  step", step,
                " L=", pred_loss,
                " ema=", self.pred_ema,
                " sig=", self.sigreg_ema,
                " var_min=", self.var_min_ema,
                " var_mean=", self.var_mean_ema,
                " gram=", self.gram_ema,
                " it/s=", sps,
            )
            if self.n_timed > 0:
                var n = Float64(self.n_timed)
                var ms_sample = Float64(self.t_sample_ns) / n / 1e6
                var ms_h2d = Float64(self.t_h2d_ns) / n / 1e6
                var ms_step = Float64(self.t_step_ns) / n / 1e6
                var ms_rest = ms_step - ms_sample - ms_h2d
                print(
                    "    [phases avg over last",
                    self.n_timed,
                    "steps] sample=",
                    ms_sample,
                    "ms  h2d=",
                    ms_h2d,
                    "ms  rest=",
                    ms_rest,
                    "ms  total=",
                    ms_step,
                    "ms",
                )
                # Reset window-averages so each log line covers just the
                # interval since the previous log line.
                self.t_sample_ns = UInt(0)
                self.t_h2d_ns = UInt(0)
                self.t_step_ns = UInt(0)
                self.n_timed = 0

        return pred_loss

    def eval_h6(
        mut self,
        mut state: Self.GPUState,
        ctx: DeviceContext,
    ) raises:
        comptime IMG_DIM = Self.GPUState.IMG_DIM
        comptime BT = Self.GPUState.BT
        comptime BTH = Self.GPUState.BTH
        comptime ENC = Self.GPUState.ENC
        comptime AE = Self.GPUState.AE
        comptime POS = Self.GPUState.POS
        comptime PROJ = Self.GPUState.PROJ

        var pixels_t = LayoutTensor[
            dtype, Layout.row_major(BT, IMG_DIM), MutAnyOrigin
        ](state.pixels_buf)
        var actions_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.T * Self.ACT), MutAnyOrigin
        ](state.actions_buf)
        var emb_t = LayoutTensor[
            dtype, Layout.row_major(BT, Self.EMB), MutAnyOrigin
        ](state.emb_buf)
        var enc_cache_t = LayoutTensor[
            dtype, Layout.row_major(BT, ENC.CACHE_SIZE), MutAnyOrigin
        ](state.enc_cache_buf)
        var act_emb_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.T * Self.EMB), MutAnyOrigin
        ](state.act_emb_buf)
        var ae_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, AE.CACHE_SIZE), MutAnyOrigin
        ](state.ae_cache_buf)
        var x_prev_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.x_prev_buf)
        var x_prev_bh_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.H * Self.EMB), MutAnyOrigin
        ](state.x_prev_buf)
        var x_prev_pe_bh_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.H * Self.EMB), MutAnyOrigin
        ](state.x_prev_pe_buf)
        var pos_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, POS.CACHE_SIZE), MutAnyOrigin
        ](state.pos_cache_buf)
        var c_in_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.c_in_buf)
        var pred_raw_bh_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.H * Self.EMB), MutAnyOrigin
        ](state.pred_raw_buf)
        var pred_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.H * Self.EMB), MutAnyOrigin
        ](state.pred_out_buf)
        var proj_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, PROJ.CACHE_SIZE), MutAnyOrigin
        ](state.proj_cache_buf)

        var silu_buf_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.silu_buf_d)
        var ln_out_buf_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.ln_out_buf_d)
        var mod_inp_buf_t = LayoutTensor[
            dtype, Layout.row_major(BTH, 3 * Self.EMB), MutAnyOrigin
        ](state.mod_inp_buf_d)
        var mod_x_buf_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.mod_x_buf_d)
        var branch_out_buf_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.branch_out_buf_d)
        var gate_inp_buf_t = LayoutTensor[
            dtype, Layout.row_major(BTH, 3 * Self.EMB), MutAnyOrigin
        ](state.gate_inp_buf_d)

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
        print()
        print("==== H6: action-shuffle diagnostic (teacher-forced) ====")
        _set_seed(self.eval_seed)

        var perm_buf = alloc[Int](Self.BATCH)

        var h6_sum_expert: Float64 = 0.0
        var h6_sum_shuf_mean: Float64 = 0.0
        var h6_sum_shuf_min: Float64 = 0.0
        var h6_sum_better: Float64 = 0.0

        for h6_iter in range(self.eval_steps):
            # H6 permutes actions_host in-place via _run_h6_diag_shots, so we
            # snapshot expert actions to actions_sample (small, ~few KB) for
            # the unshuffled MSE reference.
            self._sample_and_upload_pixels(state, ctx)
            for i in range(Self.BATCH * Self.T * Self.ACT):
                state.actions_sample[i] = state.actions_host[i]
            ctx.enqueue_copy(state.actions_buf, state.actions_host)

            ENC.forward_gpu[BT, dtype](
                ctx, emb_t, pixels_t,
                state.enc_state.params_view(), state.enc_state.model_state_view(),
                enc_cache_t, state.enc_ws_buf,
            )

            ctx.enqueue_copy(state.emb_host, state.emb_buf)
            ctx.synchronize()

            var stats = _run_h6_diag_shots[
                Self.BATCH, Self.T, Self.H, Self.N_PREDS, Self.EMB, Self.ACT, Self.SMOOTHED, Self.PROJ_H,
                Self.PRED_HEADS, Self.PRED_FF, Self.DEPTH,
            ](
                ctx,
                self.eval_samples,
                state.actions_sample,
                perm_buf,
                state.actions_host, state.actions_buf,
                state.emb_host,
                state.pred_host, state.pred_out_buf,
                state.ae_state.params_view(), state.ae_state.model_state_view(),
                actions_t, act_emb_t,
                ae_cache_t, state.ae_ws_buf,
                emb_t, state.act_emb_buf,
                x_prev_t, c_in_t,
                state.pos_state.params_view(), state.pos_state.model_state_view(),
                x_prev_bh_t, x_prev_pe_bh_t,
                pos_cache_t, state.pos_ws_buf,
                state.adaln_states, state.msa_states, state.mlp_states,
                state.x_prev_pe_buf, state.x_inter_buf, state.pred_raw_buf,
                state.silu_cache_buf, state.adaln_cache_buf,
                state.ln1_cache_buf, state.mod1_cache_buf,
                state.msa_cache_buf, state.gate1_cache_buf,
                state.ln2_cache_buf, state.mod2_cache_buf,
                state.mlp_cache_buf, state.gate2_cache_buf,
                state.raw_mod_buf, state.x_mid_buf_d,
                silu_buf_t, ln_out_buf_t, mod_inp_buf_t,
                mod_x_buf_t, branch_out_buf_t, gate_inp_buf_t,
                state.adaln_ws_buf, state.msa_ws_buf, state.mlp_ws_buf,
                state.proj_state.params_view(), state.proj_state.model_state_view(),
                proj_cache_t, state.proj_ws_buf,
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

        var avg_expert = h6_sum_expert / Float64(self.eval_steps)
        var avg_shuf_mean = h6_sum_shuf_mean / Float64(self.eval_steps)
        var avg_shuf_min = h6_sum_shuf_min / Float64(self.eval_steps)
        var avg_better = h6_sum_better / Float64(self.eval_steps)
        print()
        print(
            "H6 summary (", self.eval_steps,
            "iters x ", self.eval_samples, "shuffled samples):",
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

    def eval_random_shots(
        mut self,
        mut state: Self.GPUState,
        ctx: DeviceContext,
    ) raises:
        comptime IMG_DIM = Self.GPUState.IMG_DIM
        comptime BT = Self.GPUState.BT
        comptime BTH = Self.GPUState.BTH
        comptime ENC = Self.GPUState.ENC
        comptime AE = Self.GPUState.AE
        comptime POS = Self.GPUState.POS
        comptime PROJ = Self.GPUState.PROJ

        var pixels_t = LayoutTensor[
            dtype, Layout.row_major(BT, IMG_DIM), MutAnyOrigin
        ](state.pixels_buf)
        var actions_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.T * Self.ACT), MutAnyOrigin
        ](state.actions_buf)
        var emb_t = LayoutTensor[
            dtype, Layout.row_major(BT, Self.EMB), MutAnyOrigin
        ](state.emb_buf)
        var enc_cache_t = LayoutTensor[
            dtype, Layout.row_major(BT, ENC.CACHE_SIZE), MutAnyOrigin
        ](state.enc_cache_buf)
        var act_emb_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.T * Self.EMB), MutAnyOrigin
        ](state.act_emb_buf)
        var ae_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, AE.CACHE_SIZE), MutAnyOrigin
        ](state.ae_cache_buf)
        var x_prev_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.x_prev_buf)
        var x_prev_bh_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.H * Self.EMB), MutAnyOrigin
        ](state.x_prev_buf)
        var x_prev_pe_bh_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.H * Self.EMB), MutAnyOrigin
        ](state.x_prev_pe_buf)
        var pos_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, POS.CACHE_SIZE), MutAnyOrigin
        ](state.pos_cache_buf)
        var c_in_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.c_in_buf)
        var pred_raw_bh_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.H * Self.EMB), MutAnyOrigin
        ](state.pred_raw_buf)
        var pred_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.H * Self.EMB), MutAnyOrigin
        ](state.pred_out_buf)
        var proj_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, PROJ.CACHE_SIZE), MutAnyOrigin
        ](state.proj_cache_buf)

        var silu_buf_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.silu_buf_d)
        var ln_out_buf_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.ln_out_buf_d)
        var mod_inp_buf_t = LayoutTensor[
            dtype, Layout.row_major(BTH, 3 * Self.EMB), MutAnyOrigin
        ](state.mod_inp_buf_d)
        var mod_x_buf_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.mod_x_buf_d)
        var branch_out_buf_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.branch_out_buf_d)
        var gate_inp_buf_t = LayoutTensor[
            dtype, Layout.row_major(BTH, 3 * Self.EMB), MutAnyOrigin
        ](state.gate_inp_buf_d)

        # ------------------------------------------------------------------
        # Phase 4 eval — random action shooter (teacher-forced)
        #
        # For each eval iteration:
        #   1. Sample fresh batch.
        #   2. Forward with EXPERT actions -> expert_loss = MSE(pred, real_emb[1:Self.H+1]).
        #   3. For S random samples, replace actions with random one-hot and
        #      re-run AE + POS + Self.DEPTH + PROJ (encoder unchanged).
        #   4. Report ratio expert/random — if << 1, model is action-aware.
        #
        # This is a "teacher-forced" shooter — it scores action sequences against
        # the actual observed next-frame embeddings (NOT a goal frame). True
        # autoregressive MPC with a goal frame is Phase 4b.
        # ------------------------------------------------------------------
        print()
        print("==== Phase 4 eval: random action shooter (teacher-forced) ====")
        _set_seed(self.eval_seed)

        var mse_div = Float64(Self.BATCH * Self.H * Self.EMB)
        var sum_expert: Float64 = 0.0
        var sum_random_mean: Float64 = 0.0
        var sum_random_min: Float64 = 0.0
        var sum_better_frac: Float64 = 0.0

        for eval_iter in range(self.eval_steps):
            # Sample uint8 pixels + actions; convert pixels on GPU.
            self._sample_and_upload_pixels(state, ctx)
            ctx.enqueue_copy(state.actions_buf, state.actions_host)

            # Encoder runs once per eval iter (pixels unchanged across S).
            ENC.forward_gpu[BT, dtype](
                ctx, emb_t, pixels_t,
                state.enc_state.params_view(), state.enc_state.model_state_view(),
                enc_cache_t, state.enc_ws_buf,
            )

            # Download emb once — used as target for all S shots.
            ctx.enqueue_copy(state.emb_host, state.emb_buf)
            ctx.synchronize()

            var expert_loss: Float64 = 0.0
            var random_mean: Float64 = 0.0
            var random_min: Float64 = 1e30
            var better_count: Int = 0

            # s == 0 -> expert actions (already in actions_buf).
            # s >= 1 -> random one-hot actions.
            for s in range(1 + self.eval_samples):
                if s > 0:
                    # Generate random one-hot actions (Self.BATCH, Self.T, Self.ACT).
                    for b in range(Self.BATCH):
                        for tt in range(Self.T):
                            var r_act = Int(random_float64() * Float64(Self.ACT))
                            if r_act >= Self.ACT:
                                r_act = Self.ACT - 1
                            for k in range(Self.ACT):
                                state.actions_host[b * Self.T * Self.ACT + tt * Self.ACT + k] = (
                                    Scalar[dtype](1.0)
                                    if k == r_act
                                    else Scalar[dtype](0.0)
                                )
                    ctx.enqueue_copy(state.actions_buf, state.actions_host)

                # One shot through AE + slice + POS + Self.DEPTH × cond_block + PROJ.
                _run_eval_shot_forward[
                    Self.BATCH, Self.T, Self.H, Self.EMB, Self.ACT, Self.SMOOTHED, Self.PROJ_H,
                    Self.PRED_HEADS, Self.PRED_FF, Self.DEPTH,
                ](
                    ctx,
                    state.ae_state.params_view(), state.ae_state.model_state_view(),
                    actions_t, act_emb_t,
                    ae_cache_t, state.ae_ws_buf,
                    emb_t, state.act_emb_buf,
                    x_prev_t, c_in_t,
                    state.pos_state.params_view(), state.pos_state.model_state_view(),
                    x_prev_bh_t, x_prev_pe_bh_t,
                    pos_cache_t, state.pos_ws_buf,
                    state.adaln_states, state.msa_states, state.mlp_states,
                    state.x_prev_pe_buf, state.x_inter_buf, state.pred_raw_buf,
                    state.silu_cache_buf, state.adaln_cache_buf,
                    state.ln1_cache_buf, state.mod1_cache_buf,
                    state.msa_cache_buf, state.gate1_cache_buf,
                    state.ln2_cache_buf, state.mod2_cache_buf,
                    state.mlp_cache_buf, state.gate2_cache_buf,
                    state.raw_mod_buf, state.x_mid_buf_d,
                    silu_buf_t, ln_out_buf_t, mod_inp_buf_t,
                    mod_x_buf_t, branch_out_buf_t, gate_inp_buf_t,
                    state.adaln_ws_buf, state.msa_ws_buf, state.mlp_ws_buf,
                    state.proj_state.params_view(), state.proj_state.model_state_view(),
                    proj_cache_t, state.proj_ws_buf,
                    pred_raw_bh_t, pred_t,
                )

                # Download pred, score MSE against emb[Self.N_PREDS:Self.N_PREDS+Self.H].
                ctx.enqueue_copy(state.pred_host, state.pred_out_buf)
                ctx.synchronize()
                var l: Float64 = 0.0
                for b in range(Self.BATCH):
                    for i in range(Self.H * Self.EMB):
                        var p = Float64(state.pred_host[b * Self.H * Self.EMB + i])
                        var tgt = Float64(
                            state.emb_host[b * Self.T * Self.EMB + Self.N_PREDS * Self.EMB + i]
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

            random_mean /= Float64(self.eval_samples)
            var better_frac = (
                Float64(better_count) / Float64(self.eval_samples)
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

        var avg_expert = sum_expert / Float64(self.eval_steps)
        var avg_rand_mean = sum_random_mean / Float64(self.eval_steps)
        var avg_rand_min = sum_random_min / Float64(self.eval_steps)
        var avg_better = sum_better_frac / Float64(self.eval_steps)
        print()
        print("Phase 4 eval summary (",
            self.eval_steps, "iters x ", self.eval_samples, "random samples):"
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

    def eval_mpc_cem(
        mut self,
        mut state: Self.GPUState,
        ctx: DeviceContext,
    ) raises:
        comptime IMG_DIM = Self.GPUState.IMG_DIM
        comptime BT = Self.GPUState.BT
        comptime BTH = Self.GPUState.BTH
        comptime ENC = Self.GPUState.ENC
        comptime AE = Self.GPUState.AE
        comptime POS = Self.GPUState.POS
        comptime PROJ = Self.GPUState.PROJ

        var pixels_t = LayoutTensor[
            dtype, Layout.row_major(BT, IMG_DIM), MutAnyOrigin
        ](state.pixels_buf)
        var actions_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.T * Self.ACT), MutAnyOrigin
        ](state.actions_buf)
        var emb_t = LayoutTensor[
            dtype, Layout.row_major(BT, Self.EMB), MutAnyOrigin
        ](state.emb_buf)
        var enc_cache_t = LayoutTensor[
            dtype, Layout.row_major(BT, ENC.CACHE_SIZE), MutAnyOrigin
        ](state.enc_cache_buf)
        var act_emb_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.T * Self.EMB), MutAnyOrigin
        ](state.act_emb_buf)
        var ae_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, AE.CACHE_SIZE), MutAnyOrigin
        ](state.ae_cache_buf)
        var x_prev_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.x_prev_buf)
        var x_prev_bh_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.H * Self.EMB), MutAnyOrigin
        ](state.x_prev_buf)
        var x_prev_pe_bh_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.H * Self.EMB), MutAnyOrigin
        ](state.x_prev_pe_buf)
        var pos_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, POS.CACHE_SIZE), MutAnyOrigin
        ](state.pos_cache_buf)
        var c_in_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.c_in_buf)
        var pred_raw_bh_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.H * Self.EMB), MutAnyOrigin
        ](state.pred_raw_buf)
        var pred_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.H * Self.EMB), MutAnyOrigin
        ](state.pred_out_buf)
        var proj_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, PROJ.CACHE_SIZE), MutAnyOrigin
        ](state.proj_cache_buf)

        var silu_buf_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.silu_buf_d)
        var ln_out_buf_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.ln_out_buf_d)
        var mod_inp_buf_t = LayoutTensor[
            dtype, Layout.row_major(BTH, 3 * Self.EMB), MutAnyOrigin
        ](state.mod_inp_buf_d)
        var mod_x_buf_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.mod_x_buf_d)
        var branch_out_buf_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.branch_out_buf_d)
        var gate_inp_buf_t = LayoutTensor[
            dtype, Layout.row_major(BTH, 3 * Self.EMB), MutAnyOrigin
        ](state.gate_inp_buf_d)

        # ------------------------------------------------------------------
        # Phase 4b eval — autoregressive MPC against a goal frame.
        #
        # For each eval iter:
        #   1. Sample Self.BATCH windows of length Self.T. Use frame 0 as start, frame
        #      Self.T-1 as goal. Encode the full window once via ENC.
        #   2. For each shot (1 expert + S random):
        #        a. Build action plan of length mpc_horizon + Self.H - 1. The first
        #           Self.H actions form the initial window; subsequent actions
        #           advance the window by 1 per rollout step.
        #        b. Initialize emb_seq[b, 0..H-1] = emb_start replicated Self.H
        #           times. (We have no real history; we pad with start.)
        #        c. For k = 0..mpc_horizon-1:
        #           - Upload emb_seq[b, k..k+Self.H-1] -> emb_buf positions 0..H-1.
        #           - Upload action_plan[b, k..k+Self.H-1] -> actions_buf positions 0..H-1.
        #           - Run _run_eval_shot_forward.
        #           - Download pred; take pred[:, Self.H-1, :] as new emb.
        #           - Store at emb_seq[b, k+Self.H].
        #        d. Score: MSE(emb_seq[b, Self.H+mpc_horizon-1], emb_goal[b]).
        #   3. Aggregate expert vs random over shots.
        #
        # mpc_horizon ≤ Self.T - Self.H + 1 (limited by sampled action window length).
        # ------------------------------------------------------------------
        # mpc_horizon validation — need Self.H + mpc_horizon - 1 ≤ Self.T actions.
        var needed_actions = Self.H + self.mpc_horizon - 1
        if needed_actions > Self.T:
            raise Error(
                "mpc_horizon too large: H + mpc_horizon - 1 > T"
                " (need bigger T or smaller horizon)"
            )

        print()
        print(
            "==== Phase 4b eval: autoregressive MPC (horizon=",
            self.mpc_horizon, ") ===="
        )
        _set_seed(self.eval_seed)

        # Host scratch — start/goal/action_plan staged on host before
        # upload; sample storage for CEM elites stays on host.
        var emb_start_host_buf = alloc[Scalar[dtype]](Self.BATCH * Self.EMB)
        var emb_goal_host_buf = alloc[Scalar[dtype]](Self.BATCH * Self.EMB)
        var action_plan_host_buf = alloc[Scalar[dtype]](
            Self.BATCH * needed_actions * Self.ACT
        )
        # CEM-specific host scratch.
        var cem_active = self.cem_iters > 0
        var _cs = self.cem_samples if cem_active else 1
        var _ck = self.cem_topk if cem_active else 1
        var action_dist_host_buf = alloc[Scalar[dtype]](
            Self.BATCH * needed_actions * Self.ACT
        )
        var sample_actions_host_buf = alloc[Scalar[dtype]](
            _cs * Self.BATCH * needed_actions * Self.ACT
        )
        var sample_scores_host_buf = alloc[Float64](_cs)
        var elite_indices_host_buf = alloc[Int](_ck)

        # GPU-resident rollout state — emb_seq sized for ROLL_T_MAX = Self.T + 1
        # positions (worst case Self.H + mpc_horizon ≤ Self.T + 1).
        var emb_start_dev_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.EMB
        )
        var emb_goal_dev_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH * Self.EMB)
        var emb_seq_dev_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * (Self.T + 1) * Self.EMB
        )
        var action_plan_dev_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.T * Self.ACT
        )
        var score_dev_buf = ctx.enqueue_create_buffer[dtype](1)
        var score_host_buf = ctx.enqueue_create_host_buffer[dtype](1)
        var emb_start_stage_host = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH * Self.EMB
        )
        var emb_goal_stage_host = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH * Self.EMB
        )
        var action_plan_stage_host = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH * Self.T * Self.ACT
        )

        var emb_start_dev_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.EMB), MutAnyOrigin
        ](emb_start_dev_buf.unsafe_ptr())
        var emb_goal_dev_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.EMB), MutAnyOrigin
        ](emb_goal_dev_buf.unsafe_ptr())
        var emb_seq_dev_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, (Self.T + 1) * Self.EMB), MutAnyOrigin
        ](emb_seq_dev_buf.unsafe_ptr())
        var action_plan_dev_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.T * Self.ACT), MutAnyOrigin
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

        for eval_iter in range(self.eval_steps):
            # Sample uint8 pixels + actions; convert pixels on GPU.
            # Actions stay in actions_host (read later for shot s=0 expert plan)
            # but are NOT uploaded to actions_buf — MPC's CEM shots overwrite it.
            self._sample_and_upload_pixels(state, ctx)

            ENC.forward_gpu[BT, dtype](
                ctx, emb_t, pixels_t,
                state.enc_state.params_view(), state.enc_state.model_state_view(),
                enc_cache_t, state.enc_ws_buf,
            )
            ctx.enqueue_copy(state.emb_host, state.emb_buf)
            ctx.synchronize()

            # Extract start (frame 0) + goal (frame Self.T-1) per batch row,
            # upload both to device.
            for b in range(Self.BATCH):
                for d in range(Self.EMB):
                    emb_start_stage_host[b * Self.EMB + d] = (
                        state.emb_host[b * Self.T * Self.EMB + d]
                    )
                    emb_goal_stage_host[b * Self.EMB + d] = (
                        state.emb_host[b * Self.T * Self.EMB + (Self.T - 1) * Self.EMB + d]
                    )
            ctx.enqueue_copy(emb_start_dev_buf, emb_start_stage_host)
            ctx.enqueue_copy(emb_goal_dev_buf, emb_goal_stage_host)

            var expert_loss_mpc: Float64 = 0.0
            var random_mean_mpc: Float64 = 0.0
            var random_min_mpc: Float64 = 1e30
            var better_count_mpc: Int = 0

            for s in range(1 + self.eval_samples):
                # Build action plan (Self.BATCH, needed_actions, Self.ACT) on host.
                if s == 0:
                    for b in range(Self.BATCH):
                        for ti in range(needed_actions):
                            for k in range(Self.ACT):
                                action_plan_host_buf[
                                    b * needed_actions * Self.ACT + ti * Self.ACT + k
                                ] = state.actions_host[
                                    b * Self.T * Self.ACT + ti * Self.ACT + k
                                ]
                else:
                    for b in range(Self.BATCH):
                        for ti in range(needed_actions):
                            var r_act = Int(
                                random_float64() * Float64(Self.ACT)
                            )
                            if r_act >= Self.ACT:
                                r_act = Self.ACT - 1
                            for k in range(Self.ACT):
                                action_plan_host_buf[
                                    b * needed_actions * Self.ACT
                                    + ti * Self.ACT + k
                                ] = (
                                    Scalar[dtype](1.0)
                                    if k == r_act
                                    else Scalar[dtype](0.0)
                                )

                # Stage action_plan to (Self.BATCH, Self.T, Self.ACT) layout (positions
                # [needed_actions..T-1] zero-padded; slide_actions_window
                # only reads up to k+Self.H-1 ≤ needed_actions-1).
                for b in range(Self.BATCH):
                    for ti in range(needed_actions):
                        for k in range(Self.ACT):
                            action_plan_stage_host[
                                b * Self.T * Self.ACT + ti * Self.ACT + k
                            ] = action_plan_host_buf[
                                b * needed_actions * Self.ACT + ti * Self.ACT + k
                            ]
                    for t_pad in range(Self.T - needed_actions):
                        for k in range(Self.ACT):
                            action_plan_stage_host[
                                b * Self.T * Self.ACT
                                + (needed_actions + t_pad) * Self.ACT + k
                            ] = Scalar[dtype](0.0)
                ctx.enqueue_copy(
                    action_plan_dev_buf, action_plan_stage_host
                )

                var l = _run_mpc_shot[
                    Self.BATCH, Self.T, Self.H, Self.EMB, Self.ACT, Self.SMOOTHED, Self.PROJ_H,
                    Self.PRED_HEADS, Self.PRED_FF, Self.DEPTH,
                ](
                    ctx,
                    self.mpc_horizon, needed_actions,
                    emb_start_dev_t, emb_goal_dev_t,
                    emb_seq_dev_t, action_plan_dev_t,
                    score_dev_t, score_dev_buf, score_host_buf,
                    state.ae_state.params_view(), state.ae_state.model_state_view(),
                    actions_t, act_emb_t,
                    ae_cache_t, state.ae_ws_buf,
                    emb_t, state.act_emb_buf,
                    x_prev_t, c_in_t,
                    state.pos_state.params_view(), state.pos_state.model_state_view(),
                    x_prev_bh_t, x_prev_pe_bh_t,
                    pos_cache_t, state.pos_ws_buf,
                    state.adaln_states, state.msa_states, state.mlp_states,
                    state.x_prev_pe_buf, state.x_inter_buf, state.pred_raw_buf,
                    state.silu_cache_buf, state.adaln_cache_buf,
                    state.ln1_cache_buf, state.mod1_cache_buf,
                    state.msa_cache_buf, state.gate1_cache_buf,
                    state.ln2_cache_buf, state.mod2_cache_buf,
                    state.mlp_cache_buf, state.gate2_cache_buf,
                    state.raw_mod_buf, state.x_mid_buf_d,
                    silu_buf_t, ln_out_buf_t, mod_inp_buf_t,
                    mod_x_buf_t, branch_out_buf_t, gate_inp_buf_t,
                    state.adaln_ws_buf, state.msa_ws_buf, state.mlp_ws_buf,
                    state.proj_state.params_view(), state.proj_state.model_state_view(),
                    proj_cache_t, state.proj_ws_buf,
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

            random_mean_mpc /= Float64(self.eval_samples)
            var better_frac_mpc = (
                Float64(better_count_mpc) / Float64(self.eval_samples)
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
                    Self.BATCH, Self.T, Self.H, Self.EMB, Self.ACT,
                    Self.SMOOTHED, Self.PROJ_H,
                    Self.PRED_HEADS, Self.PRED_FF, Self.DEPTH,
                ](
                    ctx,
                    self.mpc_horizon, needed_actions,
                    self.cem_iters, self.cem_samples, self.cem_topk, self.cem_smoothing,
                    action_dist_host_buf, action_plan_host_buf,
                    sample_actions_host_buf, sample_scores_host_buf,
                    elite_indices_host_buf,
                    emb_start_dev_t, emb_goal_dev_t,
                    emb_seq_dev_t, action_plan_dev_t,
                    action_plan_dev_buf,
                    score_dev_t, score_dev_buf, score_host_buf,
                    action_plan_stage_host,
                    state.ae_state.params_view(), state.ae_state.model_state_view(),
                    actions_t, act_emb_t,
                    ae_cache_t, state.ae_ws_buf,
                    emb_t, state.act_emb_buf,
                    x_prev_t, c_in_t,
                    state.pos_state.params_view(), state.pos_state.model_state_view(),
                    x_prev_bh_t, x_prev_pe_bh_t,
                    pos_cache_t, state.pos_ws_buf,
                    state.adaln_states, state.msa_states, state.mlp_states,
                    state.x_prev_pe_buf, state.x_inter_buf, state.pred_raw_buf,
                    state.silu_cache_buf, state.adaln_cache_buf,
                    state.ln1_cache_buf, state.mod1_cache_buf,
                    state.msa_cache_buf, state.gate1_cache_buf,
                    state.ln2_cache_buf, state.mod2_cache_buf,
                    state.mlp_cache_buf, state.gate2_cache_buf,
                    state.raw_mod_buf, state.x_mid_buf_d,
                    silu_buf_t, ln_out_buf_t, mod_inp_buf_t,
                    mod_x_buf_t, branch_out_buf_t, gate_inp_buf_t,
                    state.adaln_ws_buf, state.msa_ws_buf, state.mlp_ws_buf,
                    state.proj_state.params_view(), state.proj_state.model_state_view(),
                    proj_cache_t, state.proj_ws_buf,
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

        var avg_expert_mpc = sum_expert_mpc / Float64(self.eval_steps)
        var avg_rand_mean_mpc = (
            sum_random_mean_mpc / Float64(self.eval_steps)
        )
        var avg_rand_min_mpc = sum_random_min_mpc / Float64(self.eval_steps)
        var avg_better_mpc = sum_better_frac_mpc / Float64(self.eval_steps)
        print()
        print("Phase 4b MPC eval summary (",
            self.eval_steps, "iters x ", self.eval_samples, "shots, horizon=",
            self.mpc_horizon, "):"
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
            var avg_cem = sum_cem / Float64(self.eval_steps)
            var cem_vs_expert_frac = (
                Float64(cem_better_expert) / Float64(self.eval_steps)
            )
            var cem_vs_rmin_frac = (
                Float64(cem_better_random_min) / Float64(self.eval_steps)
            )
            print()
            print("Phase 4c CEM eval summary (",
                self.eval_steps, "iters x ", self.cem_iters, "CEM iters x ",
                self.cem_samples, "samples, top", self.cem_topk, "):"
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

    def eval_h7_closed_loop_drift(
        mut self,
        mut state: Self.GPUState,
        ctx: DeviceContext,
    ) raises:
        # H7 — autoregressive closed-loop drift diagnostic.
        #
        # For each eval iter: sample a T-length clip, encode all T frames, then
        # run rollout_steps = T - H forward passes. At step k ∈ [0, T-H-1] the
        # input window slides from emb_seq[k..k+H-1]; the model emits pred
        # (BATCH, H*EMB) and we compare pred[:, H-1, :] (prediction for position
        # k+H) against the REAL encoder output at that position.
        #
        # Two modes per iter, run sequentially on the same sampled batch:
        #   - TF (teacher-forced): emb_seq stays at REAL values throughout. Every
        #     step sees real input. drift_tf[k] is the model's 1-step error.
        #   - CL (closed-loop): pred[:, H-1, :] is stored back into emb_seq[k+H]
        #     after each step (via store_pred_last_kernel), so subsequent steps
        #     see the model's own prediction in their input window.
        #
        # drift_cl[0] == drift_tf[0] by construction (step 0's input window
        # emb_seq[0..H-1] is real in both modes). For k > 0 the modes diverge;
        # cl/tf >> 1 ⇒ predictions compound errors, cl/tf ~ 1 ⇒ model is stable
        # under its own rollouts.
        comptime IMG_DIM = Self.GPUState.IMG_DIM
        comptime BT = Self.GPUState.BT
        comptime BTH = Self.GPUState.BTH
        comptime ENC = Self.GPUState.ENC
        comptime AE = Self.GPUState.AE
        comptime POS = Self.GPUState.POS
        comptime PROJ = Self.GPUState.PROJ
        comptime ROLL_T = Self.T + 1

        var rollout_steps = Self.T - Self.H
        if rollout_steps <= 0:
            print()
            print(
                "==== H7: closed-loop drift — SKIPPED (T=",
                Self.T, " <= H=", Self.H, "; no rollout positions) ===="
            )
            return

        var pixels_t = LayoutTensor[
            dtype, Layout.row_major(BT, IMG_DIM), MutAnyOrigin
        ](state.pixels_buf)
        var actions_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.T * Self.ACT), MutAnyOrigin
        ](state.actions_buf)
        var emb_t = LayoutTensor[
            dtype, Layout.row_major(BT, Self.EMB), MutAnyOrigin
        ](state.emb_buf)
        var enc_cache_t = LayoutTensor[
            dtype, Layout.row_major(BT, ENC.CACHE_SIZE), MutAnyOrigin
        ](state.enc_cache_buf)
        var act_emb_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.T * Self.EMB), MutAnyOrigin
        ](state.act_emb_buf)
        var ae_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, AE.CACHE_SIZE), MutAnyOrigin
        ](state.ae_cache_buf)
        var x_prev_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.x_prev_buf)
        var x_prev_bh_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.H * Self.EMB), MutAnyOrigin
        ](state.x_prev_buf)
        var x_prev_pe_bh_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.H * Self.EMB), MutAnyOrigin
        ](state.x_prev_pe_buf)
        var pos_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, POS.CACHE_SIZE), MutAnyOrigin
        ](state.pos_cache_buf)
        var c_in_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.c_in_buf)
        var pred_raw_bh_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.H * Self.EMB), MutAnyOrigin
        ](state.pred_raw_buf)
        var pred_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.H * Self.EMB), MutAnyOrigin
        ](state.pred_out_buf)
        var proj_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, PROJ.CACHE_SIZE), MutAnyOrigin
        ](state.proj_cache_buf)

        var silu_buf_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.silu_buf_d)
        var ln_out_buf_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.ln_out_buf_d)
        var mod_inp_buf_t = LayoutTensor[
            dtype, Layout.row_major(BTH, 3 * Self.EMB), MutAnyOrigin
        ](state.mod_inp_buf_d)
        var mod_x_buf_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.mod_x_buf_d)
        var branch_out_buf_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.branch_out_buf_d)
        var gate_inp_buf_t = LayoutTensor[
            dtype, Layout.row_major(BTH, 3 * Self.EMB), MutAnyOrigin
        ](state.gate_inp_buf_d)

        print()
        print(
            "==== H7: closed-loop drift (rollout_steps=",
            rollout_steps, ") ===="
        )
        _set_seed(self.eval_seed)

        # Local rollout buffers (allocated/freed inside this eval phase).
        var emb_seq_dev_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * ROLL_T * Self.EMB
        )
        var emb_seq_dev_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, ROLL_T * Self.EMB),
            MutAnyOrigin,
        ](emb_seq_dev_buf.unsafe_ptr())
        var action_plan_dev_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.T * Self.ACT
        )
        var action_plan_dev_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.T * Self.ACT),
            MutAnyOrigin,
        ](action_plan_dev_buf.unsafe_ptr())
        var emb_seq_host = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH * ROLL_T * Self.EMB
        )

        var sum_drift_tf = alloc[Float64](rollout_steps)
        var sum_drift_cl = alloc[Float64](rollout_steps)
        var drift_tf_iter = alloc[Float64](rollout_steps)
        var drift_cl_iter = alloc[Float64](rollout_steps)
        for k in range(rollout_steps):
            sum_drift_tf[k] = 0.0
            sum_drift_cl[k] = 0.0

        var mse_div = Float64(Self.BATCH * Self.EMB)

        for eval_iter in range(self.eval_steps):
            # ---- 1. Sample uint8 pixels + actions; convert pixels on GPU ----
            self._sample_and_upload_pixels(state, ctx)
            ctx.enqueue_copy(state.actions_buf, state.actions_host)

            ENC.forward_gpu[BT, dtype](
                ctx, emb_t, pixels_t,
                state.enc_state.params_view(), state.enc_state.model_state_view(),
                enc_cache_t, state.enc_ws_buf,
            )
            ctx.enqueue_copy(state.emb_host, state.emb_buf)
            # Snapshot real actions into action_plan (slide window source —
            # state.actions_buf will be overwritten by slide_actions_window).
            ctx.enqueue_copy(action_plan_dev_buf, state.actions_buf)
            ctx.synchronize()

            # ---- 2. Init emb_seq from real_emb (positions 0..T-1; T zeroed) ----
            for i in range(Self.BATCH * ROLL_T * Self.EMB):
                emb_seq_host[i] = Scalar[dtype](0)
            for b in range(Self.BATCH):
                for t in range(Self.T):
                    for e in range(Self.EMB):
                        emb_seq_host[
                            b * ROLL_T * Self.EMB + t * Self.EMB + e
                        ] = state.emb_host[
                            b * Self.T * Self.EMB + t * Self.EMB + e
                        ]
            ctx.enqueue_copy(emb_seq_dev_buf, emb_seq_host)
            ctx.synchronize()

            # ---- 3. TF rollout — emb_seq untouched; every step sees real input ----
            for k in range(rollout_steps):
                ctx.enqueue_function[
                    slide_emb_window_kernel[
                        Self.BATCH, Self.T, Self.H, Self.EMB, ROLL_T
                    ],
                ](
                    emb_seq_dev_t, emb_t, k,
                    grid_dim=(
                        ceildiv(Self.BATCH, TPB_X),
                        ceildiv(Self.T, TPB_Y),
                        ceildiv(Self.EMB, TPB_Z),
                    ),
                    block_dim=(TPB_X, TPB_Y, TPB_Z),
                )
                ctx.enqueue_function[
                    slide_actions_window_kernel[
                        Self.BATCH, Self.T, Self.H, Self.ACT, Self.T
                    ],
                ](
                    action_plan_dev_t, actions_t, k,
                    grid_dim=(
                        ceildiv(Self.BATCH, TPB_X),
                        ceildiv(Self.T, TPB_Y),
                        ceildiv(Self.ACT, TPB_Z),
                    ),
                    block_dim=(TPB_X, TPB_Y, TPB_Z),
                )
                _run_eval_shot_forward[
                    Self.BATCH, Self.T, Self.H, Self.EMB, Self.ACT,
                    Self.SMOOTHED, Self.PROJ_H,
                    Self.PRED_HEADS, Self.PRED_FF, Self.DEPTH,
                ](
                    ctx,
                    state.ae_state.params_view(),
                    state.ae_state.model_state_view(),
                    actions_t, act_emb_t, ae_cache_t, state.ae_ws_buf,
                    emb_t, state.act_emb_buf,
                    x_prev_t, c_in_t,
                    state.pos_state.params_view(),
                    state.pos_state.model_state_view(),
                    x_prev_bh_t, x_prev_pe_bh_t,
                    pos_cache_t, state.pos_ws_buf,
                    state.adaln_states, state.msa_states, state.mlp_states,
                    state.x_prev_pe_buf, state.x_inter_buf, state.pred_raw_buf,
                    state.silu_cache_buf, state.adaln_cache_buf,
                    state.ln1_cache_buf, state.mod1_cache_buf,
                    state.msa_cache_buf, state.gate1_cache_buf,
                    state.ln2_cache_buf, state.mod2_cache_buf,
                    state.mlp_cache_buf, state.gate2_cache_buf,
                    state.raw_mod_buf, state.x_mid_buf_d,
                    silu_buf_t, ln_out_buf_t, mod_inp_buf_t,
                    mod_x_buf_t, branch_out_buf_t, gate_inp_buf_t,
                    state.adaln_ws_buf, state.msa_ws_buf, state.mlp_ws_buf,
                    state.proj_state.params_view(),
                    state.proj_state.model_state_view(),
                    proj_cache_t, state.proj_ws_buf,
                    pred_raw_bh_t, pred_t,
                )
                ctx.enqueue_copy(state.pred_host, state.pred_out_buf)
                ctx.synchronize()
                var l: Float64 = 0.0
                for b in range(Self.BATCH):
                    for e in range(Self.EMB):
                        var p = Float64(state.pred_host[
                            b * Self.H * Self.EMB
                            + (Self.H - 1) * Self.EMB + e
                        ])
                        var tgt = Float64(state.emb_host[
                            b * Self.T * Self.EMB
                            + (k + Self.H) * Self.EMB + e
                        ])
                        var diff = p - tgt
                        l += diff * diff
                l /= mse_div
                drift_tf_iter[k] = l
                sum_drift_tf[k] += l

            # ---- 4. CL rollout — same loop, but write pred back into emb_seq ----
            # TF didn't modify emb_seq, so it still holds real values.
            for k in range(rollout_steps):
                ctx.enqueue_function[
                    slide_emb_window_kernel[
                        Self.BATCH, Self.T, Self.H, Self.EMB, ROLL_T
                    ],
                ](
                    emb_seq_dev_t, emb_t, k,
                    grid_dim=(
                        ceildiv(Self.BATCH, TPB_X),
                        ceildiv(Self.T, TPB_Y),
                        ceildiv(Self.EMB, TPB_Z),
                    ),
                    block_dim=(TPB_X, TPB_Y, TPB_Z),
                )
                ctx.enqueue_function[
                    slide_actions_window_kernel[
                        Self.BATCH, Self.T, Self.H, Self.ACT, Self.T
                    ],
                ](
                    action_plan_dev_t, actions_t, k,
                    grid_dim=(
                        ceildiv(Self.BATCH, TPB_X),
                        ceildiv(Self.T, TPB_Y),
                        ceildiv(Self.ACT, TPB_Z),
                    ),
                    block_dim=(TPB_X, TPB_Y, TPB_Z),
                )
                _run_eval_shot_forward[
                    Self.BATCH, Self.T, Self.H, Self.EMB, Self.ACT,
                    Self.SMOOTHED, Self.PROJ_H,
                    Self.PRED_HEADS, Self.PRED_FF, Self.DEPTH,
                ](
                    ctx,
                    state.ae_state.params_view(),
                    state.ae_state.model_state_view(),
                    actions_t, act_emb_t, ae_cache_t, state.ae_ws_buf,
                    emb_t, state.act_emb_buf,
                    x_prev_t, c_in_t,
                    state.pos_state.params_view(),
                    state.pos_state.model_state_view(),
                    x_prev_bh_t, x_prev_pe_bh_t,
                    pos_cache_t, state.pos_ws_buf,
                    state.adaln_states, state.msa_states, state.mlp_states,
                    state.x_prev_pe_buf, state.x_inter_buf, state.pred_raw_buf,
                    state.silu_cache_buf, state.adaln_cache_buf,
                    state.ln1_cache_buf, state.mod1_cache_buf,
                    state.msa_cache_buf, state.gate1_cache_buf,
                    state.ln2_cache_buf, state.mod2_cache_buf,
                    state.mlp_cache_buf, state.gate2_cache_buf,
                    state.raw_mod_buf, state.x_mid_buf_d,
                    silu_buf_t, ln_out_buf_t, mod_inp_buf_t,
                    mod_x_buf_t, branch_out_buf_t, gate_inp_buf_t,
                    state.adaln_ws_buf, state.msa_ws_buf, state.mlp_ws_buf,
                    state.proj_state.params_view(),
                    state.proj_state.model_state_view(),
                    proj_cache_t, state.proj_ws_buf,
                    pred_raw_bh_t, pred_t,
                )
                ctx.enqueue_copy(state.pred_host, state.pred_out_buf)
                ctx.synchronize()
                var l: Float64 = 0.0
                for b in range(Self.BATCH):
                    for e in range(Self.EMB):
                        var p = Float64(state.pred_host[
                            b * Self.H * Self.EMB
                            + (Self.H - 1) * Self.EMB + e
                        ])
                        var tgt = Float64(state.emb_host[
                            b * Self.T * Self.EMB
                            + (k + Self.H) * Self.EMB + e
                        ])
                        var diff = p - tgt
                        l += diff * diff
                l /= mse_div
                drift_cl_iter[k] = l
                sum_drift_cl[k] += l
                # Store pred[:, H-1, :] -> emb_seq[:, k+H, :] for next step.
                ctx.enqueue_function[
                    store_pred_last_kernel[
                        Self.BATCH, Self.H, Self.EMB, ROLL_T
                    ],
                ](
                    pred_t, emb_seq_dev_t, k,
                    grid_dim=(ceildiv(Self.BATCH, 16), ceildiv(Self.EMB, 16)),
                    block_dim=(16, 16),
                )

            # ---- 5. Per-iter print ----
            print("  iter", eval_iter)
            for k in range(rollout_steps):
                var ratio = drift_cl_iter[k] / (drift_tf_iter[k] + 1e-12)
                print(
                    "    step", k, " (pos=", k + Self.H, "):",
                    " tf=", drift_tf_iter[k],
                    " cl=", drift_cl_iter[k],
                    " cl/tf=", ratio,
                )

        # ---- 6. Summary ----
        print()
        print(
            "H7 summary (", self.eval_steps,
            "iters x", rollout_steps, "rollout steps):"
        )
        for k in range(rollout_steps):
            var avg_tf = sum_drift_tf[k] / Float64(self.eval_steps)
            var avg_cl = sum_drift_cl[k] / Float64(self.eval_steps)
            var ratio = avg_cl / (avg_tf + 1e-12)
            if k == 0:
                print(
                    "  step", k, "(pos=", k + Self.H, ")",
                    " avg_tf=", avg_tf,
                    " avg_cl=", avg_cl,
                    " cl/tf=", ratio,
                    " (sanity: ~1.0 — both modes see real input at step 0)",
                )
            else:
                print(
                    "  step", k, "(pos=", k + Self.H, ")",
                    " avg_tf=", avg_tf,
                    " avg_cl=", avg_cl,
                    " cl/tf=", ratio,
                    " (cl/tf > 1 = compounding; ~1 = stable rollouts)",
                )

        sum_drift_tf.free()
        sum_drift_cl.free()
        drift_tf_iter.free()
        drift_cl_iter.free()

    def _checkpoint_metadata(
        self, steps_completed: Int
    ) -> List[String]:
        """Build the metadata list saved with checkpoints.

        Records the comptime shape so the eval entry point can fail-fast
        on mismatch, plus a few useful diagnostics.
        """
        var meta = List[String]()
        meta.append("BATCH=" + String(Self.BATCH))
        meta.append("T=" + String(Self.T))
        meta.append("H=" + String(Self.H))
        meta.append("N_PREDS=" + String(Self.N_PREDS))
        meta.append("IN_CH=" + String(Self.IN_CH))
        meta.append("IMG=" + String(Self.IMG))
        meta.append("PATCH=" + String(Self.PATCH))
        meta.append("N_PATCHES=" + String(Self.N_PATCHES))
        meta.append("HIDDEN=" + String(Self.HIDDEN))
        meta.append("ENC_HEADS=" + String(Self.ENC_HEADS))
        meta.append("ENC_LAYERS=" + String(Self.ENC_LAYERS))
        meta.append("EMB=" + String(Self.EMB))
        meta.append("PROJ_H=" + String(Self.PROJ_H))
        meta.append("ACT=" + String(Self.ACT))
        meta.append("SMOOTHED=" + String(Self.SMOOTHED))
        meta.append("PRED_HEADS=" + String(Self.PRED_HEADS))
        meta.append("PRED_FF=" + String(Self.PRED_FF))
        meta.append("DEPTH=" + String(Self.DEPTH))
        meta.append("SIG_NUM_PROJ=" + String(Self.SIG_NUM_PROJ))
        meta.append("SIG_KNOTS=" + String(Self.SIG_KNOTS))
        meta.append("steps_completed=" + String(steps_completed))
        meta.append("loss_last=" + String(self.loss_last))
        meta.append("pred_ema=" + String(self.pred_ema))
        meta.append("var_min_ema=" + String(self.var_min_ema))
        return meta^

    def run_eval(
        mut self,
        mut state: Self.GPUState,
        ctx: DeviceContext,
        rng_seed: Int,
    ) raises:
        """Run all eval phases without training (assumes weights are loaded).

        Used by the eval-only entry points (`eval_lewm_offline_gpu`,
        `eval_lewm_offline_gpu_pusht`). Mirrors the eval section at
        the end of `run()` so adding a new eval phase only needs one
        edit there.
        """
        _set_seed(rng_seed)

        if self.eval_steps > 0 and self.eval_shuffle_diag:
            self.eval_h6(state, ctx)

        if self.eval_steps > 0 and self.eval_h7_closed_loop:
            self.eval_h7_closed_loop_drift(state, ctx)

        if self.eval_steps > 0 and self.mpc_horizon == 0:
            self.eval_random_shots(state, ctx)

        if self.eval_steps > 0 and self.mpc_horizon > 0:
            self.eval_mpc_cem(state, ctx)

    def run(
        mut self,
        mut state: Self.GPUState,
        ctx: DeviceContext,
        num_steps: Int,
        rng_seed: Int,
        var checkpoint_path: String = String(""),
        checkpoint_every: Int = 0,
    ) raises:
        # Print model size info (matches original at lines 1904-1927).
        print(
            "Models — ENC.PARAM=", Self.GPUState.ENC.PARAM_SIZE,
            " AE.PARAM=", Self.GPUState.AE.PARAM_SIZE,
            " POS.PARAM=", Self.GPUState.POS.PARAM_SIZE,
            " ADALN.PARAM=", Self.GPUState.ADALN.PARAM_SIZE,
            " MSA.PARAM=", Self.GPUState.MSA.PARAM_SIZE,
            " MLP.PARAM=", Self.GPUState.MLP.PARAM_SIZE,
            " PROJ.PARAM=", Self.GPUState.PROJ.PARAM_SIZE,
            " DEPTH=", Self.DEPTH,
        )
        var total_params = (
            Self.GPUState.ENC.PARAM_SIZE + Self.GPUState.AE.PARAM_SIZE
            + Self.GPUState.POS.PARAM_SIZE + Self.GPUState.PROJ.PARAM_SIZE
            + Self.DEPTH * (Self.GPUState.ADALN.PARAM_SIZE
                       + Self.GPUState.MSA.PARAM_SIZE
                       + Self.GPUState.MLP.PARAM_SIZE)
        )
        print("Total params (incl. DEPTH stack):", total_params)
        print(
            "Workspaces/sample — ENC=", Self.GPUState.ENC.WORKSPACE_SIZE_PER_SAMPLE,
            " AE=", Self.GPUState.AE.WORKSPACE_SIZE_PER_SAMPLE,
            " POS=", Self.GPUState.POS.WORKSPACE_SIZE_PER_SAMPLE,
            " ADALN=", Self.GPUState.ADALN.WORKSPACE_SIZE_PER_SAMPLE,
            " MSA=", Self.GPUState.MSA.WORKSPACE_SIZE_PER_SAMPLE,
            " MLP=", Self.GPUState.MLP.WORKSPACE_SIZE_PER_SAMPLE,
            " PROJ=", Self.GPUState.PROJ.WORKSPACE_SIZE_PER_SAMPLE,
        )

        _set_seed(rng_seed)
        self.t0_ns = perf_counter_ns()

        var has_ckpt = checkpoint_path.byte_length() > 0
        var checkpoint_path_owned = checkpoint_path^

        # ------------------------------------------------------------------
        # Step loop
        # ------------------------------------------------------------------
        for step in range(num_steps):
            _ = self.train_step(state, ctx, step, rng_seed)
            if (
                has_ckpt
                and checkpoint_every > 0
                and step > 0
                and (step + 1) % checkpoint_every == 0
            ):
                state.save_checkpoint(
                    ctx,
                    checkpoint_path_owned,
                    self._checkpoint_metadata(step + 1),
                )
                print("  [checkpoint] saved at step", step + 1, "to", checkpoint_path_owned)

        ctx.synchronize()
        var t1 = perf_counter_ns()
        var total_s = Float64(t1 - self.t0_ns) / 1e9
        print()
        print("Trained", num_steps, "steps in", total_s, "s")
        print("  loss_first =", self.loss_first)
        print("  loss_last  =", self.loss_last)
        print("  pred_ema   =", self.pred_ema)
        print(
            "  rel_drop   =",
            (self.loss_first - self.loss_last) / (self.loss_first + 1e-12),
        )
        print()
        print("Collapse probes (EMA across the run):")
        print("  var_min  =", self.var_min_ema, " (want > 0.1)")
        print("  var_mean =", self.var_mean_ema)
        print("  gram_off =", self.gram_ema, " (want < ~0.5)")

        if has_ckpt:
            state.save_checkpoint(
                ctx,
                checkpoint_path_owned,
                self._checkpoint_metadata(num_steps),
            )
            print("  [checkpoint] final save to", checkpoint_path_owned)

        if self.eval_steps > 0 and self.eval_shuffle_diag:
            self.eval_h6(state, ctx)

        if self.eval_steps > 0 and self.eval_h7_closed_loop:
            self.eval_h7_closed_loop_drift(state, ctx)

        if self.eval_steps > 0 and self.mpc_horizon == 0:
            self.eval_random_shots(state, ctx)

        if self.eval_steps > 0 and self.mpc_horizon > 0:
            self.eval_mpc_cem(state, ctx)


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
    eval_h7_closed_loop: Bool = True,
    var checkpoint_path: String = String(""),
    checkpoint_every: Int = 0,
    time_phases: Bool = False,
) raises:
    """LeWM offline GPU trainer entry point — Pong.

    Constructs a `LeWMGPUState` + `LeWMTrainer[..., PongBuffer]` and
    calls `trainer.run(...)`.

    `checkpoint_path` (non-empty) enables periodic + final checkpoint
    writes, consumed by `eval_lewm_offline_gpu`. `checkpoint_every`
    controls intermediate cadence (0 = final-only).
    """
    comptime assert DEPTH >= 1, "DEPTH must be >= 1"

    var ctx = DeviceContext()
    var state = LeWMGPUState[
        BATCH, T, H, N_PREDS, IN_CH, IMG, PATCH, N_PATCHES, HIDDEN,
        ENC_HEADS, ENC_LAYERS, EMB, PROJ_H, ACT, SMOOTHED, PRED_HEADS,
        PRED_FF, DEPTH, SIG_NUM_PROJ, SIG_KNOTS,
    ](ctx, lambda_sigreg)
    var buf = PongBuffer.load(buffer_path)
    print("Loaded Pong buffer:", buf.n_frames, "frames from", buffer_path)
    var trainer = LeWMTrainer[
        BATCH, T, H, N_PREDS, IN_CH, IMG, PATCH, N_PATCHES, HIDDEN,
        ENC_HEADS, ENC_LAYERS, EMB, PROJ_H, ACT, SMOOTHED, PRED_HEADS,
        PRED_FF, DEPTH, SIG_NUM_PROJ, SIG_KNOTS, PongBuffer,
    ](
        buf^, lambda_sigreg, log_every, eval_steps, eval_samples,
        eval_seed, mpc_horizon, cem_iters, cem_samples, cem_topk,
        cem_smoothing, eval_shuffle_diag, eval_h7_closed_loop,
        time_phases,
    )
    trainer.run(state, ctx, num_steps, rng_seed, checkpoint_path^, checkpoint_every)


def train_lewm_offline_gpu_pusht[
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
    FRAMESKIP: Int = 5,
    ACTION_DIM: Int = 2,
](
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
    eval_h7_closed_loop: Bool = True,
    var dataset_path: String = String(""),
    var checkpoint_path: String = String(""),
    checkpoint_every: Int = 0,
    time_phases: Bool = False,
) raises:
    """LeWM offline GPU trainer entry point — PushT (HF expert dataset).

    First run auto-downloads `quentinll/lewm-pusht` (~13 GB compressed,
    decompresses to ~15-25 GB at `~/.cache/mojo_rl/lewm_pusht/`); set
    `dataset_path` to point at an existing `.h5` to skip the download
    (used by fixture tests).

    Comptime invariants enforced by `comptime assert`:
      - `ACT == FRAMESKIP * ACTION_DIM` (paper's effective-action shape).
      - `IN_CH == 3` (PushT pixels are RGB, no frame stack).
      - `IMG * IMG * 3` matches the dataset's per-frame byte count.
    The dataset's runtime ``num_steps`` is set to ``T`` so the window
    buffer is sized correctly.
    """
    comptime assert DEPTH >= 1, "DEPTH must be >= 1"
    comptime assert ACT == FRAMESKIP * ACTION_DIM, \
        "ACT must equal FRAMESKIP * ACTION_DIM (effective action dim)"
    comptime assert IN_CH == 3, \
        "PushT pixels are RGB; IN_CH must be 3"

    var ctx = DeviceContext()
    var state = LeWMGPUState[
        BATCH, T, H, N_PREDS, IN_CH, IMG, PATCH, N_PATCHES, HIDDEN,
        ENC_HEADS, ENC_LAYERS, EMB, PROJ_H, ACT, SMOOTHED, PRED_HEADS,
        PRED_FF, DEPTH, SIG_NUM_PROJ, SIG_KNOTS,
    ](ctx, lambda_sigreg)
    var sampler = LewmPushTSampler(
        frameskip=FRAMESKIP, num_steps=T, path=dataset_path^,
    )
    if sampler.dataset.pixel_h != IMG or sampler.dataset.pixel_w != IMG:
        raise Error(
            "train_lewm_offline_gpu_pusht: dataset pixels are "
            + String(sampler.dataset.pixel_h)
            + "x"
            + String(sampler.dataset.pixel_w)
            + " but trainer was built with IMG="
            + String(IMG)
            + ". The `quentinll/lewm-pusht` H5 ships at 224x224; the LeWM"
            + " paper config (config/train/lewm.yaml) uses IMG=224,"
            + " PATCH=14 (256 patches)."
        )
    if sampler.dataset.action_dim != ACTION_DIM:
        raise Error(
            "train_lewm_offline_gpu_pusht: dataset action_dim="
            + String(sampler.dataset.action_dim)
            + " but trainer was built with ACTION_DIM="
            + String(ACTION_DIM)
        )
    var trainer = LeWMTrainer[
        BATCH, T, H, N_PREDS, IN_CH, IMG, PATCH, N_PATCHES, HIDDEN,
        ENC_HEADS, ENC_LAYERS, EMB, PROJ_H, ACT, SMOOTHED, PRED_HEADS,
        PRED_FF, DEPTH, SIG_NUM_PROJ, SIG_KNOTS, LewmPushTSampler,
    ](
        sampler^, lambda_sigreg, log_every, eval_steps, eval_samples,
        eval_seed, mpc_horizon, cem_iters, cem_samples, cem_topk,
        cem_smoothing, eval_shuffle_diag, eval_h7_closed_loop,
        time_phases,
    )
    trainer.run(state, ctx, num_steps, rng_seed, checkpoint_path^, checkpoint_every)


def eval_lewm_offline_gpu[
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
    checkpoint_path: String,
    eval_steps: Int = 10,
    eval_samples: Int = 32,
    eval_seed: Int = 0xBEEF,
    mpc_horizon: Int = 0,
    cem_iters: Int = 0,
    cem_samples: Int = 64,
    cem_topk: Int = 8,
    cem_smoothing: Float64 = 0.5,
    eval_shuffle_diag: Bool = True,
    eval_h7_closed_loop: Bool = True,
    lambda_sigreg: Float64 = 0.09,
) raises:
    """Load a Pong LeWM checkpoint and run only the eval phases.

    Symmetric with `train_lewm_offline_gpu` — same comptime params
    must match the binary that wrote the checkpoint. Reuses
    `PongBuffer` (loaded fresh from `buffer_path`) for the eval-time
    clip sampling.

    Returns when all enabled eval phases (`eval_h6`,
    `eval_h7_closed_loop_drift`, `eval_random_shots`, `eval_mpc_cem`)
    finish.
    """
    comptime assert DEPTH >= 1, "DEPTH must be >= 1"

    var ctx = DeviceContext()
    var state = LeWMGPUState[
        BATCH, T, H, N_PREDS, IN_CH, IMG, PATCH, N_PATCHES, HIDDEN,
        ENC_HEADS, ENC_LAYERS, EMB, PROJ_H, ACT, SMOOTHED, PRED_HEADS,
        PRED_FF, DEPTH, SIG_NUM_PROJ, SIG_KNOTS,
    ](ctx, lambda_sigreg)
    var meta = state.load_checkpoint(ctx, checkpoint_path)
    print("Loaded checkpoint from", checkpoint_path)
    for i in range(len(meta)):
        print("  meta:", meta[i])

    var buf = PongBuffer.load(buffer_path)
    print("Loaded Pong buffer:", buf.n_frames, "frames from", buffer_path)
    var trainer = LeWMTrainer[
        BATCH, T, H, N_PREDS, IN_CH, IMG, PATCH, N_PATCHES, HIDDEN,
        ENC_HEADS, ENC_LAYERS, EMB, PROJ_H, ACT, SMOOTHED, PRED_HEADS,
        PRED_FF, DEPTH, SIG_NUM_PROJ, SIG_KNOTS, PongBuffer,
    ](
        buf^, lambda_sigreg, 0, eval_steps, eval_samples,
        eval_seed, mpc_horizon, cem_iters, cem_samples, cem_topk,
        cem_smoothing, eval_shuffle_diag, eval_h7_closed_loop,
    )
    trainer.run_eval(state, ctx, eval_seed)


def eval_lewm_offline_gpu_pusht[
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
    FRAMESKIP: Int = 5,
    ACTION_DIM: Int = 2,
](
    checkpoint_path: String,
    eval_steps: Int = 10,
    eval_samples: Int = 32,
    eval_seed: Int = 0xBEEF,
    mpc_horizon: Int = 0,
    cem_iters: Int = 0,
    cem_samples: Int = 64,
    cem_topk: Int = 8,
    cem_smoothing: Float64 = 0.5,
    eval_shuffle_diag: Bool = True,
    eval_h7_closed_loop: Bool = True,
    lambda_sigreg: Float64 = 0.09,
    var dataset_path: String = String(""),
) raises:
    """Load a PushT LeWM checkpoint and run only the eval phases.

    Symmetric with `train_lewm_offline_gpu_pusht`. Comptime params
    must match the checkpoint's instantiation.
    """
    comptime assert DEPTH >= 1, "DEPTH must be >= 1"
    comptime assert ACT == FRAMESKIP * ACTION_DIM, \
        "ACT must equal FRAMESKIP * ACTION_DIM"
    comptime assert IN_CH == 3, "PushT pixels are RGB; IN_CH must be 3"

    var ctx = DeviceContext()
    var state = LeWMGPUState[
        BATCH, T, H, N_PREDS, IN_CH, IMG, PATCH, N_PATCHES, HIDDEN,
        ENC_HEADS, ENC_LAYERS, EMB, PROJ_H, ACT, SMOOTHED, PRED_HEADS,
        PRED_FF, DEPTH, SIG_NUM_PROJ, SIG_KNOTS,
    ](ctx, lambda_sigreg)
    var meta = state.load_checkpoint(ctx, checkpoint_path)
    print("Loaded checkpoint from", checkpoint_path)
    for i in range(len(meta)):
        print("  meta:", meta[i])

    var sampler = LewmPushTSampler(
        frameskip=FRAMESKIP, num_steps=T, path=dataset_path^,
    )
    if sampler.dataset.pixel_h != IMG or sampler.dataset.pixel_w != IMG:
        raise Error(
            "eval_lewm_offline_gpu_pusht: dataset pixels are "
            + String(sampler.dataset.pixel_h)
            + "x"
            + String(sampler.dataset.pixel_w)
            + " but eval driver built with IMG="
            + String(IMG)
        )
    var trainer = LeWMTrainer[
        BATCH, T, H, N_PREDS, IN_CH, IMG, PATCH, N_PATCHES, HIDDEN,
        ENC_HEADS, ENC_LAYERS, EMB, PROJ_H, ACT, SMOOTHED, PRED_HEADS,
        PRED_FF, DEPTH, SIG_NUM_PROJ, SIG_KNOTS, LewmPushTSampler,
    ](
        sampler^, lambda_sigreg, 0, eval_steps, eval_samples,
        eval_seed, mpc_horizon, cem_iters, cem_samples, cem_topk,
        cem_smoothing, eval_shuffle_diag, eval_h7_closed_loop,
    )
    trainer.run_eval(state, ctx, eval_seed)
