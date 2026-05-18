"""LeWM offline trainer — struct-based GPU training loop.

Two structs:

  - `LeWMGPUState[...]` — owns all GPU device/host buffers + per-model
    GPUNetworkStates. Comptime aliases for every model type are hoisted
    to struct level so they get instantiated once per (model-shape)
    specialization rather than once per call-site of the trainer.

  - `LeWMTrainer[..., BUF]` — owns a clip/window buffer, hyperparams,
    and per-run scalar EMAs. `BUF` is a comptime type parameter that must
    conform to `mojo_rl.core.offline_buffer.OfflineBuffer`: expose
    `INPUT_LAYOUT_HWC: Bool` comptime field and
    `sample_batch_uint8(B, T, pixels_u8_out, actions_out) raises` method.
    Concrete buffers: `PongOfflineBuffer` (in-RAM CHW uint8, in
    `mojo_rl.envs.arcade_games.pong.offline_buffer`) and
    `PushTOfflineSampler` (HDF5-backed HWC uint8, in
    `mojo_rl.envs.pusht.offline_sampler`). Each phase of training becomes
    its own method (`train_step`, `eval_h6`, `eval_random_shots`,
    `eval_mpc_cem`, `eval_h7_closed_loop_drift`, `run`, `run_eval`).

`train_lewm_offline_gpu` (Pong) and `train_lewm_offline_gpu_pusht`
(PushT HDF5) are the thin entry points that construct the appropriate
buffer + state + trainer and call `trainer.run(...)`. `eval_lewm_offline_gpu`
+ `eval_lewm_offline_gpu_pusht` are the checkpoint-load + eval-only
counterparts.

Module-level GPU kernels and per-layer training helpers
(run_cond_layer_forward/backward + the low-level kernels) live in
`kernels.mojo` — kept module-level on purpose because inlining them
explodes Mojo compile time.

Eval phases (H6 shuffle diagnostic, H7 closed-loop drift, random-shot
baseline) live in `eval_suite.LeWMEvalSuite`. Autoregressive MPC + CEM
refinement lives in `cem_planner.CEMPlanner`. The trainer constructs a
`LeWMEvalSuite` (which lazily constructs a `CEMPlanner` when
`mpc_horizon > 0`) for both `run_eval` (eval-only entry) and the
post-training eval block of `run`.
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
from ...nn.composites import TransformerBlock, MultiHeadAttention, MultiHeadAttentionXL
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
from mojo_rl.envs.arcade_games.pong.offline_buffer import (
    PongOfflineBuffer,
    PONG_FRAME_BYTES,
    PONG_NUM_ACTIONS,
)
from mojo_rl.envs.pusht.offline_sampler import PushTOfflineSampler
from mojo_rl.core.offline_buffer import OfflineBuffer
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
    slice_h_kernel, scatter_h_kernel, scatter_target_neg_kernel,
    accumulate_emb_kernel,
    pixels_uint8_to_fp32_kernel,
)
from .eval_suite import LeWMEvalSuite
from .lewm_config import LeWMConfig


@always_inline
def _max_int(a: Int, b: Int) -> Int:
    return a if a > b else b


comptime TPB_X = 4
comptime TPB_Y = 4
comptime TPB_Z = 16


struct LeWMGPUState[CONFIG: LeWMConfig]:
    """All GPU device + host buffers and per-model GPUNetworkStates.

    Comptime aliases for the model types are hoisted to struct level so
    they instantiate once per specialization (mirrors the TDMPC2Agent
    pattern of avoiding re-instantiation across `train_gpu[ENV, n_envs]`
    call sites). All 20 dimensional parameters + the encoder model type
    are read off `CONFIG` (a `LeWMConfig` instance).
    """

    # ── Model type aliases (hoisted) ─────────────────────────────────
    # Encoder pulled from CONFIG (swappable per concrete config: ViT,
    # CNN, etc.). Other networks stay local for Phase 3 — easy to lift
    # to CONFIG fields when a use case appears.
    comptime ENC = Self.CONFIG.EncoderModel

    # ── Derived dimension aliases ────────────────────────────────────
    # IMG_DIM/EMB use ENC's own IN_DIM/OUT_DIM (not arithmetic recomputed
    # off CONFIG) so LayoutTensors typed by `Layout.row_major(BT, EMB)`
    # are structurally identical to those typed by
    # `Layout.row_major(BT, ENC.OUT_DIM)` — Mojo treats `Self.EMB`
    # and `Self.ENC.OUT_DIM` as distinct comptime expressions even when
    # they evaluate to the same Int.
    comptime IMG_DIM: Int = Self.ENC.IN_DIM
    comptime EMB: Int = Self.ENC.OUT_DIM
    comptime BT: Int = Self.CONFIG.BATCH * Self.CONFIG.T
    comptime BTH: Int = Self.CONFIG.BATCH * Self.CONFIG.H
    comptime AE = ActionEmbedder[Self.CONFIG.T, Self.CONFIG.ACT, Self.CONFIG.SMOOTHED, Self.EMB]
    comptime POS = AutoDiffChain[BiasAdd[Self.CONFIG.H * Self.EMB]]
    comptime ADALN = AdaLNMod[Self.EMB]
    comptime MSA = MultiHeadAttentionXL[
        Self.EMB,
        Self.CONFIG.PRED_HEADS,
        Self.CONFIG.PRED_DIM_HEAD,
        Self.CONFIG.H,
        True,
    ]
    comptime MLP = CondMLP[Self.EMB, Self.CONFIG.PRED_FF]
    comptime _PredProjPerToken = Sequential[
        Linear[Self.EMB, Self.CONFIG.PROJ_H],
        BatchNorm1D[Self.CONFIG.PROJ_H],
        GELU[Self.CONFIG.PROJ_H],
        Linear[Self.CONFIG.PROJ_H, Self.EMB],
    ]
    comptime PROJ = Tokenwise[Self.CONFIG.H, Self._PredProjPerToken]
    comptime SIG = SIGRegOp[Self.EMB, Self.CONFIG.T, Self.CONFIG.SIG_NUM_PROJ, Self.CONFIG.SIG_KNOTS]
    comptime SIG_WS_SIZE = Self.SIG.workspace_size_for[Self.CONFIG.BATCH]()

    # ── Shared (single-instance) GPUNetworkStates ────────────────────
    var enc_state: GPUNetworkState[Self.ENC, Adam[]]
    var ae_state: GPUNetworkState[Self.AE, Adam[]]
    var pos_state: GPUNetworkState[Self.POS, Adam[]]
    var proj_state: GPUNetworkState[Self.PROJ, Adam[]]

    # ── Per-layer cond_block GPUNetworkStates (Self.CONFIG.DEPTH copies each) ────
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

    # cond_block caches — Self.CONFIG.DEPTH-fold (sliced per layer in helpers).
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
    # Intermediate x flow between layers. (Self.CONFIG.DEPTH-1) slots since layer 0 reads
    # x_prev_pe and layer Self.CONFIG.DEPTH-1 writes pred_raw directly.
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

    # cond_block backward scratch (reused across all Self.CONFIG.DEPTH layers).
    var sgg_buf: DeviceBuffer[dtype]
    var sgbo_buf: DeviceBuffer[dtype]
    var sgmx_buf: DeviceBuffer[dtype]
    var sgmi_buf: DeviceBuffer[dtype]
    var sglnout_buf: DeviceBuffer[dtype]
    var sglnin_buf: DeviceBuffer[dtype]
    var sgrm_buf: DeviceBuffer[dtype]
    var sgsc_buf: DeviceBuffer[dtype]
    var grad_x_mid_buf: DeviceBuffer[dtype]
    # Backward intermediate grad_x flow between layers (Self.CONFIG.DEPTH-1 slots).
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

        # Per-layer cond_block models — Self.CONFIG.DEPTH copies of ADALN, MSA, MLP.
        var cpu_adalns = List[NetworkState[Self.ADALN, Adam[]]](capacity=Self.CONFIG.DEPTH)
        var cpu_msas = List[NetworkState[Self.MSA, Adam[]]](capacity=Self.CONFIG.DEPTH)
        var cpu_mlps = List[NetworkState[Self.MLP, Adam[]]](capacity=Self.CONFIG.DEPTH)
        for _ in range(Self.CONFIG.DEPTH):
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

        self.adaln_states = List[GPUNetworkState[Self.ADALN, Adam[]]](capacity=Self.CONFIG.DEPTH)
        self.msa_states = List[GPUNetworkState[Self.MSA, Adam[]]](capacity=Self.CONFIG.DEPTH)
        self.mlp_states = List[GPUNetworkState[Self.MLP, Adam[]]](capacity=Self.CONFIG.DEPTH)
        for layer_idx in range(Self.CONFIG.DEPTH):
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
        self.actions_buf = ctx.enqueue_create_buffer[dtype](Self.CONFIG.BATCH * Self.CONFIG.T * Self.CONFIG.ACT)
        self.emb_buf = ctx.enqueue_create_buffer[dtype](Self.BT * Self.EMB)
        self.enc_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BT * Self.ENC.CACHE_SIZE)
        self.enc_ws_buf = ctx.enqueue_create_buffer[dtype](
            _max_int(1, Self.BT * Self.ENC.WORKSPACE_SIZE_PER_SAMPLE)
        )

        self.act_emb_buf = ctx.enqueue_create_buffer[dtype](Self.CONFIG.BATCH * Self.CONFIG.T * Self.EMB)
        self.ae_cache_buf = ctx.enqueue_create_buffer[dtype](Self.CONFIG.BATCH * Self.AE.CACHE_SIZE)
        self.ae_ws_buf = ctx.enqueue_create_buffer[dtype](
            _max_int(1, Self.CONFIG.BATCH * Self.AE.WORKSPACE_SIZE_PER_SAMPLE)
        )

        self.x_prev_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)
        self.x_prev_pe_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)
        self.pos_cache_buf = ctx.enqueue_create_buffer[dtype](
            _max_int(1, Self.CONFIG.BATCH * Self.POS.CACHE_SIZE)
        )
        self.pos_ws_buf = ctx.enqueue_create_buffer[dtype](
            _max_int(1, Self.CONFIG.BATCH * Self.POS.WORKSPACE_SIZE_PER_SAMPLE)
        )
        self.c_in_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)

        self.pred_raw_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)
        self.pred_out_buf = ctx.enqueue_create_buffer[dtype](Self.CONFIG.BATCH * Self.CONFIG.H * Self.EMB)
        self.proj_cache_buf = ctx.enqueue_create_buffer[dtype](Self.CONFIG.BATCH * Self.PROJ.CACHE_SIZE)
        self.proj_ws_buf = ctx.enqueue_create_buffer[dtype](
            _max_int(1, Self.CONFIG.BATCH * Self.PROJ.WORKSPACE_SIZE_PER_SAMPLE)
        )

        # cond_block caches — Self.CONFIG.DEPTH-fold (sliced per layer in helpers).
        self.silu_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB * Self.CONFIG.DEPTH)
        self.adaln_cache_buf = ctx.enqueue_create_buffer[dtype](
            Self.BTH * Self.ADALN.CACHE_SIZE * Self.CONFIG.DEPTH
        )
        self.ln1_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * (Self.EMB + 1) * Self.CONFIG.DEPTH)
        self.mod1_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * 2 * Self.EMB * Self.CONFIG.DEPTH)
        self.msa_cache_buf = ctx.enqueue_create_buffer[dtype](
            Self.CONFIG.BATCH * Self.MSA.CACHE_SIZE * Self.CONFIG.DEPTH
        )
        self.gate1_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * 2 * Self.EMB * Self.CONFIG.DEPTH)
        self.ln2_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * (Self.EMB + 1) * Self.CONFIG.DEPTH)
        self.mod2_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * 2 * Self.EMB * Self.CONFIG.DEPTH)
        self.mlp_cache_buf = ctx.enqueue_create_buffer[dtype](
            Self.BTH * Self.MLP.CACHE_SIZE * Self.CONFIG.DEPTH
        )
        self.gate2_cache_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * 2 * Self.EMB * Self.CONFIG.DEPTH)
        self.raw_mod_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * 6 * Self.EMB * Self.CONFIG.DEPTH)
        self.x_mid_buf_d = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB * Self.CONFIG.DEPTH)
        # Intermediate x flow between layers. (Self.CONFIG.DEPTH-1) slots since layer 0 reads
        # x_prev_pe and layer Self.CONFIG.DEPTH-1 writes pred_raw directly.
        self.x_inter_buf = ctx.enqueue_create_buffer[dtype](
            _max_int(1, Self.BTH * Self.EMB * (Self.CONFIG.DEPTH - 1))
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
            _max_int(1, Self.CONFIG.BATCH * Self.MSA.WORKSPACE_SIZE_PER_SAMPLE)
        )
        self.mlp_ws_buf = ctx.enqueue_create_buffer[dtype](
            _max_int(1, Self.BTH * Self.MLP.WORKSPACE_SIZE_PER_SAMPLE)
        )

        # cond_block backward scratch (reused across all Self.CONFIG.DEPTH layers).
        self.sgg_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * 3 * Self.EMB)
        self.sgbo_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)
        self.sgmx_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)
        self.sgmi_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * 3 * Self.EMB)
        self.sglnout_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)
        self.sglnin_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)
        self.sgrm_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * 6 * Self.EMB)
        self.sgsc_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)
        self.grad_x_mid_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)
        # Backward intermediate grad_x flow between layers (Self.CONFIG.DEPTH-1 slots).
        self.grad_x_inter_buf = ctx.enqueue_create_buffer[dtype](
            _max_int(1, Self.BTH * Self.EMB * (Self.CONFIG.DEPTH - 1))
        )
        # Per-layer grad_c output (single buffer, reused per layer; accumulated
        # into grad_c_buf via cb_accum_kernel).
        self.grad_c_layer_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)

        # SIGReg buffers (forward + backward).
        self.sigreg_out_buf = ctx.enqueue_create_buffer[dtype](Self.CONFIG.BATCH * Self.SIG.OUT_DIM)
        self.sigreg_cache_buf = ctx.enqueue_create_buffer[dtype](
            Self.CONFIG.BATCH * Self.SIG.CACHE_SIZE
        )
        self.sigreg_grad_out_buf = ctx.enqueue_create_buffer[dtype](
            Self.CONFIG.BATCH * Self.SIG.OUT_DIM
        )
        self.sigreg_grad_emb_buf = ctx.enqueue_create_buffer[dtype](
            Self.CONFIG.BATCH * Self.CONFIG.T * Self.EMB
        )
        self.sigreg_ws_buf = ctx.enqueue_create_buffer[dtype](Self.SIG_WS_SIZE)
        # Seed grad_output = λ/B (constant across all steps; chain rule produces
        # an effective G = λ at the SIGReg dLdz step). See CPU trainer line 735.
        var sigreg_grad_out_host = ctx.enqueue_create_host_buffer[dtype](
            Self.CONFIG.BATCH * Self.SIG.OUT_DIM
        )
        for i in range(Self.CONFIG.BATCH * Self.SIG.OUT_DIM):
            sigreg_grad_out_host[i] = Scalar[dtype](
                lambda_sigreg / Float64(Self.CONFIG.BATCH)
            )
        ctx.enqueue_copy(self.sigreg_grad_out_buf, sigreg_grad_out_host)

        # Gradient buffers (device).
        self.grad_pred_buf = ctx.enqueue_create_buffer[dtype](Self.CONFIG.BATCH * Self.CONFIG.H * Self.EMB)
        self.grad_pred_raw_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)
        self.grad_x_prev_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)
        self.grad_x_prev_pe_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)
        self.grad_c_buf = ctx.enqueue_create_buffer[dtype](Self.BTH * Self.EMB)
        self.grad_emb_buf = ctx.enqueue_create_buffer[dtype](Self.CONFIG.BATCH * Self.CONFIG.T * Self.EMB)
        self.grad_act_emb_buf = ctx.enqueue_create_buffer[dtype](Self.CONFIG.BATCH * Self.CONFIG.T * Self.EMB)
        self.grad_actions_buf = ctx.enqueue_create_buffer[dtype](Self.CONFIG.BATCH * Self.CONFIG.T * Self.CONFIG.ACT)
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
        self.actions_host = ctx.enqueue_create_host_buffer[dtype](Self.CONFIG.BATCH * Self.CONFIG.T * Self.CONFIG.ACT)
        self.pred_host = ctx.enqueue_create_host_buffer[dtype](Self.CONFIG.BATCH * Self.CONFIG.H * Self.EMB)
        self.target_host = ctx.enqueue_create_host_buffer[dtype](Self.CONFIG.BATCH * Self.CONFIG.H * Self.EMB)
        self.grad_pred_host = ctx.enqueue_create_host_buffer[dtype](
            Self.CONFIG.BATCH * Self.CONFIG.H * Self.EMB
        )
        self.sigreg_out_host = ctx.enqueue_create_host_buffer[dtype](
            Self.CONFIG.BATCH * Self.SIG.OUT_DIM
        )
        # emb on device has shape (BT, Self.EMB) — aliased as (Self.CONFIG.BATCH, Self.CONFIG.T*Self.EMB) for the
        # target slice. Same memory, single host buffer.
        self.emb_host = ctx.enqueue_create_host_buffer[dtype](Self.BT * Self.EMB)

        # Small scratch (BATCH*T*ACT floats — a few KB) for H6's expert-action snapshot.
        self.actions_sample = alloc[Scalar[dtype]](Self.CONFIG.BATCH * Self.CONFIG.T * Self.CONFIG.ACT)

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
            + Self.CONFIG.DEPTH * (
                self.adaln_states[0].PARAM_SIZE
                + self.msa_states[0].PARAM_SIZE
                + self.mlp_states[0].PARAM_SIZE
            )
        )
        var total_opt_state = (
            self.enc_state.OPT_STATE_SIZE + self.ae_state.OPT_STATE_SIZE
            + self.pos_state.OPT_STATE_SIZE + self.proj_state.OPT_STATE_SIZE
            + Self.CONFIG.DEPTH * (
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
        for i in range(Self.CONFIG.DEPTH):
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
        for i in range(Self.CONFIG.DEPTH):
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
    CONFIG: LeWMConfig,
    BUF: OfflineBuffer = PongOfflineBuffer,
]:
    """Owns hyperparams + per-run EMAs + a clip buffer; methods consume a
    `LeWMGPUState[CONFIG]` for the GPU-resident data.

    `CONFIG` carries the 20 dimensional parameters + swappable
    `EncoderModel` type. `BUF` is the buffer type — must implement
    `sample_batch_uint8(B, T, pixels_u8_out, actions_out) raises` and
    expose `INPUT_LAYOUT_HWC: Bool`. Concrete instances:
    `PongOfflineBuffer` (Atari-style pixel-obs replay) and
    `PushTOfflineSampler` (HDF5-backed expert clips for the LeWM paper
    recipe).
    """

    comptime GPUState = LeWMGPUState[Self.CONFIG]
    comptime EMB: Int = Self.GPUState.EMB

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
    var rh_steps: Int

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
        rh_steps: Int = 0,
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
        self.rh_steps = rh_steps

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

        self.loss_scale = Float64(Self.CONFIG.BATCH * Self.CONFIG.H * Self.EMB)
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
            Self.CONFIG.BATCH,
            Self.CONFIG.T,
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
                BT, Self.CONFIG.IN_CH, Self.CONFIG.IMG, Self.BUF.INPUT_LAYOUT_HWC,
            ],
        ](
            src_u8_t,
            dst_fp32_t,
            grid_dim=(
                ceildiv(BT, TPB_X),
                ceildiv(Self.CONFIG.IN_CH, TPB_Y),
                ceildiv(Self.CONFIG.IMG * Self.CONFIG.IMG, TPB_Z),
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
            dtype, Layout.row_major(Self.CONFIG.BATCH, Self.CONFIG.T * Self.CONFIG.ACT), MutAnyOrigin
        ](state.actions_buf)
        var emb_t = LayoutTensor[
            dtype, Layout.row_major(BT, Self.EMB), MutAnyOrigin
        ](state.emb_buf)
        var enc_cache_t = LayoutTensor[
            dtype, Layout.row_major(BT, ENC.CACHE_SIZE), MutAnyOrigin
        ](state.enc_cache_buf)

        var act_emb_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, Self.CONFIG.T * Self.EMB), MutAnyOrigin
        ](state.act_emb_buf)
        var ae_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, AE.CACHE_SIZE), MutAnyOrigin
        ](state.ae_cache_buf)

        var x_prev_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.x_prev_buf)
        var x_prev_bh_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, Self.CONFIG.H * Self.EMB), MutAnyOrigin
        ](state.x_prev_buf)
        var x_prev_pe_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.x_prev_pe_buf)
        var x_prev_pe_bh_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, Self.CONFIG.H * Self.EMB), MutAnyOrigin
        ](state.x_prev_pe_buf)
        var pos_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, POS.CACHE_SIZE), MutAnyOrigin
        ](state.pos_cache_buf)
        var c_in_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.c_in_buf)

        var pred_raw_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.pred_raw_buf)
        var pred_raw_bh_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, Self.CONFIG.H * Self.EMB), MutAnyOrigin
        ](state.pred_raw_buf)
        var pred_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, Self.CONFIG.H * Self.EMB), MutAnyOrigin
        ](state.pred_out_buf)
        var proj_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, PROJ.CACHE_SIZE), MutAnyOrigin
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
            dtype, Layout.row_major(Self.CONFIG.BATCH, Self.CONFIG.H * Self.EMB), MutAnyOrigin
        ](state.grad_pred_buf)
        var grad_pred_raw_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.grad_pred_raw_buf)
        var grad_pred_raw_bh_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, Self.CONFIG.H * Self.EMB), MutAnyOrigin
        ](state.grad_pred_raw_buf)
        var grad_x_prev_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.grad_x_prev_buf)
        var grad_x_prev_bh_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, Self.CONFIG.H * Self.EMB), MutAnyOrigin
        ](state.grad_x_prev_buf)
        var grad_x_prev_pe_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.grad_x_prev_pe_buf)
        var grad_x_prev_pe_bh_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, Self.CONFIG.H * Self.EMB), MutAnyOrigin
        ](state.grad_x_prev_pe_buf)
        var grad_c_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.grad_c_buf)
        var grad_emb_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, Self.CONFIG.T * Self.EMB), MutAnyOrigin
        ](state.grad_emb_buf)
        var grad_emb_bt_t = LayoutTensor[
            dtype, Layout.row_major(BT, Self.EMB), MutAnyOrigin
        ](state.grad_emb_buf)
        var grad_act_emb_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, Self.CONFIG.T * Self.EMB), MutAnyOrigin
        ](state.grad_act_emb_buf)
        var grad_actions_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, Self.CONFIG.T * Self.CONFIG.ACT), MutAnyOrigin
        ](state.grad_actions_buf)
        var grad_pixels_t = LayoutTensor[
            dtype, Layout.row_major(BT, IMG_DIM), MutAnyOrigin
        ](state.grad_pixels_buf)

        # SIGReg views (treat emb / grad_emb as (Self.CONFIG.BATCH, Self.CONFIG.T*Self.EMB) — same memory).
        var emb_bte_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, Self.CONFIG.T * Self.EMB), MutAnyOrigin
        ](state.emb_buf)
        var sigreg_out_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, SIG.OUT_DIM), MutAnyOrigin
        ](state.sigreg_out_buf)
        var sigreg_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, SIG.CACHE_SIZE), MutAnyOrigin
        ](state.sigreg_cache_buf)
        var sigreg_grad_out_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, SIG.OUT_DIM), MutAnyOrigin
        ](state.sigreg_grad_out_buf)
        var sigreg_grad_emb_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, Self.CONFIG.T * Self.EMB), MutAnyOrigin
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
            Self.CONFIG.BATCH,
            Self.CONFIG.T,
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
                BT_LOCAL, Self.CONFIG.IN_CH, Self.CONFIG.IMG, Self.BUF.INPUT_LAYOUT_HWC,
            ],
        ](
            src_u8_t,
            dst_fp32_t,
            grid_dim=(
                ceildiv(BT_LOCAL, TPB_X),
                ceildiv(Self.CONFIG.IN_CH, TPB_Y),
                ceildiv(Self.CONFIG.IMG * Self.CONFIG.IMG, TPB_Z),
            ),
            block_dim=(TPB_X, TPB_Y, TPB_Z),
        )
        var ts_h2d_end = perf_counter_ns()

        # Zero grads on all 6 groups.
        state.enc_state.zero_grads(ctx)
        state.ae_state.zero_grads(ctx)
        state.pos_state.zero_grads(ctx)
        state.proj_state.zero_grads(ctx)
        for layer_idx in range(Self.CONFIG.DEPTH):
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
        AE.forward_gpu[Self.CONFIG.BATCH, dtype](
            ctx, act_emb_t, actions_t,
            state.ae_state.params_view(), state.ae_state.model_state_view(),
            ae_cache_t, state.ae_ws_buf,
        )

        # Slice first Self.CONFIG.H tokens of emb + act_emb into x_prev_buf + c_in_buf.
        var act_emb_bt_t = LayoutTensor[
            dtype, Layout.row_major(BT, Self.EMB), MutAnyOrigin
        ](state.act_emb_buf)
        ctx.enqueue_function[
            slice_h_kernel[Self.CONFIG.BATCH, Self.CONFIG.T, Self.CONFIG.H, Self.EMB],
        ](
            emb_t, x_prev_t,
            grid_dim=(
                ceildiv(Self.CONFIG.BATCH, TPB_X),
                ceildiv(Self.CONFIG.H, TPB_Y),
                ceildiv(Self.EMB, TPB_Z),
            ),
            block_dim=(TPB_X, TPB_Y, TPB_Z),
        )
        ctx.enqueue_function[
            slice_h_kernel[Self.CONFIG.BATCH, Self.CONFIG.T, Self.CONFIG.H, Self.EMB],
        ](
            act_emb_bt_t, c_in_t,
            grid_dim=(
                ceildiv(Self.CONFIG.BATCH, TPB_X),
                ceildiv(Self.CONFIG.H, TPB_Y),
                ceildiv(Self.EMB, TPB_Z),
            ),
            block_dim=(TPB_X, TPB_Y, TPB_Z),
        )

        # Pos embed: x_prev_pe = x_prev + pos_bias (broadcast over Self.CONFIG.BATCH).
        POS.forward_gpu[Self.CONFIG.BATCH, dtype](
            ctx, x_prev_pe_bh_t, x_prev_bh_t,
            state.pos_state.params_view(), state.pos_state.model_state_view(),
            pos_cache_t, state.pos_ws_buf,
        )

        # cond_block stack: Self.CONFIG.DEPTH dual-branch (MSA + MLP) layers via helper.
        for d in range(Self.CONFIG.DEPTH):
            run_cond_layer_forward[
                Self.CONFIG.BATCH,
                Self.CONFIG.H,
                Self.EMB,
                Self.CONFIG.PRED_HEADS,
                Self.CONFIG.PRED_DIM_HEAD,
                Self.CONFIG.PRED_FF,
            ](
                ctx, d, Self.CONFIG.DEPTH,
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

        # PredProj: (Self.CONFIG.BATCH, Self.CONFIG.H*Self.EMB) → (Self.CONFIG.BATCH, Self.CONFIG.H*Self.EMB).
        PROJ.forward_gpu[Self.CONFIG.BATCH, dtype](
            ctx, pred_t, pred_raw_bh_t,
            state.proj_state.params_view(), state.proj_state.model_state_view(),
            proj_cache_t, state.proj_ws_buf,
        )

        # SIGReg forward over emb viewed as (Self.CONFIG.BATCH, Self.CONFIG.T*Self.EMB). Output is the
        # statistic replicated across Self.CONFIG.BATCH slots (we read [0] for logging).
        SIG.eval_gpu[Self.CONFIG.BATCH, dtype](
            ctx, sigreg_out_t, emb_bte_t,
            empty_params, sigreg_cache_t, state.sigreg_ws_buf.unsafe_ptr(),
        )

        # --------------------------------------------------------------
        # Loss + grad_pred on host (small round-trip).
        # --------------------------------------------------------------
        ctx.enqueue_copy(state.pred_host, state.pred_out_buf)
        # Download all of emb (BT, Self.EMB) — used for both target slice and probes.
        ctx.enqueue_copy(state.emb_host, state.emb_buf)
        # Download SIGReg stat (tiny — Self.CONFIG.BATCH floats) for logging.
        ctx.enqueue_copy(state.sigreg_out_host, state.sigreg_out_buf)
        ctx.synchronize()

        var pred_loss: Float64 = 0.0
        for b in range(Self.CONFIG.BATCH):
            for i in range(Self.CONFIG.H * Self.EMB):
                var p = Float64(state.pred_host[b * Self.CONFIG.H * Self.EMB + i])
                # Target = emb[b, Self.CONFIG.N_PREDS .. Self.CONFIG.N_PREDS+Self.CONFIG.H, :], flat index:
                #   b * Self.CONFIG.T * Self.EMB + Self.CONFIG.N_PREDS * Self.EMB + i
                var tgt = Float64(
                    state.emb_host[b * Self.CONFIG.T * Self.EMB + Self.CONFIG.N_PREDS * Self.EMB + i]
                )
                var diff = p - tgt
                pred_loss += diff * diff
                state.grad_pred_host[b * Self.CONFIG.H * Self.EMB + i] = self.inv_scale * (
                    Scalar[dtype](p) - Scalar[dtype](tgt)
                )
        pred_loss /= self.loss_scale

        # Read SIGReg stat (replicated across Self.CONFIG.BATCH, take [0]).
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
        PROJ.backward_gpu[Self.CONFIG.BATCH, dtype](
            ctx, grad_pred_raw_bh_t, grad_pred_t,
            state.proj_state.params_view(), state.proj_state.model_state_view(),
            proj_cache_t, proj_g, state.proj_ws_buf,
        )

        # cond_block stack backward — reverse depth loop via helper.
        # grad_c is accumulated across layers (c is shared input).
        ctx.enqueue_memset(state.grad_c_buf, 0)
        for d_rev in range(Self.CONFIG.DEPTH):
            var d = Self.CONFIG.DEPTH - 1 - d_rev
            # Bind per-layer grad views to vars (mut args can't take temps).
            var adaln_g_d = state.adaln_states[d].grads_view()
            var msa_g_d = state.msa_states[d].grads_view()
            var mlp_g_d = state.mlp_states[d].grads_view()
            run_cond_layer_backward[
                Self.CONFIG.BATCH,
                Self.CONFIG.H,
                Self.EMB,
                Self.CONFIG.PRED_HEADS,
                Self.CONFIG.PRED_DIM_HEAD,
                Self.CONFIG.PRED_FF,
            ](
                ctx, d, Self.CONFIG.DEPTH,
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
        POS.backward_gpu[Self.CONFIG.BATCH, dtype](
            ctx, grad_x_prev_bh_t, grad_x_prev_pe_bh_t,
            state.pos_state.params_view(), state.pos_state.model_state_view(),
            pos_cache_t, pos_g, state.pos_ws_buf,
        )

        # Route grad_x_prev → grad_emb's first Self.CONFIG.H tokens, grad_c → grad_act_emb's.
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
            scatter_h_kernel[Self.CONFIG.BATCH, Self.CONFIG.T, Self.CONFIG.H, Self.EMB],
        ](
            grad_x_prev_t, grad_emb_bte_to_bt,
            grid_dim=(
                ceildiv(Self.CONFIG.BATCH, TPB_X),
                ceildiv(Self.CONFIG.H, TPB_Y),
                ceildiv(Self.EMB, TPB_Z),
            ),
            block_dim=(TPB_X, TPB_Y, TPB_Z),
        )
        ctx.enqueue_function[
            scatter_h_kernel[Self.CONFIG.BATCH, Self.CONFIG.T, Self.CONFIG.H, Self.EMB],
        ](
            grad_c_t, grad_act_emb_bte_to_bt,
            grid_dim=(
                ceildiv(Self.CONFIG.BATCH, TPB_X),
                ceildiv(Self.CONFIG.H, TPB_Y),
                ceildiv(Self.EMB, TPB_Z),
            ),
            block_dim=(TPB_X, TPB_Y, TPB_Z),
        )

        # Drop stop-grad: scatter -grad_pred into target slice of grad_emb.
        # Math: pred_loss = (pred - tgt)^2 / N → d/d tgt = -grad_pred.
        # Target tokens live at b * Self.CONFIG.T*Self.EMB + Self.CONFIG.N_PREDS*Self.EMB + [0..H*Self.EMB).
        comptime TPB_TS_X = 4
        comptime TPB_TS_Y = 64
        ctx.enqueue_function[
            scatter_target_neg_kernel[Self.CONFIG.BATCH, Self.CONFIG.T, Self.CONFIG.H, Self.CONFIG.N_PREDS, Self.EMB],
        ](
            grad_pred_t, grad_emb_t,
            grid_dim=(
                ceildiv(Self.CONFIG.BATCH, TPB_TS_X),
                ceildiv(Self.CONFIG.H * Self.EMB, TPB_TS_Y),
            ),
            block_dim=(TPB_TS_X, TPB_TS_Y),
        )

        # SIGReg vjp: produces sigreg_grad_emb (Self.CONFIG.BATCH, Self.CONFIG.T*Self.EMB) from
        # `sigreg_grad_out_t` seed = λ/B (set once at init).
        SIG.vjp_gpu[Self.CONFIG.BATCH, dtype](
            ctx, sigreg_grad_out_t, sigreg_grad_emb_t,
            empty_params, sigreg_cache_t, empty_grad_params,
            state.sigreg_ws_buf.unsafe_ptr(),
        )
        # Accumulate sigreg's grad into grad_emb additively.
        comptime TPB_AC_X = 4
        comptime TPB_AC_Y = 64
        ctx.enqueue_function[
            accumulate_emb_kernel[Self.CONFIG.BATCH, Self.CONFIG.T, Self.EMB],
        ](
            sigreg_grad_emb_t, grad_emb_t,
            grid_dim=(
                ceildiv(Self.CONFIG.BATCH, TPB_AC_X),
                ceildiv(Self.CONFIG.T * Self.EMB, TPB_AC_Y),
            ),
            block_dim=(TPB_AC_X, TPB_AC_Y),
        )

        # AE.backward
        AE.backward_gpu[Self.CONFIG.BATCH, dtype](
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

        # Optimizer step — shared models + per-layer (ADALN/MSA/MLP × Self.CONFIG.DEPTH).
        state.enc_state.optimizer_step(ctx)
        state.ae_state.optimizer_step(ctx)
        state.pos_state.optimizer_step(ctx)
        state.proj_state.optimizer_step(ctx)
        for layer_idx in range(Self.CONFIG.DEPTH):
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

    def _checkpoint_metadata(
        self, steps_completed: Int
    ) -> List[String]:
        """Build the metadata list saved with checkpoints.

        Records the comptime shape so the eval entry point can fail-fast
        on mismatch, plus a few useful diagnostics.
        """
        var meta = List[String]()
        meta.append("BATCH=" + String(Self.CONFIG.BATCH))
        meta.append("T=" + String(Self.CONFIG.T))
        meta.append("H=" + String(Self.CONFIG.H))
        meta.append("N_PREDS=" + String(Self.CONFIG.N_PREDS))
        meta.append("IN_CH=" + String(Self.CONFIG.IN_CH))
        meta.append("IMG=" + String(Self.CONFIG.IMG))
        meta.append("PATCH=" + String(Self.CONFIG.PATCH))
        meta.append("N_PATCHES=" + String(Self.CONFIG.N_PATCHES))
        meta.append("HIDDEN=" + String(Self.CONFIG.HIDDEN))
        meta.append("ENC_HEADS=" + String(Self.CONFIG.ENC_HEADS))
        meta.append("ENC_LAYERS=" + String(Self.CONFIG.ENC_LAYERS))
        meta.append("EMB=" + String(Self.EMB))
        meta.append("PROJ_H=" + String(Self.CONFIG.PROJ_H))
        meta.append("ACT=" + String(Self.CONFIG.ACT))
        meta.append("SMOOTHED=" + String(Self.CONFIG.SMOOTHED))
        meta.append("PRED_HEADS=" + String(Self.CONFIG.PRED_HEADS))
        meta.append("PRED_FF=" + String(Self.CONFIG.PRED_FF))
        meta.append("DEPTH=" + String(Self.CONFIG.DEPTH))
        meta.append("SIG_NUM_PROJ=" + String(Self.CONFIG.SIG_NUM_PROJ))
        meta.append("SIG_KNOTS=" + String(Self.CONFIG.SIG_KNOTS))
        meta.append("steps_completed=" + String(steps_completed))
        meta.append("loss_last=" + String(self.loss_last))
        meta.append("pred_ema=" + String(self.pred_ema))
        meta.append("var_min_ema=" + String(self.var_min_ema))
        return meta^

    def _make_eval_suite(self) -> LeWMEvalSuite[Self.CONFIG]:
        """Construct a `LeWMEvalSuite` forwarding the trainer's eval config."""
        return LeWMEvalSuite[Self.CONFIG](
            self.eval_steps, self.eval_samples, self.eval_seed,
            self.mpc_horizon, self.cem_iters, self.cem_samples,
            self.cem_topk, self.cem_smoothing,
            self.eval_shuffle_diag, self.eval_h7_closed_loop,
            self.rh_steps,
        )

    def run_eval(
        mut self,
        mut state: Self.GPUState,
        ctx: DeviceContext,
        rng_seed: Int,
    ) raises:
        """Run all eval phases without training (assumes weights are loaded).

        Used by the eval-only entry points (`eval_lewm_offline_gpu`,
        `eval_lewm_offline_gpu_pusht`). Delegates to `LeWMEvalSuite.run_all`.
        """
        _set_seed(rng_seed)
        var suite = self._make_eval_suite()
        suite.run_all(state, self.buf, ctx)

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
            " DEPTH=", Self.CONFIG.DEPTH,
        )
        var total_params = (
            Self.GPUState.ENC.PARAM_SIZE + Self.GPUState.AE.PARAM_SIZE
            + Self.GPUState.POS.PARAM_SIZE + Self.GPUState.PROJ.PARAM_SIZE
            + Self.CONFIG.DEPTH * (Self.GPUState.ADALN.PARAM_SIZE
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

        var suite = self._make_eval_suite()
        suite.run_all(state, self.buf, ctx)


def train_lewm_offline_gpu[CONFIG: LeWMConfig](
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
    rh_steps: Int = 0,
    var checkpoint_path: String = String(""),
    checkpoint_every: Int = 0,
    time_phases: Bool = False,
) raises:
    """LeWM offline GPU trainer entry point — Pong.

    Constructs `LeWMGPUState[CONFIG]` + `LeWMTrainer[CONFIG, PongOfflineBuffer]`
    and calls `trainer.run(...)`. `CONFIG` is a `LeWMConfig` (typically
    `LeWMPongViTConfig[...]`).

    `checkpoint_path` (non-empty) enables periodic + final checkpoint
    writes, consumed by `eval_lewm_offline_gpu`. `checkpoint_every`
    controls intermediate cadence (0 = final-only).
    """
    comptime assert CONFIG.DEPTH >= 1, "CONFIG.DEPTH must be >= 1"

    var ctx = DeviceContext()
    var state = LeWMGPUState[CONFIG](ctx, lambda_sigreg)
    var buf = PongOfflineBuffer.load(buffer_path)
    print("Loaded Pong buffer:", buf.n_frames, "frames from", buffer_path)
    var trainer = LeWMTrainer[CONFIG, PongOfflineBuffer](
        buf^, lambda_sigreg, log_every, eval_steps, eval_samples,
        eval_seed, mpc_horizon, cem_iters, cem_samples, cem_topk,
        cem_smoothing, eval_shuffle_diag, eval_h7_closed_loop,
        rh_steps, time_phases,
    )
    trainer.run(state, ctx, num_steps, rng_seed, checkpoint_path^, checkpoint_every)


def train_lewm_offline_gpu_pusht[
    CONFIG: LeWMConfig,
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
    rh_steps: Int = 0,
    var dataset_path: String = String(""),
    var checkpoint_path: String = String(""),
    checkpoint_every: Int = 0,
    time_phases: Bool = False,
) raises:
    """LeWM offline GPU trainer entry point — PushT (HF expert dataset).

    First run auto-downloads `quentinll/lewm-pusht` (~13 GB compressed,
    decompresses to ~15-25 GB at `~/.cache/mojo_rl/lewm_pusht/`); set
    `dataset_path` to point at an existing `.h5` to skip the download.

    Comptime invariants enforced via `CONFIG`:
      - `CONFIG.ACT == FRAMESKIP * ACTION_DIM` (paper's effective-action shape).
      - `CONFIG.IN_CH == 3` (PushT pixels are RGB, no frame stack).
    """
    comptime assert CONFIG.DEPTH >= 1, "CONFIG.DEPTH must be >= 1"
    comptime assert CONFIG.ACT == FRAMESKIP * ACTION_DIM, \
        "CONFIG.ACT must equal FRAMESKIP * ACTION_DIM"
    comptime assert CONFIG.IN_CH == 3, \
        "PushT pixels are RGB; CONFIG.IN_CH must be 3"

    var ctx = DeviceContext()
    var state = LeWMGPUState[CONFIG](ctx, lambda_sigreg)
    var sampler = PushTOfflineSampler(
        frameskip=FRAMESKIP, num_steps=CONFIG.T, path=dataset_path^,
    )
    if sampler.dataset.pixel_h != CONFIG.IMG or sampler.dataset.pixel_w != CONFIG.IMG:
        raise Error(
            "train_lewm_offline_gpu_pusht: dataset pixels are "
            + String(sampler.dataset.pixel_h)
            + "x"
            + String(sampler.dataset.pixel_w)
            + " but CONFIG.IMG="
            + String(CONFIG.IMG)
            + ". The `quentinll/lewm-pusht` H5 ships at 224x224; the LeWM"
            + " paper config uses IMG=224, PATCH=14 (256 patches)."
        )
    if sampler.dataset.action_dim != ACTION_DIM:
        raise Error(
            "train_lewm_offline_gpu_pusht: dataset action_dim="
            + String(sampler.dataset.action_dim)
            + " but ACTION_DIM="
            + String(ACTION_DIM)
        )
    var trainer = LeWMTrainer[CONFIG, PushTOfflineSampler](
        sampler^, lambda_sigreg, log_every, eval_steps, eval_samples,
        eval_seed, mpc_horizon, cem_iters, cem_samples, cem_topk,
        cem_smoothing, eval_shuffle_diag, eval_h7_closed_loop,
        rh_steps, time_phases,
    )
    trainer.run(state, ctx, num_steps, rng_seed, checkpoint_path^, checkpoint_every)


def eval_lewm_offline_gpu[CONFIG: LeWMConfig](
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
    rh_steps: Int = 0,
    lambda_sigreg: Float64 = 0.09,
) raises:
    """Load a Pong LeWM checkpoint and run only the eval phases.

    Symmetric with `train_lewm_offline_gpu` — `CONFIG` must match the
    binary that wrote the checkpoint. Reuses `PongOfflineBuffer` (loaded fresh
    from `buffer_path`) for the eval-time clip sampling.
    """
    comptime assert CONFIG.DEPTH >= 1, "CONFIG.DEPTH must be >= 1"

    var ctx = DeviceContext()
    var state = LeWMGPUState[CONFIG](ctx, lambda_sigreg)
    var meta = state.load_checkpoint(ctx, checkpoint_path)
    print("Loaded checkpoint from", checkpoint_path)
    for i in range(len(meta)):
        print("  meta:", meta[i])

    var buf = PongOfflineBuffer.load(buffer_path)
    print("Loaded Pong buffer:", buf.n_frames, "frames from", buffer_path)
    var trainer = LeWMTrainer[CONFIG, PongOfflineBuffer](
        buf^, lambda_sigreg, 0, eval_steps, eval_samples,
        eval_seed, mpc_horizon, cem_iters, cem_samples, cem_topk,
        cem_smoothing, eval_shuffle_diag, eval_h7_closed_loop,
        rh_steps,
    )
    trainer.run_eval(state, ctx, eval_seed)


def eval_lewm_offline_gpu_pusht[
    CONFIG: LeWMConfig,
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
    rh_steps: Int = 0,
    lambda_sigreg: Float64 = 0.09,
    var dataset_path: String = String(""),
) raises:
    """Load a PushT LeWM checkpoint and run only the eval phases."""
    comptime assert CONFIG.DEPTH >= 1, "CONFIG.DEPTH must be >= 1"
    comptime assert CONFIG.ACT == FRAMESKIP * ACTION_DIM, \
        "CONFIG.ACT must equal FRAMESKIP * ACTION_DIM"
    comptime assert CONFIG.IN_CH == 3, "PushT pixels are RGB; CONFIG.IN_CH must be 3"

    var ctx = DeviceContext()
    var state = LeWMGPUState[CONFIG](ctx, lambda_sigreg)
    var meta = state.load_checkpoint(ctx, checkpoint_path)
    print("Loaded checkpoint from", checkpoint_path)
    for i in range(len(meta)):
        print("  meta:", meta[i])

    var sampler = PushTOfflineSampler(
        frameskip=FRAMESKIP, num_steps=CONFIG.T, path=dataset_path^,
    )
    if sampler.dataset.pixel_h != CONFIG.IMG or sampler.dataset.pixel_w != CONFIG.IMG:
        raise Error(
            "eval_lewm_offline_gpu_pusht: dataset pixels are "
            + String(sampler.dataset.pixel_h)
            + "x"
            + String(sampler.dataset.pixel_w)
            + " but CONFIG.IMG="
            + String(CONFIG.IMG)
        )
    var trainer = LeWMTrainer[CONFIG, PushTOfflineSampler](
        sampler^, lambda_sigreg, 0, eval_steps, eval_samples,
        eval_seed, mpc_horizon, cem_iters, cem_samples, cem_topk,
        cem_smoothing, eval_shuffle_diag, eval_h7_closed_loop,
        rh_steps,
    )
    trainer.run_eval(state, ctx, eval_seed)
