"""LeWM CEM planner — autoregressive MPC + CEM refinement over a trained world model.

Extracted from `offline_trainer.LeWMTrainer.eval_mpc_cem`. Owns MPC + CEM
scratch (host + device) persistently across `eval` calls; reuses trained
network parameters and the activation cache living on `LeWMGPUState`.

Single public method `eval[BUF: LeWMBuffer](state, buf, ctx, eval_steps,
eval_samples, eval_seed)` runs the full eval loop:
  - per eval_iter: sample → encode → 1 expert shot + N random shots → optional
    CEM refinement → accumulate stats
  - on exit: print MPC + CEM summary

The buffer is passed in (not owned) because the trainer owns the data
source; both training-time eval and standalone eval-from-checkpoint reuse
the same buffer instance.
"""

from std.math import ceildiv
from std.memory import alloc
from std.random import seed as _set_seed, random_float64
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor

from ...nn.constants import dtype

from .offline_trainer import LeWMGPUState
from .lewm_buffer import LeWMBuffer
from .kernels import (
    _run_mpc_shot, _run_cem_eval_iter,
    pixels_uint8_to_fp32_kernel,
)


comptime TPB_X = 4
comptime TPB_Y = 4
comptime TPB_Z = 16


struct CEMPlanner[
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
](Movable, ImplicitlyDestructible):
    """Autoregressive MPC + CEM planner over a trained LeWM world model.

    Comptime params mirror `LeWMGPUState` exactly so the planner can type
    its state argument as `Self.GPUState`. CEM-specific hyperparams are
    carved out of the trainer's eval config and become construction args.

    Buffer sizes:
      - emb_start_dev / emb_goal_dev: BATCH × EMB
      - emb_seq_dev:                  BATCH × (T+1) × EMB
      - action_plan_dev:              BATCH × T × ACT  (CEM samples
                                       overwrite this each shot)
      - sample_actions_host:          cem_samples × BATCH × needed × ACT
                                       (sized for 1 when cem_iters == 0)
    """

    comptime GPUState = LeWMGPUState[
        Self.BATCH, Self.T, Self.H, Self.N_PREDS,
        Self.IN_CH, Self.IMG, Self.PATCH, Self.N_PATCHES,
        Self.HIDDEN, Self.ENC_HEADS, Self.ENC_LAYERS,
        Self.EMB, Self.PROJ_H, Self.ACT, Self.SMOOTHED,
        Self.PRED_HEADS, Self.PRED_FF,
        Self.DEPTH, Self.SIG_NUM_PROJ, Self.SIG_KNOTS,
    ]

    # Hyperparams (carved out of trainer eval config).
    var mpc_horizon: Int
    var cem_iters: Int
    var cem_samples: Int
    var cem_topk: Int
    var cem_smoothing: Float64

    # Derived: H + mpc_horizon - 1, validated ≤ T in __init__.
    var needed_actions: Int

    # Host scratch (raw allocations; freed in __del__).
    var emb_start_host_buf: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var emb_goal_host_buf: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var action_plan_host_buf: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var action_dist_host_buf: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var sample_actions_host_buf: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var sample_scores_host_buf: UnsafePointer[Float64, MutAnyOrigin]
    var elite_indices_host_buf: UnsafePointer[Int, MutAnyOrigin]

    # Device-resident rollout state.
    var emb_start_dev_buf: DeviceBuffer[dtype]
    var emb_goal_dev_buf: DeviceBuffer[dtype]
    var emb_seq_dev_buf: DeviceBuffer[dtype]
    var action_plan_dev_buf: DeviceBuffer[dtype]
    var score_dev_buf: DeviceBuffer[dtype]

    # Pinned host staging (used by enqueue_copy).
    var score_host_buf: HostBuffer[dtype]
    var emb_start_stage_host: HostBuffer[dtype]
    var emb_goal_stage_host: HostBuffer[dtype]
    var action_plan_stage_host: HostBuffer[dtype]

    def __init__(
        out self,
        ctx: DeviceContext,
        mpc_horizon: Int,
        cem_iters: Int = 0,
        cem_samples: Int = 64,
        cem_topk: Int = 8,
        cem_smoothing: Float64 = 0.5,
    ) raises:
        if mpc_horizon < 1:
            raise Error("CEMPlanner requires mpc_horizon >= 1")
        var needed = Self.H + mpc_horizon - 1
        if needed > Self.T:
            raise Error(
                "mpc_horizon too large: H + mpc_horizon - 1 > T"
                " (need bigger T or smaller horizon)"
            )

        self.mpc_horizon = mpc_horizon
        self.cem_iters = cem_iters
        self.cem_samples = cem_samples
        self.cem_topk = cem_topk
        self.cem_smoothing = cem_smoothing
        self.needed_actions = needed

        var cem_active = cem_iters > 0
        var cs = cem_samples if cem_active else 1
        var ck = cem_topk if cem_active else 1

        # Host scratch.
        self.emb_start_host_buf = alloc[Scalar[dtype]](
            Self.BATCH * Self.EMB
        )
        self.emb_goal_host_buf = alloc[Scalar[dtype]](
            Self.BATCH * Self.EMB
        )
        self.action_plan_host_buf = alloc[Scalar[dtype]](
            Self.BATCH * needed * Self.ACT
        )
        self.action_dist_host_buf = alloc[Scalar[dtype]](
            Self.BATCH * needed * Self.ACT
        )
        self.sample_actions_host_buf = alloc[Scalar[dtype]](
            cs * Self.BATCH * needed * Self.ACT
        )
        self.sample_scores_host_buf = alloc[Float64](cs)
        self.elite_indices_host_buf = alloc[Int](ck)

        # Device buffers.
        self.emb_start_dev_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.EMB
        )
        self.emb_goal_dev_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.EMB
        )
        self.emb_seq_dev_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * (Self.T + 1) * Self.EMB
        )
        self.action_plan_dev_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.T * Self.ACT
        )
        self.score_dev_buf = ctx.enqueue_create_buffer[dtype](1)

        # Pinned host staging.
        self.score_host_buf = ctx.enqueue_create_host_buffer[dtype](1)
        self.emb_start_stage_host = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH * Self.EMB
        )
        self.emb_goal_stage_host = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH * Self.EMB
        )
        self.action_plan_stage_host = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH * Self.T * Self.ACT
        )

    def __del__(deinit self):
        self.emb_start_host_buf.free()
        self.emb_goal_host_buf.free()
        self.action_plan_host_buf.free()
        self.action_dist_host_buf.free()
        self.sample_actions_host_buf.free()
        self.sample_scores_host_buf.free()
        self.elite_indices_host_buf.free()

    def _sample_and_upload_pixels[BUF: LeWMBuffer](
        mut self,
        mut state: Self.GPUState,
        mut buf: BUF,
        ctx: DeviceContext,
    ) raises:
        """Sample uint8 pixels + actions from `buf` into pinned host
        buffers, upload to device, run the GPU conversion kernel.

        Mirror of `LeWMTrainer._sample_and_upload_pixels` (the trainer
        owns its own buffer reference; the planner accepts one via arg).
        Actions land in `actions_host` only — caller decides whether to
        upload, since MPC overwrites `actions_buf` per shot.
        """
        buf.sample_batch_uint8(
            Self.BATCH, Self.T,
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
                BT, Self.IN_CH, Self.IMG, BUF.INPUT_LAYOUT_HWC,
            ],
        ](
            src_u8_t, dst_fp32_t,
            grid_dim=(
                ceildiv(BT, TPB_X),
                ceildiv(Self.IN_CH, TPB_Y),
                ceildiv(Self.IMG * Self.IMG, TPB_Z),
            ),
            block_dim=(TPB_X, TPB_Y, TPB_Z),
        )

    def eval[BUF: LeWMBuffer](
        mut self,
        mut state: Self.GPUState,
        mut buf: BUF,
        ctx: DeviceContext,
        eval_steps: Int,
        eval_samples: Int,
        eval_seed: Int,
    ) raises:
        """Run `eval_steps` iterations of MPC eval (1 expert + N random
        shots per iter), with optional CEM refinement when
        `cem_iters > 0`. Prints per-iter results + final summary.

        Per iter:
          1. Sample a fresh batch via `buf` → encode all T frames.
          2. Extract frame 0 (start) and frame T-1 (goal) per batch row.
          3. For each shot s ∈ [0, eval_samples]:
                 - s=0: expert action plan (sliced from actions_host)
                 - s>0: random one-hot action plan
                 - Roll out for mpc_horizon steps; score = MSE(final, goal)
          4. If cem_iters > 0: run CEM refinement
             (cem_iters × cem_samples → top cem_topk → tighten Gaussian).
        """
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

        var needed_actions = self.needed_actions

        print()
        print(
            "==== Phase 4b eval: autoregressive MPC (horizon=",
            self.mpc_horizon, ") ===="
        )
        _set_seed(eval_seed)

        # Views over persistent device buffers.
        var emb_start_dev_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.EMB), MutAnyOrigin
        ](self.emb_start_dev_buf.unsafe_ptr())
        var emb_goal_dev_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.EMB), MutAnyOrigin
        ](self.emb_goal_dev_buf.unsafe_ptr())
        var emb_seq_dev_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, (Self.T + 1) * Self.EMB), MutAnyOrigin
        ](self.emb_seq_dev_buf.unsafe_ptr())
        var action_plan_dev_t = LayoutTensor[
            dtype, Layout.row_major(Self.BATCH, Self.T * Self.ACT), MutAnyOrigin
        ](self.action_plan_dev_buf.unsafe_ptr())
        var score_dev_t = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin
        ](self.score_dev_buf.unsafe_ptr())

        var sum_expert_mpc: Float64 = 0.0
        var sum_random_mean_mpc: Float64 = 0.0
        var sum_random_min_mpc: Float64 = 0.0
        var sum_better_frac_mpc: Float64 = 0.0
        var sum_cem: Float64 = 0.0
        var cem_better_expert: Int = 0
        var cem_better_random_min: Int = 0
        var cem_active = self.cem_iters > 0

        for eval_iter in range(eval_steps):
            self._sample_and_upload_pixels(state, buf, ctx)

            ENC.forward_gpu[BT, dtype](
                ctx, emb_t, pixels_t,
                state.enc_state.params_view(), state.enc_state.model_state_view(),
                enc_cache_t, state.enc_ws_buf,
            )
            ctx.enqueue_copy(state.emb_host, state.emb_buf)
            ctx.synchronize()

            # Extract start (frame 0) + goal (frame T-1) per batch row,
            # upload both to device.
            for b in range(Self.BATCH):
                for d in range(Self.EMB):
                    self.emb_start_stage_host[b * Self.EMB + d] = (
                        state.emb_host[b * Self.T * Self.EMB + d]
                    )
                    self.emb_goal_stage_host[b * Self.EMB + d] = (
                        state.emb_host[b * Self.T * Self.EMB + (Self.T - 1) * Self.EMB + d]
                    )
            ctx.enqueue_copy(self.emb_start_dev_buf, self.emb_start_stage_host)
            ctx.enqueue_copy(self.emb_goal_dev_buf, self.emb_goal_stage_host)

            var expert_loss_mpc: Float64 = 0.0
            var random_mean_mpc: Float64 = 0.0
            var random_min_mpc: Float64 = 1e30
            var better_count_mpc: Int = 0

            for s in range(1 + eval_samples):
                # Build action plan (Self.BATCH, needed_actions, Self.ACT) on host.
                if s == 0:
                    for b in range(Self.BATCH):
                        for ti in range(needed_actions):
                            for k in range(Self.ACT):
                                self.action_plan_host_buf[
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
                                self.action_plan_host_buf[
                                    b * needed_actions * Self.ACT
                                    + ti * Self.ACT + k
                                ] = (
                                    Scalar[dtype](1.0)
                                    if k == r_act
                                    else Scalar[dtype](0.0)
                                )

                # Stage action_plan to (Self.BATCH, Self.T, Self.ACT) layout.
                for b in range(Self.BATCH):
                    for ti in range(needed_actions):
                        for k in range(Self.ACT):
                            self.action_plan_stage_host[
                                b * Self.T * Self.ACT + ti * Self.ACT + k
                            ] = self.action_plan_host_buf[
                                b * needed_actions * Self.ACT + ti * Self.ACT + k
                            ]
                    for t_pad in range(Self.T - needed_actions):
                        for k in range(Self.ACT):
                            self.action_plan_stage_host[
                                b * Self.T * Self.ACT
                                + (needed_actions + t_pad) * Self.ACT + k
                            ] = Scalar[dtype](0.0)
                ctx.enqueue_copy(
                    self.action_plan_dev_buf, self.action_plan_stage_host
                )

                var l = _run_mpc_shot[
                    Self.BATCH, Self.T, Self.H, Self.EMB, Self.ACT, Self.SMOOTHED, Self.PROJ_H,
                    Self.PRED_HEADS, Self.PRED_FF, Self.DEPTH,
                ](
                    ctx,
                    self.mpc_horizon, needed_actions,
                    emb_start_dev_t, emb_goal_dev_t,
                    emb_seq_dev_t, action_plan_dev_t,
                    score_dev_t, self.score_dev_buf, self.score_host_buf,
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
                    Self.BATCH, Self.T, Self.H, Self.EMB, Self.ACT,
                    Self.SMOOTHED, Self.PROJ_H,
                    Self.PRED_HEADS, Self.PRED_FF, Self.DEPTH,
                ](
                    ctx,
                    self.mpc_horizon, needed_actions,
                    self.cem_iters, self.cem_samples, self.cem_topk, self.cem_smoothing,
                    self.action_dist_host_buf, self.action_plan_host_buf,
                    self.sample_actions_host_buf, self.sample_scores_host_buf,
                    self.elite_indices_host_buf,
                    emb_start_dev_t, emb_goal_dev_t,
                    emb_seq_dev_t, action_plan_dev_t,
                    self.action_plan_dev_buf,
                    score_dev_t, self.score_dev_buf, self.score_host_buf,
                    self.action_plan_stage_host,
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

        var avg_expert_mpc = sum_expert_mpc / Float64(eval_steps)
        var avg_rand_mean_mpc = (
            sum_random_mean_mpc / Float64(eval_steps)
        )
        var avg_rand_min_mpc = sum_random_min_mpc / Float64(eval_steps)
        var avg_better_mpc = sum_better_frac_mpc / Float64(eval_steps)
        print()
        print("Phase 4b MPC eval summary (",
            eval_steps, "iters x ", eval_samples, "shots, horizon=",
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
            var avg_cem = sum_cem / Float64(eval_steps)
            var cem_vs_expert_frac = (
                Float64(cem_better_expert) / Float64(eval_steps)
            )
            var cem_vs_rmin_frac = (
                Float64(cem_better_random_min) / Float64(eval_steps)
            )
            print()
            print("Phase 4c CEM eval summary (",
                eval_steps, "iters x ", self.cem_iters, "CEM iters x ",
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
