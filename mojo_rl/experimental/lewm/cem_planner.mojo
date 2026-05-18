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
from layout import Layout, LayoutTensor, TileTensor, Idx, row_major

from ...nn.constants import dtype

from mojo_rl.planners.trajectory import (
    CategoricalCEMOptimizer,
    CategoricalRandomShooter,
)

from .offline_trainer import LeWMGPUState
from .lewm_buffer import LeWMBuffer
from .lewm_config import LeWMConfig
from .lewm_rollout_callback import LeWMRolloutScoreCallback
from .kernels import (
    _run_mpc_shot,
    pixels_uint8_to_fp32_kernel,
)


comptime TPB_X = 4
comptime TPB_Y = 4
comptime TPB_Z = 16


struct CEMPlanner[CONFIG: LeWMConfig](Movable, ImplicitlyDestructible):
    """Autoregressive MPC + CEM planner over a trained LeWM world model.

    Templated on the same `CONFIG: LeWMConfig` as `LeWMGPUState` /
    `LeWMTrainer` — the planner types its state argument as
    `Self.GPUState = LeWMGPUState[Self.CONFIG]`. CEM-specific
    hyperparams are construction args, not config fields.

    Buffer sizes:
      - emb_start_dev / emb_goal_dev: BATCH × EMB
      - emb_seq_dev:                  BATCH × (T+1) × EMB
      - action_plan_dev:              BATCH × T × ACT  (CEM samples
                                       overwrite this each shot)
      - sample_actions_host:          cem_samples × BATCH × needed × ACT
                                       (sized for 1 when cem_iters == 0)
    """

    comptime GPUState = LeWMGPUState[Self.CONFIG]
    # EMB is the encoder's OUT_DIM (see comment in LeWMGPUState). Alias
    # here so this planner's method bodies can use Self.EMB without
    # tripping Mojo's "different comptime expression" type mismatch.
    comptime EMB: Int = Self.GPUState.EMB

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
        var needed = Self.CONFIG.H + mpc_horizon - 1
        if needed > Self.CONFIG.T:
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
            Self.CONFIG.BATCH * Self.EMB
        )
        self.emb_goal_host_buf = alloc[Scalar[dtype]](
            Self.CONFIG.BATCH * Self.EMB
        )
        self.action_plan_host_buf = alloc[Scalar[dtype]](
            Self.CONFIG.BATCH * needed * Self.CONFIG.ACT
        )
        self.action_dist_host_buf = alloc[Scalar[dtype]](
            Self.CONFIG.BATCH * needed * Self.CONFIG.ACT
        )
        self.sample_actions_host_buf = alloc[Scalar[dtype]](
            cs * Self.CONFIG.BATCH * needed * Self.CONFIG.ACT
        )
        self.sample_scores_host_buf = alloc[Float64](cs)
        self.elite_indices_host_buf = alloc[Int](ck)

        # Device buffers.
        self.emb_start_dev_buf = ctx.enqueue_create_buffer[dtype](
            Self.CONFIG.BATCH * Self.EMB
        )
        self.emb_goal_dev_buf = ctx.enqueue_create_buffer[dtype](
            Self.CONFIG.BATCH * Self.EMB
        )
        self.emb_seq_dev_buf = ctx.enqueue_create_buffer[dtype](
            Self.CONFIG.BATCH * (Self.CONFIG.T + 1) * Self.EMB
        )
        self.action_plan_dev_buf = ctx.enqueue_create_buffer[dtype](
            Self.CONFIG.BATCH * Self.CONFIG.T * Self.CONFIG.ACT
        )
        self.score_dev_buf = ctx.enqueue_create_buffer[dtype](1)

        # Pinned host staging.
        self.score_host_buf = ctx.enqueue_create_host_buffer[dtype](1)
        self.emb_start_stage_host = ctx.enqueue_create_host_buffer[dtype](
            Self.CONFIG.BATCH * Self.EMB
        )
        self.emb_goal_stage_host = ctx.enqueue_create_host_buffer[dtype](
            Self.CONFIG.BATCH * Self.EMB
        )
        self.action_plan_stage_host = ctx.enqueue_create_host_buffer[dtype](
            Self.CONFIG.BATCH * Self.CONFIG.T * Self.CONFIG.ACT
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
            Self.CONFIG.BATCH, Self.CONFIG.T,
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
                BT, Self.CONFIG.IN_CH, Self.CONFIG.IMG, BUF.INPUT_LAYOUT_HWC,
            ],
        ](
            src_u8_t, dst_fp32_t,
            grid_dim=(
                ceildiv(BT, TPB_X),
                ceildiv(Self.CONFIG.IN_CH, TPB_Y),
                ceildiv(Self.CONFIG.IMG * Self.CONFIG.IMG, TPB_Z),
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
        var x_prev_pe_bh_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, Self.CONFIG.H * Self.EMB), MutAnyOrigin
        ](state.x_prev_pe_buf)
        var pos_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, POS.CACHE_SIZE), MutAnyOrigin
        ](state.pos_cache_buf)
        var c_in_t = LayoutTensor[
            dtype, Layout.row_major(BTH, Self.EMB), MutAnyOrigin
        ](state.c_in_buf)
        var pred_raw_bh_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, Self.CONFIG.H * Self.EMB), MutAnyOrigin
        ](state.pred_raw_buf)
        var pred_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, Self.CONFIG.H * Self.EMB), MutAnyOrigin
        ](state.pred_out_buf)
        var proj_cache_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, PROJ.CACHE_SIZE), MutAnyOrigin
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
            dtype, Layout.row_major(Self.CONFIG.BATCH, Self.EMB), MutAnyOrigin
        ](self.emb_start_dev_buf.unsafe_ptr())
        var emb_goal_dev_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, Self.EMB), MutAnyOrigin
        ](self.emb_goal_dev_buf.unsafe_ptr())
        var emb_seq_dev_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, (Self.CONFIG.T + 1) * Self.EMB), MutAnyOrigin
        ](self.emb_seq_dev_buf.unsafe_ptr())
        var action_plan_dev_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, Self.CONFIG.T * Self.CONFIG.ACT), MutAnyOrigin
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

        # Planning pipeline (lives across all eval iters):
        #   - `shooter` runs the random-shooting MPC baseline. Always
        #     constructed since the random-vs-expert diagnostic is
        #     part of every eval, regardless of cem_active.
        #   - `optimizer` runs the CEM refinement. When cem_active is
        #     False it's still constructed but `optimize` is never
        #     called, so the cost is just a small host-scratch
        #     allocation freed at eval end.
        #   - `callback` is shared across all three legs (expert,
        #     shooter, CEM) — owns the LeWM rollout scratch and
        #     wraps `_run_mpc_shot` behind `ScorePlanCallback`.
        var shooter = CategoricalRandomShooter[
            Self.CONFIG.BATCH, Self.CONFIG.ACT,
        ](horizon=needed_actions, num_samples=eval_samples)
        var optimizer = CategoricalCEMOptimizer[
            Self.CONFIG.BATCH, Self.CONFIG.ACT,
        ](
            horizon=needed_actions,
            cem_iters=self.cem_iters,
            cem_samples=self.cem_samples,
            cem_topk=self.cem_topk,
            cem_smoothing=self.cem_smoothing,
        )
        var callback = LeWMRolloutScoreCallback[Self.CONFIG](
            state, ctx, self.mpc_horizon, needed_actions,
        )

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
            for b in range(Self.CONFIG.BATCH):
                for d in range(Self.EMB):
                    self.emb_start_stage_host[b * Self.EMB + d] = (
                        state.emb_host[b * Self.CONFIG.T * Self.EMB + d]
                    )
                    self.emb_goal_stage_host[b * Self.EMB + d] = (
                        state.emb_host[b * Self.CONFIG.T * Self.EMB + (Self.CONFIG.T - 1) * Self.EMB + d]
                    )
            ctx.enqueue_copy(self.emb_start_dev_buf, self.emb_start_stage_host)
            ctx.enqueue_copy(self.emb_goal_dev_buf, self.emb_goal_stage_host)

            # Bridge the encoded start/goal embeddings into the callback's
            # buffers — they're consumed by every leg (expert, shooter,
            # CEM) below. Two tiny device-to-device copies per eval iter.
            ctx.enqueue_copy(
                callback.emb_start_dev_buf, self.emb_start_dev_buf
            )
            ctx.enqueue_copy(
                callback.emb_goal_dev_buf, self.emb_goal_dev_buf
            )

            # ---- Leg 1: Expert plan ----
            #
            # Copy the first `needed_actions` timesteps of the recorded
            # actions from `state.actions_host` into the planner's host
            # scratch (`action_plan_host_buf`), then wrap as a TileTensor
            # of shape (BATCH, needed_actions, ACT) and score via the
            # callback. The callback handles staging + device upload +
            # `_run_mpc_shot` internally.
            for b in range(Self.CONFIG.BATCH):
                for ti in range(needed_actions):
                    for k in range(Self.CONFIG.ACT):
                        self.action_plan_host_buf[
                            b * needed_actions * Self.CONFIG.ACT
                            + ti * Self.CONFIG.ACT + k
                        ] = state.actions_host[
                            b * Self.CONFIG.T * Self.CONFIG.ACT
                            + ti * Self.CONFIG.ACT + k
                        ]
            var expert_view = TileTensor(
                self.action_plan_host_buf,
                row_major(
                    (
                        Idx[Self.CONFIG.BATCH](),
                        Idx(needed_actions),
                        Idx[Self.CONFIG.ACT](),
                    )
                ),
            )
            var expert_loss_mpc = callback.score_plan(expert_view)

            # ---- Leg 2: Random shooter ----
            #
            # `shooter.optimize` draws `eval_samples` uniform-categorical
            # plans, scores each via the callback, returns the minimum.
            # The per-sample scores stay populated in
            # `shooter.sample_scores` so we can compute the legacy mean +
            # frac_random_worse_than_expert statistics afterwards without
            # re-running rollouts. `verbose=False` suppresses the
            # shooter's own log line — we emit the unified summary below.
            var random_min_mpc = shooter.optimize(
                callback, self.action_plan_host_buf, verbose=False
            )
            var random_mean_mpc: Float64 = 0.0
            var better_count_mpc: Int = 0
            for i in range(eval_samples):
                var sc = shooter.sample_scores[i]
                random_mean_mpc += sc
                if sc > expert_loss_mpc:
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

            # ---- Leg 3: CEM refinement (optional). ----
            #
            # Sample → score → top-K → refit, repeated `cem_iters` times.
            # Same callback as the other legs (no separate bridge needed
            # — the unified bridge above already filled
            # `callback.emb_start/goal_dev_buf`).
            if cem_active:
                print("  -- CEM eval iter", eval_iter, "--")
                var cem_score = optimizer.optimize(
                    callback,
                    self.action_plan_host_buf,
                    verbose=True,
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
