"""LeWM CEM planner — autoregressive MPC + CEM refinement over a trained world model.

Extracted from `offline_trainer.LeWMTrainer.eval_mpc_cem`. Owns MPC + CEM
scratch (host + device) persistently across `eval` calls; reuses trained
network parameters and the activation cache living on `LeWMGPUState`.

Single public method `eval[BUF: OfflineBuffer](state, buf, ctx, eval_steps,
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
from layout import Layout, LayoutTensor, TileTensor, TensorLayout, Idx, row_major

from ...nn.constants import dtype

from mojo_rl.planners.trajectory import (
    CategoricalCEMOptimizer,
    CategoricalRandomShooter,
)

from mojo_rl.core.offline_buffer import OfflineBuffer

from .offline_trainer import LeWMGPUState
from .lewm_config import LeWMConfig
from .lewm_rollout_callback import LeWMRolloutScoreCallback
from .kernels import (
    _run_mpc_shot,
    pixels_uint8_to_fp32_kernel,
    extract_emb_from_seq_kernel,
    mpc_score_kernel,
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

    # Host scratch shared by the expert leg, random shooter, and CEM:
    # the optimizers write the best plan back into this buffer.
    var action_plan_host_buf: UnsafePointer[Scalar[dtype], MutAnyOrigin]

    # Device-resident bridge for the per-iter start/goal latents. We
    # upload encoded start/goal embeddings here then enqueue_copy into
    # the rollout callback's twin buffers (the callback owns all the
    # other per-shot scratch — emb_seq, action_plan_dev, score).
    var emb_start_dev_buf: DeviceBuffer[dtype]
    var emb_goal_dev_buf: DeviceBuffer[dtype]

    # Pinned host staging for the upload above.
    var emb_start_stage_host: HostBuffer[dtype]
    var emb_goal_stage_host: HostBuffer[dtype]

    # Receding-horizon MPC scratch. ``current_emb_dev_buf`` holds the
    # rolling latent state across RH execution steps; ``final_score_*``
    # are the (1,) device+host slots for the end-of-trajectory MSE.
    # Allocated unconditionally (a few KB total) so ``eval_receding_horizon``
    # can fire without touching planner construction sites.
    var current_emb_dev_buf: DeviceBuffer[dtype]
    var final_score_dev_buf: DeviceBuffer[dtype]
    var final_score_host_buf: HostBuffer[dtype]

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

        self.action_plan_host_buf = alloc[Scalar[dtype]](
            Self.CONFIG.BATCH * needed * Self.CONFIG.ACT
        )
        self.emb_start_dev_buf = ctx.enqueue_create_buffer[dtype](
            Self.CONFIG.BATCH * Self.EMB
        )
        self.emb_goal_dev_buf = ctx.enqueue_create_buffer[dtype](
            Self.CONFIG.BATCH * Self.EMB
        )
        self.emb_start_stage_host = ctx.enqueue_create_host_buffer[dtype](
            Self.CONFIG.BATCH * Self.EMB
        )
        self.emb_goal_stage_host = ctx.enqueue_create_host_buffer[dtype](
            Self.CONFIG.BATCH * Self.EMB
        )
        self.current_emb_dev_buf = ctx.enqueue_create_buffer[dtype](
            Self.CONFIG.BATCH * Self.EMB
        )
        self.final_score_dev_buf = ctx.enqueue_create_buffer[dtype](1)
        self.final_score_host_buf = ctx.enqueue_create_host_buffer[dtype](1)

    def __del__(deinit self):
        self.action_plan_host_buf.free()

    def _sample_and_upload_pixels[BUF: OfflineBuffer](
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

    def eval[BUF: OfflineBuffer](
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
        # k_max sizes the callback's K-slot scores buffer so the same
        # callback serves both the shooter (K=eval_samples) and CEM
        # iterations (K=cem_samples). Taking the max avoids ever
        # reallocating mid-eval.
        var callback_k_max = (
            self.cem_samples
            if cem_active and self.cem_samples > eval_samples
            else eval_samples
        )
        if callback_k_max < 1:
            callback_k_max = 1
        var callback = LeWMRolloutScoreCallback[Self.CONFIG](
            state, ctx, self.mpc_horizon, needed_actions,
            k_max=callback_k_max,
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
            # `shooter.optimize_batched` draws `eval_samples`
            # uniform-categorical plans, scores them in ONE batched GPU
            # call (one host sync regardless of K), returns the minimum.
            # The per-sample scores stay populated in
            # `shooter.sample_scores` so we can compute the legacy mean +
            # frac_random_worse_than_expert statistics afterwards without
            # re-running rollouts. `verbose=False` suppresses the
            # shooter's own log line — we emit the unified summary below.
            var random_min_mpc = shooter.optimize_batched(
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
            # Sample → batched score → top-K → refit, repeated
            # `cem_iters` times. ``optimize_batched`` scores all
            # ``cem_samples`` plans per iter in ONE host sync. Same
            # callback as the other legs (no separate bridge needed —
            # the unified bridge above already filled
            # ``callback.emb_start/goal_dev_buf``).
            if cem_active:
                print("  -- CEM eval iter", eval_iter, "--")
                var cem_score = optimizer.optimize_batched(
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

    def _rh_advance_one_step[L: TensorLayout](
        mut self,
        ctx: DeviceContext,
        mut callback: LeWMRolloutScoreCallback[Self.CONFIG],
        plan: TileTensor[dtype, L, MutAnyOrigin],
    ) raises -> Float64:
        """Score ``plan`` (one full ``mpc_horizon`` rollout, repopulating
        ``callback.emb_seq_dev_buf``), then extract
        ``emb_seq[:, H*EMB:(H+1)*EMB]`` (the 1-step prediction slot) into
        ``self.current_emb_dev_buf``. Returns the rollout's score —
        receding-horizon callers only use it for per-step diagnostics; the
        end-of-trajectory score is computed separately by
        ``eval_receding_horizon`` after the RH loop completes.

        Only the first action of ``plan`` actually matters for the rolling
        state (we extract slot ``H``); the remaining ``mpc_horizon - 1``
        rollout steps are wasted compute. Acceptable for v1 — re-using the
        existing single-plan rollout path is far simpler than carving a
        dedicated 1-step helper, and the overhead is fixed at
        ``mpc_horizon - 1`` wasted rollouts per RH execution step.
        """
        var score = callback.score_plan(plan)
        var emb_seq_t = LayoutTensor[
            dtype,
            Layout.row_major(
                Self.CONFIG.BATCH, (Self.CONFIG.T + 1) * Self.EMB,
            ),
            MutAnyOrigin,
        ](callback.emb_seq_dev_buf.unsafe_ptr())
        var current_emb_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, Self.EMB),
            MutAnyOrigin,
        ](self.current_emb_dev_buf.unsafe_ptr())
        ctx.enqueue_function[
            extract_emb_from_seq_kernel[
                Self.CONFIG.BATCH, Self.EMB, Self.CONFIG.T + 1,
            ],
        ](
            emb_seq_t, current_emb_t, Self.CONFIG.H,
            grid_dim=(
                ceildiv(Self.CONFIG.BATCH, 16),
                ceildiv(Self.EMB, 16),
            ),
            block_dim=(16, 16),
        )
        return score

    def _rh_final_score(
        mut self,
        ctx: DeviceContext,
        mut callback: LeWMRolloutScoreCallback[Self.CONFIG],
    ) raises -> Float64:
        """Compute MSE(``self.current_emb_dev_buf``,
        ``callback.emb_goal_dev_buf``) summed across BATCH × EMB and
        normalized — i.e. the end-of-trajectory match-to-goal score.

        Reuses ``mpc_score_kernel`` with ``ROLL_T=1`` and ``goal_pos=0``:
        the kernel treats ``current_emb`` as a degenerate ``(BATCH, 1, EMB)``
        sequence and just computes the per-batch MSE at position 0.
        """
        var current_emb_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, Self.EMB),
            MutAnyOrigin,
        ](self.current_emb_dev_buf.unsafe_ptr())
        var emb_goal_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, Self.EMB),
            MutAnyOrigin,
        ](callback.emb_goal_dev_buf.unsafe_ptr())
        var score_t = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin,
        ](self.final_score_dev_buf.unsafe_ptr())
        ctx.enqueue_function[
            mpc_score_kernel[Self.CONFIG.BATCH, Self.EMB, 1],
        ](
            current_emb_t, emb_goal_t, score_t, 0,
            grid_dim=1, block_dim=32,
        )
        ctx.enqueue_copy(self.final_score_host_buf, self.final_score_dev_buf)
        ctx.synchronize()
        return (
            Float64(self.final_score_host_buf[0])
            / Float64(Self.CONFIG.BATCH * Self.EMB)
        )

    def _build_expert_rh_plan(
        mut self,
        mut state: Self.GPUState,
        rh_step: Int,
        needed_actions: Int,
    ):
        """Stage the expert action plan for receding-horizon step ``rh_step``
        into ``self.action_plan_host_buf``: actions
        ``[rh_step .. rh_step + needed_actions)`` from
        ``state.actions_host``, zero-padded past the recorded T.

        Note: only the FIRST action of this plan actually moves the rolling
        state forward (``_rh_advance_one_step`` extracts slot H = first
        prediction). The remaining actions are present so the rollout
        kernel signature stays uniform; their predictions are computed but
        discarded. Padding past ``T`` therefore has no effect on the RH
        trajectory — it only shows up in the diagnostic score returned by
        ``_rh_advance_one_step``.
        """
        for b in range(Self.CONFIG.BATCH):
            for ti in range(needed_actions):
                var src_t = rh_step + ti
                for k in range(Self.CONFIG.ACT):
                    var dst_idx = (
                        b * needed_actions * Self.CONFIG.ACT
                        + ti * Self.CONFIG.ACT + k
                    )
                    if src_t < Self.CONFIG.T:
                        self.action_plan_host_buf[dst_idx] = (
                            state.actions_host[
                                b * Self.CONFIG.T * Self.CONFIG.ACT
                                + src_t * Self.CONFIG.ACT + k
                            ]
                        )
                    else:
                        self.action_plan_host_buf[dst_idx] = (
                            Scalar[dtype](0.0)
                        )

    def eval_receding_horizon[BUF: OfflineBuffer](
        mut self,
        mut state: Self.GPUState,
        mut buf: BUF,
        ctx: DeviceContext,
        eval_steps: Int,
        eval_samples: Int,
        rh_steps: Int,
        eval_seed: Int,
    ) raises:
        """Receding-horizon MPC eval — three RH passes per iter (expert /
        random / CEM if ``cem_iters > 0``).

        Per iter:
          1. Sample + encode T frames (same as ``eval``).
          2. Extract start (frame 0) + goal (frame T-1) per batch row,
             upload to ``callback.emb_goal_dev_buf`` and a local rolling
             ``self.current_emb_dev_buf``.
          3. For each RH pass:
             - Re-init ``current_emb := emb_start``.
             - For step in range(rh_steps):
                 a. ``callback.emb_start := current_emb``.
                 b. Pick plan (expert recorded / shooter / CEM optimizer).
                 c. Re-roll best plan via ``_rh_advance_one_step``, which
                    extracts ``emb_seq[:, H, :]`` into ``current_emb``.
             - Score the final ``current_emb`` against ``emb_goal``.
          4. Accumulate per-iter stats.

        Comparison hypothesis (per
        ``project_lewm_horizon_sweep.md``): receding-horizon CEM beats
        open-loop CEM because short-horizon plans keep optimizer competence
        high while replanning recovers the model-informativeness gain that
        long-horizon plans pick up at training scale.

        Args:
            eval_steps: Number of distinct batches to evaluate.
            eval_samples: K (candidate plans per RH step) for the random
                shooter and the CEM optimizer's inner sample count
                (overrides ``self.cem_samples`` for the RH pass).
            rh_steps: Number of receding-horizon execution steps per iter.
                Each step plans ``mpc_horizon`` ahead, executes 1 in the
                latent (via ``_rh_advance_one_step``), then replans.
            eval_seed: RNG seed for sampling + random plans.
        """
        comptime IMG_DIM = Self.GPUState.IMG_DIM
        comptime BT = Self.GPUState.BT
        comptime BTH = Self.GPUState.BTH
        comptime ENC = Self.GPUState.ENC
        comptime AE = Self.GPUState.AE
        comptime POS = Self.GPUState.POS
        comptime PROJ = Self.GPUState.PROJ

        if rh_steps < 1:
            return

        var pixels_t = LayoutTensor[
            dtype, Layout.row_major(BT, IMG_DIM), MutAnyOrigin
        ](state.pixels_buf)
        var emb_t = LayoutTensor[
            dtype, Layout.row_major(BT, Self.EMB), MutAnyOrigin
        ](state.emb_buf)
        var enc_cache_t = LayoutTensor[
            dtype, Layout.row_major(BT, ENC.CACHE_SIZE), MutAnyOrigin
        ](state.enc_cache_buf)

        var needed_actions = self.needed_actions
        var cem_active = self.cem_iters > 0

        print()
        print(
            "==== Phase 4d eval: receding-horizon MPC (rh_steps=",
            rh_steps, ", mpc_horizon=", self.mpc_horizon, ") ===="
        )
        _set_seed(eval_seed)

        var sum_rh_expert: Float64 = 0.0
        var sum_rh_random: Float64 = 0.0
        var sum_rh_cem: Float64 = 0.0

        # Optimizers + callback live for the full eval — same as open-loop
        # path. ``shooter`` always constructed; ``optimizer.optimize_batched``
        # only called when ``cem_active``.
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
        var callback_k_max = (
            self.cem_samples
            if cem_active and self.cem_samples > eval_samples
            else eval_samples
        )
        if callback_k_max < 1:
            callback_k_max = 1
        var callback = LeWMRolloutScoreCallback[Self.CONFIG](
            state, ctx, self.mpc_horizon, needed_actions,
            k_max=callback_k_max,
        )

        for eval_iter in range(eval_steps):
            self._sample_and_upload_pixels(state, buf, ctx)

            ENC.forward_gpu[BT, dtype](
                ctx, emb_t, pixels_t,
                state.enc_state.params_view(),
                state.enc_state.model_state_view(),
                enc_cache_t, state.enc_ws_buf,
            )
            ctx.enqueue_copy(state.emb_host, state.emb_buf)
            ctx.synchronize()

            for b in range(Self.CONFIG.BATCH):
                for d in range(Self.EMB):
                    self.emb_start_stage_host[b * Self.EMB + d] = (
                        state.emb_host[b * Self.CONFIG.T * Self.EMB + d]
                    )
                    self.emb_goal_stage_host[b * Self.EMB + d] = (
                        state.emb_host[
                            b * Self.CONFIG.T * Self.EMB
                            + (Self.CONFIG.T - 1) * Self.EMB + d
                        ]
                    )
            ctx.enqueue_copy(self.emb_start_dev_buf, self.emb_start_stage_host)
            ctx.enqueue_copy(self.emb_goal_dev_buf, self.emb_goal_stage_host)
            ctx.enqueue_copy(
                callback.emb_goal_dev_buf, self.emb_goal_dev_buf
            )

            # ---- Pass 1: Expert receding-horizon ----
            ctx.enqueue_copy(
                self.current_emb_dev_buf, self.emb_start_dev_buf
            )
            for rh_step in range(rh_steps):
                ctx.enqueue_copy(
                    callback.emb_start_dev_buf, self.current_emb_dev_buf
                )
                self._build_expert_rh_plan(state, rh_step, needed_actions)
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
                _ = self._rh_advance_one_step(ctx, callback, expert_view)
            var rh_expert_score = self._rh_final_score(ctx, callback)

            # ---- Pass 2: Random receding-horizon ----
            ctx.enqueue_copy(
                self.current_emb_dev_buf, self.emb_start_dev_buf
            )
            for rh_step in range(rh_steps):
                _ = rh_step
                ctx.enqueue_copy(
                    callback.emb_start_dev_buf, self.current_emb_dev_buf
                )
                # Shooter writes the best plan back to action_plan_host_buf
                # and returns its score (which we discard; the relevant
                # number is the end-of-trajectory MSE, computed below).
                _ = shooter.optimize_batched(
                    callback, self.action_plan_host_buf, verbose=False
                )
                var random_best_view = TileTensor(
                    self.action_plan_host_buf,
                    row_major(
                        (
                            Idx[Self.CONFIG.BATCH](),
                            Idx(needed_actions),
                            Idx[Self.CONFIG.ACT](),
                        )
                    ),
                )
                _ = self._rh_advance_one_step(
                    ctx, callback, random_best_view
                )
            var rh_random_score = self._rh_final_score(ctx, callback)

            # ---- Pass 3: CEM receding-horizon (optional). ----
            var rh_cem_score: Float64 = 0.0
            if cem_active:
                ctx.enqueue_copy(
                    self.current_emb_dev_buf, self.emb_start_dev_buf
                )
                for rh_step in range(rh_steps):
                    _ = rh_step
                    ctx.enqueue_copy(
                        callback.emb_start_dev_buf,
                        self.current_emb_dev_buf,
                    )
                    _ = optimizer.optimize_batched(
                        callback,
                        self.action_plan_host_buf,
                        verbose=False,
                    )
                    var cem_best_view = TileTensor(
                        self.action_plan_host_buf,
                        row_major(
                            (
                                Idx[Self.CONFIG.BATCH](),
                                Idx(needed_actions),
                                Idx[Self.CONFIG.ACT](),
                            )
                        ),
                    )
                    _ = self._rh_advance_one_step(
                        ctx, callback, cem_best_view
                    )
                rh_cem_score = self._rh_final_score(ctx, callback)

            sum_rh_expert += rh_expert_score
            sum_rh_random += rh_random_score
            if cem_active:
                sum_rh_cem += rh_cem_score

            if cem_active:
                print(
                    "  rh eval", eval_iter,
                    " expert_rh=", rh_expert_score,
                    " random_rh=", rh_random_score,
                    " cem_rh=", rh_cem_score,
                    " cem_rh/expert_rh=",
                    rh_cem_score / (rh_expert_score + 1e-12),
                    " cem_rh/random_rh=",
                    rh_cem_score / (rh_random_score + 1e-12),
                )
            else:
                print(
                    "  rh eval", eval_iter,
                    " expert_rh=", rh_expert_score,
                    " random_rh=", rh_random_score,
                    " expert_rh/random_rh=",
                    rh_expert_score / (rh_random_score + 1e-12),
                )

        var avg_rh_expert = sum_rh_expert / Float64(eval_steps)
        var avg_rh_random = sum_rh_random / Float64(eval_steps)
        print()
        print(
            "Phase 4d receding-horizon eval summary (",
            eval_steps, "iters x ", rh_steps, "RH steps x ",
            eval_samples, "samples, mpc_horizon=", self.mpc_horizon, "):"
        )
        print("  expert_rh MSE      =", avg_rh_expert)
        print("  random_rh MSE      =", avg_rh_random)
        print(
            "  expert_rh/random_rh=",
            avg_rh_expert / (avg_rh_random + 1e-12),
            " (want < 1.0 — model + expert beats model + random)",
        )
        if cem_active:
            var avg_rh_cem = sum_rh_cem / Float64(eval_steps)
            print("  cem_rh MSE         =", avg_rh_cem)
            print(
                "  cem_rh/expert_rh   =",
                avg_rh_cem / (avg_rh_expert + 1e-12),
                " (want < 1.0 — CEM-replan beats expert-replan)",
            )
            print(
                "  cem_rh/random_rh   =",
                avg_rh_cem / (avg_rh_random + 1e-12),
                " (want < 1.0 — CEM-replan beats random-replan)",
            )
