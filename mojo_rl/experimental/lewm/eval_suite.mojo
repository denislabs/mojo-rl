"""LeWM eval suite — bundles the post-training diagnostic phases.

Extracted from `offline_trainer.LeWMTrainer`. Three method-level phases
live here (each owns no GPU state — they reuse `LeWMGPUState` buffers):

  - `eval_h6`                  shuffled-action diagnostic (action-aware?)
  - `eval_random_shots`        random-action baseline (sanity check;
                                only used when mpc_horizon == 0)
  - `eval_h7_closed_loop_drift` teacher-forced vs closed-loop rollout drift

For autoregressive MPC + CEM refinement we delegate to
`cem_planner.CEMPlanner`, constructed on demand inside `run_all`.

The buffer (`PongOfflineBuffer` or `PushTOfflineSampler`) is passed in
as a method argument — the suite does not own it. The trainer owns the
buffer; both training-time eval (`LeWMTrainer.run` end) and standalone
eval-from-checkpoint (`LeWMTrainer.run_eval`) reuse the same instance.

Public entry: `run_all[BUF: OfflineBuffer](state, buf, ctx) raises` —
sequences the four phases conditionally based on `eval_steps`,
`eval_shuffle_diag`, `eval_h7_closed_loop`, and `mpc_horizon`.
"""

from std.math import ceildiv
from std.memory import alloc
from std.random import seed as _set_seed, random_float64
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor

from ...nn.constants import dtype

from mojo_rl.core.offline_buffer import OfflineBuffer

from .offline_trainer import LeWMGPUState
from .lewm_config import LeWMConfig
from .cem_planner import CEMPlanner
from .kernels import (
    _run_eval_shot_forward, _run_h6_diag_shots,
    slice_h_kernel, scatter_h_kernel, scatter_target_neg_kernel,
    accumulate_emb_kernel, store_pred_last_kernel,
    slide_emb_window_kernel, slide_actions_window_kernel,
    pixels_uint8_to_fp32_kernel,
)


comptime TPB_X = 4
comptime TPB_Y = 4
comptime TPB_Z = 16


struct LeWMEvalSuite[CONFIG: LeWMConfig](Movable, ImplicitlyDestructible):
    """Eval-only state + methods, templated on `CONFIG: LeWMConfig`.

    Constructed cheaply (no GPU allocations) per `run_all` call. CEM/MPC
    scratch is allocated lazily inside `CEMPlanner.__init__` only when
    `mpc_horizon > 0`.
    """

    comptime GPUState = LeWMGPUState[Self.CONFIG]
    comptime EMB: Int = Self.GPUState.EMB

    # Eval-phase config.
    var eval_steps: Int
    var eval_samples: Int
    var eval_seed: Int
    var eval_shuffle_diag: Bool   # H6 (action-aware) toggle
    var eval_h7_closed_loop: Bool  # H7 (drift) toggle

    # CEM/MPC config — forwarded to CEMPlanner when mpc_horizon > 0.
    # mpc_horizon == 0 means: skip MPC entirely, run random-shot baseline.
    var mpc_horizon: Int
    var cem_iters: Int
    var cem_samples: Int
    var cem_topk: Int
    var cem_smoothing: Float64

    # Receding-horizon MPC config — when > 0 and mpc_horizon > 0, the
    # suite calls ``planner.eval_receding_horizon`` after the open-loop
    # MPC eval. Hypothesis: short-horizon plans + replanning beat
    # long-horizon open-loop CEM at training scale (project_lewm_horizon_sweep).
    var rh_steps: Int

    def __init__(
        out self,
        eval_steps: Int,
        eval_samples: Int,
        eval_seed: Int,
        mpc_horizon: Int = 0,
        cem_iters: Int = 0,
        cem_samples: Int = 64,
        cem_topk: Int = 8,
        cem_smoothing: Float64 = 0.5,
        eval_shuffle_diag: Bool = True,
        eval_h7_closed_loop: Bool = True,
        rh_steps: Int = 0,
    ):
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

    def _sample_and_upload_pixels[BUF: OfflineBuffer](
        mut self,
        mut state: Self.GPUState,
        mut buf: BUF,
        ctx: DeviceContext,
    ) raises:
        """Sample uint8 pixels + actions from `buf`, upload to device,
        run the GPU conversion kernel. Mirror of the trainer's helper —
        the trainer owns its buffer; we accept one via arg.
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

    def run_all[BUF: OfflineBuffer](
        mut self,
        mut state: Self.GPUState,
        mut buf: BUF,
        ctx: DeviceContext,
    ) raises:
        """Sequence the eval phases conditionally based on config flags.

        Order matches the legacy `LeWMTrainer.run_eval` exactly:
            H6 → H7 → (random_shots if no MPC) → (CEM/MPC if MPC enabled).
        """
        if self.eval_steps > 0 and self.eval_shuffle_diag:
            self.eval_h6(state, buf, ctx)

        if self.eval_steps > 0 and self.eval_h7_closed_loop:
            self.eval_h7_closed_loop_drift(state, buf, ctx)

        if self.eval_steps > 0 and self.mpc_horizon == 0:
            self.eval_random_shots(state, buf, ctx)

        if self.eval_steps > 0 and self.mpc_horizon > 0:
            var planner = CEMPlanner[Self.CONFIG](
                ctx, self.mpc_horizon, self.cem_iters,
                self.cem_samples, self.cem_topk, self.cem_smoothing,
            )
            planner.eval(
                state, buf, ctx,
                self.eval_steps, self.eval_samples, self.eval_seed,
            )
            if self.rh_steps > 0:
                planner.eval_receding_horizon(
                    state, buf, ctx,
                    self.eval_steps, self.eval_samples,
                    self.rh_steps, self.eval_seed,
                )

    def eval_h6[BUF: OfflineBuffer](
        mut self,
        mut state: Self.GPUState,
        mut buf: BUF,
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

        var perm_buf = alloc[Int](Self.CONFIG.BATCH)

        var h6_sum_expert: Float64 = 0.0
        var h6_sum_shuf_mean: Float64 = 0.0
        var h6_sum_shuf_min: Float64 = 0.0
        var h6_sum_better: Float64 = 0.0

        for h6_iter in range(self.eval_steps):
            # H6 permutes actions_host in-place via _run_h6_diag_shots, so we
            # snapshot expert actions to actions_sample (small, ~few KB) for
            # the unshuffled MSE reference.
            self._sample_and_upload_pixels(state, buf, ctx)
            for i in range(Self.CONFIG.BATCH * Self.CONFIG.T * Self.CONFIG.ACT):
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
                Self.CONFIG.BATCH, Self.CONFIG.T, Self.CONFIG.H, Self.CONFIG.N_PREDS, Self.EMB, Self.CONFIG.ACT, Self.CONFIG.SMOOTHED, Self.CONFIG.PROJ_H,
                Self.CONFIG.PRED_HEADS, Self.CONFIG.PRED_DIM_HEAD,
                Self.CONFIG.PRED_FF, Self.CONFIG.DEPTH,
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

    def eval_random_shots[BUF: OfflineBuffer](
        mut self,
        mut state: Self.GPUState,
        mut buf: BUF,
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

        # ------------------------------------------------------------------
        # Phase 4 eval — random action shooter (teacher-forced)
        #
        # For each eval iteration:
        #   1. Sample fresh batch.
        #   2. Forward with EXPERT actions -> expert_loss = MSE(pred, real_emb[1:Self.CONFIG.H+1]).
        #   3. For S random samples, replace actions with random one-hot and
        #      re-run AE + POS + Self.CONFIG.DEPTH + PROJ (encoder unchanged).
        #   4. Report ratio expert/random — if << 1, model is action-aware.
        #
        # This is a "teacher-forced" shooter — it scores action sequences against
        # the actual observed next-frame embeddings (NOT a goal frame). True
        # autoregressive MPC with a goal frame is Phase 4b.
        # ------------------------------------------------------------------
        print()
        print("==== Phase 4 eval: random action shooter (teacher-forced) ====")
        _set_seed(self.eval_seed)

        var mse_div = Float64(Self.CONFIG.BATCH * Self.CONFIG.H * Self.EMB)
        var sum_expert: Float64 = 0.0
        var sum_random_mean: Float64 = 0.0
        var sum_random_min: Float64 = 0.0
        var sum_better_frac: Float64 = 0.0

        for eval_iter in range(self.eval_steps):
            # Sample uint8 pixels + actions; convert pixels on GPU.
            self._sample_and_upload_pixels(state, buf, ctx)
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
                    # Generate random one-hot actions (Self.CONFIG.BATCH, Self.CONFIG.T, Self.CONFIG.ACT).
                    for b in range(Self.CONFIG.BATCH):
                        for tt in range(Self.CONFIG.T):
                            var r_act = Int(random_float64() * Float64(Self.CONFIG.ACT))
                            if r_act >= Self.CONFIG.ACT:
                                r_act = Self.CONFIG.ACT - 1
                            for k in range(Self.CONFIG.ACT):
                                state.actions_host[b * Self.CONFIG.T * Self.CONFIG.ACT + tt * Self.CONFIG.ACT + k] = (
                                    Scalar[dtype](1.0)
                                    if k == r_act
                                    else Scalar[dtype](0.0)
                                )
                    ctx.enqueue_copy(state.actions_buf, state.actions_host)

                # One shot through AE + slice + POS + Self.CONFIG.DEPTH × cond_block + PROJ.
                _run_eval_shot_forward[
                    Self.CONFIG.BATCH, Self.CONFIG.T, Self.CONFIG.H, Self.EMB, Self.CONFIG.ACT, Self.CONFIG.SMOOTHED, Self.CONFIG.PROJ_H,
                    Self.CONFIG.PRED_HEADS, Self.CONFIG.PRED_DIM_HEAD,
                    Self.CONFIG.PRED_FF, Self.CONFIG.DEPTH,
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

                # Download pred, score MSE against emb[Self.CONFIG.N_PREDS:Self.CONFIG.N_PREDS+Self.CONFIG.H].
                ctx.enqueue_copy(state.pred_host, state.pred_out_buf)
                ctx.synchronize()
                var l: Float64 = 0.0
                for b in range(Self.CONFIG.BATCH):
                    for i in range(Self.CONFIG.H * Self.EMB):
                        var p = Float64(state.pred_host[b * Self.CONFIG.H * Self.EMB + i])
                        var tgt = Float64(
                            state.emb_host[b * Self.CONFIG.T * Self.EMB + Self.CONFIG.N_PREDS * Self.EMB + i]
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
    def eval_h7_closed_loop_drift[BUF: OfflineBuffer](
        mut self,
        mut state: Self.GPUState,
        mut buf: BUF,
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
        comptime ROLL_T = Self.CONFIG.T + 1

        var rollout_steps = Self.CONFIG.T - Self.CONFIG.H
        if rollout_steps <= 0:
            print()
            print(
                "==== H7: closed-loop drift — SKIPPED (T=",
                Self.CONFIG.T, " <= H=", Self.CONFIG.H, "; no rollout positions) ===="
            )
            return

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

        print()
        print(
            "==== H7: closed-loop drift (rollout_steps=",
            rollout_steps, ") ===="
        )
        _set_seed(self.eval_seed)

        # Local rollout buffers (allocated/freed inside this eval phase).
        var emb_seq_dev_buf = ctx.enqueue_create_buffer[dtype](
            Self.CONFIG.BATCH * ROLL_T * Self.EMB
        )
        var emb_seq_dev_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, ROLL_T * Self.EMB),
            MutAnyOrigin,
        ](emb_seq_dev_buf.unsafe_ptr())
        var action_plan_dev_buf = ctx.enqueue_create_buffer[dtype](
            Self.CONFIG.BATCH * Self.CONFIG.T * Self.CONFIG.ACT
        )
        var action_plan_dev_t = LayoutTensor[
            dtype, Layout.row_major(Self.CONFIG.BATCH, Self.CONFIG.T * Self.CONFIG.ACT),
            MutAnyOrigin,
        ](action_plan_dev_buf.unsafe_ptr())
        var emb_seq_host = ctx.enqueue_create_host_buffer[dtype](
            Self.CONFIG.BATCH * ROLL_T * Self.EMB
        )

        var sum_drift_tf = alloc[Float64](rollout_steps)
        var sum_drift_cl = alloc[Float64](rollout_steps)
        var drift_tf_iter = alloc[Float64](rollout_steps)
        var drift_cl_iter = alloc[Float64](rollout_steps)
        for k in range(rollout_steps):
            sum_drift_tf[k] = 0.0
            sum_drift_cl[k] = 0.0

        var mse_div = Float64(Self.CONFIG.BATCH * Self.EMB)

        for eval_iter in range(self.eval_steps):
            # ---- 1. Sample uint8 pixels + actions; convert pixels on GPU ----
            self._sample_and_upload_pixels(state, buf, ctx)
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
            for i in range(Self.CONFIG.BATCH * ROLL_T * Self.EMB):
                emb_seq_host[i] = Scalar[dtype](0)
            for b in range(Self.CONFIG.BATCH):
                for t in range(Self.CONFIG.T):
                    for e in range(Self.EMB):
                        emb_seq_host[
                            b * ROLL_T * Self.EMB + t * Self.EMB + e
                        ] = state.emb_host[
                            b * Self.CONFIG.T * Self.EMB + t * Self.EMB + e
                        ]
            ctx.enqueue_copy(emb_seq_dev_buf, emb_seq_host)
            ctx.synchronize()

            # ---- 3. TF rollout — emb_seq untouched; every step sees real input ----
            for k in range(rollout_steps):
                ctx.enqueue_function[
                    slide_emb_window_kernel[
                        Self.CONFIG.BATCH, Self.CONFIG.T, Self.CONFIG.H, Self.EMB, ROLL_T
                    ],
                ](
                    emb_seq_dev_t, emb_t, k,
                    grid_dim=(
                        ceildiv(Self.CONFIG.BATCH, TPB_X),
                        ceildiv(Self.CONFIG.T, TPB_Y),
                        ceildiv(Self.EMB, TPB_Z),
                    ),
                    block_dim=(TPB_X, TPB_Y, TPB_Z),
                )
                ctx.enqueue_function[
                    slide_actions_window_kernel[
                        Self.CONFIG.BATCH, Self.CONFIG.T, Self.CONFIG.H, Self.CONFIG.ACT, Self.CONFIG.T
                    ],
                ](
                    action_plan_dev_t, actions_t, k,
                    grid_dim=(
                        ceildiv(Self.CONFIG.BATCH, TPB_X),
                        ceildiv(Self.CONFIG.T, TPB_Y),
                        ceildiv(Self.CONFIG.ACT, TPB_Z),
                    ),
                    block_dim=(TPB_X, TPB_Y, TPB_Z),
                )
                _run_eval_shot_forward[
                    Self.CONFIG.BATCH, Self.CONFIG.T, Self.CONFIG.H, Self.EMB, Self.CONFIG.ACT,
                    Self.CONFIG.SMOOTHED, Self.CONFIG.PROJ_H,
                    Self.CONFIG.PRED_HEADS, Self.CONFIG.PRED_DIM_HEAD,
                    Self.CONFIG.PRED_FF, Self.CONFIG.DEPTH,
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
                for b in range(Self.CONFIG.BATCH):
                    for e in range(Self.EMB):
                        var p = Float64(state.pred_host[
                            b * Self.CONFIG.H * Self.EMB
                            + (Self.CONFIG.H - 1) * Self.EMB + e
                        ])
                        var tgt = Float64(state.emb_host[
                            b * Self.CONFIG.T * Self.EMB
                            + (k + Self.CONFIG.H) * Self.EMB + e
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
                        Self.CONFIG.BATCH, Self.CONFIG.T, Self.CONFIG.H, Self.EMB, ROLL_T
                    ],
                ](
                    emb_seq_dev_t, emb_t, k,
                    grid_dim=(
                        ceildiv(Self.CONFIG.BATCH, TPB_X),
                        ceildiv(Self.CONFIG.T, TPB_Y),
                        ceildiv(Self.EMB, TPB_Z),
                    ),
                    block_dim=(TPB_X, TPB_Y, TPB_Z),
                )
                ctx.enqueue_function[
                    slide_actions_window_kernel[
                        Self.CONFIG.BATCH, Self.CONFIG.T, Self.CONFIG.H, Self.CONFIG.ACT, Self.CONFIG.T
                    ],
                ](
                    action_plan_dev_t, actions_t, k,
                    grid_dim=(
                        ceildiv(Self.CONFIG.BATCH, TPB_X),
                        ceildiv(Self.CONFIG.T, TPB_Y),
                        ceildiv(Self.CONFIG.ACT, TPB_Z),
                    ),
                    block_dim=(TPB_X, TPB_Y, TPB_Z),
                )
                _run_eval_shot_forward[
                    Self.CONFIG.BATCH, Self.CONFIG.T, Self.CONFIG.H, Self.EMB, Self.CONFIG.ACT,
                    Self.CONFIG.SMOOTHED, Self.CONFIG.PROJ_H,
                    Self.CONFIG.PRED_HEADS, Self.CONFIG.PRED_DIM_HEAD,
                    Self.CONFIG.PRED_FF, Self.CONFIG.DEPTH,
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
                for b in range(Self.CONFIG.BATCH):
                    for e in range(Self.EMB):
                        var p = Float64(state.pred_host[
                            b * Self.CONFIG.H * Self.EMB
                            + (Self.CONFIG.H - 1) * Self.EMB + e
                        ])
                        var tgt = Float64(state.emb_host[
                            b * Self.CONFIG.T * Self.EMB
                            + (k + Self.CONFIG.H) * Self.EMB + e
                        ])
                        var diff = p - tgt
                        l += diff * diff
                l /= mse_div
                drift_cl_iter[k] = l
                sum_drift_cl[k] += l
                # Store pred[:, H-1, :] -> emb_seq[:, k+H, :] for next step.
                ctx.enqueue_function[
                    store_pred_last_kernel[
                        Self.CONFIG.BATCH, Self.CONFIG.H, Self.EMB, ROLL_T
                    ],
                ](
                    pred_t, emb_seq_dev_t, k,
                    grid_dim=(ceildiv(Self.CONFIG.BATCH, 16), ceildiv(Self.EMB, 16)),
                    block_dim=(16, 16),
                )

            # ---- 5. Per-iter print ----
            print("  iter", eval_iter)
            for k in range(rollout_steps):
                var ratio = drift_cl_iter[k] / (drift_tf_iter[k] + 1e-12)
                print(
                    "    step", k, " (pos=", k + Self.CONFIG.H, "):",
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
                    "  step", k, "(pos=", k + Self.CONFIG.H, ")",
                    " avg_tf=", avg_tf,
                    " avg_cl=", avg_cl,
                    " cl/tf=", ratio,
                    " (sanity: ~1.0 — both modes see real input at step 0)",
                )
            else:
                print(
                    "  step", k, "(pos=", k + Self.CONFIG.H, ")",
                    " avg_tf=", avg_tf,
                    " avg_cl=", avg_cl,
                    " cl/tf=", ratio,
                    " (cl/tf > 1 = compounding; ~1 = stable rollouts)",
                )

        sum_drift_tf.free()
        sum_drift_cl.free()
        drift_tf_iter.free()
        drift_cl_iter.free()
