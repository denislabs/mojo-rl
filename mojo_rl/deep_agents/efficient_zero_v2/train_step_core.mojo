"""K-step BPTT body of EZ-V2's GPU train step, factored out of
`GenericEfficientZeroV2Agent.train_step_gpu`.

This is the action-agnostic core of the training loop — sections 2-9 of
the original `train_step_gpu` body. The agent retains:

  • Section 1: priority-weighted host sampling + (optional) SARSA
    target-net bootstrap forward + mixed-value-target precompute.
    These touch CPU-resident replay state (`self.state.priorities`,
    `self.state.buffer`, target networks) and stay on the agent.

  • Section 10: priorities writeback to `self.state.priorities[]` plus
    `self.train_step_count` increment.

The action-space dispatch happens at one place inside this core:
section 5.1 (policy CE) goes through `Config.ActSpace.policy_loss_grad_gpu`.
Everything else — rep / dyn / pred forward, SimSiam consistency,
backward, optimizer, priority refresh — is action-agnostic.

See `docs/EZV2_MODULAR_ARCHITECTURE.md` and `docs/EZV2_TRAIN_STEP_EXTRACTION.md`
for the design rationale + call-site map.
"""

from std.math import sqrt
from layout import Layout, LayoutTensor
from std.gpu.host import DeviceContext, DeviceBuffer
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import LSTMCell
from mojo_rl.nn.training import Network
from mojo_rl.deep_agents.efficient_zero_v2.configs import EZV2DiscreteConfig
from mojo_rl.deep_agents.efficient_zero_v2.networks import RewardPrefixHeadMLP
from mojo_rl.deep_agents.efficient_zero_v2.state import EZV2GPUStateBase
from mojo_rl.deep_agents.efficient_zero_v2.kernels import (
    ezv2_copy_obs_at_step_kernel,
    ezv2_build_dyn_input_kernel,
    ezv2_extract_hidden_after_dyn_kernel,
    ezv2_value_loss_grad_kernel,
    ezv2_reward_loss_grad_kernel,
    ezv2_cosine_loss_grad_kernel,
    ezv2_reduce_add_kernel,
    ezv2_add_kernel,
    ezv2_assemble_grad_dyn_step_kernel,
    ezv2_accumulate_dyn_grad_in_kernel,
    ezv2_gather_reward_at_step_kernel,
    ezv2_gather_value_target_kernel,
    ezv2_gather_policy_target_kernel,
    ezv2_gather_fullpi_targets_kernel,
    ezv2_policy_loss_grad_continuous_fullpi_kernel,
    ezv2_priority_from_v_loss_kernel,
    ezv2_copy_lstm_input_kernel,
    ezv2_reward_prefix_loss_grad_kernel,
)
from mojo_rl.deep_agents.muzero.kernels import (
    scalar_transform_kernel,
    two_hot_encode_kernel,
)
from mojo_rl.deep_agents.core.kernels import (
    gradient_norm_kernel,
    gradient_reduce_apply_fused_kernel,
)


def ezv2_train_step_gpu_core[
    Config: EZV2DiscreteConfig,
    SKIP_UPLOAD: Bool = False,
](
    mut gpu: EZV2GPUStateBase[Config],
    ctx: DeviceContext,
    v_min: Float64,
    v_max: Float64,
    max_grad_norm: Float64,
    rng_seed: UInt64 = 0,
) raises -> Tuple[Float64, Float64, Float64, Float64]:
    """Sections 2-9 of `train_step_gpu`: upload + zero accumulators +
    forward + (optional) LSTM forward + per-output gradients + backward
    + optimizer step + priority refresh + download.

    The action-space hook fires at section 5.1 — `Config.ActSpace.policy_loss_grad_gpu`
    is dispatched there, the only point in the body where discrete and
    continuous diverge.

    Returns `(L_R, L_P, L_V, L_G)` already divided by their respective
    sample counts. The caller composes them via `Config.lambda_*` and
    handles the agent-side priority writeback.

    Parameters:
        Config: EZ-V2 config trait.
        SKIP_UPLOAD: When `False` (default), section 2 uploads the host
            pinned `gpu.batch_*_host` / `gpu.value_target_full_host` /
            `gpu.cum_rewards_host` buffers to their device counterparts
            — the legacy host-sampling path. When `True`, section 2 is
            elided: caller is responsible for ensuring the device
            `gpu.batch_*_buf` and `gpu.value_target_full_buf` (+
            `gpu.cum_rewards_buf` if reward-prefix) already hold the
            sampled batch (e.g. the GPU-sampling path that ran
            `ezv2_gpu_sample_and_gather` writes them directly).

    Pre-conditions:
      • `gpu.batch_*_host` filled by section 1 on the agent (when
        `SKIP_UPLOAD=False`); or `gpu.batch_*_buf` filled by GPU
        sampling kernels (when `SKIP_UPLOAD=True`).
      • `gpu.value_target_full_host` filled by section 1's mixed-value /
        SARSA precompute (`SKIP_UPLOAD=False`); or
        `gpu.value_target_full_buf` populated on device
        (`SKIP_UPLOAD=True`).
      • Network params on `gpu` reflect current online weights.

    Post-conditions:
      • `gpu.priorities_out_host[b]` holds per-sample priority refresh
        values (= per-sample value-CE loss at unroll position k=0
        + 1e-3). The caller writes these back into the CPU-resident
        priorities array.
    """
    comptime BATCH = Config.batch_size
    comptime K = Config.unroll_steps
    comptime N_TD = Config.td_steps
    comptime OBS = Config.obs_dim
    comptime ACT = Config.action_dim
    comptime LATENT = Config.latent_dim
    comptime PROJ = Config.proj_dim
    comptime BINS = Config.num_bins
    comptime DYN_IN = LATENT + ACT
    comptime DYN_OUT = LATENT + BINS
    # PRED_OUT is the per-sample stride of the prediction network output —
    # for discrete this is `ACT + BINS` (policy logits + value logits), for
    # continuous it is `2 * ACT_DIM + BINS` (μ_raw ‖ σ_raw ‖ value logits).
    # Use `Config.PRED_OUT` directly so the same body works for both.
    comptime PRED_OUT = Config.PRED_OUT
    # Offset of the value-logits slice within each sample's pred-out row.
    # Equals the policy-output width (= `ACT` for discrete, = `2 * ACT_DIM`
    # for continuous), which is exactly `PRED_OUT - BINS`.
    comptime VALUE_OFF = PRED_OUT - BINS
    comptime CAP = 50000

    comptime TPB: Int = 256
    comptime BATCH_BLOCKS = (BATCH + TPB - 1) // TPB
    comptime BATCH_BINS_BLOCKS = (BATCH * BINS + TPB - 1) // TPB
    comptime LATENT_BLOCKS = (BATCH * LATENT + TPB - 1) // TPB

    # ── 2. Upload host → device ─────────────────────────────────────
    # Skipped under `SKIP_UPLOAD=True` (GPU-sampling path) — caller's
    # `ezv2_gpu_sample_and_gather` already populated the device-side
    # `batch_*_buf` and the GPU-sampling agent method memcpys
    # `value_target_full_buf` from `batch_mcts_val_buf` (SEARCH mode).
    comptime if not SKIP_UPLOAD:
        ctx.enqueue_copy(gpu.batch_obs_buf, gpu.batch_obs_host)
        ctx.enqueue_copy(gpu.batch_actions_buf, gpu.batch_actions_host)
        ctx.enqueue_copy(gpu.batch_rewards_buf, gpu.batch_rewards_host)
        ctx.enqueue_copy(gpu.batch_mcts_pol_buf, gpu.batch_mcts_pol_host)
        ctx.enqueue_copy(gpu.batch_mcts_val_buf, gpu.batch_mcts_val_host)
        ctx.enqueue_copy(gpu.batch_age_buf, gpu.batch_age_host)
        # Full-π targets (paper Eq. 6) for continuous ACT_DIM==1 — extra
        # uploads. For other configs the buffers are still allocated but
        # the kernel isn't dispatched, so contents don't matter.
        ctx.enqueue_copy(
            gpu.batch_mcts_samp_act_buf, gpu.batch_mcts_samp_act_host
        )
        ctx.enqueue_copy(
            gpu.batch_mcts_imp_pi_buf, gpu.batch_mcts_imp_pi_host
        )
        ctx.enqueue_copy(
            gpu.value_target_full_buf, gpu.value_target_full_host
        )
        comptime if Config.use_reward_prefix:
            ctx.enqueue_copy(gpu.cum_rewards_buf, gpu.cum_rewards_host)

    # ── 3. Zero loss accumulators + per-network grads ───────────────
    ctx.enqueue_memset(gpu.L_R_buf, 0)
    ctx.enqueue_memset(gpu.L_P_buf, 0)
    ctx.enqueue_memset(gpu.L_V_buf, 0)
    ctx.enqueue_memset(gpu.L_G_buf, 0)
    ctx.enqueue_memset(gpu.grad_hidden_buf, 0)
    # The CE/cosine kernels overwrite (not accumulate into) the
    # per-output grad slices they're responsible for, so we don't
    # need to zero the grad_pred_out / grad_dyn_out / grad_pred_dyn
    # buffers across train steps — every position they care about
    # gets re-written before being read.
    gpu.representation.zero_grads(ctx)
    gpu.dynamics.zero_grads(ctx)
    gpu.prediction.zero_grads(ctx)
    gpu.projector.zero_grads(ctx)
    gpu.predictor.zero_grads(ctx)

    # ── 4. Forward pass ─────────────────────────────────────────────

    # 4.0 Pre-built LayoutTensor views that we reuse across calls.
    var batch_obs_t = LayoutTensor[
        dtype,
        Layout.row_major(BATCH * (K + 1) * OBS),
        MutAnyOrigin,
    ](gpu.batch_obs_buf.unsafe_ptr())
    var batch_actions_t = LayoutTensor[
        dtype,
        Layout.row_major(BATCH * K * ACT),
        MutAnyOrigin,
    ](gpu.batch_actions_buf.unsafe_ptr())
    var batch_rewards_t = LayoutTensor[
        dtype,
        Layout.row_major(BATCH * K),
        MutAnyOrigin,
    ](gpu.batch_rewards_buf.unsafe_ptr())
    var batch_mcts_pol_t = LayoutTensor[
        dtype,
        Layout.row_major(BATCH * (K + 1) * ACT),
        MutAnyOrigin,
    ](gpu.batch_mcts_pol_buf.unsafe_ptr())
    var value_target_full_t = LayoutTensor[
        dtype,
        Layout.row_major(BATCH * (K + 1)),
        MutAnyOrigin,
    ](gpu.value_target_full_buf.unsafe_ptr())

    var rep_input_t_in = LayoutTensor[
        dtype,
        Layout.row_major(BATCH, Config.RepModel.IN_DIM),
        MutAnyOrigin,
    ](gpu.rep_input_buf.unsafe_ptr())
    var rep_input_t_flat = LayoutTensor[
        dtype,
        Layout.row_major(BATCH * OBS),
        MutAnyOrigin,
    ](gpu.rep_input_buf.unsafe_ptr())
    var dyn_input_t_in = LayoutTensor[
        dtype,
        Layout.row_major(BATCH, Config.DynModel.IN_DIM),
        MutAnyOrigin,
    ](gpu.dyn_input_buf.unsafe_ptr())
    var dyn_input_t_flat = LayoutTensor[
        dtype,
        Layout.row_major(BATCH * DYN_IN),
        MutAnyOrigin,
    ](gpu.dyn_input_buf.unsafe_ptr())
    var obs_step_t_in = LayoutTensor[
        dtype,
        Layout.row_major(BATCH, Config.RepModel.IN_DIM),
        MutAnyOrigin,
    ](gpu.obs_step_buf.unsafe_ptr())
    var obs_step_t_flat = LayoutTensor[
        dtype,
        Layout.row_major(BATCH * OBS),
        MutAnyOrigin,
    ](gpu.obs_step_buf.unsafe_ptr())
    var rep_obs_t_out = LayoutTensor[
        dtype,
        Layout.row_major(BATCH, Config.RepModel.OUT_DIM),
        MutAnyOrigin,
    ](gpu.rep_obs_buf.unsafe_ptr())
    var rep_obs_t_in_proj = LayoutTensor[
        dtype,
        Layout.row_major(BATCH, Config.ProjectorModel.IN_DIM),
        MutAnyOrigin,
    ](gpu.rep_obs_buf.unsafe_ptr())

    # 4.1 rep(o[0]) → hidden[0]
    # Gather batch_obs[:, 0, :] into rep_input_buf.
    comptime gather_obs = ezv2_copy_obs_at_step_kernel[
        BATCH, K + 1, OBS, dtype
    ]
    ctx.enqueue_function[gather_obs](
        batch_obs_t,
        rep_input_t_flat,
        0,
        grid_dim=(BATCH_BLOCKS,),
        block_dim=(TPB,),
    )

    var hidden0_t_out = LayoutTensor[
        dtype,
        Layout.row_major(BATCH, Config.RepModel.OUT_DIM),
        MutAnyOrigin,
    ](gpu.hidden_buf.unsafe_ptr())
    var rep_cache_t_w = LayoutTensor[
        dtype,
        Layout.row_major(BATCH, Config.RepModel.CACHE_SIZE),
        MutAnyOrigin,
    ](gpu.rep_cache_buf.unsafe_ptr())
    Network[Config.RepModel, Config.OptType].forward_gpu_with_cache[
        BATCH
    ](
        ctx,
        rep_input_t_in,
        hidden0_t_out,
        gpu.representation.params_view(),
        gpu.representation.model_state_view(),
        rep_cache_t_w,
        gpu.workspace_buf,
    )

    # 4.2 K dynamics steps: for k=0..K-1 build dyn_input from hidden[k]
    # + actions[:, k, :], call dyn.forward_with_cache → dyn_out[k],
    # extract hidden[k+1].
    comptime build_dyn_in = ezv2_build_dyn_input_kernel[
        BATCH, LATENT, ACT, K, dtype
    ]
    comptime extract_hidden = ezv2_extract_hidden_after_dyn_kernel[
        BATCH, LATENT, BINS, dtype
    ]
    comptime DYN_CS = Config.DynModel.CACHE_SIZE

    for k in range(K):
        # hidden[k] view (single time-slice).
        var hidden_k_flat = LayoutTensor[
            dtype,
            Layout.row_major(BATCH * LATENT),
            MutAnyOrigin,
        ](gpu.hidden_buf.unsafe_ptr() + k * BATCH * LATENT)

        ctx.enqueue_function[build_dyn_in](
            hidden_k_flat,
            batch_actions_t,
            dyn_input_t_flat,
            k,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        var dyn_out_k_in = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Config.DynModel.OUT_DIM),
            MutAnyOrigin,
        ](gpu.dyn_out_buf.unsafe_ptr() + k * BATCH * DYN_OUT)
        var dyn_cache_k_w = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Config.DynModel.CACHE_SIZE),
            MutAnyOrigin,
        ](gpu.dyn_caches_buf.unsafe_ptr() + k * BATCH * DYN_CS)
        Network[
            Config.DynModel, Config.OptType
        ].forward_gpu_with_cache[BATCH](
            ctx,
            dyn_input_t_in,
            dyn_out_k_in,
            gpu.dynamics.params_view(),
            gpu.dynamics.model_state_view(),
            dyn_cache_k_w,
            gpu.workspace_buf,
        )

        # Residual: hidden[k+1] = dyn_out_buf[k][:, :LATENT] + hidden[k].
        # Matches reference `ez_dmc_state.py:270` (`state = hidden + x`).
        var dyn_out_k_flat = LayoutTensor[
            dtype,
            Layout.row_major(BATCH * DYN_OUT),
            MutAnyOrigin,
        ](gpu.dyn_out_buf.unsafe_ptr() + k * BATCH * DYN_OUT)
        var hidden_kp1_flat = LayoutTensor[
            dtype,
            Layout.row_major(BATCH * LATENT),
            MutAnyOrigin,
        ](gpu.hidden_buf.unsafe_ptr() + (k + 1) * BATCH * LATENT)
        ctx.enqueue_function[extract_hidden](
            dyn_out_k_flat,
            hidden_k_flat,
            hidden_kp1_flat,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

    # 4.3 Pred at k = 0..K
    comptime PRED_CS = Config.PredModel.CACHE_SIZE
    for k in range(K + 1):
        var hidden_k_in = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Config.PredModel.IN_DIM),
            MutAnyOrigin,
        ](gpu.hidden_buf.unsafe_ptr() + k * BATCH * LATENT)
        var pred_out_k_in = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Config.PredModel.OUT_DIM),
            MutAnyOrigin,
        ](gpu.pred_out_buf.unsafe_ptr() + k * BATCH * PRED_OUT)
        var pred_cache_k_w = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Config.PredModel.CACHE_SIZE),
            MutAnyOrigin,
        ](gpu.pred_caches_buf.unsafe_ptr() + k * BATCH * PRED_CS)
        Network[
            Config.PredModel, Config.OptType
        ].forward_gpu_with_cache[BATCH](
            ctx,
            hidden_k_in,
            pred_out_k_in,
            gpu.prediction.params_view(),
            gpu.prediction.model_state_view(),
            pred_cache_k_w,
            gpu.workspace_buf,
        )

    # 4.4 SimSiam branches for k_offset = 0..K-1 (k = k_offset + 1).
    comptime PROJ_CS = Config.ProjectorModel.CACHE_SIZE
    comptime PREDR_CS = Config.PredictorModel.CACHE_SIZE

    for k_offset in range(K):
        var k = k_offset + 1

        # Online projector: projector(hidden[k]) → proj_dyn[k_offset]
        var hidden_k_in_proj = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Config.ProjectorModel.IN_DIM),
            MutAnyOrigin,
        ](gpu.hidden_buf.unsafe_ptr() + k * BATCH * LATENT)
        var proj_dyn_k_in = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Config.ProjectorModel.OUT_DIM),
            MutAnyOrigin,
        ](gpu.proj_dyn_buf.unsafe_ptr() + k_offset * BATCH * PROJ)
        var proj_dyn_cache_k_w = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Config.ProjectorModel.CACHE_SIZE),
            MutAnyOrigin,
        ](
            gpu.proj_dyn_caches_buf.unsafe_ptr()
            + k_offset * BATCH * PROJ_CS
        )
        Network[
            Config.ProjectorModel, Config.OptType
        ].forward_gpu_with_cache[BATCH](
            ctx,
            hidden_k_in_proj,
            proj_dyn_k_in,
            gpu.projector.params_view(),
            gpu.projector.model_state_view(),
            proj_dyn_cache_k_w,
            gpu.workspace_buf,
        )

        # Predictor: predictor(proj_dyn[k_offset]) → pred_dyn[k_offset]
        var proj_dyn_k_input = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Config.PredictorModel.IN_DIM),
            MutAnyOrigin,
        ](gpu.proj_dyn_buf.unsafe_ptr() + k_offset * BATCH * PROJ)
        var pred_dyn_k_in = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Config.PredictorModel.OUT_DIM),
            MutAnyOrigin,
        ](gpu.pred_dyn_buf.unsafe_ptr() + k_offset * BATCH * PROJ)
        var pred_dyn_cache_k_w = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Config.PredictorModel.CACHE_SIZE),
            MutAnyOrigin,
        ](
            gpu.pred_dyn_caches_buf.unsafe_ptr()
            + k_offset * BATCH * PREDR_CS
        )
        Network[
            Config.PredictorModel, Config.OptType
        ].forward_gpu_with_cache[BATCH](
            ctx,
            proj_dyn_k_input,
            pred_dyn_k_in,
            gpu.predictor.params_view(),
            gpu.predictor.model_state_view(),
            pred_dyn_cache_k_w,
            gpu.workspace_buf,
        )

        # Target branch (no cache, no gradient).
        # Gather batch_obs[:, k, :] → obs_step.
        ctx.enqueue_function[gather_obs](
            batch_obs_t,
            obs_step_t_flat,
            k,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # rep(obs_step) → rep_obs (no cache; stop-grad target)
        Network[
            Config.RepModel, Config.OptType
        ].forward_gpu[BATCH](
            ctx,
            obs_step_t_in,
            rep_obs_t_out,
            gpu.representation.params_view(),
            gpu.representation.model_state_view(),
            gpu.workspace_buf,
        )

        # projector(rep_obs) → proj_obs[k_offset] (no cache)
        var proj_obs_k_in = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Config.ProjectorModel.OUT_DIM),
            MutAnyOrigin,
        ](gpu.proj_obs_buf.unsafe_ptr() + k_offset * BATCH * PROJ)
        Network[
            Config.ProjectorModel, Config.OptType
        ].forward_gpu[BATCH](
            ctx,
            rep_obs_t_in_proj,
            proj_obs_k_in,
            gpu.projector.params_view(),
            gpu.projector.model_state_view(),
            gpu.workspace_buf,
        )

    # ── 4.5 Reward-prefix LSTM forward (paper App. G, when enabled) ─
    comptime LSTM_HIDDEN = Config.lstm_hidden
    comptime LSTM_HORIZON = Config.lstm_horizon_len
    comptime _LSTMHead = LSTMCell[LATENT, LSTM_HIDDEN]
    comptime _RewardPrefixMLP = RewardPrefixHeadMLP[
        LSTM_HIDDEN,
        Config.lstm_mlp_hidden,
        BINS,
    ]
    comptime _LSTM_CS = _LSTMHead.CACHE_SIZE
    comptime _MLP_HEAD_CS = _RewardPrefixMLP.CACHE_SIZE
    comptime LSTM_HIDDEN_BLOCKS = (
        BATCH * LSTM_HIDDEN + TPB - 1
    ) // TPB
    comptime copy_lstm_input = ezv2_copy_lstm_input_kernel[
        BATCH, LSTM_HIDDEN, dtype
    ]
    comptime add_kernel_lstm_h = ezv2_add_kernel[
        BATCH * LSTM_HIDDEN, dtype
    ]
    comptime add_kernel_latent = ezv2_add_kernel[BATCH * LATENT, dtype]

    comptime if Config.use_reward_prefix:
        # Reset h_lstm[0], c_lstm[0] = 0 at the start of every batch.
        ctx.enqueue_memset(gpu.lstm_h_states_buf, 0)
        ctx.enqueue_memset(gpu.lstm_c_states_buf, 0)

        var lstm_params_v_f = LayoutTensor[
            dtype,
            Layout.row_major(_LSTMHead.PARAM_SIZE),
            MutAnyOrigin,
        ](gpu.lstm_params_buf.unsafe_ptr())

        for k in range(K):
            var reset_now = (k > 0) and (k % LSTM_HORIZON == 0)
            if reset_now:
                ctx.enqueue_memset(gpu.lstm_h_input_buf, 0)
                ctx.enqueue_memset(gpu.lstm_c_input_buf, 0)
            else:
                # Copy lstm_h_states[k], lstm_c_states[k] → input scratch.
                var hs_k_flat = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH * LSTM_HIDDEN),
                    MutAnyOrigin,
                ](
                    gpu.lstm_h_states_buf.unsafe_ptr()
                    + k * BATCH * LSTM_HIDDEN
                )
                var cs_k_flat = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH * LSTM_HIDDEN),
                    MutAnyOrigin,
                ](
                    gpu.lstm_c_states_buf.unsafe_ptr()
                    + k * BATCH * LSTM_HIDDEN
                )
                var h_in_flat = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH * LSTM_HIDDEN),
                    MutAnyOrigin,
                ](gpu.lstm_h_input_buf.unsafe_ptr())
                var c_in_flat = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH * LSTM_HIDDEN),
                    MutAnyOrigin,
                ](gpu.lstm_c_input_buf.unsafe_ptr())
                ctx.enqueue_function[copy_lstm_input](
                    hs_k_flat,
                    cs_k_flat,
                    h_in_flat,
                    c_in_flat,
                    grid_dim=(LSTM_HIDDEN_BLOCKS,),
                    block_dim=(TPB,),
                )

            # LSTM step k: input is hidden[k+1] (post-dyn-step-k latent).
            var z_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
            ](gpu.hidden_buf.unsafe_ptr() + (k + 1) * BATCH * LATENT)
            var h_prev_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, LSTM_HIDDEN),
                MutAnyOrigin,
            ](gpu.lstm_h_input_buf.unsafe_ptr())
            var c_prev_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, LSTM_HIDDEN),
                MutAnyOrigin,
            ](gpu.lstm_c_input_buf.unsafe_ptr())
            var h_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, LSTM_HIDDEN),
                MutAnyOrigin,
            ](
                gpu.lstm_h_states_buf.unsafe_ptr()
                + (k + 1) * BATCH * LSTM_HIDDEN
            )
            var c_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, LSTM_HIDDEN),
                MutAnyOrigin,
            ](
                gpu.lstm_c_states_buf.unsafe_ptr()
                + (k + 1) * BATCH * LSTM_HIDDEN
            )
            var lstm_cache_t_f = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, _LSTM_CS),
                MutAnyOrigin,
            ](gpu.lstm_caches_buf.unsafe_ptr() + k * BATCH * _LSTM_CS)
            _LSTMHead.step_forward_gpu[BATCH](
                ctx,
                z_t,
                h_prev_t,
                c_prev_t,
                lstm_params_v_f,
                h_t,
                c_t,
                lstm_cache_t_f,
            )

            # MLP head forward(h_lstm[k+1]) → rew_pref_logits[k]
            var mlp_in_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, _RewardPrefixMLP.IN_DIM),
                MutAnyOrigin,
            ](
                gpu.lstm_h_states_buf.unsafe_ptr()
                + (k + 1) * BATCH * LSTM_HIDDEN
            )
            var mlp_out_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, _RewardPrefixMLP.OUT_DIM),
                MutAnyOrigin,
            ](
                gpu.rew_pref_logits_buf.unsafe_ptr()
                + k * BATCH * BINS
            )
            var mlp_cache_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, _MLP_HEAD_CS),
                MutAnyOrigin,
            ](
                gpu.mlp_head_caches_buf.unsafe_ptr()
                + k * BATCH * _MLP_HEAD_CS
            )
            Network[
                _RewardPrefixMLP, Config.OptType
            ].forward_gpu_with_cache[BATCH](
                ctx,
                mlp_in_t,
                mlp_out_t,
                gpu.reward_prefix_mlp_gpu.params_view(),
                gpu.reward_prefix_mlp_gpu.model_state_view(),
                mlp_cache_t,
                gpu.workspace_buf,
            )

    # ── 5. Per-output upstream gradients + per-sample loss reductions ─

    # 5.1 Policy CE at every k = 0..K
    # The policy-head loss + grad goes through `Config.ActSpace` —
    # the only action-space-specific hook in the K-step BPTT body.
    # Discrete uses the existing CE kernel; continuous (Phase 3)
    # supplies a different one. See `docs/EZV2_MODULAR_ARCHITECTURE.md`.
    #
    # Continuous ACT_DIM==1 (Pendulum etc., paper Eq. 6 branch in
    # `ez/utils/loss.py:continuous_loss`) takes a separate full-π path:
    # K root-sampled candidates + improved-policy weights from the MCTS
    # search produce a weighted NLL instead of a simple-best NLL.
    comptime POL_TGT_DIM = Config.ActSpace.POLICY_TARGET_DIM
    comptime _ACT = Config.action_dim
    comptime _K_ROOT_CFG = Config.num_root_candidates
    comptime USE_FULLPI = (
        Config.ActSpace.IS_CONTINUOUS and _ACT == 1
    )
    comptime gather_pol = ezv2_gather_policy_target_kernel[
        BATCH, K + 1, POL_TGT_DIM, dtype
    ]
    comptime gather_fullpi = ezv2_gather_fullpi_targets_kernel[
        BATCH, K + 1, _K_ROOT_CFG, _ACT, dtype
    ]
    comptime fullpi_kernel = (
        ezv2_policy_loss_grad_continuous_fullpi_kernel[
            BATCH, _ACT, _K_ROOT_CFG, PRED_OUT, dtype
        ]
    )
    comptime reduce_one = ezv2_reduce_add_kernel[BATCH, dtype]
    var policy_target_step_t = LayoutTensor[
        dtype, Layout.row_major(BATCH * POL_TGT_DIM), MutAnyOrigin
    ](gpu.policy_target_step_buf.unsafe_ptr())
    var fullpi_target_actions_step_t = LayoutTensor[
        dtype,
        Layout.row_major(BATCH * _K_ROOT_CFG * _ACT),
        MutAnyOrigin,
    ](gpu.fullpi_target_actions_step_buf.unsafe_ptr())
    var fullpi_target_policy_step_t = LayoutTensor[
        dtype, Layout.row_major(BATCH * _K_ROOT_CFG), MutAnyOrigin
    ](gpu.fullpi_target_policy_step_buf.unsafe_ptr())
    var batch_mcts_samp_act_t = LayoutTensor[
        dtype,
        Layout.row_major(BATCH * (K + 1) * _K_ROOT_CFG * _ACT),
        MutAnyOrigin,
    ](gpu.batch_mcts_samp_act_buf.unsafe_ptr())
    var batch_mcts_imp_pi_t = LayoutTensor[
        dtype,
        Layout.row_major(BATCH * (K + 1) * _K_ROOT_CFG),
        MutAnyOrigin,
    ](gpu.batch_mcts_imp_pi_buf.unsafe_ptr())
    var per_sample_loss_t = LayoutTensor[
        dtype, Layout.row_major(BATCH), MutAnyOrigin
    ](gpu.per_sample_loss_scratch_buf.unsafe_ptr())
    var L_P_t = LayoutTensor[
        dtype, Layout.row_major(1), MutAnyOrigin
    ](gpu.L_P_buf.unsafe_ptr())

    var n_P = Float64(BATCH * (K + 1))
    var lp_scale = Config.lambda_policy / n_P
    var ent_scale = Config.entropy_weight / n_P
    # `MAX_ACTION` / `MIN_STD` live on `ContinuousActionSpace` only —
    # bind them at comptime so the discrete path doesn't try to resolve
    # them on `DiscreteActionSpace`.
    comptime MAX_ACTION_F: Float64 = (
        Config.ActSpace.MAX_ACTION if USE_FULLPI else Float64(0.0)
    )
    comptime MIN_STD_F: Float64 = (
        Config.ActSpace.MIN_STD if USE_FULLPI else Float64(0.0)
    )
    var max_action_s = Scalar[dtype](MAX_ACTION_F)
    var min_std_s = Scalar[dtype](MIN_STD_F)
    for k in range(K + 1):
        var pred_out_k_flat = LayoutTensor[
            dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
        ](gpu.pred_out_buf.unsafe_ptr() + k * BATCH * PRED_OUT)
        var grad_pred_out_k_flat = LayoutTensor[
            dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
        ](gpu.grad_pred_out_buf.unsafe_ptr() + k * BATCH * PRED_OUT)
        # Mix `rng_seed` with the unroll-step index `k` so each per-slice
        # MC entropy estimator gets a distinct Philox stream.
        var policy_seed_k = (
            rng_seed * UInt64(0x9E3779B97F4A7C15)
            + UInt64(k) * UInt64(2862933555777941757)
        )
        comptime if USE_FULLPI:
            ctx.enqueue_function[gather_fullpi](
                batch_mcts_samp_act_t,
                batch_mcts_imp_pi_t,
                fullpi_target_actions_step_t,
                fullpi_target_policy_step_t,
                k,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
            ctx.enqueue_function[fullpi_kernel](
                pred_out_k_flat,
                fullpi_target_actions_step_t,
                fullpi_target_policy_step_t,
                grad_pred_out_k_flat,
                per_sample_loss_t,
                Scalar[dtype](lp_scale),
                Scalar[dtype](ent_scale),
                max_action_s,
                min_std_s,
                policy_seed_k,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
        else:
            ctx.enqueue_function[gather_pol](
                batch_mcts_pol_t,
                policy_target_step_t,
                k,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
            Config.ActSpace.policy_loss_grad_gpu[
                BATCH, PRED_OUT, POL_TGT_DIM, dtype
            ](
                ctx,
                pred_out_k_flat,
                policy_target_step_t,
                grad_pred_out_k_flat,
                per_sample_loss_t,
                Scalar[dtype](lp_scale),
                Scalar[dtype](ent_scale),
                policy_seed_k,
            )
        ctx.enqueue_function[reduce_one](
            per_sample_loss_t,
            L_P_t,
            grid_dim=(1,),
            block_dim=(1,),
        )

    # 5.2 Value CE at every k = 0..K, using pre-uploaded mixed value
    # target. Build two-hot dist by gathering scalar → scalar_transform
    # → two_hot_encode, then run value_loss_grad_kernel. At k = 0,
    # additionally copy per_sample_loss to per_sample_v_loss_k0 for
    # the priority refresh.
    comptime gather_val = ezv2_gather_value_target_kernel[
        BATCH, K + 1, dtype
    ]
    comptime st_kernel = scalar_transform_kernel[BATCH, dtype]
    comptime th_kernel = two_hot_encode_kernel[BATCH, BINS, dtype]
    comptime value_grad = ezv2_value_loss_grad_kernel[
        BATCH, BINS, VALUE_OFF, PRED_OUT, dtype
    ]
    comptime BATCH_BLOCKS_BS = (BATCH + TPB - 1) // TPB
    comptime BINS_BLOCKS = (BATCH + TPB - 1) // TPB

    var value_target_scalar_t = LayoutTensor[
        dtype, Layout.row_major(BATCH), MutAnyOrigin
    ](gpu.value_target_scalar_buf.unsafe_ptr())
    var value_target_dist_t = LayoutTensor[
        dtype, Layout.row_major(BATCH * BINS), MutAnyOrigin
    ](gpu.value_target_dist_buf.unsafe_ptr())
    var L_V_t = LayoutTensor[
        dtype, Layout.row_major(1), MutAnyOrigin
    ](gpu.L_V_buf.unsafe_ptr())
    var per_sample_v_loss_k0_t = LayoutTensor[
        dtype, Layout.row_major(BATCH), MutAnyOrigin
    ](gpu.per_sample_v_loss_k0_buf.unsafe_ptr())

    var n_V = Float64(BATCH * (K + 1))
    var lv_scale = Config.lambda_value / n_V
    comptime copy_kernel_b = ezv2_add_kernel[BATCH, dtype]
    # NOTE: at k=0 we want a *copy* of per_sample_loss into
    # per_sample_v_loss_k0_buf, not an add. Easiest: zero
    # per_sample_v_loss_k0_buf via memset before the loop, then add.
    ctx.enqueue_memset(gpu.per_sample_v_loss_k0_buf, 0)

    for k in range(K + 1):
        ctx.enqueue_function[gather_val](
            value_target_full_t,
            value_target_scalar_t,
            k,
            grid_dim=(BATCH_BLOCKS_BS,),
            block_dim=(TPB,),
        )
        ctx.enqueue_function[st_kernel](
            value_target_scalar_t,
            Scalar[dtype](0.001),
            grid_dim=(BATCH_BLOCKS_BS,),
            block_dim=(TPB,),
        )
        ctx.enqueue_function[th_kernel](
            value_target_dist_t,
            value_target_scalar_t,
            Scalar[dtype](v_min),
            Scalar[dtype](v_max),
            grid_dim=(BINS_BLOCKS,),
            block_dim=(TPB,),
        )
        var pred_out_k_flat = LayoutTensor[
            dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
        ](gpu.pred_out_buf.unsafe_ptr() + k * BATCH * PRED_OUT)
        var grad_pred_out_k_flat = LayoutTensor[
            dtype, Layout.row_major(BATCH * PRED_OUT), MutAnyOrigin
        ](gpu.grad_pred_out_buf.unsafe_ptr() + k * BATCH * PRED_OUT)
        ctx.enqueue_function[value_grad](
            pred_out_k_flat,
            value_target_dist_t,
            grad_pred_out_k_flat,
            per_sample_loss_t,
            Scalar[dtype](lv_scale),
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )
        ctx.enqueue_function[reduce_one](
            per_sample_loss_t,
            L_V_t,
            grid_dim=(1,),
            block_dim=(1,),
        )
        if k == 0:
            ctx.enqueue_function[copy_kernel_b](
                per_sample_v_loss_k0_t,
                per_sample_loss_t,
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )

    # 5.3 Reward CE at every k = 0..K-1.
    # Two paths (matches CPU):
    #   • use_reward_prefix=False: classic per-step reward CE through
    #     the dyn-network's reward output slice.
    #   • use_reward_prefix=True: cumulative-reward CE through the
    #     reward-prefix LSTM head (rew_pref_logits_buf). The dyn-
    #     output's reward grad slice MUST be zeroed so dyn-backward
    #     sees no gradient through that path.
    comptime gather_rew = ezv2_gather_reward_at_step_kernel[
        BATCH, K, dtype
    ]
    comptime reward_grad = ezv2_reward_loss_grad_kernel[
        BATCH, BINS, LATENT, DYN_OUT, dtype
    ]
    comptime rew_pref_grad = ezv2_reward_prefix_loss_grad_kernel[
        BATCH, BINS, dtype
    ]
    var reward_target_scalar_t = LayoutTensor[
        dtype, Layout.row_major(BATCH), MutAnyOrigin
    ](gpu.reward_target_scalar_buf.unsafe_ptr())
    var reward_target_dist_t = LayoutTensor[
        dtype, Layout.row_major(BATCH * BINS), MutAnyOrigin
    ](gpu.reward_target_dist_buf.unsafe_ptr())
    var L_R_t = LayoutTensor[
        dtype, Layout.row_major(1), MutAnyOrigin
    ](gpu.L_R_buf.unsafe_ptr())
    var cum_rewards_t = LayoutTensor[
        dtype, Layout.row_major(BATCH * K), MutAnyOrigin
    ](gpu.cum_rewards_buf.unsafe_ptr())
    var rew_pref_target_dist_t = LayoutTensor[
        dtype, Layout.row_major(BATCH * BINS), MutAnyOrigin
    ](gpu.rew_pref_target_dist_buf.unsafe_ptr())

    var n_R = Float64(BATCH * K)
    var lr_scale = Config.lambda_reward / n_R

    comptime if not Config.use_reward_prefix:
        for k in range(K):
            ctx.enqueue_function[gather_rew](
                batch_rewards_t,
                reward_target_scalar_t,
                k,
                grid_dim=(BATCH_BLOCKS_BS,),
                block_dim=(TPB,),
            )
            ctx.enqueue_function[st_kernel](
                reward_target_scalar_t,
                Scalar[dtype](0.001),
                grid_dim=(BATCH_BLOCKS_BS,),
                block_dim=(TPB,),
            )
            ctx.enqueue_function[th_kernel](
                reward_target_dist_t,
                reward_target_scalar_t,
                Scalar[dtype](v_min),
                Scalar[dtype](v_max),
                grid_dim=(BINS_BLOCKS,),
                block_dim=(TPB,),
            )
            var dyn_out_k_flat = LayoutTensor[
                dtype, Layout.row_major(BATCH * DYN_OUT), MutAnyOrigin
            ](gpu.dyn_out_buf.unsafe_ptr() + k * BATCH * DYN_OUT)
            var grad_dyn_out_k_flat = LayoutTensor[
                dtype, Layout.row_major(BATCH * DYN_OUT), MutAnyOrigin
            ](gpu.grad_dyn_out_buf.unsafe_ptr() + k * BATCH * DYN_OUT)
            ctx.enqueue_function[reward_grad](
                dyn_out_k_flat,
                reward_target_dist_t,
                grad_dyn_out_k_flat,
                per_sample_loss_t,
                Scalar[dtype](lr_scale),
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
            ctx.enqueue_function[reduce_one](
                per_sample_loss_t,
                L_R_t,
                grid_dim=(1,),
                block_dim=(1,),
            )
    else:
        # Zero the dyn-output reward grad slice so dyn-backward sees
        # no gradient through that path. (Without this, stale data
        # from previous train_step calls would flow through.) The
        # whole grad_dyn_out_buf is fine to zero — the pred-out
        # grad and the K-1 hidden-slice piece get rebuilt anyway.
        ctx.enqueue_memset(gpu.grad_dyn_out_buf, 0)

        for k in range(K):
            # Gather cum_rewards[:, k] → reward_target_scalar.
            ctx.enqueue_function[gather_rew](
                cum_rewards_t,
                reward_target_scalar_t,
                k,
                grid_dim=(BATCH_BLOCKS_BS,),
                block_dim=(TPB,),
            )
            ctx.enqueue_function[st_kernel](
                reward_target_scalar_t,
                Scalar[dtype](0.001),
                grid_dim=(BATCH_BLOCKS_BS,),
                block_dim=(TPB,),
            )
            ctx.enqueue_function[th_kernel](
                rew_pref_target_dist_t,
                reward_target_scalar_t,
                Scalar[dtype](v_min),
                Scalar[dtype](v_max),
                grid_dim=(BINS_BLOCKS,),
                block_dim=(TPB,),
            )
            var rpl_k_flat = LayoutTensor[
                dtype,
                Layout.row_major(BATCH * BINS),
                MutAnyOrigin,
            ](
                gpu.rew_pref_logits_buf.unsafe_ptr()
                + k * BATCH * BINS
            )
            var grad_rpl_k_flat = LayoutTensor[
                dtype,
                Layout.row_major(BATCH * BINS),
                MutAnyOrigin,
            ](
                gpu.grad_rew_pref_logits_buf.unsafe_ptr()
                + k * BATCH * BINS
            )
            ctx.enqueue_function[rew_pref_grad](
                rpl_k_flat,
                rew_pref_target_dist_t,
                grad_rpl_k_flat,
                per_sample_loss_t,
                Scalar[dtype](lr_scale),
                grid_dim=(BATCH_BLOCKS,),
                block_dim=(TPB,),
            )
            ctx.enqueue_function[reduce_one](
                per_sample_loss_t,
                L_R_t,
                grid_dim=(1,),
                block_dim=(1,),
            )

    # 5.4 Cosine consistency at every k_offset = 0..K-1.
    comptime cosine_grad = ezv2_cosine_loss_grad_kernel[
        BATCH, PROJ, dtype
    ]
    var L_G_t = LayoutTensor[
        dtype, Layout.row_major(1), MutAnyOrigin
    ](gpu.L_G_buf.unsafe_ptr())

    var n_G = Float64(BATCH * K)
    var lg_scale = Config.lambda_consistency / n_G
    for k_offset in range(K):
        var pred_dyn_k_flat = LayoutTensor[
            dtype, Layout.row_major(BATCH * PROJ), MutAnyOrigin
        ](gpu.pred_dyn_buf.unsafe_ptr() + k_offset * BATCH * PROJ)
        var proj_obs_k_flat = LayoutTensor[
            dtype, Layout.row_major(BATCH * PROJ), MutAnyOrigin
        ](gpu.proj_obs_buf.unsafe_ptr() + k_offset * BATCH * PROJ)
        var grad_pred_dyn_k_flat = LayoutTensor[
            dtype, Layout.row_major(BATCH * PROJ), MutAnyOrigin
        ](gpu.grad_pred_dyn_buf.unsafe_ptr() + k_offset * BATCH * PROJ)
        ctx.enqueue_function[cosine_grad](
            pred_dyn_k_flat,
            proj_obs_k_flat,
            grad_pred_dyn_k_flat,
            per_sample_loss_t,
            Scalar[dtype](lg_scale),
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )
        ctx.enqueue_function[reduce_one](
            per_sample_loss_t,
            L_G_t,
            grid_dim=(1,),
            block_dim=(1,),
        )

    # ── 6. Backward pass ────────────────────────────────────────────

    # 6.1 pred backward at k = 0..K → adds into grad_hidden[k].
    comptime add_kernel_lat = ezv2_add_kernel[BATCH * LATENT, dtype]
    for k in range(K + 1):
        var grad_pred_out_k_in = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Config.PredModel.OUT_DIM),
            MutAnyOrigin,
        ](gpu.grad_pred_out_buf.unsafe_ptr() + k * BATCH * PRED_OUT)
        var grad_pred_in_step_t_in = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Config.PredModel.IN_DIM),
            MutAnyOrigin,
        ](gpu.grad_pred_in_step_buf.unsafe_ptr())
        var pred_cache_k_r = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Config.PredModel.CACHE_SIZE),
            MutAnyOrigin,
        ](gpu.pred_caches_buf.unsafe_ptr() + k * BATCH * PRED_CS)
        var pred_grads_v = gpu.prediction.grads_view()
        Network[
            Config.PredModel, Config.OptType
        ].backward_gpu[BATCH](
            ctx,
            grad_pred_out_k_in,
            grad_pred_in_step_t_in,
            gpu.prediction.params_view(),
            gpu.prediction.model_state_view(),
            pred_cache_k_r,
            pred_grads_v,
            gpu.workspace_buf,
        )
        # grad_hidden[k] += grad_pred_in_step
        var grad_hidden_k_flat = LayoutTensor[
            dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
        ](gpu.grad_hidden_buf.unsafe_ptr() + k * BATCH * LATENT)
        var grad_pred_in_step_flat = LayoutTensor[
            dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
        ](gpu.grad_pred_in_step_buf.unsafe_ptr())
        ctx.enqueue_function[add_kernel_lat](
            grad_hidden_k_flat,
            grad_pred_in_step_flat,
            grid_dim=(LATENT_BLOCKS,),
            block_dim=(TPB,),
        )

    # 6.2 SimSiam backward (predictor + projector online branch) at
    # k_offset = 0..K-1 → adds into grad_hidden[k_offset + 1].
    for k_offset in range(K):
        var k = k_offset + 1
        # predictor.backward(grad_pred_dyn[k_offset]) → grad_predr_in_step
        var grad_pred_dyn_k_in = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Config.PredictorModel.OUT_DIM),
            MutAnyOrigin,
        ](gpu.grad_pred_dyn_buf.unsafe_ptr() + k_offset * BATCH * PROJ)
        var grad_predr_in_step_t_in = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Config.PredictorModel.IN_DIM),
            MutAnyOrigin,
        ](gpu.grad_predr_in_step_buf.unsafe_ptr())
        var pred_dyn_cache_k_r = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Config.PredictorModel.CACHE_SIZE),
            MutAnyOrigin,
        ](
            gpu.pred_dyn_caches_buf.unsafe_ptr()
            + k_offset * BATCH * PREDR_CS
        )
        var predr_grads_v = gpu.predictor.grads_view()
        Network[
            Config.PredictorModel, Config.OptType
        ].backward_gpu[BATCH](
            ctx,
            grad_pred_dyn_k_in,
            grad_predr_in_step_t_in,
            gpu.predictor.params_view(),
            gpu.predictor.model_state_view(),
            pred_dyn_cache_k_r,
            predr_grads_v,
            gpu.workspace_buf,
        )

        # projector.backward(grad_predr_in_step → grad_proj_in_step)
        var grad_predr_in_step_in_proj = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Config.ProjectorModel.OUT_DIM),
            MutAnyOrigin,
        ](gpu.grad_predr_in_step_buf.unsafe_ptr())
        var grad_proj_in_step_t_in = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Config.ProjectorModel.IN_DIM),
            MutAnyOrigin,
        ](gpu.grad_proj_in_step_buf.unsafe_ptr())
        var proj_dyn_cache_k_r = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Config.ProjectorModel.CACHE_SIZE),
            MutAnyOrigin,
        ](
            gpu.proj_dyn_caches_buf.unsafe_ptr()
            + k_offset * BATCH * PROJ_CS
        )
        var proj_grads_v = gpu.projector.grads_view()
        Network[
            Config.ProjectorModel, Config.OptType
        ].backward_gpu[BATCH](
            ctx,
            grad_predr_in_step_in_proj,
            grad_proj_in_step_t_in,
            gpu.projector.params_view(),
            gpu.projector.model_state_view(),
            proj_dyn_cache_k_r,
            proj_grads_v,
            gpu.workspace_buf,
        )

        # grad_hidden[k] += grad_proj_in_step
        var grad_hidden_k_flat = LayoutTensor[
            dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
        ](gpu.grad_hidden_buf.unsafe_ptr() + k * BATCH * LATENT)
        var grad_proj_in_step_flat = LayoutTensor[
            dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
        ](gpu.grad_proj_in_step_buf.unsafe_ptr())
        ctx.enqueue_function[add_kernel_lat](
            grad_hidden_k_flat,
            grad_proj_in_step_flat,
            grid_dim=(LATENT_BLOCKS,),
            block_dim=(TPB,),
        )

    # 6.2'. Reward-prefix LSTM head backward (when use_reward_prefix=True)
    # — pass 1: MLP head backward at every k → adds to grad_h_lstm[k+1].
    # pass 2: LSTM step_backward in REVERSE time, threading dh/dc back to
    # the previous step (or discarding at horizon-reset boundary), and
    # accumulating ∂L/∂hidden[k+1] into the existing grad_hidden buffer.
    comptime if Config.use_reward_prefix:
        ctx.enqueue_memset(gpu.lstm_grads_buf, 0)
        gpu.reward_prefix_mlp_gpu.zero_grads(ctx)
        ctx.enqueue_memset(gpu.grad_h_lstm_buf, 0)
        ctx.enqueue_memset(gpu.grad_c_lstm_buf, 0)

        # Pass 1: MLP head backward k = 0..K-1.
        for k in range(K):
            var grad_logits_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, _RewardPrefixMLP.OUT_DIM),
                MutAnyOrigin,
            ](
                gpu.grad_rew_pref_logits_buf.unsafe_ptr()
                + k * BATCH * BINS
            )
            var grad_mlp_in_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, _RewardPrefixMLP.IN_DIM),
                MutAnyOrigin,
            ](gpu.grad_mlp_in_step_buf.unsafe_ptr())
            var mlp_cache_t_b = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, _MLP_HEAD_CS),
                MutAnyOrigin,
            ](
                gpu.mlp_head_caches_buf.unsafe_ptr()
                + k * BATCH * _MLP_HEAD_CS
            )
            var mlp_grads_v = gpu.reward_prefix_mlp_gpu.grads_view()
            Network[
                _RewardPrefixMLP, Config.OptType
            ].backward_gpu[BATCH](
                ctx,
                grad_logits_t,
                grad_mlp_in_t,
                gpu.reward_prefix_mlp_gpu.params_view(),
                gpu.reward_prefix_mlp_gpu.model_state_view(),
                mlp_cache_t_b,
                mlp_grads_v,
                gpu.workspace_buf,
            )

            # Accumulate grad_mlp_in_step → grad_h_lstm[k+1]
            var grad_h_lstm_kp1_flat = LayoutTensor[
                dtype,
                Layout.row_major(BATCH * LSTM_HIDDEN),
                MutAnyOrigin,
            ](
                gpu.grad_h_lstm_buf.unsafe_ptr()
                + (k + 1) * BATCH * LSTM_HIDDEN
            )
            var grad_mlp_in_step_flat = LayoutTensor[
                dtype,
                Layout.row_major(BATCH * LSTM_HIDDEN),
                MutAnyOrigin,
            ](gpu.grad_mlp_in_step_buf.unsafe_ptr())
            ctx.enqueue_function[add_kernel_lstm_h](
                grad_h_lstm_kp1_flat,
                grad_mlp_in_step_flat,
                grid_dim=(LSTM_HIDDEN_BLOCKS,),
                block_dim=(TPB,),
            )

        # Pass 2: LSTM backward in REVERSE time, kk = 0..K-1, k = K-1..0.
        var lstm_params_v_b = LayoutTensor[
            dtype,
            Layout.row_major(_LSTMHead.PARAM_SIZE),
            MutAnyOrigin,
        ](gpu.lstm_params_buf.unsafe_ptr())
        var lstm_grads_v_b = LayoutTensor[
            dtype,
            Layout.row_major(_LSTMHead.PARAM_SIZE),
            MutAnyOrigin,
        ](gpu.lstm_grads_buf.unsafe_ptr())

        for kk in range(K):
            var k = K - 1 - kk
            var reset_now_b = (k > 0) and (k % LSTM_HORIZON == 0)
            if reset_now_b:
                ctx.enqueue_memset(gpu.lstm_h_input_buf, 0)
                ctx.enqueue_memset(gpu.lstm_c_input_buf, 0)
            else:
                # Re-fill the input scratch with h_lstm[k] / c_lstm[k]
                # (same values used during forward at this step).
                var hs_k_flat_b = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH * LSTM_HIDDEN),
                    MutAnyOrigin,
                ](
                    gpu.lstm_h_states_buf.unsafe_ptr()
                    + k * BATCH * LSTM_HIDDEN
                )
                var cs_k_flat_b = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH * LSTM_HIDDEN),
                    MutAnyOrigin,
                ](
                    gpu.lstm_c_states_buf.unsafe_ptr()
                    + k * BATCH * LSTM_HIDDEN
                )
                var h_in_flat_b = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH * LSTM_HIDDEN),
                    MutAnyOrigin,
                ](gpu.lstm_h_input_buf.unsafe_ptr())
                var c_in_flat_b = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH * LSTM_HIDDEN),
                    MutAnyOrigin,
                ](gpu.lstm_c_input_buf.unsafe_ptr())
                ctx.enqueue_function[copy_lstm_input](
                    hs_k_flat_b,
                    cs_k_flat_b,
                    h_in_flat_b,
                    c_in_flat_b,
                    grid_dim=(LSTM_HIDDEN_BLOCKS,),
                    block_dim=(TPB,),
                )

            var dh_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, LSTM_HIDDEN),
                MutAnyOrigin,
            ](
                gpu.grad_h_lstm_buf.unsafe_ptr()
                + (k + 1) * BATCH * LSTM_HIDDEN
            )
            var dc_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, LSTM_HIDDEN),
                MutAnyOrigin,
            ](
                gpu.grad_c_lstm_buf.unsafe_ptr()
                + (k + 1) * BATCH * LSTM_HIDDEN
            )
            var z_t_b = LayoutTensor[
                dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
            ](gpu.hidden_buf.unsafe_ptr() + (k + 1) * BATCH * LATENT)
            var h_prev_t_b = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, LSTM_HIDDEN),
                MutAnyOrigin,
            ](gpu.lstm_h_input_buf.unsafe_ptr())
            var c_prev_t_b = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, LSTM_HIDDEN),
                MutAnyOrigin,
            ](gpu.lstm_c_input_buf.unsafe_ptr())
            var lstm_cache_t_bw = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, _LSTM_CS),
                MutAnyOrigin,
            ](gpu.lstm_caches_buf.unsafe_ptr() + k * BATCH * _LSTM_CS)
            var grad_x_t = LayoutTensor[
                dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
            ](gpu.grad_x_lstm_buf.unsafe_ptr())
            var grad_h_prev_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, LSTM_HIDDEN),
                MutAnyOrigin,
            ](gpu.grad_h_prev_lstm_buf.unsafe_ptr())
            var grad_c_prev_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, LSTM_HIDDEN),
                MutAnyOrigin,
            ](gpu.grad_c_prev_lstm_buf.unsafe_ptr())
            var d_combined_ws_t = LayoutTensor[
                dtype,
                Layout.row_major(BATCH, 4 * LSTM_HIDDEN),
                MutAnyOrigin,
            ](gpu.lstm_d_combined_ws_buf.unsafe_ptr())
            _LSTMHead.step_backward_gpu[BATCH](
                ctx,
                dh_t,
                dc_t,
                z_t_b,
                h_prev_t_b,
                c_prev_t_b,
                lstm_params_v_b,
                lstm_cache_t_bw,
                grad_x_t,
                grad_h_prev_t,
                grad_c_prev_t,
                lstm_grads_v_b,
                d_combined_ws_t,
            )

            # Accumulate grad_x → grad_hidden[k+1].
            var grad_hidden_kp1_flat_lstm = LayoutTensor[
                dtype,
                Layout.row_major(BATCH * LATENT),
                MutAnyOrigin,
            ](
                gpu.grad_hidden_buf.unsafe_ptr()
                + (k + 1) * BATCH * LATENT
            )
            var grad_x_lstm_flat = LayoutTensor[
                dtype,
                Layout.row_major(BATCH * LATENT),
                MutAnyOrigin,
            ](gpu.grad_x_lstm_buf.unsafe_ptr())
            ctx.enqueue_function[add_kernel_latent](
                grad_hidden_kp1_flat_lstm,
                grad_x_lstm_flat,
                grid_dim=(LATENT_BLOCKS,),
                block_dim=(TPB,),
            )

            # Thread dh_prev / dc_prev → grad_h_lstm[k] / grad_c_lstm[k]
            # unless this step was a reset boundary (input was zero).
            if not reset_now_b:
                var grad_h_lstm_k_flat = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH * LSTM_HIDDEN),
                    MutAnyOrigin,
                ](
                    gpu.grad_h_lstm_buf.unsafe_ptr()
                    + k * BATCH * LSTM_HIDDEN
                )
                var grad_c_lstm_k_flat = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH * LSTM_HIDDEN),
                    MutAnyOrigin,
                ](
                    gpu.grad_c_lstm_buf.unsafe_ptr()
                    + k * BATCH * LSTM_HIDDEN
                )
                var grad_h_prev_flat = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH * LSTM_HIDDEN),
                    MutAnyOrigin,
                ](gpu.grad_h_prev_lstm_buf.unsafe_ptr())
                var grad_c_prev_flat = LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH * LSTM_HIDDEN),
                    MutAnyOrigin,
                ](gpu.grad_c_prev_lstm_buf.unsafe_ptr())
                ctx.enqueue_function[add_kernel_lstm_h](
                    grad_h_lstm_k_flat,
                    grad_h_prev_flat,
                    grid_dim=(LSTM_HIDDEN_BLOCKS,),
                    block_dim=(TPB,),
                )
                ctx.enqueue_function[add_kernel_lstm_h](
                    grad_c_lstm_k_flat,
                    grad_c_prev_flat,
                    grid_dim=(LSTM_HIDDEN_BLOCKS,),
                    block_dim=(TPB,),
                )

    # 6.3 dyn backward in REVERSE topological order: kk = K-1..0.
    # Walks BACKWARD in time so grad_hidden[k+1] is fully accumulated
    # (pred + projector + later dyn-backwards) before we consume it
    # at the current step.
    comptime assemble_dyn_grad = (
        ezv2_assemble_grad_dyn_step_kernel[BATCH, LATENT, BINS, dtype]
    )
    comptime accum_dyn_grad_in = (
        ezv2_accumulate_dyn_grad_in_kernel[BATCH, LATENT, ACT, dtype]
    )
    for kk in range(K):
        var k = K - 1 - kk
        # Build grad_dyn_out_step[b] = [grad_hidden[k+1, b] || grad_dyn_out[k, b, LATENT:]]
        var grad_hidden_kp1_flat = LayoutTensor[
            dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
        ](
            gpu.grad_hidden_buf.unsafe_ptr() + (k + 1) * BATCH * LATENT
        )
        var grad_dyn_out_k_full = LayoutTensor[
            dtype, Layout.row_major(BATCH * DYN_OUT), MutAnyOrigin
        ](gpu.grad_dyn_out_buf.unsafe_ptr() + k * BATCH * DYN_OUT)
        var grad_dyn_out_step_t = LayoutTensor[
            dtype, Layout.row_major(BATCH * DYN_OUT), MutAnyOrigin
        ](gpu.grad_dyn_out_step_buf.unsafe_ptr())
        ctx.enqueue_function[assemble_dyn_grad](
            grad_hidden_kp1_flat,
            grad_dyn_out_k_full,
            grad_dyn_out_step_t,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # dyn.backward(grad_dyn_out_step → grad_dyn_in_step)
        var grad_dyn_out_step_t_in = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Config.DynModel.OUT_DIM),
            MutAnyOrigin,
        ](gpu.grad_dyn_out_step_buf.unsafe_ptr())
        var grad_dyn_in_step_t_in = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Config.DynModel.IN_DIM),
            MutAnyOrigin,
        ](gpu.grad_dyn_in_step_buf.unsafe_ptr())
        var dyn_cache_k_r = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, Config.DynModel.CACHE_SIZE),
            MutAnyOrigin,
        ](gpu.dyn_caches_buf.unsafe_ptr() + k * BATCH * DYN_CS)
        var dyn_grads_v = gpu.dynamics.grads_view()
        Network[
            Config.DynModel, Config.OptType
        ].backward_gpu[BATCH](
            ctx,
            grad_dyn_out_step_t_in,
            grad_dyn_in_step_t_in,
            gpu.dynamics.params_view(),
            gpu.dynamics.model_state_view(),
            dyn_cache_k_r,
            dyn_grads_v,
            gpu.workspace_buf,
        )

        # grad_hidden[k] += grad_dyn_in_step[:, :LATENT]
        var grad_dyn_in_step_flat = LayoutTensor[
            dtype, Layout.row_major(BATCH * DYN_IN), MutAnyOrigin
        ](gpu.grad_dyn_in_step_buf.unsafe_ptr())
        var grad_hidden_k_flat = LayoutTensor[
            dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
        ](gpu.grad_hidden_buf.unsafe_ptr() + k * BATCH * LATENT)
        ctx.enqueue_function[accum_dyn_grad_in](
            grad_dyn_in_step_flat,
            grad_hidden_k_flat,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # Residual skip-path gradient: hidden[k+1] = dyn(hidden[k]) + hidden[k]
        # → ∂L/∂hidden[k] picks up an extra +∂L/∂hidden[k+1] term beyond the
        # through-dyn path handled above. `grad_hidden[k+1]` is fully
        # accumulated by this point (used as input to `assemble_dyn_grad`
        # at the top of this iteration; not read by any subsequent step in
        # the reverse loop since k decreases), so it's safe to read here.
        var grad_hidden_kp1_skip = LayoutTensor[
            dtype, Layout.row_major(BATCH * LATENT), MutAnyOrigin
        ](gpu.grad_hidden_buf.unsafe_ptr() + (k + 1) * BATCH * LATENT)
        ctx.enqueue_function[add_kernel_lat](
            grad_hidden_k_flat,
            grad_hidden_kp1_skip,
            grid_dim=(LATENT_BLOCKS,),
            block_dim=(TPB,),
        )

    # 6.4 rep backward at k=0 (only — the target-branch rep forwards
    # at k=1..K were stop-grad / no-cache, so no gradients flow
    # through them).
    var grad_hidden_0_in = LayoutTensor[
        dtype,
        Layout.row_major(BATCH, Config.RepModel.OUT_DIM),
        MutAnyOrigin,
    ](gpu.grad_hidden_buf.unsafe_ptr())
    var grad_rep_in_t = LayoutTensor[
        dtype,
        Layout.row_major(BATCH, Config.RepModel.IN_DIM),
        MutAnyOrigin,
    ](gpu.grad_rep_in_buf.unsafe_ptr())
    var rep_grads_v = gpu.representation.grads_view()
    Network[
        Config.RepModel, Config.OptType
    ].backward_gpu[BATCH](
        ctx,
        grad_hidden_0_in,
        grad_rep_in_t,
        gpu.representation.params_view(),
        gpu.representation.model_state_view(),
        rep_cache_t_w,
        rep_grads_v,
        gpu.workspace_buf,
    )

    # ── 7. Optimizer step on every network. ─────────────────────────
    # Per-network gradient clipping (paper: max_grad_norm=5.0). Same
    # threshold applied independently per network — same pattern as
    # `offpolicy_agent`. Reuses `gpu.grad_clip_ps` sequentially.
    if max_grad_norm > 0.0:
        comptime _CLIP_TPB: Int = TPB

        comptime _R_PS = Config.RepModel.PARAM_SIZE
        comptime _R_BLK = (_R_PS + _CLIP_TPB - 1) // _CLIP_TPB
        comptime _r_norm_k = gradient_norm_kernel[
            dtype, _R_PS, _R_BLK, _CLIP_TPB
        ]
        comptime _r_clip_k = gradient_reduce_apply_fused_kernel[
            dtype, _R_PS, _R_BLK, _CLIP_TPB
        ]
        var _r_ps_t = LayoutTensor[
            dtype, Layout.row_major(_R_BLK), MutAnyOrigin
        ](gpu.grad_clip_ps.unsafe_ptr())
        var _r_g = gpu.representation.grads_view()
        ctx.enqueue_function[_r_norm_k](
            _r_ps_t, _r_g, grid_dim=(_R_BLK,), block_dim=(_CLIP_TPB,)
        )
        ctx.enqueue_function[_r_clip_k](
            _r_g,
            _r_ps_t,
            Scalar[dtype](max_grad_norm),
            grid_dim=(_R_BLK,),
            block_dim=(_CLIP_TPB,),
        )

        comptime _D_PS = Config.DynModel.PARAM_SIZE
        comptime _D_BLK = (_D_PS + _CLIP_TPB - 1) // _CLIP_TPB
        comptime _d_norm_k = gradient_norm_kernel[
            dtype, _D_PS, _D_BLK, _CLIP_TPB
        ]
        comptime _d_clip_k = gradient_reduce_apply_fused_kernel[
            dtype, _D_PS, _D_BLK, _CLIP_TPB
        ]
        var _d_ps_t = LayoutTensor[
            dtype, Layout.row_major(_D_BLK), MutAnyOrigin
        ](gpu.grad_clip_ps.unsafe_ptr())
        var _d_g = gpu.dynamics.grads_view()
        ctx.enqueue_function[_d_norm_k](
            _d_ps_t, _d_g, grid_dim=(_D_BLK,), block_dim=(_CLIP_TPB,)
        )
        ctx.enqueue_function[_d_clip_k](
            _d_g,
            _d_ps_t,
            Scalar[dtype](max_grad_norm),
            grid_dim=(_D_BLK,),
            block_dim=(_CLIP_TPB,),
        )

        comptime _P_PS = Config.PredModel.PARAM_SIZE
        comptime _P_BLK = (_P_PS + _CLIP_TPB - 1) // _CLIP_TPB
        comptime _p_norm_k = gradient_norm_kernel[
            dtype, _P_PS, _P_BLK, _CLIP_TPB
        ]
        comptime _p_clip_k = gradient_reduce_apply_fused_kernel[
            dtype, _P_PS, _P_BLK, _CLIP_TPB
        ]
        var _p_ps_t = LayoutTensor[
            dtype, Layout.row_major(_P_BLK), MutAnyOrigin
        ](gpu.grad_clip_ps.unsafe_ptr())
        var _p_g = gpu.prediction.grads_view()
        ctx.enqueue_function[_p_norm_k](
            _p_ps_t, _p_g, grid_dim=(_P_BLK,), block_dim=(_CLIP_TPB,)
        )
        ctx.enqueue_function[_p_clip_k](
            _p_g,
            _p_ps_t,
            Scalar[dtype](max_grad_norm),
            grid_dim=(_P_BLK,),
            block_dim=(_CLIP_TPB,),
        )

        comptime _PJ_PS = Config.ProjectorModel.PARAM_SIZE
        comptime _PJ_BLK = (_PJ_PS + _CLIP_TPB - 1) // _CLIP_TPB
        comptime _pj_norm_k = gradient_norm_kernel[
            dtype, _PJ_PS, _PJ_BLK, _CLIP_TPB
        ]
        comptime _pj_clip_k = gradient_reduce_apply_fused_kernel[
            dtype, _PJ_PS, _PJ_BLK, _CLIP_TPB
        ]
        var _pj_ps_t = LayoutTensor[
            dtype, Layout.row_major(_PJ_BLK), MutAnyOrigin
        ](gpu.grad_clip_ps.unsafe_ptr())
        var _pj_g = gpu.projector.grads_view()
        ctx.enqueue_function[_pj_norm_k](
            _pj_ps_t, _pj_g, grid_dim=(_PJ_BLK,), block_dim=(_CLIP_TPB,)
        )
        ctx.enqueue_function[_pj_clip_k](
            _pj_g,
            _pj_ps_t,
            Scalar[dtype](max_grad_norm),
            grid_dim=(_PJ_BLK,),
            block_dim=(_CLIP_TPB,),
        )

        comptime _PR_PS = Config.PredictorModel.PARAM_SIZE
        comptime _PR_BLK = (_PR_PS + _CLIP_TPB - 1) // _CLIP_TPB
        comptime _pr_norm_k = gradient_norm_kernel[
            dtype, _PR_PS, _PR_BLK, _CLIP_TPB
        ]
        comptime _pr_clip_k = gradient_reduce_apply_fused_kernel[
            dtype, _PR_PS, _PR_BLK, _CLIP_TPB
        ]
        var _pr_ps_t = LayoutTensor[
            dtype, Layout.row_major(_PR_BLK), MutAnyOrigin
        ](gpu.grad_clip_ps.unsafe_ptr())
        var _pr_g = gpu.predictor.grads_view()
        ctx.enqueue_function[_pr_norm_k](
            _pr_ps_t, _pr_g, grid_dim=(_PR_BLK,), block_dim=(_CLIP_TPB,)
        )
        ctx.enqueue_function[_pr_clip_k](
            _pr_g,
            _pr_ps_t,
            Scalar[dtype](max_grad_norm),
            grid_dim=(_PR_BLK,),
            block_dim=(_CLIP_TPB,),
        )

    gpu.representation.optimizer_step(ctx)
    gpu.dynamics.optimizer_step(ctx)
    gpu.prediction.optimizer_step(ctx)
    gpu.projector.optimizer_step(ctx)
    gpu.predictor.optimizer_step(ctx)

    comptime if Config.use_reward_prefix:
        # LSTM Adam step (the cell isn't in a GPUNetworkState so we
        # call OptType.step_gpu directly with the device buffers).
        var lstm_params_v_o = LayoutTensor[
            dtype,
            Layout.row_major(_LSTMHead.PARAM_SIZE),
            MutAnyOrigin,
        ](gpu.lstm_params_buf.unsafe_ptr())
        var lstm_grads_v_o = LayoutTensor[
            dtype,
            Layout.row_major(_LSTMHead.PARAM_SIZE),
            MutAnyOrigin,
        ](gpu.lstm_grads_buf.unsafe_ptr())
        var lstm_opt_state_v_o = LayoutTensor[
            dtype,
            Layout.row_major(
                _LSTMHead.PARAM_SIZE,
                Config.OptType.STATE_PER_PARAM,
            ),
            MutAnyOrigin,
        ](gpu.lstm_opt_state_buf.unsafe_ptr())
        var lstm_opt_global_v_o = LayoutTensor[
            dtype,
            Layout.row_major(Config.OptType.GLOBAL_STATE_SIZE),
            MutAnyOrigin,
        ](gpu.lstm_opt_global_buf.unsafe_ptr())
        gpu.lstm_step_num += 1
        Config.OptType.step_gpu[_LSTMHead.PARAM_SIZE](
            ctx,
            lstm_params_v_o,
            lstm_grads_v_o,
            lstm_opt_state_v_o,
            lstm_opt_global_v_o,
            gpu.lstm_step_num,
        )

        # MLP head Adam step.
        gpu.reward_prefix_mlp_gpu.optimizer_step(ctx)


    # ── 8. Priority refresh — kernel writes priorities_out_buf[b] =
    #       per_sample_v_loss_k0[b] + 1e-3, downloaded below.
    comptime prio_kernel = ezv2_priority_from_v_loss_kernel[
        BATCH, dtype
    ]
    var priorities_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH), MutAnyOrigin
    ](gpu.priorities_out_buf.unsafe_ptr())
    ctx.enqueue_function[prio_kernel](
        per_sample_v_loss_k0_t,
        priorities_out_t,
        grid_dim=(BATCH_BLOCKS,),
        block_dim=(TPB,),
    )

    # ── 9. Download losses + per-sample priorities ───────────────────
    ctx.enqueue_copy(gpu.L_R_host, gpu.L_R_buf)
    ctx.enqueue_copy(gpu.L_P_host, gpu.L_P_buf)
    ctx.enqueue_copy(gpu.L_V_host, gpu.L_V_buf)
    ctx.enqueue_copy(gpu.L_G_host, gpu.L_G_buf)
    ctx.enqueue_copy(gpu.priorities_out_host, gpu.priorities_out_buf)
    ctx.synchronize()

    var L_R_sum = Float64(gpu.L_R_host[0])
    var L_P_sum = Float64(gpu.L_P_host[0])
    var L_V_sum = Float64(gpu.L_V_host[0])
    var L_G_sum = Float64(gpu.L_G_host[0])

    var L_R = L_R_sum / n_R if n_R > 0.0 else 0.0
    var L_P = L_P_sum / n_P if n_P > 0.0 else 0.0
    var L_V = L_V_sum / n_V if n_V > 0.0 else 0.0
    var L_G = L_G_sum / n_G if n_G > 0.0 else 0.0

    return (L_R, L_P, L_V, L_G)
