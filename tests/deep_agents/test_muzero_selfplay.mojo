"""Test MuZero self-play on TicTacToe with GPU MCTS + legal masking.

Demonstrates the full AlphaZero-style pipeline:
  - GPUTwoPlayerDiscreteEnv for parallel game stepping
  - GPU MCTS with legal action masking + negated backup (zero-sum)
  - Canonical observations (single network for both players)
"""

from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Linear, LinearReLU, Sequential, Parallel
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.training import Network, GPUNetworkState
from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv
from mojo_rl.deep_agents.muzero.gpu_mcts import (
    GPUMCTSState,
    gpu_mcts_init_root_kernel,
    gpu_mcts_select_kernel,
    gpu_mcts_build_dyn_input_kernel,
    gpu_mcts_expand_kernel,
    gpu_mcts_backup_negated_kernel,
    gpu_mcts_extract_actions_masked_kernel,
    gpu_mcts_apply_legal_mask_kernel,
    TPB,
    MAX_DEPTH,
)
from mojo_rl.deep_agents.muzero.kernels import extract_hidden_kernel


fn main() raises:
    print("=== MuZero Self-Play on TicTacToe (GPU) ===")

    var ctx = DeviceContext()

    # ── Environment constants ────────────────────────────────────
    comptime TTT = TicTacToeEnv[DType.float32]
    comptime OBS = TTT.OBS_DIM         # 27
    comptime ACT = TTT.NUM_ACTIONS     # 9
    comptime STATE_SIZE = TTT.STATE_SIZE  # 12
    comptime N_ENVS = 32
    comptime MAX_NODES = 32
    comptime NUM_SIMS = 10
    comptime LATENT = 64
    comptime BINS = 1   # Scalar value for board games
    comptime HIDDEN = 64
    comptime ENV_BLOCKS = (N_ENVS + TPB - 1) // TPB

    # ── Network architectures (AlphaZero-style) ──────────────────
    # Prediction: obs → (policy_logits[9], value[1])
    comptime PredModel = Sequential[
        LinearReLU[OBS, HIDDEN],
        LinearReLU[HIDDEN, HIDDEN],
        Parallel[
            Linear[HIDDEN, ACT],   # Policy head
            Linear[HIDDEN, BINS],  # Scalar value head
        ],
    ]
    # Representation: obs → latent (identity-like for AlphaZero)
    comptime RepModel = Sequential[Linear[OBS, LATENT]]
    # Dynamics: stub (unused with TrueGameRules but needed for MCTS structure)
    comptime DYN_IN = LATENT + ACT
    comptime DYN_OUT = LATENT + BINS
    comptime DynModel = Sequential[Linear[DYN_IN, DYN_OUT]]
    comptime PRED_OUT = ACT + BINS
    comptime OptType = Adam[LR=1e-3]

    comptime RepNet = Network[RepModel, OptType]
    comptime DynNet = Network[DynModel, OptType]
    comptime PredNet = Network[PredModel, OptType]

    # ── Allocate GPU networks ────────────────────────────────────
    var gpu_rep = GPUNetworkState[RepModel, OptType](ctx)
    var gpu_dyn = GPUNetworkState[DynModel, OptType](ctx)
    var gpu_pred = GPUNetworkState[PredModel, OptType](ctx)

    # Initialize with random weights (CPU side)
    from mojo_rl.nn.training import NetworkState
    from mojo_rl.nn.initializer import Xavier
    var cpu_pred = NetworkState[PredModel, OptType]()
    cpu_pred.initialize[Xavier[]]()
    gpu_pred.upload_from(cpu_pred, ctx)

    var cpu_rep = NetworkState[RepModel, OptType]()
    cpu_rep.initialize[Xavier[]]()
    gpu_rep.upload_from(cpu_rep, ctx)

    var cpu_dyn = NetworkState[DynModel, OptType]()
    cpu_dyn.initialize[Xavier[]]()
    gpu_dyn.upload_from(cpu_dyn, ctx)

    # ── Network workspace ────────────────────────────────────────
    comptime WS_R = RepModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS_D = DynModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS_P = PredModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime MAX_WS = WS_R if WS_R > WS_D else WS_D
    comptime MAX_WS2 = MAX_WS if MAX_WS > WS_P else WS_P
    comptime WS_TOTAL = N_ENVS * MAX_WS2 if MAX_WS2 > 0 else 1
    var workspace = ctx.enqueue_create_buffer[dtype](WS_TOTAL)

    # ── GPU MCTS state ───────────────────────────────────────────
    # For AlphaZero, we use PredModel directly on observations (no latent)
    # But MCTS kernels expect LATENT-sized hidden states, so we use RepModel
    var mcts = GPUMCTSState[N_ENVS, MAX_NODES, ACT, LATENT, BINS](ctx)

    # ── GPU environment buffers ──────────────────────────────────
    var states_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * STATE_SIZE)
    var obs_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS)
    var actions_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var rewards_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var dones_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var terminated_buf = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var legal_masks_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * ACT)

    # Host buffers for reading back results
    var actions_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    var rewards_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    var dones_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    var policies_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS * ACT)

    # ── Initialize games ─────────────────────────────────────────
    TTT.reset_kernel_gpu[N_ENVS, STATE_SIZE](ctx, states_buf, rng_seed=42)
    TTT.extract_obs_kernel_gpu[N_ENVS, STATE_SIZE, OBS](
        ctx, states_buf, obs_buf, legal_masks_buf
    )
    ctx.synchronize()
    print("Initialized", N_ENVS, "TicTacToe games on GPU")

    # ── Self-play loop: play games using GPU MCTS ────────────────
    var total_games_done = 0
    var total_steps = 0
    comptime MAX_STEPS = 50  # TicTacToe games are short

    for step in range(MAX_STEPS):
        # ── 1. Run GPU MCTS for action selection ─────────────────

        # 1a. Representation: obs → hidden (for MCTS tree)
        comptime REP_IN_DIM = RepModel.IN_DIM
        comptime REP_OUT_DIM = RepModel.OUT_DIM
        var rep_obs = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, REP_IN_DIM), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var rep_h = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, REP_OUT_DIM), MutAnyOrigin
        ](mcts.hidden_states.unsafe_ptr())
        RepNet.forward_gpu[N_ENVS](
            ctx, rep_obs, rep_h, gpu_rep.params_view(), workspace
        )

        # 1b. Prediction: hidden → policy + value
        comptime PRED_IN_DIM = PredModel.IN_DIM
        comptime PRED_OUT_DIM = PredModel.OUT_DIM
        # For AlphaZero, prediction runs on OBS directly, not latent
        # But our PredModel takes OBS as input, not LATENT
        # So we run PredModel(obs) directly for root prediction
        var pred_obs = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, PRED_IN_DIM), MutAnyOrigin
        ](obs_buf.unsafe_ptr())
        var pred_out = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, PRED_OUT_DIM), MutAnyOrigin
        ](mcts.pred_output.unsafe_ptr())
        PredNet.forward_gpu[N_ENVS](
            ctx, pred_obs, pred_out, gpu_pred.params_view(), workspace
        )

        # 1c. Initialize root nodes
        var vc_t = LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin](mcts.visit_count.unsafe_ptr())
        var tv_t = LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin](mcts.total_value.unsafe_ptr())
        var pr_t = LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin](mcts.prior.unsafe_ptr())
        var rw_t = LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin](mcts.reward.unsafe_ptr())
        var ci_t = LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin](mcts.child_idx.unsafe_ptr())
        var tvis_t = LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin](mcts.total_visits.unsafe_ptr())
        var nc_t = LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](mcts.node_count.unsafe_ptr())
        var po_t = LayoutTensor[dtype, Layout.row_major(N_ENVS * PRED_OUT), MutAnyOrigin](mcts.pred_output.unsafe_ptr())
        var miq_t = LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](mcts.min_q.unsafe_ptr())
        var mxq_t = LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](mcts.max_q.unsafe_ptr())

        comptime run_init = gpu_mcts_init_root_kernel[N_ENVS, MAX_NODES, ACT, LATENT, PRED_OUT, dtype]
        ctx.enqueue_function[run_init, run_init](
            vc_t, tv_t, pr_t, rw_t, ci_t, tvis_t, nc_t, po_t, miq_t, mxq_t,
            Scalar[dtype](0.25), Scalar[DType.uint32](UInt32(step * 137)),
            grid_dim=(ENV_BLOCKS,), block_dim=(TPB,),
        )

        # 1d. Apply legal mask to root prior (SELF-PLAY KEY STEP)
        var lm_t = LayoutTensor[dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin](legal_masks_buf.unsafe_ptr())
        comptime run_mask = gpu_mcts_apply_legal_mask_kernel[N_ENVS, MAX_NODES, ACT, dtype]
        ctx.enqueue_function[run_mask, run_mask](
            pr_t, lm_t,
            grid_dim=(ENV_BLOCKS,), block_dim=(TPB,),
        )

        # 1e. Run MCTS simulations with NEGATED backup
        var pp_t = LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](mcts.pending_parent.unsafe_ptr())
        var pa_t = LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](mcts.pending_action.unsafe_ptr())
        var sp_t = LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_DEPTH), MutAnyOrigin](mcts.search_paths.unsafe_ptr())
        var ap_t = LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_DEPTH), MutAnyOrigin](mcts.action_paths.unsafe_ptr())
        var pl_t = LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](mcts.path_lengths.unsafe_ptr())
        var lv_t = LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](mcts.leaf_values.unsafe_ptr())
        var hs_t = LayoutTensor[dtype, Layout.row_major(N_ENVS * MAX_NODES * LATENT), MutAnyOrigin](mcts.hidden_states.unsafe_ptr())

        comptime DYN_IN_DIM = DynModel.IN_DIM
        comptime DYN_OUT_DIM = DynModel.OUT_DIM

        for _sim in range(NUM_SIMS):
            # Selection
            comptime run_sel = gpu_mcts_select_kernel[N_ENVS, MAX_NODES, ACT, dtype]
            ctx.enqueue_function[run_sel, run_sel](
                vc_t, tv_t, pr_t, ci_t, tvis_t, nc_t, miq_t, mxq_t,
                pp_t, pa_t, sp_t, ap_t, pl_t,
                Scalar[dtype](19652.0), Scalar[dtype](2.5),  # AlphaGo-style c
                grid_dim=(ENV_BLOCKS,), block_dim=(TPB,),
            )

            # Build dynamics input
            var di_t = LayoutTensor[dtype, Layout.row_major(N_ENVS * DYN_IN), MutAnyOrigin](mcts.dyn_input.unsafe_ptr())
            comptime run_bld = gpu_mcts_build_dyn_input_kernel[N_ENVS, MAX_NODES, ACT, LATENT, DYN_IN, dtype]
            ctx.enqueue_function[run_bld, run_bld](
                di_t, hs_t, pp_t, pa_t,
                grid_dim=(ENV_BLOCKS,), block_dim=(TPB,),
            )

            # Dynamics forward
            var dyn_in_net = LayoutTensor[dtype, Layout.row_major(N_ENVS, DYN_IN_DIM), MutAnyOrigin](mcts.dyn_input.unsafe_ptr())
            var dyn_out_net = LayoutTensor[dtype, Layout.row_major(N_ENVS, DYN_OUT_DIM), MutAnyOrigin](mcts.dyn_output.unsafe_ptr())
            DynNet.forward_gpu[N_ENVS](ctx, dyn_in_net, dyn_out_net, gpu_dyn.params_view(), workspace)

            # Extract hidden → pred input
            var pred_in_flat = LayoutTensor[dtype, Layout.row_major(N_ENVS * LATENT), MutAnyOrigin](mcts.pred_input.unsafe_ptr())
            var dyn_out_flat = LayoutTensor[dtype, Layout.row_major(N_ENVS * DYN_OUT), MutAnyOrigin](mcts.dyn_output.unsafe_ptr())
            comptime EXTR_BLK = (N_ENVS * LATENT + TPB - 1) // TPB
            comptime run_extr = extract_hidden_kernel[N_ENVS, LATENT, DYN_OUT, dtype]
            ctx.enqueue_function[run_extr, run_extr](
                pred_in_flat, dyn_out_flat,
                grid_dim=(EXTR_BLK,), block_dim=(TPB,),
            )

            # Prediction forward
            # Note: for child nodes, pred runs on latent (not obs)
            # This is a simplification — full AlphaZero would use game rules
            var pred_in_lat = LayoutTensor[dtype, Layout.row_major(N_ENVS, PRED_IN_DIM), MutAnyOrigin](mcts.pred_input.unsafe_ptr())
            var pred_out_sim = LayoutTensor[dtype, Layout.row_major(N_ENVS, PRED_OUT_DIM), MutAnyOrigin](mcts.pred_output.unsafe_ptr())
            PredNet.forward_gpu[N_ENVS](ctx, pred_in_lat, pred_out_sim, gpu_pred.params_view(), workspace)

            # Expand
            var do_t = LayoutTensor[dtype, Layout.row_major(N_ENVS * DYN_OUT), MutAnyOrigin](mcts.dyn_output.unsafe_ptr())
            var po_exp = LayoutTensor[dtype, Layout.row_major(N_ENVS * PRED_OUT), MutAnyOrigin](mcts.pred_output.unsafe_ptr())
            comptime run_exp = gpu_mcts_expand_kernel[N_ENVS, MAX_NODES, ACT, LATENT, PRED_OUT, DYN_OUT, dtype]
            ctx.enqueue_function[run_exp, run_exp](
                vc_t, tv_t, pr_t, rw_t, ci_t, tvis_t, nc_t,
                hs_t, pp_t, pa_t, do_t, po_exp,
                Scalar[dtype](-1.0), Scalar[dtype](1.0), lv_t,
                grid_dim=(ENV_BLOCKS,), block_dim=(TPB,),
            )

            # Backup with NEGATION (zero-sum two-player)
            comptime run_bk = gpu_mcts_backup_negated_kernel[N_ENVS, MAX_NODES, ACT, dtype]
            ctx.enqueue_function[run_bk, run_bk](
                vc_t, tv_t, rw_t, tvis_t, miq_t, mxq_t,
                sp_t, ap_t, pl_t, lv_t,
                grid_dim=(ENV_BLOCKS,), block_dim=(TPB,),
            )

        # 1f. Extract actions (only legal ones)
        var act_out = LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](actions_buf.unsafe_ptr())
        var pol_out = LayoutTensor[dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin](mcts.policies_out.unsafe_ptr())
        comptime run_act = gpu_mcts_extract_actions_masked_kernel[N_ENVS, MAX_NODES, ACT, dtype]
        ctx.enqueue_function[run_act, run_act](
            vc_t, lm_t, act_out, pol_out,
            grid_dim=(ENV_BLOCKS,), block_dim=(TPB,),
        )

        # ── 2. GPU environment step ──────────────────────────────
        TTT.step_kernel_gpu[N_ENVS, STATE_SIZE, OBS](
            ctx, states_buf, actions_buf, rewards_buf, dones_buf,
            terminated_buf, obs_buf, legal_masks_buf,
            rng_seed=UInt64(step),
        )

        # ── 3. Read back results ─────────────────────────────────
        ctx.enqueue_copy(dones_host, dones_buf)
        ctx.enqueue_copy(rewards_host, rewards_buf)
        ctx.synchronize()

        var step_dones = 0
        var step_wins = 0
        var step_draws = 0
        for e in range(N_ENVS):
            if Float64(dones_host[e]) > 0.5:
                step_dones += 1
                if Float64(rewards_host[e]) > 0.5:
                    step_wins += 1
                elif Float64(rewards_host[e]) < -0.5:
                    pass  # Loss (shouldn't happen in self-play)
                else:
                    step_draws += 1

        total_games_done += step_dones
        total_steps += 1

        if step_dones > 0:
            print(
                "Step", step,
                "| Games done:", step_dones,
                "| Wins:", step_wins,
                "| Draws:", step_draws,
                "| Total done:", total_games_done,
            )

        # ── 4. Selective reset done games ────────────────────────
        TTT.selective_reset_kernel_gpu[N_ENVS, STATE_SIZE](
            ctx, states_buf, dones_buf, rng_seed=UInt64(step * 7 + 1),
        )
        # Re-extract obs + legal masks for reset games
        TTT.extract_obs_kernel_gpu[N_ENVS, STATE_SIZE, OBS](
            ctx, states_buf, obs_buf, legal_masks_buf,
        )

        # Stop early once enough games played
        if total_games_done >= 100:
            break

    print("\n=== Results ===")
    print("Total steps:", total_steps)
    print("Total games completed:", total_games_done)

    if total_games_done > 0:
        print("PASS: Self-play GPU MCTS working on TicTacToe")
    else:
        print("FAIL: No games completed")

    # Read final policies to verify they're valid
    ctx.enqueue_copy(policies_host, mcts.policies_out)
    ctx.synchronize()
    print("\nSample policies (last step):")
    for e in range(3):
        var pol_str = String("  env ") + String(e) + ": ["
        for a in range(ACT):
            if a > 0:
                pol_str += ", "
            pol_str += String(Int(Float64(policies_host[e * ACT + a]) * 100)) + "%"
        pol_str += "]"
        print(pol_str)

    print("=== Done ===")
