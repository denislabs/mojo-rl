"""Inspect raw MCTS state after search — dump visit counts, priors, node count.

Sets up a single MCTS search from the empty TicTacToe board, then downloads
and prints all internal MCTS data to find where the bug is.
"""

from std.memory import alloc, memset
from std.math import abs, sqrt, exp
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Linear, LinearReLU, Sequential, Parallel
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.training import Network, NetworkState, GPUNetworkState
from mojo_rl.deep_agents.alphazero.configs import AlphaZeroConfig
from mojo_rl.planners.tree_search.strategies import (
    DirichletNoise,
    AlphaGoPUCT,
    SelfPlay,
)
from mojo_rl.planners.tree_search.mcts_gpu import (
    GPUMCTSState,
    gpu_mcts_init_root_kernel,
    gpu_mcts_apply_legal_mask_kernel,
    gpu_mcts_copy_root_state_kernel,
    gpu_mcts_batched_select_and_copy_kernel,
    gpu_mcts_batched_expand_backup_kernel,
    TPB,
    MAX_DEPTH,
)
from mojo_rl.envs.board_games.tic_tac_toe import TicTacToeEnv


struct InspectConfig(AlphaZeroConfig):
    comptime NAME: String = "AZ-Inspect"
    comptime obs_dim: Int = 27
    comptime action_dim: Int = 9
    comptime PredModel = Sequential[
        LinearReLU[27, 64],
        LinearReLU[64, 64],
        Parallel[Linear[64, 9], Linear[64, 1]],
    ]
    comptime OptType = Adam[LR=0.001]
    comptime batch_size: Int = 32
    comptime buffer_capacity: Int = 1000
    comptime history_window: Int = 20
    comptime num_simulations: Int = 25
    comptime max_nodes: Int = 64
    comptime temp_threshold: Int = 15
    comptime temp_min: Float64 = 0.0
    comptime batch_sims: Int = 8
    comptime invalid_action_penalty: Float64 = 0.0
    comptime max_grad_norm: Float64 = 0.0
    comptime value_target_q_weight: Float64 = 0.0
    comptime max_episode_length: Int = 9
    comptime board_rows: Int = 3
    comptime board_cols: Int = 3
    comptime board_planes: Int = 3
    comptime num_symmetries: Int = 2
    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = AlphaGoPUCT[1.0]
    comptime Players = SelfPlay


def main() raises:
    print("=== MCTS State Inspector (SharedMemory) ===")
    print()

    var ctx = DeviceContext()

    comptime N_ENVS = 4  # Small number for readability
    comptime ACT = 9
    comptime OBS = 27
    comptime MAX_NODES = 64
    comptime SIMS = 25
    comptime BATCH_SIMS = 8
    comptime NUM_ROUNDS = SIMS // BATCH_SIMS  # 3 rounds
    comptime GS = TicTacToeEnv[DType.float32].STATE_SIZE
    comptime ENV_BLOCKS = (N_ENVS + TPB - 1) // TPB
    comptime PredModel = InspectConfig.PredModel
    comptime OptType = InspectConfig.OptType
    comptime PRED_IN = PredModel.IN_DIM
    comptime PRED_OUT_DIM = PredModel.OUT_DIM
    comptime MCTS_PRED_OUT = ACT + 1
    comptime PredNet = Network[PredModel, OptType]

    # Initialize network
    var net_state = NetworkState[PredModel, OptType]()
    from mojo_rl.nn.initializer import Kaiming

    net_state.initialize[Kaiming[]]()
    var gpu_net = GPUNetworkState[PredModel, OptType](ctx)
    gpu_net.upload_from(net_state, ctx)

    # Setup env
    comptime E = TicTacToeEnv[DType.float32]
    var states_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * GS)
    var obs_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS)
    var legal_masks_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * ACT)

    # Reset all envs to initial state
    E.reset_kernel_gpu[N_ENVS, GS](ctx, states_buf, rng_seed=42)
    E.extract_obs_kernel_gpu[N_ENVS, GS, OBS](
        ctx, states_buf, obs_buf, legal_masks_buf
    )
    ctx.synchronize()

    # Verify obs
    var obs_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS * OBS)
    ctx.enqueue_copy(obs_host, obs_buf)
    var legal_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS * ACT)
    ctx.enqueue_copy(legal_host, legal_masks_buf)
    ctx.synchronize()

    print("--- Legal masks (env 0) ---")
    for a in range(ACT):
        print("  Action", a, ":", legal_host[a])

    # Setup MCTS
    var gpu_mcts = GPUMCTSState[N_ENVS, MAX_NODES, ACT, OBS, 1, GS](ctx)

    comptime WS = PredModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS_SIZE = N_ENVS * BATCH_SIMS * WS if WS > 0 else 1
    var mcts_ws = ctx.enqueue_create_buffer[dtype](WS_SIZE)

    # Run prediction on initial obs
    var pred_obs = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, PRED_IN), MutAnyOrigin
    ](obs_buf.unsafe_ptr())
    var pred_out = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, PRED_OUT_DIM), MutAnyOrigin
    ](gpu_mcts.pred_output.unsafe_ptr())
    PredNet.forward_gpu[N_ENVS](
        ctx, pred_obs, pred_out, gpu_net.params_view(), gpu_net.model_state_view(), mcts_ws
    )

    # Download and print raw prediction
    var pred_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS * PRED_OUT_DIM)
    ctx.enqueue_copy(pred_host, gpu_mcts.pred_output)
    ctx.synchronize()

    print()
    print("--- Raw network output (env 0) ---")
    print("  Policy logits:", end="")
    for a in range(ACT):
        print(" ", pred_host[a], end="")
    print()
    print("  Value raw:", pred_host[ACT])

    # Init root
    var vc = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](gpu_mcts.visit_count.unsafe_ptr())
    var tv = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](gpu_mcts.total_value.unsafe_ptr())
    var pr = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](gpu_mcts.prior.unsafe_ptr())
    var rw = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](gpu_mcts.reward.unsafe_ptr())
    var ci = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](gpu_mcts.child_idx.unsafe_ptr())
    var tvis = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ](gpu_mcts.total_visits.unsafe_ptr())
    var nc = LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](
        gpu_mcts.node_count.unsafe_ptr()
    )
    var po = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MCTS_PRED_OUT), MutAnyOrigin
    ](gpu_mcts.pred_output.unsafe_ptr())
    var miq = LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](
        gpu_mcts.min_q.unsafe_ptr()
    )
    var mxq = LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](
        gpu_mcts.max_q.unsafe_ptr()
    )

    comptime run_init = gpu_mcts_init_root_kernel[
        N_ENVS, MAX_NODES, ACT, OBS, MCTS_PRED_OUT, dtype
    ]
    ctx.enqueue_function[run_init, run_init](
        vc,
        tv,
        pr,
        rw,
        ci,
        tvis,
        nc,
        po,
        miq,
        mxq,
        Scalar[dtype](0.25),  # noise fraction
        Scalar[DType.uint32](UInt32(42)),
        grid_dim=(ENV_BLOCKS,),
        block_dim=(TPB,),
    )

    # Apply legal mask
    var lm = LayoutTensor[dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin](
        legal_masks_buf.unsafe_ptr()
    )
    comptime run_mask = gpu_mcts_apply_legal_mask_kernel[
        N_ENVS, MAX_NODES, ACT, dtype
    ]
    ctx.enqueue_function[run_mask, run_mask](
        pr, lm, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
    )

    # Copy root game states
    var gs = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * GS), MutAnyOrigin
    ](gpu_mcts.game_states.unsafe_ptr())
    var es = LayoutTensor[dtype, Layout.row_major(N_ENVS * GS), MutAnyOrigin](
        states_buf.unsafe_ptr()
    )
    comptime run_rs = gpu_mcts_copy_root_state_kernel[
        N_ENVS, MAX_NODES, GS, dtype
    ]
    ctx.enqueue_function[run_rs, run_rs](
        gs, es, grid_dim=(ENV_BLOCKS,), block_dim=(TPB,)
    )

    # Download and print root prior after masking + noise
    var prior_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * MAX_NODES * ACT
    )
    ctx.enqueue_copy(prior_host, gpu_mcts.prior)
    ctx.synchronize()

    print()
    print("--- Root prior after legal mask + noise (env 0) ---")
    for a in range(ACT):
        print("  Action", a, ": prior =", prior_host[a])

    # Download node count before MCTS
    var nc_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    ctx.enqueue_copy(nc_host, gpu_mcts.node_count)
    ctx.synchronize()
    print()
    print("--- Before MCTS ---")
    print("  Node count (env 0):", nc_host[0])

    # Expansion scratch buffers
    comptime TOTAL_EXPAND = N_ENVS * BATCH_SIMS
    var exp_rewards = ctx.enqueue_create_buffer[dtype](TOTAL_EXPAND)
    var exp_dones = ctx.enqueue_create_buffer[dtype](TOTAL_EXPAND)
    var exp_terminated = ctx.enqueue_create_buffer[dtype](TOTAL_EXPAND)
    var exp_obs = ctx.enqueue_create_buffer[dtype](TOTAL_EXPAND * OBS)

    # Batched MCTS buffers
    var b_pp = LayoutTensor[
        dtype, Layout.row_major(TOTAL_EXPAND), MutAnyOrigin
    ](gpu_mcts.pending_parent.unsafe_ptr())
    var b_pa = LayoutTensor[
        dtype, Layout.row_major(TOTAL_EXPAND), MutAnyOrigin
    ](gpu_mcts.pending_action.unsafe_ptr())
    var b_sp = LayoutTensor[
        dtype, Layout.row_major(TOTAL_EXPAND * MAX_DEPTH), MutAnyOrigin
    ](gpu_mcts.search_paths.unsafe_ptr())
    var b_ap = LayoutTensor[
        dtype, Layout.row_major(TOTAL_EXPAND * MAX_DEPTH), MutAnyOrigin
    ](gpu_mcts.action_paths.unsafe_ptr())
    var b_pl = LayoutTensor[
        dtype, Layout.row_major(TOTAL_EXPAND), MutAnyOrigin
    ](gpu_mcts.path_lengths.unsafe_ptr())
    var b_exp_st = LayoutTensor[
        dtype, Layout.row_major(TOTAL_EXPAND * GS), MutAnyOrigin
    ](gpu_mcts.expansion_states.unsafe_ptr())

    # Run MCTS rounds
    for round_idx in range(NUM_ROUNDS):
        # 1. Select + copy
        comptime run_sel = gpu_mcts_batched_select_and_copy_kernel[
            N_ENVS, MAX_NODES, ACT, BATCH_SIMS, GS, dtype
        ]
        ctx.enqueue_function[run_sel, run_sel](
            vc,
            tv,
            pr,
            ci,
            tvis,
            nc,
            miq,
            mxq,
            gs,
            b_pp,
            b_pa,
            b_exp_st,
            b_sp,
            b_ap,
            b_pl,
            Scalar[dtype](InspectConfig.PUCT.C_BASE),
            Scalar[dtype](InspectConfig.PUCT.C_INIT),
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # 2. Env step
        E.step_kernel_gpu[TOTAL_EXPAND, GS, OBS](
            ctx,
            gpu_mcts.expansion_states,
            gpu_mcts.pending_action,
            exp_rewards,
            exp_dones,
            exp_terminated,
            exp_obs,
            gpu_mcts.expansion_legal_masks,
            rng_seed=UInt64(round_idx),
        )

        # 3. Prediction on expanded states
        var p_in = LayoutTensor[
            dtype, Layout.row_major(TOTAL_EXPAND, PRED_IN), MutAnyOrigin
        ](exp_obs.unsafe_ptr())
        var p_out = LayoutTensor[
            dtype, Layout.row_major(TOTAL_EXPAND, PRED_OUT_DIM), MutAnyOrigin
        ](gpu_mcts.pred_output.unsafe_ptr())
        PredNet.forward_gpu[TOTAL_EXPAND](
            ctx, p_in, p_out, gpu_net.params_view(), gpu_net.model_state_view(), mcts_ws
        )

        # 4. Expand + backup
        var b_po = LayoutTensor[
            dtype, Layout.row_major(TOTAL_EXPAND * MCTS_PRED_OUT), MutAnyOrigin
        ](gpu_mcts.pred_output.unsafe_ptr())
        var b_rew = LayoutTensor[
            dtype, Layout.row_major(TOTAL_EXPAND), MutAnyOrigin
        ](exp_rewards.unsafe_ptr())
        comptime run_exp = gpu_mcts_batched_expand_backup_kernel[
            N_ENVS, MAX_NODES, ACT, BATCH_SIMS, MCTS_PRED_OUT, GS, dtype
        ]
        ctx.enqueue_function[run_exp, run_exp](
            vc,
            tv,
            pr,
            rw,
            ci,
            tvis,
            nc,
            miq,
            mxq,
            gs,
            b_exp_st,
            b_pp,
            b_pa,
            b_po,
            b_rew,
            b_sp,
            b_ap,
            b_pl,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # Download after each round to inspect
        ctx.enqueue_copy(nc_host, gpu_mcts.node_count)
        var vc_host = ctx.enqueue_create_host_buffer[dtype](
            N_ENVS * MAX_NODES * ACT
        )
        ctx.enqueue_copy(vc_host, gpu_mcts.visit_count)
        ctx.synchronize()

        print()
        print("--- After round", round_idx, "(", BATCH_SIMS, "sims) ---")
        print("  Node count (env 0):", nc_host[0])
        print("  Root visit counts (env 0):")
        var total_visits_root: Float64 = 0.0
        for a in range(ACT):
            var v = Float64(vc_host[a])
            total_visits_root += v
            if v > 0.5:
                print("    Action", a, ":", Int(v), "visits")
        print("  Total root visits:", Int(total_visits_root))

    # Final state
    var vc_final = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * MAX_NODES * ACT
    )
    ctx.enqueue_copy(vc_final, gpu_mcts.visit_count)
    var tv_final = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * MAX_NODES * ACT
    )
    ctx.enqueue_copy(tv_final, gpu_mcts.total_value)
    ctx.synchronize()

    print()
    print("=== Final MCTS state (env 0) ===")
    print("  Node count:", nc_host[0])
    print("  Root visit counts + Q-values:")
    var total_v: Float64 = 0.0
    for a in range(ACT):
        var visits = Float64(vc_final[a])
        total_v += visits
        if visits > 0.5:
            var q = Float64(tv_final[a]) / visits
            print("    Action", a, ":", Int(visits), "visits, Q =", q)
    print("  Total:", Int(total_v))

    # Check child nodes of root
    var ci_host = ctx.enqueue_create_host_buffer[dtype](
        N_ENVS * MAX_NODES * ACT
    )
    ctx.enqueue_copy(ci_host, gpu_mcts.child_idx)
    ctx.synchronize()

    print("  Root children:")
    for a in range(ACT):
        var child = Float64(ci_host[a])
        if child >= 0:
            print("    Action", a, "→ node", Int(child))

    print()
    print("=== Done ===")
