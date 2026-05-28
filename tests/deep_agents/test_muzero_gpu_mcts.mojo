"""Test fully GPU-resident MCTS for MuZero."""

from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.training import Network, GPUNetworkState
from mojo_rl.nn.optimizer import Adam
from mojo_rl.deep_agents.muzero.state import MuZeroCPUState, MuZeroGPUState
from mojo_rl.planners.tree_search.mcts_gpu import (
    GPUMCTSState,
    gpu_mcts_init_root_kernel,
    gpu_mcts_select_kernel,
    gpu_mcts_build_dyn_input_kernel,
    gpu_mcts_expand_kernel,
    gpu_mcts_backup_kernel,
    gpu_mcts_extract_actions_kernel,
    TPB,
    MAX_DEPTH,
)
from mojo_rl.deep_agents.muzero.kernels import extract_hidden_kernel


def main() raises:
    print("=== GPU MCTS Test ===")

    var ctx = DeviceContext()

    comptime OBS = 4
    comptime ACT = 2
    comptime LATENT = 32
    comptime HIDDEN = 32
    comptime BINS = 21
    comptime N_ENVS = 8
    comptime MAX_NODES = 64
    comptime NUM_SIMS = 10

    comptime StateType = MuZeroCPUState[OBS, ACT, LATENT, HIDDEN, BINS]
    comptime OptType = StateType.OptType
    comptime RepModel = StateType.RepModel
    comptime DynModel = StateType.DynModel
    comptime PredModel = StateType.PredModel
    comptime PRED_OUT = StateType.PRED_OUT
    comptime DYN_IN = StateType.DYN_IN
    comptime DYN_OUT = StateType.DYN_OUT

    comptime RepNet = Network[RepModel, OptType]
    comptime DynNet = Network[DynModel, OptType]
    comptime PredNet = Network[PredModel, OptType]

    # Create CPU state + upload to GPU
    var cpu_state = StateType()
    var gpu_rep = GPUNetworkState[RepModel, OptType](ctx)
    var gpu_dyn = GPUNetworkState[DynModel, OptType](ctx)
    var gpu_pred = GPUNetworkState[PredModel, OptType](ctx)
    gpu_rep.upload_from(cpu_state.representation, ctx)
    gpu_dyn.upload_from(cpu_state.dynamics, ctx)
    gpu_pred.upload_from(cpu_state.prediction, ctx)

    # Create GPU MCTS state
    var mcts = GPUMCTSState[N_ENVS, MAX_NODES, ACT, LATENT, BINS](ctx)
    print("GPU MCTS state created")

    # Create dummy observations [N_ENVS × OBS]
    var obs_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS)
    obs_buf.enqueue_fill(Scalar[dtype](0.1))

    # Workspace for network forward
    comptime WS_REP = RepModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS_DYN = DynModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime WS_PRED = PredModel.WORKSPACE_SIZE_PER_SAMPLE
    comptime MAX_WS = WS_REP if WS_REP > WS_DYN else WS_DYN
    comptime MAX_WS2 = MAX_WS if MAX_WS > WS_PRED else WS_PRED
    comptime WS_TOTAL = N_ENVS * MAX_WS2 if MAX_WS2 > 0 else 1
    var workspace = ctx.enqueue_create_buffer[dtype](WS_TOTAL)

    # Step 1: Representation forward on all obs → root hidden states
    comptime REP_IN_DIM = RepModel.IN_DIM
    comptime REP_OUT_DIM = RepModel.OUT_DIM
    var obs_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, REP_IN_DIM), MutAnyOrigin
    ](obs_buf.unsafe_ptr())
    # Write root hidden states to position 0 of each env's hidden pool
    var root_h_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, REP_OUT_DIM), MutAnyOrigin
    ](mcts.hidden_states.unsafe_ptr())
    RepNet.forward_gpu[N_ENVS](
        ctx, obs_t, root_h_t, gpu_rep.params_view(), workspace
    )

    # Step 2: Prediction forward on root hidden → root policy + value
    comptime PRED_IN_DIM = PredModel.IN_DIM
    comptime PRED_OUT_DIM = PredModel.OUT_DIM
    var root_pred_in = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, PRED_IN_DIM), MutAnyOrigin
    ](mcts.hidden_states.unsafe_ptr())
    var root_pred_out = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, PRED_OUT_DIM), MutAnyOrigin
    ](mcts.pred_output.unsafe_ptr())
    PredNet.forward_gpu[N_ENVS](
        ctx, root_pred_in, root_pred_out, gpu_pred.params_view(), workspace
    )

    # Step 3: Initialize root nodes
    comptime ENV_BLOCKS = (N_ENVS + TPB - 1) // TPB

    var vc_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](mcts.visit_count.unsafe_ptr())
    var tv_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](mcts.total_value.unsafe_ptr())
    var pr_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](mcts.prior.unsafe_ptr())
    var rw_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](mcts.reward.unsafe_ptr())
    var ci_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](mcts.child_idx.unsafe_ptr())
    var tvis_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ](mcts.total_visits.unsafe_ptr())
    var nc_t = LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](
        mcts.node_count.unsafe_ptr()
    )
    var po_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * PRED_OUT), MutAnyOrigin
    ](mcts.pred_output.unsafe_ptr())
    var miq_t = LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](
        mcts.min_q.unsafe_ptr()
    )
    var mxq_t = LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](
        mcts.max_q.unsafe_ptr()
    )

    comptime run_init = gpu_mcts_init_root_kernel[
        N_ENVS, MAX_NODES, ACT, LATENT, PRED_OUT, dtype
    ]
    ctx.enqueue_function[run_init, run_init](
        vc_t,
        tv_t,
        pr_t,
        rw_t,
        ci_t,
        tvis_t,
        nc_t,
        po_t,
        miq_t,
        mxq_t,
        Scalar[dtype](0.25),
        Scalar[DType.uint32](42),
        grid_dim=(ENV_BLOCKS,),
        block_dim=(TPB,),
    )
    print("Root nodes initialized")

    # Step 4: Run MCTS simulations
    var pp_t = LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](
        mcts.pending_parent.unsafe_ptr()
    )
    var pa_t = LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](
        mcts.pending_action.unsafe_ptr()
    )
    var sp_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_DEPTH), MutAnyOrigin
    ](mcts.search_paths.unsafe_ptr())
    var ap_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_DEPTH), MutAnyOrigin
    ](mcts.action_paths.unsafe_ptr())
    var pl_t = LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](
        mcts.path_lengths.unsafe_ptr()
    )
    var lv_t = LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](
        mcts.leaf_values.unsafe_ptr()
    )

    comptime DYN_IN_DIM = DynModel.IN_DIM
    comptime DYN_OUT_DIM = DynModel.OUT_DIM
    var hs_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * LATENT), MutAnyOrigin
    ](mcts.hidden_states.unsafe_ptr())

    for sim in range(NUM_SIMS):
        # Selection
        comptime run_select = gpu_mcts_select_kernel[
            N_ENVS, MAX_NODES, ACT, dtype
        ]
        ctx.enqueue_function[run_select, run_select](
            vc_t,
            tv_t,
            pr_t,
            ci_t,
            tvis_t,
            nc_t,
            miq_t,
            mxq_t,
            pp_t,
            pa_t,
            sp_t,
            ap_t,
            pl_t,
            Scalar[dtype](19652.0),
            Scalar[dtype](1.25),
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # Build dynamics input
        var di_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS * DYN_IN), MutAnyOrigin
        ](mcts.dyn_input.unsafe_ptr())
        comptime run_build = gpu_mcts_build_dyn_input_kernel[
            N_ENVS, MAX_NODES, ACT, LATENT, DYN_IN, dtype
        ]
        ctx.enqueue_function[run_build, run_build](
            di_t,
            hs_t,
            pp_t,
            pa_t,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # Dynamics forward
        var dyn_in_net = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, DYN_IN_DIM), MutAnyOrigin
        ](mcts.dyn_input.unsafe_ptr())
        var dyn_out_net = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, DYN_OUT_DIM), MutAnyOrigin
        ](mcts.dyn_output.unsafe_ptr())
        DynNet.forward_gpu[N_ENVS](
            ctx, dyn_in_net, dyn_out_net, gpu_dyn.params_view(), workspace
        )

        # Expand nodes
        var do_t = LayoutTensor[
            dtype, Layout.row_major(N_ENVS * DYN_OUT), MutAnyOrigin
        ](mcts.dyn_output.unsafe_ptr())
        comptime run_expand = gpu_mcts_expand_kernel[
            N_ENVS, MAX_NODES, ACT, LATENT, PRED_OUT, DYN_OUT, dtype
        ]

        # Prediction forward on new child hidden states
        # First copy child hidden to pred_input via expand kernel (it writes to hidden_states)
        # Then do prediction forward
        # Actually, expand kernel needs pred_output as input. So we need to:
        # 1. Extract child hidden from dyn_output → hidden_states (done in expand)
        # 2. Copy child hidden to pred_input
        # 3. Run prediction forward
        # 4. Run expand with pred_output

        # For simplicity: expand extracts hidden + links, then we do pred forward separately
        # and run a second expand-like kernel for prior setting...
        # OR: we do pred forward BEFORE expand and pass both dyn_output + pred_output to expand

        # Let's do pred forward first using dyn_output's hidden portion as input
        # Copy first LATENT elements of each env's dyn_output to pred_input
        var pred_in_net = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, PRED_IN_DIM), MutAnyOrigin
        ](mcts.pred_input.unsafe_ptr())
        # Quick copy kernel: extract first LATENT from DYN_OUT for each env
        for e_idx in range(N_ENVS):
            pass  # This is on CPU — not ideal, but for testing

        # Actually, let's use a simpler approach: extract hidden from dyn_output to pred_input
        # using our existing extract_hidden_kernel
        var pred_in_flat = LayoutTensor[
            dtype, Layout.row_major(N_ENVS * LATENT), MutAnyOrigin
        ](mcts.pred_input.unsafe_ptr())
        var dyn_out_flat = LayoutTensor[
            dtype, Layout.row_major(N_ENVS * DYN_OUT), MutAnyOrigin
        ](mcts.dyn_output.unsafe_ptr())
        comptime EXTRACT_BLOCKS = (N_ENVS * LATENT + TPB - 1) // TPB
        comptime run_extract = extract_hidden_kernel[
            N_ENVS, LATENT, DYN_OUT, dtype
        ]
        ctx.enqueue_function[run_extract, run_extract](
            pred_in_flat,
            dyn_out_flat,
            grid_dim=(EXTRACT_BLOCKS,),
            block_dim=(TPB,),
        )

        # Prediction forward
        var pred_out_net = LayoutTensor[
            dtype, Layout.row_major(N_ENVS, PRED_OUT_DIM), MutAnyOrigin
        ](mcts.pred_output.unsafe_ptr())
        PredNet.forward_gpu[N_ENVS](
            ctx, pred_in_net, pred_out_net, gpu_pred.params_view(), workspace
        )

        # Expand nodes (creates child, sets prior, decodes value)
        var pred_out_expand = LayoutTensor[
            dtype, Layout.row_major(N_ENVS * PRED_OUT), MutAnyOrigin
        ](mcts.pred_output.unsafe_ptr())
        ctx.enqueue_function[run_expand, run_expand](
            vc_t,
            tv_t,
            pr_t,
            rw_t,
            ci_t,
            tvis_t,
            nc_t,
            hs_t,
            pp_t,
            pa_t,
            do_t,
            pred_out_expand,
            Scalar[dtype](-10.0),
            Scalar[dtype](10.0),
            lv_t,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # Backup
        comptime run_backup = gpu_mcts_backup_kernel[
            N_ENVS, MAX_NODES, ACT, dtype
        ]
        ctx.enqueue_function[run_backup, run_backup](
            vc_t,
            tv_t,
            rw_t,
            tvis_t,
            miq_t,
            mxq_t,
            sp_t,
            ap_t,
            pl_t,
            lv_t,
            Scalar[dtype](0.99),
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )

    print("Completed", NUM_SIMS, "GPU MCTS simulations for", N_ENVS, "envs")

    # Step 5: Extract actions
    var act_out_t = LayoutTensor[dtype, Layout.row_major(N_ENVS), MutAnyOrigin](
        mcts.actions_out.unsafe_ptr()
    )
    var pol_out_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin
    ](mcts.policies_out.unsafe_ptr())
    comptime run_extract_act = gpu_mcts_extract_actions_kernel[
        N_ENVS, MAX_NODES, ACT, dtype
    ]
    ctx.enqueue_function[run_extract_act, run_extract_act](
        vc_t,
        act_out_t,
        pol_out_t,
        grid_dim=(ENV_BLOCKS,),
        block_dim=(TPB,),
    )

    # Read back actions
    var actions_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS)
    var policies_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS * ACT)
    ctx.enqueue_copy(actions_host, mcts.actions_out)
    ctx.enqueue_copy(policies_host, mcts.policies_out)
    ctx.synchronize()

    print("Actions:")
    for e in range(N_ENVS):
        var a = Int(Float64(actions_host[e]))
        var p0 = Float64(policies_host[e * ACT])
        var p1 = Float64(policies_host[e * ACT + 1])
        print("  env", e, ": action=", a, "policy=[", p0, ",", p1, "]")

    # Verify all policies sum to ~1
    var all_valid = True
    for e in range(N_ENVS):
        var sum_p = Float64(0.0)
        for a in range(ACT):
            sum_p += Float64(policies_host[e * ACT + a])
        if sum_p < 0.99 or sum_p > 1.01:
            all_valid = False
            print("FAIL: env", e, "policy sum =", sum_p)

    if all_valid:
        print("PASS: All GPU MCTS policies valid")
    else:
        print("FAIL: Some policies invalid")

    print("=== Done ===")
