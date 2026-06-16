"""Legacy EZv2/MuZero Gumbel-MCTS GPU driver (sunset-scoped).

`run_gumbel_search_gpu` and its per-sim body `_run_one_sim_gpu` drive the
shared Gumbel kernels (`mcts_gpu_gumbel.mojo`) over legacy `nn` networks
(`Network` / `GPUNetworkState` / `Model` / `Optimizer`). This was the only
remaining legacy-`nn` dependency in `planners/tree_search/`; it was split
out of `mcts_gpu_gumbel.mojo` so that file — and the `GumbelGPUMCTS`
orchestrator built on its kernels, which `deep_agents2` depends on — is
legacy-free.

Consumers: legacy `deep_agents/efficient_zero_v2`, the `GumbelGPUMCTS`
byte-parity test, and two legacy examples. DELETE this file together with
the legacy `nn` / `deep_agents` packages at sunset (Phase 4).
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor
from mojo_rl.nn2.constants import DT as dtype
from mojo_rl.nn.model.model import Model
from mojo_rl.nn.optimizer.optimizer import Optimizer
from mojo_rl.nn.training import Network, GPUNetworkState
from .mcts_gpu_gumbel import (
    MAX_DEPTH,
    EZV2GPUMCTSState,
    gz_scatter_root_hidden_kernel,
    gz_init_root_kernel,
    gz_select_kernel,
    gz_copy_pred_input_kernel,
    gz_expand_kernel,
    gz_backup_kernel,
    gz_halve_active_kernel,
    gz_extract_policy_kernel,
)

comptime TPB = 256  # preserved from legacy nn.constants (nn2.TPB == 128)


def run_gumbel_search_gpu[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    MAX_K: Int,
    NUM_SIMULATIONS: Int,
    RepModel: Model,
    DynModel: Model,
    PredModel: Model,
    RepOpt: Optimizer,
    DynOpt: Optimizer,
    PredOpt: Optimizer,
](
    ctx: DeviceContext,
    mut state: EZV2GPUMCTSState[N_ENVS, MAX_NODES, ACT, LATENT, BINS, MAX_K],
    obs_buf: DeviceBuffer[dtype],
    rep_state: GPUNetworkState[RepModel, RepOpt],
    dyn_state: GPUNetworkState[DynModel, DynOpt],
    pred_state: GPUNetworkState[PredModel, PredOpt],
    workspace_buf: DeviceBuffer[dtype],
    v_min: Float64,
    v_max: Float64,
    apply_legal: Bool = False,
    k_actual: Int = MAX_K,
    c_visit: Float64 = 50.0,
    c_scale: Float64 = 0.1,
    gamma: Float64 = 0.997,
    rng_seed: UInt32 = UInt32(0),
    qnorm_per_node: Bool = True,
) raises:
    """Run Gumbel search across all envs in `state`. Writes the improved
    policy distribution to `state.policies_out`.

    Caller is responsible for:
      • populating `obs_buf` with `[N_ENVS × OBS]` (contiguous batch);
      • optionally populating `state.legal_mask` if `apply_legal=True`;
      • calling `state.zero_tree(ctx)` is done internally;
      • allocating `workspace_buf` sized for the max of the three networks'
        per-sample workspace * `N_ENVS`.
    """
    comptime PRED_OUT = ACT + BINS
    comptime DYN_IN = LATENT + ACT
    comptime DYN_OUT = LATENT + BINS
    comptime ENV_BLOCKS = (N_ENVS + TPB - 1) // TPB

    # ── 0. Reset tree ────────────────────────────────────────────────────
    state.zero_tree(ctx)

    # ── 1. Rep forward (obs → root_hidden, contiguous [N_ENVS × LATENT]) ─
    var obs_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, RepModel.IN_DIM), MutAnyOrigin
    ](obs_buf.unsafe_ptr())
    var root_hidden_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, RepModel.OUT_DIM), MutAnyOrigin
    ](state.root_hidden.unsafe_ptr())
    Network[RepModel, RepOpt].forward_gpu[N_ENVS](
        ctx,
        obs_t,
        root_hidden_t,
        rep_state.params_view(),
        rep_state.model_state_view(),
        workspace_buf,
    )

    # ── 2. Pred forward (root_hidden → pred_output, contiguous) ──────────
    var pred_in_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, PredModel.IN_DIM), MutAnyOrigin
    ](state.root_hidden.unsafe_ptr())
    var pred_out_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, PredModel.OUT_DIM), MutAnyOrigin
    ](state.pred_output.unsafe_ptr())
    Network[PredModel, PredOpt].forward_gpu[N_ENVS](
        ctx,
        pred_in_t,
        pred_out_t,
        pred_state.params_view(),
        pred_state.model_state_view(),
        workspace_buf,
    )

    # ── 3. Scatter root_hidden into hidden_states[e][0] ──────────────────
    var rh_flat = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * LATENT), MutAnyOrigin
    ](state.root_hidden.unsafe_ptr())
    var hs_flat = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * LATENT), MutAnyOrigin
    ](state.hidden_states.unsafe_ptr())
    comptime run_scatter = gz_scatter_root_hidden_kernel[
        N_ENVS, MAX_NODES, LATENT, dtype
    ]
    ctx.enqueue_function[run_scatter](
        rh_flat,
        hs_flat,
        grid_dim=(ENV_BLOCKS,),
        block_dim=(TPB,),
    )

    # ── 4. Init root: logits + Gumbel-Top-k + value + per-env scalars ────
    var nl_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](state.node_logits.unsafe_ptr())
    var nv_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ](state.node_value.unsafe_ptr())
    var nc_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.node_count.unsafe_ptr())
    var miq_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.min_q.unsafe_ptr())
    var mxq_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.max_q.unsafe_ptr())
    var lm_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin
    ](state.legal_mask.unsafe_ptr())
    var rc_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_K), MutAnyOrigin
    ](state.root_candidates.unsafe_ptr())
    var rg_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_K), MutAnyOrigin
    ](state.root_gumbels.unsafe_ptr())
    var ra_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_K), MutAnyOrigin
    ](state.root_active.unsafe_ptr())
    var po_full_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * PRED_OUT), MutAnyOrigin
    ](state.pred_output.unsafe_ptr())

    var k_clipped = k_actual
    if k_clipped > MAX_K:
        k_clipped = MAX_K
    if k_clipped > ACT:
        k_clipped = ACT
    # Round down to power of two for clean log2(K) phases.
    k_clipped = _largest_power_of_two_le(k_clipped)
    if k_clipped < 1:
        k_clipped = 1

    comptime run_init = gz_init_root_kernel[
        N_ENVS, MAX_NODES, ACT, BINS, MAX_K, PRED_OUT, dtype
    ]
    ctx.enqueue_function[run_init](
        nl_t,
        nv_t,
        nc_t,
        miq_t,
        mxq_t,
        lm_t,
        rc_t,
        rg_t,
        ra_t,
        po_full_t,
        Scalar[dtype](v_min),
        Scalar[dtype](v_max),
        Scalar[DType.int32](k_clipped),
        Scalar[DType.uint8](1 if apply_legal else 0),
        rng_seed,
        Scalar[dtype](1.0),  # gumbel_scale: GPU enqueue ignores kernel defaults
        grid_dim=(ENV_BLOCKS,),
        block_dim=(TPB,),
    )

    # ── 5. Sequential Halving simulation loop ────────────────────────────
    var num_phases = _ilog2(k_clipped)
    if num_phases < 1:
        num_phases = 1
    var per_phase_budget = NUM_SIMULATIONS // num_phases
    if per_phase_budget < 1:
        per_phase_budget = 1

    var sims_used = 0
    var active_size = k_clipped
    for phase in range(num_phases):
        var per_action = per_phase_budget // active_size
        if per_action < 1:
            per_action = 1

        for _rep in range(per_action):
            for slot in range(active_size):
                if sims_used >= NUM_SIMULATIONS:
                    break
                _run_one_sim_gpu[
                    N_ENVS,
                    MAX_NODES,
                    ACT,
                    LATENT,
                    BINS,
                    MAX_K,
                    DynModel,
                    PredModel,
                    DynOpt,
                    PredOpt,
                ](
                    ctx,
                    state,
                    dyn_state,
                    pred_state,
                    workspace_buf,
                    slot,
                    apply_legal,
                    v_min,
                    v_max,
                    c_visit,
                    c_scale,
                    gamma,
                    qnorm_per_node,
                )
                sims_used += 1

        # Halve the active set, except in the last phase.
        if phase + 1 < num_phases and active_size > 1:
            var keep = active_size // 2
            if keep < 1:
                keep = 1
            comptime run_halve = gz_halve_active_kernel[
                N_ENVS, MAX_NODES, ACT, MAX_K, dtype
            ]
            var vc_t = LayoutTensor[
                dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
            ](state.visit_count.unsafe_ptr())
            var tv_t = LayoutTensor[
                dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
            ](state.total_value.unsafe_ptr())
            var tvis_t = LayoutTensor[
                dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
            ](state.total_visits.unsafe_ptr())
            ctx.enqueue_function[run_halve](
                vc_t,
                tv_t,
                nl_t,
                tvis_t,
                nv_t,
                miq_t,
                mxq_t,
                rc_t,
                rg_t,
                ra_t,
                Scalar[DType.int32](active_size),
                Scalar[DType.int32](keep),
                Scalar[dtype](c_visit),
                Scalar[dtype](c_scale),
                Scalar[DType.uint8](1 if qnorm_per_node else 0),
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )
            active_size = keep

    # Spend any leftover simulations on slot 0 of the (now size-1) active set.
    while sims_used < NUM_SIMULATIONS:
        _run_one_sim_gpu[
            N_ENVS,
            MAX_NODES,
            ACT,
            LATENT,
            BINS,
            MAX_K,
            DynModel,
            PredModel,
            DynOpt,
            PredOpt,
        ](
            ctx,
            state,
            dyn_state,
            pred_state,
            workspace_buf,
            0,
            apply_legal,
            v_min,
            v_max,
            c_visit,
            c_scale,
            gamma,
            qnorm_per_node,
        )
        sims_used += 1

    # ── 6. Extract improved policy ───────────────────────────────────────
    var po_extract_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin
    ](state.policies_out.unsafe_ptr())
    var vc_t2 = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](state.visit_count.unsafe_ptr())
    var tv_t2 = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](state.total_value.unsafe_ptr())
    var tvis_t2 = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ](state.total_visits.unsafe_ptr())
    comptime run_extract = gz_extract_policy_kernel[
        N_ENVS, MAX_NODES, ACT, dtype
    ]
    ctx.enqueue_function[run_extract](
        vc_t2,
        tv_t2,
        nl_t,
        tvis_t2,
        nv_t,
        miq_t,
        mxq_t,
        lm_t,
        po_extract_t,
        Scalar[DType.uint8](1 if apply_legal else 0),
        Scalar[dtype](c_visit),
        Scalar[dtype](c_scale),
        Scalar[DType.uint8](1 if qnorm_per_node else 0),
        grid_dim=(ENV_BLOCKS,),
        block_dim=(TPB,),
    )


def _run_one_sim_gpu[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    MAX_K: Int,
    DynModel: Model,
    PredModel: Model,
    DynOpt: Optimizer,
    PredOpt: Optimizer,
](
    ctx: DeviceContext,
    mut state: EZV2GPUMCTSState[N_ENVS, MAX_NODES, ACT, LATENT, BINS, MAX_K],
    dyn_state: GPUNetworkState[DynModel, DynOpt],
    pred_state: GPUNetworkState[PredModel, PredOpt],
    workspace_buf: DeviceBuffer[dtype],
    slot: Int,
    apply_legal: Bool,
    v_min: Float64,
    v_max: Float64,
    c_visit: Float64,
    c_scale: Float64,
    gamma: Float64,
    qnorm_per_node: Bool = True,
) raises:
    """One simulation across all envs: select → dyn → pred → expand →
    backup. The root candidate slot is shared across envs (that's safe
    because Sequential Halving keeps the active sets the same size for all
    envs in any given phase)."""
    comptime PRED_OUT = ACT + BINS
    comptime DYN_IN = LATENT + ACT
    comptime DYN_OUT = LATENT + BINS
    comptime ENV_BLOCKS = (N_ENVS + TPB - 1) // TPB

    var vc_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](state.visit_count.unsafe_ptr())
    var tv_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](state.total_value.unsafe_ptr())
    var nl_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](state.node_logits.unsafe_ptr())
    var rw_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](state.reward.unsafe_ptr())
    var ci_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](state.child_idx.unsafe_ptr())
    var tvis_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ](state.total_visits.unsafe_ptr())
    var nv_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ](state.node_value.unsafe_ptr())
    var nc_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.node_count.unsafe_ptr())
    var miq_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.min_q.unsafe_ptr())
    var mxq_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.max_q.unsafe_ptr())
    var lm_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin
    ](state.legal_mask.unsafe_ptr())
    var rc_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_K), MutAnyOrigin
    ](state.root_candidates.unsafe_ptr())
    var ra_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_K), MutAnyOrigin
    ](state.root_active.unsafe_ptr())
    var hs_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * LATENT), MutAnyOrigin
    ](state.hidden_states.unsafe_ptr())
    var di_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * DYN_IN), MutAnyOrigin
    ](state.dyn_input.unsafe_ptr())
    var pp_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.pending_parent.unsafe_ptr())
    var pa_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.pending_action.unsafe_ptr())
    var sp_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_DEPTH), MutAnyOrigin
    ](state.search_paths.unsafe_ptr())
    var ap_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_DEPTH), MutAnyOrigin
    ](state.action_paths.unsafe_ptr())
    var pl_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.path_lengths.unsafe_ptr())

    # Selection.
    comptime run_select = gz_select_kernel[
        N_ENVS, MAX_NODES, ACT, MAX_K, LATENT, DYN_IN, dtype
    ]
    ctx.enqueue_function[run_select](
        vc_t,
        tv_t,
        nl_t,
        ci_t,
        tvis_t,
        nv_t,
        miq_t,
        mxq_t,
        lm_t,
        rc_t,
        ra_t,
        hs_t,
        di_t,
        pp_t,
        pa_t,
        sp_t,
        ap_t,
        pl_t,
        Scalar[DType.int32](slot),
        Scalar[DType.uint8](1 if apply_legal else 0),
        Scalar[dtype](c_visit),
        Scalar[dtype](c_scale),
        Scalar[DType.uint8](1 if qnorm_per_node else 0),
        grid_dim=(ENV_BLOCKS,),
        block_dim=(TPB,),
    )

    # Dynamics forward.
    var dyn_in_b = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, DynModel.IN_DIM), MutAnyOrigin
    ](state.dyn_input.unsafe_ptr())
    var dyn_out_b = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, DynModel.OUT_DIM), MutAnyOrigin
    ](state.dyn_output.unsafe_ptr())
    Network[DynModel, DynOpt].forward_gpu[N_ENVS](
        ctx,
        dyn_in_b,
        dyn_out_b,
        dyn_state.params_view(),
        dyn_state.model_state_view(),
        workspace_buf,
    )

    # Copy dyn_output's hidden prefix into pred_input, then prediction forward.
    comptime run_copy = gz_copy_pred_input_kernel[
        N_ENVS, LATENT, DYN_OUT, dtype
    ]
    var pred_in_flat = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * LATENT), MutAnyOrigin
    ](state.pred_input.unsafe_ptr())
    var dyn_out_flat = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * DYN_OUT), MutAnyOrigin
    ](state.dyn_output.unsafe_ptr())
    ctx.enqueue_function[run_copy](
        pred_in_flat,
        dyn_out_flat,
        grid_dim=(ENV_BLOCKS,),
        block_dim=(TPB,),
    )

    var pred_in_b = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, PredModel.IN_DIM), MutAnyOrigin
    ](state.pred_input.unsafe_ptr())
    var pred_out_b = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, PredModel.OUT_DIM), MutAnyOrigin
    ](state.pred_output.unsafe_ptr())
    Network[PredModel, PredOpt].forward_gpu[N_ENVS](
        ctx,
        pred_in_b,
        pred_out_b,
        pred_state.params_view(),
        pred_state.model_state_view(),
        workspace_buf,
    )

    # Expand.
    var lv_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.leaf_values.unsafe_ptr())
    var po_full_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * PRED_OUT), MutAnyOrigin
    ](state.pred_output.unsafe_ptr())
    comptime run_expand = gz_expand_kernel[
        N_ENVS, MAX_NODES, ACT, LATENT, BINS, PRED_OUT, DYN_OUT, dtype
    ]
    ctx.enqueue_function[run_expand](
        vc_t,
        tv_t,
        nl_t,
        rw_t,
        ci_t,
        tvis_t,
        nv_t,
        nc_t,
        hs_t,
        pp_t,
        pa_t,
        dyn_out_flat,
        po_full_t,
        lv_t,
        Scalar[dtype](v_min),
        Scalar[dtype](v_max),
        grid_dim=(ENV_BLOCKS,),
        block_dim=(TPB,),
    )

    # Backup.
    comptime run_backup = gz_backup_kernel[
        N_ENVS, MAX_NODES, ACT, dtype
    ]
    ctx.enqueue_function[run_backup](
        vc_t,
        tv_t,
        rw_t,
        tvis_t,
        nv_t,
        miq_t,
        mxq_t,
        sp_t,
        ap_t,
        pl_t,
        lv_t,
        Scalar[dtype](gamma),
        grid_dim=(ENV_BLOCKS,),
        block_dim=(TPB,),
    )


# ═════════════════════════════════════════════════════════════════════════
# Host helpers (mirror efficient_zero_v2/mcts.mojo)
# ═════════════════════════════════════════════════════════════════════════


def _ilog2(n: Int) -> Int:
    var x = n
    var r = 0
    while x > 1:
        x = x // 2
        r += 1
    return r


def _largest_power_of_two_le(n: Int) -> Int:
    var x = 1
    while x * 2 <= n:
        x *= 2
    return x
