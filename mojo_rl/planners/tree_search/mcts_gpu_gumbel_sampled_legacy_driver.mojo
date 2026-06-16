"""Legacy EZv2 sampled-Gumbel-MCTS GPU driver (sunset-scoped).

`run_sampled_gumbel_search_gpu` and its per-sim body `_run_one_sim_gpu`
drive the shared sampled-Gumbel kernels (`mcts_gpu_gumbel_sampled.mojo`)
over legacy `nn` networks (`Network` / `GPUNetworkState` / `Model` /
`Optimizer`). Split out of `mcts_gpu_gumbel_sampled.mojo` so that file —
and the `SampledGumbelGPUMCTS` orchestrator built on its kernels, which
`deep_agents2` depends on — is legacy-free.

Consumers: legacy `deep_agents/efficient_zero_v2`, the
`SampledGumbelGPUMCTS` byte-parity test, and several legacy examples.
DELETE this file together with the legacy `nn` / `deep_agents` packages at
sunset (Phase 4).
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor
from mojo_rl.nn2.constants import DT as dtype
from mojo_rl.nn.model.model import Model
from mojo_rl.nn.optimizer.optimizer import Optimizer
from mojo_rl.nn.training import Network, GPUNetworkState
from .mcts_gpu_gumbel_sampled import (
    MAX_DEPTH,
    EZV2GPUSampledMCTSState,
    gs_scatter_root_hidden_kernel,
    gs_init_root_kernel,
    gs_select_kernel,
    gs_copy_pred_input_kernel,
    gs_expand_kernel,
    gs_backup_kernel,
    gs_halve_active_kernel,
    gs_extract_kernel,
)

comptime TPB = 256  # preserved from legacy nn.constants (nn2.TPB == 128)


def run_sampled_gumbel_search_gpu[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT_DIM: Int,
    LATENT: Int,
    BINS: Int,
    K_ROOT: Int,
    K_NON_ROOT: Int,
    NUM_SIMULATIONS: Int,
    RepModel: Model,
    DynModel: Model,
    PredModel: Model,
    RepOpt: Optimizer,
    DynOpt: Optimizer,
    PredOpt: Optimizer,
    # Root sampling mode selector — see `gs_init_root_kernel` for the
    # legacy-vs-DMC dispatch. Default `K_ROOT` preserves legacy magnified
    # behavior so existing positional callers stay unchanged. Positioned
    # at the end of the template list so the legacy positional ordering
    # of the Model/Optimizer params keeps working.
    N_POLICY_AT_ROOT: Int = K_ROOT,
](
    ctx: DeviceContext,
    mut state: EZV2GPUSampledMCTSState[
        N_ENVS, MAX_NODES, ACT_DIM, LATENT, BINS, K_ROOT, K_NON_ROOT
    ],
    obs_buf: DeviceBuffer[dtype],
    rep_state: GPUNetworkState[RepModel, RepOpt],
    dyn_state: GPUNetworkState[DynModel, DynOpt],
    pred_state: GPUNetworkState[PredModel, PredOpt],
    workspace_buf: DeviceBuffer[dtype],
    v_min: Float64,
    v_max: Float64,
    # Separate reward-head support, in TRANSFORMED scalar space (paper
    # `dmc_state.yaml: reward_support: range=[-2, 2]`, transformed via h
    # gives ≈ ±0.732). Decoupled from `v_min/v_max` 2026-05-14; the prior
    # shared range left MCTS reading decoded rewards ~100× too coarse.
    reward_min: Float64 = -0.732_050_807_568_877_3,
    reward_max: Float64 = 0.732_050_807_568_877_3,
    max_action: Float64 = 1.0,
    min_std: Float64 = 0.1,
    std_magnification: Float64 = 3.0,
    # Dreamer-v3 soft clamp on μ_pre and softplus bias on σ_raw — must
    # match the training loss kernel and CPU MCTS (reference 5.0 / 1.0).
    soft_clamp: Float64 = 5.0,
    init_std: Float64 = 1.0,
    c_visit: Float64 = 50.0,
    c_scale: Float64 = 0.1,
    gamma: Float64 = 0.997,
    deterministic: Bool = False,
    rng_seed: UInt32 = UInt32(0),
) raises:
    """Run the sampled-Gumbel MCTS across all envs in `state`. Writes
    `state.chosen_actions[N_ENVS, ACT_DIM]` and
    `state.root_visits[N_ENVS, K_ROOT]`.

    Caller responsibilities:
      • populate `obs_buf` with `[N_ENVS × OBS]`,
      • size `workspace_buf` for the largest of the three networks'
        per-sample workspace × N_ENVS,
      • construct `state` with the matching template parameters.
    """
    comptime PRED_OUT = 2 * ACT_DIM + BINS
    comptime DYN_IN = LATENT + ACT_DIM
    comptime DYN_OUT = LATENT + BINS
    comptime ENV_BLOCKS = (N_ENVS + TPB - 1) // TPB
    comptime K_PAD = K_ROOT

    state.zero_tree(ctx)

    # Rep forward.
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

    # Pred forward at root.
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

    # Scatter root hidden into hidden_states[e][0].
    var rh_flat = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * LATENT), MutAnyOrigin
    ](state.root_hidden.unsafe_ptr())
    var hs_flat = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * LATENT), MutAnyOrigin
    ](state.hidden_states.unsafe_ptr())
    comptime run_scatter = gs_scatter_root_hidden_kernel[
        N_ENVS, MAX_NODES, LATENT, dtype
    ]
    ctx.enqueue_function[run_scatter](
        rh_flat,
        hs_flat,
        grid_dim=(ENV_BLOCKS,),
        block_dim=(TPB,),
    )

    # Init root: sample candidates + log_prior + value + per-env scalars.
    var act_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * K_PAD * ACT_DIM),
        MutAnyOrigin,
    ](state.actions.unsafe_ptr())
    var lp_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * K_PAD), MutAnyOrigin
    ](state.log_prior.unsafe_ptr())
    var nv_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ](state.node_value.unsafe_ptr())
    var ak_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ](state.active_k.unsafe_ptr())
    var nc_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.node_count.unsafe_ptr())
    var miq_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.min_q.unsafe_ptr())
    var mxq_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.max_q.unsafe_ptr())
    var rg_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * K_ROOT), MutAnyOrigin
    ](state.root_gumbels.unsafe_ptr())
    var ra_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * K_ROOT), MutAnyOrigin
    ](state.root_active.unsafe_ptr())
    var po_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * PRED_OUT), MutAnyOrigin
    ](state.pred_output.unsafe_ptr())

    comptime run_init = gs_init_root_kernel[
        N_ENVS, MAX_NODES, ACT_DIM, BINS, K_ROOT, K_PAD, PRED_OUT,
        N_POLICY_AT_ROOT, dtype
    ]
    ctx.enqueue_function[run_init](
        act_t,
        lp_t,
        nv_t,
        ak_t,
        nc_t,
        miq_t,
        mxq_t,
        rg_t,
        ra_t,
        po_t,
        Scalar[dtype](v_min),
        Scalar[dtype](v_max),
        Scalar[dtype](max_action),
        Scalar[dtype](min_std),
        Scalar[dtype](std_magnification),
        Scalar[dtype](soft_clamp),
        Scalar[dtype](init_std),
        rng_seed,
        grid_dim=(ENV_BLOCKS,),
        block_dim=(TPB,),
    )

    # Sequential Halving simulation loop.
    var num_phases = _ilog2(_largest_power_of_two_le(K_ROOT))
    if num_phases < 1:
        num_phases = 1
    var per_phase_budget = NUM_SIMULATIONS // num_phases
    if per_phase_budget < 1:
        per_phase_budget = 1

    var sims_used = 0
    var active_size = _largest_power_of_two_le(K_ROOT)
    if active_size < 1:
        active_size = 1
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
                    ACT_DIM,
                    K_ROOT,
                    K_NON_ROOT,
                    K_PAD,
                    LATENT,
                    BINS,
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
                    v_min,
                    v_max,
                    reward_min,
                    reward_max,
                    max_action,
                    min_std,
                    soft_clamp,
                    init_std,
                    c_visit,
                    c_scale,
                    gamma,
                    rng_seed,
                    UInt32(sims_used),
                )
                sims_used += 1

        if phase + 1 < num_phases and active_size > 1:
            var keep = active_size // 2
            if keep < 1:
                keep = 1
            comptime run_halve = gs_halve_active_kernel[
                N_ENVS, MAX_NODES, K_ROOT, K_PAD, dtype
            ]
            var vc_t = LayoutTensor[
                dtype, Layout.row_major(N_ENVS * MAX_NODES * K_PAD),
                MutAnyOrigin,
            ](state.visit_count.unsafe_ptr())
            var tv_t = LayoutTensor[
                dtype, Layout.row_major(N_ENVS * MAX_NODES * K_PAD),
                MutAnyOrigin,
            ](state.total_value.unsafe_ptr())
            var tvis_t = LayoutTensor[
                dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
            ](state.total_visits.unsafe_ptr())
            ctx.enqueue_function[run_halve](
                vc_t,
                tv_t,
                lp_t,
                tvis_t,
                nv_t,
                miq_t,
                mxq_t,
                rg_t,
                ra_t,
                Scalar[DType.int32](active_size),
                Scalar[DType.int32](keep),
                Scalar[dtype](c_visit),
                Scalar[dtype](c_scale),
                grid_dim=(ENV_BLOCKS,),
                block_dim=(TPB,),
            )
            active_size = keep

    # Spend leftover budget on slot 0.
    while sims_used < NUM_SIMULATIONS:
        _run_one_sim_gpu[
            N_ENVS,
            MAX_NODES,
            ACT_DIM,
            K_ROOT,
            K_NON_ROOT,
            K_PAD,
            LATENT,
            BINS,
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
            v_min,
            v_max,
            reward_min,
            reward_max,
            max_action,
            min_std,
            soft_clamp,
            init_std,
            c_visit,
            c_scale,
            gamma,
            rng_seed,
            UInt32(sims_used),
        )
        sims_used += 1

    # Extract chosen action + visit distribution.
    var ca_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * ACT_DIM), MutAnyOrigin
    ](state.chosen_actions.unsafe_ptr())
    var rv_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * K_ROOT), MutAnyOrigin
    ](state.root_visits.unsafe_ptr())
    var vc_extract_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * K_PAD), MutAnyOrigin
    ](state.visit_count.unsafe_ptr())
    comptime run_extract = gs_extract_kernel[
        N_ENVS, MAX_NODES, ACT_DIM, K_ROOT, K_PAD, dtype
    ]
    ctx.enqueue_function[run_extract](
        vc_extract_t,
        act_t,
        ca_t,
        rv_t,
        Scalar[DType.uint8](1 if deterministic else 0),
        rng_seed,
        grid_dim=(ENV_BLOCKS,),
        block_dim=(TPB,),
    )


def _run_one_sim_gpu[
    N_ENVS: Int,
    MAX_NODES: Int,
    ACT_DIM: Int,
    K_ROOT: Int,
    K_NON_ROOT: Int,
    K_PAD: Int,
    LATENT: Int,
    BINS: Int,
    DynModel: Model,
    PredModel: Model,
    DynOpt: Optimizer,
    PredOpt: Optimizer,
](
    ctx: DeviceContext,
    mut state: EZV2GPUSampledMCTSState[
        N_ENVS, MAX_NODES, ACT_DIM, LATENT, BINS, K_ROOT, K_NON_ROOT
    ],
    dyn_state: GPUNetworkState[DynModel, DynOpt],
    pred_state: GPUNetworkState[PredModel, PredOpt],
    workspace_buf: DeviceBuffer[dtype],
    slot: Int,
    v_min: Float64,
    v_max: Float64,
    reward_min: Float64,
    reward_max: Float64,
    max_action: Float64,
    min_std: Float64,
    soft_clamp: Float64,
    init_std: Float64,
    c_visit: Float64,
    c_scale: Float64,
    gamma: Float64,
    rng_seed: UInt32,
    sim_index: UInt32,
) raises:
    """One simulation: select → dyn → pred → expand → backup."""
    comptime PRED_OUT = 2 * ACT_DIM + BINS
    comptime DYN_IN = LATENT + ACT_DIM
    comptime DYN_OUT = LATENT + BINS
    comptime ENV_BLOCKS = (N_ENVS + TPB - 1) // TPB

    # Tensor views (reused across kernels).
    var vc_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * K_PAD), MutAnyOrigin
    ](state.visit_count.unsafe_ptr())
    var tv_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * K_PAD), MutAnyOrigin
    ](state.total_value.unsafe_ptr())
    var lp_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * K_PAD), MutAnyOrigin
    ](state.log_prior.unsafe_ptr())
    var rw_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * K_PAD), MutAnyOrigin
    ](state.reward.unsafe_ptr())
    var ci_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * K_PAD), MutAnyOrigin
    ](state.child_idx.unsafe_ptr())
    var act_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * K_PAD * ACT_DIM),
        MutAnyOrigin,
    ](state.actions.unsafe_ptr())
    var tvis_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ](state.total_visits.unsafe_ptr())
    var nv_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ](state.node_value.unsafe_ptr())
    var ak_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ](state.active_k.unsafe_ptr())
    var nc_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.node_count.unsafe_ptr())
    var miq_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.min_q.unsafe_ptr())
    var mxq_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.max_q.unsafe_ptr())
    var ra_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * K_ROOT), MutAnyOrigin
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
    var pc_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.pending_cand.unsafe_ptr())
    var sp_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_DEPTH), MutAnyOrigin
    ](state.search_paths.unsafe_ptr())
    var cp_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_DEPTH), MutAnyOrigin
    ](state.cand_paths.unsafe_ptr())
    var pl_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.path_lengths.unsafe_ptr())

    # Selection.
    comptime run_select = gs_select_kernel[
        N_ENVS, MAX_NODES, ACT_DIM, K_ROOT, K_PAD, LATENT, DYN_IN, dtype
    ]
    ctx.enqueue_function[run_select](
        vc_t,
        tv_t,
        lp_t,
        ci_t,
        act_t,
        tvis_t,
        nv_t,
        ak_t,
        miq_t,
        mxq_t,
        ra_t,
        hs_t,
        di_t,
        pp_t,
        pc_t,
        sp_t,
        cp_t,
        pl_t,
        Scalar[DType.int32](slot),
        Scalar[dtype](c_visit),
        Scalar[dtype](c_scale),
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

    # Copy LATENT prefix into pred_input + prediction forward.
    comptime run_copy = gs_copy_pred_input_kernel[
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
    comptime run_expand = gs_expand_kernel[
        N_ENVS, MAX_NODES, ACT_DIM, K_ROOT, K_NON_ROOT, K_PAD, LATENT,
        BINS, PRED_OUT, DYN_OUT, dtype,
    ]
    ctx.enqueue_function[run_expand](
        vc_t,
        tv_t,
        lp_t,
        rw_t,
        ci_t,
        act_t,
        tvis_t,
        nv_t,
        ak_t,
        nc_t,
        hs_t,
        pp_t,
        pc_t,
        dyn_out_flat,
        po_full_t,
        lv_t,
        Scalar[dtype](v_min),
        Scalar[dtype](v_max),
        Scalar[dtype](reward_min),
        Scalar[dtype](reward_max),
        Scalar[dtype](max_action),
        Scalar[dtype](min_std),
        Scalar[dtype](soft_clamp),
        Scalar[dtype](init_std),
        rng_seed,
        sim_index,
        grid_dim=(ENV_BLOCKS,),
        block_dim=(TPB,),
    )

    # Backup.
    comptime run_backup = gs_backup_kernel[
        N_ENVS, MAX_NODES, K_PAD, dtype
    ]
    ctx.enqueue_function[run_backup](
        vc_t,
        tv_t,
        rw_t,
        tvis_t,
        miq_t,
        mxq_t,
        sp_t,
        cp_t,
        pl_t,
        lv_t,
        Scalar[dtype](gamma),
        grid_dim=(ENV_BLOCKS,),
        block_dim=(TPB,),
    )


# ═════════════════════════════════════════════════════════════════════════
# Host helpers
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
