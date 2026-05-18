"""Phase 3b: ``GenericGPUMCTS`` vs legacy inline orchestration — bit parity.

Mirrors the CPU bit-parity test (`test_mcts_cpu_parity_muzero.mojo`) for
the GPU pipeline. Both paths use:

* The same kernels (``gpu_mcts_init_root_kernel``,
  ``gpu_mcts_batched_select_and_build_dyn_kernel``,
  ``gpu_mcts_batched_expand_backup_muzero_kernel``,
  ``gpu_mcts_extract_actions_kernel``,
  ``gpu_mcts_extract_root_value_kernel``).
* The same stub ``Sequential[Linear]`` networks for representation /
  dynamics / prediction. Params are constructed once and shared via
  ``enqueue_copy`` into a second ``GPUNetworkState`` set so both paths
  see byte-identical weights.
* The same RNG seed for ``init_root``'s Dirichlet noise (NoNoise here ⇒
  fraction=0, but the noise sampler still touches the buffer with the
  given seed).
* The same gamma / v_min / v_max / PUCT constants.

The legacy "path" is inlined directly into this test, exactly mirroring
the kernel sequence in ``muzero.mojo``'s training loop (~250 lines). The
purpose is to lock in that ``GenericGPUMCTS`` is byte-equivalent to the
inline orchestration before the agent rewires use it.

What we compare (byte-level Float32 equality on host):

* ``visit_count``     [N_ENVS × MAX_NODES × ACT]
* ``total_value``     [N_ENVS × MAX_NODES × ACT]
* ``prior``           [N_ENVS × MAX_NODES × ACT]
* ``reward``          [N_ENVS × MAX_NODES × ACT]
* ``child_idx``       [N_ENVS × MAX_NODES × ACT]
* ``total_visits``    [N_ENVS × MAX_NODES]
* ``node_count``      [N_ENVS]
* ``min_q`` / ``max_q``  [N_ENVS]
* ``hidden_states``   [N_ENVS × MAX_NODES × LATENT]
* ``actions_out``     [N_ENVS]
* ``policies_out``    [N_ENVS × ACT]
* ``root_value_out``  [N_ENVS]

Usage:
    pixi run -e apple mojo run -I . tests/planners/tree_search/test_mcts_gpu_parity_muzero.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.testing import assert_true, assert_equal
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Linear, Sequential
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.training import Network, GPUNetworkState

from mojo_rl.planners.tree_search import (
    GenericGPUMCTS,
    RepresentationGPU,
    DynamicsGPU,
    PredictionGPU,
    MuZeroPUCT,
    NoNoise,
    SinglePlayer,
    TPB,
    MAX_DEPTH,
    GPUMCTSState,
    gpu_mcts_init_root_kernel,
    gpu_mcts_batched_select_and_build_dyn_kernel,
    gpu_mcts_batched_expand_backup_muzero_kernel,
    gpu_mcts_extract_actions_kernel,
    gpu_mcts_extract_root_value_kernel,
    mcts_gpu_scale_hidden_kernel,
    mcts_gpu_extract_hidden_kernel,
)


# ─── Tiny config (kept small for fast parity verification) ────────────────


comptime OBS: Int = 4
comptime ACT: Int = 2
comptime LATENT: Int = 4
comptime BINS: Int = 3
comptime DYN_IN: Int = LATENT + ACT
comptime DYN_OUT: Int = LATENT + BINS
comptime PRED_OUT: Int = ACT + BINS
comptime N_ENVS: Int = 2
comptime MAX_NODES: Int = 32
comptime BATCH_SIMS: Int = 4
comptime NUM_SIMS: Int = 16  # = 4 rounds × 4 sims
comptime MCTS_ROUNDS: Int = NUM_SIMS // BATCH_SIMS
comptime MCTS_TOTAL: Int = N_ENVS * BATCH_SIMS
comptime ENV_BLOCKS: Int = (N_ENVS + TPB - 1) // TPB
comptime EXTR_TOTAL: Int = MCTS_TOTAL * LATENT
comptime EXTR_BLK: Int = (EXTR_TOTAL + TPB - 1) // TPB

comptime RepModel = Sequential[Linear[OBS, LATENT]]
comptime DynModel = Sequential[Linear[DYN_IN, DYN_OUT]]
comptime PredModel = Sequential[Linear[LATENT, PRED_OUT]]
comptime OptType = Adam[LR=1e-3]


# ─── GPU trait adapters (identical to orchestrator smoke test) ────────────


@fieldwise_init
struct StubRepGPU(Movable, ImplicitlyDestructible, RepresentationGPU):
    comptime OBS_DIM: Int = OBS
    comptime LATENT_DIM: Int = LATENT

    var params: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var model_state: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var workspace: DeviceBuffer[dtype]

    def encode_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        obs: LayoutTensor[
            dtype, Layout.row_major(B, Self.OBS_DIM), MutAnyOrigin
        ],
        mut hidden_out: LayoutTensor[
            dtype, Layout.row_major(B, Self.LATENT_DIM), MutAnyOrigin
        ],
    ) raises:
        var p_t = LayoutTensor[
            dtype, Layout.row_major(RepModel.PARAM_SIZE), MutAnyOrigin
        ](self.params)
        var s_t = LayoutTensor[
            dtype, Layout.row_major(RepModel.STATE_SIZE), MutAnyOrigin
        ](self.model_state)
        Network[RepModel, OptType].forward_gpu[B](
            ctx, obs, hidden_out, p_t, s_t, self.workspace
        )


@fieldwise_init
struct StubDynGPU(Movable, ImplicitlyDestructible, DynamicsGPU):
    comptime LATENT_DIM: Int = LATENT
    comptime ACTION_DIM: Int = ACT
    comptime DYN_IN_DIM: Int = DYN_IN
    comptime DYN_OUT_DIM: Int = DYN_OUT

    var params: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var model_state: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var workspace: DeviceBuffer[dtype]

    def step_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        dyn_in: LayoutTensor[
            dtype, Layout.row_major(B, Self.DYN_IN_DIM), MutAnyOrigin
        ],
        mut dyn_out: LayoutTensor[
            dtype, Layout.row_major(B, Self.DYN_OUT_DIM), MutAnyOrigin
        ],
    ) raises:
        var p_t = LayoutTensor[
            dtype, Layout.row_major(DynModel.PARAM_SIZE), MutAnyOrigin
        ](self.params)
        var s_t = LayoutTensor[
            dtype, Layout.row_major(DynModel.STATE_SIZE), MutAnyOrigin
        ](self.model_state)
        Network[DynModel, OptType].forward_gpu[B](
            ctx, dyn_in, dyn_out, p_t, s_t, self.workspace
        )


@fieldwise_init
struct StubPredGPU(Movable, ImplicitlyDestructible, PredictionGPU):
    comptime LATENT_DIM: Int = LATENT
    comptime ACTION_DIM: Int = ACT
    comptime PRED_OUT_DIM: Int = PRED_OUT

    var params: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var model_state: UnsafePointer[Scalar[dtype], MutAnyOrigin]
    var workspace: DeviceBuffer[dtype]

    def predict_gpu[B: Int](
        mut self,
        ctx: DeviceContext,
        hidden: LayoutTensor[
            dtype, Layout.row_major(B, Self.LATENT_DIM), MutAnyOrigin
        ],
        mut pred_out: LayoutTensor[
            dtype, Layout.row_major(B, Self.PRED_OUT_DIM), MutAnyOrigin
        ],
    ) raises:
        var p_t = LayoutTensor[
            dtype, Layout.row_major(PredModel.PARAM_SIZE), MutAnyOrigin
        ](self.params)
        var s_t = LayoutTensor[
            dtype, Layout.row_major(PredModel.STATE_SIZE), MutAnyOrigin
        ](self.model_state)
        Network[PredModel, OptType].forward_gpu[B](
            ctx, hidden, pred_out, p_t, s_t, self.workspace
        )


# ─── Helpers ──────────────────────────────────────────────────────────────


@always_inline
def _max3(a: Int, b: Int, c: Int) -> Int:
    var m = a if a > b else b
    return m if m > c else c


# ─── Legacy inline orchestration ──────────────────────────────────────────


def run_legacy_inline_search(
    mut ctx: DeviceContext,
    mut state: GPUMCTSState[
        N_ENVS, MAX_NODES, ACT, LATENT, BINS, 0, BATCH_SIMS
    ],
    mut actions_out: DeviceBuffer[dtype],
    mut policies_out: DeviceBuffer[dtype],
    mut root_value_out: DeviceBuffer[dtype],
    mut rep: StubRepGPU,
    mut dyn: StubDynGPU,
    mut pred: StubPredGPU,
    obs: LayoutTensor[dtype, Layout.row_major(N_ENVS, OBS), MutAnyOrigin],
    rng_seed: UInt32,
    gamma: Float64,
    v_min: Float64,
    v_max: Float64,
) raises:
    """Inline kernel-by-kernel orchestration that mirrors
    ``muzero.mojo``'s training-loop MCTS path. Used as the reference for
    ``GenericGPUMCTS.search_gpu`` parity.
    """

    # ── 1. Root encode → hidden_states[node 0] ───────────────────────
    var hidden_root = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, LATENT), MutAnyOrigin
    ](state.hidden_states.unsafe_ptr())
    rep.encode_gpu[N_ENVS](ctx, obs, hidden_root)

    # 1a. Post-encode min-max scaling.
    comptime BATCH_BLOCKS = (N_ENVS + TPB - 1) // TPB
    var hidden_root_flat = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * LATENT), MutAnyOrigin
    ](state.hidden_states.unsafe_ptr())
    comptime run_scale_root = mcts_gpu_scale_hidden_kernel[
        N_ENVS, LATENT, dtype
    ]
    ctx.enqueue_function[run_scale_root](
        hidden_root_flat,
        grid_dim=(BATCH_BLOCKS,),
        block_dim=(TPB,),
    )

    # ── 2. Root predict ──────────────────────────────────────────────
    var pred_root_in = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, LATENT), MutAnyOrigin
    ](state.hidden_states.unsafe_ptr())
    var pred_root_out = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, PRED_OUT), MutAnyOrigin
    ](state.pred_output.unsafe_ptr())
    pred.predict_gpu[N_ENVS](ctx, pred_root_in, pred_root_out)

    # ── 3. Zero tree + init root ─────────────────────────────────────
    state.zero_tree(ctx)

    var vc_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](state.visit_count.unsafe_ptr())
    var tv_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](state.total_value.unsafe_ptr())
    var pr_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](state.prior.unsafe_ptr())
    var rw_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](state.reward.unsafe_ptr())
    var ci_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * ACT), MutAnyOrigin
    ](state.child_idx.unsafe_ptr())
    var tvis_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ](state.total_visits.unsafe_ptr())
    var nc_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.node_count.unsafe_ptr())
    var po_root_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * PRED_OUT), MutAnyOrigin
    ](state.pred_output.unsafe_ptr())
    var miq_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.min_q.unsafe_ptr())
    var mxq_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](state.max_q.unsafe_ptr())

    comptime run_init = gpu_mcts_init_root_kernel[
        N_ENVS, MAX_NODES, ACT, LATENT, PRED_OUT, dtype
    ]
    # NoNoise ⇒ fraction = 0 (matches `GenericGPUMCTS` with NoNoise).
    ctx.enqueue_function[run_init](
        vc_t, tv_t, pr_t, rw_t, ci_t, tvis_t, nc_t, po_root_t,
        miq_t, mxq_t,
        Scalar[dtype](0.0),
        rng_seed,
        grid_dim=(ENV_BLOCKS,),
        block_dim=(TPB,),
    )

    # ── 4. Simulation rounds ─────────────────────────────────────────
    var hs_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES * LATENT), MutAnyOrigin
    ](state.hidden_states.unsafe_ptr())
    var b_pp = LayoutTensor[
        dtype, Layout.row_major(MCTS_TOTAL), MutAnyOrigin
    ](state.pending_parent.unsafe_ptr())
    var b_pa = LayoutTensor[
        dtype, Layout.row_major(MCTS_TOTAL), MutAnyOrigin
    ](state.pending_action.unsafe_ptr())
    var b_sp = LayoutTensor[
        dtype, Layout.row_major(MCTS_TOTAL * MAX_DEPTH), MutAnyOrigin
    ](state.search_paths.unsafe_ptr())
    var b_ap = LayoutTensor[
        dtype, Layout.row_major(MCTS_TOTAL * MAX_DEPTH), MutAnyOrigin
    ](state.action_paths.unsafe_ptr())
    var b_pl = LayoutTensor[
        dtype, Layout.row_major(MCTS_TOTAL), MutAnyOrigin
    ](state.path_lengths.unsafe_ptr())
    var b_di = LayoutTensor[
        dtype, Layout.row_major(MCTS_TOTAL * DYN_IN), MutAnyOrigin
    ](state.dyn_input.unsafe_ptr())

    for _round in range(MCTS_ROUNDS):
        # 4a. select + build dyn
        comptime run_sel_dyn = gpu_mcts_batched_select_and_build_dyn_kernel[
            N_ENVS, MAX_NODES, ACT, BATCH_SIMS, LATENT, DYN_IN, dtype
        ]
        ctx.enqueue_function[run_sel_dyn](
            vc_t, tv_t, pr_t, ci_t, tvis_t, nc_t, miq_t, mxq_t, hs_t,
            b_di, b_pp, b_pa, b_sp, b_ap, b_pl,
            Scalar[dtype](MuZeroPUCT[].C_BASE),
            Scalar[dtype](MuZeroPUCT[].C_INIT),
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # 4b. dynamics forward
        var dyn_in_b = LayoutTensor[
            dtype, Layout.row_major(MCTS_TOTAL, DYN_IN), MutAnyOrigin
        ](state.dyn_input.unsafe_ptr())
        var dyn_out_b = LayoutTensor[
            dtype, Layout.row_major(MCTS_TOTAL, DYN_OUT), MutAnyOrigin
        ](state.dyn_output.unsafe_ptr())
        dyn.step_gpu[MCTS_TOTAL](ctx, dyn_in_b, dyn_out_b)

        # 4c. extract hidden → pred input
        var pred_in_b = LayoutTensor[
            dtype, Layout.row_major(MCTS_TOTAL * LATENT), MutAnyOrigin
        ](state.pred_input.unsafe_ptr())
        var dyn_out_b_flat = LayoutTensor[
            dtype, Layout.row_major(MCTS_TOTAL * DYN_OUT), MutAnyOrigin
        ](state.dyn_output.unsafe_ptr())
        comptime run_extr = mcts_gpu_extract_hidden_kernel[
            MCTS_TOTAL, LATENT, DYN_OUT, dtype
        ]
        ctx.enqueue_function[run_extr](
            pred_in_b, dyn_out_b_flat,
            grid_dim=(EXTR_BLK,),
            block_dim=(TPB,),
        )

        # 4d. prediction forward
        var pred_in_net = LayoutTensor[
            dtype, Layout.row_major(MCTS_TOTAL, LATENT), MutAnyOrigin
        ](state.pred_input.unsafe_ptr())
        var pred_out_net = LayoutTensor[
            dtype, Layout.row_major(MCTS_TOTAL, PRED_OUT), MutAnyOrigin
        ](state.pred_output.unsafe_ptr())
        pred.predict_gpu[MCTS_TOTAL](ctx, pred_in_net, pred_out_net)

        # 4e. expand + backup
        var b_do = LayoutTensor[
            dtype, Layout.row_major(MCTS_TOTAL * DYN_OUT), MutAnyOrigin
        ](state.dyn_output.unsafe_ptr())
        var b_po = LayoutTensor[
            dtype, Layout.row_major(MCTS_TOTAL * PRED_OUT), MutAnyOrigin
        ](state.pred_output.unsafe_ptr())
        comptime run_exp_bk = gpu_mcts_batched_expand_backup_muzero_kernel[
            N_ENVS, MAX_NODES, ACT, BATCH_SIMS, LATENT, PRED_OUT, DYN_OUT,
            dtype,
        ]
        ctx.enqueue_function[run_exp_bk](
            vc_t, tv_t, pr_t, rw_t, ci_t, tvis_t, nc_t, miq_t, mxq_t, hs_t,
            b_pp, b_pa, b_do, b_po, b_sp, b_ap, b_pl,
            Scalar[dtype](v_min),
            Scalar[dtype](v_max),
            Scalar[dtype](gamma),
            Scalar[DType.bool](False),
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )

    # ── 5. Extract actions + policies ────────────────────────────────
    var act_out_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](actions_out.unsafe_ptr())
    var pol_out_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS * ACT), MutAnyOrigin
    ](policies_out.unsafe_ptr())
    comptime run_act = gpu_mcts_extract_actions_kernel[
        N_ENVS, MAX_NODES, ACT, dtype
    ]
    ctx.enqueue_function[run_act](
        vc_t, act_out_t, pol_out_t,
        grid_dim=(ENV_BLOCKS,),
        block_dim=(TPB,),
    )

    # ── 6. Extract root value ────────────────────────────────────────
    var rv_out_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ](root_value_out.unsafe_ptr())
    comptime run_rv = gpu_mcts_extract_root_value_kernel[
        N_ENVS, MAX_NODES, ACT, dtype
    ]
    ctx.enqueue_function[run_rv](
        vc_t, tv_t, rv_out_t,
        grid_dim=(ENV_BLOCKS,),
        block_dim=(TPB,),
    )


# ─── Buffer-equality helpers ──────────────────────────────────────────────


def _copy_to_host(
    mut ctx: DeviceContext, buf: DeviceBuffer[dtype], size: Int
) raises -> List[Float64]:
    """Snapshot a device buffer to a Python-side List[Float64] for
    byte-level comparison.
    """
    var host = ctx.enqueue_create_host_buffer[dtype](size)
    ctx.enqueue_copy(host, buf)
    ctx.synchronize()
    var out = List[Float64]()
    for i in range(size):
        out.append(Float64(host[i]))
    return out^


def _assert_buffers_equal(
    name: String, a: List[Float64], b: List[Float64]
) raises:
    assert_equal(
        len(a), len(b), "buffer " + name + " length mismatch"
    )
    for i in range(len(a)):
        if a[i] != b[i]:
            assert_true(
                False,
                "buffer "
                + name
                + " mismatch at index "
                + String(i)
                + ": legacy="
                + String(a[i])
                + " new="
                + String(b[i]),
            )


# ─── The test ─────────────────────────────────────────────────────────────


def test_gpu_parity_muzero_inline() raises:
    """Drive both ``GenericGPUMCTS`` and the inline legacy-style path on
    the same inputs / params / RNG seed and assert byte-identical
    tree state + outputs.
    """
    var ctx = DeviceContext()

    # ── Networks (single copy of params, shared by both paths) ────────
    comptime BATCHED: Int = N_ENVS * BATCH_SIMS
    var ws_per_sample = _max3(
        RepModel.WORKSPACE_SIZE_PER_SAMPLE,
        DynModel.WORKSPACE_SIZE_PER_SAMPLE,
        PredModel.WORKSPACE_SIZE_PER_SAMPLE,
    )
    if ws_per_sample <= 0:
        ws_per_sample = 1
    var workspace = ctx.enqueue_create_buffer[dtype](BATCHED * ws_per_sample)

    var rep_state = GPUNetworkState[RepModel, OptType](ctx)
    var dyn_state = GPUNetworkState[DynModel, OptType](ctx)
    var pred_state = GPUNetworkState[PredModel, OptType](ctx)

    # Same GPUNetworkState instance underlies both paths — params buffer
    # identity guarantees the two orchestrations see byte-identical
    # weights without any extra copying.
    var rep = StubRepGPU(
        params=rep_state.params_buf.unsafe_ptr(),
        model_state=rep_state.model_state_buf.unsafe_ptr(),
        workspace=workspace,
    )
    var dyn = StubDynGPU(
        params=dyn_state.params_buf.unsafe_ptr(),
        model_state=dyn_state.model_state_buf.unsafe_ptr(),
        workspace=workspace,
    )
    var pred = StubPredGPU(
        params=pred_state.params_buf.unsafe_ptr(),
        model_state=pred_state.model_state_buf.unsafe_ptr(),
        workspace=workspace,
    )

    # ── Common obs input ──────────────────────────────────────────────
    var obs_buf = ctx.enqueue_create_buffer[dtype](N_ENVS * OBS)
    var obs_host = ctx.enqueue_create_host_buffer[dtype](N_ENVS * OBS)
    for e in range(N_ENVS):
        for d in range(OBS):
            obs_host[e * OBS + d] = Scalar[dtype](
                0.1 if (e == 0) else 0.5
            )
    ctx.enqueue_copy(obs_buf, obs_host)
    var obs_t = LayoutTensor[
        dtype, Layout.row_major(N_ENVS, OBS), MutAnyOrigin
    ](obs_buf.unsafe_ptr())

    var seed = UInt32(7)
    var gamma = Float64(0.997)
    var v_min = Float64(-5.0)
    var v_max = Float64(5.0)

    # ── Path A: GenericGPUMCTS ────────────────────────────────────────
    var planner = GenericGPUMCTS[
        N_ENVS, ACT, LATENT, BINS, MAX_NODES, NUM_SIMS, BATCH_SIMS,
        MuZeroPUCT[],
        NoNoise,
        SinglePlayer,
    ](ctx, gamma=gamma, v_min=v_min, v_max=v_max)
    planner.search_gpu[StubRepGPU, StubDynGPU, StubPredGPU](
        ctx, rep, dyn, pred, obs_t, rng_seed=seed,
    )
    ctx.synchronize()

    # Snapshot every tree field + output buffer for path A.
    comptime NA = N_ENVS * MAX_NODES * ACT
    comptime NN = N_ENVS * MAX_NODES
    comptime HS = N_ENVS * MAX_NODES * LATENT
    var a_visit_count = _copy_to_host(ctx, planner.state.visit_count, NA)
    var a_total_value = _copy_to_host(ctx, planner.state.total_value, NA)
    var a_prior = _copy_to_host(ctx, planner.state.prior, NA)
    var a_reward = _copy_to_host(ctx, planner.state.reward, NA)
    var a_child_idx = _copy_to_host(ctx, planner.state.child_idx, NA)
    var a_total_visits = _copy_to_host(ctx, planner.state.total_visits, NN)
    var a_node_count = _copy_to_host(ctx, planner.state.node_count, N_ENVS)
    var a_min_q = _copy_to_host(ctx, planner.state.min_q, N_ENVS)
    var a_max_q = _copy_to_host(ctx, planner.state.max_q, N_ENVS)
    var a_hidden = _copy_to_host(ctx, planner.state.hidden_states, HS)
    var a_actions = _copy_to_host(ctx, planner.actions_out, N_ENVS)
    var a_policies = _copy_to_host(ctx, planner.policies_out, N_ENVS * ACT)
    var a_rv = _copy_to_host(ctx, planner.root_value_out, N_ENVS)

    # ── Path B: inline legacy orchestration ───────────────────────────
    # Fresh GPUMCTSState + output buffers so we don't reuse path A's
    # device memory and accidentally pass parity by aliasing.
    var legacy_state = GPUMCTSState[
        N_ENVS, MAX_NODES, ACT, LATENT, BINS, 0, BATCH_SIMS
    ](ctx)
    var legacy_actions = ctx.enqueue_create_buffer[dtype](N_ENVS)
    var legacy_policies = ctx.enqueue_create_buffer[dtype](N_ENVS * ACT)
    var legacy_rv = ctx.enqueue_create_buffer[dtype](N_ENVS)

    run_legacy_inline_search(
        ctx, legacy_state, legacy_actions, legacy_policies, legacy_rv,
        rep, dyn, pred, obs_t,
        rng_seed=seed, gamma=gamma, v_min=v_min, v_max=v_max,
    )
    ctx.synchronize()

    var b_visit_count = _copy_to_host(ctx, legacy_state.visit_count, NA)
    var b_total_value = _copy_to_host(ctx, legacy_state.total_value, NA)
    var b_prior = _copy_to_host(ctx, legacy_state.prior, NA)
    var b_reward = _copy_to_host(ctx, legacy_state.reward, NA)
    var b_child_idx = _copy_to_host(ctx, legacy_state.child_idx, NA)
    var b_total_visits = _copy_to_host(ctx, legacy_state.total_visits, NN)
    var b_node_count = _copy_to_host(ctx, legacy_state.node_count, N_ENVS)
    var b_min_q = _copy_to_host(ctx, legacy_state.min_q, N_ENVS)
    var b_max_q = _copy_to_host(ctx, legacy_state.max_q, N_ENVS)
    var b_hidden = _copy_to_host(ctx, legacy_state.hidden_states, HS)
    var b_actions = _copy_to_host(ctx, legacy_actions, N_ENVS)
    var b_policies = _copy_to_host(ctx, legacy_policies, N_ENVS * ACT)
    var b_rv = _copy_to_host(ctx, legacy_rv, N_ENVS)

    # ── Compare every buffer byte-for-byte ────────────────────────────
    _assert_buffers_equal("visit_count", b_visit_count, a_visit_count)
    _assert_buffers_equal("total_value", b_total_value, a_total_value)
    _assert_buffers_equal("prior", b_prior, a_prior)
    _assert_buffers_equal("reward", b_reward, a_reward)
    _assert_buffers_equal("child_idx", b_child_idx, a_child_idx)
    _assert_buffers_equal("total_visits", b_total_visits, a_total_visits)
    _assert_buffers_equal("node_count", b_node_count, a_node_count)
    _assert_buffers_equal("min_q", b_min_q, a_min_q)
    _assert_buffers_equal("max_q", b_max_q, a_max_q)
    _assert_buffers_equal("hidden_states", b_hidden, a_hidden)
    _assert_buffers_equal("actions_out", b_actions, a_actions)
    _assert_buffers_equal("policies_out", b_policies, a_policies)
    _assert_buffers_equal("root_value_out", b_rv, a_rv)

    # Sanity: visit counts at root must sum to NUM_SIMS per env.
    for e in range(N_ENVS):
        var s: Int = 0
        for a in range(ACT):
            s += Int(a_visit_count[e * MAX_NODES * ACT + a])
        assert_equal(
            s, NUM_SIMS, "env " + String(e) + " visit sum != NUM_SIMS"
        )


def main() raises:
    print("=== Phase 3b: GenericGPUMCTS vs inline legacy — bit parity ===")
    test_gpu_parity_muzero_inline()
    print(
        "  PASS GenericGPUMCTS byte-identical to inline kernel"
        " sequence (visit_count + total_value + prior + reward +"
        " child_idx + total_visits + node_count + min_q + max_q +"
        " hidden + actions + policies + root_value)"
    )
    print("OK")
