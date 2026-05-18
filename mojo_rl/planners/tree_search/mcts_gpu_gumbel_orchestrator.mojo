"""GumbelGPUMCTS — discrete-action Gumbel-MCTS orchestrator for EZv2.

The agent-side counterpart of ``GenericGPUMCTS`` for the EfficientZero V2
discrete planner. Owns an ``EZV2GPUMCTSState`` plus a ``root_value_out``
device buffer, exposes ``search_gpu[REP, DYN, PRED]`` that drives the
full encode → predict (root) → init_root (Gumbel-Top-k) → Sequential
Halving phases → extract policy pipeline.

Same trait surface as ``GenericGPUMCTS`` (``RepresentationGPU`` /
``DynamicsGPU`` / ``PredictionGPU``), so the EZv2 agent's existing
GPUNetworkState + the EZv2 trait adapters drop into the orchestrator
without any agent-side network code changes.

What's different vs ``GenericGPUMCTS.search_gpu``:
  • Selection rule: deterministic visit-balance
    ``argmax_a [π_improved(a) − N(s,a)/(1+ΣN(s,b))]`` instead of PUCT.
  • Root expansion: restricted to ``K`` candidates via Gumbel-Top-k —
    ``init_root`` samples them, ``halve_active`` halves the active set
    between Sequential Halving phases.
  • Sims structure: ``log2(K)`` host-orchestrated phases; per-action
    budget is ``NUM_SIMULATIONS / num_phases / active_size``. Leftover
    sims spend on slot 0 of the final survivor.
  • No virtual-loss batching — sims run one at a time within a phase.
    Kept verbatim from ``run_gumbel_search_gpu`` so the EZv2 perf
    profile is unchanged through the rewiring.
  • Policy readout: ``gz_extract_policy_kernel`` builds the
    ``π̂ = softmax(logits + σ(completed_Q))`` improved policy. The agent
    samples (or argmaxes) from it host-side — the orchestrator does NOT
    write a separate ``actions_out`` because EZv2 selects stochastically
    during data collection.

Output buffers exposed:
  • ``policies_view()`` → ``[N_ENVS × ACT]`` improved policy.
  • ``root_value_view()`` → ``[N_ENVS]`` scalar root value (scattered
    from ``state.node_value[e * MAX_NODES]``).
  • ``legal_mask_view()`` → ``[N_ENVS × ACT]``; the caller populates it
    before ``search_gpu`` when calling with ``apply_legal=True``.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype, TPB

from .model_traits_gpu import (
    RepresentationGPU,
    DynamicsGPU,
    PredictionGPU,
)
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


# ═════════════════════════════════════════════════════════════════════════
# Helpers (moved from `run_gumbel_search_gpu` so the orchestrator method is
# pure orchestration). Kept module-private so the legacy driver still
# imports its own copies from `mcts_gpu_gumbel.mojo`.
# ═════════════════════════════════════════════════════════════════════════


def _ilog2(n: Int) -> Int:
    var x = n
    var r = 0
    while x > 1:
        x = x // 2
        r += 1
    return r


def _largest_power_of_two_le(n: Int) -> Int:
    if n < 1:
        return 1
    var p = 1
    while p * 2 <= n:
        p *= 2
    return p


# ═════════════════════════════════════════════════════════════════════════
# Root-value extraction kernel
# ═════════════════════════════════════════════════════════════════════════
#
# Lives next to the orchestrator (not in `mcts_gpu_gumbel.mojo`) because it
# only exists to populate the orchestrator's `root_value_out` buffer — the
# legacy driver writes nothing equivalent.


def gz_extract_root_value_kernel[
    N_ENVS: Int, MAX_NODES: Int, dtype: DType,
](
    node_value: LayoutTensor[
        dtype, Layout.row_major(N_ENVS * MAX_NODES), MutAnyOrigin
    ],
    root_value_out: LayoutTensor[
        dtype, Layout.row_major(N_ENVS), MutAnyOrigin
    ],
) where dtype.is_floating_point():
    """Scatter ``node_value[e * MAX_NODES + 0]`` into ``root_value_out[e]``.
    One thread per env. Decoded scalar root value was already produced by
    ``gz_init_root_kernel``; this just hoists it for callers that don't
    want to walk the per-env stride."""
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return
    root_value_out[e] = node_value[e * MAX_NODES]


# ═════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═════════════════════════════════════════════════════════════════════════


struct GumbelGPUMCTS[
    N_ENVS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    MAX_NODES: Int,
    MAX_K: Int,
    NUM_SIMULATIONS: Int,
](Movable, ImplicitlyDestructible):
    """GPU Gumbel-MCTS orchestrator (EZv2 discrete planner).

    Comptime params:
        N_ENVS: Number of parallel envs / trees.
        ACT: Discrete action count.
        LATENT: Hidden state dim.
        BINS: Categorical reward / value bins.
        MAX_NODES: Per-tree node arena size.
        MAX_K: Max Gumbel-Top-k root candidates (must be a power of two
            and ≤ ``ACT``; the driver clips at runtime).
        NUM_SIMULATIONS: Total sims per ``search_gpu`` call. Budget split
            across ``log2(K)`` phases by Sequential Halving.

    Runtime ctor args:
        gamma, v_min, v_max — categorical decode + discounting.
        c_visit, c_scale — π-improvement σ(Q) scaling (paper defaults
            50.0 and 0.1).
    """

    comptime PRED_OUT: Int = Self.ACT + Self.BINS
    comptime DYN_IN: Int = Self.LATENT + Self.ACT
    comptime DYN_OUT: Int = Self.LATENT + Self.BINS
    comptime ENV_BLOCKS: Int = (Self.N_ENVS + TPB - 1) // TPB

    var state: EZV2GPUMCTSState[
        Self.N_ENVS, Self.MAX_NODES, Self.ACT, Self.LATENT, Self.BINS, Self.MAX_K,
    ]
    """Per-env trees + scratch (legal mask, root candidates, search
    paths, network I/O staging, ``policies_out``). See
    ``EZV2GPUMCTSState`` for field-by-field layout."""

    var root_value_out: DeviceBuffer[dtype]
    """``[N_ENVS]`` scalar root value scattered from
    ``state.node_value[e * MAX_NODES]`` (decoded by
    ``gz_init_root_kernel``)."""

    var gamma: Float64
    var v_min: Float64
    var v_max: Float64
    var c_visit: Float64
    var c_scale: Float64

    def __init__(
        out self,
        ctx: DeviceContext,
        gamma: Float64 = 0.997,
        v_min: Float64 = -10.0,
        v_max: Float64 = 10.0,
        c_visit: Float64 = 50.0,
        c_scale: Float64 = 0.1,
    ) raises:
        if Self.MAX_K > Self.ACT:
            raise Error("GumbelGPUMCTS: MAX_K must be <= ACT")
        if Self.MAX_K < 1:
            raise Error("GumbelGPUMCTS: MAX_K must be >= 1")
        self.state = EZV2GPUMCTSState[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, Self.LATENT, Self.BINS,
            Self.MAX_K,
        ](ctx)
        self.root_value_out = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.gamma = gamma
        self.v_min = v_min
        self.v_max = v_max
        self.c_visit = c_visit
        self.c_scale = c_scale

    def __init__(out self, *, deinit take: Self):
        self.state = take.state^
        self.root_value_out = take.root_value_out^
        self.gamma = take.gamma
        self.v_min = take.v_min
        self.v_max = take.v_max
        self.c_visit = take.c_visit
        self.c_scale = take.c_scale

    # ══════════════════════════════════════════════════════════════════════
    # Views
    # ══════════════════════════════════════════════════════════════════════

    def policies_view(self) -> DeviceBuffer[dtype]:
        """``[N_ENVS × ACT]`` improved-policy distribution."""
        return self.state.policies_out

    def root_value_view(self) -> DeviceBuffer[dtype]:
        """``[N_ENVS]`` scalar root value (decoded from value bins at
        init_root time)."""
        return self.root_value_out

    def legal_mask_view(self) -> DeviceBuffer[dtype]:
        """``[N_ENVS × ACT]`` legal-action mask; caller populates before
        ``search_gpu(apply_legal=True)`` and re-reads any time."""
        return self.state.legal_mask

    # ══════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════

    def search_gpu[
        REP: RepresentationGPU,
        DYN: DynamicsGPU,
        PRED: PredictionGPU,
    ](
        mut self,
        ctx: DeviceContext,
        mut rep: REP,
        mut dyn: DYN,
        mut pred: PRED,
        obs: LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS, REP.OBS_DIM), MutAnyOrigin
        ],
        apply_legal: Bool = False,
        k_actual: Int = Self.MAX_K,
        rng_seed: UInt32 = UInt32(0),
    ) raises:
        """Run Gumbel-MCTS for all ``N_ENVS`` envs in parallel.

        Pipeline (mirrors ``run_gumbel_search_gpu``):
          1. ``rep.encode_gpu`` → root hidden (contiguous ``[N_ENVS × LATENT]``)
          2. ``pred.predict_gpu`` → root pred output (logits + value bins)
          3. Scatter root hidden into ``state.hidden_states[e][0]``
          4. ``gz_init_root_kernel`` — logits + Gumbel-Top-k + decoded value
          5. Sequential Halving: ``log2(K)`` phases, each
             ``per_phase_budget // active_size`` sims per slot, then
             ``gz_halve_active_kernel``.
             Each sim: ``gz_select_kernel`` → ``dyn.step_gpu`` →
             ``gz_copy_pred_input_kernel`` → ``pred.predict_gpu`` →
             ``gz_expand_kernel`` → ``gz_backup_kernel``.
          6. Leftover sims on slot 0 of the size-1 survivor.
          7. ``gz_extract_policy_kernel`` → improved policy.
          8. ``gz_extract_root_value_kernel`` → root_value_out.

        ``apply_legal=True`` reads the caller-populated
        ``state.legal_mask`` and applies it inside ``init_root`` (Gumbel
        sampling skips illegal actions) and ``extract_policy``.

        ``k_actual`` is clipped to ``[1, MAX_K]`` and rounded down to a
        power of two by the driver.
        """

        # ── 0. Reset tree ────────────────────────────────────────────────
        self.state.zero_tree(ctx)

        # ── 1. Rep forward ───────────────────────────────────────────────
        var root_hidden_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS, REP.LATENT_DIM), MutAnyOrigin,
        ](self.state.root_hidden.unsafe_ptr())
        rep.encode_gpu[Self.N_ENVS](ctx, obs, root_hidden_t)

        # ── 2. Pred forward at the root ──────────────────────────────────
        var pred_root_in = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS, PRED.LATENT_DIM), MutAnyOrigin,
        ](self.state.root_hidden.unsafe_ptr())
        var pred_root_out = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS, PRED.PRED_OUT_DIM), MutAnyOrigin,
        ](self.state.pred_output.unsafe_ptr())
        pred.predict_gpu[Self.N_ENVS](ctx, pred_root_in, pred_root_out)

        # ── 3. Scatter root_hidden → hidden_states[e][0] ─────────────────
        var rh_flat = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.LATENT), MutAnyOrigin
        ](self.state.root_hidden.unsafe_ptr())
        var hs_flat = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.LATENT),
            MutAnyOrigin,
        ](self.state.hidden_states.unsafe_ptr())
        comptime run_scatter = gz_scatter_root_hidden_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.LATENT, dtype,
        ]
        ctx.enqueue_function[run_scatter](
            rh_flat,
            hs_flat,
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── 4. Init root ─────────────────────────────────────────────────
        var nl_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
            MutAnyOrigin,
        ](self.state.node_logits.unsafe_ptr())
        var nv_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.MAX_NODES), MutAnyOrigin
        ](self.state.node_value.unsafe_ptr())
        var nc_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.node_count.unsafe_ptr())
        var miq_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.min_q.unsafe_ptr())
        var mxq_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.max_q.unsafe_ptr())
        var lm_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.ACT), MutAnyOrigin
        ](self.state.legal_mask.unsafe_ptr())
        var rc_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.MAX_K), MutAnyOrigin
        ](self.state.root_candidates.unsafe_ptr())
        var rg_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.MAX_K), MutAnyOrigin
        ](self.state.root_gumbels.unsafe_ptr())
        var ra_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.MAX_K), MutAnyOrigin
        ](self.state.root_active.unsafe_ptr())
        var po_full_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.PRED_OUT), MutAnyOrigin
        ](self.state.pred_output.unsafe_ptr())

        var k_clipped = k_actual
        if k_clipped > Self.MAX_K:
            k_clipped = Self.MAX_K
        if k_clipped > Self.ACT:
            k_clipped = Self.ACT
        k_clipped = _largest_power_of_two_le(k_clipped)
        if k_clipped < 1:
            k_clipped = 1

        comptime run_init = gz_init_root_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, Self.BINS, Self.MAX_K,
            Self.PRED_OUT, dtype,
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
            Scalar[dtype](self.v_min),
            Scalar[dtype](self.v_max),
            Scalar[DType.int32](k_clipped),
            Scalar[DType.uint8](1 if apply_legal else 0),
            rng_seed,
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── 5. Sequential Halving simulation loop ────────────────────────
        var num_phases = _ilog2(k_clipped)
        if num_phases < 1:
            num_phases = 1
        var per_phase_budget = Self.NUM_SIMULATIONS // num_phases
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
                    if sims_used >= Self.NUM_SIMULATIONS:
                        break
                    self._run_one_sim_gpu[REP, DYN, PRED](
                        ctx, dyn, pred, slot, apply_legal
                    )
                    sims_used += 1

            # Halve the active set, except in the last phase.
            if phase + 1 < num_phases and active_size > 1:
                var keep = active_size // 2
                if keep < 1:
                    keep = 1
                comptime run_halve = gz_halve_active_kernel[
                    Self.N_ENVS, Self.MAX_NODES, Self.ACT, Self.MAX_K, dtype,
                ]
                var vc_t = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
                    MutAnyOrigin,
                ](self.state.visit_count.unsafe_ptr())
                var tv_t = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
                    MutAnyOrigin,
                ](self.state.total_value.unsafe_ptr())
                var tvis_t = LayoutTensor[
                    dtype,
                    Layout.row_major(Self.N_ENVS * Self.MAX_NODES),
                    MutAnyOrigin,
                ](self.state.total_visits.unsafe_ptr())
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
                    Scalar[dtype](self.c_visit),
                    Scalar[dtype](self.c_scale),
                    grid_dim=(Self.ENV_BLOCKS,),
                    block_dim=(TPB,),
                )
                active_size = keep

        # Spend any leftover simulations on slot 0 of the size-1 survivor.
        while sims_used < Self.NUM_SIMULATIONS:
            self._run_one_sim_gpu[REP, DYN, PRED](
                ctx, dyn, pred, 0, apply_legal
            )
            sims_used += 1

        # ── 6. Extract improved policy ───────────────────────────────────
        var po_extract_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.ACT), MutAnyOrigin
        ](self.state.policies_out.unsafe_ptr())
        var vc_t2 = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
            MutAnyOrigin,
        ](self.state.visit_count.unsafe_ptr())
        var tv_t2 = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
            MutAnyOrigin,
        ](self.state.total_value.unsafe_ptr())
        var tvis_t2 = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.MAX_NODES), MutAnyOrigin
        ](self.state.total_visits.unsafe_ptr())
        comptime run_extract = gz_extract_policy_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, dtype,
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
            Scalar[dtype](self.c_visit),
            Scalar[dtype](self.c_scale),
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── 7. Extract root scalar value ─────────────────────────────────
        var rv_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.root_value_out.unsafe_ptr())
        comptime run_root_value = gz_extract_root_value_kernel[
            Self.N_ENVS, Self.MAX_NODES, dtype,
        ]
        ctx.enqueue_function[run_root_value](
            nv_t,
            rv_t,
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

    # ══════════════════════════════════════════════════════════════════════
    # Internal — one MCTS simulation across all envs.
    # ══════════════════════════════════════════════════════════════════════

    def _run_one_sim_gpu[
        REP: RepresentationGPU,
        DYN: DynamicsGPU,
        PRED: PredictionGPU,
    ](
        mut self,
        ctx: DeviceContext,
        mut dyn: DYN,
        mut pred: PRED,
        slot: Int,
        apply_legal: Bool,
    ) raises:
        """One sim across all envs: select → dyn → pred → expand → backup.

        Mirrors the per-sim body of the legacy ``_run_one_sim_gpu``.
        ``slot`` is the Gumbel-Top-k root candidate slot to descend into
        (shared across envs — Sequential Halving keeps active sets in
        sync, so the same slot index is valid for every env).
        """
        var vc_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
            MutAnyOrigin,
        ](self.state.visit_count.unsafe_ptr())
        var tv_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
            MutAnyOrigin,
        ](self.state.total_value.unsafe_ptr())
        var nl_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
            MutAnyOrigin,
        ](self.state.node_logits.unsafe_ptr())
        var rw_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
            MutAnyOrigin,
        ](self.state.reward.unsafe_ptr())
        var ci_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
            MutAnyOrigin,
        ](self.state.child_idx.unsafe_ptr())
        var tvis_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.MAX_NODES), MutAnyOrigin
        ](self.state.total_visits.unsafe_ptr())
        var nv_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.MAX_NODES), MutAnyOrigin
        ](self.state.node_value.unsafe_ptr())
        var nc_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.node_count.unsafe_ptr())
        var miq_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.min_q.unsafe_ptr())
        var mxq_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.max_q.unsafe_ptr())
        var lm_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.ACT), MutAnyOrigin
        ](self.state.legal_mask.unsafe_ptr())
        var rc_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.MAX_K), MutAnyOrigin
        ](self.state.root_candidates.unsafe_ptr())
        var ra_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.MAX_K), MutAnyOrigin
        ](self.state.root_active.unsafe_ptr())
        var hs_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.LATENT),
            MutAnyOrigin,
        ](self.state.hidden_states.unsafe_ptr())
        var di_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.DYN_IN), MutAnyOrigin
        ](self.state.dyn_input.unsafe_ptr())
        var pp_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.pending_parent.unsafe_ptr())
        var pa_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.pending_action.unsafe_ptr())
        var sp_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * MAX_DEPTH), MutAnyOrigin
        ](self.state.search_paths.unsafe_ptr())
        var ap_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * MAX_DEPTH), MutAnyOrigin
        ](self.state.action_paths.unsafe_ptr())
        var pl_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.path_lengths.unsafe_ptr())

        # Selection.
        comptime run_select = gz_select_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, Self.MAX_K, Self.LATENT,
            Self.DYN_IN, dtype,
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
            Scalar[dtype](self.c_visit),
            Scalar[dtype](self.c_scale),
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # Dynamics forward (via trait adapter).
        var dyn_in_b = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS, DYN.DYN_IN_DIM),
            MutAnyOrigin,
        ](self.state.dyn_input.unsafe_ptr())
        var dyn_out_b = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS, DYN.DYN_OUT_DIM),
            MutAnyOrigin,
        ](self.state.dyn_output.unsafe_ptr())
        dyn.step_gpu[Self.N_ENVS](ctx, dyn_in_b, dyn_out_b)

        # Copy dyn_output's hidden prefix into pred_input.
        comptime run_copy = gz_copy_pred_input_kernel[
            Self.N_ENVS, Self.LATENT, Self.DYN_OUT, dtype,
        ]
        var pred_in_flat = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.LATENT), MutAnyOrigin
        ](self.state.pred_input.unsafe_ptr())
        var dyn_out_flat = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.DYN_OUT), MutAnyOrigin
        ](self.state.dyn_output.unsafe_ptr())
        ctx.enqueue_function[run_copy](
            pred_in_flat,
            dyn_out_flat,
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # Prediction forward.
        var pred_in_b = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS, PRED.LATENT_DIM),
            MutAnyOrigin,
        ](self.state.pred_input.unsafe_ptr())
        var pred_out_b = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS, PRED.PRED_OUT_DIM),
            MutAnyOrigin,
        ](self.state.pred_output.unsafe_ptr())
        pred.predict_gpu[Self.N_ENVS](ctx, pred_in_b, pred_out_b)

        # Expand.
        var lv_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.leaf_values.unsafe_ptr())
        var po_full_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.PRED_OUT), MutAnyOrigin
        ](self.state.pred_output.unsafe_ptr())
        comptime run_expand = gz_expand_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, Self.LATENT, Self.BINS,
            Self.PRED_OUT, Self.DYN_OUT, dtype,
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
            Scalar[dtype](self.v_min),
            Scalar[dtype](self.v_max),
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # Backup.
        comptime run_backup = gz_backup_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, dtype,
        ]
        ctx.enqueue_function[run_backup](
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
            Scalar[dtype](self.gamma),
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )
