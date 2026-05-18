"""Generic GPU MCTS orchestrator — ``GenericGPUMCTS``.

The agent-side counterpart of ``GenericCPUMCTS``. Owns a
``GPUMCTSState`` plus its action / policy / root-value output buffers,
and a ``search_gpu`` method that drives the full kernel-sequence-+-net-
forward pipeline for one MCTS search across ``N_ENVS`` environments in
parallel.

Phase 3b second slice. Replaces the ~600-line inline orchestration in
``muzero.mojo`` for the **MuZero batched single-player path**:

  root setup
    ┌── encode_gpu (REP) → hidden_states[node 0]
    ├── mcts_gpu_scale_hidden_kernel
    ├── predict_gpu (PRED, batch=N_ENVS) → pred_output
    ├── zero_tree
    └── gpu_mcts_init_root_kernel  ← adds Dirichlet noise if NOISE != NoNoise
  for each round (NUM_SIMS / BATCH_SIMS):
    ┌── gpu_mcts_batched_select_and_build_dyn_kernel
    ├── step_gpu (DYN, batch=N_ENVS·BATCH_SIMS)
    ├── mcts_gpu_extract_hidden_kernel
    ├── predict_gpu (PRED, batch=N_ENVS·BATCH_SIMS)
    └── gpu_mcts_batched_expand_backup_muzero_kernel
  readout
    ├── gpu_mcts_extract_actions_kernel
    └── gpu_mcts_extract_root_value_kernel

Other paths (AlphaZero `env.step` expansion, self-play negated backup,
temperature-weighted action sampling, legal-mask variants) are
deliberately not wired here yet — they're separate methods on the
struct, landing as their respective agents get rewired. The single-player
MuZero path is the most-used + easiest to validate against the existing
muzero CartPole training run.

The struct doesn't own a workspace buffer; the adapters do (they know
their network's ``WORKSPACE_SIZE_PER_SAMPLE``). This avoids the
orchestrator having to peek at trait-internal model dimensions.
"""

from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import dtype

from .model_traits_gpu import (
    RepresentationGPU,
    DynamicsGPU,
    PredictionGPU,
)
from .strategies import (
    PUCTFormula,
    ExplorationNoise,
    PlayerMode,
)
from .mcts_gpu import (
    TPB,
    MAX_DEPTH,
    GPUMCTSState,
    gpu_mcts_init_root_kernel,
    gpu_mcts_extract_actions_kernel,
    gpu_mcts_extract_actions_masked_kernel,
    gpu_mcts_extract_root_value_kernel,
    gpu_mcts_apply_legal_mask_kernel,
    gpu_mcts_apply_legal_mask_with_noise_kernel,
    gpu_mcts_batched_select_and_build_dyn_kernel,
    gpu_mcts_batched_expand_backup_muzero_kernel,
    mcts_gpu_scale_hidden_kernel,
    mcts_gpu_extract_hidden_kernel,
)


struct GenericGPUMCTS[
    N_ENVS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    MAX_NODES: Int,
    NUM_SIMULATIONS: Int,
    BATCH_SIMS: Int,
    PUCT: PUCTFormula,
    NOISE: ExplorationNoise,
    PLAYER: PlayerMode,
](Movable, ImplicitlyDestructible):
    """GPU MCTS orchestrator for the MuZero batched single-player path.

    Comptime params:
        N_ENVS: Number of parallel envs / trees.
        ACT: Discrete action count.
        LATENT: Hidden state dim.
        BINS: Categorical reward / value bins (= ``NUM_BINS``).
        MAX_NODES: Per-tree node arena size.
        NUM_SIMULATIONS: Sims per ``search_gpu`` call. Must be a
            multiple of ``BATCH_SIMS``.
        BATCH_SIMS: Leaves selected per round (virtual-loss batching).
        PUCT: PUCT formula trait — supplies ``C_BASE`` / ``C_INIT`` to
            the selection kernel. ``MuZeroPUCT`` (log-based) is the
            default for MuZero.
        NOISE: Root exploration noise. ``NOISE_FRACTION`` is forwarded
            to ``gpu_mcts_init_root_kernel``; ``NoNoise`` ⇒ 0.
        PLAYER: ``SinglePlayer`` for now. ``SelfPlay`` requires the
            negated-backup variant (Phase 3b follow-up).

    Runtime ctor args: ``gamma``, ``v_min``, ``v_max``.

    The orchestrator does **not** own a workspace buffer — adapters
    own theirs because they know the network's
    ``WORKSPACE_SIZE_PER_SAMPLE``.
    """

    comptime PRED_OUT: Int = Self.ACT + Self.BINS
    comptime DYN_IN: Int = Self.LATENT + Self.ACT
    comptime DYN_OUT: Int = Self.LATENT + Self.BINS
    comptime MCTS_TOTAL: Int = Self.N_ENVS * Self.BATCH_SIMS
    comptime MCTS_ROUNDS: Int = Self.NUM_SIMULATIONS // Self.BATCH_SIMS
    comptime ENV_BLOCKS: Int = (Self.N_ENVS + TPB - 1) // TPB

    var state: GPUMCTSState[
        Self.N_ENVS, Self.MAX_NODES, Self.ACT, Self.LATENT, Self.BINS, 0,
        Self.BATCH_SIMS,
    ]
    """Tree node arena + per-batch scratch buffers, all device-resident."""

    var actions_out: DeviceBuffer[dtype]
    """``[N_ENVS]`` argmax action per env (output of
    ``gpu_mcts_extract_actions_kernel``)."""

    var policies_out: DeviceBuffer[dtype]
    """``[N_ENVS × ACT]`` visit-count policy per env."""

    var root_value_out: DeviceBuffer[dtype]
    """``[N_ENVS]`` visit-weighted root Q per env."""

    var gamma: Float64
    var v_min: Float64
    var v_max: Float64

    def __init__(
        out self,
        ctx: DeviceContext,
        gamma: Float64 = 0.997,
        v_min: Float64 = -10.0,
        v_max: Float64 = 10.0,
    ) raises:
        # NUM_SIMULATIONS must be a clean multiple of BATCH_SIMS.
        # Enforced at construction so a misconfigured agent fails loudly.
        if Self.NUM_SIMULATIONS % Self.BATCH_SIMS != 0:
            raise Error(
                "GenericGPUMCTS: NUM_SIMULATIONS must be divisible by"
                " BATCH_SIMS"
            )
        if Self.MCTS_ROUNDS < 1:
            raise Error(
                "GenericGPUMCTS: NUM_SIMULATIONS >= BATCH_SIMS required"
            )

        self.state = GPUMCTSState[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, Self.LATENT, Self.BINS, 0,
            Self.BATCH_SIMS,
        ](ctx)
        self.actions_out = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.policies_out = ctx.enqueue_create_buffer[dtype](
            Self.N_ENVS * Self.ACT
        )
        self.root_value_out = ctx.enqueue_create_buffer[dtype](Self.N_ENVS)
        self.gamma = gamma
        self.v_min = v_min
        self.v_max = v_max

    def __init__(out self, *, deinit take: Self):
        self.state = take.state^
        self.actions_out = take.actions_out^
        self.policies_out = take.policies_out^
        self.root_value_out = take.root_value_out^
        self.gamma = take.gamma
        self.v_min = take.v_min
        self.v_max = take.v_max

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
        rng_seed: UInt32 = UInt32(0),
    ) raises:
        """Run ``NUM_SIMULATIONS`` MCTS sims for every env in parallel.

        Writes results into ``self.actions_out``, ``self.policies_out``,
        ``self.root_value_out``. Caller reads via the corresponding
        ``…_view()`` methods (or directly off the device buffers).

        Single-player MuZero-style: gamma-discounted backup, categorical
        reward + value decode, MinMax Q-normalization, virtual-loss
        batched leaf selection. Root exploration noise is gated on the
        ``NOISE`` trait — ``NoNoise`` ⇒ no noise added.
        """

        # ── 1. Root encode → hidden_states[node 0 for each env] ──────
        # Use the trait's ``LATENT_DIM`` for the view so the
        # method-template binding sees one comptime expression, not
        # ``Self.LATENT`` vs ``REP.LATENT_DIM`` (same value, distinct
        # expressions in Mojo's type system).
        var hidden_root = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS, REP.LATENT_DIM),
            MutAnyOrigin,
        ](self.state.hidden_states.unsafe_ptr())
        rep.encode_gpu[Self.N_ENVS](ctx, obs, hidden_root)

        # 1a. Post-encode MinMax scaling. MuZero networks bake this into
        # the model (``MinMaxNorm`` tail layer) so the kernel is a no-op
        # on already-normalized output; for stub / Linear-only test
        # networks it does real work. Always-on is the safer default —
        # the kernel is idempotent on [0, 1] values.
        comptime BATCH_BLOCKS = (Self.N_ENVS + TPB - 1) // TPB
        var hidden_root_flat = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.LATENT),
            MutAnyOrigin,
        ](self.state.hidden_states.unsafe_ptr())
        comptime run_scale_root = mcts_gpu_scale_hidden_kernel[
            Self.N_ENVS, Self.LATENT, dtype
        ]
        ctx.enqueue_function[run_scale_root](
            hidden_root_flat,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── 2. Root predict → policy logits + value bins ─────────────
        var pred_root_in = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS, PRED.LATENT_DIM),
            MutAnyOrigin,
        ](self.state.hidden_states.unsafe_ptr())
        var pred_root_out = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS, PRED.PRED_OUT_DIM),
            MutAnyOrigin,
        ](self.state.pred_output.unsafe_ptr())
        pred.predict_gpu[Self.N_ENVS](ctx, pred_root_in, pred_root_out)

        # ── 3. Zero tree + init root from pred output ────────────────
        self.state.zero_tree(ctx)

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
        var pr_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
            MutAnyOrigin,
        ](self.state.prior.unsafe_ptr())
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
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES),
            MutAnyOrigin,
        ](self.state.total_visits.unsafe_ptr())
        var nc_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.node_count.unsafe_ptr())
        var po_root_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.PRED_OUT), MutAnyOrigin
        ](self.state.pred_output.unsafe_ptr())
        var miq_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.min_q.unsafe_ptr())
        var mxq_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.max_q.unsafe_ptr())

        comptime run_init = gpu_mcts_init_root_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, Self.LATENT, Self.PRED_OUT,
            dtype,
        ]
        # NOISE_FRACTION = 0 for NoNoise; the kernel internally still
        # generates Dirichlet samples and blends them in, but with
        # fraction 0 the original softmax prior is preserved.
        var noise_fraction = Scalar[dtype](Self.NOISE.NOISE_FRACTION)
        ctx.enqueue_function[run_init](
            vc_t, tv_t, pr_t, rw_t, ci_t, tvis_t, nc_t, po_root_t,
            miq_t, mxq_t,
            noise_fraction,
            rng_seed,
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── 4. MCTS simulation rounds ────────────────────────────────
        var hs_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.LATENT),
            MutAnyOrigin,
        ](self.state.hidden_states.unsafe_ptr())
        var b_pp = LayoutTensor[
            dtype, Layout.row_major(Self.MCTS_TOTAL), MutAnyOrigin
        ](self.state.pending_parent.unsafe_ptr())
        var b_pa = LayoutTensor[
            dtype, Layout.row_major(Self.MCTS_TOTAL), MutAnyOrigin
        ](self.state.pending_action.unsafe_ptr())
        var b_sp = LayoutTensor[
            dtype,
            Layout.row_major(Self.MCTS_TOTAL * MAX_DEPTH),
            MutAnyOrigin,
        ](self.state.search_paths.unsafe_ptr())
        var b_ap = LayoutTensor[
            dtype,
            Layout.row_major(Self.MCTS_TOTAL * MAX_DEPTH),
            MutAnyOrigin,
        ](self.state.action_paths.unsafe_ptr())
        var b_pl = LayoutTensor[
            dtype, Layout.row_major(Self.MCTS_TOTAL), MutAnyOrigin
        ](self.state.path_lengths.unsafe_ptr())
        var b_di = LayoutTensor[
            dtype,
            Layout.row_major(Self.MCTS_TOTAL * Self.DYN_IN),
            MutAnyOrigin,
        ](self.state.dyn_input.unsafe_ptr())

        for _round in range(Self.MCTS_ROUNDS):
            # 4a. Batched select + build dynamics input
            comptime run_sel_dyn = gpu_mcts_batched_select_and_build_dyn_kernel[
                Self.N_ENVS, Self.MAX_NODES, Self.ACT, Self.BATCH_SIMS,
                Self.LATENT, Self.DYN_IN, dtype,
            ]
            ctx.enqueue_function[run_sel_dyn](
                vc_t, tv_t, pr_t, ci_t, tvis_t, nc_t, miq_t, mxq_t, hs_t,
                b_di, b_pp, b_pa, b_sp, b_ap, b_pl,
                Scalar[dtype](Self.PUCT.C_BASE),
                Scalar[dtype](Self.PUCT.C_INIT),
                grid_dim=(Self.ENV_BLOCKS,),
                block_dim=(TPB,),
            )

            # 4b. Batched dynamics forward.
            # Build the LayoutTensors against the trait's comptime
            # dimensions so the method-template binding doesn't see
            # ``Self.DYN_IN`` vs ``DYN.DYN_IN_DIM`` as distinct
            # expressions (Mojo's type system treats unfolded comptime
            # exprs structurally — equal values aren't enough).
            var dyn_in_b = LayoutTensor[
                dtype,
                Layout.row_major(Self.MCTS_TOTAL, DYN.DYN_IN_DIM),
                MutAnyOrigin,
            ](self.state.dyn_input.unsafe_ptr())
            var dyn_out_b = LayoutTensor[
                dtype,
                Layout.row_major(Self.MCTS_TOTAL, DYN.DYN_OUT_DIM),
                MutAnyOrigin,
            ](self.state.dyn_output.unsafe_ptr())
            dyn.step_gpu[Self.MCTS_TOTAL](ctx, dyn_in_b, dyn_out_b)

            # 4c. Extract hidden slice → pred input
            var pred_in_b = LayoutTensor[
                dtype,
                Layout.row_major(Self.MCTS_TOTAL * Self.LATENT),
                MutAnyOrigin,
            ](self.state.pred_input.unsafe_ptr())
            var dyn_out_b_flat = LayoutTensor[
                dtype,
                Layout.row_major(Self.MCTS_TOTAL * Self.DYN_OUT),
                MutAnyOrigin,
            ](self.state.dyn_output.unsafe_ptr())
            comptime EXTR_TOTAL = Self.MCTS_TOTAL * Self.LATENT
            comptime EXTR_BLK = (EXTR_TOTAL + TPB - 1) // TPB
            comptime run_extr = mcts_gpu_extract_hidden_kernel[
                Self.MCTS_TOTAL, Self.LATENT, Self.DYN_OUT, dtype
            ]
            ctx.enqueue_function[run_extr](
                pred_in_b, dyn_out_b_flat,
                grid_dim=(EXTR_BLK,),
                block_dim=(TPB,),
            )

            # 4d. Batched prediction forward — same trait-dim trick as
            # the dynamics call above.
            var pred_in_net = LayoutTensor[
                dtype,
                Layout.row_major(Self.MCTS_TOTAL, PRED.LATENT_DIM),
                MutAnyOrigin,
            ](self.state.pred_input.unsafe_ptr())
            var pred_out_net = LayoutTensor[
                dtype,
                Layout.row_major(Self.MCTS_TOTAL, PRED.PRED_OUT_DIM),
                MutAnyOrigin,
            ](self.state.pred_output.unsafe_ptr())
            pred.predict_gpu[Self.MCTS_TOTAL](
                ctx, pred_in_net, pred_out_net
            )

            # 4e. Batched expand + backup + remove virtual loss
            var b_do = LayoutTensor[
                dtype,
                Layout.row_major(Self.MCTS_TOTAL * Self.DYN_OUT),
                MutAnyOrigin,
            ](self.state.dyn_output.unsafe_ptr())
            var b_po = LayoutTensor[
                dtype,
                Layout.row_major(Self.MCTS_TOTAL * Self.PRED_OUT),
                MutAnyOrigin,
            ](self.state.pred_output.unsafe_ptr())
            comptime run_exp_bk = gpu_mcts_batched_expand_backup_muzero_kernel[
                Self.N_ENVS, Self.MAX_NODES, Self.ACT, Self.BATCH_SIMS,
                Self.LATENT, Self.PRED_OUT, Self.DYN_OUT, dtype,
            ]
            ctx.enqueue_function[run_exp_bk](
                vc_t, tv_t, pr_t, rw_t, ci_t, tvis_t, nc_t, miq_t, mxq_t,
                hs_t, b_pp, b_pa, b_do, b_po, b_sp, b_ap, b_pl,
                Scalar[dtype](self.v_min),
                Scalar[dtype](self.v_max),
                Scalar[dtype](self.gamma),
                Scalar[DType.bool](Self.PLAYER.NEGATE_BACKUP),
                grid_dim=(Self.ENV_BLOCKS,),
                block_dim=(TPB,),
            )

        # ── 5. Extract actions + visit-count policy ──────────────────
        var act_out_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.actions_out.unsafe_ptr())
        var pol_out_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.ACT), MutAnyOrigin
        ](self.policies_out.unsafe_ptr())
        comptime run_act = gpu_mcts_extract_actions_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, dtype
        ]
        ctx.enqueue_function[run_act](
            vc_t, act_out_t, pol_out_t,
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── 6. Extract root value (visit-weighted Q at root) ─────────
        var rv_out_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.root_value_out.unsafe_ptr())
        comptime run_rv = gpu_mcts_extract_root_value_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, dtype
        ]
        ctx.enqueue_function[run_rv](
            vc_t, tv_t, rv_out_t,
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

    # ══════════════════════════════════════════════════════════════════════
    # Self-play variant — legal masks + negated backup
    # ══════════════════════════════════════════════════════════════════════

    def search_gpu_selfplay[
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
        legal_masks: LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.ACT), MutAnyOrigin
        ],
        rng_seed: UInt32 = UInt32(0),
    ) raises:
        """Self-play variant of ``search_gpu`` for board games.

        Differences vs ``search_gpu``:
          * After ``init_root``, applies the legal mask to the root prior
            (zeroes illegal actions, renormalizes). With ``DirichletNoise``
            the noisy variant is used so the noise budget lands only on
            legal actions; ``init_root`` is then called with fraction=0
            to avoid double-noise.
          * Backup is negated (``PLAYER.NEGATE_BACKUP=True`` for
            ``SelfPlay``) — already wired in the kernel call.
          * Action extraction uses ``gpu_mcts_extract_actions_masked_kernel``
            so the argmax + visit-count policy only considers legal
            actions.

        The MCTS simulation rounds themselves are identical to
        ``search_gpu``. AlphaZero ``env.step`` expansion is a separate
        follow-up path (``search_gpu_alphazero``) because it replaces
        the dynamics network call.
        """

        # ── 1. Root encode → hidden[node 0] ──────────────────────────
        var hidden_root = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS, REP.LATENT_DIM),
            MutAnyOrigin,
        ](self.state.hidden_states.unsafe_ptr())
        rep.encode_gpu[Self.N_ENVS](ctx, obs, hidden_root)

        # 1a. Post-encode MinMax scaling.
        comptime BATCH_BLOCKS = (Self.N_ENVS + TPB - 1) // TPB
        var hidden_root_flat = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.LATENT),
            MutAnyOrigin,
        ](self.state.hidden_states.unsafe_ptr())
        comptime run_scale_root = mcts_gpu_scale_hidden_kernel[
            Self.N_ENVS, Self.LATENT, dtype
        ]
        ctx.enqueue_function[run_scale_root](
            hidden_root_flat,
            grid_dim=(BATCH_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── 2. Root predict ──────────────────────────────────────────
        var pred_root_in = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS, PRED.LATENT_DIM),
            MutAnyOrigin,
        ](self.state.hidden_states.unsafe_ptr())
        var pred_root_out = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS, PRED.PRED_OUT_DIM),
            MutAnyOrigin,
        ](self.state.pred_output.unsafe_ptr())
        pred.predict_gpu[Self.N_ENVS](ctx, pred_root_in, pred_root_out)

        # ── 3. Zero tree + init root with noise=0 (legal-mask path
        #       owns noise injection so it can keep the budget on legal
        #       actions). ────────────────────────────────────────────
        self.state.zero_tree(ctx)

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
        var pr_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.ACT),
            MutAnyOrigin,
        ](self.state.prior.unsafe_ptr())
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
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES),
            MutAnyOrigin,
        ](self.state.total_visits.unsafe_ptr())
        var nc_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.node_count.unsafe_ptr())
        var po_root_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.PRED_OUT), MutAnyOrigin
        ](self.state.pred_output.unsafe_ptr())
        var miq_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.min_q.unsafe_ptr())
        var mxq_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.state.max_q.unsafe_ptr())

        comptime run_init = gpu_mcts_init_root_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, Self.LATENT, Self.PRED_OUT,
            dtype,
        ]
        # Always init_root with fraction=0 here — the legal-mask kernel
        # injects noise restricted to legal actions if NOISE is on.
        ctx.enqueue_function[run_init](
            vc_t, tv_t, pr_t, rw_t, ci_t, tvis_t, nc_t, po_root_t,
            miq_t, mxq_t,
            Scalar[dtype](0.0),
            rng_seed,
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── 3a. Apply legal mask (+ Dirichlet noise on legal-only if
        #         NOISE is enabled). ──────────────────────────────────
        comptime if Self.NOISE.NOISE_TYPE == 2:
            comptime run_mask = gpu_mcts_apply_legal_mask_kernel[
                Self.N_ENVS, Self.MAX_NODES, Self.ACT, dtype
            ]
            ctx.enqueue_function[run_mask](
                pr_t, legal_masks,
                grid_dim=(Self.ENV_BLOCKS,),
                block_dim=(TPB,),
            )
        else:
            comptime run_mask_noise = (
                gpu_mcts_apply_legal_mask_with_noise_kernel[
                    Self.N_ENVS, Self.MAX_NODES, Self.ACT, dtype
                ]
            )
            ctx.enqueue_function[run_mask_noise](
                pr_t, legal_masks,
                Scalar[dtype](Self.NOISE.NOISE_FRACTION),
                rng_seed,
                grid_dim=(Self.ENV_BLOCKS,),
                block_dim=(TPB,),
            )

        # ── 4. Simulation rounds (identical to single-player path) ──
        var hs_t = LayoutTensor[
            dtype,
            Layout.row_major(Self.N_ENVS * Self.MAX_NODES * Self.LATENT),
            MutAnyOrigin,
        ](self.state.hidden_states.unsafe_ptr())
        var b_pp = LayoutTensor[
            dtype, Layout.row_major(Self.MCTS_TOTAL), MutAnyOrigin
        ](self.state.pending_parent.unsafe_ptr())
        var b_pa = LayoutTensor[
            dtype, Layout.row_major(Self.MCTS_TOTAL), MutAnyOrigin
        ](self.state.pending_action.unsafe_ptr())
        var b_sp = LayoutTensor[
            dtype,
            Layout.row_major(Self.MCTS_TOTAL * MAX_DEPTH),
            MutAnyOrigin,
        ](self.state.search_paths.unsafe_ptr())
        var b_ap = LayoutTensor[
            dtype,
            Layout.row_major(Self.MCTS_TOTAL * MAX_DEPTH),
            MutAnyOrigin,
        ](self.state.action_paths.unsafe_ptr())
        var b_pl = LayoutTensor[
            dtype, Layout.row_major(Self.MCTS_TOTAL), MutAnyOrigin
        ](self.state.path_lengths.unsafe_ptr())
        var b_di = LayoutTensor[
            dtype,
            Layout.row_major(Self.MCTS_TOTAL * Self.DYN_IN),
            MutAnyOrigin,
        ](self.state.dyn_input.unsafe_ptr())

        for _round in range(Self.MCTS_ROUNDS):
            comptime run_sel_dyn = gpu_mcts_batched_select_and_build_dyn_kernel[
                Self.N_ENVS, Self.MAX_NODES, Self.ACT, Self.BATCH_SIMS,
                Self.LATENT, Self.DYN_IN, dtype,
            ]
            ctx.enqueue_function[run_sel_dyn](
                vc_t, tv_t, pr_t, ci_t, tvis_t, nc_t, miq_t, mxq_t, hs_t,
                b_di, b_pp, b_pa, b_sp, b_ap, b_pl,
                Scalar[dtype](Self.PUCT.C_BASE),
                Scalar[dtype](Self.PUCT.C_INIT),
                grid_dim=(Self.ENV_BLOCKS,),
                block_dim=(TPB,),
            )

            var dyn_in_b = LayoutTensor[
                dtype,
                Layout.row_major(Self.MCTS_TOTAL, DYN.DYN_IN_DIM),
                MutAnyOrigin,
            ](self.state.dyn_input.unsafe_ptr())
            var dyn_out_b = LayoutTensor[
                dtype,
                Layout.row_major(Self.MCTS_TOTAL, DYN.DYN_OUT_DIM),
                MutAnyOrigin,
            ](self.state.dyn_output.unsafe_ptr())
            dyn.step_gpu[Self.MCTS_TOTAL](ctx, dyn_in_b, dyn_out_b)

            var pred_in_b = LayoutTensor[
                dtype,
                Layout.row_major(Self.MCTS_TOTAL * Self.LATENT),
                MutAnyOrigin,
            ](self.state.pred_input.unsafe_ptr())
            var dyn_out_b_flat = LayoutTensor[
                dtype,
                Layout.row_major(Self.MCTS_TOTAL * Self.DYN_OUT),
                MutAnyOrigin,
            ](self.state.dyn_output.unsafe_ptr())
            comptime EXTR_TOTAL = Self.MCTS_TOTAL * Self.LATENT
            comptime EXTR_BLK = (EXTR_TOTAL + TPB - 1) // TPB
            comptime run_extr = mcts_gpu_extract_hidden_kernel[
                Self.MCTS_TOTAL, Self.LATENT, Self.DYN_OUT, dtype
            ]
            ctx.enqueue_function[run_extr](
                pred_in_b, dyn_out_b_flat,
                grid_dim=(EXTR_BLK,),
                block_dim=(TPB,),
            )

            var pred_in_net = LayoutTensor[
                dtype,
                Layout.row_major(Self.MCTS_TOTAL, PRED.LATENT_DIM),
                MutAnyOrigin,
            ](self.state.pred_input.unsafe_ptr())
            var pred_out_net = LayoutTensor[
                dtype,
                Layout.row_major(Self.MCTS_TOTAL, PRED.PRED_OUT_DIM),
                MutAnyOrigin,
            ](self.state.pred_output.unsafe_ptr())
            pred.predict_gpu[Self.MCTS_TOTAL](
                ctx, pred_in_net, pred_out_net
            )

            var b_do = LayoutTensor[
                dtype,
                Layout.row_major(Self.MCTS_TOTAL * Self.DYN_OUT),
                MutAnyOrigin,
            ](self.state.dyn_output.unsafe_ptr())
            var b_po = LayoutTensor[
                dtype,
                Layout.row_major(Self.MCTS_TOTAL * Self.PRED_OUT),
                MutAnyOrigin,
            ](self.state.pred_output.unsafe_ptr())
            comptime run_exp_bk = gpu_mcts_batched_expand_backup_muzero_kernel[
                Self.N_ENVS, Self.MAX_NODES, Self.ACT, Self.BATCH_SIMS,
                Self.LATENT, Self.PRED_OUT, Self.DYN_OUT, dtype,
            ]
            ctx.enqueue_function[run_exp_bk](
                vc_t, tv_t, pr_t, rw_t, ci_t, tvis_t, nc_t, miq_t, mxq_t,
                hs_t, b_pp, b_pa, b_do, b_po, b_sp, b_ap, b_pl,
                Scalar[dtype](self.v_min),
                Scalar[dtype](self.v_max),
                Scalar[dtype](self.gamma),
                Scalar[DType.bool](Self.PLAYER.NEGATE_BACKUP),
                grid_dim=(Self.ENV_BLOCKS,),
                block_dim=(TPB,),
            )

        # ── 5. Masked action extraction ──────────────────────────────
        var act_out_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.actions_out.unsafe_ptr())
        var pol_out_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.ACT), MutAnyOrigin
        ](self.policies_out.unsafe_ptr())
        comptime run_act_masked = gpu_mcts_extract_actions_masked_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, dtype
        ]
        ctx.enqueue_function[run_act_masked](
            vc_t, legal_masks, act_out_t, pol_out_t,
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

        # ── 6. Root value (same kernel, no masking needed since the
        #       backup already encoded negation). ─────────────────────
        var rv_out_t = LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.root_value_out.unsafe_ptr())
        comptime run_rv = gpu_mcts_extract_root_value_kernel[
            Self.N_ENVS, Self.MAX_NODES, Self.ACT, dtype
        ]
        ctx.enqueue_function[run_rv](
            vc_t, tv_t, rv_out_t,
            grid_dim=(Self.ENV_BLOCKS,),
            block_dim=(TPB,),
        )

    # ══════════════════════════════════════════════════════════════════════
    # Output views
    # ══════════════════════════════════════════════════════════════════════

    def actions_view(
        self,
    ) -> LayoutTensor[dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin]:
        """``[N_ENVS]`` argmax action — caller copies to host to read."""
        return LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.actions_out.unsafe_ptr())

    def policies_view(
        self,
    ) -> LayoutTensor[
        dtype, Layout.row_major(Self.N_ENVS * Self.ACT), MutAnyOrigin
    ]:
        """``[N_ENVS × ACT]`` visit-count policy per env (sums to 1)."""
        return LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS * Self.ACT), MutAnyOrigin
        ](self.policies_out.unsafe_ptr())

    def root_value_view(
        self,
    ) -> LayoutTensor[dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin]:
        """``[N_ENVS]`` visit-weighted Q at the root."""
        return LayoutTensor[
            dtype, Layout.row_major(Self.N_ENVS), MutAnyOrigin
        ](self.root_value_out.unsafe_ptr())
