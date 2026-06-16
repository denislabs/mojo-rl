"""LeWM `ScorePlanCallback` adapter.

Wraps LeWM's autoregressive MPC shot (``_run_mpc_shot`` in
``kernels.mojo``) behind the generic
``mojo_rl.planners.trajectory.score_callback.ScorePlanCallback`` trait
so that the agent-agnostic ``CategoricalCEMOptimizer`` can score plans
without seeing any LeWM-specific types.

Construction takes:

- a ``LeWMGPUState`` (by ``UnsafePointer`` — the state outlives the
  callback, which is only live during one ``CEMPlanner.eval`` call),
- the MPC horizon + ``needed_actions``,
- the host/device scratch buffers ``_run_mpc_shot`` writes through:
  staged-action upload + score scratch. These were historically owned
  by ``CEMPlanner`` and are *not* duplicated — the planner hands them
  in by ``mut`` ref/copy, depending on their movability.

Per-call work in ``score_plan``:
  1. Stage the ``(BATCH, HORIZON, ACT)`` host plan into the trainer's
     ``(BATCH, T, ACT)`` device layout (zero-padding the tail).
  2. Build LayoutTensor views for all the LeWM-state fields ``_run_mpc_shot``
     reads/writes (cheap — these are pointer-plus-shape wrappers).
  3. Call ``_run_mpc_shot``; return the Float64 MSE-to-goal it produced.

The view-rebuild on every call is intentional: ``LayoutTensor`` construction
is free at runtime, and avoiding stored fields keeps the callback struct
small (5 fields vs 30+).

Note on Mojo nightly compile time: per
``feedback_lewm_eval_block_compile_explosion.md``, large
``def-raises``-call clusters inside method bodies blow up the inliner.
``_run_mpc_shot`` is already a module-level helper, and the view-rebuild
in ``score_plan`` only constructs ``LayoutTensor`` values (no
``def-raises`` calls), so this method stays well under threshold.
"""

from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor, TileTensor, TensorLayout

from mojo_rl.nn2.constants import DT as dtype  # float32 (legacy nn.constants.dtype)
from mojo_rl.planners.trajectory.score_callback import (
    ScorePlanCallback, BatchedScorePlanCallback,
)

from .offline_trainer import LeWMGPUState
from .lewm_config import LeWMConfig
from .kernels import _run_mpc_shot, _run_mpc_rollout_no_readback


struct LeWMRolloutScoreCallback[CONFIG: LeWMConfig](
    Movable, ImplicitlyDestructible,
    ScorePlanCallback, BatchedScorePlanCallback,
):
    """Score a categorical action plan via LeWM's autoregressive MPC shot.

    The callback owns no LeWM-state buffers — it carries a ``Pointer``
    to the trainer's persistent ``LeWMGPUState`` and the per-call scratch
    ``_run_mpc_shot`` needs (action-plan staging + score scratch).
    Implements both ``ScorePlanCallback`` (single plan / call — used by
    the expert leg) and ``BatchedScorePlanCallback`` (K plans / call,
    one host sync — used by the random shooter and CEM loops where K is
    large). The view-rebuild on every score call is what keeps the
    struct small and the build time bounded.
    """

    comptime GPUState = LeWMGPUState[Self.CONFIG]
    comptime EMB: Int = Self.GPUState.EMB

    var state_ref: Pointer[Self.GPUState, MutAnyOrigin]
    """Safe pointer to the trainer's persistent GPU state. Caller
    guarantees it outlives the callback (which lives only inside one
    ``CEMPlanner.eval`` call). ``Pointer`` is the checked alternative
    to ``UnsafePointer``; we use it here because the callback never
    does pointer arithmetic on the state, only a single dereference."""

    var ctx: DeviceContext
    var mpc_horizon: Int
    var needed_actions: Int
    var k_max: Int
    """Maximum number of plans `score_plans_batched` will be called with.
    Sizes ``scores_dev_buf`` / ``scores_host_buf`` at construction. The
    single-plan path uses slot 0 of the same buffers."""

    # ── Per-call scratch (re-used across all score calls). ──
    var emb_start_dev_buf: DeviceBuffer[dtype]
    var emb_goal_dev_buf: DeviceBuffer[dtype]
    var emb_seq_dev_buf: DeviceBuffer[dtype]
    var action_plan_dev_buf: DeviceBuffer[dtype]
    var scores_dev_buf: DeviceBuffer[dtype]
    """K-sized device scratch — slot ``k`` receives the score for the
    k-th plan in a batched call. Single-plan ``score_plan`` writes slot
    0 and reads it back immediately."""
    var scores_host_buf: HostBuffer[dtype]
    """Pinned host mirror of ``scores_dev_buf`` for the readback at the
    end of the K-loop (or after slot-0 on the single-plan path)."""
    var action_plan_stage_host: HostBuffer[dtype]

    def __init__(
        out self,
        mut state: Self.GPUState,
        ctx: DeviceContext,
        mpc_horizon: Int,
        needed_actions: Int,
        k_max: Int = 1,
    ) raises:
        """Construct a fresh callback. Allocates its own per-call scratch
        (emb_start/goal/seq, action plan staging, K-slot score buffers).
        ``k_max`` should be the largest K the caller will pass to
        ``score_plans_batched`` — typically ``max(num_samples,
        cem_samples)``. Defaults to 1 (single-plan-only callers).

        The caller is expected to fill ``emb_start_dev_buf`` /
        ``emb_goal_dev_buf`` before the first score call (typically via
        a device-to-device copy from the trainer's encoded embeddings).
        """
        if k_max < 1:
            raise Error("LeWMRolloutScoreCallback: k_max must be >= 1")
        self.state_ref = Pointer(to=state)
        self.ctx = ctx
        self.mpc_horizon = mpc_horizon
        self.needed_actions = needed_actions
        self.k_max = k_max
        self.emb_start_dev_buf = ctx.enqueue_create_buffer[dtype](
            Self.CONFIG.BATCH * Self.EMB
        )
        self.emb_goal_dev_buf = ctx.enqueue_create_buffer[dtype](
            Self.CONFIG.BATCH * Self.EMB
        )
        self.emb_seq_dev_buf = ctx.enqueue_create_buffer[dtype](
            Self.CONFIG.BATCH * (Self.CONFIG.T + 1) * Self.EMB
        )
        self.action_plan_dev_buf = ctx.enqueue_create_buffer[dtype](
            Self.CONFIG.BATCH * Self.CONFIG.T * Self.CONFIG.ACT
        )
        self.scores_dev_buf = ctx.enqueue_create_buffer[dtype](k_max)
        self.scores_host_buf = ctx.enqueue_create_host_buffer[dtype](k_max)
        self.action_plan_stage_host = ctx.enqueue_create_host_buffer[dtype](
            Self.CONFIG.BATCH * Self.CONFIG.T * Self.CONFIG.ACT
        )

    def __init__(out self, *, deinit take: Self):
        self.state_ref = take.state_ref
        self.ctx = take.ctx^
        self.mpc_horizon = take.mpc_horizon
        self.needed_actions = take.needed_actions
        self.k_max = take.k_max
        self.emb_start_dev_buf = take.emb_start_dev_buf^
        self.emb_goal_dev_buf = take.emb_goal_dev_buf^
        self.emb_seq_dev_buf = take.emb_seq_dev_buf^
        self.action_plan_dev_buf = take.action_plan_dev_buf^
        self.scores_dev_buf = take.scores_dev_buf^
        self.scores_host_buf = take.scores_host_buf^
        self.action_plan_stage_host = take.action_plan_stage_host^

    def score_plan[L: TensorLayout](
        mut self,
        action_plan: TileTensor[dtype, L, MutAnyOrigin],
    ) raises -> Float64:
        """Stage `action_plan` (BATCH, needed_actions, ACT) → device
        (BATCH, T, ACT) with zero-pad, then run a single
        autoregressive MPC shot and return the score.

        ``action_plan`` is a 3D tile-tensor; the layout type ``L`` is
        generic because the optimizer's runtime ``horizon`` makes
        the type non-uniform across CEMPlanner instances. We index
        ``action_plan[b, ti, k]`` and trust the caller's contract.
        """
        return _lewm_score_plan_helper[Self.CONFIG, L](
            self.ctx,
            self.state_ref[],
            self.mpc_horizon,
            self.needed_actions,
            action_plan,
            self.emb_start_dev_buf,
            self.emb_goal_dev_buf,
            self.emb_seq_dev_buf,
            self.action_plan_dev_buf,
            self.scores_dev_buf,
            self.scores_host_buf,
            self.action_plan_stage_host,
        )

    def score_plans_batched[L: TensorLayout](
        mut self,
        action_plans: TileTensor[dtype, L, MutAnyOrigin],
        mut scores_out: List[Float64],
    ) raises:
        """Score K candidate plans on the GPU stream with ONE host sync.

        ``action_plans`` is a 4D tile-tensor of shape
        ``(K, BATCH, needed_actions, ACT)``; the optimizer-side
        ``CategoricalCEMOptimizer`` / ``CategoricalRandomShooter`` build
        it directly over their ``sample_actions`` storage.

        Implementation: K sequential ``_run_mpc_rollout_no_readback``
        calls — each writes its scalar MSE into ``scores_dev_buf[k]``
        without any host stall. After the K-loop a single
        ``enqueue_copy + synchronize`` ferries all K scores back, then
        we normalize and write into ``scores_out``.
        """
        return _lewm_score_plans_batched_helper[Self.CONFIG, L](
            self.ctx,
            self.state_ref[],
            self.mpc_horizon,
            self.needed_actions,
            action_plans,
            scores_out,
            self.k_max,
            self.emb_start_dev_buf,
            self.emb_goal_dev_buf,
            self.emb_seq_dev_buf,
            self.action_plan_dev_buf,
            self.scores_dev_buf,
            self.scores_host_buf,
            self.action_plan_stage_host,
        )


# ==============================================================================
# Module-level helper.
#
# Extracted out of `score_plan` per `feedback_lewm_eval_block_compile_explosion.md`
# — we have ~30 LayoutTensor constructions + one heavy `_run_mpc_shot` call.
# Inlining all of that into a method body of a trait-conformant struct is the
# exact pattern that blew up MPC eval compile time before. Keep it module-level.
# ==============================================================================


def _lewm_score_plan_helper[CONFIG: LeWMConfig, L: TensorLayout](
    ctx: DeviceContext,
    mut state: LeWMGPUState[CONFIG],
    mpc_horizon: Int,
    needed_actions: Int,
    action_plan: TileTensor[dtype, L, MutAnyOrigin],
    mut emb_start_dev_buf: DeviceBuffer[dtype],
    mut emb_goal_dev_buf: DeviceBuffer[dtype],
    mut emb_seq_dev_buf: DeviceBuffer[dtype],
    mut action_plan_dev_buf: DeviceBuffer[dtype],
    mut scores_dev_buf: DeviceBuffer[dtype],
    mut scores_host_buf: HostBuffer[dtype],
    mut action_plan_stage_host: HostBuffer[dtype],
) raises -> Float64:
    """Score one (BATCH, needed_actions, ACT) one-hot plan via LeWM MPC.

    Stages host plan → uploads → runs ``_run_mpc_shot`` (which writes
    the scalar score into slot 0 of ``scores_dev_buf`` and copies
    ``scores_host_buf`` back). ``scores_dev_buf`` / ``scores_host_buf``
    may be K-sized (the same buffers that ``score_plans_batched``
    uses) — single-plan path only consumes slot 0; the extra K-1 byte
    transfer is negligible (~256B at paper-K=64).

    Note: ``EMB`` is read off ``LeWMGPUState[CONFIG].EMB`` rather than passed
    as a separate comptime param — Mojo treats two textually distinct
    comptime expressions as distinct types even when numerically equal, so
    ``_run_mpc_shot`` only accepts LayoutTensors typed against the *same*
    ``LeWMGPUState[CONFIG].EMB`` expression that ``ae_state.params_view()``
    returns.
    """
    comptime assert action_plan.flat_rank == 3, (
        "_lewm_score_plan_helper expects a 3D (BATCH, needed_actions, ACT) plan"
    )
    comptime EMB = LeWMGPUState[CONFIG].EMB
    comptime BT = CONFIG.BATCH * CONFIG.T
    comptime BTH = CONFIG.BATCH * CONFIG.H
    comptime AE = LeWMGPUState[CONFIG].AE
    comptime POS = LeWMGPUState[CONFIG].POS
    comptime PROJ = LeWMGPUState[CONFIG].PROJ

    # Stage action_plan (BATCH, needed_actions, ACT) tile-tensor →
    # action_plan_stage_host (BATCH, T, ACT) with zero-padding.
    for b in range(CONFIG.BATCH):
        for ti in range(needed_actions):
            for k in range(CONFIG.ACT):
                action_plan_stage_host[
                    b * CONFIG.T * CONFIG.ACT + ti * CONFIG.ACT + k
                ] = action_plan[b, ti, k]
        for t_pad in range(CONFIG.T - needed_actions):
            for k in range(CONFIG.ACT):
                action_plan_stage_host[
                    b * CONFIG.T * CONFIG.ACT
                    + (needed_actions + t_pad) * CONFIG.ACT + k
                ] = Scalar[dtype](0.0)
    ctx.enqueue_copy(action_plan_dev_buf, action_plan_stage_host)

    # ── Build LayoutTensor views over the persistent state buffers. ──
    var actions_t = LayoutTensor[
        dtype, Layout.row_major(CONFIG.BATCH, CONFIG.T * CONFIG.ACT),
        MutAnyOrigin,
    ](state.actions_buf)
    var act_emb_t = LayoutTensor[
        dtype, Layout.row_major(CONFIG.BATCH, CONFIG.T * EMB),
        MutAnyOrigin,
    ](state.act_emb_buf)
    var ae_cache_t = LayoutTensor[
        dtype, Layout.row_major(CONFIG.BATCH, AE.CACHE_SIZE),
        MutAnyOrigin,
    ](state.ae_cache_buf)
    var emb_t = LayoutTensor[
        dtype, Layout.row_major(BT, EMB), MutAnyOrigin,
    ](state.emb_buf)
    var x_prev_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin,
    ](state.x_prev_buf)
    var x_prev_bh_t = LayoutTensor[
        dtype, Layout.row_major(CONFIG.BATCH, CONFIG.H * EMB),
        MutAnyOrigin,
    ](state.x_prev_buf)
    var x_prev_pe_bh_t = LayoutTensor[
        dtype, Layout.row_major(CONFIG.BATCH, CONFIG.H * EMB),
        MutAnyOrigin,
    ](state.x_prev_pe_buf)
    var pos_cache_t = LayoutTensor[
        dtype, Layout.row_major(CONFIG.BATCH, POS.CACHE_SIZE),
        MutAnyOrigin,
    ](state.pos_cache_buf)
    var c_in_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin,
    ](state.c_in_buf)
    var pred_raw_bh_t = LayoutTensor[
        dtype, Layout.row_major(CONFIG.BATCH, CONFIG.H * EMB),
        MutAnyOrigin,
    ](state.pred_raw_buf)
    var pred_t = LayoutTensor[
        dtype, Layout.row_major(CONFIG.BATCH, CONFIG.H * EMB),
        MutAnyOrigin,
    ](state.pred_out_buf)
    var proj_cache_t = LayoutTensor[
        dtype, Layout.row_major(CONFIG.BATCH, PROJ.CACHE_SIZE),
        MutAnyOrigin,
    ](state.proj_cache_buf)
    var silu_buf_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin,
    ](state.silu_buf_d)
    var ln_out_buf_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin,
    ](state.ln_out_buf_d)
    var mod_inp_buf_t = LayoutTensor[
        dtype, Layout.row_major(BTH, 3 * EMB), MutAnyOrigin,
    ](state.mod_inp_buf_d)
    var mod_x_buf_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin,
    ](state.mod_x_buf_d)
    var branch_out_buf_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin,
    ](state.branch_out_buf_d)
    var gate_inp_buf_t = LayoutTensor[
        dtype, Layout.row_major(BTH, 3 * EMB), MutAnyOrigin,
    ](state.gate_inp_buf_d)

    # Views over the per-call rollout scratch (callback-owned).
    var emb_start_dev_t = LayoutTensor[
        dtype, Layout.row_major(CONFIG.BATCH, EMB), MutAnyOrigin,
    ](emb_start_dev_buf.unsafe_ptr())
    var emb_goal_dev_t = LayoutTensor[
        dtype, Layout.row_major(CONFIG.BATCH, EMB), MutAnyOrigin,
    ](emb_goal_dev_buf.unsafe_ptr())
    var emb_seq_dev_t = LayoutTensor[
        dtype, Layout.row_major(CONFIG.BATCH, (CONFIG.T + 1) * EMB),
        MutAnyOrigin,
    ](emb_seq_dev_buf.unsafe_ptr())
    var action_plan_dev_t = LayoutTensor[
        dtype, Layout.row_major(CONFIG.BATCH, CONFIG.T * CONFIG.ACT),
        MutAnyOrigin,
    ](action_plan_dev_buf.unsafe_ptr())
    var score_dev_t = LayoutTensor[
        dtype, Layout.row_major(1), MutAnyOrigin,
    ](scores_dev_buf.unsafe_ptr())

    return _run_mpc_shot[
        CONFIG.BATCH, CONFIG.T, CONFIG.H, EMB, CONFIG.ACT, CONFIG.SMOOTHED,
        CONFIG.PROJ_H, CONFIG.PRED_HEADS, CONFIG.PRED_DIM_HEAD,
        CONFIG.PRED_FF, CONFIG.DEPTH,
    ](
        ctx, mpc_horizon, needed_actions,
        emb_start_dev_t, emb_goal_dev_t,
        emb_seq_dev_t, action_plan_dev_t,
        score_dev_t, scores_dev_buf, scores_host_buf,
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


def _lewm_score_plans_batched_helper[CONFIG: LeWMConfig, L: TensorLayout](
    ctx: DeviceContext,
    mut state: LeWMGPUState[CONFIG],
    mpc_horizon: Int,
    needed_actions: Int,
    action_plans: TileTensor[dtype, L, MutAnyOrigin],
    mut scores_out: List[Float64],
    k_max: Int,
    mut emb_start_dev_buf: DeviceBuffer[dtype],
    mut emb_goal_dev_buf: DeviceBuffer[dtype],
    mut emb_seq_dev_buf: DeviceBuffer[dtype],
    mut action_plan_dev_buf: DeviceBuffer[dtype],
    mut scores_dev_buf: DeviceBuffer[dtype],
    mut scores_host_buf: HostBuffer[dtype],
    mut action_plan_stage_host: HostBuffer[dtype],
) raises:
    """Score K plans on the GPU stream with ONE host sync.

    K is taken from ``len(scores_out)``; the caller is expected to size
    ``scores_out`` exactly to the number of plans they want scored. The
    leading dim of ``action_plans`` must match.

    Body shape: for each of K plans, stage host → upload → run
    ``_run_mpc_rollout_no_readback`` writing to a (1,) view at slot
    ``k_idx`` of the K-sized device scores buffer. After the loop, a
    single ``enqueue_copy + synchronize`` ferries all K scores back.
    """
    comptime assert action_plans.flat_rank == 4, (
        "_lewm_score_plans_batched_helper expects a 4D"
        " (K, BATCH, needed_actions, ACT) plan"
    )
    comptime EMB = LeWMGPUState[CONFIG].EMB
    comptime BT = CONFIG.BATCH * CONFIG.T
    comptime BTH = CONFIG.BATCH * CONFIG.H
    comptime AE = LeWMGPUState[CONFIG].AE
    comptime POS = LeWMGPUState[CONFIG].POS
    comptime PROJ = LeWMGPUState[CONFIG].PROJ

    var num_plans = len(scores_out)
    if num_plans > k_max:
        raise Error(
            "LeWMRolloutScoreCallback.score_plans_batched: K="
            + String(num_plans) + " exceeds k_max=" + String(k_max)
            + ". Construct with a larger k_max."
        )
    if num_plans < 1:
        return

    # ── Persistent-state views (rebuilt once per batched call; the
    # K-loop body re-uses them). ────────────────────────────────────
    var actions_t = LayoutTensor[
        dtype, Layout.row_major(CONFIG.BATCH, CONFIG.T * CONFIG.ACT),
        MutAnyOrigin,
    ](state.actions_buf)
    var act_emb_t = LayoutTensor[
        dtype, Layout.row_major(CONFIG.BATCH, CONFIG.T * EMB),
        MutAnyOrigin,
    ](state.act_emb_buf)
    var ae_cache_t = LayoutTensor[
        dtype, Layout.row_major(CONFIG.BATCH, AE.CACHE_SIZE),
        MutAnyOrigin,
    ](state.ae_cache_buf)
    var emb_t = LayoutTensor[
        dtype, Layout.row_major(BT, EMB), MutAnyOrigin,
    ](state.emb_buf)
    var x_prev_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin,
    ](state.x_prev_buf)
    var x_prev_bh_t = LayoutTensor[
        dtype, Layout.row_major(CONFIG.BATCH, CONFIG.H * EMB),
        MutAnyOrigin,
    ](state.x_prev_buf)
    var x_prev_pe_bh_t = LayoutTensor[
        dtype, Layout.row_major(CONFIG.BATCH, CONFIG.H * EMB),
        MutAnyOrigin,
    ](state.x_prev_pe_buf)
    var pos_cache_t = LayoutTensor[
        dtype, Layout.row_major(CONFIG.BATCH, POS.CACHE_SIZE),
        MutAnyOrigin,
    ](state.pos_cache_buf)
    var c_in_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin,
    ](state.c_in_buf)
    var pred_raw_bh_t = LayoutTensor[
        dtype, Layout.row_major(CONFIG.BATCH, CONFIG.H * EMB),
        MutAnyOrigin,
    ](state.pred_raw_buf)
    var pred_t = LayoutTensor[
        dtype, Layout.row_major(CONFIG.BATCH, CONFIG.H * EMB),
        MutAnyOrigin,
    ](state.pred_out_buf)
    var proj_cache_t = LayoutTensor[
        dtype, Layout.row_major(CONFIG.BATCH, PROJ.CACHE_SIZE),
        MutAnyOrigin,
    ](state.proj_cache_buf)
    var silu_buf_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin,
    ](state.silu_buf_d)
    var ln_out_buf_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin,
    ](state.ln_out_buf_d)
    var mod_inp_buf_t = LayoutTensor[
        dtype, Layout.row_major(BTH, 3 * EMB), MutAnyOrigin,
    ](state.mod_inp_buf_d)
    var mod_x_buf_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin,
    ](state.mod_x_buf_d)
    var branch_out_buf_t = LayoutTensor[
        dtype, Layout.row_major(BTH, EMB), MutAnyOrigin,
    ](state.branch_out_buf_d)
    var gate_inp_buf_t = LayoutTensor[
        dtype, Layout.row_major(BTH, 3 * EMB), MutAnyOrigin,
    ](state.gate_inp_buf_d)
    var emb_start_dev_t = LayoutTensor[
        dtype, Layout.row_major(CONFIG.BATCH, EMB), MutAnyOrigin,
    ](emb_start_dev_buf.unsafe_ptr())
    var emb_goal_dev_t = LayoutTensor[
        dtype, Layout.row_major(CONFIG.BATCH, EMB), MutAnyOrigin,
    ](emb_goal_dev_buf.unsafe_ptr())
    var emb_seq_dev_t = LayoutTensor[
        dtype, Layout.row_major(CONFIG.BATCH, (CONFIG.T + 1) * EMB),
        MutAnyOrigin,
    ](emb_seq_dev_buf.unsafe_ptr())
    var action_plan_dev_t = LayoutTensor[
        dtype, Layout.row_major(CONFIG.BATCH, CONFIG.T * CONFIG.ACT),
        MutAnyOrigin,
    ](action_plan_dev_buf.unsafe_ptr())

    # ── K-loop: stage / upload / rollout per plan, scores accumulate in
    # scores_dev_buf[0..K-1] without host stalls. ────────────────────
    for k_idx in range(num_plans):
        # Stage plan k_idx (BATCH, needed_actions, ACT) into
        # action_plan_stage_host (BATCH, T, ACT) with zero-padding.
        for b in range(CONFIG.BATCH):
            for ti in range(needed_actions):
                for j in range(CONFIG.ACT):
                    action_plan_stage_host[
                        b * CONFIG.T * CONFIG.ACT + ti * CONFIG.ACT + j
                    ] = action_plans[k_idx, b, ti, j]
            for t_pad in range(CONFIG.T - needed_actions):
                for j in range(CONFIG.ACT):
                    action_plan_stage_host[
                        b * CONFIG.T * CONFIG.ACT
                        + (needed_actions + t_pad) * CONFIG.ACT + j
                    ] = Scalar[dtype](0.0)
        ctx.enqueue_copy(action_plan_dev_buf, action_plan_stage_host)

        # (1,)-shaped device view pointing at slot k_idx of the K-sized
        # scores buffer — the rollout's mpc_score_kernel writes
        # warp.sum(...) into score_dev_slot_t[0] which maps to
        # scores_dev_buf[k_idx] in the underlying memory.
        var score_dev_slot_t = LayoutTensor[
            dtype, Layout.row_major(1), MutAnyOrigin,
        ](scores_dev_buf.unsafe_ptr() + k_idx)

        _run_mpc_rollout_no_readback[
            CONFIG.BATCH, CONFIG.T, CONFIG.H, EMB, CONFIG.ACT,
            CONFIG.SMOOTHED, CONFIG.PROJ_H,
            CONFIG.PRED_HEADS, CONFIG.PRED_DIM_HEAD,
            CONFIG.PRED_FF, CONFIG.DEPTH,
        ](
            ctx, mpc_horizon, needed_actions,
            emb_start_dev_t, emb_goal_dev_t,
            emb_seq_dev_t, action_plan_dev_t,
            score_dev_slot_t,
            state.ae_state.params_view(), state.ae_state.model_state_view(),
            actions_t, act_emb_t,
            ae_cache_t, state.ae_ws_buf,
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

    # One bulk readback + sync at end of K-loop — the whole point of the
    # batched path. K Float32s land in scores_host_buf, then we normalize.
    ctx.enqueue_copy(scores_host_buf, scores_dev_buf)
    ctx.synchronize()
    var denom = Float64(CONFIG.BATCH * EMB)
    for k_idx in range(num_plans):
        scores_out[k_idx] = Float64(scores_host_buf[k_idx]) / denom
