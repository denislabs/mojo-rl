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

from mojo_rl.nn.constants import dtype
from mojo_rl.planners.trajectory.score_callback import ScorePlanCallback

from .offline_trainer import LeWMGPUState
from .lewm_config import LeWMConfig
from .kernels import _run_mpc_shot


struct LeWMRolloutScoreCallback[CONFIG: LeWMConfig](
    Movable, ImplicitlyDestructible, ScorePlanCallback,
):
    """Score a categorical action plan via LeWM's autoregressive MPC shot.

    The callback owns no LeWM-state buffers — it carries an
    ``UnsafePointer`` to the trainer's persistent ``LeWMGPUState`` and
    the per-call scratch ``_run_mpc_shot`` needs (action-plan staging +
    score scratch). The view-rebuild on every ``score_plan`` is what
    keeps the struct small and the build time bounded.
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

    # ── Per-call scratch (re-used across all `score_plan` calls). ────
    # These came from `CEMPlanner` originally; the planner now hands them
    # in by `^` transfer to the callback for the duration of the eval.
    var emb_start_dev_buf: DeviceBuffer[dtype]
    var emb_goal_dev_buf: DeviceBuffer[dtype]
    var emb_seq_dev_buf: DeviceBuffer[dtype]
    var action_plan_dev_buf: DeviceBuffer[dtype]
    var score_dev_buf: DeviceBuffer[dtype]
    var score_host_buf: HostBuffer[dtype]
    var action_plan_stage_host: HostBuffer[dtype]

    def __init__(
        out self,
        mut state: Self.GPUState,
        ctx: DeviceContext,
        mpc_horizon: Int,
        needed_actions: Int,
    ) raises:
        """Construct a fresh callback. Allocates its own per-call scratch
        (emb_start/goal/seq, action plan staging, score buffers). The caller
        is expected to fill ``emb_start_dev_buf`` and ``emb_goal_dev_buf``
        before the first ``score_plan`` call (typically via a
        device-to-device copy from the trainer's encoded embeddings).
        """
        self.state_ref = Pointer(to=state)
        self.ctx = ctx
        self.mpc_horizon = mpc_horizon
        self.needed_actions = needed_actions
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
        self.score_dev_buf = ctx.enqueue_create_buffer[dtype](1)
        self.score_host_buf = ctx.enqueue_create_host_buffer[dtype](1)
        self.action_plan_stage_host = ctx.enqueue_create_host_buffer[dtype](
            Self.CONFIG.BATCH * Self.CONFIG.T * Self.CONFIG.ACT
        )

    def __init__(out self, *, deinit take: Self):
        self.state_ref = take.state_ref
        self.ctx = take.ctx^
        self.mpc_horizon = take.mpc_horizon
        self.needed_actions = take.needed_actions
        self.emb_start_dev_buf = take.emb_start_dev_buf^
        self.emb_goal_dev_buf = take.emb_goal_dev_buf^
        self.emb_seq_dev_buf = take.emb_seq_dev_buf^
        self.action_plan_dev_buf = take.action_plan_dev_buf^
        self.score_dev_buf = take.score_dev_buf^
        self.score_host_buf = take.score_host_buf^
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
            self.score_dev_buf,
            self.score_host_buf,
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
    mut score_dev_buf: DeviceBuffer[dtype],
    mut score_host_buf: HostBuffer[dtype],
    mut action_plan_stage_host: HostBuffer[dtype],
) raises -> Float64:
    """Score one (BATCH, needed_actions, ACT) one-hot plan via LeWM MPC.

    Mirrors the per-sample logic that used to live inside
    ``_run_cem_eval_iter``: stage host plan, upload, run MPC shot.

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
    ](score_dev_buf.unsafe_ptr())

    return _run_mpc_shot[
        CONFIG.BATCH, CONFIG.T, CONFIG.H, EMB, CONFIG.ACT, CONFIG.SMOOTHED,
        CONFIG.PROJ_H, CONFIG.PRED_HEADS, CONFIG.PRED_FF, CONFIG.DEPTH,
    ](
        ctx, mpc_horizon, needed_actions,
        emb_start_dev_t, emb_goal_dev_t,
        emb_seq_dev_t, action_plan_dev_t,
        score_dev_t, score_dev_buf, score_host_buf,
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
