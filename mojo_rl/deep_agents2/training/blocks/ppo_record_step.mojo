"""PPORecordStep — push N_ENVS transitions into the rollout buffer.

Reads N_ENVS-wide (obs, reward, done, next_obs) from the driver + the
per-step caches (cached_action / cached_log_prob / cached_value) filled
by PPOActStep, writes into the [ROLLOUT_LEN, N_ENVS]-sized buffers at
state.rollout_idx (all envs at row `rollout_idx`), then advances the
single shared cursor.

Layout: T-major. Slot at time t, env e:
  obs_buf[ (t*N_ENVS + e) * OBS + d ] = obs_e[d]
  act_buf[ (t*N_ENVS + e) * ACT + j ] = cached_action_e[j]
  olp_buf[ t*N_ENVS + e ] = cached_log_prob_e
  ... etc

Driver contract:
  - `action_ptr` arg is the env-ready action vector — IGNORED (the
    rollout uses cached unbounded action for PPO math).
  - `done_ptr` is treated as truncation by default. Real terminals
    need `mark_terminal(env_idx)`.
  - `next_obs_ptr` is cached in `bootstrap_obs` (N_ENVS × OBS) every
    step — at rollout end it already holds the right per-env value.
"""

from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from ..onpolicy_state import OnPolicyState


struct PPORecordStep[
    OBS_: Int,
    ACT_: Int,
    ROLLOUT_LEN_: Int,
](Defaultable & Movable & ImplicitlyDestructible):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime ROLLOUT_LEN = Self.ROLLOUT_LEN_

    def __init__(out self):
        pass

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "PPORecordStep: target must be 'cpu' or 'gpu'"
        )
        return Self()

    def step[
        target: StaticString,
        MINIBATCH: Int,
        N_ENVS: Int,
    ](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, Self.ROLLOUT_LEN, MINIBATCH, N_ENVS,
        ],
        obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        reward_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        next_obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        done_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Push N_ENVS transitions into rollout buffers at row
        state.rollout_idx, advance the single shared cursor by 1."""
        var t = state.rollout_idx
        if t >= Self.ROLLOUT_LEN:
            return
        # Rollout buffers always live host-side (GPU train_target only
        # uploads the gathered minibatch).
        var obs_p   = state.obs_buf.cpu_ptr()
        var act_p   = state.act_buf.cpu_ptr()
        var ca_p    = state.cached_action.cpu_ptr()
        var olp_p   = state.olp_buf.cpu_ptr()
        var val_p   = state.val_buf.cpu_ptr()
        var rew_p   = state.rew_buf.cpu_ptr()
        var done_p  = state.done_buf.cpu_ptr()
        var boot_p  = state.bootstrap_obs.cpu_ptr()
        var clp_p   = state.cached_log_prob.cpu_ptr()
        var cval_p  = state.cached_value.cpu_ptr()
        # T-major layout: row `t` holds all N_ENVS rows back-to-back.
        var row_base = t * N_ENVS
        for e in range(N_ENVS):
            for d in range(Self.OBS):
                obs_p[(row_base + e) * Self.OBS + d] = obs_ptr[e * Self.OBS + d]
            for j in range(Self.ACT):
                act_p[(row_base + e) * Self.ACT + j] = ca_p[e * Self.ACT + j]
            olp_p[row_base + e]  = clp_p[e]
            val_p[row_base + e]  = cval_p[e]
            rew_p[row_base + e]  = reward_ptr[e]
            done_p[row_base + e] = done_ptr[e]
            # term_buf stays at 0 unless caller marks terminal explicitly.
            for d in range(Self.OBS):
                boot_p[e * Self.OBS + d] = next_obs_ptr[e * Self.OBS + d]
        state.rollout_idx += 1

    def mark_terminal[
        target: StaticString,
        MINIBATCH: Int,
        N_ENVS: Int,
    ](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, Self.ROLLOUT_LEN, MINIBATCH, N_ENVS,
        ],
        env_idx: Int,
    ) raises:
        """Mark the last-recorded transition for `env_idx` as a real
        terminal (V=0 bootstrap). No-op if the cursor is at 0."""
        if state.rollout_idx > 0:
            var term_p = state.term_buf.cpu_ptr()
            var row = (state.rollout_idx - 1) * N_ENVS + env_idx
            term_p[row] = Scalar[DT](1.0)

    def reset_rollout[
        target: StaticString,
        MINIBATCH: Int,
        N_ENVS: Int,
    ](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, Self.ROLLOUT_LEN, MINIBATCH, N_ENVS,
        ],
    ) raises:
        """Called by the trainer once the K-epoch update has fired —
        zeros the term buffer and resets the cursor."""
        state.rollout_idx = 0
        var term_p = state.term_buf.cpu_ptr()
        for k in range(Self.ROLLOUT_LEN * N_ENVS):
            term_p[k] = Scalar[DT](0.0)
