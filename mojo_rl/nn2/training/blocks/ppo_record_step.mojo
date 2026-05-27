"""PPORecordStep — push one transition into the rollout buffer.

Reads (obs, reward, done, next_obs) from the driver + the per-step
cache (cached_action, cached_log_prob, cached_value) filled by
PPOActStep, writes into the ROLLOUT_LEN-sized buffers at
state.rollout_idx, then advances the cursor.

Driver contract:
  - `action` arg is the env-ready action — IGNORED (the rollout uses
    cached unbounded action for PPO math).
  - `done` is treated as truncation by default (Pendulum-style). Real
    terminals need an explicit `mark_terminal()` call.
  - `next_obs` is cached in `bootstrap_obs` every step — at rollout
    end it already holds the right value.
"""

from std.gpu.host import DeviceContext

from ...constants import DT
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
        comptime assert target == "cpu", (
            "PPORecordStep: P.1 is CPU-only (GPU lands in P.2)"
        )
        return Self()

    def step[
        target: StaticString,
        MINIBATCH: Int,
    ](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, Self.ROLLOUT_LEN, MINIBATCH,
        ],
        ref obs: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        var t = state.rollout_idx
        if t >= Self.ROLLOUT_LEN:
            return
        var obs_p   = state.obs_buf.target_ptr[target]()
        var act_p   = state.act_buf.target_ptr[target]()
        var ca_p    = state.cached_action.target_ptr[target]()
        var olp_p   = state.olp_buf.target_ptr[target]()
        var val_p   = state.val_buf.target_ptr[target]()
        var rew_p   = state.rew_buf.target_ptr[target]()
        var done_p  = state.done_buf.target_ptr[target]()
        var boot_p  = state.bootstrap_obs.target_ptr[target]()
        for d in range(Self.OBS):
            obs_p[t * Self.OBS + d] = obs[d]
        for j in range(Self.ACT):
            act_p[t * Self.ACT + j] = ca_p[j]
        olp_p[t]  = state.cached_log_prob
        val_p[t]  = state.cached_value
        rew_p[t]  = reward
        done_p[t] = done
        # term_buf stays at 0 unless caller marks terminal explicitly.
        for d in range(Self.OBS):
            boot_p[d] = next_obs[d]
        state.rollout_idx += 1

    def mark_terminal[
        target: StaticString,
        MINIBATCH: Int,
    ](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, Self.ROLLOUT_LEN, MINIBATCH,
        ],
    ) raises:
        """Mark the last-recorded transition as a real terminal (V=0
        bootstrap). No-op if the cursor is at 0."""
        if state.rollout_idx > 0:
            var term_p = state.term_buf.target_ptr[target]()
            term_p[state.rollout_idx - 1] = Scalar[DT](1.0)

    def reset_rollout[
        target: StaticString,
        MINIBATCH: Int,
    ](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, Self.ROLLOUT_LEN, MINIBATCH,
        ],
    ) raises:
        """Called by the trainer once the K-epoch update has fired —
        zeros the term buffer and resets the cursor."""
        state.rollout_idx = 0
        var term_p = state.term_buf.target_ptr[target]()
        for k in range(Self.ROLLOUT_LEN):
            term_p[k] = Scalar[DT](0.0)
