"""PPOMinibatchGatherStep — Fisher-Yates shuffle + minibatch gather.

Three methods:
  - `reset_indices[target]` — write [0..ROLLOUT_LEN) into state.indices
    (called once per rollout, BEFORE the K-epoch loop — matches legacy
    PPOTrainer ordering for bit-identity).
  - `shuffle_epoch[target]` — in-place Fisher-Yates over state.indices
    (called once per K-epoch, AFTER reset_indices on the first epoch).
  - `gather[target]` — gather the `mb_idx`-th minibatch into mb_obs /
    mb_act / mb_olp / mb_adv / mb_ret, then mean/std normalise mb_adv
    (CleanRL per-minibatch normalisation).

Indices are Int32 (not DT) so they live on the raw `state.indices`
pointer rather than a Scratch.
"""

from std.gpu.host import DeviceContext
from std.random import random_float64

from ...constants import DT
from ..gae import normalize_in_place
from ..onpolicy_state import OnPolicyState


struct PPOMinibatchGatherStep[
    OBS_: Int,
    ACT_: Int,
    ROLLOUT_LEN_: Int,
    MINIBATCH_: Int,
](Defaultable & Movable & ImplicitlyDestructible):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_
    comptime ROLLOUT_LEN = Self.ROLLOUT_LEN_
    comptime MINIBATCH = Self.MINIBATCH_

    def __init__(out self):
        pass

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu", (
            "PPOMinibatchGatherStep: P.1 is CPU-only (GPU lands in P.2)"
        )
        return Self()

    def reset_indices[target: StaticString](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, Self.ROLLOUT_LEN, Self.MINIBATCH,
        ],
    ) raises:
        """Write [0..ROLLOUT_LEN) into state.indices. Caller invokes
        this ONCE per rollout, before the K-epoch loop. Subsequent
        epoch shuffles operate on whatever state the previous epoch
        left behind — bit-identity-critical (legacy resets once per
        rollout, not once per epoch)."""
        for k in range(Self.ROLLOUT_LEN):
            state.indices[k] = Int32(k)

    def shuffle_epoch[target: StaticString](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, Self.ROLLOUT_LEN, Self.MINIBATCH,
        ],
    ) raises:
        """In-place Fisher-Yates over state.indices. Caller invokes
        this once at the top of each K-epoch (after `reset_indices` on
        the first epoch)."""
        for t in range(Self.ROLLOUT_LEN - 1, 0, -1):
            var j = Int(random_float64() * Float64(t + 1))
            if j > t:
                j = t
            var tmp = state.indices[t]
            state.indices[t] = state.indices[j]
            state.indices[j] = tmp

    def gather[target: StaticString](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, Self.ROLLOUT_LEN, Self.MINIBATCH,
        ],
        mb_idx: Int,
    ) raises:
        """Gather the `mb_idx`-th minibatch into mb_obs/mb_act/mb_olp/
        mb_adv/mb_ret, then mean/std normalise mb_adv in place."""
        var obs_p = state.obs_buf.target_ptr[target]()
        var act_p = state.act_buf.target_ptr[target]()
        var olp_p = state.olp_buf.target_ptr[target]()
        var adv_p = state.adv_buf.target_ptr[target]()
        var ret_p = state.ret_buf.target_ptr[target]()
        var mb_obs_p = state.mb_obs.target_ptr[target]()
        var mb_act_p = state.mb_act.target_ptr[target]()
        var mb_olp_p = state.mb_olp.target_ptr[target]()
        var mb_adv_p = state.mb_adv.target_ptr[target]()
        var mb_ret_p = state.mb_ret.target_ptr[target]()
        for k in range(Self.MINIBATCH):
            var src = Int(state.indices[mb_idx * Self.MINIBATCH + k])
            for d in range(Self.OBS):
                mb_obs_p[k * Self.OBS + d] = obs_p[src * Self.OBS + d]
            for j in range(Self.ACT):
                mb_act_p[k * Self.ACT + j] = act_p[src * Self.ACT + j]
            mb_olp_p[k] = olp_p[src]
            mb_adv_p[k] = adv_p[src]
            mb_ret_p[k] = ret_p[src]
        normalize_in_place(Self.MINIBATCH, mb_adv_p)
