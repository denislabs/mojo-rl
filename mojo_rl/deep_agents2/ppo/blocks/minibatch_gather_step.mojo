"""PPOMinibatchGatherStep — Fisher-Yates shuffle + minibatch gather.

Three methods:
  - `reset_indices[target]` — write [0..ROLLOUT_LEN) into state.indices
    (called once per rollout, BEFORE the K-epoch loop — epoch shuffles
    operate on whatever state the previous epoch left behind).
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

from mojo_rl.nn2.constants import DT
from ...training.gae import normalize_in_place
from ...training.onpolicy_state import OnPolicyState


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
        comptime assert target == "cpu" or target == "gpu", (
            "PPOMinibatchGatherStep: target must be 'cpu' or 'gpu'"
        )
        return Self()

    def reset_indices[target: StaticString, N_ENVS: Int](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, Self.ROLLOUT_LEN, Self.MINIBATCH, N_ENVS,
        ],
    ) raises:
        """Write [0..ROLLOUT_LEN*N_ENVS) into state.indices. Caller
        invokes this ONCE per rollout before the K-epoch loop.
        Subsequent epoch shuffles operate on whatever state the
        previous epoch left behind — bit-identity-critical (legacy
        resets once per rollout, not once per epoch)."""
        var idx_p = state.indices.value()
        for k in range(Self.ROLLOUT_LEN * N_ENVS):
            idx_p[k] = Int32(k)

    def shuffle_epoch[target: StaticString, N_ENVS: Int](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, Self.ROLLOUT_LEN, Self.MINIBATCH, N_ENVS,
        ],
    ) raises:
        """In-place Fisher-Yates over state.indices (length
        ROLLOUT_LEN*N_ENVS). Caller invokes this once at the top of
        each K-epoch (after `reset_indices` on the first epoch)."""
        var n_total = Self.ROLLOUT_LEN * N_ENVS
        var idx_p = state.indices.value()
        for t in range(n_total - 1, 0, -1):
            var j = Int(random_float64() * Float64(t + 1))
            if j > t:
                j = t
            var tmp = idx_p[t]
            idx_p[t] = idx_p[j]
            idx_p[j] = tmp

    def gather[target: StaticString, N_ENVS: Int](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, Self.ROLLOUT_LEN, Self.MINIBATCH, N_ENVS,
        ],
        mb_idx: Int,
    ) raises:
        """Gather the `mb_idx`-th minibatch from the flat
        ROLLOUT_LEN*N_ENVS pool into mb_obs/mb_act/mb_olp/mb_adv/mb_ret,
        mean/std normalise mb_adv in place, then (on GPU) H2D upload
        the populated mb_* host mirrors.

        At N_ENVS=1 the flat index space equals the per-env time index,
        so the math reduces to the N=1 case bit-identically."""
        var obs_p = state.obs_buf.cpu_ptr()
        var act_p = state.act_buf.cpu_ptr()
        var olp_p = state.olp_buf.cpu_ptr()
        var adv_p = state.adv_buf.cpu_ptr()
        var ret_p = state.ret_buf.cpu_ptr()
        var mb_obs_p = state.mb_obs.cpu_ptr()
        var mb_act_p = state.mb_act.cpu_ptr()
        var mb_olp_p = state.mb_olp.cpu_ptr()
        var mb_adv_p = state.mb_adv.cpu_ptr()
        var mb_ret_p = state.mb_ret.cpu_ptr()
        var idx_p = state.indices.value()
        for k in range(Self.MINIBATCH):
            var src = Int(idx_p[mb_idx * Self.MINIBATCH + k])
            for d in range(Self.OBS):
                mb_obs_p[k * Self.OBS + d] = obs_p[src * Self.OBS + d]
            for j in range(Self.ACT):
                mb_act_p[k * Self.ACT + j] = act_p[src * Self.ACT + j]
            mb_olp_p[k] = olp_p[src]
            mb_adv_p[k] = adv_p[src]
            mb_ret_p[k] = ret_p[src]
        normalize_in_place(Self.MINIBATCH, mb_adv_p)

        comptime if target == "gpu":
            # H2D the populated minibatch so the train steps read
            # device-side pointers via state.mb_*.target_ptr["gpu"]().
            var ctx = state.ctx.value()
            ctx.enqueue_copy(state.mb_obs.dev.value(), mb_obs_p)
            ctx.enqueue_copy(state.mb_act.dev.value(), mb_act_p)
            ctx.enqueue_copy(state.mb_olp.dev.value(), mb_olp_p)
            ctx.enqueue_copy(state.mb_adv.dev.value(), mb_adv_p)
            ctx.enqueue_copy(state.mb_ret.dev.value(), mb_ret_p)
