"""PPOMinibatchGatherStep — Fisher-Yates shuffle + minibatch gather (STORAGE).

Three methods:
  - `reset_indices[target]` — write [0..ROLLOUT_LEN) into state.indices
    (called once per rollout, BEFORE the K-epoch loop — epoch shuffles
    operate on whatever state the previous epoch left behind).
  - `shuffle_epoch[target]` — in-place Fisher-Yates over state.indices
    (called once per K-epoch, AFTER reset_indices on the first epoch).
  - `gather[target]` — gather the `mb_idx`-th minibatch into mb_obs /
    mb_act / mb_olp / mb_adv / mb_ret, then mean/std normalise mb_adv
    (CleanRL per-minibatch normalisation).

STORAGE migration: the rollout pool + the minibatch staging both index the
storage tensors' host `.data` Lists directly. On GPU the populated mb_* host
mirrors are `upload`ed so the actor/critic train steps read the device buffers.
Indices stay an Int32 raw pointer on `state.indices` (Tensor is DT-only).
"""

from std.gpu.host import DeviceContext
from std.random import random_float64
from std.math import sqrt as fsqrt

from mojo_rl.nn.constants import DT
from ...training.onpolicy_state import OnPolicyState


struct PPOMinibatchGatherStep[
    OBS_: Int,
    ACT_: Int,
    ROLLOUT_LEN_: Int,
    MINIBATCH_: Int,
](Defaultable & Movable & ImplicitlyDeletable):
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
        # Gather works on the rollout pool + minibatch staging host `.data`
        # Lists directly (no raw pointers). `state.indices` is an Int32 array
        # (Tensor is DT-only) so it stays a raw pointer.
        ref obs = state.obs_buf.data
        ref act = state.act_buf.data
        ref olp = state.olp_buf.data
        ref adv = state.adv_buf.data
        ref ret = state.ret_buf.data
        ref mb_obs = state.mb_obs.data
        ref mb_act = state.mb_act.data
        ref mb_olp = state.mb_olp.data
        ref mb_adv = state.mb_adv.data
        ref mb_ret = state.mb_ret.data
        var idx_p = state.indices.value()
        for k in range(Self.MINIBATCH):
            var src = Int(idx_p[mb_idx * Self.MINIBATCH + k])
            for d in range(Self.OBS):
                mb_obs[k * Self.OBS + d] = obs[src * Self.OBS + d]
            for j in range(Self.ACT):
                mb_act[k * Self.ACT + j] = act[src * Self.ACT + j]
            mb_olp[k] = olp[src]
            mb_adv[k] = adv[src]
            mb_ret[k] = ret[src]
        # CleanRL per-minibatch advantage normalisation (subtract mean, divide
        # by std + 1e-8) — inlined over the `mb_adv` List (was a pointer-taking
        # `normalize_in_place` helper).
        var s: Scalar[DT] = 0.0
        for t in range(Self.MINIBATCH):
            s += mb_adv[t]
        var mean = s / Scalar[DT](Self.MINIBATCH)
        var sq: Scalar[DT] = 0.0
        for t in range(Self.MINIBATCH):
            var d = mb_adv[t] - mean
            sq += d * d
        var std = fsqrt(sq / Scalar[DT](Self.MINIBATCH))
        for t in range(Self.MINIBATCH):
            mb_adv[t] = (mb_adv[t] - mean) / (std + Scalar[DT](1e-8))

        comptime if target == "gpu":
            # H2D the populated minibatch so the train steps read the device
            # buffers (state.mb_*.lt["gpu", ...] / .dev.value()).
            var c = state.ctx.value()
            state.mb_obs.upload(c)
            state.mb_act.upload(c)
            state.mb_olp.upload(c)
            state.mb_adv.upload(c)
            state.mb_ret.upload(c)
