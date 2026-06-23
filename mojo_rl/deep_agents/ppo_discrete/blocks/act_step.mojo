"""PPODiscreteActStep — N_ENVS-batched categorical action selection (STORAGE).

Discrete sibling of `ppo/blocks/act_step.mojo`. Per env-step:
  1. actor.forward[BATCH=N_ENVS] on `state.ob1` → own `logits` Tensor
     (N_ENVS × N_ACTIONS).
  2. critic.forward[BATCH=N_ENVS] on `state.ob1` → `state.v1`.
  3. Host-side: per env, softmax(logits) → categorical sample via a
     U(0,1) draw → write the action INDEX (as a float) into both the
     env-ready `action_ptr` and the per-env cache; log p(a) into the
     log-prob cache; V(s) into the value cache.

The action index lives in `state.cached_action` (width ACT=1) and is
pushed into the rollout by `PPORecordStep` exactly like the continuous
sample. No Gaussian noise buffer is used — `state.z` / `state.ao1` are
left untouched (they are sized for the degenerate ACT=1 case and unused
on the discrete path).

The actor's logit output is wider than `state.ao1` (which is sized for
the continuous [mu|log_std]=2·ACT layout), so this block owns its own
`logits` storage `Tensor` (sized at struct level via `N_ENVS_`), giving
both a host mirror and — on GPU — a device buffer for the on-device forward.

STORAGE migration: nets are storage `Module`s (`forward[target, B, POLICY](
TensorRefs[1](ob1), logits, ctx)`). The obs/output staging works on the storage
tensors' host `.data` (sanctioned host loops). On GPU `ob1.upload(ctx)` stages
H2D, the actor/critic forward runs on device, then `logits.download` /
`v1.download` read the result back on host for the sampling walk.

Greedy variant: deterministic argmax over logits — no sampling, no
cache writes (eval bypasses the rollout buffer).
"""

from std.math import exp as fexp, log as flog
from std.gpu.host import DeviceContext
from std.random import random_float64

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.amp import AMPPolicy, NoAMP
from ...training.onpolicy_state import OnPolicyState


struct PPODiscreteActStep[
    OBS_: Int,
    N_ACTIONS_: Int,
    N_ENVS_: Int,
    ACTOR: Module,
    CRITIC: Module,
](Defaultable & Movable & ImplicitlyDeletable):
    comptime OBS = Self.OBS_
    comptime N_ACTIONS = Self.N_ACTIONS_
    comptime N_ENVS = Self.N_ENVS_

    # Own logit buffer (host mirror + device buffer on GPU). Sized for
    # the full N_ENVS sweep; the greedy N=1 path uses the first row.
    var logits: Tensor

    def __init__(out self):
        self.logits = Tensor()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "PPODiscreteActStep: target must be 'cpu' or 'gpu'"
        )
        comptime if target == "gpu":
            if not ctx:
                raise Error("PPODiscreteActStep.make[target='gpu']: ctx required")
        var s = Self()
        s.logits = Tensor.make[target](Self.N_ENVS_ * Self.N_ACTIONS_, ctx)
        return s^

    def step[
        target: StaticString,
        ROLLOUT_LEN: Int,
        MINIBATCH: Int,
        N_ENVS: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, 1, ROLLOUT_LEN, MINIBATCH, N_ENVS,
        ],
        mut actor: Self.ACTOR,
        mut critic: Self.CRITIC,
        obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Sample N_ENVS categorical actions. Reads N_ENVS × OBS from
        obs_ptr, writes N_ENVS action indices (as floats) into action_ptr,
        caches per-env (index, log_prob, value) into state for the
        upcoming PPORecordStep."""
        comptime assert N_ENVS == Self.N_ENVS_, (
            "PPODiscreteActStep.step: method N_ENVS must equal struct N_ENVS_"
        )
        comptime N = Self.N_ACTIONS

        # Stage N_ENVS × OBS into the host mirror of ob1 (index `.data`
        # directly; obs_ptr is the driver trait ABI).
        for e in range(N_ENVS):
            for d in range(Self.OBS):
                state.ob1.data[e * Self.OBS + d] = obs_ptr[e * Self.OBS + d]

        comptime if target == "gpu":
            var ctx = state.ctx.value()
            state.ob1.upload(ctx)
            actor.forward[target, N_ENVS, POLICY=POLICY](
                TensorRefs[Self.ACTOR.ARITY](state.ob1), self.logits, state.ctx
            )
            critic.forward[target, N_ENVS, POLICY=POLICY](
                TensorRefs[Self.CRITIC.ARITY](state.ob1), state.v1, state.ctx
            )
            self.logits.download(ctx)
            state.v1.download(ctx)
        else:
            actor.forward[target, N_ENVS, POLICY=POLICY](
                TensorRefs[Self.ACTOR.ARITY](state.ob1), self.logits, state.ctx
            )
            critic.forward[target, N_ENVS, POLICY=POLICY](
                TensorRefs[Self.CRITIC.ARITY](state.ob1), state.v1, state.ctx
            )

        # Host-side softmax + categorical sample per env (index `.data`).
        ref lg = self.logits.data
        ref v1 = state.v1.data
        ref ca = state.cached_action.data
        ref clp = state.cached_log_prob.data
        ref cval = state.cached_value.data
        for e in range(N_ENVS):
            var base = e * N
            var max_l = lg[base]
            for j in range(1, N):
                var lj = lg[base + j]
                if lj > max_l:
                    max_l = lj
            var sum_exp: Scalar[DT] = 0.0
            for j in range(N):
                sum_exp += fexp(lg[base + j] - max_l)
            # Sample via inverse-CDF over the softmax.
            var u = Scalar[DT](random_float64())
            var cum: Scalar[DT] = 0.0
            var a_idx: Int = N - 1
            for j in range(N):
                var p_j = fexp(lg[base + j] - max_l) / sum_exp
                cum += p_j
                if u <= cum:
                    a_idx = j
                    break
            var log_sum = flog(sum_exp)
            var log_p_a = (lg[base + a_idx] - max_l) - log_sum
            ca[e] = Scalar[DT](a_idx)
            action_ptr[e] = Scalar[DT](a_idx)
            clp[e] = log_p_a
            cval[e] = v1[e]

    def step_greedy_n1[
        target: StaticString,
        ROLLOUT_LEN: Int,
        MINIBATCH: Int,
        N_ENVS: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, 1, ROLLOUT_LEN, MINIBATCH, N_ENVS,
        ],
        mut actor: Self.ACTOR,
        ref obs: List[Scalar[DT]],
    ) raises -> Int:
        """Deterministic single-env action for eval — argmax over the
        logits of env 0. Returns the action index. Does not touch the
        cache. Always BATCH=1."""
        comptime N = Self.N_ACTIONS
        for d in range(Self.OBS):
            state.ob1.data[d] = obs[d]
        comptime if target == "gpu":
            var ctx = state.ctx.value()
            state.ob1.upload(ctx)
            actor.forward[target, 1, POLICY=POLICY](
                TensorRefs[Self.ACTOR.ARITY](state.ob1), self.logits, state.ctx
            )
            self.logits.download(ctx)
        else:
            actor.forward[target, 1, POLICY=POLICY](
                TensorRefs[Self.ACTOR.ARITY](state.ob1), self.logits, state.ctx
            )
        ref lg = self.logits.data
        var best: Int = 0
        var best_l = lg[0]
        for j in range(1, N):
            if lg[j] > best_l:
                best_l = lg[j]
                best = j
        return best
