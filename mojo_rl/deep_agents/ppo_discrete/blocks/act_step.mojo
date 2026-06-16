"""PPODiscreteActStep — N_ENVS-batched categorical action selection.

Discrete sibling of `ppo/blocks/act_step.mojo`. Per env-step:
  1. actor.forward[BATCH=N_ENVS] on `state.ob1` → own `logits` scratch
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
`logits` Scratch (sized at struct level via `N_ENVS_`), giving both a
host mirror and — on GPU — a device buffer for the on-device forward.

Greedy variant: deterministic argmax over logits — no sampling, no
cache writes (eval bypasses the rollout buffer).
"""

from std.math import exp as fexp, log as flog
from std.gpu.host import DeviceContext
from std.random import random_float64
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.core.scratch import Scratch
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
    var logits: Scratch["disc_logits", Self.N_ENVS_ * Self.N_ACTIONS_, True]

    def __init__(out self):
        self.logits = Scratch[
            "disc_logits", Self.N_ENVS_ * Self.N_ACTIONS_, True
        ]()

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
        s.logits.init_with[target](ctx)
        return s^

    def step[
        target: StaticString,
        ROLLOUT_LEN: Int,
        MINIBATCH: Int,
        N_ENVS: Int,
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
        var ob1_cpu_p  = state.ob1.cpu_ptr()
        var v1_cpu_p   = state.v1.cpu_ptr()
        var lg_cpu_p   = self.logits.cpu_ptr()
        var ca_cpu_p   = state.cached_action.cpu_ptr()
        var clp_cpu_p  = state.cached_log_prob.cpu_ptr()
        var cval_cpu_p = state.cached_value.cpu_ptr()

        # Stage N_ENVS × OBS into host mirror of ob1.
        for e in range(N_ENVS):
            for d in range(Self.OBS):
                ob1_cpu_p[e * Self.OBS + d] = obs_ptr[e * Self.OBS + d]

        comptime if target == "cpu":
            var ob1_t = TileTensor(ob1_cpu_p, row_major[N_ENVS, Self.OBS]())
            var lg_t  = TileTensor(lg_cpu_p, row_major[N_ENVS, N]())
            actor.forward[target, N_ENVS](ob1_t, output=lg_t)
            var v1_t = TileTensor(v1_cpu_p, row_major[N_ENVS, 1]())
            critic.forward[target, N_ENVS](ob1_t, output=v1_t)
        else:
            var ctx = state.ctx.value()
            ctx.enqueue_copy(state.ob1.dev.value(), ob1_cpu_p)
            var ob1_dev_t = TileTensor(
                state.ob1.dev_ptr(), row_major[N_ENVS, Self.OBS](),
            )
            var lg_dev_t = TileTensor(
                self.logits.dev_ptr(), row_major[N_ENVS, N](),
            )
            var v1_dev_t = TileTensor(
                state.v1.dev_ptr(), row_major[N_ENVS, 1](),
            )
            actor.forward[target, N_ENVS](ob1_dev_t, output=lg_dev_t)
            critic.forward[target, N_ENVS](ob1_dev_t, output=v1_dev_t)
            ctx.enqueue_copy(lg_cpu_p, self.logits.dev.value())
            ctx.enqueue_copy(v1_cpu_p, state.v1.dev.value())
            ctx.synchronize()

        # Host-side softmax + categorical sample per env.
        for e in range(N_ENVS):
            var base = e * N
            var max_l = lg_cpu_p[base]
            for j in range(1, N):
                var lj = lg_cpu_p[base + j]
                if lj > max_l:
                    max_l = lj
            var sum_exp: Scalar[DT] = 0.0
            for j in range(N):
                sum_exp += fexp(lg_cpu_p[base + j] - max_l)
            # Sample via inverse-CDF over the softmax.
            var u = Scalar[DT](random_float64())
            var cum: Scalar[DT] = 0.0
            var a_idx: Int = N - 1
            for j in range(N):
                var p_j = fexp(lg_cpu_p[base + j] - max_l) / sum_exp
                cum += p_j
                if u <= cum:
                    a_idx = j
                    break
            var log_sum = flog(sum_exp)
            var log_p_a = (lg_cpu_p[base + a_idx] - max_l) - log_sum
            ca_cpu_p[e]   = Scalar[DT](a_idx)
            action_ptr[e] = Scalar[DT](a_idx)
            clp_cpu_p[e]  = log_p_a
            cval_cpu_p[e] = v1_cpu_p[e]

    def step_greedy_n1[
        target: StaticString,
        ROLLOUT_LEN: Int,
        MINIBATCH: Int,
        N_ENVS: Int,
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
        var ob1_cpu_p = state.ob1.cpu_ptr()
        var lg_cpu_p  = self.logits.cpu_ptr()
        for d in range(Self.OBS):
            ob1_cpu_p[d] = obs[d]
        comptime if target == "cpu":
            var ob1_t = TileTensor(ob1_cpu_p, row_major[1, Self.OBS]())
            var lg_t  = TileTensor(lg_cpu_p, row_major[1, N]())
            actor.forward[target, 1](ob1_t, output=lg_t)
        else:
            var ctx = state.ctx.value()
            ctx.enqueue_copy(state.ob1.dev.value(), ob1_cpu_p)
            var ob1_dev_t = TileTensor(
                state.ob1.dev_ptr(), row_major[1, Self.OBS](),
            )
            var lg_dev_t = TileTensor(
                self.logits.dev_ptr(), row_major[1, N](),
            )
            actor.forward[target, 1](ob1_dev_t, output=lg_dev_t)
            ctx.enqueue_copy(lg_cpu_p, self.logits.dev.value())
            ctx.synchronize()
        var best: Int = 0
        var best_l = lg_cpu_p[0]
        for j in range(1, N):
            if lg_cpu_p[j] > best_l:
                best_l = lg_cpu_p[j]
                best = j
        return best
