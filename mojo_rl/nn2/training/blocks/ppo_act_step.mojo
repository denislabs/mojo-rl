"""PPOActStep — one-env action selection (actor sample + critic eval).

Per-step entry point. Runs:
  1. actor.forward[BATCH=1] on `state.ob1` → `state.ao1` ([mu, log_std]).
  2. box_muller_normal → `state.z`.
  3. For each act dim: clamped log_std, sample = mu + exp(ls) * z,
     env_action = clamp(sample, ±action_scale), accumulate log_prob.
  4. critic.forward[BATCH=1] on `state.ob1` → `state.v1`.
  5. Writes env_action to caller's `action_out` List.
  6. Caches (sample, log_prob, value) into state for the upcoming
     `PPORecordStep.step`.

Greedy variant: deterministic — uses mu directly, no sampling, doesn't
touch the cache.
"""

from std.math import exp as fexp
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from ...constants import DT
from ...core.module import Module
from ...random.box_muller import box_muller_normal
from ..onpolicy_state import OnPolicyState


comptime LOG_2PI: Scalar[DT] = 1.8378770664093453
comptime EPS_STD: Scalar[DT] = 1e-6
comptime LOG_STD_MIN_F: Scalar[DT] = -5.0
comptime LOG_STD_MAX_F: Scalar[DT] = 2.0


def _clamp_log_std(ls: Scalar[DT]) -> Scalar[DT]:
    if ls < LOG_STD_MIN_F:
        return LOG_STD_MIN_F
    elif ls > LOG_STD_MAX_F:
        return LOG_STD_MAX_F
    return ls


struct PPOActStep[
    OBS_: Int,
    ACT_: Int,
    ACTOR: Module,
    CRITIC: Module,
](Defaultable & Movable & ImplicitlyDestructible):
    comptime OBS = Self.OBS_
    comptime ACT = Self.ACT_

    def __init__(out self):
        pass

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "PPOActStep: target must be 'cpu' or 'gpu'"
        )
        comptime if target == "gpu":
            if not ctx:
                raise Error("PPOActStep.make[target='gpu']: ctx required")
        return Self()

    def step[
        target: StaticString,
        ROLLOUT_LEN: Int,
        MINIBATCH: Int,
    ](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, ROLLOUT_LEN, MINIBATCH,
        ],
        mut actor: Self.ACTOR,
        mut critic: Self.CRITIC,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        action_scale: Scalar[DT],
    ) raises:
        """Sample one PPO action. Writes env-ready (clamped, scaled)
        action into `action_out`; caches the unbounded sample,
        log_prob, and V(s) into state for the next PPORecordStep.

        GPU path (hybrid N=1): H2D obs → device ob1, actor.forward and
        critic.forward on device, D2H ao1 + v1 → host mirrors, host-
        side sample/log_prob/clamp. cached_action stays on the host
        mirror (rollout buffer is host-only on GPU train_target — see
        OnPolicyState docstring)."""
        # Host mirror pointers (always valid; STAGING=True ensures the
        # CPU mirror exists on GPU too).
        var ob1_cpu_p = state.ob1.cpu_ptr()
        var ao1_cpu_p = state.ao1.cpu_ptr()
        var v1_cpu_p  = state.v1.cpu_ptr()
        var z_cpu_p   = state.z.cpu_ptr()
        var ca_cpu_p  = state.cached_action.cpu_ptr()

        # Stage obs into host mirror.
        for d in range(Self.OBS):
            ob1_cpu_p[d] = obs[d]

        comptime if target == "cpu":
            var ob1_t = TileTensor(ob1_cpu_p, row_major[1, Self.OBS]())
            var ao1_t = TileTensor(ao1_cpu_p, row_major[1, 2 * Self.ACT]())
            actor.forward[target, 1](ob1_t, output=ao1_t)
            var v1_t = TileTensor(v1_cpu_p, row_major[1, 1]())
            critic.forward[target, 1](ob1_t, output=v1_t)
        else:
            var ctx = state.ctx.value()
            ctx.enqueue_copy(state.ob1.dev.value(), ob1_cpu_p)
            var ob1_dev_t = TileTensor(
                state.ob1.dev_ptr(), row_major[1, Self.OBS](),
            )
            var ao1_dev_t = TileTensor(
                state.ao1.dev_ptr(), row_major[1, 2 * Self.ACT](),
            )
            var v1_dev_t = TileTensor(
                state.v1.dev_ptr(), row_major[1, 1](),
            )
            actor.forward[target, 1](ob1_dev_t, output=ao1_dev_t)
            critic.forward[target, 1](ob1_dev_t, output=v1_dev_t)
            ctx.enqueue_copy(ao1_cpu_p, state.ao1.dev.value())
            ctx.enqueue_copy(v1_cpu_p,  state.v1.dev.value())
            ctx.synchronize()

        box_muller_normal(z_cpu_p, Self.ACT)
        var lp_total: Scalar[DT] = 0.0
        for j in range(Self.ACT):
            var mu = ao1_cpu_p[j]
            var ls = _clamp_log_std(ao1_cpu_p[Self.ACT + j])
            var sample = mu + fexp(ls) * z_cpu_p[j]
            ca_cpu_p[j] = sample
            var env_a = sample
            if env_a > action_scale:
                env_a = action_scale
            elif env_a < -action_scale:
                env_a = -action_scale
            action_out[j] = env_a
            var zz = (sample - mu) / (fexp(ls) + EPS_STD)
            lp_total += Scalar[DT](-0.5) * (
                LOG_2PI + Scalar[DT](2.0) * ls + zz * zz
            )
        state.cached_log_prob = lp_total
        state.cached_value = v1_cpu_p[0]

    def step_greedy[
        target: StaticString,
        ROLLOUT_LEN: Int,
        MINIBATCH: Int,
    ](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, ROLLOUT_LEN, MINIBATCH,
        ],
        mut actor: Self.ACTOR,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        action_scale: Scalar[DT],
    ) raises:
        """Deterministic action for eval — uses mu directly. Does not
        touch the cache (eval bypasses the rollout buffer)."""
        var ob1_cpu_p = state.ob1.cpu_ptr()
        var ao1_cpu_p = state.ao1.cpu_ptr()
        for d in range(Self.OBS):
            ob1_cpu_p[d] = obs[d]
        comptime if target == "cpu":
            var ob1_t = TileTensor(ob1_cpu_p, row_major[1, Self.OBS]())
            var ao1_t = TileTensor(ao1_cpu_p, row_major[1, 2 * Self.ACT]())
            actor.forward[target, 1](ob1_t, output=ao1_t)
        else:
            var ctx = state.ctx.value()
            ctx.enqueue_copy(state.ob1.dev.value(), ob1_cpu_p)
            var ob1_dev_t = TileTensor(
                state.ob1.dev_ptr(), row_major[1, Self.OBS](),
            )
            var ao1_dev_t = TileTensor(
                state.ao1.dev_ptr(), row_major[1, 2 * Self.ACT](),
            )
            actor.forward[target, 1](ob1_dev_t, output=ao1_dev_t)
            ctx.enqueue_copy(ao1_cpu_p, state.ao1.dev.value())
            ctx.synchronize()
        for j in range(Self.ACT):
            var env_a = ao1_cpu_p[j]
            if env_a > action_scale:
                env_a = action_scale
            elif env_a < -action_scale:
                env_a = -action_scale
            action_out[j] = env_a
