"""PPOActStep — N_ENVS-batched action selection (actor sample + critic eval).

Per-step entry point. Runs:
  1. actor.forward[BATCH=N_ENVS] on `state.ob1` → `state.ao1`.
  2. box_muller_normal → `state.z` (N_ENVS * ACT noise samples).
  3. For each env, each act dim: clamped log_std, sample, clamp, log_prob.
  4. critic.forward[BATCH=N_ENVS] on `state.ob1` → `state.v1`.
  5. Writes env-ready actions; caches per-env (sample, log_prob, value).

Per-env caches (cached_action / cached_log_prob / cached_value) live as
Scratches sized N_ENVS, consumed by the next `PPORecordStep.step`.

Greedy variant: deterministic — uses mu directly, no sampling, doesn't
touch the cache. N=1 greedy path exposed via list-based wrapper for
single-env eval.
"""

from std.math import exp as fexp
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import Module
from mojo_rl.nn.random.box_muller import box_muller_normal
from ...training.onpolicy_state import OnPolicyState


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
        N_ENVS: Int,
    ](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, ROLLOUT_LEN, MINIBATCH, N_ENVS,
        ],
        mut actor: Self.ACTOR,
        mut critic: Self.CRITIC,
        obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_scale: Scalar[DT],
    ) raises:
        """Sample N_ENVS PPO actions. Reads N_ENVS × OBS from obs_ptr,
        writes N_ENVS × ACT env-ready (clamped, scaled) actions into
        action_ptr, caches per-env (sample, log_prob, value) into state
        for the upcoming PPORecordStep.

        Both pointers are host-side (rollout buffer is host-only on
        GPU train_target — see OnPolicyState docstring). On GPU,
        actor + critic forward run on device via H2D/D2H staging."""
        var ob1_cpu_p  = state.ob1.cpu_ptr()
        var ao1_cpu_p  = state.ao1.cpu_ptr()
        var v1_cpu_p   = state.v1.cpu_ptr()
        var z_cpu_p    = state.z.cpu_ptr()
        var ca_cpu_p   = state.cached_action.cpu_ptr()
        var clp_cpu_p  = state.cached_log_prob.cpu_ptr()
        var cval_cpu_p = state.cached_value.cpu_ptr()

        # Stage N_ENVS × OBS into host mirror of ob1.
        for e in range(N_ENVS):
            for d in range(Self.OBS):
                ob1_cpu_p[e * Self.OBS + d] = obs_ptr[e * Self.OBS + d]

        comptime if target == "cpu":
            var ob1_t = TileTensor(ob1_cpu_p, row_major[N_ENVS, Self.OBS]())
            var ao1_t = TileTensor(ao1_cpu_p, row_major[N_ENVS, 2 * Self.ACT]())
            actor.forward[target, N_ENVS](ob1_t, output=ao1_t)
            var v1_t = TileTensor(v1_cpu_p, row_major[N_ENVS, 1]())
            critic.forward[target, N_ENVS](ob1_t, output=v1_t)
        else:
            var ctx = state.ctx.value()
            ctx.enqueue_copy(state.ob1.dev.value(), ob1_cpu_p)
            var ob1_dev_t = TileTensor(
                state.ob1.dev_ptr(), row_major[N_ENVS, Self.OBS](),
            )
            var ao1_dev_t = TileTensor(
                state.ao1.dev_ptr(), row_major[N_ENVS, 2 * Self.ACT](),
            )
            var v1_dev_t = TileTensor(
                state.v1.dev_ptr(), row_major[N_ENVS, 1](),
            )
            actor.forward[target, N_ENVS](ob1_dev_t, output=ao1_dev_t)
            critic.forward[target, N_ENVS](ob1_dev_t, output=v1_dev_t)
            ctx.enqueue_copy(ao1_cpu_p, state.ao1.dev.value())
            ctx.enqueue_copy(v1_cpu_p,  state.v1.dev.value())
            ctx.synchronize()

        box_muller_normal(z_cpu_p, N_ENVS * Self.ACT)
        for e in range(N_ENVS):
            var lp_total: Scalar[DT] = 0.0
            for j in range(Self.ACT):
                var mu = ao1_cpu_p[e * 2 * Self.ACT + j]
                var ls = _clamp_log_std(
                    ao1_cpu_p[e * 2 * Self.ACT + Self.ACT + j]
                )
                var sample = mu + fexp(ls) * z_cpu_p[e * Self.ACT + j]
                ca_cpu_p[e * Self.ACT + j] = sample
                var env_a = sample
                if env_a > action_scale:
                    env_a = action_scale
                elif env_a < -action_scale:
                    env_a = -action_scale
                action_ptr[e * Self.ACT + j] = env_a
                var zz = (sample - mu) / (fexp(ls) + EPS_STD)
                lp_total += Scalar[DT](-0.5) * (
                    LOG_2PI + Scalar[DT](2.0) * ls + zz * zz
                )
            clp_cpu_p[e]  = lp_total
            cval_cpu_p[e] = v1_cpu_p[e]

    def step_greedy_n1[
        target: StaticString,
        ROLLOUT_LEN: Int,
        MINIBATCH: Int,
        N_ENVS: Int,
    ](
        mut self,
        mut state: OnPolicyState[
            Self.OBS, Self.ACT, ROLLOUT_LEN, MINIBATCH, N_ENVS,
        ],
        mut actor: Self.ACTOR,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        action_scale: Scalar[DT],
    ) raises:
        """Deterministic single-env action for eval — uses mu of env 0
        directly. Does not touch the cache (eval bypasses the rollout
        buffer). Always BATCH=1 even when state was sized for N_ENVS>1
        (writes only the first OBS/ACT slot of ob1/ao1)."""
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
