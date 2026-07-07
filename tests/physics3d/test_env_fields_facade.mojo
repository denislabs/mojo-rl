"""P5 gate: Phyics3dEnvFields (physics on the per-field path) vs the legacy
Phyics3dEnv, InvertedPendulum, identical injected state + identical
deterministic action sequence, full episodes.

Both envs run MODEL_DEF-default RK4 (legacy: CPU RK4Integrator via the
config substep; fields: RK4IntegratorFields) at float64 — different
implementations of the same integrator, so the comparison is tolerance-
based (like the legacy-CPU cross-checks in the vs_mujoco gates), with
rewards and done timing required to MATCH EXACTLY over the horizon.

Also proves BoxContinuousActionEnv conformance by driving the fields env
through a generic function (the driver-shaped seam).

Run: pixi run -e apple mojo run -I . tests/physics3d/test_env_fields_facade.mojo
"""

from std.math import abs, sin
from std.gpu.host import DeviceContext

from mojo_rl.core.env_traits import BoxContinuousActionEnv
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.envs.phyics3d_env_fields import Phyics3dEnvFields
from mojo_rl.envs.inverted_pendulum.inverted_pendulum_xml import (
    InvertedPendulumModel,
)
from mojo_rl.envs.inverted_pendulum.inverted_pendulum_config import (
    InvertedPendulumConfig,
)

comptime DTYPE = DType.float64
comptime IPM = InvertedPendulumModel
comptime NQ = IPM.NQ
comptime NV = IPM.NV
comptime OBS_DIM = IPM.OBS_DIM
comptime N_STEPS = 100

comptime LegacyEnv = Phyics3dEnv[
    IPM, InvertedPendulumConfig, DTYPE, TERMINATE_ON_UNHEALTHY=True
]
comptime FieldsEnv = Phyics3dEnvFields[
    IPM, InvertedPendulumConfig, DTYPE, TERMINATE_ON_UNHEALTHY=True
]


def _action(t: Int) -> Float64:
    return 0.8 * sin(Float64(t) * 0.13)


def _generic_episode[
    E: BoxContinuousActionEnv
](mut env: E, n: Int) raises -> Float64:
    """Drive any BoxContinuousActionEnv generically (driver-shaped seam)."""
    _ = env.reset_obs_list()
    var total = Float64(0)
    for t in range(n):
        var acts = List[Scalar[DType.float64]]()
        for _ in range(env.action_dim()):
            acts.append(Scalar[DType.float64](_action(t)))
        var r = env.step_continuous_vec[DType.float64](acts)
        total += Float64(r[1])
        if r[2]:
            break
    return total


def main() raises:
    print("--- P5 facade: fields env vs legacy env, InvertedPendulum ---")
    var ctx = DeviceContext()

    var env_l = LegacyEnv()
    var env_f = FieldsEnv(ctx)

    # Deterministic identical start (bypass noisy reset on both).
    _ = env_l.reset_obs_list()
    _ = env_f.reset_obs_list()
    var qpos0 = List[Float64]()
    qpos0.append(0.0)
    qpos0.append(0.08)  # pole tilt
    var qvel0 = List[Float64]()
    qvel0.append(0.0)
    qvel0.append(0.0)
    for i in range(NQ):
        env_l.data.qpos[i] = Scalar[DTYPE](qpos0[i])
    for i in range(NV):
        env_l.data.qvel[i] = Scalar[DTYPE](qvel0[i])
    env_f.set_state(qpos0, qvel0)

    var worst_obs = Float64(0)
    var reward_l = Float64(0)
    var reward_f = Float64(0)
    var done_step_l = -1
    var done_step_f = -1
    for t in range(N_STEPS):
        var acts = List[Scalar[DTYPE]]()
        acts.append(Scalar[DTYPE](_action(t)))
        var rl = env_l.step_continuous_vec[DTYPE](acts)
        var rf = env_f.step_continuous_vec[DTYPE](acts)
        for i in range(OBS_DIM):
            var e = abs(Float64(rf[0][i]) - Float64(rl[0][i]))
            if e > worst_obs:
                worst_obs = e
        reward_l += Float64(rl[1])
        reward_f += Float64(rf[1])
        if rl[2] and done_step_l < 0:
            done_step_l = t
        if rf[2] and done_step_f < 0:
            done_step_f = t
        if rl[2] or rf[2]:
            break

    print(
        "  worst obs err over", N_STEPS, "steps:", worst_obs,
        " returns: legacy=", reward_l, " fields=", reward_f,
        " done@ legacy=", done_step_l, " fields=", done_step_f,
    )
    if done_step_l != done_step_f:
        raise Error("done timing differs between legacy and fields env")
    if reward_l != reward_f:
        raise Error("episode returns differ")
    if worst_obs > 1e-6:
        raise Error("obs trajectory diverged beyond f64 RK4 impl budget")
    print("  PASS: obs within 1e-6, rewards + done timing exact")

    # Generic-trait conformance: run the fields env through a generic fn.
    var env_g = FieldsEnv(ctx)
    var total = _generic_episode(env_g, 30)
    print("  generic BoxContinuousActionEnv episode return:", total)
    if total <= 0:
        raise Error("generic episode returned no reward")
    print("  PASS: BoxContinuousActionEnv conformance via generic driver fn")

    print("test_env_fields_facade: ALL PASS")
