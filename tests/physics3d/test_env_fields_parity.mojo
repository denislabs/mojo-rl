"""Stage-C parity gate: Phyics3dEnvFields vs the legacy Phyics3dEnv across the
MuJoCo envs, before re-homing the env aliases onto the fields engine.

For each env: construct both facades, seed BOTH from the legacy env's
post-reset state (identical qpos/qvel — the fields env via set_state), then
drive K identical deterministic-action steps. Compare per-step observations,
per-step rewards, episode return, and done timing. Both run the model-default
integrator (RK4/Euler) at float64 — different implementations of the same
integrator, so tiny roundoff is expected; a STRUCTURAL difference (a missing
reward term, dropped tendon/contact physics) shows up as an O(1e-2+) gap at
step 1, far above roundoff.

Specifically targets the two flagged risks: Ant's contact-cost reward
(cvel/cfrc_ext) and Humanoid's tendon constraints.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_env_fields_parity.mojo
"""

from std.math import abs, sin
from std.gpu.host import DeviceContext

from mojo_rl.physics3d.model.model_def import ModelDefLike
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.envs.phyics3d_env_fields import Phyics3dEnvFields
from mojo_rl.envs.phyics3d_env_config import Phyics3dEnvConfig

from mojo_rl.envs.half_cheetah import HalfCheetahModel, HalfCheetahConfig
from mojo_rl.envs.ant import AntModel, AntConfig
from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel
from mojo_rl.envs.walker2d.walker2d_config import Walker2dConfig
from mojo_rl.envs.humanoid.humanoid_xml import HumanoidModel
from mojo_rl.envs.humanoid.humanoid_config import HumanoidConfig
from mojo_rl.envs.swimmer.swimmer_xml import SwimmerModel
from mojo_rl.envs.swimmer.swimmer_config import SwimmerConfig
from mojo_rl.envs.reacher.reacher_xml import ReacherModel
from mojo_rl.envs.reacher.reacher_config import ReacherConfig

comptime DTYPE = DType.float64
comptime K = 5


def _action(t: Int, k: Int) -> Float64:
    return 0.5 * sin(Float64(t * 7 + k) * 0.11)


def _parity[
    MODEL: ModelDefLike, CONFIG: Phyics3dEnvConfig, TERM: Bool
](ctx: DeviceContext, name: String, obs_tol: Float64, rew_tol: Float64) raises:
    comptime LE = Phyics3dEnv[MODEL, CONFIG, DTYPE, TERM]
    comptime FE = Phyics3dEnvFields[MODEL, CONFIG, DTYPE, TERM]
    comptime NQ = MODEL.NQ
    comptime NV = MODEL.NV
    comptime OBS = MODEL.OBS_DIM

    var el = LE()
    var ef = FE(ctx)
    _ = el.reset_obs_list()
    _ = ef.reset_obs_list()

    # Shared start = legacy's post-reset state (a valid, normalized pose).
    var qpos = List[Float64]()
    for i in range(NQ):
        qpos.append(Float64(el.data.qpos[i]))
    var qvel = List[Float64]()
    for i in range(NV):
        qvel.append(Float64(el.data.qvel[i]))
    ef.set_state(qpos, qvel)

    var worst_obs = Float64(0)
    var worst_obs_step = -1
    var worst_rew = Float64(0)
    var worst_rew_step = -1
    var ret_l = Float64(0)
    var ret_f = Float64(0)
    var done_l = -1
    var done_f = -1
    for t in range(K):
        var acts = List[Scalar[DTYPE]]()
        for k in range(el.action_dim()):
            acts.append(Scalar[DTYPE](_action(t, k)))
        var rl = el.step_continuous_vec[DTYPE](acts)
        var rf = ef.step_continuous_vec[DTYPE](acts)
        for i in range(OBS):
            var e = abs(Float64(rf[0][i]) - Float64(rl[0][i]))
            if e > worst_obs:
                worst_obs = e
                worst_obs_step = t
        var re = abs(Float64(rf[1]) - Float64(rl[1]))
        if re > worst_rew:
            worst_rew = re
            worst_rew_step = t
        ret_l += Float64(rl[1])
        ret_f += Float64(rf[1])
        if rl[2] and done_l < 0:
            done_l = t
        if rf[2] and done_f < 0:
            done_f = t
        if rl[2] or rf[2]:
            break

    print(
        "  ", name, ": worst obs err", worst_obs, "@", worst_obs_step,
        " worst rew err", worst_rew, "@", worst_rew_step,
    )
    print(
        "      return legacy", ret_l, " fields", ret_f, " done l", done_l,
        "f", done_f,
    )
    if done_l != done_f:
        raise Error(name + ": done timing differs (legacy vs fields)")
    if worst_obs > obs_tol:
        raise Error(name + ": obs parity exceeds tol " + String(obs_tol))
    if worst_rew > rew_tol:
        raise Error(name + ": reward parity exceeds tol " + String(rew_tol))
    print("     ", name, "PASS")


def main() raises:
    print("--- Phyics3dEnvFields vs legacy Phyics3dEnv parity (K=", K, ") ---")
    var ctx = DeviceContext()

    _parity[HalfCheetahModel, HalfCheetahConfig, False](
        ctx, "HalfCheetah", 1e-2, 1e-3
    )
    _parity[Walker2dModel, Walker2dConfig, True](ctx, "Walker2d", 1e-2, 1e-3)
    _parity[ReacherModel, ReacherConfig, False](ctx, "Reacher", 1e-2, 1e-3)
    _parity[SwimmerModel, SwimmerConfig, False](ctx, "Swimmer", 1e-2, 1e-3)
    _parity[AntModel, AntConfig, True](ctx, "Ant", 1e-2, 1e-3)
    _parity[HumanoidModel, HumanoidConfig, True](ctx, "Humanoid", 1e-2, 1e-3)

    print("test_env_fields_parity: ALL PASS")
