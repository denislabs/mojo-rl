"""Coverage gate: single-env `Phyics3dEnv` round-trip smoke over the
MuJoCo envs that previously had only KERNEL-level fields coverage (FK, contact,
integrator kernels) but no facade-level test. For each env we reset, drive N
steps with a deterministic sinusoidal action, and assert:
  * every observation component and every reward stays finite (no NaN/Inf),
  * mid-episode terminations re-reset cleanly, and
  * a fresh reset after the loop does not crash.

Pure integration smoke on the CPU fields path (deterministic) — NOT convergence.
Complements test_env_fields_hopper_smoke (Hopper, contacts) and
test_env_fields_sac_smoke (InvertedPendulum). Legacy-free, so it survives the
physics3d sunset and anchors the facade plumbing for these six envs.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_env_fields_mujoco_roundtrip.mojo
"""

from std.math import sin, abs
from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.core.env_traits import BoxContinuousActionEnv
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.envs.half_cheetah import HalfCheetahModel, HalfCheetahConfig
from mojo_rl.envs.ant import AntModel, AntConfig
from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel
from mojo_rl.envs.walker2d.walker2d_config import Walker2dConfig

# Swimmer (fluid drag) is now supported — fluid forces are applied inside the
# fields integrator step (Stage A). Humanoid exercises tendons + sites on the
# facade.
from mojo_rl.envs.swimmer.swimmer_xml import SwimmerModel
from mojo_rl.envs.swimmer.swimmer_config import SwimmerConfig
from mojo_rl.envs.humanoid.humanoid_xml import HumanoidModel
from mojo_rl.envs.humanoid.humanoid_config import HumanoidConfig
from mojo_rl.envs.inverted_double_pendulum.inverted_double_pendulum_xml import (
    InvertedDoublePendulumModel,
)
from mojo_rl.envs.inverted_double_pendulum.inverted_double_pendulum_config import (
    InvertedDoublePendulumConfig,
)
from mojo_rl.envs.reacher.reacher_xml import ReacherModel
from mojo_rl.envs.reacher.reacher_config import ReacherConfig

comptime N_STEPS = 60


def _finite(x: Float64) -> Bool:
    return x == x and abs(x) < 1e30


def _smoke[E: BoxContinuousActionEnv, OBS: Int](
    mut env: E, name: StringLiteral
) raises:
    _ = env.reset_obs_list()
    var total_r = Float64(0)
    var resets = 0
    for t in range(N_STEPS):
        var acts = List[Scalar[DT]]()
        for k in range(env.action_dim()):
            acts.append(Scalar[DT](0.5 * sin(Float64(t * 7 + k) * 0.11)))
        var r = env.step_continuous_vec[DT](acts)
        for i in range(OBS):
            if not _finite(Float64(r[0][i])):
                raise Error(
                    String(name) + ": non-finite obs[" + String(i)
                    + "] @step " + String(t)
                )
        if not _finite(Float64(r[1])):
            raise Error(String(name) + ": non-finite reward @step " + String(t))
        total_r += Float64(r[1])
        if r[2]:
            _ = env.reset_obs_list()  # mid-episode reset must stay finite too
            resets += 1
    _ = env.reset_obs_list()  # clean terminal reset must not crash
    print(
        "  ", name, ": OK  steps=", N_STEPS, " sum_r=", total_r,
        " mid-resets=", resets,
    )


def main() raises:
    print("--- Phyics3dEnv facade round-trip smoke (5 MuJoCo envs) ---")
    var ctx = DeviceContext()

    comptime HC = Phyics3dEnv[
        HalfCheetahModel, HalfCheetahConfig, DT, TERMINATE_ON_UNHEALTHY=False
    ]
    var hc = HC(ctx)
    _smoke[HC, HC.OBS_DIM](hc, "HalfCheetah")

    comptime AN = Phyics3dEnv[
        AntModel, AntConfig, DT, TERMINATE_ON_UNHEALTHY=True
    ]
    var an = AN(ctx)
    _smoke[AN, AN.OBS_DIM](an, "Ant")

    comptime WK = Phyics3dEnv[
        Walker2dModel, Walker2dConfig, DT, TERMINATE_ON_UNHEALTHY=True
    ]
    var wk = WK(ctx)
    _smoke[WK, WK.OBS_DIM](wk, "Walker2d")

    comptime IDP = Phyics3dEnv[
        InvertedDoublePendulumModel,
        InvertedDoublePendulumConfig,
        DT,
        TERMINATE_ON_UNHEALTHY=True,
    ]
    var idp = IDP(ctx)
    _smoke[IDP, IDP.OBS_DIM](idp, "InvertedDoublePendulum")

    comptime RE = Phyics3dEnv[
        ReacherModel, ReacherConfig, DT, TERMINATE_ON_UNHEALTHY=False
    ]
    var re = RE(ctx)
    _smoke[RE, RE.OBS_DIM](re, "Reacher")

    # Swimmer: fluid drag active (density=4000, viscosity=0.1) — exercises the
    # Stage-A fluid path through the facade.
    comptime SW = Phyics3dEnv[
        SwimmerModel, SwimmerConfig, DT, TERMINATE_ON_UNHEALTHY=False
    ]
    var sw = SW(ctx)
    _smoke[SW, SW.OBS_DIM](sw, "Swimmer")

    # Humanoid: tendons (max_tendon=2) + sites, threaded through the fields
    # integrator/solver.
    comptime HU = Phyics3dEnv[
        HumanoidModel, HumanoidConfig, DT, TERMINATE_ON_UNHEALTHY=True
    ]
    var hu = HU(ctx)
    _smoke[HU, HU.OBS_DIM](hu, "Humanoid")

    print("test_env_fields_mujoco_roundtrip: ALL PASS")
