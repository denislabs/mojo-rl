"""dm_control `point_mass-easy` task config — port of `suite/point_mass.py`.

    observation = [qpos(2), qvel(2)]                                     (4)
    reward      = near_target * small_control
                  near_target   = tolerance(||target - mass||,
                                            bounds=(0, .015), margin=.015)
                  small_control = (mean_i tolerance(ctrl_i, margin=1,
                                       value_at_margin=0, quadratic) + 4) / 5
    reset       = randomize_limited_and_rotational_joints
    episode     = 1000 control steps (20 s / 0.02 s), no early termination

Only `easy` is ported. `hard` randomizes the tendon mixing matrix per episode
(`model.wrap_prm`), which our engine cannot express — see `point_mass_xml` for
why the tendons are written as joint motors here.

The reward is far sharper than "smooth" suggests: `margin` is the 1.5 cm target
radius, so a mass 10 cm away scores ~1e-245, not "a bit less than 1". An
untrained policy therefore returns a flat zero for a long time — that is the
reference's behaviour, gated to 2.7e-11 against dm_control's own
`rewards.tolerance`, not a broken env.
"""

from std.random import random_float64
from std.math import pi, sqrt
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.physics3d.fields import Data
from mojo_rl.physics3d.kinematics.geom_xpos import geom_xpos
from mojo_rl.physics3d.joint_types import JNT_HINGE, JNT_SLIDE
from mojo_rl.physics3d.gpu.constants import (
    METADATA_SIZE,
    MODEL_CURRICULUM_SIZE,
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
)

from .point_mass_xml import (
    DMPointMassModel,
    POINTMASS_GEOM_IDX,
    TARGET_GEOM_IDX,
    TARGET_SIZE,
)

from ...phyics3d_env_config import Phyics3dEnvConfig
from ..rewards import tolerance, SIGMOID_QUADRATIC


struct DMPointMassConfig(Phyics3dEnvConfig):
    # === Physics ===
    # point_mass.py passes no control_timestep, so one env step is one
    # physics step of 0.02 s.
    comptime FRAME_SKIP: Int = 1
    # _DEFAULT_TIME_LIMIT = 20 s / 0.02 s = 1000 steps.
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    comptime SYNC_FK_AFTER_STEP: Bool = True
    # `<option timestep="0.02">` carries no `integrator`, so MuJoCo's Euler
    # default applies.
    comptime INTEGRATOR: StaticString = "euler"

    # === CPU: Observation ===
    @staticmethod
    def custom_extract_obs_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        mut obs: List[Scalar[DTYPE]],
    ) -> Bool:
        """`PointMass.get_observation`: position then velocity, both whole."""
        for i in range(NQ):
            obs.append(d.qpos.data[i])
        for i in range(NV):
            obs.append(d.qvel.data[i])
        return True

    # === CPU: Reset ===
    @staticmethod
    def custom_reset_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        mut d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
    ):
        """`randomizers.randomize_limited_and_rotational_joints` — identical to
        the walker config's; both of point_mass's joints are limited slides, so
        only the first branch is ever taken here."""
        var njoint = len(m_joints) // MODEL_JOINT_SIZE
        for j in range(njoint):
            var jtype = Int(m_joints[j * MODEL_JOINT_SIZE + JOINT_IDX_TYPE])
            if jtype != JNT_HINGE and jtype != JNT_SLIDE:
                continue
            var adr = Int(m_joints[j * MODEL_JOINT_SIZE + JOINT_IDX_QPOS_ADR])
            var lo = Float64(
                m_joints[j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MIN]
            )
            var hi = Float64(
                m_joints[j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MAX]
            )
            var limited = lo > -1e9 and hi < 1e9
            if limited:
                d.qpos.data[adr] = Scalar[DTYPE](
                    lo + random_float64() * (hi - lo)
                )
            elif jtype == JNT_HINGE:
                d.qpos.data[adr] = Scalar[DTYPE](
                    -pi + random_float64() * 2.0 * pi
                )

    # === CPU: Reward ===
    @staticmethod
    def compute_reward_and_done_cpu[
        DTYPE: DType,
        NQ: Int,
        NV: Int,
        NBODY: Int,
        MAX_CONTACTS: Int,
        NSITE: Int = 0,
    ](
        d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
        m_bodies: List[Scalar[DTYPE]],
        m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]],
        m_sites: List[Scalar[DTYPE]],
        prev_x: Scalar[DTYPE],
        actions: List[Float64],
        step_count: Int,
        frame_skip: Int,
    ) -> Tuple[Scalar[DTYPE], Bool]:
        # `Physics.mass_to_target_dist`.
        var tp = geom_xpos(d, m_geoms, TARGET_GEOM_IDX)
        var mp = geom_xpos(d, m_geoms, POINTMASS_GEOM_IDX)
        var dx = tp[0] - mp[0]
        var dy = tp[1] - mp[1]
        var dz = tp[2] - mp[2]
        var dist = sqrt(dx * dx + dy * dy + dz * dz)
        var near_target = tolerance(dist, 0.0, TARGET_SIZE, TARGET_SIZE)

        # `tolerance(physics.control(), margin=1, value_at_margin=0,
        #            sigmoid='quadratic').mean()`. `physics.control()` is
        # mjData.ctrl, which MuJoCo has already clamped to ctrlrange.
        var acc = 0.0
        comptime nact = DMPointMassModel.nact
        for a in range(nact):
            var c = actions[a] if a < len(actions) else 0.0
            if c > 1.0:
                c = 1.0
            elif c < -1.0:
                c = -1.0
            acc += tolerance[SIGMOID_QUADRATIC, 0.0](c, 0.0, 0.0, 1.0)
        var small_control = (acc / Float64(nact) + 4.0) / 5.0

        # dm_control tasks never terminate early.
        return (Scalar[DTYPE](near_target * small_control), False)

    # === CPU: Float getters ===
    @staticmethod
    def get_timestep() -> Float64:
        return Float64(DMPointMassModel.TIMESTEP)
