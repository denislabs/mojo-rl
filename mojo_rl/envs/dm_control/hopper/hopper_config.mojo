"""dm_control `hopper` task configs — port of `suite/hopper.py` (`Hopper`).

One parameterized config covers both registered tasks:

    stand = DMHopperConfig[HOPPING=False]
    hop   = DMHopperConfig[HOPPING=True]

    observation = [qpos[1:] (6), qvel (7), log1p(touch) (2)]           (15)
    reward      = standing * small_control          (stand)
                  standing * hopping                (hop)
                  standing = tolerance(torso_z - foot_z, (0.6, 2))  [margin 0]
    reset       = randomize_limited_and_rotational_joints
    episode     = 1000 control steps (20 s / 0.02 s), no early termination

`standing` has NO margin, so it is a hard indicator: the reward is exactly 0
until the torso is 0.6 m above the foot. Both tasks therefore start at a flat
zero return, as with pendulum and cartpole-sparse.

The touch terms are the first use of `physics3d.sensors.touch`; see that module
for the zone semantics and for why contacts must be read post-solve.
"""

from std.random import random_float64
from std.math import pi, log, inf
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.physics3d.fields import Data
from mojo_rl.physics3d.joint_types import JNT_HINGE, JNT_SLIDE
from mojo_rl.physics3d.sensors.subtree import subtree_linvel
from mojo_rl.physics3d.sensors.touch import touch_sphere_site
from mojo_rl.physics3d.gpu.constants import (
    METADATA_SIZE,
    MODEL_CURRICULUM_SIZE,
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
)

from .hopper_xml import (
    DMHopperModel,
    TORSO_BODY_IDX,
    FOOT_BODY_IDX,
    TOUCH_TOE_SITE_IDX,
    TOUCH_HEEL_SITE_IDX,
)

from ...phyics3d_env_config import Phyics3dEnvConfig
from ..rewards import tolerance, SIGMOID_QUADRATIC, SIGMOID_LINEAR


# `hopper.py`: minimal torso-over-foot height scoring 1, and the hopping speed
# above which the hop reward saturates.
comptime STAND_HEIGHT: Float64 = 0.6
comptime HOP_SPEED: Float64 = 2.0

# `CONTACT_IDX_FORCE_N` is already a FORCE, in the same units as
# `mj_contactForce`'s normal component — measured, not assumed: on a settling
# drop our raw values track MuJoCo's within the solver's own disagreement
# (274 vs 336, 174 vs 208), whereas a 1/timestep scale would be 200x out.
# No conversion, then.
comptime TOUCH_FORCE_SCALE: Float64 = 1.0


struct DMHopperConfig[HOPPING: Bool](Phyics3dEnvConfig):
    # === Physics ===
    # `_CONTROL_TIMESTEP = .02` over the model's 0.005 s step => 4 substeps.
    comptime FRAME_SKIP: Int = 4
    # _DEFAULT_TIME_LIMIT = 20 s / 0.02 s = 1000 steps.
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    comptime SYNC_FK_AFTER_STEP: Bool = True
    # `<option timestep="0.005"/>` names no integrator => MuJoCo's Euler.
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
        act: List[Scalar[DTYPE]],
        mut obs: List[Scalar[DTYPE]],
    ) -> Bool:
        """`Hopper.get_observation`: position, velocity, touch.

        Position drops qpos[0] (rootx) for translational invariance, exactly as
        the reference comments say.
        """
        for i in range(1, NQ):
            obs.append(d.qpos.data[i])
        for i in range(NV):
            obs.append(d.qvel.data[i])

        # `np.log1p(sensordata[['touch_toe', 'touch_heel']])`.
        try:
            var toe = touch_sphere_site(
                d, m_sites, TOUCH_TOE_SITE_IDX, TOUCH_FORCE_SCALE
            )
            var heel = touch_sphere_site(
                d, m_sites, TOUCH_HEEL_SITE_IDX, TOUCH_FORCE_SCALE
            )
            obs.append(Scalar[DTYPE](log(1.0 + toe)))
            obs.append(Scalar[DTYPE](log(1.0 + heel)))
        except:
            # touch_sphere_site only raises on a non-sphere site, which is a
            # model-authoring error rather than a runtime condition. Keep the
            # observation the right LENGTH so a mis-edited model fails as an
            # obviously-dead sensor rather than a shape mismatch.
            obs.append(Scalar[DTYPE](0))
            obs.append(Scalar[DTYPE](0))
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
        the walker and point_mass configs'."""
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
        # `Physics.height`: xipos['torso','z'] - xipos['foot','z'].
        var height = Float64(d.xipos.data[TORSO_BODY_IDX * 3 + 2]) - Float64(
            d.xipos.data[FOOT_BODY_IDX * 3 + 2]
        )
        # margin defaults to 0 => hard indicator.
        var standing = tolerance(height, STAND_HEIGHT, 2.0, 0.0)

        comptime if Self.HOPPING:
            # `physics.speed()` = sensordata['torso_subtreelinvel'][0].
            var vx = Float64(0)
            var vy = Float64(0)
            var vz = Float64(0)
            subtree_linvel(
                d.xvel.data, m_bodies, NBODY, TORSO_BODY_IDX, vx, vy, vz
            )
            var hopping = tolerance[SIGMOID_LINEAR, 0.5](
                Float64(vx), HOP_SPEED, inf[DType.float64](), HOP_SPEED / 2.0
            )
            return (Scalar[DTYPE](standing * hopping), False)
        else:
            # small_control = (mean_i tolerance(ctrl_i, margin=1,
            #                    value_at_margin=0, quadratic) + 4) / 5
            var acc = 0.0
            comptime nact = DMHopperModel.nact
            for a in range(nact):
                var c = actions[a] if a < len(actions) else 0.0
                if c > 1.0:
                    c = 1.0
                elif c < -1.0:
                    c = -1.0
                acc += tolerance[SIGMOID_QUADRATIC, 0.0](c, 0.0, 0.0, 1.0)
            var small_control = (acc / Float64(nact) + 4.0) / 5.0
            return (Scalar[DTYPE](standing * small_control), False)

    # === CPU: Float getters ===
    @staticmethod
    def get_timestep() -> Float64:
        return Float64(DMHopperModel.TIMESTEP)
