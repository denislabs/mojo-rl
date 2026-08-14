"""`dm_control` `finger` task configs — port of `suite/finger.py`.

Two task shapes over one model:

    spin      = DMFingerSpinConfig                       (obs 9)
    turn_easy = DMFingerTurnConfig[TARGET_SIZE=0.07]     (obs 12)
    turn_hard = DMFingerTurnConfig[TARGET_SIZE=0.03]     (obs 12)

    position = [jointpos proximal, jointpos distal, tip_position (x, z)]
    velocity = [jointvel proximal, jointvel distal, jointvel hinge]
    touch    = log1p(touch sensors at touchtop, touchbottom)
    turn adds target_position (x, z) and dist_to_target (1)

    tip_position    = framepos('tip')    - framepos('spinner')   [x, z]
    target_position = framepos('target') - framepos('spinner')   [x, z]
    dist_to_target  = ||target_position - tip_position|| - target_radius

    reward (spin) = float(hinge_velocity <= -15.0)
    reward (turn) = float(dist_to_target <= 0)
    episode       = 1000 control steps (20 s / 0.02 s), no early termination

Both rewards are hard indicators — exactly 0 or exactly 1, no `tolerance`
margin anywhere in this domain.

SENSORS. dm_control routes every finger observation through `<sensor>` so the
whole thing is finite-differenceable, but each one we need is a direct read:
`jointpos` -> `d.qpos[adr]`, `jointvel` -> `d.qvel[adr]`, `framepos` of a site
-> `d.site_xpos`, of an xbody -> `d.xpos`. Only `touch` needs real work, and
that already landed for hopper.

⚠ THE TOUCH ZONES ARE ELLIPSOIDS TREATED AS SPHERES. `childclass="finger"`
types `touchtop`/`touchbottom` as `ellipsoid size=".025 .03 .025"`, and
`_geom_type_from_str` has no `ellipsoid` case — it falls through to
`_GEOM_SPHERE` SILENTLY, so `touch_sphere_site` measures a sphere of radius
size[0] rather than raising. Exact here only because the x and z semi-axes are
equal and the model is planar in x-z; `test_finger_vs_dm_control` pins both
facts so this cannot rot unnoticed.

⚠ `Spin.initialize_episode` writes `dof_damping['hinge'] = .03`, down from the
XML's .5. That is a real dynamics change, not cosmetics, and our `fields.Model`
is shared and unbatched so a config cannot write it per episode. The spin model
therefore compiles from its OWN XML with the .03 already substituted — see
`finger_xml.dm_finger_spin_xml`.
"""

from std.random import random_float64
from std.math import pi, sqrt, sin, cos
from ..dtype_math import log1p_dt

from mojo_rl.physics3d.fields import Data
from mojo_rl.physics3d.sensors.touch import touch_sphere_site
from mojo_rl.physics3d.joint_types import JNT_HINGE, JNT_SLIDE
from mojo_rl.physics3d.gpu.constants import (
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
)

from .finger_xml import (
    DMFingerSpinModel,
    DMFingerTurnModel,
    SPINNER_BODY_IDX,
    TARGET_BODY_IDX,
    TIP_SITE_IDX,
    TARGET_SITE_IDX,
    TOUCHTOP_SITE_IDX,
    TOUCHBOTTOM_SITE_IDX,
    PROXIMAL_ADR,
    DISTAL_ADR,
    HINGE_ADR,
    SPINNER_RADIUS,
    TARGET_Z,
)

from ...phyics3d_env_config import Phyics3dEnvConfig


# `float(physics.hinge_velocity() <= -_SPIN_VELOCITY)`.
comptime SPIN_VELOCITY: Float64 = 15.0


def _tip_position[
    DTYPE: DType, NQ: Int, NV: Int, NBODY: Int, MAX_CONTACTS: Int, NSITE: Int
](
    d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
) -> Tuple[Float64, Float64]:
    """`Physics.tip_position` — (x, z) of the tip site RELATIVE to the spinner.

    Both sides come from `framepos` sensors in the reference: the tip is a
    SITE, the spinner an XBODY, so they read from different arrays here.
    """
    var tx = Float64(d.site_xpos.data[TIP_SITE_IDX * 3 + 0])
    var tz = Float64(d.site_xpos.data[TIP_SITE_IDX * 3 + 2])
    var sx = Float64(d.xpos.data[SPINNER_BODY_IDX * 3 + 0])
    var sz = Float64(d.xpos.data[SPINNER_BODY_IDX * 3 + 2])
    return (tx - sx, tz - sz)


def _target_position[
    DTYPE: DType, NQ: Int, NV: Int, NBODY: Int, MAX_CONTACTS: Int, NSITE: Int
](
    d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
) -> Tuple[Float64, Float64]:
    """`Physics.target_position` — (x, z) of the target site vs the spinner."""
    var tx = Float64(d.site_xpos.data[TARGET_SITE_IDX * 3 + 0])
    var tz = Float64(d.site_xpos.data[TARGET_SITE_IDX * 3 + 2])
    var sx = Float64(d.xpos.data[SPINNER_BODY_IDX * 3 + 0])
    var sz = Float64(d.xpos.data[SPINNER_BODY_IDX * 3 + 2])
    return (tx - sx, tz - sz)


def _append_shared_obs[
    DTYPE: DType, NQ: Int, NV: Int, NBODY: Int, MAX_CONTACTS: Int, NSITE: Int
](
    d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
    m_sites: List[Scalar[DTYPE]],
    mut obs: List[Scalar[DTYPE]],
) raises:
    """position (4) + velocity (3) + touch (2) — common to spin and turn."""
    # `bounded_position`: the two finger joints, then the tip position, which
    # REPLACES the hinge angle (an unbounded, wrapping quantity).
    obs.append(d.qpos.data[PROXIMAL_ADR])
    obs.append(d.qpos.data[DISTAL_ADR])
    var tip = _tip_position(d)
    obs.append(Scalar[DTYPE](tip[0]))
    obs.append(Scalar[DTYPE](tip[1]))

    obs.append(d.qvel.data[PROXIMAL_ADR])
    obs.append(d.qvel.data[DISTAL_ADR])
    obs.append(d.qvel.data[HINGE_ADR])

    # `np.log1p(sensordata[['touchtop', 'touchbottom']])`. Reading the contact
    # force requires POST-SOLVE contacts, which the facade guarantees by
    # extracting obs after the integrator step.
    var top = touch_sphere_site(d, m_sites, TOUCHTOP_SITE_IDX, 1.0)
    var bot = touch_sphere_site(d, m_sites, TOUCHBOTTOM_SITE_IDX, 1.0)
    obs.append(Scalar[DTYPE](log1p_dt[DTYPE](Scalar[DTYPE](top))))
    obs.append(Scalar[DTYPE](log1p_dt[DTYPE](Scalar[DTYPE](bot))))


def _randomize_joints[
    DTYPE: DType, NQ: Int, NV: Int, NBODY: Int, MAX_CONTACTS: Int, NSITE: Int
](
    mut d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
    m_joints: List[Scalar[DTYPE]],
):
    """`randomizers.randomize_limited_and_rotational_joints`.

    ⚠ The reference then REJECTION-SAMPLES on `physics.data.ncon == 0` (up to
    1000 attempts), which we cannot reproduce inside a reset hook: collision
    detection has not run at this point, so `d.contacts` is stale. The initial
    STATE DISTRIBUTION therefore differs from the reference's whenever a draw
    self-intersects — the dynamics from any given state are unaffected. Same
    limitation ball_in_cup's reset will hit; see docs/DM_CONTROL_PORT.md.
    """
    var njoint = len(m_joints) // MODEL_JOINT_SIZE
    for j in range(njoint):
        var jbase = j * MODEL_JOINT_SIZE
        var jtype = Int(m_joints[jbase + JOINT_IDX_TYPE])
        if jtype != JNT_HINGE and jtype != JNT_SLIDE:
            continue
        var adr = Int(m_joints[jbase + JOINT_IDX_QPOS_ADR])
        var lo = Float64(m_joints[jbase + JOINT_IDX_RANGE_MIN])
        var hi = Float64(m_joints[jbase + JOINT_IDX_RANGE_MAX])
        if lo > -1e9 and hi < 1e9:
            d.qpos.data[adr] = Scalar[DTYPE](lo + random_float64() * (hi - lo))
        elif jtype == JNT_HINGE:
            # Unlimited hinge (`hinge`) -> uniform on the full circle.
            d.qpos.data[adr] = Scalar[DTYPE](-pi + random_float64() * 2.0 * pi)


struct DMFingerSpinConfig(Phyics3dEnvConfig):
    """`Spin`: reward the spinner turning fast in the negative direction."""

    # === Physics ===
    # _CONTROL_TIMESTEP .02 over a .01 physics step => 2 substeps.
    comptime FRAME_SKIP: Int = 2
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    comptime SYNC_FK_AFTER_STEP: Bool = True
    comptime INTEGRATOR: StaticString = "euler"

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
        """`Spin.get_observation`: position, velocity, touch."""
        try:
            _append_shared_obs(d, m_sites, obs)
        except:
            return False
        return True

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
        """`Spin.initialize_episode` — joints only.

        The hinge damping write (.5 -> .03) is baked into the spin XML, and
        the two `site_rgba` writes are visual-only.
        """
        _randomize_joints(d, m_joints)

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
        # `float(hinge_velocity <= -_SPIN_VELOCITY)` — sign matters, only one
        # spin direction is rewarded.
        var hv = Float64(d.qvel.data[HINGE_ADR])
        var r = 1.0 if hv <= -SPIN_VELOCITY else 0.0
        return (Scalar[DTYPE](r), False)

    @staticmethod
    def get_timestep() -> Float64:
        return Float64(DMFingerSpinModel.TIMESTEP)


struct DMFingerTurnConfig[TARGET_SIZE: Float64](Phyics3dEnvConfig):
    """`Turn`: reward the tip reaching a target angle on the spinner rim."""

    comptime FRAME_SKIP: Int = 2
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    comptime SYNC_FK_AFTER_STEP: Bool = True
    comptime INTEGRATOR: StaticString = "euler"

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
        """`Turn.get_observation`: + target_position, dist_to_target."""
        try:
            _append_shared_obs(d, m_sites, obs)
        except:
            return False
        var tgt = _target_position(d)
        obs.append(Scalar[DTYPE](tgt[0]))
        obs.append(Scalar[DTYPE](tgt[1]))
        var tip = _tip_position(d)
        var dx = tgt[0] - tip[0]
        var dz = tgt[1] - tip[1]
        obs.append(
            Scalar[DTYPE](sqrt(dx * dx + dz * dz) - Self.TARGET_SIZE)
        )
        return True

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
        """`Turn.initialize_episode`: a target angle, then the joints.

        The reference places the target at
        `hinge_xanchor + radius * (sin a, cos a)` in (x, z). `hinge` sits at
        the spinner body origin with no `pos` offset, so its anchor IS the
        spinner body position — read from `m_bodies` is not needed, the mocap
        write is absolute and the spinner never moves (it only rotates).
        """
        var angle = -pi + random_float64() * 2.0 * pi
        # Spinner body origin = the hinge anchor, `pos=".2 0 .4"`.
        var hinge_x = 0.2
        var hinge_z = TARGET_Z
        d.mocap_pos.data[TARGET_BODY_IDX * 3 + 0] = Scalar[DTYPE](
            hinge_x + SPINNER_RADIUS * sin(angle)
        )
        d.mocap_pos.data[TARGET_BODY_IDX * 3 + 1] = Scalar[DTYPE](0)
        d.mocap_pos.data[TARGET_BODY_IDX * 3 + 2] = Scalar[DTYPE](
            hinge_z + SPINNER_RADIUS * cos(angle)
        )
        d.mocap_quat.data[TARGET_BODY_IDX * 4 + 0] = Scalar[DTYPE](0)
        d.mocap_quat.data[TARGET_BODY_IDX * 4 + 1] = Scalar[DTYPE](0)
        d.mocap_quat.data[TARGET_BODY_IDX * 4 + 2] = Scalar[DTYPE](0)
        d.mocap_quat.data[TARGET_BODY_IDX * 4 + 3] = Scalar[DTYPE](1)

        _randomize_joints(d, m_joints)

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
        # `float(dist_to_target <= 0)` — the tip inside the target disc.
        var tgt = _target_position(d)
        var tip = _tip_position(d)
        var dx = tgt[0] - tip[0]
        var dz = tgt[1] - tip[1]
        var dist = sqrt(dx * dx + dz * dz) - Self.TARGET_SIZE
        return (Scalar[DTYPE](1.0 if dist <= 0.0 else 0.0), False)

    @staticmethod
    def get_timestep() -> Float64:
        return Float64(DMFingerTurnModel.TIMESTEP)
