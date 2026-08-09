"""`dm_control` `dog` task config — port of `suite/dog.py`'s `Stand` and `Move`.

    stand = DMDogStandConfig
    walk  = DMDogMoveConfig[MOVE_SPEED=1.0]
    trot  = DMDogMoveConfig[MOVE_SPEED=3.0]
    run   = DMDogMoveConfig[MOVE_SPEED=9.0]

    observation = [joint_angles(73), joint_velocites(73),
                   torso_pelvis_height(2), z_projection(9),
                   torso_com_velocity(3), inertial_sensors(9),
                   foot_forces(12), touch_sensors(4),
                   actuator_state(38)]                              (223)
    reward      = prod(torso, pelvis, upright x3, touch)   [x forward, Move]
    reset       = qpos0, then random yaw, random planar/yaw root velocity,
                  and a random activation for EVERY actuator
    episode     = 1000 control steps (15 s / 0.015 s), no early termination

`Move` SUBCLASSES `Stand` and its `get_reward_factors` calls `super()`, so the
Move reward is the six Stand factors TIMES a seventh. It is not a different
reward with a speed term bolted on, and the six-factor product must be
identical between them — `DMDogMoveConfig` therefore calls the same
`_stand_factors` helper rather than re-deriving it.

THREE THINGS HERE ARE EASY TO GET WRONG AND STILL LOOK RIGHT

1. **`upright` is THREE factors, not one.** `physics.upright()` returns
   `z_projection()[:, 2]`, a 3-vector over skull / torso / pelvis, and
   `rewards.tolerance` broadcasts over it. `np.prod` of the `hstack` then
   multiplies all three in. Collapsing it to the torso alone gives a reward
   that is too generous by two factors and is otherwise indistinguishable.

2. **`torso_com_velocity` is a WORLD→BODY rotation.** The reference is
   `self.center_of_mass_velocity().dot(torso_frame)` with
   `torso_frame = xmat['torso'].reshape(3, 3)`. A numpy 1-D `.dot(M)` is a ROW
   vector times a matrix, i.e. `Mᵀ v` — the transpose of the obvious reading.
   Writing `R v` yields three numbers of the right magnitude and units, and
   `com_forward_velocity()` (which is just `[0]` of this) then drives the whole
   Move reward off the wrong axis.

3. **`_stand_height` and `_body_weight` are model constants**, despite being
   measured in `initialize_episode` — see the note at `DOG_STAND_HEIGHT_TORSO`
   in `dog_xml.mojo`. `initialize_episode` measures them AFTER `physics.reset()`
   and BEFORE the randomization, so no episode ever sees a different value.

⚠ THE RESET RANDOMIZES `data.act`, which no earlier ported domain does.
`for actuator_id in range(nu): act[id] = uniform(*ctrlrange)`. Since every dog
actuator is `dyntype="filter"` and its force is `gainprm[0] * act`, this starts
each episode with 38 non-zero torques already applied — a reset that left `act`
at zero would be a materially easier task, and no parity gate that sets qpos
directly would ever notice (reset infidelities are invisible to all of them).
"""

from std.math import log, sqrt, cos, sin, pi, inf
from std.random import random_float64
from std.collections import InlineArray

from mojo_rl.physics3d.fields import Data
from mojo_rl.physics3d.kinematics.xmat import xmat_elem, XMAT_ZX, XMAT_ZY, XMAT_ZZ
from mojo_rl.physics3d.sensors.frame_vel import site_frame_velocity
from mojo_rl.physics3d.sensors.site_acc import (
    site_accelerometer,
    site_force_torque,
)
from mojo_rl.physics3d.sensors.subtree import subtree_linvel
from mojo_rl.physics3d.sensors.touch import touch_sphere_site

from ..rewards import tolerance, SIGMOID_LINEAR
from .dog_xml import (
    dsp,
    DOG_OBS_DIM,
    DOG_FRAME_SKIP,
    DOG_MAX_STEPS,
    DOG_N_HINGE,
    DOG_HINGE_QPOS_0,
    DOG_HINGE_DOF_0,
    DOG_TORSO_BODY_IDX,
    DOG_PELVIS_BODY_IDX,
    DOG_SKULL_BODY_IDX,
    DOG_SITE_HEAD,
    DOG_SITE_PALM_L,
    DOG_SITE_PALM_R,
    DOG_SITE_SOLE_L,
    DOG_SITE_SOLE_R,
    DOG_SITE_FOOT_ANCHOR_L,
    DOG_SITE_FOOT_ANCHOR_R,
    DOG_SITE_HAND_ANCHOR_L,
    DOG_SITE_HAND_ANCHOR_R,
    DOG_BODY_FOOT_ANCHOR_L,
    DOG_BODY_FOOT_ANCHOR_R,
    DOG_BODY_HAND_ANCHOR_L,
    DOG_BODY_HAND_ANCHOR_R,
    DOG_MIN_UPRIGHT_COSINE,
    DOG_STAND_HEIGHT_TORSO,
    DOG_STAND_HEIGHT_PELVIS,
    DOG_BODY_WEIGHT,
)
from ...phyics3d_env_config import Phyics3dEnvConfig


@always_inline
def _randn_pair() -> Tuple[Float64, Float64]:
    """Two independent standard normals (Box-Muller), as `np.random.randn`."""
    var u1 = random_float64()
    if u1 < 1e-300:
        u1 = 1e-300
    var u2 = random_float64()
    var r = sqrt(-2.0 * log(u1))
    return (r * cos(2.0 * pi * u2), r * sin(2.0 * pi * u2))


@always_inline
def _world_to_torso[
    DTYPE: DType, NQ: Int, NV: Int, NBODY: Int, MAX_CONTACTS: Int, NSITE: Int
](
    d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
    vx: Float64,
    vy: Float64,
    vz: Float64,
) -> Tuple[Float64, Float64, Float64]:
    """`v.dot(xmat['torso'])` — a ROW vector times R, which is `Rᵀ v`.

    Spelled out longhand rather than through a matrix helper precisely because
    the transpose is the trap: component k of the result is the dot product of
    v with COLUMN k of R, i.e. with R's k-th basis vector expressed in world
    coordinates. See note 2 in the module docstring.
    """
    var r = InlineArray[Float64, 9](fill=0.0)
    for k in range(9):
        r[k] = xmat_elem(d, DOG_TORSO_BODY_IDX, k)
    return (
        vx * r[0] + vy * r[3] + vz * r[6],
        vx * r[1] + vy * r[4] + vz * r[7],
        vx * r[2] + vy * r[5] + vz * r[8],
    )


@always_inline
def _com_velocity_torso_frame[
    DTYPE: DType, NQ: Int, NV: Int, NBODY: Int, MAX_CONTACTS: Int, NSITE: Int
](
    d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
    m_bodies: List[Scalar[DTYPE]],
    nbody: Int,
) -> Tuple[Float64, Float64, Float64]:
    """`torso_com_velocity()` = `subtreelinvel('torso')` rotated into torso."""
    var vx = Float64(0)
    var vy = Float64(0)
    var vz = Float64(0)
    subtree_linvel[DTYPE](
        d.xvel.data, m_bodies, nbody, DOG_TORSO_BODY_IDX, vx, vy, vz
    )
    return _world_to_torso[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE](
        d, vx, vy, vz
    )


@always_inline
def _touch_sum[
    DTYPE: DType, NQ: Int, NV: Int, NBODY: Int, MAX_CONTACTS: Int, NSITE: Int
](
    d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
    m_sites: List[Scalar[DTYPE]],
) raises -> Float64:
    """`touch_sensors().sum()` — palm_L + palm_R + sole_L + sole_R.

    All four sites are `type="box"`, which the touch sensor measures exactly;
    it is the ELLIPSOID case that is approximated.
    """
    return (
        touch_sphere_site[DTYPE](d, m_sites, DOG_SITE_PALM_L, 1.0)
        + touch_sphere_site[DTYPE](d, m_sites, DOG_SITE_PALM_R, 1.0)
        + touch_sphere_site[DTYPE](d, m_sites, DOG_SITE_SOLE_L, 1.0)
        + touch_sphere_site[DTYPE](d, m_sites, DOG_SITE_SOLE_R, 1.0)
    )


@always_inline
def _stand_factors[
    DTYPE: DType, NQ: Int, NV: Int, NBODY: Int, MAX_CONTACTS: Int, NSITE: Int
](
    d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
    m_bodies: List[Scalar[DTYPE]],
    m_sites: List[Scalar[DTYPE]],
) raises -> Float64:
    """`Stand.get_reward_factors`, multiplied out — SIX factors.

        torso    tolerance(h_torso,  (sh_torso,  inf), margin=sh_torso)
        pelvis   tolerance(h_pelvis, (sh_pelvis, inf), margin=sh_pelvis)
        upright  tolerance(zz, (min_cos, inf), linear, margin=min_cos+1,
                           value_at_margin=0)   -- ONCE PER BODY, three times
        touch    tolerance(sum, (weight, inf), linear, margin=weight,
                           value_at_margin=0.9)

    `torso` and `pelvis` take the DEFAULT gaussian sigmoid and the default
    `value_at_margin` of 0.1; only the last two override them.
    """
    var h_t = Float64(d.xpos.data[DOG_TORSO_BODY_IDX * 3 + 2])
    var h_p = Float64(d.xpos.data[DOG_PELVIS_BODY_IDX * 3 + 2])

    var f_torso = tolerance(
        h_t, DOG_STAND_HEIGHT_TORSO, inf[DType.float64](),
        DOG_STAND_HEIGHT_TORSO,
    )
    var f_pelvis = tolerance(
        h_p, DOG_STAND_HEIGHT_PELVIS, inf[DType.float64](),
        DOG_STAND_HEIGHT_PELVIS,
    )

    # `upright()` is z_projection()[:, 2] over skull, torso, pelvis — the 'zz'
    # element of each body's xmat. Three separate reward factors.
    var f_upright = Float64(1.0)
    var uprights = InlineArray[Int, 3](fill=0)
    uprights[0] = DOG_SKULL_BODY_IDX
    uprights[1] = DOG_TORSO_BODY_IDX
    uprights[2] = DOG_PELVIS_BODY_IDX
    for k in range(3):
        var zz = xmat_elem(d, uprights[k], XMAT_ZZ)
        f_upright *= tolerance[SIGMOID_LINEAR, 0.0](
            zz, DOG_MIN_UPRIGHT_COSINE, inf[DType.float64](),
            DOG_MIN_UPRIGHT_COSINE + 1.0,
        )

    var f_touch = tolerance[SIGMOID_LINEAR, 0.9](
        _touch_sum[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE](d, m_sites),
        DOG_BODY_WEIGHT, inf[DType.float64](), DOG_BODY_WEIGHT,
    )

    return f_torso * f_pelvis * f_upright * f_touch


@always_inline
def _dog_obs_cpu[
    DTYPE: DType, NQ: Int, NV: Int, NBODY: Int, MAX_CONTACTS: Int, NSITE: Int
](
    d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1],
    m_bodies: List[Scalar[DTYPE]],
    m_sites: List[Scalar[DTYPE]],
    act: List[Scalar[DTYPE]],
    mut obs: List[Scalar[DTYPE]],
) -> Bool:
    """`Stand.get_observation_components` — nine blocks, in order, 223 numbers.

    `Move` and `Fetch` both inherit this; `Fetch` (Phase 5) appends two more
    blocks after it.
    """
    try:
        # --- joint_angles / joint_velocites: HINGES ONLY -------------------
        # `self.model.jnt_type == _HINGE_TYPE` skips the free root, so these
        # start at qpos 7 / dof 6 and run contiguously (asserted in the gate).
        for k in range(DOG_N_HINGE):
            obs.append(d.qpos.data[DOG_HINGE_QPOS_0 + k])
        for k in range(DOG_N_HINGE):
            obs.append(d.qvel.data[DOG_HINGE_DOF_0 + k])

        # --- torso_pelvis_height -------------------------------------------
        obs.append(d.xpos.data[DOG_TORSO_BODY_IDX * 3 + 2])
        obs.append(d.xpos.data[DOG_PELVIS_BODY_IDX * 3 + 2])

        # --- z_projection: xmat[[skull, torso, pelvis], ['zx','zy','zz']] --
        #     row-major indices 6, 7, 8 of each body's rotation matrix.
        var zbodies = InlineArray[Int, 3](fill=0)
        zbodies[0] = DOG_SKULL_BODY_IDX
        zbodies[1] = DOG_TORSO_BODY_IDX
        zbodies[2] = DOG_PELVIS_BODY_IDX
        for k in range(3):
            obs.append(Scalar[DTYPE](xmat_elem(d, zbodies[k], XMAT_ZX)))
            obs.append(Scalar[DTYPE](xmat_elem(d, zbodies[k], XMAT_ZY)))
            obs.append(Scalar[DTYPE](xmat_elem(d, zbodies[k], XMAT_ZZ)))

        # --- torso_com_velocity --------------------------------------------
        var cv = _com_velocity_torso_frame[
            DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE
        ](d, m_bodies, NBODY)
        obs.append(Scalar[DTYPE](cv[0]))
        obs.append(Scalar[DTYPE](cv[1]))
        obs.append(Scalar[DTYPE](cv[2]))

        # --- inertial_sensors: accelerometer, velocimeter, gyro ------------
        #     All three sit on the `head` site, and dm_control reads them BY
        #     NAME in that order — which happens to be declaration order too.
        # ⚠ `site_xpos_acc` / `xquat_acc`, NOT the live FK products —
        # defect 19. This sensor transports `cacc` to the site and rotates
        # into the site frame, so it needs the geometry FROM THE INSTANT
        # `cacc` was written. `d.site_xpos`/`d.xquat` have since been moved to
        # the post-integration state by `_fields_fk`, which the
        # position/velocity-stage dims below require and this one must not
        # see. Mixing them read 1.484 where dm_control reads -6.386.
        var acc = site_accelerometer[DTYPE](
            d.cvel.data, d.cacc.data, d.subtree_com.data,
            d.site_xpos_acc.data, d.xquat_acc.data, m_bodies, m_sites,
            DOG_SKULL_BODY_IDX, DOG_SITE_HEAD,
        )
        var fv = site_frame_velocity[DTYPE](
            d.xvel.data, d.xangvel.data, d.xipos.data, d.xquat.data,
            d.site_xpos.data, m_sites, DOG_SKULL_BODY_IDX, DOG_SITE_HEAD,
        )
        # ⚠ Unpacked, not looped: a Tuple subscript needs a COMPTIME index,
        # so `acc[k]` with a loop variable is "cannot use a dynamic value in
        # type parameter". `quadruped_config` hit the same thing.
        obs.append(Scalar[DTYPE](acc[0]))
        obs.append(Scalar[DTYPE](acc[1]))
        obs.append(Scalar[DTYPE](acc[2]))
        # velocimeter = linear half of the site frame velocity, gyro = angular.
        obs.append(Scalar[DTYPE](fv[0]))
        obs.append(Scalar[DTYPE](fv[1]))
        obs.append(Scalar[DTYPE](fv[2]))
        obs.append(Scalar[DTYPE](fv[3]))
        obs.append(Scalar[DTYPE](fv[4]))
        obs.append(Scalar[DTYPE](fv[5]))

        # --- foot_forces: <force> at foot_L, foot_R, hand_L, hand_R --------
        #     A `<force>` sensor is THREE numbers; `site_force_torque` returns
        #     six and the torque half belongs to a `<torque>` sensor dog does
        #     not declare.
        var f_bodies = InlineArray[Int, 4](fill=0)
        f_bodies[0] = DOG_BODY_FOOT_ANCHOR_L
        f_bodies[1] = DOG_BODY_FOOT_ANCHOR_R
        f_bodies[2] = DOG_BODY_HAND_ANCHOR_L
        f_bodies[3] = DOG_BODY_HAND_ANCHOR_R
        var f_sites = InlineArray[Int, 4](fill=0)
        f_sites[0] = DOG_SITE_FOOT_ANCHOR_L
        f_sites[1] = DOG_SITE_FOOT_ANCHOR_R
        f_sites[2] = DOG_SITE_HAND_ANCHOR_L
        f_sites[3] = DOG_SITE_HAND_ANCHOR_R
        for t in range(4):
            # Acceleration stage, same as the accelerometer above: the
            # snapshot, not the live FK products.
            var ft = site_force_torque[DTYPE](
                d.cfrc_int.data, d.subtree_com.data, d.site_xpos_acc.data,
                d.xquat_acc.data, m_bodies, m_sites, f_bodies[t], f_sites[t],
            )
            obs.append(Scalar[DTYPE](ft[0]))
            obs.append(Scalar[DTYPE](ft[1]))
            obs.append(Scalar[DTYPE](ft[2]))

        # --- touch_sensors: palm_L, palm_R, sole_L, sole_R -----------------
        var t_sites = InlineArray[Int, 4](fill=0)
        t_sites[0] = DOG_SITE_PALM_L
        t_sites[1] = DOG_SITE_PALM_R
        t_sites[2] = DOG_SITE_SOLE_L
        t_sites[3] = DOG_SITE_SOLE_R
        for t in range(4):
            obs.append(
                Scalar[DTYPE](touch_sphere_site[DTYPE](d, m_sites, t_sites[t], 1.0))
            )

        # --- actuator_state: data.act, all 38 ------------------------------
        for k in range(len(act)):
            obs.append(act[k])
    except:
        return False
    return True


@always_inline
def _dog_reset_cpu[
    DTYPE: DType, NQ: Int, NV: Int, NBODY: Int, MAX_CONTACTS: Int, NSITE: Int
](mut d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, 1]):
    """`Stand.initialize_episode`, minus the `data.act` draw.

        azimuth     = uniform(0, 2*pi)
        qpos['root'][3:] = (cos(a/2), 0, 0, sin(a/2))     -- YAW ONLY
        qvel[0], qvel[1], qvel[5] = 2 * randn()  each

    ⚠ THE ORIENTATION IS A YAW, NOT A UNIFORM QUATERNION. quadruped's `Move`
    draws a normalized 4-vector of normals (uniform on SO(3)); dog draws a
    single azimuth and builds the rotation about z. Reusing quadruped's draw
    here would start the dog upside-down a third of the time and make the
    upright factors — three of the six — mostly zero.

    The `act` draw needs the actuator ctrlrange and so lives in the config's
    own hook, which has the model tables.
    """
    var azimuth = 2.0 * pi * random_float64()
    d.qpos.data[3] = Scalar[DTYPE](cos(azimuth * 0.5))
    d.qpos.data[4] = Scalar[DTYPE](0)
    d.qpos.data[5] = Scalar[DTYPE](0)
    d.qpos.data[6] = Scalar[DTYPE](sin(azimuth * 0.5))

    var ab = _randn_pair()
    var c = _randn_pair()
    d.qvel.data[0] = Scalar[DTYPE](2.0 * ab[0])
    d.qvel.data[1] = Scalar[DTYPE](2.0 * ab[1])
    d.qvel.data[5] = Scalar[DTYPE](2.0 * c[0])


struct DMDogStandConfig(Phyics3dEnvConfig):
    """`Stand` — hold the default pose, upright, with the feet loaded."""

    comptime FRAME_SKIP: Int = DOG_FRAME_SKIP
    comptime MAX_STEPS: Int = DOG_MAX_STEPS
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    # dog.xml sets no `integrator`, so MuJoCo's default Euler applies.
    comptime INTEGRATOR: StaticString = "euler"
    # dm_control runs mj_step1 after the last substep.
    comptime SYNC_FK_AFTER_STEP: Bool = True
    # accelerometer + the four force sensors need `mj_rnePostConstraint`.
    comptime RNE_POST: Bool = True
    # dog's reset does NOT raise the body — it spawns at qpos0's height and
    # lets the solver sort out the contacts, so no height search.
    comptime RESET_FIND_HEIGHT: Bool = False

    @staticmethod
    def get_timestep() -> Float64:
        return Float64(dsp.TIMESTEP)

    @staticmethod
    def get_reset_noise() -> Float64:
        # The randomization is structured (yaw, three root velocities, and the
        # activations) rather than uniform jitter on every joint.
        return 0.0

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
        _dog_reset_cpu[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE](d)

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
        return _dog_obs_cpu[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE](
            d, m_bodies, m_sites, act, obs
        )

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
        """`np.prod(get_reward_factors())`; never terminates early."""
        var r: Float64
        try:
            r = _stand_factors[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE](
                d, m_bodies, m_sites
            )
        except:
            r = 0.0
        return (Scalar[DTYPE](r), False)


struct DMDogMoveConfig[MOVE_SPEED: Float64](Phyics3dEnvConfig):
    """`Move` — `Stand`'s six factors times a forward-speed seventh.

        walk 1.0, trot 3.0, run 9.0

        speed_margin = max(1.0, move_speed)
        forward = tolerance(com_forward_velocity,
                            bounds=(move_speed, 2*move_speed),
                            margin=speed_margin, value_at_margin=0,
                            sigmoid='linear')
        forward = (4*forward + 1) / 5

    ⚠ THE UPPER BOUND IS `2 * move_speed`, NOT INFINITY. Running faster than
    twice the target falls back off the plateau. Every other Move-style reward
    in this suite port is one-sided, so this is the one place the habit is
    wrong.
    """

    comptime FRAME_SKIP: Int = DOG_FRAME_SKIP
    comptime MAX_STEPS: Int = DOG_MAX_STEPS
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    comptime INTEGRATOR: StaticString = "euler"
    comptime SYNC_FK_AFTER_STEP: Bool = True
    comptime RNE_POST: Bool = True
    comptime RESET_FIND_HEIGHT: Bool = False

    @staticmethod
    def get_timestep() -> Float64:
        return Float64(dsp.TIMESTEP)

    @staticmethod
    def get_reset_noise() -> Float64:
        return 0.0

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
        """`Move` does not override `initialize_episode` — this is `Stand`'s."""
        _dog_reset_cpu[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE](d)

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
        """`Move` does not override `get_observation_components` either."""
        return _dog_obs_cpu[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE](
            d, m_bodies, m_sites, act, obs
        )

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
        var r: Float64
        try:
            var standing = _stand_factors[
                DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE
            ](d, m_bodies, m_sites)

            var cv = _com_velocity_torso_frame[
                DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE
            ](d, m_bodies, NBODY)

            comptime speed_margin = (
                1.0 if Self.MOVE_SPEED < 1.0 else Self.MOVE_SPEED
            )
            var fwd = tolerance[SIGMOID_LINEAR, 0.0](
                cv[0], Self.MOVE_SPEED, 2.0 * Self.MOVE_SPEED, speed_margin
            )
            fwd = (4.0 * fwd + 1.0) / 5.0
            r = standing * fwd
        except:
            r = 0.0
        return (Scalar[DTYPE](r), False)
