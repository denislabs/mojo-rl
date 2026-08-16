"""`dm_control` `quadruped` models — port of `dm_control/suite/quadruped.xml`.

The reference does NOT load that file as written. `quadruped.make_model()`
(suite/quadruped.py:55) parses it with lxml and DELETES elements per task:

    walk / run:  make_model(floor_size=_DEFAULT_TIME_LIMIT * speed)
                 -> terrain=False, rangefinders=False, walls_and_ball=False

so both tasks get the same stripped model, differing only in the floor plane's
half-extent (walk 20*0.5 = 10, run 20*5 = 100). What the strip removes:

  - the four `wall_*` plane geoms,
  - the `target` site,
  - the whole `ball` body (freejoint `ball_root`, `ball` geom, `ball_light`),
  - the `terrain` hfield geom,
  - all twenty `<rangefinder>` SENSORS.

Note what it does NOT remove, and which this module therefore keeps verbatim:
the twenty `rf_*` SITES (only the sensors go), the `terrain` hfield ASSET, and
the `ball` texture/material assets. None of the three touches the dynamics, but
dropping the sites would shift every site index off MuJoCo's — and the
`force`/`torque` sensors are addressed by site.

WHAT THIS MODEL NEEDS THAT THE OTHER PORTS DID NOT
--------------------------------------------------
1. `<general>` actuators with `dyntype="filter"`. Every one of the twelve is

       <general ctrllimited="true" gainprm="1000" biasprm="0 -1000"
                biastype="affine" dyntype="filter" dynprm=".1"/>

   i.e. a position servo (force = 1000*(act - length)) whose setpoint `act` is
   a first-order lag of `ctrl` with a 0.1 s time constant. That ACTIVATION
   STATE is a new piece of `Data` — and it is observable: `egocentric_state()`
   concatenates `data.act` onto the hinge qpos/qvel.

2. Actuator transmission through a fixed tendon for eight of the twelve
   (`lift_*`, `extend_*`), which point_mass already exercised, and directly
   through a joint for the four `yaw_*`.

3. `<equality><tendon>` on the four `coupling_*` tendons, which constrains
   .333*(pitch + knee + ankle) to zero per leg.

4. `accelerometer` and `force`/`torque` sensors — the first two that need
   MuJoCo's `mj_rnePostConstraint` pass rather than a kinematic read.

The `<default>` tree is deep and load-bearing: `class="body"` supplies capsule
type/size/condim/density and every joint's damping/armature/limited, and the
actuator classes (`yaw_act`, `lift_act`, `extend_act`) supply nothing but
`ctrlrange` on top of a bare `<general>` default. Nothing here is spelled out
inline that the reference leaves to a class.
"""

from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.physics3d.types import ConeType

from mojo_rl.envs.dm_control.quadruped.quadruped_dims import (
    DM_QUADRUPED_FETCH_DIMS,
    DM_QUADRUPED_WALK_DIMS,
    DM_QUADRUPED_RUN_DIMS,
)



# ⚠ THE FLOOR IS THE ONLY PER-TASK DIFFERENCE, and it is now a difference
# between three FILES rather than three comptime concatenations.
# dm_control computes `f'{floor_size} {floor_size} .5'` from
# `_DEFAULT_TIME_LIMIT * speed` — walk: 20 * 0.5 = 10.0 (a Python float, hence
# "10.0"); run: 20 * 5 = 100 (both ints, hence "100"). The TEXT differs; the
# number does not. Both spellings are preserved verbatim in
# `assets/quadruped_walk.xml` and `assets/quadruped_run.xml`.



# --- fetch: the arena the other two tasks strip out -------------------------
# `make_model(walls_and_ball=True)` keeps four WALLS, the ball and the target
# site; walk/run call `make_model(floor_size=...)` with the default
# `walls_and_ball=False`, which deletes all three. fetch passes NO floor_size,
# so the floor keeps quadruped.xml's own `15 15 .5` rather than the
# `_DEFAULT_TIME_LIMIT * speed` the other two compute.
#
# ⚠ THE WALLS ARE TILTED PLANES, not boxes — `class="wall"` resolves to
# `<geom type="plane" material="decoration"/>` and each carries a `zaxis` that
# leans it inward by 45 degrees. They are the reason the oriented-plane work
# had to land first: a plane whose normal is not +z used to be treated as
# though it were, so the ball would have passed straight through all four.
#
# ⚠ THE TARGET SITE IS DECLARED BEFORE THE TORSO BODY, so it takes site id 0
# and shifts EVERY other site index by one relative to walk/run. Constants
# below are re-derived for this model rather than reused.

# The ball rides in as its own fragment so it lands LAST in the accumulated
# <worldbody> — which is where quadruped.xml declares it, after the torso body
# closes. That makes it the last body, and `ball_root` the last joint, so its
# qpos/qvel occupy the tail of the state vector and nothing above them moves.
#
# ⚠ `condim="6"` HERE IS THE WHOLE REASON THE PYRAMIDAL EDGE BUILDER HAD TO BE
# GENERALISED. Until 004fe439 the torsional and rolling rows were built into a
# workspace the solver never read, and this ball would have spun and rolled
# without resistance while every other model in the tree stayed exact. See
# tests/physics3d/test_rolling_friction_vs_mujoco.mojo.
#
# ⚠ `priority="1"` makes the ball's condim, friction AND solref win over the
# floor's outright, rather than being mixed — including the DIRECT-form
# `solref="-10000 -30"` (negative = stiffness/damping given literally).
# ⚠ NO <asset> HERE. The ball texture and material are already in
# `_QUADRUPED_HEAD` — `make_model` strips the ball BODY but leaves its asset
# behind, so walk and run carry an unused ball material too. Re-declaring it
# fails the compile outright ("repeated name 'ball' in texture"), which is the
# friendly version of this mistake.



comptime qfp = DM_QUADRUPED_FETCH_DIMS

comptime qwp = DM_QUADRUPED_WALK_DIMS

comptime qrp = DM_QUADRUPED_RUN_DIMS

# obs = egocentric_state (16 hinge qpos + 16 hinge qvel + 12 act = 44)
#     + torso_velocity (3) + torso_upright (1) + imu (6) + force_torque (24)
#     = 78
comptime QUADRUPED_OBS_DIM: Int = 78


comptime DMQuadrupedWalkModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/quadruped_walk.xml",
    nbody=qwp.NBODY, njoint=qwp.NJOINT, nq=qwp.NQ, nv=qwp.NV,
    ngeom=qwp.NGEOM, nact=qwp.NACT, ntex=qwp.NTEX, nmat=qwp.NMAT,
    nlight=qwp.NLIGHT, ncam=qwp.NCAM, nsite=qwp.NSITE,
    max_tendon=qwp.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    # Four toes on the floor, plus the torso ellipsoid when it falls over.
    max_contacts=16,
    obs_dim_override=QUADRUPED_OBS_DIM,
    obs_qpos_skip=0,
    timestep=qwp.TIMESTEP,
    # MuJoCo `m->na`: 12 `<general dyntype="filter">` servos, one activation
    # each. Hand-supplied because `parse_xml` does not compute it; `init_fields`
    # asserts it against the parsed XML.
    na = 12,
]

comptime DMQuadrupedRunModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/quadruped_run.xml",
    nbody=qrp.NBODY, njoint=qrp.NJOINT, nq=qrp.NQ, nv=qrp.NV,
    ngeom=qrp.NGEOM, nact=qrp.NACT, ntex=qrp.NTEX, nmat=qrp.NMAT,
    nlight=qrp.NLIGHT, ncam=qrp.NCAM, nsite=qrp.NSITE,
    max_tendon=qrp.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=16,
    obs_dim_override=QUADRUPED_OBS_DIM,
    obs_qpos_skip=0,
    timestep=qrp.TIMESTEP,
    # MuJoCo `m->na`: 12 `<general dyntype="filter">` servos, one activation
    # each. Hand-supplied because `parse_xml` does not compute it; `init_fields`
    # asserts it against the parsed XML.
    na = 12,
]


# --- Task constants, transcribed from suite/quadruped.py --------------------
comptime QUADRUPED_RUN_SPEED: Float64 = 5.0
comptime QUADRUPED_WALK_SPEED: Float64 = 0.5

# The torso body carries the IMU/velocimeter site; the four toes carry the
# force/torque sites. Indices are OURS and are pinned by the parity test
# (`test_rne_post_sensors_vs_mujoco` proves our body and site ORDER equals
# MuJoCo's for this XML, so `mj_name2id` values are valid here).
comptime TORSO_BODY_IDX: Int = 1
comptime TORSO_SITE_IDX: Int = 24

# Toes in SENSOR-ID order — front-left, front-right, back-right, back-left —
# which is how `physics.force_torque()` lays them out. NOT the reference's
# `_TOES` list, which is FL, BL, BR, FR and is only used by `toe_positions()`.
# Each leg is four bodies (hip, knee, ankle, toe), declared in the same order,
# so the toes are evenly spaced. Sites are contiguous because the four toe
# sites are the last four declared.
comptime TOE_BODY_0: Int = 5
comptime TOE_BODY_STRIDE: Int = 4
comptime TOE_SITE_0: Int = 25

# Joint 0 is the free root; joints 1..16 are the leg hinges, so their qpos and
# dof blocks are contiguous. `egocentric_state` reads exactly these.
comptime N_HINGE: Int = 16
comptime HINGE_QPOS_0: Int = 7
comptime HINGE_DOF_0: Int = 6

# --- fetch model + its own indices ------------------------------------------
# obs = _common_observations (78) + ball_state (9) + target_position (3)
comptime QUADRUPED_FETCH_OBS_DIM: Int = 90

comptime DMQuadrupedFetchModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/quadruped_fetch.xml",
    nbody=qfp.NBODY, njoint=qfp.NJOINT, nq=qfp.NQ, nv=qfp.NV,
    ngeom=qfp.NGEOM, nact=qfp.NACT, ntex=qfp.NTEX, nmat=qfp.NMAT,
    nlight=qfp.NLIGHT, ncam=qfp.NCAM, nsite=qfp.NSITE,
    max_tendon=qfp.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    # Four toes and the torso ellipsoid as in walk/run, plus the ball against
    # the floor, the four walls, and any leg it is being nudged with.
    max_contacts=24,
    obs_dim_override=QUADRUPED_FETCH_OBS_DIM,
    obs_qpos_skip=0,
    timestep=qfp.TIMESTEP,
    # ⚠ WITHOUT THIS THE BALL'S condim=6 IS SILENTLY DOWNGRADED to four
    # pyramid edges and it spins and rolls unopposed. Derived by scanning the
    # XML, never hand-written — passing the number by hand is the defect
    # 004fe439 fixed, in a new dress.
    max_condim=qfp.MAX_CONDIM,
    # MuJoCo `m->na`: 12 `<general dyntype="filter">` servos, one activation
    # each. Hand-supplied because `parse_xml` does not compute it; `init_fields`
    # asserts it against the parsed XML.
    na = 12,
]

# ⚠ SITE IDS ARE NOT walk/run's. `<site name="target">` is declared before the
# torso body, so it takes id 0 and pushes every other site up by one. These are
# read out of a compiled mjModel, not counted by hand.
comptime FETCH_TARGET_SITE_IDX: Int = 0
comptime FETCH_WORKSPACE_SITE_IDX: Int = 3
comptime FETCH_TORSO_SITE_IDX: Int = 25
comptime FETCH_TOE_SITE_0: Int = 26

# The ball is the last body and `ball_root` the last joint, so the quadruped's
# own qpos/qvel layout is untouched and only the tail is new.
comptime FETCH_BALL_BODY_IDX: Int = 18
comptime FETCH_BALL_QPOS_0: Int = 23
comptime FETCH_BALL_DOF_0: Int = 22

# Reward geometry, read from the compiled model rather than from the XML text.
comptime FETCH_FLOOR_HALF: Float64 = 15.0
comptime FETCH_WORKSPACE_RADIUS: Float64 = 0.3
comptime FETCH_TARGET_RADIUS: Float64 = 0.4
comptime FETCH_BALL_RADIUS: Float64 = 0.15
