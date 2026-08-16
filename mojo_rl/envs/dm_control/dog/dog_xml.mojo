"""`dm_control` `dog` models — port of `dog.py` + `dog.xml` (Phase 4).

`dog.py::make_model(floor_size, remove_ball)` parses dog.xml with lxml and,
for stand / walk / trot / run, DELETES the ball body, the target geom, the
`ball` and `head` cameras and the four `wall_*` geoms, then rewrites the floor
plane's half-extent. The only per-task difference is that half-extent:

    floor_size = move_speed * _DEFAULT_TIME_LIMIT (15)

    stand  _WALK_SPEED 1 -> 15      trot  _TROT_SPEED 3 -> 45
    walk   _WALK_SPEED 1 -> 15      run   _RUN_SPEED  9 -> 135

so stand and walk share ONE model. `fetch` (Phase 5) keeps the ball and uses
dog.xml's own default of 10.

⚠ THIS FILE IS GENERATED, and deliberately so — 69 kB of MJCF is not something
to retype. `tests/dm_control/dog_ref.py::port_fragment` emits it; that function
is where the four text-level deviations are documented and is the only place
they can be changed. Regenerate rather than hand-edit.

THE MESH-INERTIA BAKE — a labelled deviation
--------------------------------------------
dog.xml is 82 kB: 62 bodies, 74 joints, 290 geoms, 162 STL meshes, 38
actuators. **The 162 mesh geoms are gone from this port.** They are `<default
class="bone">`, which sets `contype="0" conaffinity="0"`, so they never
collide; their only contribution to physics is INERTIA, because dog declares
just 3 explicit `<inertial>` elements and lets MuJoCo derive 59 bodies\' mass
and inertia tensor from mesh volume at `density="1100"`/`"300"`.

So the bake states that result explicitly — `<inertial pos quat mass
diaginertia>` per body at 17 significant digits — and deletes the meshes. The
port carries 128 geoms and no asset tree.

⚠ A BAKED CONSTANT THAT OUTLIVES ITS JUSTIFICATION IS A BUG WAITING TO HAPPEN
(`point_mass`\'s tendon workaround is the precedent). What keeps this one
honest is `dog_ref.check_bake`, run by the parity test: it compiles baked and
unbaked and diffs EVERY table, exempting only the geom-indexing columns, with
the surviving geoms matched BY NAME so an id shift fails rather than passes.
Measured: 0 mismatches, and a 300-step rollout agrees bit-for-bit (max|dqpos|
and max|dqvel| both exactly 0.0).

⚠ THE BAKE IS NOT AVAILABLE TO MANIPULATION (Phase 7). Jaco\'s meshes COLLIDE
(`<geom type="mesh" condim="3" contype="3" conaffinity="2">`), so that layer
needs a real mesh narrow phase. Do not cite dog as a precedent there.

WHAT DOG NEEDS THAT EARLIER PORTS DID NOT
-----------------------------------------
1. `noslip_iterations="4"`. A post-solver pass that removes residual slip in
   the friction dimensions with the normal forces frozen. NOT a refinement to
   round off: measured against MuJoCo with it disabled, `max|dqvel|` is 2.9e-2
   on the FIRST contacting step, so it is first-order, not chaos.
2. `<geom priority>` with `condim="6"`. dog has 42 teeth on
   `class="tooth_primitive"` (`condim="6" priority="2" friction="0.5 .01 .01"`)
   against 77 condim-1 primitives and a condim-3 floor. Priority means the
   teeth dictate condim, friction AND solref wholesale wherever they touch.
   Both halves of this landed in Phase 3 — the condim>=4 rows were structurally
   present and completely INERT until then.
3. A `subtreeangmom` sensor. Declared here for model fidelity; see
   `dog_config` for why nothing reads it.
4. 74 joints, which is what raised `MAX_COMPTIME_JOINTS` from 64 to 96.

Comptime build cost, since `docs/DM_CONTROL_PORT_PHASE2.md` §9 ranked it risk
2: **14.8 s** for the model def. humanoid_CMU\'s 23 kB took ~2 m 50 s, so the
scaling is not in the XML length and dog is CHEAPER despite being 3x the text.
The risk is retired.
"""

from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.physics3d.types import ConeType

from mojo_rl.envs.dm_control.dog.dog_dims import (
    DM_DOG_STAND_WALK_DIMS,
    DM_DOG_TROT_DIMS,
    DM_DOG_RUN_DIMS,
)


# --- everything before the floor geom -----------------------------------

# --- the floor plane, the ONLY per-task difference -----------------------

# --- the dog itself, plus actuators / tendons / sensors ------------------

# dog's deformable envelope — the thing dm_control actually shows you.
#
# ⚠ WITHOUT THIS, DOG RENDERS AS A SKELETON. The reference hides everything else:
# its 162 bone meshes sit in geom group 5 and its collision capsules in group 3,
# both of which MuJoCo's viewer leaves invisible (`mjv_defaultOption` enables
# groups 0-2 only). So the picture is the skin, plus the 23 `class="visible_bone"`
# geoms in group 1 — and this port dropped the meshes, which leaves the skin
# carrying the whole image.
#
# ⚠ PATHS ARE REPO-ROOT-RELATIVE, matching `sawyer_scene_xml.mojo`. Every command
# in this project runs from the project root; the renderer opens these verbatim.
#
# ⚠ NOT PART OF THE PHYSICS, and it must stay that way. A `<skin>` has no
# collision, no inertia and no DOF — MuJoCo treats it as pure visualization. It
# is appended to the model's assets rather than merged into the reference body
# text so that the parity tests, which compare against a MuJoCo model built from
# that same body text, cannot be perturbed by it.



comptime dsp = DM_DOG_STAND_WALK_DIMS

comptime dtp = DM_DOG_TROT_DIMS

comptime drp = DM_DOG_RUN_DIMS

# --- observation layout, transcribed from dog.py::get_observation_components
#
#   joint_angles         73   every HINGE qpos (the free root is skipped)
#   joint_velocites      73   every HINGE qvel  [sic - dm_control's spelling]
#   torso_pelvis_height   2   xpos[['torso','pelvis'], 'z']
#   z_projection          9   xmat[['skull','torso','pelvis'], 'zx zy zz']
#   torso_com_velocity    3   subtreelinvel('torso') rotated into torso frame
#   inertial_sensors      9   accelerometer + velocimeter + gyro (site 'head')
#   foot_forces          12   force sensors foot_L, foot_R, hand_L, hand_R
#   touch_sensors         4   touch sensors palm_L, palm_R, sole_L, sole_R
#   actuator_state       38   data.act - every actuator is dyntype="filter"
comptime DOG_OBS_DIM: Int = 223

# 24 contacts: four feet is the steady state, but dog has 120 colliding
# primitives and falls over a lot while it learns.
comptime _DOG_MAX_CONTACTS: Int = 24

comptime DMDogStandWalkModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/dog_stand_walk.xml",
    nbody=dsp.NBODY, njoint=dsp.NJOINT, nq=dsp.NQ, nv=dsp.NV,
    ngeom=dsp.NGEOM, nact=dsp.NACT, ntex=dsp.NTEX, nmat=dsp.NMAT,
    nlight=dsp.NLIGHT, ncam=dsp.NCAM, nsite=dsp.NSITE,
    # ⚠ BOTH OF THESE DEFAULT TO 0 AND NOTHING CHECKS THEM.
    # Omitting nexclude builds an exclusion-free model silently —
    # dog declares THIRTY <exclude> pairs, so 30 body pairs would
    # collide that MuJoCo never lets collide.
    neq=dsp.NEQ, nexclude=dsp.NEXCLUDE,
    max_tendon=dsp.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=_DOG_MAX_CONTACTS,
    max_condim=dsp.MAX_CONDIM,
    # dog.xml sets noslip_iterations=4; `mj_solNoSlip` is now implemented
    # (solver/noslip.mojo) and wired into the pyramidal Newton path, so this
    # is a live request, not a declared gap.
    noslip_iter=dsp.NOSLIP_ITER,
    obs_dim_override=DOG_OBS_DIM,
    obs_qpos_skip=0,
    timestep=dsp.TIMESTEP,
    # MuJoCo `m->na`: every one of dog's 38 actuators is a filtered
    # `<general>`, so na == nu here. Asserted in `init_fields`.
    na = 38,
]

comptime DMDogTrotModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/dog_trot.xml",
    nbody=dtp.NBODY, njoint=dtp.NJOINT, nq=dtp.NQ, nv=dtp.NV,
    ngeom=dtp.NGEOM, nact=dtp.NACT, ntex=dtp.NTEX, nmat=dtp.NMAT,
    nlight=dtp.NLIGHT, ncam=dtp.NCAM, nsite=dtp.NSITE,
    # ⚠ BOTH OF THESE DEFAULT TO 0 AND NOTHING CHECKS THEM.
    # Omitting nexclude builds an exclusion-free model silently —
    # dog declares THIRTY <exclude> pairs, so 30 body pairs would
    # collide that MuJoCo never lets collide.
    neq=dtp.NEQ, nexclude=dtp.NEXCLUDE,
    max_tendon=dtp.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=_DOG_MAX_CONTACTS,
    max_condim=dtp.MAX_CONDIM,
    # dog.xml sets noslip_iterations=4; `mj_solNoSlip` is now implemented
    # (solver/noslip.mojo) and wired into the pyramidal Newton path, so this
    # is a live request, not a declared gap.
    noslip_iter=dtp.NOSLIP_ITER,
    obs_dim_override=DOG_OBS_DIM,
    obs_qpos_skip=0,
    timestep=dtp.TIMESTEP,
    # MuJoCo `m->na`: every one of dog's 38 actuators is a filtered
    # `<general>`, so na == nu here. Asserted in `init_fields`.
    na = 38,
]

comptime DMDogRunModel = ModelDefFromXML[
    xml_path="mojo_rl/envs/dm_control/assets/dog_run.xml",
    nbody=drp.NBODY, njoint=drp.NJOINT, nq=drp.NQ, nv=drp.NV,
    ngeom=drp.NGEOM, nact=drp.NACT, ntex=drp.NTEX, nmat=drp.NMAT,
    nlight=drp.NLIGHT, ncam=drp.NCAM, nsite=drp.NSITE,
    # ⚠ BOTH OF THESE DEFAULT TO 0 AND NOTHING CHECKS THEM.
    # Omitting nexclude builds an exclusion-free model silently —
    # dog declares THIRTY <exclude> pairs, so 30 body pairs would
    # collide that MuJoCo never lets collide.
    neq=drp.NEQ, nexclude=drp.NEXCLUDE,
    max_tendon=drp.NTENDON,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=_DOG_MAX_CONTACTS,
    max_condim=drp.MAX_CONDIM,
    # dog.xml sets noslip_iterations=4; `mj_solNoSlip` is now implemented
    # (solver/noslip.mojo) and wired into the pyramidal Newton path, so this
    # is a live request, not a declared gap.
    noslip_iter=drp.NOSLIP_ITER,
    obs_dim_override=DOG_OBS_DIM,
    obs_qpos_skip=0,
    timestep=drp.TIMESTEP,
    # MuJoCo `m->na`: every one of dog's 38 actuators is a filtered
    # `<general>`, so na == nu here. Asserted in `init_fields`.
    na = 38,
]


# --- task constants, transcribed from suite/dog.py -------------------------
comptime DOG_WALK_SPEED: Float64 = 1.0
comptime DOG_TROT_SPEED: Float64 = 3.0
comptime DOG_RUN_SPEED: Float64 = 9.0

# `_MAX_UPRIGHT_ANGLE = 30` degrees; the reward uses its COSINE.
comptime DOG_MIN_UPRIGHT_COSINE: Float64 = 0.8660254037844387
comptime DOG_STAND_HEIGHT_FRACTION: Float64 = 0.9

# control_timestep .015 / physics timestep .005; time_limit 15 s.
comptime DOG_FRAME_SKIP: Int = 3
comptime DOG_MAX_STEPS: Int = 1000

# --- indices, all read back from the compiled model and pinned by the gate --
# The free root joint is joint 0, then 73 hinges occupying a CONTIGUOUS block
# of qpos and dof (checked in the parity test, not assumed).
comptime DOG_N_HINGE: Int = 73
comptime DOG_HINGE_QPOS_0: Int = 7
comptime DOG_HINGE_DOF_0: Int = 6

comptime DOG_TORSO_BODY_IDX: Int = 1
comptime DOG_PELVIS_BODY_IDX: Int = 9
comptime DOG_SKULL_BODY_IDX: Int = 48

# Sites, in MuJoCo's order for this XML.
comptime DOG_SITE_FOOT_ANCHOR_L: Int = 0
comptime DOG_SITE_SOLE_L: Int = 1
comptime DOG_SITE_FOOT_ANCHOR_R: Int = 2
comptime DOG_SITE_SOLE_R: Int = 3
comptime DOG_SITE_TAIL_TIP: Int = 4
comptime DOG_SITE_HEAD: Int = 5
comptime DOG_SITE_UPPER_BITE: Int = 6
comptime DOG_SITE_LOWER_BITE: Int = 7
comptime DOG_SITE_HAND_ANCHOR_L: Int = 8
comptime DOG_SITE_PALM_L: Int = 9
comptime DOG_SITE_HAND_ANCHOR_R: Int = 10
comptime DOG_SITE_PALM_R: Int = 11

# `sensordata` addresses. dm_control reads these BY NAME, so the order here is
# the model's, not the observation's — `foot_forces` is (foot_L, foot_R,
# hand_L, hand_R) and `touch_sensors` is (palm_L, palm_R, sole_L, sole_R),
# neither of which is contiguous-in-model order by accident.
comptime DOG_SENS_ACCELEROMETER: Int = 0
comptime DOG_SENS_VELOCIMETER: Int = 3
comptime DOG_SENS_GYRO: Int = 6
comptime DOG_SENS_TORSO_LINVEL: Int = 9
comptime DOG_SENS_TORSO_ANGMOM: Int = 12
comptime DOG_SENS_TOUCH_PALM_L: Int = 15
comptime DOG_SENS_TOUCH_PALM_R: Int = 16
comptime DOG_SENS_TOUCH_SOLE_L: Int = 17
comptime DOG_SENS_TOUCH_SOLE_R: Int = 18
comptime DOG_SENS_FORCE_FOOT_L: Int = 19
comptime DOG_SENS_FORCE_FOOT_R: Int = 22
comptime DOG_SENS_FORCE_HAND_L: Int = 25
comptime DOG_SENS_FORCE_HAND_R: Int = 28

# The four `<force>` sensors hang off sites on their OWN bodies (each anchor is
# a body, not a site on the paw), and `site_force_torque` needs the body index
# as well as the site index.
comptime DOG_BODY_FOOT_ANCHOR_L: Int = 13
comptime DOG_BODY_FOOT_ANCHOR_R: Int = 18
comptime DOG_BODY_HAND_ANCHOR_L: Int = 53
comptime DOG_BODY_HAND_ANCHOR_R: Int = 59

# --- reward constants that `Stand.initialize_episode` MEASURES -------------
#
# The reference computes these per episode, but both are constants of the
# MODEL, so they are pinned here and gated against MuJoCo rather than
# recomputed every reset:
#
#   `initialize_episode` calls `physics.reset()` FIRST, which restores
#   `qpos0` — so `torso_pelvis_height()` is read at the default pose, before
#   the orientation/velocity/act randomization that follows it. And
#   `body_subtreemass['torso']` and `opt.gravity` never change at all.
#
#   _stand_height = torso_pelvis_height() * _STAND_HEIGHT_FRACTION(0.9)
#   _body_weight  = -gravity[2] * body_subtreemass['torso']
#
# ⚠ `body_subtreemass['torso']` is the WHOLE model's mass (10.2409 kg): torso
# is the root body, so its subtree is every body. It is not the torso's own
# mass, and reading it as such would put the touch-reward threshold an order
# of magnitude low.
comptime DOG_STAND_HEIGHT_TORSO: Float64 = 0.373698
comptime DOG_STAND_HEIGHT_PELVIS: Float64 = 0.42854840999999994
comptime DOG_BODY_WEIGHT: Float64 = 100.46366448928629
