"""Physics3D GPU constants - Flat buffer layout for GPU kernels.

Primary state is qpos/qvel (joint space). Body positions (xpos, xquat, xipos)
are computed via forward kinematics and stored for collision detection.

State buffer layout per environment:
  [qpos: NQ | qvel: NV | qacc: NV | qfrc: NV |
   xpos: NBODY*3 | xquat: NBODY*4 | xipos: NBODY*3 |
   xvel: NBODY*3 | xangvel: NBODY*3 |
   contacts: MAX_CONTACTS*CONTACT_SIZE | metadata: METADATA_SIZE |
   site_xpos: NSITE*3 |
   cfrc_ext: NBODY*6 | cvel: NBODY*6 | cinert: NBODY*10 | qfrc_actuator: NV]

Model buffer (static, same for all environments):
  Per body (MODEL_BODY_SIZE=26): [mass, inv_mass, inertia(3), inv_inertia(3),
    pos(3), quat(4), parent, ipos(3), iquat(4), rootid, weldid, mocap]
  Per joint (MODEL_JOINT_SIZE=26): [type, body_id, qpos_adr, dof_adr,
    pos(3), axis(3), tau_limit, range_min/max, armature, damping, stiffness, springref, frictionloss,
    solref_limit(2), solimp_limit(5), qpos0]
  Metadata (MODEL_META_SIZE): [NBODY, NJOINT, gravity(3), timestep, _reserved(2),
    solref_contact(2), solimp_contact(5), solref_limit(2), solimp_limit(5), impratio, nequality,
    ntendon, nexclude, meaninertia, npair, noslip_tolerance, ccd_tolerance, ccd_iterations,
    ctrl_min, ctrl_max, multiccd_disabled]
  Curriculum (MODEL_CURRICULUM_SIZE=8): [up to 8 curriculum parameters]
  Per geom (MODEL_GEOM_SIZE=29): [type, body, pos(3), quat(4), radius, half_length,
    half_x/y/z, friction, contype, conaffinity, condim, friction_spin, friction_roll,
    rbound, solref(2), solimp(5), margin]
"""

# =============================================================================
# GPU Configuration
# =============================================================================

comptime TPB: Int = 256  # Threads per block (optimal for most GPUs)
comptime TILE: Int = 8  # Tile size for 2D operations


# =============================================================================
# Physics Defaults
# =============================================================================

comptime DEFAULT_GRAVITY_Z: Float32 = -9.81
comptime DEFAULT_TIMESTEP: Float32 = 0.01
comptime DEFAULT_RESTITUTION: Float32 = 0.0
comptime MAX_POS_CORRECTION_VEL: Float32 = 10.0  # Legacy, unused after accel-level migration


# =============================================================================
# State Buffer Layout - Joint Space (qpos, qvel, qacc, qfrc)
# =============================================================================

# These are computed as offsets based on NQ and NV parameters
# For a system with NQ total qpos and NV total qvel:
#
#   qpos: [0, NQ)
#   qvel: [NQ, NQ + NV)
#   qacc: [NQ + NV, NQ + 2*NV)
#   qfrc: [NQ + 2*NV, NQ + 3*NV)


# =============================================================================
# State Buffer Layout - World Space (xpos, xquat, xvel, xangvel)
# =============================================================================


# =============================================================================
# State Buffer Layout - Contacts
# =============================================================================

# Contact layout (same as Cartesian engine: 12 floats per contact)
comptime CONTACT_SIZE: Int = 30

comptime CONTACT_IDX_BODY_A: Int = 0
comptime CONTACT_IDX_BODY_B: Int = 1
comptime CONTACT_IDX_POS_X: Int = 2
comptime CONTACT_IDX_POS_Y: Int = 3
comptime CONTACT_IDX_POS_Z: Int = 4
comptime CONTACT_IDX_NX: Int = 5
comptime CONTACT_IDX_NY: Int = 6
comptime CONTACT_IDX_NZ: Int = 7
comptime CONTACT_IDX_DIST: Int = 8
comptime CONTACT_IDX_INCLUDEMARGIN: Int = 9
comptime CONTACT_IDX_FORCE_N: Int = 10
comptime CONTACT_IDX_FORCE_T1: Int = 11
comptime CONTACT_IDX_FORCE_T2: Int = 12
comptime CONTACT_IDX_FRICTION: Int = 13
comptime CONTACT_IDX_FRICTION_SPIN: Int = 14
comptime CONTACT_IDX_FRICTION_ROLL: Int = 15
comptime CONTACT_IDX_CONDIM: Int = 16
comptime CONTACT_IDX_FORCE_TORSION: Int = 17
comptime CONTACT_IDX_FORCE_ROLL1: Int = 18
comptime CONTACT_IDX_FORCE_ROLL2: Int = 19
comptime CONTACT_IDX_FRAME_T1_X: Int = 20  # T1 hint for tangent frame (capsule axis)
comptime CONTACT_IDX_FRAME_T1_Y: Int = 21
comptime CONTACT_IDX_FRAME_T1_Z: Int = 22
# ── Per-contact solver parameters, appended 2026-08-03 ──────────────────────
#
# Until now every contact row read ONE MODEL-LEVEL solref/solimp
# (`MODEL_META_IDX_SOLREF_CONTACT_*`), while the record already carried
# per-contact friction, condim and margin. So `<geom solref=... solimp=.../>`
# was parsed, written into the GEOM record (`GEOM_IDX_SOLREF_0`..`SOLIMP_4`)
# and then read by NOTHING — dead data on every build.
#
# These slots hold the pair's MIXED values, computed in the narrow phase by
# MuJoCo's rule (`engine_collision_driver.c:1426-1480`), so the solver reads
# them the same way it already reads friction.
#
# ⚠ APPENDED, NOT INSERTED. Every index 0..22 keeps its value, which is what
# lets a layout change like this be verified as inert: nothing that does not
# read 23..29 can behave differently.
comptime CONTACT_IDX_SOLREF_0: Int = 23  # mixed solref timeconst (or -stiffness)
comptime CONTACT_IDX_SOLREF_1: Int = 24  # mixed solref dampratio (or -damping)
comptime CONTACT_IDX_SOLIMP_0: Int = 25  # mixed solimp dmin
comptime CONTACT_IDX_SOLIMP_1: Int = 26  # mixed solimp dmax
comptime CONTACT_IDX_SOLIMP_2: Int = 27  # mixed solimp width
comptime CONTACT_IDX_SOLIMP_3: Int = 28  # mixed solimp midpoint
comptime CONTACT_IDX_SOLIMP_4: Int = 29  # mixed solimp power


# =============================================================================
# State Buffer Layout - Metadata
# =============================================================================

comptime METADATA_SIZE: Int = 16
"""Per-env metadata words: 4 fixed slots plus `META_IDX_TASK_PARAM_0..11`.

⚠ RAISED FROM 8 FOR `reassemble_5_bricks_random_order`, which stores TWO
five-entry orders — `desired_order` and `initial_order`, the second because its
relabeling is built from the first entry of it. Four slots were not enough for
even one of them, and the encodings that would have made four enough (a Lehmer
code, base-5 packing, deriving the last entry from the other four) all buy the
space by hiding the layout.

⚠ EVERY INDEX INTO `meta` IS `env * METADATA_SIZE + META_IDX_*`, never a
literal, so widening is a buffer-size change and nothing else. It costs 8
floats per env — `Data.meta` is `alloc(B * METADATA_SIZE)` — which is noise
next to a single body's state."""

comptime META_IDX_NUM_CONTACTS: Int = 0
comptime META_IDX_STEP_COUNT: Int = 1  # Episode step counter for truncation
comptime META_IDX_PREV_X: Int = 2  # Previous x position for velocity computation
comptime META_IDX_PREV_COM_X: Int = 3  # Reserved for prev CoM x (unused with cvel approach)

# ── Per-episode TASK-RANDOMIZED MODEL PARAMETERS (the G4 workaround) ──────
#
# ⚠ WHY THESE LIVE IN PER-ENV STATE AND NOT IN `Model`. A few dm_control
# tasks randomize a MODEL field per episode, not just the state:
# `point_mass-hard` redraws the two fixed-tendon coefficient vectors
# (`model.wrap_prm`) so each control drives a random linear combination of
# root_x/root_y. `Model` is deliberately SHARED across the batch — the design
# batches STATE, not MODEL — so a lane cannot own a different `Model.tendons`
# row without batching the whole record set.
#
# These four slots are the narrow escape: the randomized quantity is a handful
# of floats, it is per-episode (written by `init_qpos_gpu`, which already runs
# per lane at reset), and the ONLY consumer is the config's own actuation
# hook. That last part is a property to CHECK, not to assume — a tendon that
# is also `limited`, spring-loaded, or named in an `<equality>` would be read
# by the SOLVER out of `Model.tendons`, where these writes are invisible. The
# configs that use these slots carry a comptime assert to that effect.
#
# NOT ZEROED between episodes beyond what the writer does: `_reset_env_lane`
# sets `META_IDX_STEP_COUNT` and leaves the rest, exactly as it always did for
# `META_IDX_PREV_X`. A hook that reads a slot it never wrote gets the previous
# episode's value.
# ⚠ ALSO THE HOME OF PER-EPISODE TASK STATE THAT IS NOT A MODEL FIELD. The
# brick tasks' `desired_order` and relabeling live here for the same reason:
# they are per-episode, they are a handful of floats, and `prev_x` — the only
# other per-env scalar — is rewritten every step and would lose them.
comptime META_IDX_TASK_PARAM_0: Int = 4
comptime META_IDX_TASK_PARAM_1: Int = 5
comptime META_IDX_TASK_PARAM_2: Int = 6
comptime META_IDX_TASK_PARAM_3: Int = 7
comptime META_IDX_TASK_PARAM_4: Int = 8
comptime META_IDX_TASK_PARAM_5: Int = 9
comptime META_IDX_TASK_PARAM_6: Int = 10
comptime META_IDX_TASK_PARAM_7: Int = 11
comptime META_IDX_TASK_PARAM_8: Int = 12
comptime META_IDX_TASK_PARAM_9: Int = 13
comptime META_IDX_TASK_PARAM_10: Int = 14
comptime META_IDX_TASK_PARAM_11: Int = 15


# =============================================================================
# Total State Size Computation
# =============================================================================


# =============================================================================
# Model Buffer Layout - Per Body
# =============================================================================

comptime MODEL_BODY_SIZE: Int = 26

comptime BODY_IDX_MASS: Int = 0
comptime BODY_IDX_INV_MASS: Int = 1
comptime BODY_IDX_IXX: Int = 2
comptime BODY_IDX_IYY: Int = 3
comptime BODY_IDX_IZZ: Int = 4
comptime BODY_IDX_INV_IXX: Int = 5
comptime BODY_IDX_INV_IYY: Int = 6
comptime BODY_IDX_INV_IZZ: Int = 7
comptime BODY_IDX_POS_X: Int = 8  # Local position in parent frame
comptime BODY_IDX_POS_Y: Int = 9
comptime BODY_IDX_POS_Z: Int = 10
comptime BODY_IDX_QUAT_X: Int = 11  # Local orientation in parent frame
comptime BODY_IDX_QUAT_Y: Int = 12
comptime BODY_IDX_QUAT_Z: Int = 13
comptime BODY_IDX_QUAT_W: Int = 14
comptime BODY_IDX_PARENT: Int = 15  # Parent body index (-1 for world)
comptime BODY_IDX_IPOS_X: Int = 16  # CoM offset from body origin (body frame)
comptime BODY_IDX_IPOS_Y: Int = 17
comptime BODY_IDX_IPOS_Z: Int = 18
comptime BODY_IDX_IQUAT_X: Int = 19  # Inertia frame quaternion (body frame)
comptime BODY_IDX_IQUAT_Y: Int = 20
comptime BODY_IDX_IQUAT_Z: Int = 21
comptime BODY_IDX_IQUAT_W: Int = 22
comptime BODY_IDX_ROOTID: Int = 23  # Root body index (child of worldbody)
comptime BODY_IDX_WELDID: Int = 24  # Weld body index (MuJoCo body_weldid)
comptime BODY_IDX_MOCAP: Int = 25  # 1.0 if body pose is externally set (mocap)


# =============================================================================
# Model Buffer Layout - Per Joint
# =============================================================================

comptime MODEL_JOINT_SIZE: Int = 26  # +7 for per-joint solref/solimp limits (5 params) + qpos0

comptime JOINT_IDX_TYPE: Int = 0  # JNT_FREE, JNT_BALL, JNT_SLIDE, JNT_HINGE
comptime JOINT_IDX_BODY_ID: Int = 1
comptime JOINT_IDX_QPOS_ADR: Int = 2
comptime JOINT_IDX_DOF_ADR: Int = 3
comptime JOINT_IDX_POS_X: Int = 4
comptime JOINT_IDX_POS_Y: Int = 5
comptime JOINT_IDX_POS_Z: Int = 6
comptime JOINT_IDX_AXIS_X: Int = 7
comptime JOINT_IDX_AXIS_Y: Int = 8
comptime JOINT_IDX_AXIS_Z: Int = 9
comptime JOINT_IDX_TAU_LIMIT: Int = 10
comptime JOINT_IDX_RANGE_MIN: Int = 11  # Minimum position (radians for hinge, meters for slide)
comptime JOINT_IDX_RANGE_MAX: Int = 12  # Maximum position
comptime JOINT_IDX_ARMATURE: Int = 13  # Rotor inertia (added to M diagonal)
comptime JOINT_IDX_DAMPING: Int = 14  # Passive joint damping
comptime JOINT_IDX_STIFFNESS: Int = 15  # Passive joint stiffness (spring)
comptime JOINT_IDX_SPRINGREF: Int = 16  # Spring reference position (rest position)
comptime JOINT_IDX_FRICTIONLOSS: Int = 17  # Dry friction loss (Coulomb friction)
comptime JOINT_IDX_SOLREF_LIMIT_0: Int = 18  # Per-joint limit solref timeconst
comptime JOINT_IDX_SOLREF_LIMIT_1: Int = 19  # Per-joint limit solref dampratio
comptime JOINT_IDX_SOLIMP_LIMIT_0: Int = 20  # Per-joint limit solimp dmin
comptime JOINT_IDX_SOLIMP_LIMIT_1: Int = 21  # Per-joint limit solimp dmax
comptime JOINT_IDX_SOLIMP_LIMIT_2: Int = 22  # Per-joint limit solimp width
comptime JOINT_IDX_SOLIMP_LIMIT_3: Int = 23  # Per-joint limit solimp midpoint
comptime JOINT_IDX_SOLIMP_LIMIT_4: Int = 24  # Per-joint limit solimp power
comptime JOINT_IDX_QPOS0: Int = 25  # Joint reference position (MuJoCo qpos0 / ref)

# ⚠⚠ HOW AN UNLIMITED JOINT IS ENCODED, AND IT IS NOT MuJoCo'S ENCODING.
# The record has NO `limited` flag: `FlatModelDef`'s `JointData.is_limited` is
# known by the parser and dropped by `fields_build`. An unlimited joint instead
# carries a range of `[-JOINT_RANGE_UNLIMITED, +JOINT_RANGE_UNLIMITED]`, wide
# enough that the limit row can never activate.
#
# MuJoCo stores the opposite: `jnt_range = [0, 0]` with `jnt_limited = 0`. So
# **`range_min < range_max` DOES NOT MEAN "limited" HERE** — it is true for
# every joint in every model. Code ported from a routine that tests MuJoCo's
# `[0, 0]` (or resolves `limited="auto"` that way) reads every unlimited joint
# as limited to +-1e10, which is not a compile error and not a wrong number
# anywhere the range is only a clamp — it goes wrong where the range is used as
# a SAMPLING BOUND.
#
# That is exactly how it was found: `manipulation_reach_config` draws IK retry
# poses uniformly over each arm joint's range, and four of Jaco's six arm
# joints are unlimited. dm_control gives an unlimited HINGE `[0, 2*pi]`
# (`entities/manipulators/base._get_joint_pos_sampling_bounds`); we drew from
# +-1e10, so every retry started from a meaningless pose and the TCP
# initializer exhausted on 7 of 24 resets with 10/10 IK failures. dm_control's
# own IK reaches 30/30 of the same targets in 2.4 attempts.
#
# Test against THIS constant, never against `min < max`.
comptime JOINT_RANGE_UNLIMITED: Float64 = 1e10


# =============================================================================
# Model Buffer Layout - Global Metadata
# =============================================================================

comptime MODEL_META_SIZE: Int = 38

comptime MODEL_META_IDX_NBODY: Int = 0
comptime MODEL_META_IDX_NJOINT: Int = 1
comptime MODEL_META_IDX_GRAVITY_X: Int = 2
comptime MODEL_META_IDX_GRAVITY_Y: Int = 3
comptime MODEL_META_IDX_GRAVITY_Z: Int = 4
comptime MODEL_META_IDX_TIMESTEP: Int = 5
# Fluid dynamics parameters (MuJoCo option.density / option.viscosity)
# These occupy the previously-reserved slots 6 and 7.
comptime MODEL_META_IDX_DENSITY: Int = 6  # Fluid mass density (kg/m³), 0 = disabled
comptime MODEL_META_IDX_VISCOSITY: Int = 7  # Fluid dynamic viscosity (Pa·s), 0 = disabled
# solref/solimp contact parameters (MuJoCo impedance model)
comptime MODEL_META_IDX_SOLREF_CONTACT_0: Int = 8  # timeconst
comptime MODEL_META_IDX_SOLREF_CONTACT_1: Int = 9  # dampratio
comptime MODEL_META_IDX_SOLIMP_CONTACT_0: Int = 10  # dmin
comptime MODEL_META_IDX_SOLIMP_CONTACT_1: Int = 11  # dmax
comptime MODEL_META_IDX_SOLIMP_CONTACT_2: Int = 12  # width
comptime MODEL_META_IDX_SOLIMP_CONTACT_3: Int = 13  # midpoint
comptime MODEL_META_IDX_SOLIMP_CONTACT_4: Int = 14  # power
# solref/solimp limit parameters (MuJoCo impedance model)
comptime MODEL_META_IDX_SOLREF_LIMIT_0: Int = 15  # timeconst
comptime MODEL_META_IDX_SOLREF_LIMIT_1: Int = 16  # dampratio
comptime MODEL_META_IDX_SOLIMP_LIMIT_0: Int = 17  # dmin
comptime MODEL_META_IDX_SOLIMP_LIMIT_1: Int = 18  # dmax
comptime MODEL_META_IDX_SOLIMP_LIMIT_2: Int = 19  # width
comptime MODEL_META_IDX_SOLIMP_LIMIT_3: Int = 20  # midpoint
comptime MODEL_META_IDX_SOLIMP_LIMIT_4: Int = 21  # power
# Friction cone model
comptime MODEL_META_IDX_IMPRATIO: Int = 22  # MuJoCo impratio
# ⚠ WHICH SOLVER THE **MODEL** ASKS FOR, as opposed to the one the caller
# happened to build. `<option cone/solver/integrator>` went unparsed until
# 2026-08-19, so on the runtime path there was nothing to compare a built
# integrator against and the studio ran every model on ELLIPTIC + PGS + Euler.
# Appended (34 -> 37), so every index 0..33 keeps its value.
comptime MODEL_META_IDX_CONE: Int = 34  # ConeType, MuJoCo default PYRAMIDAL
comptime MODEL_META_IDX_SOLVER: Int = 35  # SolverType, MuJoCo default NEWTON
comptime MODEL_META_IDX_INTEGRATOR: Int = 36  # IntegratorType, default EULER
# ⚠⚠ THE CONDIM THE **MODEL** NEEDS, versus the `MAX_CONDIM` a caller built.
# `contact_solve` clamps `condim > MAX_CONDIM` down to it SILENTLY, in both
# cone branches — so spot's `condim="6"` feet, which want torsional and rolling
# friction, were solved as plain condim 3 with no indication. The comptime path
# has `ParsedModel.MAX_CONDIM` and every def passes it; the runtime path had no
# equivalent at all, so the studio's hardcoded 3 could not even be compared
# against what the file asked for.
comptime MODEL_META_IDX_MAX_CONDIM: Int = 37  # max geom condim, >= 3
# Equality constraints
comptime MODEL_META_IDX_NEQUALITY: Int = 23  # Number of equality constraints
# Fixed tendons
comptime MODEL_META_IDX_NTENDON: Int = 24  # Number of fixed tendons
comptime MODEL_META_IDX_NEXCLUDE: Int = 25  # Number of contact exclude pairs
# `mjModel.stat.meaninertia` — the MEAN OF THE MASS-MATRIX DIAGONAL at qpos0,
# armature included (`engine_setconst.c:1139-1146`):
#
#     meaninertia = (1/nv) * sum_i qM[dof_Madr[i]]
#
# Its only consumer is `mj_solNoSlip`'s convergence test, which scales the
# per-iteration improvement by `1 / (meaninertia * max(1, nv))` before
# comparing against `opt.noslip_tolerance`. Getting it wrong does not corrupt
# the sweep, it changes WHEN the sweep stops — measured on dm_control's dog,
# suppressing the early exit entirely moves a 120-step rollout by 2.2e-6 of
# qvel, so the stopping rule is not something to approximate.
comptime MODEL_META_IDX_MEANINERTIA: Int = 26
# Number of `<contact><pair>` records — MuJoCo's `npair`.
comptime MODEL_META_IDX_NPAIR: Int = 27
# `mjModel.opt.noslip_tolerance` — the improvement threshold `mj_solNoSlip`
# breaks on. Carried in META rather than as a comptime parameter because it is
# a plain runtime number the solver reads next to MEANINERTIA, and threading a
# Float64 through env -> integrator -> solver -> kernel would touch every
# caller for something that never needs to be known at compile time.
#
# ⚠ 0 IS A REAL SETTING, NOT "UNSET". dm_control's manipulation models use it
# to mean "run all `noslip_iterations`". Any consumer that treats a 0 here as
# "fall back to the default" reintroduces the truncation this slot fixes.
comptime MODEL_META_IDX_NOSLIP_TOLERANCE: Int = 28
# `mjModel.opt.ccd_tolerance` / `.ccd_iterations` — EPA's stopping rule.
# Defaults 1e-6 and 35 (`mjcPhysics/schema.usda`, and confirmed against the
# 3.10.0 runtime: `m.opt.ccd_tolerance == 1e-06`, `m.opt.ccd_iterations == 35`).
#
# ⚠⚠ THESE WERE HARDCODED AT 1e-8 AND 64, i.e. TIGHTER THAN MUJOCO'S, and a
# model that sets them was ignored outright. That is a parity gap in the
# direction that is easy to mistake for safety: EPA's stopping rule decides
# WHICH boundary face it settles on, and the contact NORMAL is that face's,
# so running longer than the reference does not converge toward it — it
# converges away from it. Measured on Jaco `reach_site_features` pose 38,
# ours at 1e-8 sits 9.4e-3 from MuJoCo's normal and at 1e-6 sits 7.6e-3.
#
# ⚠ MATCHING THE TOLERANCE IS NECESSARY, NOT SUFFICIENT. The polytope
# expansion, face ordering and horizon construction all differ from
# `engine_collision_gjk.c`'s, so identical stopping rules still stop on
# different faces. See `test_epa_optimality_cylinder_mesh`, which gates the
# quantity that IS well-posed.
#
# In META rather than as comptime parameters for the reason
# `MODEL_META_IDX_NOSLIP_TOLERANCE` gives above: plain runtime numbers, read
# next to the geoms they apply to, with no compile-time consumer.
comptime MODEL_META_IDX_CCD_TOLERANCE: Int = 29
comptime MODEL_META_IDX_CCD_ITERATIONS: Int = 30

# The ROOT `<default>`'s motor ctrlrange — the scalar action bounds an env
# advertises through `BoxContinuousActionEnv.action_low/action_high`.
#
# ⚠⚠ A SUMMARY, NOT THE CLAMP. `apply_actions` clamps each actuator to its own
# range; this pair only sizes the box a policy samples from, and it is
# knowingly wrong on models that set ranges per actuator or per default class
# (reach_site_features, quadruped). Kept bit-for-bit as it was — see
# `FlatModelDef.default_motor_ctrl_min`.
#
# Here rather than as comptime members of `ModelDefLike` because phase 1b
# removed the last comptime readers of the MJCF: these were
# `_xml_default_motor_ctrlrange[Self.xml]()`, and a comptime reader of the XML
# is exactly what pins a model to a `String` in Mojo source.
comptime MODEL_META_IDX_CTRL_MIN: Int = 31
comptime MODEL_META_IDX_CTRL_MAX: Int = 32

# `mjDSBL_MULTICCD` (1<<19) — non-zero when the model carries
# `<option><flag multiccd="disable"/></option>`.
#
# ⚠⚠ THE SENSE IS "DISABLED", NOT "ENABLED", AND THAT IS DELIBERATE. MuJoCo's
# multi-point convex manifold is ON by default on the 3.10.0 runtime, so 0 —
# what an unseeded slot and every pre-existing builder gives — is the correct
# default and leaves every model that does not set the flag untouched. Storing
# "enabled" here would have made a zeroed slot silently switch the feature off
# for every hand-made fixture and GPU env spec.
#
# ⚠ WHY IT NEEDS A SLOT AT ALL. `collision/multi_ccd.mojo` implemented the
# default-on behaviour UNCONDITIONALLY, so a model asking for single-point
# convex contacts got a 4-point manifold anyway. Measured on
# `manipulation/reassemble5`: ours 437 contacts against MuJoCo's 111, and 3701
# ms per control step against 13-49 ms. Every dm_control manipulation model
# sets this flag; 9 of the 11 baked ones carry it.
#
# ⚠ `nativeccd` IS PARSED TOO BUT HAS NO SLOT, because it has no consumer.
# `mjDSBL_NATIVECCD` only decides whether `mjc_Convex` takes its early return
# before the perturbation loop, and the native `multicontact()` polygon-clipping
# path that early return protects is NOT PORTED (see `multi_ccd.mojo`'s header).
# With multiccd disabled the loop does not run either way, so on every model in
# this tree the two flags agree. A model setting `nativeccd` ALONE — the baked
# `reach_site_features` and `lift_large_box` do — would route BOX/MESH and
# MESH/MESH pairs into the perturbation loop in MuJoCo while we still exclude
# them, a SMALLER divergence in the opposite direction. Recorded, not fixed.
comptime MODEL_META_IDX_MULTICCD_DISABLED: Int = 33

# MuJoCo's defaults, used wherever a Model is built without a parser (hand-made
# fixtures, the GPU env specs) so that those paths behave like the reference
# rather than like whatever the old constants happened to be.
comptime MJ_CCD_TOLERANCE: Float64 = 1e-6
comptime MJ_CCD_ITERATIONS: Int = 35


# =============================================================================
# Model Buffer Layout - Unified Geoms (body-attached + static)
# =============================================================================

comptime MODEL_GEOM_SIZE: Int = 31  # Per unified geom (+7 solref/solimp, +1 margin, +1 mesh_id, +1 priority)

comptime GEOM_IDX_TYPE: Int = 0
comptime GEOM_IDX_BODY: Int = 1  # Body index (-1 for static)
comptime GEOM_IDX_POS_X: Int = 2
comptime GEOM_IDX_POS_Y: Int = 3
comptime GEOM_IDX_POS_Z: Int = 4
comptime GEOM_IDX_QUAT_X: Int = 5
comptime GEOM_IDX_QUAT_Y: Int = 6
comptime GEOM_IDX_QUAT_Z: Int = 7
comptime GEOM_IDX_QUAT_W: Int = 8
comptime GEOM_IDX_RADIUS: Int = 9
comptime GEOM_IDX_HALF_LENGTH: Int = 10
comptime GEOM_IDX_HALF_X: Int = 11
comptime GEOM_IDX_HALF_Y: Int = 12
comptime GEOM_IDX_HALF_Z: Int = 13
comptime GEOM_IDX_FRICTION: Int = 14
comptime GEOM_IDX_CONTYPE: Int = 15
comptime GEOM_IDX_CONAFFINITY: Int = 16
comptime GEOM_IDX_CONDIM: Int = 17
comptime GEOM_IDX_FRICTION_SPIN: Int = 18
comptime GEOM_IDX_FRICTION_ROLL: Int = 19
comptime GEOM_IDX_RBOUND: Int = 20
comptime GEOM_IDX_SOLREF_0: Int = 21  # Per-geom solref timeconst
comptime GEOM_IDX_SOLREF_1: Int = 22  # Per-geom solref dampratio
comptime GEOM_IDX_SOLIMP_0: Int = 23  # Per-geom solimp dmin
comptime GEOM_IDX_SOLIMP_1: Int = 24  # Per-geom solimp dmax
comptime GEOM_IDX_SOLIMP_2: Int = 25  # Per-geom solimp width
comptime GEOM_IDX_SOLIMP_3: Int = 26  # Per-geom solimp midpoint
comptime GEOM_IDX_SOLIMP_4: Int = 27  # Per-geom solimp power
comptime GEOM_IDX_MARGIN: Int = 28  # Per-geom contact margin
comptime GEOM_IDX_MESH_ID: Int = 29  # Mesh hull index (-1 if not mesh)
# `<geom priority="...">`, default 0. When two geoms differ, the HIGHER
# priority one dictates condim, solref, solimp AND friction wholesale — no
# mixing at all (`engine_collision_driver.c:1427-1438`). dm_control's quadruped
# and dog are the only suite models that set it; quadruped's ball uses it to
# force its own `condim="6"` and `solref="-10000 -30"` onto every contact it
# takes part in, including against the floor.
comptime GEOM_IDX_PRIORITY: Int = 30


# =============================================================================
# Model Buffer Layout - Equality Constraints
# =============================================================================

comptime MODEL_EQ_SIZE: Int = 22  # Per equality constraint

# ⚠ THE SLOTS ARE REUSED PER TYPE, exactly as MuJoCo reuses `eq_obj1id` /
# `eq_obj2id` / `eq_data` for every `mjtEq`. Read the type FIRST:
#
#   EQ_CONNECT / EQ_WELD : BODY_A/BODY_B are BODY indices, ANCHOR_A* and
#                          ANCHOR_B* are `eq_data[0:3]` / `eq_data[3:6]`.
#   EQ_JOINT             : BODY_A/BODY_B are JOINT indices (BODY_B = -1 for
#                          the single-joint form), and ANCHOR_AX..ANCHOR_BY
#                          are `polycoef[0..4]` — MuJoCo's `eq_data[0:5]`,
#                          the same five floats in the same order.
#
# Naming them BODY_*/ANCHOR_* is a wart inherited from when connect and weld
# were the only types; the alternative was renaming them across nine files
# mid-arc. The rule that matters is: NEVER read these without branching on
# EQ_IDX_TYPE.
comptime EQ_IDX_TYPE: Int = 0  # EQ_CONNECT=0, EQ_WELD=1, EQ_JOINT=2 (mjtEq)
comptime EQ_IDX_BODY_A: Int = 1  # body index, or JOINT index when EQ_JOINT
comptime EQ_IDX_BODY_B: Int = 2  # -1 for world / for a single-joint EQ_JOINT
comptime EQ_IDX_ANCHOR_AX: Int = 3  # EQ_JOINT: polycoef[0]
comptime EQ_IDX_ANCHOR_AY: Int = 4  # EQ_JOINT: polycoef[1]
comptime EQ_IDX_ANCHOR_AZ: Int = 5  # EQ_JOINT: polycoef[2]
comptime EQ_IDX_ANCHOR_BX: Int = 6  # EQ_JOINT: polycoef[3]
comptime EQ_IDX_ANCHOR_BY: Int = 7  # EQ_JOINT: polycoef[4]
comptime EQ_IDX_ANCHOR_BZ: Int = 8
comptime EQ_IDX_RELPOSE_X: Int = 9
comptime EQ_IDX_RELPOSE_Y: Int = 10
comptime EQ_IDX_RELPOSE_Z: Int = 11
comptime EQ_IDX_RELPOSE_W: Int = 12
comptime EQ_IDX_SOLREF_0: Int = 13
comptime EQ_IDX_SOLREF_1: Int = 14
comptime EQ_IDX_SOLIMP_0: Int = 15
comptime EQ_IDX_SOLIMP_1: Int = 16
comptime EQ_IDX_SOLIMP_2: Int = 17
comptime EQ_IDX_SOLIMP_3: Int = 18  # solimp midpoint
comptime EQ_IDX_SOLIMP_4: Int = 19  # solimp power

# Weld only — MuJoCo's `eq_data[10]`, scaling the three ORIENTATION rows.
#
# ⚠ NOT COSMETIC AND NOT ALWAYS 1. MuJoCo applies it twice: to the orientation
# residual (`mju_scl3(cpos+3, quat2+1, torquescale)`) and to the rotational
# Jacobian (`mju_scl(jac+3*NV, ..., torquescale, 3*NV)`), so it scales the
# whole rotational half of the constraint. MJCF defaults it to 1, which is why
# ignoring it went unnoticed — but MetaWorld's `reset_mocap_welds` sets **5.0**,
# so sawyer's weld orientation was 5x too soft against the environment we port.
# Unimplemented until 2026-08-12.
comptime EQ_IDX_TORQUESCALE: Int = 20

# MuJoCo's `eq_objtype` — BODY or SITE semantics. MJCF lets both `connect` and
# `weld` name either two bodies (+ an `anchor` in body1's frame) or two SITES,
# and `mj_instantiateEquality` branches on it: the body form builds the anchor
# as `xmat[b]*eq_data + xpos[b]`, the site form reads `site_xpos` directly and
# ignores `eq_data` entirely (engine_core_constraint.c:448).
#
# WE STORE THE SITE FORM REDUCED TO THE BODY FORM: at parse time a site
# reference becomes `(body = site_bodyid, anchor = site local pos)`, which is
# exactly what `site_xpos` expands to in FK. That keeps the row builder and
# every solver path unchanged. The flag still has to be carried because the
# qpos0 derivation below must NOT run on the site form — MuJoCo zeroes
# `eq_data` there, and re-deriving would overwrite the site offsets with the
# anchor MuJoCo never computed.
comptime EQ_IDX_OBJTYPE: Int = 21  # EQ_OBJ_BODY=0 or EQ_OBJ_SITE=1


# =============================================================================
# Model Buffer Layout - Tendons
# =============================================================================
#
# Indices 0..16 are the ORIGINAL fixed-tendon record and keep their offsets;
# 17..35 were appended 2026-07-31 for dm_control's `ball_in_cup`, the first
# model with a SPATIAL (site-routed) tendon and the first with a tendon LIMIT.
# Same append-don't-renumber discipline the site record used for type+size.
#
# A fixed tendon uses NUM_JOINTS/JOINT_*/COEF_*; a spatial one uses
# NUM_SITES/SITE_*. KIND says which. The two halves are mutually exclusive.
#
# ⚠ IS_EQUALITY exists because `_tendon_env` treats every populated record as a
# BILATERAL EQUALITY (ten_length == LENGTH_REF). That was harmless only while
# `fields_build` hardcoded `ntendon = 0`. humanoid and humanoid_standup both
# declare <fixed> tendons that MuJoCo constrains in NO way, so honestly
# populating the count would have silently welded their hips together. Only
# <equality><tendon> sets this flag; `_tendon_env` skips rows without it.

# ⚠ THE WRAP CAP IS ONE CONSTANT AND EVERY OFFSET BELOW IS DERIVED FROM IT.
# It was 4, hardcoded in five places that had to agree: this layout, the loop
# bounds in `full_parser._fill_tendons`, `TendonData`'s three `InlineArray`s,
# the explicit `JOINT_0..3` writes in `fields_build`, and `TENDON_MAX_JOINTS`
# in `constraints/tendon_limit.mojo`. dog's `caudal_extend` wraps ELEVEN
# joints, so it was silently truncated to four on this path exactly as it was
# on the comptime path before defect 17.
#
# 16 matches `MAX_COMPTIME_TENDON_WRAPS`, deliberately: the two parsers must
# not disagree about how wide a tendon may be, and having them differ is its
# own class of bug (see `feedback_physics3d_two_parser_paths`).
#
# ⚠ THE WRAP SLOTS MUST STAY CONTIGUOUS. Consumers read
# `TENDON_IDX_JOINT_0 + k` / `TENDON_IDX_SITE_0 + k` in a loop
# (`tendon_limit.mojo`, `dynamics/tendon.mojo`), so appending new slots at the
# END of the record instead of widening in place would make k=4 silently read
# COEF_0. That is why this is a renumber rather than an append.
comptime TENDON_MAX_WRAPS: Int = 16

# ⚠⚠ A SEPARATE CAP FOR SPATIAL ROUTING, AND THE SPLIT IS THE POINT.
# `TENDON_MAX_WRAPS` was doing three jobs at once: how many joints a FIXED
# tendon may combine, how many waypoints a SPATIAL one may route through, and
# the stride of the ACTUATOR transmission arrays (`motor_trn_qadr` is sized
# `na * TENDON_MAX_WRAPS`). They shared a number, not a meaning.
#
# iit_softfoot routes a tendon through 39 waypoints — 21 sites and 18 wrap
# geoms — so the spatial cap has to be ~3x what it was. Raising the shared
# constant would have tripled ms_human_700's 700 actuator transmission rows
# for a quantity that has nothing to do with tendon routing. Measured across
# Menagerie's 881 spatial tendons: 726 use under 8 waypoints, 150 use 8-15,
# and 5 (softfoot's) use 32-39.
# ── what a spatial waypoint IS ───────────────────────────────────────────
# MuJoCo's `mjtWrap`, restricted to the kinds we route. Defined HERE rather
# than beside `mju_wrap` because the parser writes these values and the
# dynamics reads them: two spellings of one enum is how a wrap geom ends up
# read as a site.
comptime WRAP_NONE: Int = 0
comptime WRAP_SITE: Int = 1
comptime WRAP_SPHERE: Int = 2
comptime WRAP_CYLINDER: Int = 3
comptime WRAP_PULLEY: Int = 4

comptime TENDON_MAX_SPATIAL_WRAPS: Int = 48

# 24 scalar fields, two FIXED runs (joint, coef) and three SPATIAL runs
# (wrap object, wrap type, wrap parameter).
comptime MODEL_TENDON_SIZE: Int = (
    24 + 2 * TENDON_MAX_WRAPS + 3 * TENDON_MAX_SPATIAL_WRAPS
)

comptime TENDON_IDX_NUM_JOINTS: Int = 0
comptime TENDON_IDX_JOINT_0: Int = 1
comptime TENDON_IDX_JOINT_1: Int = TENDON_IDX_JOINT_0 + 1
comptime TENDON_IDX_JOINT_2: Int = TENDON_IDX_JOINT_0 + 2
comptime TENDON_IDX_JOINT_3: Int = TENDON_IDX_JOINT_0 + 3
comptime TENDON_IDX_COEF_0: Int = TENDON_IDX_JOINT_0 + TENDON_MAX_WRAPS
comptime TENDON_IDX_COEF_1: Int = TENDON_IDX_COEF_0 + 1
comptime TENDON_IDX_COEF_2: Int = TENDON_IDX_COEF_0 + 2
comptime TENDON_IDX_COEF_3: Int = TENDON_IDX_COEF_0 + 3
comptime TENDON_IDX_LENGTH_REF: Int = TENDON_IDX_COEF_0 + TENDON_MAX_WRAPS
comptime TENDON_IDX_SOLREF_0: Int = TENDON_IDX_LENGTH_REF + 1
comptime TENDON_IDX_SOLREF_1: Int = TENDON_IDX_LENGTH_REF + 2
comptime TENDON_IDX_SOLIMP_0: Int = TENDON_IDX_LENGTH_REF + 3
comptime TENDON_IDX_SOLIMP_1: Int = TENDON_IDX_LENGTH_REF + 4
comptime TENDON_IDX_SOLIMP_2: Int = TENDON_IDX_LENGTH_REF + 5
comptime TENDON_IDX_SOLIMP_3: Int = TENDON_IDX_LENGTH_REF + 6  # solimp midpoint
comptime TENDON_IDX_SOLIMP_4: Int = TENDON_IDX_LENGTH_REF + 7  # solimp power

# --- appended 2026-07-31 (spatial routing + limits) --------------------------

comptime TENDON_KIND_FIXED: Int = 0
comptime TENDON_KIND_SPATIAL: Int = 1

comptime TENDON_IDX_KIND: Int = TENDON_IDX_SOLIMP_4 + 1  # TENDON_KIND_*
comptime TENDON_IDX_IS_EQUALITY: Int = TENDON_IDX_KIND + 1  # 1 => `_tendon_env` owns this row
comptime TENDON_IDX_NUM_WRAPS: Int = TENDON_IDX_KIND + 2  # spatial only

# ── the spatial routing sequence, three parallel runs ─────────────────────
# MuJoCo's `wrap_type` / `wrap_objid` / `wrap_prm`, flattened into the tendon
# record. Entry `k` is a SITE (`WRAP_SITE`, obj = site id, prm unused) or a
# WRAP GEOM (`WRAP_SPHERE`/`WRAP_CYLINDER`, obj = geom id, prm = the sidesite
# id or -1).
#
# ⚠ THE TYPE RUN IS NOT REDUNDANT WITH THE OBJECT RUN. A site id and a geom
# id are both non-negative integers indexing different tables; without the
# type, entry `k` reads as a site whose position happens to be a geom's, and
# the tendon quietly routes through the wrong point rather than around the
# object. The previous layout had no type run because everything was a site.
comptime TENDON_IDX_WOBJ_0: Int = TENDON_IDX_KIND + 3
comptime TENDON_IDX_WTYPE_0: Int = (
    TENDON_IDX_WOBJ_0 + TENDON_MAX_SPATIAL_WRAPS
)
comptime TENDON_IDX_WPRM_0: Int = (
    TENDON_IDX_WTYPE_0 + TENDON_MAX_SPATIAL_WRAPS
)
comptime TENDON_IDX_LIMITED: Int = (
    TENDON_IDX_WPRM_0 + TENDON_MAX_SPATIAL_WRAPS
)
comptime TENDON_IDX_RANGE_MIN: Int = TENDON_IDX_LIMITED + 1
comptime TENDON_IDX_RANGE_MAX: Int = TENDON_IDX_LIMITED + 2
comptime TENDON_IDX_MARGIN: Int = TENDON_IDX_LIMITED + 3
# J M^-1 J^T at qpos0 — the limit row's diagApprox (engine_setconst.c:256).
comptime TENDON_IDX_INVWEIGHT0: Int = TENDON_IDX_LIMITED + 4
# The LIMIT solref/solimp pair, distinct from the equality pair above
# (MuJoCo keeps tendon_solref_lim separate from tendon_solref_fri).
comptime TENDON_IDX_SOLREF_LIM_0: Int = TENDON_IDX_LIMITED + 5
comptime TENDON_IDX_SOLREF_LIM_1: Int = TENDON_IDX_LIMITED + 6
comptime TENDON_IDX_SOLIMP_LIM_0: Int = TENDON_IDX_LIMITED + 7
comptime TENDON_IDX_SOLIMP_LIM_1: Int = TENDON_IDX_LIMITED + 8
comptime TENDON_IDX_SOLIMP_LIM_2: Int = TENDON_IDX_LIMITED + 9
comptime TENDON_IDX_SOLIMP_LIM_3: Int = TENDON_IDX_LIMITED + 10
comptime TENDON_IDX_SOLIMP_LIM_4: Int = TENDON_IDX_LIMITED + 11

# ⚠ `TENDON_MAX_SITES` IS GONE, not renamed: it was an alias of
# `TENDON_MAX_WRAPS` from when a waypoint could only be a site. The spatial
# cap is `TENDON_MAX_SPATIAL_WRAPS` above and it is a DIFFERENT NUMBER now.


# =============================================================================
# Model Buffer Layout - Curriculum Parameters
# =============================================================================

# Fixed-size curriculum section (environments use what they need)
comptime MODEL_CURRICULUM_SIZE: Int = 8  # Up to 8 curriculum parameters

# Generic curriculum parameter indices (environments define their own semantics)
comptime CURRICULUM_IDX_PARAM_0: Int = 0
comptime CURRICULUM_IDX_PARAM_1: Int = 1
comptime CURRICULUM_IDX_PARAM_2: Int = 2
comptime CURRICULUM_IDX_PARAM_3: Int = 3
comptime CURRICULUM_IDX_PARAM_4: Int = 4
comptime CURRICULUM_IDX_PARAM_5: Int = 5
comptime CURRICULUM_IDX_PARAM_6: Int = 6
comptime CURRICULUM_IDX_PARAM_7: Int = 7


# =============================================================================
# Model Buffer Layout - Sites
# =============================================================================

# Site layout: [body_idx, pos(3), type, size(3), quat(4)]
#
# type + size were appended 2026-07-29 for the `touch` sensor, which needs the
# site's ZONE (MuJoCo casts a ray from each contact point along the contact
# normal and asks whether it hits the site volume). `SiteData` carried both all
# along; only the serialized record was truncated. Appending keeps every
# existing `SITE_IDX_*` offset put.
#
# quat was appended 2026-08-01 for manipulator, whose `thumb_touch` /
# `finger_touch` zones are BOXES carrying `euler="0 15 0"`. A box zone is
# orientation-dependent, so the sphere-only scope that let the record ship
# without an orientation ended there. Three files had been substituting the
# site's BODY quaternion in the meantime and saying so in their docstrings —
# `sensors/touch.mojo`, `sensors/frame_vel.mojo`, `sensors/site_acc.mojo`.
# Stored (x, y, z, w), the order `BODY_IDX_QUAT_*` and `GEOM_IDX_QUAT_*`
# already use — MuJoCo's own `site_quat` is (w, x, y, z), so a parity test
# reading both has to reorder.
comptime MODEL_SITE_SIZE: Int = 12  # body + pos(3) + type + size(3) + quat(4)

comptime SITE_IDX_BODY: Int = 0  # Body index the site is attached to
comptime SITE_IDX_POS_X: Int = 1  # Local position in body frame
comptime SITE_IDX_POS_Y: Int = 2
comptime SITE_IDX_POS_Z: Int = 3
comptime SITE_IDX_TYPE: Int = 4  # GEOM_* code (sphere/capsule/box/...)
comptime SITE_IDX_SIZE_0: Int = 5  # radius, or half-x for a box
comptime SITE_IDX_SIZE_1: Int = 6  # half-length, or half-y
comptime SITE_IDX_SIZE_2: Int = 7  # half-z (box only)
comptime SITE_IDX_QUAT_X: Int = 8  # Local orientation in body frame
comptime SITE_IDX_QUAT_Y: Int = 9
comptime SITE_IDX_QUAT_Z: Int = 10
comptime SITE_IDX_QUAT_W: Int = 11


comptime MODEL_EXCLUDE_PAIR_SIZE: Int = 2  # body1, body2


# =============================================================================
# Model Buffer Layout - Predefined contact pairs (`<contact><pair>`)
# =============================================================================
#
# One record per `<contact><pair>`, mirroring MuJoCo's `m->pair_*` arrays.
#
# ⚠ A PREDEFINED PAIR IS NOT A FILTERED PAIR. It collides UNCONDITIONALLY:
# `mj_collideGeoms` skips the contype/conaffinity test whenever `ipair >= 0`
# (`engine_collision_driver.c:1583`), and the whole merge loop runs BEFORE the
# `canCollide2` / `exclude_signature` tests at `:398-412`. Confirmed against the
# 3.10.0 runtime, which emits the contact for every one of: masks cleared to
# `contype=0 conaffinity=0`, an `<exclude>` naming both bodies, two geoms on the
# SAME body, and a welded parent/child. So the pair path must bypass
# `pair_body_filtered` AND the mask AND the plane-vs-world skips — not just one
# of them.
#
# ⚠ THE PARAMETERS ARE PLAIN DEFAULTS, NOT DERIVED FROM THE TWO GEOMS.
# `mjCPair::Compile` (`user/user_objects.cc`) reads as though an omitted
# attribute is filled in from the geoms — max margin, max gap, max condim, max
# friction, solmix-weighted solref/solimp. That code is DEAD on the XML path:
# `mjs_defaultPair` (`user/user_init.c`) memsets the spec and writes concrete
# defaults (condim 3, friction 1/1/0.005/1e-4/1e-4, `mj_defaultSolRefImp`), so
# `mjuu_defined()` is true for every field and no derivation branch is ever
# taken. Measured on 3.10.0 with two geoms deliberately given DIFFERENT solref,
# friction, margin and condim:
#
#     dynamic geom pair -> condim 6, friction 1.5,  solref 0.0125  (mixed)
#     <pair> no attrs   -> condim 3, friction 1.0,  solref 0.02    (defaults)
#
# Transcribing `mjCPair::Compile` would therefore have silently given every
# attribute-less pair the WRONG friction and condim. ToddlerBot's `scene*.xml`
# pairs carry `geom1`/`geom2` and nothing else, so this is the path that matters
# and the error would have been invisible in the geometry.
#
# ⚠ `gap` IS NOT STORED. `mj_setContact` is called with `margin-gap` in every
# reference tree here (3.3.6, 3.6.0, main), but the 3.10.0 runtime reports
# `includemargin == margin` for a pair with `margin=.05 gap=.02`, and 3.11.0
# filters on `margin + gap` instead — three different behaviours across the
# versions. This engine models no gap anywhere (there is no `GEOM_IDX_GAP`), so
# `full_parser` REJECTS a non-zero `gap` on a pair rather than pick one of the
# three and silently drop it, the same way it rejects a non-default `solmix`.
comptime MODEL_PAIR_SIZE: Int = 14

comptime PAIR_IDX_GEOM1: Int = 0  # Geom index (compiler-sorted, g1 < g2)
comptime PAIR_IDX_GEOM2: Int = 1
comptime PAIR_IDX_CONDIM: Int = 2
comptime PAIR_IDX_FRICTION: Int = 3  # Sliding (MuJoCo pair_friction[0..1])
comptime PAIR_IDX_FRICTION_SPIN: Int = 4  # Torsional (pair_friction[2])
comptime PAIR_IDX_FRICTION_ROLL: Int = 5  # Rolling (pair_friction[3..4])
comptime PAIR_IDX_SOLREF_0: Int = 6
comptime PAIR_IDX_SOLREF_1: Int = 7
comptime PAIR_IDX_SOLIMP_0: Int = 8
comptime PAIR_IDX_SOLIMP_1: Int = 9
comptime PAIR_IDX_SOLIMP_2: Int = 10
comptime PAIR_IDX_SOLIMP_3: Int = 11
comptime PAIR_IDX_SOLIMP_4: Int = 12
comptime PAIR_IDX_MARGIN: Int = 13


# =============================================================================
# Model Buffer Layout - Actuation (phase 1a.2)
# =============================================================================

# The runtime replacement for `ComptimeActData`'s actuator arrays — the record
# layout behind `fields/spec_fields.mojo::SpecFields`. Everything BOTH
# `ModelDefFromXML.apply_actions` (CPU) and `apply_actions_kernel_gpu` read.
#
# ⚠ THESE ARE DELIBERATELY NOT PART OF `fields.Model`. `Model` is the operand
# bundle the integrator / solver / collision kernels bind; actuation is read by
# exactly one function per target and by nothing else. Widening `Model` would
# also have added a fifteenth type parameter to a struct named in 48 files,
# every one of which would have had to thread `NACT` through to keep compiling.
#
# ⚠ THE WRAP STRIDE IS `TENDON_MAX_WRAPS`, SHARED WITH THE TENDON RECORD AND
# WITH `FlatModelDef.motor_trn_*` (which is indexed `ai * TENDON_MAX_WRAPS + k`
# already). The comptime twin uses `_WRAPS`, which collapses to 1 on a model
# with no tendons — so the two strides AGREE ONLY WHEN THE MODEL HAS TENDONS.
# Anything diffing the two must convert; the equivalence gate does.
comptime MODEL_ACTUATOR_SIZE: Int = 16 + 3 * TENDON_MAX_WRAPS

comptime ACT_IDX_KIND: Int = 0  # ACT_KIND_*
comptime ACT_IDX_GEAR: Int = 1
comptime ACT_IDX_CTRL_MIN: Int = 2
comptime ACT_IDX_CTRL_MAX: Int = 3
# ⚠ READ THIS BEFORE READING THE RANGE. MuJoCo's `ctrllimited` defaults to
# "auto", so an actuator declaring no range is UNLIMITED and the stored range
# is a (-1, 1) fallback nobody should clamp to.
comptime ACT_IDX_CTRL_LIMITED: Int = 4
comptime ACT_IDX_FORCE_MIN: Int = 5
comptime ACT_IDX_FORCE_MAX: Int = 6
comptime ACT_IDX_FORCE_LIMITED: Int = 7
# MuJoCo `gainprm[0]` and `-biasprm[2]`. INDEPENDENT — `<velocity>` happens to
# set both to K, but `gainprm="5 0 0" biasprm="0 0 -3"` is legal.
# ⚠ `kp` DEFAULTS TO 1, NOT 0: a plain `<motor>` never writes it and its force
# is `kp * ctrl`, so a zero here silently disables every bare motor.
comptime ACT_IDX_KP: Int = 8
comptime ACT_IDX_KV: Int = 9
# mjDYN_FILTER: `act_dot = (ctrl - act) / dyn_tau`. `act_adr >= 0` means this
# actuator owns one activation variable at that index; -1 means none.
comptime ACT_IDX_DYN_TAU: Int = 10
comptime ACT_IDX_ACT_ADR: Int = 11
comptime ACT_IDX_TRN_N: Int = 12  # 0 => no resolvable transmission, skip
comptime ACT_IDX_DOF_ADR: Int = 13  # the single dof the actuator reports on
comptime ACT_IDX_TENDON_ID: Int = 14  # -1 unless a `tendon=` transmission
comptime ACT_IDX_JOINT_ID: Int = 15  # -1 unless a `joint=` transmission
# The transmission triples, `+ k` for k in [0, TRN_N). A `joint=` actuator is
# ONE triple with coef 1; a `tendon=` one copies the tendon's whole wrap list.
comptime ACT_IDX_TRN_QADR_0: Int = 16
comptime ACT_IDX_TRN_DADR_0: Int = ACT_IDX_TRN_QADR_0 + TENDON_MAX_WRAPS
comptime ACT_IDX_TRN_COEF_0: Int = ACT_IDX_TRN_QADR_0 + 2 * TENDON_MAX_WRAPS

# The tendon SPRING half of actuation (`engine_passive.c`), kept in its own
# record rather than folded into `MODEL_TENDON_SIZE`.
#
# ⚠ SEPARATE FROM `Model.tendons` ON PURPOSE. That record stores wraps as
# JOINT IDS (`TENDON_IDX_JOINT_0 + k`) because its consumers — `tendon_limit`,
# `dynamics/tendon` — want joints. The spring path wants qpos/dof ADDRESSES,
# and resolving one to the other inside the actuation kernel would mean binding
# `Model.joints` as a second operand purely to do a lookup the parser already
# did. Storing the addresses is what the comptime twin does
# (`tendon_trn_qadr` / `_dadr`) and this mirrors it.
comptime MODEL_ACT_TENDON_SIZE: Int = 4 + 3 * TENDON_MAX_WRAPS

comptime ACTTEN_IDX_STIFFNESS: Int = 0  # 0 => no spring, skip the row
# The deadband bounds. ⚠ WHEN `springlength` IS ABSENT BOTH DEFAULT TO the
# tendon's rest length `sum(coef * joint.ref)`, NOT to zero.
comptime ACTTEN_IDX_SPRING_LO: Int = 1
comptime ACTTEN_IDX_SPRING_HI: Int = 2
comptime ACTTEN_IDX_TRN_N: Int = 3
comptime ACTTEN_IDX_TRN_QADR_0: Int = 4
comptime ACTTEN_IDX_TRN_DADR_0: Int = ACTTEN_IDX_TRN_QADR_0 + TENDON_MAX_WRAPS
comptime ACTTEN_IDX_TRN_COEF_0: Int = (
    ACTTEN_IDX_TRN_QADR_0 + 2 * TENDON_MAX_WRAPS
)

# --- reference pose + keyframes (phase 1a.4) ---------------------------------
#
# `qpos0` itself is a bare `[NQ]` tensor; these are the two scalars that go
# with it. ⚠ `FREE_JOINT_QPOS_ADR` IS -1 WHEN ABSENT AND ZERO IS A VALID
# ADDRESS, so the record is seeded rather than left as `alloc` wrote it.
comptime POSE_META_SIZE: Int = 2
comptime POSE_IDX_QPOS0_NQ: Int = 0
comptime POSE_IDX_FREE_JOINT_QPOS_ADR: Int = 1

# One row per `<keyframe><key>`. ⚠ `NQPOS`/`NQVEL`/`NCTRL` ARE PRESENCE FLAGS
# AS MUCH AS LENGTHS: MuJoCo fills an absent `qpos=` from qpos0 and an absent
# `qvel=`/`ctrl=` with zero, so `key_qpos_at` must know the attribute was
# missing rather than read a row of zeros as a real pose. `init_fields`
# already refuses any length other than the full one.
comptime KEY_META_SIZE: Int = 4
comptime KEY_IDX_TIME: Int = 0
comptime KEY_IDX_NQPOS: Int = 1
comptime KEY_IDX_NQVEL: Int = 2
comptime KEY_IDX_NCTRL: Int = 3

# --- joint limits, for the `enforce_limits` clamp (phase 1a.4) ---------------
#
# ⚠ NOT THE SAME THING AS THE SOLVER'S LIMIT ROWS. `Model.joints` carries
# `JOINT_IDX_RANGE_MIN/MAX` and the per-joint solref/solimp that the
# CONSTRAINT path uses; this record is the hard clamp `enforce_limits` applies
# to `qpos` directly, and it needs one thing the joint record does not have: a
# LIMITED flag. `range_min < range_max` is not that test — MuJoCo spells an
# unlimited joint BOTH as `[0, 0]` and as `[-1e10, 1e10]`, and the second
# satisfies it.
comptime JLIM_SIZE: Int = 8
comptime JLIM_IDX_LIMITED: Int = 0
comptime JLIM_IDX_QPOS_ADR: Int = 1
comptime JLIM_IDX_RANGE_MIN: Int = 2
comptime JLIM_IDX_RANGE_MAX: Int = 3
# --- `<joint actuatorfrcrange>` — MuJoCo's `jnt_actfrcrange` ----------------
#
# ⚠⚠ A SECOND, INDEPENDENT FORCE LIMIT, AND IT IS NOT THE ACTUATOR'S.
# `mj_fwdActuation` clamps TWICE: `actuator_forcerange` on each actuator's own
# scalar force (engine_forward.c:417), and then
#
#     clampVec(d->qfrc_actuator, m->jnt_actfrcrange, m->jnt_actfrclimited,
#              m->njnt, m->jnt_dofadr);                            // :477
#
# on the ACCUMULATED `qfrc_actuator`, per JOINT, at that joint's dof address.
# The two are unrelated: on unitree_g1 `actuator_forcelimited` is FALSE on all
# 29 actuators while `jnt_actfrclimited` is TRUE on 29 of 30 joints, so the
# joint-level clamp is the ONLY force limit the model has. 481 of this tree's
# 2519 joints declare one, across 20 robots — and the tightest are tiny: g1's
# wrists are +-5 N.m against a `kp=500` servo, sharpa_wave's are +-0.19.
#
# ⚠ IT LIVES HERE, NOT ON `Model.joints`, because `apply_actions_fields` is
# handed `sf` and not the model — and because this record already carries the
# LIMITED flag idiom that the joint record deliberately does not have.
# `JLIM_IDX_DOF_ADR` is what `jnt_dofadr` is in the clampVec call above; the
# existing `QPOS_ADR` column is the wrong address for it.
comptime JLIM_IDX_DOF_ADR: Int = 4
comptime JLIM_IDX_ACTFRC_LIMITED: Int = 5
comptime JLIM_IDX_ACTFRC_MIN: Int = 6
comptime JLIM_IDX_ACTFRC_MAX: Int = 7


# =============================================================================
# Model Buffer Layout - Mesh Collision Hull Data
# =============================================================================

# Mesh hull vertices stored AFTER exclude pairs in the model buffer.
# Layout: [mesh_meta(NMESH*2)] [mesh_verts(total_verts*3)]
# mesh_meta: [vertadr, vertnum] per mesh
# mesh_verts: flattened [x0,y0,z0, x1,y1,z1, ...] in local frame
comptime MAX_HULL_VERTS_PER_MESH: Int = 256
"""⚠⚠ DEAD, AND MISLEADING IF READ AS LIVE. Nothing references this. Hulls are
NOT capped per mesh — so_arm100 loads one of 2094 vertices — and the total is
bounded by `dims.get_nmesh_verts()`, a runtime budget. Kept only because
deleting a public constant is a separate change; do not reintroduce a use."""

comptime MAX_GPU_MESHES: Int = 256
"""How many COLLIDABLE meshes one model may have. Was 16.

⚠ IT SIZES ONE TABLE AND NOTHING ELSE. `mesh_meta` is `[MAX_GPU_MESHES, 4]` —
8 KB at 256 and float64, against 512 bytes at 16 — and every other use is a
`Layout` over that table. No `InlineArray` is keyed on it, so raising it costs
memory and nothing else. 16 was never a hardware limit; it was a guess that
predates mesh-heavy models.

⚠⚠ AND EXCEEDING IT NOW RAISES. It used to print `ERROR:` and continue, which
leaves meshes past the cap with a hull built and an id assigned but NO
`mesh_meta` row — so every consumer reads vertadr/vertnum 0 and collides
against an EMPTY mesh. That is wrong physics that runs, and this tree has paid
for the same shape twice (the 16-asset `<mesh>` cap cost SO-ARM100 two
collision surfaces, found only by diffing `rbound` against MuJoCo). A model
that will not open is a better outcome than one that opens and is wrong.

Headroom, measured 2026-08-19: ToddlerBot 2, so_arm100 8, sawyer 10 — the
mesh-heaviest models in or near this tree. See
`tests/physics3d/test_mesh_cap_is_loud.mojo`."""
# vertadr, vertnum, polyadr, polynum per mesh — the last two added with the
# native multi-contact path, mirroring `mesh_polyadr` / `mesh_polynum`.
comptime MODEL_MESH_META_SIZE: Int = 4
comptime MESH_META_IDX_VERTADR: Int = 0
comptime MESH_META_IDX_VERTNUM: Int = 1
comptime MESH_META_IDX_POLYADR: Int = 2
comptime MESH_META_IDX_POLYNUM: Int = 3

# ---- Mesh POLYGON topology (native multi-contact) ---------------------------
#
# `multicontact` (`engine_collision_gjk.c:2111`) recovers the face a contact
# came from and clips it against the opposing face. For a mesh that needs the
# hull's polygons, which `collision/mesh_polygons.mojo` builds at model load.
#
# ⚠ THE CAPS ARE EULER'S FORMULA, NOT A GUESS, which is why none of these need
# to become `Model` type parameters. For a convex polyhedron with V vertices a
# TRIANGULATED hull has at most F = 2V - 4 faces and E = 3V - 6 edges; merging
# coplanar triangles into polygons only ever REDUCES F, and the total number of
# polygon-vertex incidences is exactly 2E <= 6V - 12. The vertex -> polygon map
# holds the same 2E entries. So sizing off NMESH_VERTS is exact, not generous,
# and a mesh cannot overflow these without violating convexity.
comptime MODEL_MESH_POLY_SIZE: Int = 5  # vertadr, vertnum, nx, ny, nz
comptime MESH_POLY_IDX_VERTADR: Int = 0
comptime MESH_POLY_IDX_VERTNUM: Int = 1
comptime MESH_POLY_IDX_NX: Int = 2
comptime MESH_POLY_IDX_NY: Int = 3
comptime MESH_POLY_IDX_NZ: Int = 4


def mesh_max_poly(nmesh_verts: Int) -> Int:
    """Polygon capacity for a hull budget of `nmesh_verts` vertices."""
    return 2 * nmesh_verts if nmesh_verts > 0 else 1


def mesh_max_polyvert(nmesh_verts: Int) -> Int:
    """Polygon-vertex (and vertex->polygon map) capacity. See above."""
    return 6 * nmesh_verts if nmesh_verts > 0 else 1


def mesh_max_edge(nmesh_verts: Int) -> Int:
    """Hull edge-graph capacity for a budget of `nmesh_verts` vertices.

    MuJoCo sizes the same block at `numvert + 3*numface`; a triangulated
    polytope has `F = 2V - 4`, so that is `7V - 12`. 8V leaves headroom rather
    than trusting the identity, and `fields_build` still raises on overflow.
    """
    return 8 * nmesh_verts if nmesh_verts > 0 else 1


# =============================================================================
# Workspace Buffer Layout (per-environment scratch space for GPU kernels)
# =============================================================================
# Moves all integrator temporaries and solver arrays from InlineArrays
# (register pressure) to device memory.
#
# Layout per environment:
#   [integrator_temps | M_inv: NV*NV | solver workspace: SOLVER.solver_workspace_size()]
#
# Integrator temps section:
#   [cdof: NV*6 | crb: NBODY*10 | M: NV*NV | L: NV*NV | D: NV |
#    bias: NV | f_net: NV | qacc_ws: NV | qacc_constrained: NV]


# =============================================================================
# Implicit Integrator Extra Workspace
# =============================================================================
# Additional workspace for the full implicit integrator's RNE velocity
# derivative computation. Placed AFTER solver workspace so existing offsets
# are unchanged.
#
# Layout within implicit extra section:
#   [qDeriv: NV*NV | cdof_origin: NV*6 | cvel_origin: NBODY*6 |
#    cinert: NBODY*10 | cdof_dot: NV*6 |
#    Dcvel: NBODY*6*NV | Dcdofdot: NV*6*NV |
#    Dcacc: NBODY*6*NV | Dcfrcbody: NBODY*6*NV]


def rk4_extra_workspace_size[NQ: Int, NV: Int]() -> Int:
    """Total RK4-extra workspace size per environment."""
    return NQ + 7 * NV



