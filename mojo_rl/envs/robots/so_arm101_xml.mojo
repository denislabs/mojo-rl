"""SO-ARM101 (5 DOF + gripper) — model definition, `new_calib` variant.

Reference: `references/SO-ARM100-main/Simulation/SO101/`, vendored from
https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101 — the
model is in The Robot Studio's OWN repository, not in Menagerie and not in
`johnsutor/so101-nexus`. Generated upstream by `onshape-to-robot`.

Measured against MuJoCo 3.10.0: nbody 8, njnt 6 (all hinge), nq/nv/nu 6,
ngeom 31 (30 mesh, 1 plane), nsite 2, neq/ntendon/npair/nexclude/nkey 0,
cone pyramidal, impratio 1.

`so101_new_calib` is upstream's default: each joint's virtual zero is the
MIDDLE of its range. `so101_old_calib` puts zero at the fully-extended
horizontal pose. Only `new` is ported; `so_arm_bake.py` takes a `calib`
argument if the other is ever wanted.

RELATIONSHIP TO SO-100 — same robot, and NOT interchangeable. Same topology
(nbody 8 / njnt 6 / nq 6 / nu 6), and the two long links are BIT-IDENTICAL
(shoulder->elbow 0.116 m, elbow->wrist 0.135 m). But the base mount is 4 cm
shorter, and the inertials differ: **moving mass 0.485 kg here vs 0.609 kg**
for SO-100, with the moving jaw 68% apart. Under a stiff position servo a
policy will not notice; for torque-level sim-to-real it is the difference that
matters. ⚠ `qpos` is NOT portable between the two — SO-100 uses per-joint axes
`(0,1,0)/(1,0,0)/(0,0,1)`, SO-101 puts every axis on `(0,0,1)` and absorbs the
rest into body quaternions. Any cross-model comparison needs an explicit joint
mapping, not an array copy. See `docs/SO_ARM101_PORT_ASSESSMENT.md` §3.

⚠⚠ THE `fullinertia` DEVIATION IS THE ONE THAT MATTERS HERE. All seven bodies
spell their inertia as `<inertial ... fullinertia="ixx iyy izz ixy ixz iyz"/>`,
which `full_parser.mojo` RAISES on — honestly, and on both parser paths, since
`ModelDefFromXML.init_fields` goes through `parse_xml_full`. Rather than block
this port on a parser feature that belongs to ToddlerBot's phase 1a
(`docs/TODDLERBOT_PORT_PLAN.md` §4.5), `tests/robots/so_arm_bake.py`
diagonalises each tensor WITH MUJOCO and emits `quat` + `diaginertia` at 17
significant digits — the dog mesh-inertia bake precedent.

⚠ Gate the QUATERNION, not just the moments. A wrong `iquat` with a correct
`diaginertia` leaves total mass and every scalar moment right while silently
rotating each body's inertia frame. `so_arm_ref.py` diffs `body_iquat` AND
`body_inertia` per body at tolerance 0.0 for exactly that reason — §32.9 of
`docs/DM_CONTROL_PORT_PHASE2.md` is the precedent, where eigenvalues were
already right on 6 of 9 meshes while the frame was a different valid one.

⚠ WHEN `fullinertia` LANDS, switch `so_arm_bake.py` to emit the raw spelling
and keep the baked values as a REGRESSION FIXTURE — 7 near-symmetric robot
links are a much better probe of `mjuu_eig3`'s tie-breaking than a synthetic
tensor, and near-symmetric is exactly where those details bite.

⚠ THE TASK BODY IS INLINED BY THE BAKE, NOT BY `merge_mjcf`. The bug that
prompted this is FIXED (2026-08-13, comments are stripped before scanning);
direct emission is kept only because it is fewer moving parts. The reason
first recorded here was WRONG. That call did mangle this model (`<default>`
vanished; MuJoCo rejected it with "unknown default class name 'sts3215'"), but
NOT because the defaults are nested. `_extract_section_inner` depth-counts raw
text without stripping comments, and the comment the bake inserted contained
the literal `<default>` — that alone deleted the section. Measured: the same
fixture with an angle-bracket-free comment merges fine, and a CLEAN nested
model merges fine too. ⚠ Do not inherit "merge_mjcf cannot do nested defaults"
from this file; it can. Direct emission is kept anyway, as one less dependency
on a function with three recorded silent section drops. Full analysis:
`docs/PHYSICS3D_PARSER_GAPS_2026_08_13.md` §3.

⚠⚠ COLLISION IS 10x SO-100'S AND BUYS NOTHING PHYSICAL. Ten collidable meshes
totalling **26 198 convex-hull vertices** (raw 136 832) against SO-100's
2 456 — because upstream uses the RAW VISUAL MESHES as collision geometry.
Upstream's own README records that this behaved badly enough that they deleted
the base collision meshes outright. `mesh_max_poly/polyvert/edge` are linear
(2V/6V/8V) so this is compile time and support-function cost, not memory. If it
proves unworkable, the documented third option is grafting SO-100's
hand-authored collision onto these kinematics — label it a deviation if taken.
"""

from mojo_rl.physics3d.parser import ModelDefFromXML
from mojo_rl.physics3d.types import ConeType
from mojo_rl.envs.robots.so_arm101_dims import SO_ARM101_DIMS


# --- BEGIN GENERATED XML (tests/robots/so_arm_bake.py) ---

# --- END GENERATED XML ---


comptime _pm = SO_ARM101_DIMS

comptime SoArm101Model = ModelDefFromXML[
    xml_path="mojo_rl/envs/robots/assets/so_arm101.xml",
    nbody=_pm.NBODY,
    njoint=_pm.NJOINT,
    nq=_pm.NQ,
    nv=_pm.NV,
    ngeom=_pm.NGEOM,
    nact=_pm.NACT,
    ntex=_pm.NTEX,
    nmat=_pm.NMAT,
    nlight=_pm.NLIGHT,
    ncam=_pm.NCAM,
    nsite=_pm.NSITE,
    neq=_pm.NEQ,
    # ⚠⚠ `nexclude` AND `npair` DEFAULT TO 0, AND THE DROP IS SILENT. Omitting
    # `nexclude` here left SO-100's `<exclude body1="Base"
    # body2="Rotation_Pitch"/>` unbuilt — `parse_xml` reported NEXCLUDE 1 while
    # the MODEL carried 0, so the two adjacent base geoms would have collided
    # with each other forever. Caught by printing the model's counts against
    # `parse_xml`'s, which is why `test_so_arm10x_vs_mujoco` asserts BOTH.
    nexclude=_pm.NEXCLUDE,
    npair=_pm.NPAIR,
    timestep=_pm.TIMESTEP,
    # ⚠ PYRAMIDAL, unlike SO-100. Upstream sets no `<option cone>`, so this is
    # MuJoCo's default and `opt.cone` is gated at layer 1. Do not "align" the
    # two arms by making this elliptic — that would be a silent model change.
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=16,
    obs_dim_override=SO_ARM101_OBS_DIM,
    action_dim_override=6,
]

# Body indices, worldbody DFS order with world at 0.
comptime BASE_BODY_IDX: Int = 1
comptime SHOULDER_BODY_IDX: Int = 2
comptime UPPER_ARM_BODY_IDX: Int = 3
comptime LOWER_ARM_BODY_IDX: Int = 4
comptime WRIST_BODY_IDX: Int = 5
comptime GRIPPER_BODY_IDX: Int = 6
comptime MOVING_JAW_BODY_IDX: Int = 7
comptime TARGET_BODY_IDX: Int = 8

# qpos/qvel addresses. Six hinges in XML order.
comptime SHOULDER_PAN_ADR: Int = 0
comptime SHOULDER_LIFT_ADR: Int = 1
comptime ELBOW_FLEX_ADR: Int = 2
comptime WRIST_FLEX_ADR: Int = 3
comptime WRIST_ROLL_ADR: Int = 4
comptime GRIPPER_ADR: Int = 5

# Same layout as SO-100's, so one config shape serves both.
comptime SO_ARM101_OBS_DIM: Int = 27

# ⚠⚠ SIZED FROM **OUR** HULL: `fields_build` needs **32 934** vertices where
# MuJoCo's `mesh_graph` totals 26 198 — ours keeps 26% more. 33 280 is that
# rounded to a multiple of 512. A budget copied from `mjModel` raises at env
# construction; see `so_arm100_xml.mojo` for how to read the exact figure.
#
# ⚠ This is 13x SO-100's 2 551 and is the one place the two models genuinely
# diverge in cost — see the module docstring.
comptime SO_ARM101_NMESH_VERTS: Int = 33280
