"""dm_control `humanoid_CMU` parity: our envs vs MuJoCo + the reference tasks.

Three tasks (stand, walk, run) over `suite/humanoid_CMU.xml` verbatim. The task
class is `humanoid`'s — `diff humanoid.py humanoid_CMU.py` over `get_reward`
reports the two functions IDENTICAL — so what this file gates is the model and
the accessors underneath, not the reward algebra.

TWO LAYERS, and conflating them is how quadruped's bug 44 hid behind six green
gates for two days:

  * LAYER 1, `test_humanoid_cmu_xml_matches_reference` — our merged XML string
    and the reference `.xml` are BOTH compiled by MuJoCo and diffed over every
    mjModel table plus the counts, `<option>` and the element order of all seven
    named object types. Both sides are MuJoCo, so our parser and engine are not
    in the loop: a mismatch isolates the XML TEXT.
  * LAYER 2, the rest — our `fields.Model` against MuJoCo compiled from that
    same string, which is only a valid reference BECAUSE layer 1 proves the
    string is the reference model.

WHAT THIS DOMAIN EXERCISES THAT NO EARLIER PORTED ONE DOES:

  - A NAMED TOP-LEVEL DEFAULT CLASS, `<default class="main">`. It is the only
    one in all nineteen suite domains. Wrong resolution here would hand every
    geom and joint the wrong defaults and still simulate.
  - 56 ACTUATORS AND 57 JOINTS against comptime tables that held 32 of each
    until 2026-08-03. Both scans were `while count < CAP` while `ParsedModel`
    counted the tags independently, so the pre-fix behaviour was a model with
    the right `nu`, the full action space exposed, and ZERO FORCE through 24
    actuators. `test_humanoid_cmu_actuator_constants_match_mujoco` is the gate
    on the widening — it checks the LAST actuator, which is the one truncation
    drops.
  - `<contact><exclude>` THROUGH `merge_mjcf`. Two independent bugs were
    stacked here, both found by this port and both silent:
      1. `merge_mjcf` dropped the entire `<contact>` section, on the stale
         grounds of "no exclude/pair support yet" — false at both ends, since
         `full_parser._fill_excludes` fills the record and
         `contact_detection` skips the pair.
      2. `ModelDefFromXML`'s `nexclude` parameter DEFAULTS TO 0 and nothing
         checks it, so omitting it builds an exclusion-free model silently.
    Either alone produces `nexclude == 0`; the symptom is five body pairs
    colliding that MuJoCo never collides, which reads as a solver divergence.

⚠ COUNT MODEL ELEMENTS WITH MuJoCo, NOT WITH grep. On this file `grep -c
'<joint '` says 60 against `njnt` 57, `<motor` says 57 against `nu` 56, `<geom`
52 against 50, `<site` 6 against 5 — every difference is an element inside a
`<default>` block. Three comptime caps were sized off those greps in the first
draft of this port.

The rollout gates the CONTACT-FREE PREFIX and reports the rest — the split
walker and humanoid already use, because the contact solver is the one
component known to disagree with MuJoCo at a level that would swamp everything
else. ⚠ On this model the prefix is SHORT (4-25 steps) and it is short for a
measured reason: the CMU skeleton folds into itself under gravity and its own
joint springs within ~32 steps even at zero action. Both the action amplitude
and the three initial poses were chosen from sweeps against MuJoCo rather than
carried over from `humanoid`'s test; the notes at `AMP` and at `quats` give the
numbers.

Run with:
    pixi run mojo run -I . tests/dm_control/test_humanoid_cmu_vs_dm_control.mojo
"""

from std.math import abs, sin, sqrt, inf
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.humanoid_cmu import (
    DMHumanoidCMUStand,
    DMHumanoidCMUWalk,
    DMHumanoidCMURun,
    DMHumanoidCMUModel,
    HUMANOID_CMU_OBS_DIM,
    THORAX_BODY_IDX,
    HEAD_BODY_IDX,
    extremity_body_indices,
    STAND_HEIGHT,
    WALK_SPEED,
    RUN_SPEED,
)
from mojo_rl.physics3d.fields import Model, Dims
from mojo_rl.physics3d.joint_types import JNT_FREE
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    BODY_IDX_MASS,
    BODY_IDX_IPOS_X,
    BODY_IDX_IXX,
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    JOINT_IDX_STIFFNESS,
    JOINT_IDX_ARMATURE,
    JOINT_IDX_DAMPING,
    MODEL_GEOM_SIZE,
    GEOM_IDX_TYPE,
)
from mojo_rl.physics3d.constants import (
    GEOM_PLANE,
    GEOM_SPHERE,
    GEOM_CAPSULE,
    GEOM_ELLIPSOID,
    GEOM_CYLINDER,
    GEOM_BOX,
    GEOM_MESH,
)
from mojo_rl.physics3d.gpu.constants import (
    MODEL_ACTUATOR_SIZE,
    ACT_IDX_GEAR,
    ACT_IDX_CTRL_MIN,
    ACT_IDX_CTRL_MAX,
    ACT_IDX_TRN_N,
)

# The PARENT of the `dm_control` package, so `import dm_control.utils.rewards`
# resolves. Pointing at the package directory itself makes the import fail.
comptime REF_PATH = "references/dm_control-main"
comptime TEST_PATH = "tests/dm_control"

comptime NQ = DMHumanoidCMUModel.NQ
comptime NV = DMHumanoidCMUModel.NV
comptime NBODY = DMHumanoidCMUModel.NBODY
comptime NJOINT = DMHumanoidCMUModel.NJOINT
comptime NGEOM = DMHumanoidCMUModel.NGEOM
comptime NSITE = DMHumanoidCMUModel.NSITE
comptime NEXCLUDE = DMHumanoidCMUModel.NEXCLUDE
comptime NACT = DMHumanoidCMUModel.nact
comptime FRAME_SKIP = 10

comptime EnvStand = DMHumanoidCMUStand[DType.float64]
comptime EnvWalk = DMHumanoidCMUWalk[DType.float64]
comptime EnvRun = DMHumanoidCMURun[DType.float64]

# Parsed constants are exact — both sides read the same decimal literals — so
# anything above rounding is a real divergence.
comptime MODEL_TOL: Float64 = 1e-12
# `invweight0` is COMPUTED from the mass matrix at qpos0, not parsed, so it
# carries the whole FK -> CRBA chain. Since the quaternion epsilon fixes of
# 2026-08-03 that chain lands at ~5e-15 on quadruped; this is ~200x that.
comptime INVWEIGHT_TOL: Float64 = 1e-12

# ⚠ AMP IS 0.08, NOT humanoid's 0.6, AND IT WAS MEASURED RATHER THAN CHOSEN.
# This skeleton self-collides far more readily than `humanoid`: 56 actuators
# with gears up to 120, fingers and thumbs adjacent, toes under the feet, and
# only five `<exclude>` pairs (all in the neck/clavicle region). Swept against
# MuJoCo, the shortest contact-free prefix over the three inits below runs
#     amp 0.6 -> 0   0.3 -> 0   0.15 -> 1   0.08 -> 4   0.04 -> 4   0.0 -> 32
# so even ZERO action gives only 32 steps — the thing folds under gravity and
# its own joint springs alone. 0.08 is the largest amplitude that still leaves
# a prefix to gate, and with gears of 20..120 it is 1.6..9.6 N*m of real
# actuator torque, so the actuator path is genuinely exercised rather than
# nulled out. `small_control` varies little at this amplitude; the FULL-run
# numbers below cover the large-action regime, un-gated.
comptime AMP: Float64 = 0.08
comptime N_STEPS: Int = 60
# The prefix must be long enough to gate something. Measured min is 4.
comptime MIN_SMOOTH_STEPS: Int = 4


def _action_at(step: Int, k: Int) -> Float64:
    return AMP * sin(0.07 * Float64(step) + 0.41 * Float64(k))


def _mj_geom_type(ours: Int) -> Int:
    """Our `GEOM_*` code -> MuJoCo's `mjtGeom`.

    ⚠ THE TWO ENUMS ARE NOT THE SAME NUMBERING — MuJoCo interleaves HFIELD at 1
    and orders ellipsoid/cylinder/box differently, so comparing the raw codes
    would agree on plane and disagree on everything else for the wrong reason.
    Copied from `test_quadruped_vs_dm_control` rather than shared, because a
    silently-wrong mapping in one place is better than in five.
    """
    if ours == GEOM_PLANE:
        return 0
    if ours == GEOM_SPHERE:
        return 2
    if ours == GEOM_CAPSULE:
        return 3
    if ours == GEOM_ELLIPSOID:
        return 4
    if ours == GEOM_CYLINDER:
        return 5
    if ours == GEOM_BOX:
        return 6
    if ours == GEOM_MESH:
        return 7
    return -1


def _rel(ours: Float64, want: Float64) -> Float64:
    var scale = abs(want)
    if scale < 1.0:
        scale = 1.0
    return abs(ours - want) / scale


def _ref_module() raises -> PythonObject:
    """`humanoid_cmu_ref`, importable from the test directory."""
    var sys = Python.import_module("sys")
    sys.path.insert(0, TEST_PATH)
    return Python.import_module("humanoid_cmu_ref")


def _mj_from_our_xml() raises -> PythonObject:
    """MuJoCo compiled from OUR merged XML — the layer-2 reference.

    Valid only because layer 1 proves this string compiles to the reference
    model; on its own it would compare our engine against our own parser.
    """
    var mujoco = Python.import_module("mujoco")
    return mujoco.MjModel.from_xml_path("mojo_rl/envs/dm_control/assets/humanoid_cmu.xml")


def _build() raises -> Model[DType.float64, Dims[nv=DMHumanoidCMUModel.NV, nbody=DMHumanoidCMUModel.NBODY, njoint=DMHumanoidCMUModel.NJOINT, ngeom=DMHumanoidCMUModel.NGEOM, nequality=DMHumanoidCMUModel.MAX_EQUALITY, ntendon=DMHumanoidCMUModel.MAX_TENDON, nsite=DMHumanoidCMUModel.NSITE, nexclude=DMHumanoidCMUModel.NEXCLUDE, nmesh_verts=0]]:
    var ctx = DeviceContext()
    var mf = Model[DType.float64, Dims[nv=DMHumanoidCMUModel.NV, nbody=DMHumanoidCMUModel.NBODY, njoint=DMHumanoidCMUModel.NJOINT, ngeom=DMHumanoidCMUModel.NGEOM, nequality=DMHumanoidCMUModel.MAX_EQUALITY, ntendon=DMHumanoidCMUModel.MAX_TENDON, nsite=DMHumanoidCMUModel.NSITE, nexclude=DMHumanoidCMUModel.NEXCLUDE, nmesh_verts=0]]()
    DMHumanoidCMUModel.init_fields[DType.float64, 0](ctx, mf)
    return mf^


# =============================================================================
# LAYER 1 — the XML text
# =============================================================================


def test_humanoid_cmu_xml_matches_reference() raises:
    """Our merged XML vs `humanoid_CMU.xml`, both compiled by MuJoCo.

    Diffs every mjModel table plus the counts, `<option>` and the element order
    of the seven named object types, at tolerance EXACTLY ZERO — both sides ran
    the same compiler on the same numbers, so any difference at all is a
    finding.

    THE ONE DEVIATION IS THE `<sensor>` BLOCK, and it is handled by removing it
    from BOTH sides rather than by skipping the sensor tables — skipping would
    leave the deviation unmeasured. The block's contents are then asserted
    explicitly, so the day upstream adds a sensor the tasks read, this fails
    instead of quietly continuing to drop it.
    """
    print("--- humanoid_CMU: layer 1, XML text vs the reference ---")
    var refmod = _ref_module()
    var diff = Python.import_module("mjmodel_diff")

    var bad = refmod.compare_xml_to_reference(
        "mojo_rl/envs/dm_control/assets/humanoid_cmu.xml"
    )
    var n_bad = Int(py=Python.import_module("builtins").len(bad))
    if n_bad > 0:
        for i in range(n_bad):
            print("   MISMATCH:", String(py=bad[i]))
    print("  mismatches =", n_bad, " over",
          Int(py=diff.n_checks()), "checks;",
          Int(py=diff.n_tables()), "of them mjModel tables")
    assert_true(
        n_bad == 0,
        "our XML does not compile to the reference model — see the list above",
    )

    # Non-vacuity: a comparison over an empty table list would also report 0.
    assert_true(
        Int(py=diff.n_tables()) >= 97,
        "the shared layer-1 table list shrank — a subset comparison is how"
        " quadruped's bug 44 hid behind six green gates for two days",
    )

    # The dropped sensor block, stated rather than assumed.
    var sensors = refmod.sensor_block_contents()
    var expected = refmod.EXPECTED_SENSORS
    var builtins = Python.import_module("builtins")
    assert_true(
        Int(py=builtins.len(sensors)) == Int(py=builtins.len(expected)),
        "the reference's <sensor> block changed size — the port drops it, so"
        " re-check that no task reads a sensor we do not compute",
    )
    assert_true(
        Bool(py=(sensors == expected)),
        "the reference's <sensor> block changed contents — see above",
    )
    print("  dropped <sensor> block:", Int(py=builtins.len(sensors)),
          "sensors, of which the tasks read",
          Int(py=builtins.len(refmod.READ_BY_TASKS)),
          "(thorax_subtreelinvel, computed from Data.xvel)")


# =============================================================================
# LAYER 2 — our fields.Model
# =============================================================================


def test_humanoid_cmu_model_matches_mujoco() raises:
    """Dims, body order, inertials, passive-force parameters, exclusions."""
    print("--- humanoid_CMU: layer 2, fields.Model vs MuJoCo ---")
    var mujoco = Python.import_module("mujoco")
    var m = _mj_from_our_xml()

    assert_true(Int(py=m.nq) == NQ, "nq mismatch")
    assert_true(Int(py=m.nv) == NV, "nv mismatch")
    assert_true(Int(py=m.nbody) == NBODY, "nbody mismatch")
    assert_true(Int(py=m.njnt) == NJOINT, "njnt mismatch")
    assert_true(Int(py=m.ngeom) == NGEOM, "ngeom mismatch")
    assert_true(Int(py=m.nsite) == NSITE, "nsite mismatch")
    assert_true(Int(py=m.nu) == NACT, "nu mismatch")
    assert_true(
        Int(py=m.nexclude) == NEXCLUDE,
        "nexclude mismatch — merge_mjcf dropping <contact>, or ModelDefFromXML"
        " built without nexclude=? Both default to 0 silently.",
    )
    assert_true(
        NEXCLUDE == 5,
        "humanoid_CMU declares five <exclude> pairs; zero here means the"
        " clavicles and neck collide where MuJoCo never lets them",
    )

    var mf = _build()

    # Joint 0 must be the free root, and the body order must be the tree DFS
    # our body-index comptimes assume.
    assert_true(
        Int(mf.joints.data[0 * MODEL_JOINT_SIZE + JOINT_IDX_TYPE]) == JNT_FREE,
        "joint 0 is not the free root — did <freejoint> normalization break?",
    )
    var named = [
        ("thorax", THORAX_BODY_IDX),
        ("head", HEAD_BODY_IDX),
    ]
    for nb in named:
        var ref_id = Int(
            py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nb[0])
        )
        assert_true(
            ref_id == nb[1],
            String("body index drifted from MuJoCo's for ") + nb[0],
        )
    # The extremity bodies too — a permutation here silently scrambles 12
    # observation slots.
    var limb_names = ["lhand", "lfoot", "rhand", "rfoot"]
    var limbs = extremity_body_indices()
    for li in range(len(limbs)):
        var ref_id = Int(
            py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, limb_names[li])
        )
        assert_true(
            ref_id == limbs[li],
            String("extremity order drifted at ") + limb_names[li],
        )

    # Inertials.
    var worst_mass = 0.0
    var worst_ipos = 0.0
    var worst_inertia = 0.0
    for b in range(NBODY):
        var dm = abs(
            Float64(mf.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_MASS])
            - Float64(py=m.body_mass[b])
        )
        if dm > worst_mass:
            worst_mass = dm
        for k in range(3):
            var dp = abs(
                Float64(
                    mf.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_IPOS_X + k]
                )
                - Float64(py=m.body_ipos[b][k])
            )
            if dp > worst_ipos:
                worst_ipos = dp
            var di = abs(
                Float64(mf.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_IXX + k])
                - Float64(py=m.body_inertia[b][k])
            )
            if di > worst_inertia:
                worst_inertia = di
    print("  max |d(mass)| =", worst_mass, " |d(ipos)| =", worst_ipos,
          " |d(inertia)| =", worst_inertia)
    assert_true(worst_mass <= MODEL_TOL, "masses differ from MuJoCo")
    assert_true(worst_ipos <= MODEL_TOL, "body CoMs differ from MuJoCo")
    assert_true(worst_inertia <= MODEL_TOL, "inertias differ from MuJoCo")

    # Passive-force parameters, per joint. `dof_*` is indexed by DOF and the
    # free root occupies DOFs 0..5, so hinge j maps to DOF j + 5.
    #
    # THIS IS THE GATE ON `<default class="main">`. Every joint's stiffness,
    # armature and damping is inherited — from `main` itself, or from one of
    # the three `stiff_*` classes nested inside it. A named top-level class
    # resolving wrongly shows up here as a wholesale zero, not as noise.
    var worst_stiff = 0.0
    var worst_arm = 0.0
    var worst_damp = 0.0
    var worst_range = 0.0
    var nonzero_stiffness = 0
    var n_limited = 0
    for j in range(1, NJOINT):
        var dof = j + 5
        var ds = abs(
            Float64(mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_STIFFNESS])
            - Float64(py=m.jnt_stiffness[j])
        )
        if ds > worst_stiff:
            worst_stiff = ds
        if Float64(py=m.jnt_stiffness[j]) != 0.0:
            nonzero_stiffness += 1
        if Int(py=m.jnt_limited[j]) != 0:
            n_limited += 1
        var da = abs(
            Float64(mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_ARMATURE])
            - Float64(py=m.dof_armature[dof])
        )
        if da > worst_arm:
            worst_arm = da
        var dd = abs(
            Float64(mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_DAMPING])
            - Float64(py=m.dof_damping[dof])
        )
        if dd > worst_damp:
            worst_damp = dd
        var dlo = abs(
            Float64(mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MIN])
            - Float64(py=m.jnt_range[j][0])
        )
        var dhi = abs(
            Float64(mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MAX])
            - Float64(py=m.jnt_range[j][1])
        )
        if dlo > worst_range:
            worst_range = dlo
        if dhi > worst_range:
            worst_range = dhi
    print("  max |d(stiffness)| =", worst_stiff, " over", nonzero_stiffness,
          "springs;  |d(armature)| =", worst_arm,
          " |d(damping)| =", worst_damp)
    print("  max |d(jnt_range)| =", worst_range, " over", n_limited,
          "limited joints")
    # Every one of the 56 hinges inherits `stiffness` from `main` (0.1) or a
    # `stiff_*` class (0.5 / 10 / 30), so all 56 are nonzero. A wholesale zero
    # is what a broken named-top-level-class lookup produces.
    assert_true(
        nonzero_stiffness == 56,
        "expected all 56 hinges to carry a spring inherited from"
        " <default class='main'> or a stiff_* class — a zero here means the"
        " named top-level default class did not resolve",
    )
    assert_true(
        n_limited == 56,
        "every hinge declares a range and `main` says limited='true'",
    )
    assert_true(worst_stiff <= MODEL_TOL, "joint stiffness differs")
    assert_true(worst_arm <= MODEL_TOL, "armature differs")
    assert_true(worst_damp <= MODEL_TOL, "damping differs")
    assert_true(
        worst_range <= MODEL_TOL,
        "joint ranges differ — degree->radian conversion missing?"
        " humanoid_CMU declares no <compiler>, so `angle` must default to"
        " DEGREE (MuJoCo's MJCF default), not radian",
    )

    # Geom types, PER GEOM against MuJoCo's, not as a count.
    #
    # ⚠ An earlier draft of this test only tallied MuJoCo's OWN type mix and
    # asserted the tally — which compares MuJoCo to itself and would pass with
    # every one of our geoms set to sphere. The per-geom comparison below is
    # the actual check; the tally is kept only as a coverage report.
    #
    # The two ellipsoids matter most: the hands are the only ellipsoids in the
    # model, `ellipsoid` used to fall through to GEOM_SPHERE silently, and
    # every capsule here inherits `type=` from `<default class="humanoid">`
    # rather than declaring it.
    var n_plane = 0
    var n_sphere = 0
    var n_capsule = 0
    var n_ellipsoid = 0
    for g in range(NGEOM):
        var ours_t = Int(mf.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_TYPE])
        var mj_t = Int(py=m.geom_type[g])
        assert_true(
            _mj_geom_type(ours_t) == mj_t,
            String("geom_type mismatch on geom ") + String(g)
            + " — a `type=` inherited from the wrong default class changes"
            " this without changing ngeom",
        )
        if mj_t == 0:
            n_plane += 1
        elif mj_t == 2:
            n_sphere += 1
        elif mj_t == 3:
            n_capsule += 1
        elif mj_t == 4:
            n_ellipsoid += 1
    print("  geoms: plane", n_plane, "sphere", n_sphere, "capsule", n_capsule,
          "ellipsoid", n_ellipsoid)
    # Non-vacuity: the mix must still be the one this model is known to have,
    # so the per-geom loop above cannot go quiet on a model that changed shape.
    assert_true(
        n_plane == 1 and n_sphere == 8 and n_capsule == 39 and n_ellipsoid == 2,
        "geom type mix drifted from MuJoCo's (1 plane, 8 sphere, 39 capsule,"
        " 2 ellipsoid)",
    )

    # The five exclusion pairs, by body id.
    var ref_pairs = List[Int]()
    for e in range(NEXCLUDE):
        var sig = Int(py=m.exclude_signature[e])
        ref_pairs.append(sig >> 16)
        ref_pairs.append(sig & 0xFFFF)
    var matched = 0
    for e in range(NEXCLUDE):
        var b1 = Int(mf.excludes.data[e * 2 + 0])
        var b2 = Int(mf.excludes.data[e * 2 + 1])
        for f in range(NEXCLUDE):
            var r1 = ref_pairs[f * 2 + 0]
            var r2 = ref_pairs[f * 2 + 1]
            if (b1 == r1 and b2 == r2) or (b1 == r2 and b2 == r1):
                matched += 1
                break
    print("  exclusion pairs matched:", matched, "/", NEXCLUDE)
    assert_true(
        matched == NEXCLUDE,
        "our <contact><exclude> pairs are not MuJoCo's — an unordered match,"
        " because our record's body order is not specified",
    )


def test_humanoid_cmu_actuator_constants_match_mujoco() raises:
    """The 56 motors: gear, transmission target, ctrlrange.

    THE POINT OF THIS TEST IS THE TAIL. Before the 2026-08-03 widening the
    comptime scan stopped at 32 actuators, so actuators 32..55 held their
    fill values — gear 1.0, `motor_trn_n == 0` — and `apply_actions` skipped
    them entirely. Everything below index 32 was correct, `nu` was correct, and
    the action space was the right width. Only the tail discriminates.

    ⚠ `Mdl._acd` is a COMPTIME value and RE-MATERIALIZES ON EVERY READ; in a
    function this size that yields garbage. One explicit `materialize` into a
    local, then read the local.
    """
    print("--- humanoid_CMU: actuator constants (the widened comptime table) ---")
    var m = _mj_from_our_xml()
    var sf = DMHumanoidCMUModel.make_spec_fields[DType.float64]()

    var worst_gear = 0.0
    var n_unresolved = 0
    var worst_ctrl = 0.0
    for a in range(NACT):
        var dg = abs(
            Float64(Float64(sf.actuators.data[(a) * MODEL_ACTUATOR_SIZE + ACT_IDX_GEAR])) - Float64(py=m.actuator_gear[a][0])
        )
        if dg > worst_gear:
            worst_gear = dg
        if Int(sf.actuators.data[(a) * MODEL_ACTUATOR_SIZE + ACT_IDX_TRN_N]) == 0:
            n_unresolved += 1
        var dlo = abs(
            Float64(Float64(sf.actuators.data[(a) * MODEL_ACTUATOR_SIZE + ACT_IDX_CTRL_MIN]))
            - Float64(py=m.actuator_ctrlrange[a][0])
        )
        var dhi = abs(
            Float64(Float64(sf.actuators.data[(a) * MODEL_ACTUATOR_SIZE + ACT_IDX_CTRL_MAX]))
            - Float64(py=m.actuator_ctrlrange[a][1])
        )
        if dlo > worst_ctrl:
            worst_ctrl = dlo
        if dhi > worst_ctrl:
            worst_ctrl = dhi
    print("  max |d(gear)| =", worst_gear, " max |d(ctrlrange)| =", worst_ctrl,
          " unresolved transmissions =", n_unresolved, "/", NACT)
    assert_true(
        n_unresolved == 0,
        "some actuator resolved to no transmission — `apply_actions` skips"
        " those SILENTLY, which is exactly what the 32-actuator truncation did",
    )
    assert_true(worst_gear <= MODEL_TOL, "actuator gears differ from MuJoCo")
    assert_true(
        worst_ctrl <= MODEL_TOL,
        "ctrlrange differs — it is declared only on <default class='main'>,"
        " so this is a second probe on the named-top-level-class lookup",
    )
    # Non-vacuity: the tail must genuinely be past the old cap.
    assert_true(
        NACT > 32,
        "this model no longer exceeds the old 32-actuator cap — the test has"
        " stopped covering the truncation it exists for",
    )
    # The gears are not all equal, so a fill-value table cannot pass above.
    var g0 = Float64(Float64(sf.actuators.data[(0) * MODEL_ACTUATOR_SIZE + ACT_IDX_GEAR]))
    var n_diff = 0
    for a in range(NACT):
        if abs(Float64(Float64(sf.actuators.data[(a) * MODEL_ACTUATOR_SIZE + ACT_IDX_GEAR])) - g0) > 1e-9:
            n_diff += 1
    assert_true(
        n_diff > 0,
        "every gear is identical — a table of fill values would pass the gear"
        " check vacuously",
    )


def test_humanoid_cmu_invweight0_matches_mujoco() raises:
    """`body_invweight0` / `dof_invweight0`.

    Run for every newly ported model since bug 20, which was a 64x silent
    contact-stiffness multiplier living exactly here. Contacts read
    `body_invweight0` and the 56 hinge limits read `dof_invweight0`, so on this
    model both are live.

    These are COMPUTED, not parsed — `mj_setConst` builds them from the mass
    matrix at qpos0 — so they carry the entire FK -> CRBA chain. Agreement here
    at 1e-15 is the strongest single statement that the chain is exact.
    """
    print("--- humanoid_CMU: invweight0 ---")
    var m = _mj_from_our_xml()
    var mf = _build()

    var biw = m.body_invweight0.tolist()
    var diw = m.dof_invweight0.tolist()
    var worst = Float64(0)
    for b in range(NBODY):
        for k in range(2):
            var rel = _rel(
                Float64(mf.body_invweight0.data[2 * b + k]),
                Float64(py=biw[b][k]),
            )
            worst = max(worst, rel)
            assert_true(
                rel <= INVWEIGHT_TOL,
                String("body_invweight0 mismatch on body ") + String(b),
            )
    for i in range(NV):
        var rel = _rel(Float64(mf.dof_invweight0.data[i]), Float64(py=diw[i]))
        worst = max(worst, rel)
        assert_true(
            rel <= INVWEIGHT_TOL,
            String("dof_invweight0 mismatch on dof ") + String(i),
        )
    print("  worst invweight0 rel err =", worst)


# =============================================================================
# Dynamics / observation / reward
# =============================================================================


def _setup() raises -> PythonObject:
    var sys = Python.import_module("sys")
    sys.path.insert(0, REF_PATH)
    var mujoco = Python.import_module("mujoco")
    var rw = Python.import_module("dm_control.utils.rewards")
    var refmod = _ref_module()
    # ⚠ WITH sensors: the rollout needs `thorax_subtreelinvel` as ground truth
    # for `com_velocity`. Layer 1 compares the sensor-free pair; this one is a
    # different model object for a different purpose, and using the layer-1 one
    # here would leave `com_velocity` ungated.
    var model = refmod.model(True)
    var data = mujoco.MjData(model)
    var tol = Python.evaluate(
        "lambda rw: lambda x, lo, hi, m, s, v: float("
        "rw.tolerance(x, bounds=(lo, hi), margin=m, sigmoid=s,"
        " value_at_margin=v))"
    )(rw)
    return Python.tuple(mujoco, model, data, tol)


def test_humanoid_cmu_dynamics_vs_dm_control() raises:
    """Physics / observation / reward parity over the contact-free prefix.

    The observation rebuild here is the reference's, expressed against MuJoCo's
    own arrays, and it is where the two humanoid domains genuinely differ:

      * the reference body is `thorax` (14), not `torso` (`humanoid`'s index 1
        is this model's free-jointed `root`);
      * the reward's `upright` term reads `xmat['thorax', 'zy']` — element 7 —
        where `humanoid`'s reads `zz`, element 8. The OBSERVATION still takes
        the whole z row, so obs and reward disagree on which element matters.
        Both are transcribed from the reference; the asymmetry is upstream's.
      * `small_control` divides by 56, not 21.

    Each of those produces a smooth, plausible number when wrong.
    """
    var handle = _setup()
    var mujoco = handle[0]
    var model = handle[1]
    var data = handle[2]
    var tol = handle[3]

    var svl_adr = Int(
        py=model.sensor_adr[
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SENSOR, "thorax_subtreelinvel"
            )
        ]
    )

    var max_state = 0.0
    var max_obs = 0.0
    var max_r_stand = 0.0
    var max_r_walk = 0.0
    var max_r_run = 0.0
    var max_state_s = 0.0
    var max_obs_s = 0.0
    var max_r_stand_s = 0.0
    var max_r_walk_s = 0.0
    var max_r_run_s = 0.0
    var min_smooth = N_STEPS
    var max_ncon = 0
    var r_stand_lo = 1e9
    var r_stand_hi = -1e9

    # qpos = [x, y, z, qw, qx, qy, qz, 56 joint angles].
    #
    # ⚠ THE ORIENTATIONS ARE CHOSEN FROM A MEASUREMENT OF WHAT THE REWARD
    # READS, not copied from `humanoid`'s test. This skeleton is authored y-up
    # inside its own frame, so a rotation of +90 deg about x is what stands it
    # up, and the reward's `upright` term reads the thorax's ZY element:
    #     x+90     -> thorax_zy = +1.0   (upright term 1.0)
    #     identity ->              0.0   (upright term 0.526)
    #     x-90     ->             -1.0   (upright term 0.0, the margin)
    # `humanoid`'s quaternions leave zy pinned at 0.0 for two of its three
    # inits. The spawn heights then put `head_height` on both sides of
    # _STAND_HEIGHT (1.4) while keeping every geom off the floor.
    #
    # Measured effect: the stand reward sweeps 0.0 .. 0.999 over the prefix.
    # The first draft of this test used humanoid's inits at z = 3 and swept
    # 0.501 .. 0.515 — a gate that agrees to 1e-14 on a quantity that never
    # moves is not evidence.
    var quats = [
        [0.7071, 0.7071, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.7071, -0.7071, 0.0, 0.0],
    ]
    var spawn_z = [1.3, 1.0, 2.0]
    var joint_seeds = [0.0, 0.15, -0.2]

    for t in range(3):
        var quat = quats[t].copy()
        var seed = joint_seeds[t]
        var z0 = spawn_z[t]

        mujoco.mj_resetData(model, data)
        data.qpos[0] = 0.0
        data.qpos[1] = 0.0
        data.qpos[2] = z0
        for k in range(4):
            data.qpos[3 + k] = quat[k]
        for i in range(7, NQ):
            data.qpos[i] = seed * sin(0.7 * Float64(i))
        mujoco.mj_forward(model, data)

        var qs = List[Float64]()
        var vs = List[Float64]()
        qs.append(0.0)
        qs.append(0.0)
        qs.append(z0)
        for k in range(4):
            qs.append(quat[k])
        for i in range(7, NQ):
            qs.append(seed * sin(0.7 * Float64(i)))
        for _ in range(NV):
            vs.append(0.0)

        var e_stand = EnvStand()
        _ = e_stand.reset()
        e_stand.set_state(qs, vs)
        var e_walk = EnvWalk()
        _ = e_walk.reset()
        e_walk.set_state(qs, vs)
        var e_run = EnvRun()
        _ = e_run.reset()
        e_run.set_state(qs, vs)

        var smooth = True
        var smooth_steps = 0
        for step in range(N_STEPS):
            var a_stand = EnvStand.ActionType()
            var a_walk = EnvWalk.ActionType()
            var a_run = EnvRun.ActionType()
            for k in range(NACT):
                var a = _action_at(step, k)
                data.ctrl[k] = a
                a_stand.data[k] = a
                a_walk.data[k] = a
                a_run.data[k] = a
            for _ in range(FRAME_SKIP):
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)

            var o_stand = e_stand.step(a_stand)
            var o_walk = e_walk.step(a_walk)
            var o_run = e_run.step(a_run)

            var ncon = Int(py=data.ncon)
            if ncon > max_ncon:
                max_ncon = ncon
            if ncon > 0:
                smooth = False
            if smooth:
                smooth_steps += 1

            for i in range(NQ):
                var dq = abs(
                    Float64(py=data.qpos[i]) - Float64(e_stand.d.qpos.data[i])
                )
                if dq > max_state:
                    max_state = dq
                if smooth and dq > max_state_s:
                    max_state_s = dq
            for i in range(NV):
                var dv = abs(
                    Float64(py=data.qvel[i]) - Float64(e_stand.d.qvel.data[i])
                )
                if dv > max_state:
                    max_state = dv
                if smooth and dv > max_state_s:
                    max_state_s = dv

            # ── the reference observation, rebuilt from MuJoCo ──
            var ref_obs = List[Float64]()
            for i in range(7, NQ):  # joint_angles()
                ref_obs.append(Float64(py=data.qpos[i]))
            var head_z = Float64(py=data.xpos[HEAD_BODY_IDX][2])
            ref_obs.append(head_z)  # head_height()

            # extremities(): (limb - thorax) . xmat[thorax], i.e. R^T v.
            var tp = [
                Float64(py=data.xpos[THORAX_BODY_IDX][0]),
                Float64(py=data.xpos[THORAX_BODY_IDX][1]),
                Float64(py=data.xpos[THORAX_BODY_IDX][2]),
            ]
            var rm = List[Float64]()
            for k in range(9):
                rm.append(Float64(py=data.xmat[THORAX_BODY_IDX][k]))
            var limbs = extremity_body_indices()
            for li in range(len(limbs)):
                var b = limbs[li]
                var v0 = Float64(py=data.xpos[b][0]) - tp[0]
                var v1 = Float64(py=data.xpos[b][1]) - tp[1]
                var v2 = Float64(py=data.xpos[b][2]) - tp[2]
                for c in range(3):
                    ref_obs.append(
                        v0 * rm[0 * 3 + c]
                        + v1 * rm[1 * 3 + c]
                        + v2 * rm[2 * 3 + c]
                    )
            for k in range(6, 9):  # torso_vertical: zx, zy, zz
                ref_obs.append(rm[k])
            var com = [
                Float64(py=data.sensordata[svl_adr + 0]),
                Float64(py=data.sensordata[svl_adr + 1]),
                Float64(py=data.sensordata[svl_adr + 2]),
            ]
            for k in range(3):
                ref_obs.append(com[k])
            for i in range(NV):  # velocity()
                ref_obs.append(Float64(py=data.qvel[i]))

            assert_true(
                len(ref_obs) == HUMANOID_CMU_OBS_DIM,
                "the reference observation is not 137 wide",
            )
            var obs = o_stand[0]
            for i in range(HUMANOID_CMU_OBS_DIM):
                var d_o = abs(ref_obs[i] - Float64(obs.data[i]))
                if d_o > max_obs:
                    max_obs = d_o
                if smooth and d_o > max_obs_s:
                    max_obs_s = d_o

            # ── the reference rewards ──
            var standing = Float64(
                py=tol(
                    head_z,
                    STAND_HEIGHT,
                    Float64(py=Python.evaluate("float('inf')")),
                    STAND_HEIGHT / 4.0,
                    String("gaussian"),
                    0.1,
                )
            )
            # ⚠ rm[7] is 'zy'. `humanoid`'s equivalent line uses rm[8], 'zz'.
            var upright = Float64(
                py=tol(
                    rm[7],
                    0.9,
                    Float64(py=Python.evaluate("float('inf')")),
                    1.9,
                    String("linear"),
                    0.0,
                )
            )
            var stand_reward = standing * upright
            var acc = 0.0
            for k in range(NACT):
                acc += Float64(
                    py=tol(
                        Float64(py=data.ctrl[k]),
                        0.0,
                        0.0,
                        1.0,
                        String("quadratic"),
                        0.0,
                    )
                )
            var small_control = (4.0 + acc / Float64(NACT)) / 5.0

            var dm0 = Float64(
                py=tol(com[0], 0.0, 0.0, 2.0, String("gaussian"), 0.1)
            )
            var dm1 = Float64(
                py=tol(com[1], 0.0, 0.0, 2.0, String("gaussian"), 0.1)
            )
            var r_stand = small_control * stand_reward * (dm0 + dm1) / 2.0
            var d_rs = abs(r_stand - Float64(o_stand[1]))
            if d_rs > max_r_stand:
                max_r_stand = d_rs
            if smooth:
                if d_rs > max_r_stand_s:
                    max_r_stand_s = d_rs
                if r_stand < r_stand_lo:
                    r_stand_lo = r_stand
                if r_stand > r_stand_hi:
                    r_stand_hi = r_stand

            var speed = sqrt(com[0] * com[0] + com[1] * com[1])
            var mv_w = Float64(
                py=tol(
                    speed,
                    WALK_SPEED,
                    Float64(py=Python.evaluate("float('inf')")),
                    WALK_SPEED,
                    String("linear"),
                    0.0,
                )
            )
            var r_walk = small_control * stand_reward * (5.0 * mv_w + 1.0) / 6.0
            var d_rw = abs(r_walk - Float64(o_walk[1]))
            if d_rw > max_r_walk:
                max_r_walk = d_rw
            if smooth and d_rw > max_r_walk_s:
                max_r_walk_s = d_rw

            var mv_r = Float64(
                py=tol(
                    speed,
                    RUN_SPEED,
                    Float64(py=Python.evaluate("float('inf')")),
                    RUN_SPEED,
                    String("linear"),
                    0.0,
                )
            )
            var r_run = small_control * stand_reward * (5.0 * mv_r + 1.0) / 6.0
            var d_rr = abs(r_run - Float64(o_run[1]))
            if d_rr > max_r_run:
                max_r_run = d_rr
            if smooth and d_rr > max_r_run_s:
                max_r_run_s = d_rr

        if smooth_steps < min_smooth:
            min_smooth = smooth_steps

    print("humanoid_CMU vs MuJoCo, 3 x", N_STEPS, "steps:")
    print("  contact-free prefix: shortest =", min_smooth, "steps;",
          " reference max ncon over the full run =", max_ncon)
    print("  PREFIX  max |d(state)| =", max_state_s, " |d(obs)| =", max_obs_s)
    print("  PREFIX  max |d(reward)| stand =", max_r_stand_s,
          " walk =", max_r_walk_s, " run =", max_r_run_s)
    print("  FULL    max |d(state)| =", max_state, " |d(obs)| =", max_obs)
    print("  FULL    max |d(reward)| stand =", max_r_stand,
          " walk =", max_r_walk, " run =", max_r_run)
    print("  stand reward range over the prefix =", r_stand_lo, "..",
          r_stand_hi)

    # ── Gates ────────────────────────────────────────────────────────────
    assert_true(
        min_smooth >= MIN_SMOOTH_STEPS,
        "the contact-free prefix is too short to gate anything — raise the"
        " spawn height or reduce the joint seed",
    )
    # `MAX_CONTACTS` is 64 on this model; if MuJoCo ever needs more than that
    # our narrow phase would be dropping contacts silently.
    assert_true(
        max_ncon < 64,
        "MuJoCo exceeded our MAX_CONTACTS — raise it in humanoid_cmu_xml",
    )

    assert_true(max_r_stand_s < 1e-8, "stand reward diverged over the prefix")
    assert_true(max_r_walk_s < 1e-8, "walk reward diverged over the prefix")
    assert_true(max_r_run_s < 1e-8, "run reward diverged over the prefix")

    # A reward that never moves would pass the gates above vacuously.
    assert_true(
        r_stand_hi - r_stand_lo > 0.5,
        "stand reward is degenerate over the prefix — gate is vacuous. The"
        " inits are sized to sweep 0.0 .. 0.999; anything much narrower means"
        " the orientation set stopped moving `upright` or `standing`.",
    )

    # The observation must add no error of its own beyond the state's. If
    # `|d(obs)|` exceeds `|d(state)|` materially, an accessor is wrong (the
    # thorax/torso mix-up shows up exactly here).
    assert_true(
        max_obs_s <= max_state_s * 10.0 + 1e-12,
        "the observation carries error the state does not — check the thorax"
        " body index and the extremities transpose",
    )

    # ── The prefix state budget, and why it is 5.98e-07 rather than 1e-14 ──
    #
    # Budgeted like humanoid's (5.66e-08 against a 3e-6 bound): the residual is
    # the JOINT-LIMIT constraint path, the one gap this port has never closed.
    #
    # ⚠ THE ATTRIBUTION WAS MEASURED, NOT ASSUMED. An earlier draft of this
    # test spawned at z = 3 with humanoid's quaternions and reported
    # 1.9e-14 — seven orders tighter. That was not a better engine, it was a
    # rollout that never engaged a limit. Counting MuJoCo's
    # `efc_type == mjCNSTR_LIMIT_JOINT` rows per prefix step under the inits
    # actually used here gives
    #     init 0:  6 6 6 6 6 6 6 6 6 6      (then contact)
    #     init 1:  5 5 5 5                  (then contact)
    #     init 2:  5 5 5 6 6 4 4 5 5 6 6 6
    # i.e. five or six limits live on EVERY step of EVERY prefix. The tighter
    # number came from a gate that was not testing the thing it appeared to
    # test, which is the failure mode `test_euler_fields_vs_mujoco`'s inherited
    # 1e-4 budget already cost this project once.
    #
    # Bound is 5x the observed value, deliberately tighter than this file's
    # usual ~50x: these are deterministic float64 rollouts with no platform
    # noise, and a limit-path regression is exactly what it exists to catch.
    assert_true(
        max_state_s < 3e-6,
        "humanoid_CMU prefix state diverged beyond the joint-limit budget",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
