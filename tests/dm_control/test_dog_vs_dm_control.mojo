"""dm_control `dog` parity: our model vs MuJoCo + the reference tasks (Phase 4).

Four tasks (stand, walk, trot, run) over `suite/dog.xml`. `dog.py::make_model`
deletes the ball, target, two cameras and four walls, then rewrites the floor
plane's half-extent to `move_speed * _DEFAULT_TIME_LIMIT`, so there are THREE
models, not four — stand and walk both use `_WALK_SPEED * 15 = 15`.

THREE LAYERS here, not the usual two, because this port carries a deviation:

  * LAYER 0, `test_dog_bake_is_inert` — the MESH-INERTIA BAKE. dog.xml has 162
    STL mesh geoms; this port has none. See `dog_xml.mojo` for the argument and
    `dog_ref.check_bake` for the proof. Layers 1 and 2 compare against the
    BAKED reference, so if layer 0 is not sound neither of them means anything.
  * LAYER 1, `test_dog_xml_matches_reference` — our merged XML and the baked
    reference, BOTH compiled by MuJoCo, diffed over every mjModel table plus
    counts, `<option>` and element order. Our parser and engine are not in the
    loop: a mismatch isolates the XML TEXT.
  * LAYER 2, the rest — our `fields.Model` against MuJoCo compiled from that
    same string.

WHAT THIS DOMAIN EXERCISES THAT NO EARLIER PORTED ONE DOES

  - **74 joints**, against a comptime table that held 64 until this port
    (`MAX_COMPTIME_JOINTS`). The pre-fix failure mode is silent: the scan is
    `while count < CAP` while `ParsedModel` counts the tags independently, so
    joints past the cap keep their degrees of freedom and quietly lose their
    STOPS.
  - **`<geom priority>` driving `condim="6"`.** 42 of dog's 120 colliding
    geoms are teeth on `class="tooth_primitive"` (`condim="6" priority="2"
    friction="0.5 0.01 0.01"`); the other 77 primitives are condim 1 and the
    floor is condim 3. Priority means a tooth dictates condim, friction AND
    solref wholesale wherever it touches. ⚠ The condim>=4 friction rows were
    STRUCTURALLY PRESENT AND COMPLETELY INERT until Phase 3 — see
    `test_rolling_friction_vs_mujoco.mojo`. Reading the enum said they worked;
    measuring said they did nothing.
  - **`noslip_iterations="4"`.** A post-solver pass, and NOT a rounding
    refinement: with it disabled MuJoCo's own rollout moves by `max|dqvel|`
    2.9e-2 on the FIRST contacting step. `test_dog_noslip_is_load_bearing`
    pins that number so nobody re-derives "it is probably negligible".

⚠ COUNT MODEL ELEMENTS WITH MuJoCo, NOT WITH grep. `docs/DM_CONTROL_PORT_PHASE2.md`
§4.3 sized this model's caps off greps and was wrong for every field: it said
58 actuators / 147 joints / ~153 nq against the true 38 / 74 / 80. `<default>`
blocks match every one of those greps.

⚠ AND COUNT THE RIGHT VARIANT. dog is two models. stand/walk/trot/run have
njnt 74 / nq 80; fetch keeps the ball's free joint and has 75 / 87. A cap sized
off one of those silently truncates the other.

Run with:
    pixi run mojo run -I . tests/dm_control/test_dog_vs_dm_control.mojo
"""

from std.math import abs, min
from std.collections import InlineArray
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.envs.dm_control.dog.dog_xml import (
    dm_dog_stand_walk_xml,
    dm_dog_trot_xml,
    dm_dog_run_xml,
    DMDogStandWalkModel,
    DOG_OBS_DIM,
    DOG_N_HINGE,
    DOG_HINGE_QPOS_0,
    DOG_HINGE_DOF_0,
    DOG_TORSO_BODY_IDX,
    DOG_PELVIS_BODY_IDX,
    DOG_SKULL_BODY_IDX,
    DOG_SITE_HEAD,
    DOG_SITE_PALM_L,
    DOG_SITE_SOLE_R,
    DOG_FRAME_SKIP,
    dsp,
)
from mojo_rl.physics3d.fields import Model
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    BODY_IDX_MASS,
    BODY_IDX_IPOS_X,
    BODY_IDX_IXX,
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_STIFFNESS,
    JOINT_IDX_ARMATURE,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DAMPING,
    JOINT_IDX_SPRINGREF,
    JOINT_IDX_FRICTIONLOSS,
    MODEL_GEOM_SIZE,
    GEOM_IDX_CONDIM,
    GEOM_IDX_PRIORITY,
)
from mojo_rl.physics3d.joint_types import JNT_FREE
from std.gpu.host import DeviceContext


comptime TEST_PATH = "tests/dm_control"

comptime NQ = DMDogStandWalkModel.NQ
comptime NV = DMDogStandWalkModel.NV
comptime NBODY = DMDogStandWalkModel.NBODY
comptime NJOINT = DMDogStandWalkModel.NJOINT
comptime NGEOM = DMDogStandWalkModel.NGEOM
comptime NSITE = DMDogStandWalkModel.NSITE
comptime NACT = DMDogStandWalkModel.nact
comptime NEXCLUDE = DMDogStandWalkModel.NEXCLUDE

# Both sides are float64 and, for layers 0 and 1, the SAME compiler on the same
# numbers — so the model-constant budget is a rounding epsilon, not a physics
# tolerance.
comptime MODEL_TOL = 1e-14


def _ref_module() raises -> PythonObject:
    var sys = Python.import_module("sys")
    sys.path.insert(0, TEST_PATH)
    return Python.import_module("dog_ref")


def _mj_from_our_xml() raises -> PythonObject:
    """MuJoCo compiled from OUR merged XML — the layer-2 reference.

    Valid only because layer 1 proves this string compiles to the reference
    model; on its own it would compare our engine against our own parser.
    """
    var mujoco = Python.import_module("mujoco")
    return mujoco.MjModel.from_xml_string(materialize[dm_dog_stand_walk_xml]())


def _build() raises -> Model[
    DType.float64,
    DMDogStandWalkModel.NV,
    DMDogStandWalkModel.NBODY,
    DMDogStandWalkModel.NJOINT,
    DMDogStandWalkModel.NGEOM,
    DMDogStandWalkModel.MAX_EQUALITY,
    DMDogStandWalkModel.MAX_TENDON,
    DMDogStandWalkModel.NSITE,
    DMDogStandWalkModel.NEXCLUDE,
    0,
]:
    var ctx = DeviceContext()
    var mf = Model[
        DType.float64,
        DMDogStandWalkModel.NV,
        DMDogStandWalkModel.NBODY,
        DMDogStandWalkModel.NJOINT,
        DMDogStandWalkModel.NGEOM,
        DMDogStandWalkModel.MAX_EQUALITY,
        DMDogStandWalkModel.MAX_TENDON,
        DMDogStandWalkModel.NSITE,
        DMDogStandWalkModel.NEXCLUDE,
        0,
    ]()
    DMDogStandWalkModel.init_fields[DType.float64, 0](ctx, mf)
    return mf^


# =============================================================================
# LAYER 0 — the mesh-inertia bake
# =============================================================================


def test_dog_bake_is_inert() raises:
    """The 162 deleted mesh geoms changed nothing that can affect physics.

    `check_bake` compiles dog.xml with and without its meshes and diffs every
    mjModel table at tolerance EXACTLY ZERO, exempting only `body_geomadr` /
    `body_geomnum` (deleting geoms necessarily renumbers them) and the geom
    tables, which it re-checks with the surviving geoms matched BY NAME so an
    id shift fails rather than passes.

    ⚠ THE PREMISE IS CHECKED, NOT ASSUMED. `check_bake` fails outright if any
    mesh geom has `contype` or `conaffinity` set — if one collided, deleting it
    would change the physics and the whole deviation would be invalid.

    ⚠ NON-VACUITY. A bake that deleted nothing would also report zero
    mismatches, so the count of deleted geoms is asserted too. That is the same
    failure shape as `test_oriented_plane`'s axis-aligned control, which could
    not fail either way until it was made to.
    """
    print("--- dog: layer 0, the mesh-inertia bake ---")
    var refmod = _ref_module()
    var builtins = Python.import_module("builtins")

    var bad = refmod.check_bake()
    var n_bad = Int(py=builtins.len(bad))
    if n_bad > 0:
        for i in range(n_bad):
            print("   MISMATCH:", String(py=bad[i]))
    assert_true(
        n_bad == 0,
        "the bake changed the model — the deviation in dog_xml.mojo is not"
        " sound as written; see the list above",
    )

    var raw = refmod.raw_model()
    var baked = refmod.model()
    var n_raw = Int(py=raw.ngeom)
    var n_baked = Int(py=baked.ngeom)
    var n_mesh = Int(py=raw.nmesh)
    print(
        "  ngeom", n_raw, "->", n_baked,
        " nmesh", n_mesh, "->", Int(py=baked.nmesh),
    )
    assert_true(
        n_raw - n_baked == 162 and n_mesh == 162,
        "the bake did not delete dog's 162 mesh geoms — a bake that deletes"
        " nothing passes the diff above trivially",
    )
    assert_true(
        Int(py=baked.nmesh) == 0,
        "mesh assets survive the bake — the port cannot carry an STL tree",
    )
    assert_true(
        n_baked == NGEOM,
        "our model's geom count is not the baked reference's",
    )


def test_dog_noslip_is_load_bearing() raises:
    """`noslip_iterations="4"` is first-order, measured, not assumed.

    This test asserts a fact about MuJoCo, not about us, and it exists so that
    the cost of the noslip pass is never re-litigated from a reading of the
    source. Both rollouts below are MuJoCo's; the only difference is the option.

    ⚠ THE HORIZON MATTERS. Over 200 steps the two diverge by `max|dqvel|` 2.7,
    which proves nothing on its own — a contact-rich rollout is chaotic and any
    perturbation grows. The number that settles it is the divergence on the
    FIRST step that has contacts at all, which is ~2.9e-2 and cannot be
    amplification because there is nothing yet to amplify.
    """
    print("--- dog: noslip is load-bearing (a fact about MuJoCo) ---")
    var refmod = _ref_module()
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")

    var m4 = refmod.model()
    var m0 = refmod.model()
    m0.opt.noslip_iterations = 0
    var d4 = mujoco.MjData(m4)
    var d0 = mujoco.MjData(m0)

    assert_true(
        Int(py=m4.opt.noslip_iterations) == 4,
        "dog.xml no longer sets noslip_iterations=4 — re-derive this port's"
        " solver requirements before trusting the rest of the file",
    )

    var rng = np.random.RandomState(0)
    var ctrl = rng.uniform(-1.0, 1.0, m4.nu) * 0.3
    d4.ctrl[:] = ctrl
    d0.ctrl[:] = ctrl

    var first_contact_dv = 0.0
    var seen_contact = False
    for _ in range(12):
        mujoco.mj_step(m4, d4)
        mujoco.mj_step(m0, d0)
        if not seen_contact and Int(py=d4.ncon) > 0:
            seen_contact = True
            first_contact_dv = Float64(
                py=np.abs(np.subtract(d4.qvel, d0.qvel)).max()
            )

    print("  first contacting step: max|d(qvel)| =", first_contact_dv)
    assert_true(
        seen_contact,
        "no contacts in 12 steps — this measurement is vacuous",
    )
    assert_true(
        first_contact_dv > 1e-3,
        "noslip no longer changes the first contacting step; if that is real,"
        " the pass can be skipped and this port gets much cheaper — but"
        " re-measure before believing it",
    )


# =============================================================================
# LAYER 1 — the XML text
# =============================================================================


def test_dog_xml_matches_reference() raises:
    """Our three merged XMLs vs the baked reference, every table, tolerance 0.

    All THREE floor sizes are compared, not just one. The floor half-extent is
    the only per-task difference, so checking a single model would leave the
    other two entirely ungated while looking like full coverage.
    """
    print("--- dog: layer 1, XML text vs the reference ---")
    var refmod = _ref_module()
    var diff = Python.import_module("mjmodel_diff")
    var builtins = Python.import_module("builtins")

    var names = ["stand/walk", "trot", "run"]
    var floors = [15, 45, 135]
    var xmls = [
        materialize[dm_dog_stand_walk_xml](),
        materialize[dm_dog_trot_xml](),
        materialize[dm_dog_run_xml](),
    ]
    for t in range(3):
        var bad = refmod.compare_xml_to_reference(xmls[t], floors[t])
        var n_bad = Int(py=builtins.len(bad))
        if n_bad > 0:
            for i in range(n_bad):
                print("   MISMATCH(", names[t], "):", String(py=bad[i]))
        print("  ", names[t], "floor", floors[t], "-> mismatches =", n_bad)
        assert_true(
            n_bad == 0,
            "our XML does not compile to the reference model — see above",
        )

    print(
        "  compared", Int(py=diff.n_checks()), "checks;",
        Int(py=diff.n_tables()), "of them mjModel tables",
    )
    # Non-vacuity: a comparison over an empty table list also reports 0.
    assert_true(
        Int(py=diff.n_tables()) >= 97,
        "the shared layer-1 table list shrank — a subset comparison is how"
        " quadruped's bug 44 hid behind six green gates for two days",
    )

    # The three floors must actually DIFFER, or the loop above compared one
    # model to itself three times.
    var m_walk = Python.import_module("mujoco").MjModel.from_xml_string(
        materialize[dm_dog_stand_walk_xml]()
    )
    var m_run = Python.import_module("mujoco").MjModel.from_xml_string(
        materialize[dm_dog_run_xml]()
    )
    var floor_walk = Float64(py=m_walk.geom_size[0][0])
    var floor_run = Float64(py=m_run.geom_size[0][0])
    print("  floor half-extent: stand/walk", floor_walk, " run", floor_run)
    assert_true(
        floor_walk == 15.0 and floor_run == 135.0,
        "the per-task floor sizes are not move_speed * 15 — stand/walk share"
        " 15 because stand uses _WALK_SPEED",
    )


def test_dog_subtreeangmom_is_declared_and_unread() raises:
    """The one sensor dog declares that this engine does not implement.

    `<subtreeangmom name="torso_angmom" body="torso"/>` is in the model — the
    port keeps it so the layer-1 sensor tables diff clean — but the engine has
    no angular-momentum sensor, only `subtreelinvel`.

    That is safe here for a reason worth stating rather than assuming: NO dog
    observation and NO dog reward reads it. `Physics.inertial_sensors` reads
    accelerometer/velocimeter/gyro and `center_of_mass_velocity` reads
    `torso_linvel`; nothing reads `torso_angmom`. Our configs also read the
    underlying fields directly rather than a packed `sensordata` array, so an
    unimplemented sensor cannot shift the offset of any sensor after it — the
    usual way this kind of gap does damage.

    ⚠ IF A LATER TASK READS IT, THIS TEST IS THE PLACE THAT FINDS OUT. It fails
    the moment the reference's dog.py mentions the sensor name.
    """
    print("--- dog: subtreeangmom declared, and read by nothing ---")
    var builtins = Python.import_module("builtins")
    var mujoco = Python.import_module("mujoco")
    var m = _mj_from_our_xml()

    var sid = Int(
        py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "torso_angmom")
    )
    assert_true(
        sid >= 0,
        "torso_angmom is missing from our XML — the port keeps it for model"
        " fidelity even though nothing computes it",
    )
    print("  torso_angmom is sensor", sid, "of", Int(py=m.nsensor))

    # The load-bearing half: the reference must not read it.
    var os = Python.import_module("os")
    var path = os.path.join(
        TEST_PATH, "..", "..", "references", "dm_control-main",
        "dm_control", "suite", "dog.py",
    )
    var src = String(py=builtins.open(path).read())
    assert_true(
        "torso_angmom" not in src,
        "dog.py now mentions torso_angmom — a task reads the one sensor this"
        " engine does not implement; subtreeangmom must be built before that"
        " task is ported",
    )
    assert_true(
        "torso_linvel" in src,
        "dog.py no longer reads torso_linvel — this check is looking at the"
        " wrong file, so its negative result above means nothing",
    )


# =============================================================================
# LAYER 2 — our fields.Model
# =============================================================================


def test_dog_model_matches_mujoco() raises:
    """Dims, body/joint order, inertials, and the priority/condim columns."""
    print("--- dog: layer 2, fields.Model vs MuJoCo ---")
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
        "nexclude mismatch — merge_mjcf dropping <contact>, or"
        " ModelDefFromXML built without nexclude=? Both default to 0 silently.",
    )
    assert_true(
        NEXCLUDE == 30,
        "dog declares thirty <exclude> pairs; zero here means thirty body"
        " pairs collide that MuJoCo never lets collide",
    )
    assert_true(
        NJOINT == 74 and NQ == 80,
        "these are the stand/walk/trot/run counts; 75/87 would mean the ball"
        " survived make_model and this is the fetch model",
    )

    var mf = _build()

    # Joint 0 must be the free root: `<freejoint>` is sugar our scanners only
    # see after `merge_mjcf` normalizes it, and missing it yields a model that
    # welds the torso to the world and comes out 1 joint / 7 qpos / 6 dof short
    # — which reads as a dimension mismatch far from the cause.
    assert_true(
        Int(mf.joints.data[0 * MODEL_JOINT_SIZE + JOINT_IDX_TYPE]) == JNT_FREE,
        "joint 0 is not the free root — <freejoint> normalization broke",
    )

    # The three bodies the observation projects onto world z, plus the sites
    # the sensors hang off. A permutation here silently scrambles the
    # observation while every number stays finite and plausible.
    var named = [
        ("torso", DOG_TORSO_BODY_IDX),
        ("pelvis", DOG_PELVIS_BODY_IDX),
        ("skull", DOG_SKULL_BODY_IDX),
    ]
    for nb in named:
        var ref_id = Int(
            py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nb[0])
        )
        assert_true(
            ref_id == nb[1],
            String("body index drifted from MuJoCo's for ") + nb[0],
        )
    var named_sites = [
        ("head", DOG_SITE_HEAD),
        ("palm_L", DOG_SITE_PALM_L),
        ("sole_R", DOG_SITE_SOLE_R),
    ]
    for ns in named_sites:
        var ref_id = Int(
            py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, ns[0])
        )
        assert_true(
            ref_id == ns[1],
            String("site index drifted from MuJoCo's for ") + ns[0],
        )

    # The hinge block. `joint_angles`/`joint_velocities` slice qpos and qvel by
    # `jnt_type == HINGE`, and this port indexes that block by its base rather
    # than re-deriving it per step — so the contiguity is an assumption, and
    # assumptions of this shape get checked.
    var n_hinge = 0
    var first_hinge = -1
    var contiguous_q = True
    var contiguous_v = True
    for j in range(NJOINT):
        if Int(py=m.jnt_type[j]) == 3:  # mjJNT_HINGE
            if first_hinge < 0:
                first_hinge = j
            if Int(py=m.jnt_qposadr[j]) != DOG_HINGE_QPOS_0 + n_hinge:
                contiguous_q = False
            if Int(py=m.jnt_dofadr[j]) != DOG_HINGE_DOF_0 + n_hinge:
                contiguous_v = False
            n_hinge += 1
    print("  hinges =", n_hinge, " qpos base", DOG_HINGE_QPOS_0,
          " dof base", DOG_HINGE_DOF_0)
    assert_true(n_hinge == DOG_N_HINGE, "hinge count drifted from MuJoCo's")
    assert_true(
        contiguous_q and contiguous_v,
        "the hinge qpos/dof block is not contiguous — DOG_HINGE_QPOS_0 and"
        " DOG_HINGE_DOF_0 slice it as one run and would now be wrong",
    )

    # Passive-force parameters per joint. dog's root `<joint>` default sets
    # `stiffness="0.1" armature="0.0001" damping="0.01"` for all 73 hinges, and
    # armature in particular feeds `dof_invweight0` — a dropped armature on a
    # 1.5e-3 kg tooth joint inflates that by orders of magnitude while every
    # mass and inertia stays exact. Nothing checked it before.
    var worst_arm = 0.0
    var worst_stiff = 0.0
    var n_nonzero_arm = 0
    for j in range(1, NJOINT):
        var dof = j + 5
        var da = abs(
            Float64(mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_ARMATURE])
            - Float64(py=m.dof_armature[dof])
        )
        if da > worst_arm:
            worst_arm = da
        if Float64(py=m.dof_armature[dof]) != 0.0:
            n_nonzero_arm += 1
        var ds = abs(
            Float64(mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_STIFFNESS])
            - Float64(py=m.jnt_stiffness[j])
        )
        if ds > worst_stiff:
            worst_stiff = ds
    # ⚠ IS THIS AN ORDER PROBLEM OR A VALUE PROBLEM? Ask first, because the
    # geoms above ARE permuted and the same question decides how to read the
    # armature/stiffness numbers. `jnt_range` is order-sensitive and comes
    # straight from the XML, so if our joint order equals MuJoCo's these agree
    # elementwise; if they do not, every per-index comparison below is
    # comparing two different joints and says nothing about the parser.
    # ⚠ LIMITED JOINTS ONLY. For an UNLIMITED joint MuJoCo stores
    # `jnt_range = (0, 0)` — not a range, just an unused slot — while we store
    # the sentinel +-1e10. Measured on dog, the free root is the only joint
    # where they differ, and every one of the 73 limited hinges agrees to the
    # last digit. Comparing the unlimited slot would fail on a difference that
    # cannot affect anything: `is_limited` gates whether a limit row is built
    # at all, so nothing ever reads an unlimited joint's range.
    var worst_range = 0.0
    var n_limited = 0
    for j in range(NJOINT):
        if Int(py=m.jnt_limited[j]) == 0:
            continue
        n_limited += 1
        for k in range(2):
            var dr = abs(
                Float64(
                    mf.joints.data[
                        j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MIN + k
                    ]
                )
                - Float64(py=m.jnt_range[j][k])
            )
            if dr > worst_range:
                worst_range = dr
    # ⚠ `jnt_range` IS NO LONGER AN ORDER DISCRIMINATOR. It was one while the
    # parser emitted elements in XML text order; now that `full_parser` groups
    # by body, ARMATURE is the order proof — it is per-dof and matches at
    # exactly 0.0, which cannot happen under a permutation. A `jnt_range`
    # mismatch at this point means a wrong RANGE, not a wrong order, and
    # reading it as "order differs" sends the next person to the wrong file.
    print("  max |d(jnt_range)| =", worst_range,
          " over", n_limited, "LIMITED joints")
    var first_r = -1
    for j in range(NJOINT):
        if Int(py=m.jnt_limited[j]) == 0:
            continue
        var d0 = abs(
            Float64(mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MIN])
            - Float64(py=m.jnt_range[j][0])
        )
        var d1 = abs(
            Float64(
                mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MIN + 1]
            )
            - Float64(py=m.jnt_range[j][1])
        )
        if d0 > MODEL_TOL or d1 > MODEL_TOL:
            first_r = j
            break
    if first_r >= 0:
        for j in range(first_r, min(first_r + 4, NJOINT)):
            print(
                "      jnt", j,
                String(py=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)),
                " type", Int(py=m.jnt_type[j]),
                " limited", Int(py=m.jnt_limited[j]),
                " range ours",
                Float64(
                    mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MIN]
                ),
                Float64(
                    mf.joints.data[
                        j * MODEL_JOINT_SIZE + JOINT_IDX_RANGE_MIN + 1
                    ]
                ),
                " mj", Float64(py=m.jnt_range[j][0]),
                Float64(py=m.jnt_range[j][1]),
            )
    # Same for stiffness, which is off by ~0.06 on at least one joint.
    var first_s = -1
    for j in range(NJOINT):
        if abs(
            Float64(mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_STIFFNESS])
            - Float64(py=m.jnt_stiffness[j])
        ) > MODEL_TOL:
            first_s = j
            break
    if first_s >= 0:
        for j in range(first_s, min(first_s + 4, NJOINT)):
            print(
                "      jnt", j,
                String(py=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)),
                " stiffness ours",
                Float64(
                    mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_STIFFNESS]
                ),
                " mj", Float64(py=m.jnt_stiffness[j]),
            )
    print("  max |d(armature)| =", worst_arm, " |d(stiffness)| =", worst_stiff)
    # Name the first offender. `armature` feeds `dof_invweight0`, so a wrong
    # one shows up twice and the two failures look independent when they are
    # not.
    var first_j = -1
    for j in range(1, NJOINT):
        var dof = j + 5
        if abs(
            Float64(mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_ARMATURE])
            - Float64(py=m.dof_armature[dof])
        ) > MODEL_TOL:
            first_j = j
            break
    if first_j >= 0:
        for j in range(first_j, min(first_j + 5, NJOINT)):
            var dof = j + 5
            print(
                "      jnt", j,
                String(py=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)),
                " armature ours",
                Float64(
                    mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_ARMATURE]
                ),
                " mj", Float64(py=m.dof_armature[dof]),
                " | stiffness ours",
                Float64(
                    mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_STIFFNESS]
                ),
                " mj", Float64(py=m.jnt_stiffness[j]),
            )
    print("  joints with non-zero armature:", n_nonzero_arm, "/", NJOINT - 1)

    # ── the three passive tables nothing compared until 2026-08-05 ──────────
    #
    # ⚠ A MATCHING `stiffness` IS NO EVIDENCE ABOUT `springref` OR `damping`.
    # This test compared armature and stiffness and stopped there, and both of
    # the others were wrong in ways stiffness could not show:
    #
    #   * `springref` was never converted from DEGREES. dog's jaw spells
    #     `springref="-11.0"`, so its spring pulled towards -11 RADIANS. That
    #     one number is what made the whole loaded-pose step diverge.
    #   * `AutoSpringDamper` derives stiffness AND damping from `springdamper`
    #     by two DIFFERENT formulas (`inertia/(tc^2 dr^2)` vs `2 inertia/tc`),
    #     so a correct stiffness says nothing about the damping beside it.
    #
    # `frictionloss` is included because it is the third passive scalar on the
    # same element and the same class-resolution path builds all three.
    var worst_sref = 0.0
    var worst_damp = 0.0
    var worst_floss = 0.0
    var first_sref = -1
    var n_sref_nonzero = 0
    for j in range(NJOINT):
        var o = j * MODEL_JOINT_SIZE
        var adr = Int(Float64(mf.joints.data[o + JOINT_IDX_QPOS_ADR]))
        # MuJoCo keeps the spring reference in `qpos_spring`, indexed by qpos
        # address — NOT in a per-joint table. For FREE and BALL joints it
        # holds a copy of `qpos0` (a position / quaternion, not a scalar
        # reference), so only the hinges are comparable here; dog's free root
        # carries `stiffness = 0`, which makes the slot inert anyway.
        if Int(py=m.jnt_type[j]) != 3:  # mjJNT_HINGE
            continue
        if Float64(py=m.qpos_spring[adr]) != 0.0:
            n_sref_nonzero += 1
        var ds = abs(
            Float64(mf.joints.data[o + JOINT_IDX_SPRINGREF])
            - Float64(py=m.qpos_spring[adr])
        )
        if ds > worst_sref:
            worst_sref = ds
        if ds > MODEL_TOL and first_sref < 0:
            first_sref = j
    for j in range(1, NJOINT):
        var o = j * MODEL_JOINT_SIZE
        var dof = j + 5
        var dd = abs(
            Float64(mf.joints.data[o + JOINT_IDX_DAMPING])
            - Float64(py=m.dof_damping[dof])
        )
        if dd > worst_damp:
            worst_damp = dd
        var df = abs(
            Float64(mf.joints.data[o + JOINT_IDX_FRICTIONLOSS])
            - Float64(py=m.dof_frictionloss[dof])
        )
        if df > worst_floss:
            worst_floss = df
    print("  max |d(springref)| =", worst_sref,
          " over hinges with a non-zero spring reference:", n_sref_nonzero)
    print("  max |d(damping)| =", worst_damp,
          " |d(frictionloss)| =", worst_floss)
    if first_sref >= 0:
        for j in range(first_sref, min(first_sref + 3, NJOINT)):
            var o = j * MODEL_JOINT_SIZE
            var adr = Int(Float64(mf.joints.data[o + JOINT_IDX_QPOS_ADR]))
            print(
                "      jnt", j,
                String(py=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)),
                " springref ours",
                Float64(mf.joints.data[o + JOINT_IDX_SPRINGREF]),
                " mj", Float64(py=m.qpos_spring[adr]),
                " (a ratio of ~57.3 means the DEGREE conversion is missing)",
            )
    # NON-VACUITY: dog has exactly one hinge with a non-zero spring reference
    # (the jaw). If that ever drops to zero the comparison above is `0 == 0`
    # on all 73 hinges and proves nothing about the conversion.
    assert_true(
        n_sref_nonzero >= 1,
        "no dog hinge has a non-zero spring reference — the springref"
        " comparison is vacuous and the degree conversion is untested here",
    )
    assert_true(
        worst_sref <= MODEL_TOL,
        "joint SPRINGREF differs from MuJoCo's qpos_spring — a degree model's"
        " hinge springref must be scaled by pi/180"
        " (tests/physics3d/test_springref_degrees_vs_mujoco.mojo isolates it)",
    )
    assert_true(
        worst_damp <= MODEL_TOL,
        "dof DAMPING differs from MuJoCo — check AutoSpringDamper's"
        " `2*inertia/tc`, which is a different formula from the stiffness"
        " beside it and is not covered by a matching stiffness",
    )
    assert_true(
        worst_floss <= MODEL_TOL, "dof frictionloss differs from MuJoCo"
    )
    # Armature is the ORDER proof: per-dof, and any permutation shows in it.
    assert_true(
        worst_arm <= MODEL_TOL,
        "dof armature differs from MuJoCo — if this is nonzero AND jnt_range"
        " is too, suspect element ORDER (`_stable_group_by_body_*`); if only"
        " armature is off, suspect default-class resolution",
    )
    assert_true(
        n_limited == NJOINT - 1,
        "dog's root <joint> default sets limited=true, so all 73 hinges are"
        " limited and only the free root is not — if that changed, the range"
        " comparison above is covering a different set than it claims",
    )
    assert_true(
        worst_range <= MODEL_TOL,
        "joint RANGE differs from MuJoCo (order is proved fine by armature"
        " above) — see the per-joint dump for which joint and whether it is a"
        " limited/unlimited representation difference",
    )
    assert_true(worst_stiff <= MODEL_TOL, "joint stiffness differs from MuJoCo")
    assert_true(
        n_nonzero_arm == NJOINT - 1,
        "some hinge has zero armature — dog's root <joint> default sets"
        " armature=0.0001 on every one of them",
    )

    # Inertials. THIS IS THE GATE ON THE BAKE, from our side: every one of
    # these numbers was written into the XML by `bake_xml` rather than derived
    # by a compiler from mesh volume.
    var worst_mass = 0.0
    var worst_ipos = 0.0
    var worst_inertia = 0.0
    var total_mass = 0.0
    for b in range(NBODY):
        var our_mass = Float64(
            mf.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_MASS]
        )
        total_mass += our_mass
        var dm = abs(our_mass - Float64(py=m.body_mass[b]))
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
    print("  total mass =", total_mass, "kg")
    assert_true(worst_mass <= MODEL_TOL, "masses differ from MuJoCo")
    assert_true(worst_ipos <= MODEL_TOL, "body CoMs differ from MuJoCo")
    assert_true(worst_inertia <= MODEL_TOL, "inertias differ from MuJoCo")
    assert_true(
        total_mass > 1.0,
        "total mass is near zero — the baked <inertial> elements are not"
        " reaching our parser, which is exactly what the bake risks",
    )


def test_dog_priority_and_condim_reach_our_model() raises:
    """The 42 teeth keep `condim=6` and `priority=2` through OUR parser.

    dog is the first ported model where `<geom priority>` decides anything:
    a tooth against the condim-3 floor takes the TOOTH's condim, friction and
    solref wholesale, so a dropped priority column silently downgrades those
    contacts to a blend.

    ⚠ `grep` CANNOT ANSWER THIS. `condim="6"` and `priority="2"` appear ONCE
    each in dog.xml, inside `<default class="tooth_primitive">`; the 42 geoms
    that carry them say only `class="tooth_primitive"`. Both counts below come
    from the compiled model.
    """
    print("--- dog: priority + condim survive our parser ---")
    var mujoco = Python.import_module("mujoco")
    var m = _mj_from_our_xml()
    var mf = _build()

    var ref_c6 = 0
    var ref_prio = 0
    var our_c6 = 0
    var our_prio = 0
    var worst = 0
    for g in range(NGEOM):
        var rc = Int(py=m.geom_condim[g])
        var rp = Int(py=m.geom_priority[g])
        var oc = Int(mf.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_CONDIM])
        var op = Int(mf.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_PRIORITY])
        if rc == 6:
            ref_c6 += 1
        if rp != 0:
            ref_prio += 1
        if oc == 6:
            our_c6 += 1
        if op != 0:
            our_prio += 1
        if abs(rc - oc) > worst:
            worst = abs(rc - oc)
        if abs(rp - op) > worst:
            worst = abs(rp - op)

    print("  condim==6 geoms: MuJoCo", ref_c6, " ours", our_c6)
    print("  priority!=0 geoms: MuJoCo", ref_prio, " ours", our_prio)

    # Histogram both sides. `ours 0` alone cannot distinguish "the class
    # fallback failed" from "our geoms table is defaults" from "our geom
    # ORDER differs" — and a three-body control model gets condim, priority
    # AND invweight exactly right, so the failure is specific to this model
    # and the shape of it is the whole question.
    var hist_ours = InlineArray[Int, 8](fill=0)
    var hist_ref = InlineArray[Int, 8](fill=0)
    for g in range(NGEOM):
        var oc = Int(mf.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_CONDIM])
        var rc = Int(py=m.geom_condim[g])
        if oc >= 0 and oc < 8:
            hist_ours[oc] += 1
        if rc >= 0 and rc < 8:
            hist_ref[rc] += 1
    for k in range(8):
        if hist_ours[k] != 0 or hist_ref[k] != 0:
            print("    condim", k, ": ours", hist_ours[k], " MuJoCo", hist_ref[k])
    var first_bad = -1
    for g in range(NGEOM):
        if Int(mf.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_CONDIM]) != Int(
            py=m.geom_condim[g]
        ):
            first_bad = g
            break
    print("    first differing geom index:", first_bad, "of", NGEOM)
    if first_bad >= 0:
        # Counts agree, so this is a PERMUTATION, not lost data. Print a
        # window around the divergence with MuJoCo's geom NAMES so the shape
        # of the permutation is visible — our `fields.Model` carries no names,
        # which is exactly why this has to be read off the reference side.
        var lo = first_bad - 2
        if lo < 0:
            lo = 0
        var hi = lo + 8
        if hi > NGEOM:
            hi = NGEOM
        for g in range(lo, hi):
            print(
                "      g", g,
                " ours", Int(mf.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_CONDIM]),
                " mj", Int(py=m.geom_condim[g]),
                " mj_name",
                String(py=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)),
                " mj_body",
                String(
                    py=mujoco.mj_id2name(
                        m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]
                    )
                ),
            )

    # The COUNTS are order-independent; the per-index diff is not. Our parser
    # orders geoms by XML text and MuJoCo orders them by body, so separating
    # the two says which kind of failure this is: equal counts with a nonzero
    # per-index diff means the VALUES are right and the ORDER is not.
    assert_true(
        our_c6 == ref_c6 and our_prio == ref_prio,
        "our parser lost condim-6 or priority geoms entirely — check the"
        " default-CLASS fallback, which `priority` did not have until dog",
    )
    # ⚠ DO NOT ASSERT PER-INDEX EQUALITY HERE. Our parser emits geoms in XML
    # TEXT order and MuJoCo emits them grouped BY BODY, and dog is the first
    # ported model where the two differ: `skull` declares its 42 teeth AFTER
    # its nested child bodies, so from index 64 our sequence runs four ahead
    # of MuJoCo's (measured — the eye geoms `iris_*`/`pupil_*` are the four).
    # That is a pre-existing, documented property of this parser
    # (`mjmodel_diff`'s docstring says so) and it is benign for dog, whose
    # config addresses no geom by index. The MULTISET is what has to match, and
    # the histogram above is that comparison.
    print("    (per-index diff is expected: text order vs body order)")
    assert_true(
        ref_c6 == 42 and ref_prio == 42,
        "dog no longer has 42 condim-6 priority-2 teeth — if that is real this"
        " model stopped exercising the Phase 3 friction rows entirely",
    )
    assert_true(
        DMDogStandWalkModel.MAX_CONDIM == 6,
        "MAX_CONDIM did not reach 6 — `_scan_max_condim` missed the"
        " `<default class=\"tooth_primitive\">` block, and the pyramidal edge"
        " builder is back to 4 edges with mu_slide on all of them",
    )


def test_dog_invweight_matches_mujoco() raises:
    """`body_invweight0` / `dof_invweight0`, per the standing rule.

    That rule exists because a 64x contact-stiffness error once hid behind five
    green domains. These are derived quantities — a mass or inertia that is
    individually within tolerance can still compose into the wrong effective
    inertia at a contact — so they are checked separately from the inertials.
    """
    print("--- dog: invweight0 vs MuJoCo ---")
    var np = Python.import_module("numpy")
    var m = _mj_from_our_xml()
    var mf = _build()

    # ⚠ THIS MUST READ OUR `fields.Model`. The first draft compared
    # `_mj_from_our_xml()` against `dog_ref.model()` — MuJoCo against MuJoCo,
    # which is layer 1's job and which layer 1 already does over the very same
    # two tables. It reported 0.0 and gated NOTHING about our engine.
    var worst_b = 0.0
    var scale_b = 0.0
    for b in range(NBODY):
        var d0 = abs(
            Float64(mf.body_invweight0.data[b * 2 + 0])
            - Float64(py=m.body_invweight0[b][0])
        )
        var d1 = abs(
            Float64(mf.body_invweight0.data[b * 2 + 1])
            - Float64(py=m.body_invweight0[b][1])
        )
        if d0 > worst_b:
            worst_b = d0
        if d1 > worst_b:
            worst_b = d1
        var r0 = abs(Float64(py=m.body_invweight0[b][0]))
        var r1 = abs(Float64(py=m.body_invweight0[b][1]))
        if r0 > scale_b:
            scale_b = r0
        if r1 > scale_b:
            scale_b = r1

    var worst_d = 0.0
    var scale = 0.0
    # ⚠ PER-ELEMENT RELATIVE, NOT GLOBAL-MAX RELATIVE.
    # This test used to divide the worst ABSOLUTE error by the LARGEST
    # `dof_invweight0` in the model. dog's span ten orders — up to ~1e10 — so
    # normalising by the maximum means a dof whose invweight is 1.4e3 could be
    # wrong by 100% and still score 1e-10 and pass. That is not a strict gate,
    # it is a gate that cannot see most of the model, and the JOINT LIMIT rows
    # read exactly those small entries (`R = (1-imp)/imp * dof_invweight0`).
    var worst_rel_d = 0.0
    var worst_rel_i = -1
    for i in range(NV):
        var ref_v = Float64(py=m.dof_invweight0[i])
        var dd = abs(Float64(mf.dof_invweight0.data[i]) - ref_v)
        if dd > worst_d:
            worst_d = dd
        if abs(ref_v) > scale:
            scale = abs(ref_v)
        var denom = abs(ref_v) if abs(ref_v) > 1e-30 else 1.0
        var rel_i = dd / denom
        if rel_i > worst_rel_d:
            worst_rel_d = rel_i
            worst_rel_i = i
    if worst_rel_i >= 0:
        print(
            "  worst PER-ELEMENT dof_invweight0: dof", worst_rel_i,
            " ours", Float64(mf.dof_invweight0.data[worst_rel_i]),
            " mj", Float64(py=m.dof_invweight0[worst_rel_i]),
            " rel", worst_rel_d,
        )

    print("  max |d(body_invweight0)| =", worst_b)
    print("  max |d(dof_invweight0)|  =", worst_d,
          " (dof_invweight0 range up to", scale, ")")
    # ⚠ RELATIVE, NOT ABSOLUTE. `invweight0` is ~1/mass and dog's lightest
    # bodies are ~1.5e-3 kg, so the quantity spans ten orders and reaches
    # 1e10. An absolute 1e-14 budget on a value of 1e10 is below float64's own
    # resolution — it can never pass, which makes it a broken gate rather than
    # a strict one. Measured on a three-body control model, our invweight
    # agrees with MuJoCo to ~1e-12 relative across that whole range.
    var rel_b = worst_b / (scale_b if scale_b > 1.0 else 1.0)
    var rel_d = worst_d / (scale if scale > 1.0 else 1.0)
    print("  relative: body", rel_b, " dof", rel_d)
    assert_true(rel_b <= 1e-10, "body_invweight0 differs from MuJoCo")
    assert_true(rel_d <= 1e-10, "dof_invweight0 differs from MuJoCo")
    assert_true(
        worst_rel_d <= 1e-10,
        "dof_invweight0 differs from MuJoCo on a SINGLE dof by more than 1e-10"
        " RELATIVE TO THAT DOF — the global-max form of this check above"
        " cannot see it, and the joint-limit rows read these entries directly",
    )
    assert_true(
        scale > 0.0,
        "every dof_invweight0 is zero — this comparison is vacuous",
    )


def test_dog_shape_constants() raises:
    """The observation width and frame skip, derived rather than asserted."""
    print("--- dog: observation width and timing ---")
    var expect = (
        DOG_N_HINGE  # joint_angles
        + DOG_N_HINGE  # joint_velocites
        + 2  # torso_pelvis_height
        + 9  # z_projection
        + 3  # torso_com_velocity
        + 9  # inertial_sensors
        + 12  # foot_forces
        + 4  # touch_sensors
        + NACT  # actuator_state
    )
    print("  obs dim =", DOG_OBS_DIM, " recomputed", expect)
    assert_true(DOG_OBS_DIM == expect, "DOG_OBS_DIM does not add up")
    assert_true(
        DOG_FRAME_SKIP == 3,
        "control_timestep .015 / timestep .005 = 3",
    )
    assert_true(
        abs(dsp.TIMESTEP - 0.005) < 1e-15,
        "dog.xml sets timestep .005",
    )
    # Every actuator has an activation state, and the observation exposes all
    # of them — `assert physics.model.nu == physics.model.na` is in dog.py.
    assert_true(
        NACT == 38,
        "dog has 38 dyntype=filter actuators; data.act is 38 wide and is 38 of"
        " the observation's 223 numbers",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
