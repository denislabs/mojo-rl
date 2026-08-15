"""dm_control `swimmer` — MODEL + DYNAMICS parity against the reference.

Swimmer is the first PROCEDURAL model in the port: there is no `swimmer6.xml`
to load, so both sides run a generator. Ours is `_swimmer_body_xml(n)` at
comptime; the reference side is `tests/dm_control/swimmer_ref.py`, a verbatim
copy of `suite/swimmer.py::_make_model` with lxml swapped for the stdlib
ElementTree (see that file for why an import is not possible here).

It is also the first model that turns the FLUID path on (`density="3000"`) and
the first with contacts disabled model-wide, so `dynamics/fluid_forces.mojo` —
MuJoCo's `mj_inertiaBoxFluidModel` — goes from never-executed to being the only
force that converts joint torque into locomotion. `test_swimmer_fluid_drag_*`
gates it in isolation before the rollouts gate it in composition.

TWO ENGINE BUGS THIS DOMAIN FOUND, both silent, both gated below:

  * BUG 24 — the comptime XML path's `<default>` lookups were not nested-default
    aware, so a top-level `<default>` element declared AFTER a named
    `<default class="...">` block was invisible. swimmer declares
    `<motor gear="5e-4"/>` exactly there, so gear fell back to MuJoCo's 1.0:
    a 2000x actuator force error, and the entire dynamics of the domain. The
    same truncation read the `swimmer` class's `limited="true"` as the global
    default, marking the three UNLIMITED root DOFs limited with a (0, 0) range.
    Fixed by `xml_parser._root_defaults`; gated by
    `test_swimmer_actuators_match_mujoco` and `test_swimmer_joints_match_mujoco`.

  * The `limited` fallback now follows MuJoCo's `compiler/autolimits` (a joint
    with a `range` and no explicit `limited` IS limited), which is what lets a
    class-blind scan agree with the compiler on a model that puts `limited` on
    two different default classes.

Run:
    pixi run mojo run -I . tests/dm_control/test_swimmer_vs_dm_control.mojo
"""

from std.testing import assert_true, assert_equal, TestSuite
from std.python import Python, PythonObject
from std.math import abs, sin, sqrt
from max.gpu.host import DeviceContext

from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.physics3d.fields import Model
from mojo_rl.physics3d.model.model_def import ModelDefLike
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    BODY_IDX_MASS,
    BODY_IDX_IXX,
    BODY_IDX_IYY,
    BODY_IDX_IZZ,
    BODY_IDX_IPOS_X,
    BODY_IDX_IPOS_Y,
    BODY_IDX_IPOS_Z,
    BODY_IDX_IQUAT_X,
    BODY_IDX_IQUAT_Y,
    BODY_IDX_IQUAT_Z,
    BODY_IDX_IQUAT_W,
    BODY_IDX_POS_X,
    BODY_IDX_POS_Y,
    BODY_IDX_POS_Z,
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    JOINT_IDX_ARMATURE,
    JOINT_IDX_DAMPING,
    JOINT_IDX_SOLREF_LIMIT_0,
    JOINT_IDX_SOLREF_LIMIT_1,
    JOINT_IDX_SOLIMP_LIMIT_0,
    JOINT_IDX_SOLIMP_LIMIT_1,
    JOINT_IDX_SOLIMP_LIMIT_2,
    MODEL_GEOM_SIZE,
    GEOM_IDX_BODY,
    GEOM_IDX_POS_X,
    GEOM_IDX_POS_Y,
    GEOM_IDX_POS_Z,
    MODEL_META_IDX_DENSITY,
    MODEL_META_IDX_VISCOSITY,
)

from mojo_rl.envs.dm_control.rewards import tolerance, SIGMOID_LONG_TAIL
from mojo_rl.envs.dm_control.swimmer.swimmer_config import DMSwimmerConfig
from mojo_rl.envs.dm_control.swimmer.swimmer_xml import (
    DMSwimmer6Model,
    DMSwimmer15Model,
    HEAD_BODY_IDX,
    NOSE_GEOM_IDX,
    N_ROOT_DOF,
    TARGET_Z,
)
from mojo_rl.physics3d.gpu.constants import (
    MODEL_ACTUATOR_SIZE,
    ACT_IDX_GEAR,
    ACT_IDX_CTRL_MIN,
    ACT_IDX_CTRL_MAX,
    ACT_IDX_DOF_ADR,
)

comptime DTYPE = DType.float64
comptime REF_HELPER_PATH: StaticString = "tests/dm_control"

comptime MODEL_TOL: Float64 = 1e-13

# Rollout gates. `_CONTROL_TIMESTEP` .03 over a .002 physics step => 15
# substeps, so 80 control steps is 1200 integrator steps of accumulation.
comptime FRAME_SKIP_S: Int = 15
comptime STATE_TOL: Float64 = 1e-8
comptime OBS_TOL: Float64 = 1e-8
comptime REWARD_TOL: Float64 = 1e-10


def _ref(n_bodies: Int) raises -> PythonObject:
    """The reference `mjModel` for an `n_bodies`-link swimmer."""
    var sys = Python.import_module("sys")
    sys.path.insert(0, REF_HELPER_PATH)
    var swimmer_ref = Python.import_module("swimmer_ref")
    return swimmer_ref.model(n_bodies)


def _close(a: Float64, b: Float64) -> Bool:
    return abs(a - b) <= MODEL_TOL * (1.0 + abs(b))


# ── counts + structure ───────────────────────────────────────────────────────


def test_swimmer6_counts() raises:
    """Counts match, with the mocap target the single accounted-for extra."""
    comptime M = DMSwimmer6Model
    var mj = _ref(6)
    print(
        "  ours  NBODY", M.NBODY, " NJOINT", M.NJOINT, " NQ", M.NQ,
        " NV", M.NV, " NGEOM", M.NGEOM, " NSITE", M.NSITE,
        " NACT", M.ACTION_DIM,
    )
    print(
        "  mjcf  nbody", Int(py=mj.nbody), " njnt", Int(py=mj.njnt),
        " nq", Int(py=mj.nq), " nv", Int(py=mj.nv),
        " ngeom", Int(py=mj.ngeom), " nsite", Int(py=mj.nsite),
        " nu", Int(py=mj.nu),
    )
    # The target moves from a static worldbody geom onto a mocap BODY, so we
    # gain one body and no geoms.
    assert_equal(
        M.NBODY, Int(py=mj.nbody) + 1,
        "NBODY should be the reference's + 1 (the mocap target body)",
    )
    assert_equal(M.NJOINT, Int(py=mj.njnt), "joint count")
    assert_equal(M.NQ, Int(py=mj.nq), "nq — the mocap body must add no DOF")
    assert_equal(M.NV, Int(py=mj.nv), "nv — the mocap body must add no DOF")
    assert_equal(M.NGEOM, Int(py=mj.ngeom), "geom count")
    assert_equal(M.NSITE, Int(py=mj.nsite), "site count")
    assert_equal(M.ACTION_DIM, Int(py=mj.nu), "actuator count")
    # obs = joints (5) + to_target (2) + body_velocities (18)
    assert_equal(M.OBS_DIM, 25, "swimmer6 observation width")


def test_swimmer15_counts() raises:
    """The generator has to scale: 15 links, not a hardcoded 6."""
    comptime M = DMSwimmer15Model
    var mj = _ref(15)
    print(
        "  ours  NBODY", M.NBODY, " NJOINT", M.NJOINT, " NQ", M.NQ,
        " NGEOM", M.NGEOM, " NSITE", M.NSITE, " NACT", M.ACTION_DIM,
    )
    assert_equal(M.NBODY, Int(py=mj.nbody) + 1, "NBODY (+1 mocap target)")
    assert_equal(M.NJOINT, Int(py=mj.njnt), "joint count")
    assert_equal(M.NQ, Int(py=mj.nq), "nq")
    assert_equal(M.NV, Int(py=mj.nv), "nv")
    assert_equal(M.NGEOM, Int(py=mj.ngeom), "geom count")
    assert_equal(M.NSITE, Int(py=mj.nsite), "site count")
    assert_equal(M.ACTION_DIM, Int(py=mj.nu), "actuator count")
    # obs = joints (14) + to_target (2) + body_velocities (45)
    assert_equal(M.OBS_DIM, 61, "swimmer15 observation width")


def test_swimmer_fluid_option_is_parsed() raises:
    """`<option density="3000">` — without it the domain has no locomotion."""
    comptime M = DMSwimmer6Model
    var mj = _ref(6)
    var ctx = DeviceContext()
    var mf = Model[
        DTYPE, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields[DTYPE, 0](ctx, mf)

    var dens = Float64(mf.meta.data[MODEL_META_IDX_DENSITY])
    var visc = Float64(mf.meta.data[MODEL_META_IDX_VISCOSITY])
    print("  density", dens, " viscosity", visc, " timestep", M.TIMESTEP)
    assert_true(_close(dens, Float64(py=mj.opt.density)), "option/density")
    assert_true(_close(visc, Float64(py=mj.opt.viscosity)), "option/viscosity")
    assert_true(
        _close(Float64(M.TIMESTEP), Float64(py=mj.opt.timestep)),
        "option/timestep",
    )
    # Non-vacuity: this whole domain is the density term.
    assert_true(
        dens > 0.0,
        "density parsed as 0 — the fluid path early-outs and the swimmer"
        " becomes a frictionless linkage that cannot swim",
    )


def test_swimmer_bodies_match_mujoco() raises:
    """Mass, diagonal inertia, `ipos`/`iquat` and body pose, per link.

    `iquat` matters beyond bookkeeping: `mj_inertiaBoxFluidModel` rotates the
    drag wrench with `ximat` (the INERTIAL frame), while our port rotates with
    `xquat` (the BODY frame). Those agree only while `iquat` is identity, which
    it is for a box-inertia swimmer — asserted here rather than assumed.
    """
    comptime M = DMSwimmer6Model
    var mj = _ref(6)
    var ctx = DeviceContext()
    var mf = Model[
        DTYPE, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields[DTYPE, 0](ctx, mf)

    var mm = mj.body_mass.tolist()
    var mi = mj.body_inertia.tolist()
    var mip = mj.body_ipos.tolist()
    var miq = mj.body_iquat.tolist()
    var mbp = mj.body_pos.tolist()

    var worst = Float64(0)
    for b in range(Int(py=mj.nbody)):
        var bo = b * MODEL_BODY_SIZE
        var pairs = [
            (Float64(mf.bodies.data[bo + BODY_IDX_MASS]), Float64(py=mm[b])),
            (Float64(mf.bodies.data[bo + BODY_IDX_IXX]), Float64(py=mi[b][0])),
            (Float64(mf.bodies.data[bo + BODY_IDX_IYY]), Float64(py=mi[b][1])),
            (Float64(mf.bodies.data[bo + BODY_IDX_IZZ]), Float64(py=mi[b][2])),
            (Float64(mf.bodies.data[bo + BODY_IDX_IPOS_X]), Float64(py=mip[b][0])),
            (Float64(mf.bodies.data[bo + BODY_IDX_IPOS_Y]), Float64(py=mip[b][1])),
            (Float64(mf.bodies.data[bo + BODY_IDX_IPOS_Z]), Float64(py=mip[b][2])),
            (Float64(mf.bodies.data[bo + BODY_IDX_POS_X]), Float64(py=mbp[b][0])),
            (Float64(mf.bodies.data[bo + BODY_IDX_POS_Y]), Float64(py=mbp[b][1])),
            (Float64(mf.bodies.data[bo + BODY_IDX_POS_Z]), Float64(py=mbp[b][2])),
        ]
        for k in range(len(pairs)):
            var e = abs(pairs[k][0] - pairs[k][1])
            if e > worst:
                worst = e
            assert_true(
                _close(pairs[k][0], pairs[k][1]),
                String("body ") + String(b) + " field " + String(k),
            )
        # MuJoCo stores iquat as (w, x, y, z); ours as (x, y, z, w).
        assert_true(
            abs(Float64(mf.bodies.data[bo + BODY_IDX_IQUAT_W]) - Float64(py=miq[b][0])) <= 1e-12
            and abs(Float64(mf.bodies.data[bo + BODY_IDX_IQUAT_X]) - Float64(py=miq[b][1])) <= 1e-12
            and abs(Float64(mf.bodies.data[bo + BODY_IDX_IQUAT_Y]) - Float64(py=miq[b][2])) <= 1e-12
            and abs(Float64(mf.bodies.data[bo + BODY_IDX_IQUAT_Z]) - Float64(py=miq[b][3])) <= 1e-12,
            String("body ") + String(b) + " iquat",
        )
        # The fluid model's frame assumption (see the docstring).
        assert_true(
            abs(Float64(py=miq[b][0]) - 1.0) <= 1e-12,
            String("body ") + String(b) + ": iquat is no longer identity, so"
            " fluid_forces.mojo rotating drag by `xquat` instead of MuJoCo's"
            " `ximat` is no longer exact",
        )
    print("  worst body field abs err =", worst)


def test_swimmer_joints_match_mujoco() raises:
    """Type, limits, armature, damping and the LIMIT solver parameters.

    `solimplimit="0 .8 .1"` puts `dmin` at 0, i.e. the case that only exists
    because solimp is clamped to [mjMINIMP, mjMAXIMP] before use (bug 21).
    Asserted as a value here and exercised for real by the saturating rollout.

    The three root DOFs are the bug-24 half: they are UNLIMITED, and the
    truncated `<default>` scan used to mark them limited with a (0, 0) range.
    """
    comptime M = DMSwimmer6Model
    var mj = _ref(6)
    var ctx = DeviceContext()
    var mf = Model[
        DTYPE, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields[DTYPE, 0](ctx, mf)

    var jr = mj.jnt_range.tolist()
    var jlim = mj.jnt_limited.tolist()
    var jtype = mj.jnt_type.tolist()
    var jsolref = mj.jnt_solref.tolist()
    var jsolimp = mj.jnt_solimp.tolist()
    var mj_arm = mj.dof_armature.tolist()
    var mj_damp = mj.dof_damping.tolist()
    var mj_dadr = mj.jnt_dofadr.tolist()

    var saw_unlimited = False
    var saw_zero_dmin = False
    for j in range(M.NJOINT):
        var jo = j * MODEL_JOINT_SIZE
        var lo = Float64(mf.joints.data[jo + JOINT_IDX_RANGE_MIN])
        var hi = Float64(mf.joints.data[jo + JOINT_IDX_RANGE_MAX])
        var d_adr = Int(py=mj_dadr[j])
        print(
            "   joint", j, " type", Int(mf.joints.data[jo + JOINT_IDX_TYPE]),
            " range [", lo, ",", hi, "] armature",
            Float64(mf.joints.data[jo + JOINT_IDX_ARMATURE]),
            " solimp dmin",
            Float64(mf.joints.data[jo + JOINT_IDX_SOLIMP_LIMIT_0]),
        )
        if Int(py=jlim[j]) != 0:
            assert_true(_close(lo, Float64(py=jr[j][0])), "jnt_range min")
            assert_true(_close(hi, Float64(py=jr[j][1])), "jnt_range max")
        else:
            saw_unlimited = True
            assert_true(
                lo < -1e9 and hi > 1e9,
                String("joint ") + String(j) + " is UNLIMITED in MuJoCo but"
                " does not carry our unlimited sentinel — the limit builder"
                " would invent a constraint row pinning it (bug 24)",
            )
        assert_true(
            _close(
                Float64(mf.joints.data[jo + JOINT_IDX_ARMATURE]),
                Float64(py=mj_arm[d_adr]),
            ),
            "dof_armature",
        )
        assert_true(
            _close(
                Float64(mf.joints.data[jo + JOINT_IDX_DAMPING]),
                Float64(py=mj_damp[d_adr]),
            ),
            "dof_damping",
        )
        # Limit solver parameters, only meaningful on a limited joint.
        if Int(py=jlim[j]) != 0:
            assert_true(
                _close(
                    Float64(mf.joints.data[jo + JOINT_IDX_SOLREF_LIMIT_0]),
                    Float64(py=jsolref[j][0]),
                )
                and _close(
                    Float64(mf.joints.data[jo + JOINT_IDX_SOLREF_LIMIT_1]),
                    Float64(py=jsolref[j][1]),
                ),
                "jnt_solref (solreflimit)",
            )
            assert_true(
                _close(
                    Float64(mf.joints.data[jo + JOINT_IDX_SOLIMP_LIMIT_0]),
                    Float64(py=jsolimp[j][0]),
                )
                and _close(
                    Float64(mf.joints.data[jo + JOINT_IDX_SOLIMP_LIMIT_1]),
                    Float64(py=jsolimp[j][1]),
                )
                and _close(
                    Float64(mf.joints.data[jo + JOINT_IDX_SOLIMP_LIMIT_2]),
                    Float64(py=jsolimp[j][2]),
                ),
                "jnt_solimp (solimplimit)",
            )
            if abs(Float64(py=jsolimp[j][0])) <= 1e-15:
                saw_zero_dmin = True

    assert_true(
        saw_unlimited,
        "no unlimited joint left in the model — this gate no longer covers the"
        " root-DOF half of bug 24",
    )
    assert_true(
        saw_zero_dmin,
        "no limited joint has solimp dmin == 0 any more — this model no longer"
        " exercises the impedance clamp (bug 21)",
    )


def test_swimmer_actuators_match_mujoco() raises:
    """THE BUG 24 GATE: `gear` lives in a top-level `<default><motor>` that is
    declared AFTER two named `<default class="...">` blocks.

    The comptime `<default>` scan was not depth aware, so it never saw the
    `<motor>` at all and gear silently fell back to MuJoCo's 1.0 — 2000x the
    real 5e-4. Nothing raised; the swimmer simply thrashed. Compared against
    `actuator_gear` rather than the literal so the XML and the parser have to
    agree with the compiler, not with each other.
    """
    comptime M = DMSwimmer6Model
    var mj = _ref(6)
    var gear = mj.actuator_gear.tolist()
    var cr = mj.actuator_ctrlrange.tolist()
    var dofadr = mj.actuator_trnid.tolist()
    var jdof = mj.jnt_dofadr.tolist()

    # ⚠ THE `comptime for` IS GONE AND SO IS THE REASON FOR IT. These used to
    # read `M._acd`, a COMPTIME value whose `Array` fields cannot be indexed by
    # a runtime `i` (rc2 dropped `ImplicitlyCopyable` on `Array`), so every
    # element had to be hoisted under a comptime index. `SpecFields` is a
    # runtime tensor; a plain loop reads it.
    var sf = M.make_spec_fields[DType.float64]()
    var a_gears = List[Float64]()
    var a_cmin = List[Float64]()
    var a_cmax = List[Float64]()
    var a_dof_adr = List[Int]()
    for a in range(M.ACTION_DIM):
        a_gears.append(
            Float64(sf.actuators.data[a * MODEL_ACTUATOR_SIZE + ACT_IDX_GEAR])
        )
        a_cmin.append(
            Float64(
                sf.actuators.data[a * MODEL_ACTUATOR_SIZE + ACT_IDX_CTRL_MIN]
            )
        )
        a_cmax.append(
            Float64(
                sf.actuators.data[a * MODEL_ACTUATOR_SIZE + ACT_IDX_CTRL_MAX]
            )
        )
        a_dof_adr.append(
            Int(sf.actuators.data[a * MODEL_ACTUATOR_SIZE + ACT_IDX_DOF_ADR])
        )

    for i in range(M.ACTION_DIM):
        var ours = a_gears[i]
        var mref = Float64(py=gear[i][0])
        print(
            "   motor", i, " gear ours", ours, " mj", mref,
            " ctrl [", a_cmin[i], ",",
            a_cmax[i], "] dof", a_dof_adr[i],
        )
        assert_true(
            _close(ours, mref),
            String("actuator ") + String(i) + " gear — a default-class gear"
            " declared after a nested <default class=...> block (bug 24)",
        )
        assert_true(
            _close(a_cmin[i], Float64(py=cr[i][0]))
            and _close(a_cmax[i], Float64(py=cr[i][1])),
            "actuator ctrlrange",
        )
        # The motor drives the joint the reference says it does.
        var mj_jnt = Int(py=dofadr[i][0])
        assert_equal(
            a_dof_adr[i], Int(py=jdof[mj_jnt]),
            "actuator transmission dof",
        )

    # Non-vacuity: gear 1.0 is exactly the value the bug produced, so a model
    # whose real gear IS 1.0 could not detect the regression.
    assert_true(
        abs(a_gears[0] - 1.0) > 1e-9,
        "swimmer's gear is 1.0 now — this gate can no longer distinguish a"
        " parsed gear from the fallback",
    )


def test_swimmer_sites_are_unrotated() raises:
    """`sensors/frame_vel.mojo` uses the BODY quaternion as the site frame.

    ⚠ This assertion USED TO exist because `site_frame_velocity` substituted
    the site's BODY quaternion for its own, so the velocimeter/gyro pair was
    exact only for an identity-oriented site. That is fixed — the sensors now
    compose `xquat[body] * site_quat` via
    `kinematics/site_frame.site_world_quat_list`, and a rotated site is handled
    correctly. The check is KEPT, with its reason rewritten: swimmer's sites
    ARE all bare under `class="swimmer"` (which sets `size`/`rgba` only), and
    pinning that means a `quat=`/`euler=` appearing upstream shows up here as a
    MODEL change to look at rather than passing unnoticed. It is no longer a
    statement about what the engine can do.
    """
    var mj = _ref(6)
    var sq = mj.site_quat.tolist()
    var sp = mj.site_pos.tolist()
    var sb = mj.site_bodyid.tolist()
    for s in range(Int(py=mj.nsite)):
        print(
            "   site", s, " body", Int(py=sb[s]),
            " quat", Float64(py=sq[s][0]), Float64(py=sq[s][1]),
            Float64(py=sq[s][2]), Float64(py=sq[s][3]),
        )
        assert_true(
            abs(Float64(py=sq[s][0]) - 1.0) <= 1e-15
            and abs(Float64(py=sq[s][1])) <= 1e-15
            and abs(Float64(py=sq[s][2])) <= 1e-15
            and abs(Float64(py=sq[s][3])) <= 1e-15,
            String("site ") + String(s) + " is ROTATED relative to its body."
            " The sensors handle that correctly now (site_world_quat_list), so"
            " this is a MODEL CHANGE upstream, not an engine limit — check"
            " what moved and then relax this assertion.",
        )
        # Site i rides body i + 1 (head, then one per segment): the mapping
        # `body_velocities` walks.
        assert_equal(
            Int(py=sb[s]), s + HEAD_BODY_IDX,
            "site/body pairing — body_velocities reads them by index",
        )
        assert_true(
            abs(Float64(py=sp[s][0])) <= 1e-15
            and abs(Float64(py=sp[s][1])) <= 1e-15
            and abs(Float64(py=sp[s][2])) <= 1e-15,
            String("site ") + String(s) + " local pos is nonzero",
        )


def test_swimmer_sensor_layout_matches_body_velocities_slice() raises:
    """`body_velocities()` is `sensordata[12:].reshape(-1, 6)[:, [0, 1, 5]]`.

    That slice is only the per-link [vx, vy, wz] if the first 12 floats are the
    four `frame*` sensors and everything after is a strict
    (velocimeter, gyro) alternation, head first. Both facts come from
    `_make_model`'s append order, which our generator has to reproduce; a
    transposition here would corrupt 18 of 25 observation entries with no
    other symptom.
    """
    var mj = _ref(6)
    var mujoco = Python.import_module("mujoco")
    var sadr = mj.sensor_adr.tolist()
    var sdim = mj.sensor_dim.tolist()

    var head_vel = Int(
        py=mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SENSOR, "head_vel")
    )
    assert_equal(
        Int(py=sadr[head_vel]), 12,
        "head_vel must start at sensordata[12] — the literal offset"
        " `Physics.body_velocities` slices from",
    )

    # Rows: (head_vel, head_gyro), then (velocimeter_i, gyro_i) per segment.
    var names: List[String] = ["head_vel", "head_gyro"]
    for i in range(5):
        names.append(String("velocimeter_") + String(i))
        names.append(String("gyro_") + String(i))
    var expect = 12
    for k in range(len(names)):
        var sid = Int(
            py=mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SENSOR, names[k])
        )
        assert_true(sid >= 0, String("missing sensor ") + names[k])
        assert_equal(
            Int(py=sadr[sid]), expect,
            String("sensor ") + names[k] + " is not at the expected address —"
            " the velocimeter/gyro interleave changed",
        )
        assert_equal(Int(py=sdim[sid]), 3, "sensor dim")
        expect += 3
    assert_equal(
        expect, Int(py=mj.nsensordata),
        "trailing sensordata beyond the velocimeter/gyro block — the"
        " reshape(-1, 6) would silently mis-group",
    )


def test_swimmer_head_ellipsoid_is_inert() raises:
    """`geom head` is an `ellipsoid`, and it must contribute nothing.

    HISTORY: `ellipsoid` had no case in `_geom_type_from_str` and fell through
    to SPHERE silently. Harmless here — but load-bearing in fish, whose tail
    and fins ARE ellipsoids with density-derived mass, where it cost tail1
    1/128th of its mass. `GEOM_ELLIPSOID` is a real geom type now (bug 26), so
    this geom is modelled as one.

    Either way the head ellipsoid must stay inert, for two independent reasons
    pinned here so neither can quietly stop holding: it carries `mass="0"` so
    it contributes no inertia, and contacts are disabled model-wide so no
    narrow phase ever reads its shape (there is no ellipsoid narrow phase —
    `init_fields` raises if a collidable ellipsoid ever appears).
    """
    comptime M = DMSwimmer6Model
    var mj = _ref(6)
    var mujoco = Python.import_module("mujoco")
    var gid = Int(py=mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "head"))
    var gmass = mj.body_mass.tolist()
    var gbody = mj.geom_bodyid.tolist()
    print(
        "   head geom id", gid, " type", Int(py=mj.geom_type.tolist()[gid]),
        " body", Int(py=gbody[gid]),
    )
    # The reference's head body mass is exactly the inertial box's .01, i.e.
    # the ellipsoid contributed nothing.
    assert_true(
        _close(Float64(py=gmass[Int(py=gbody[gid])]), 0.01),
        "the head body's mass is no longer just the inertial box — the"
        " ellipsoid-as-sphere substitution now changes the dynamics",
    )
    assert_true(
        Int(py=mj.opt.disableflags) & Int(py=mujoco.mjtDisableBit.mjDSBL_CONTACT)
        != 0,
        "contacts are no longer disabled — the ellipsoid-as-sphere geom would"
        " now be collided as the wrong shape",
    )
    # And our own nose geom index, which the observation reads by index.
    var ctx = DeviceContext()
    var mf = Model[
        DTYPE, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields[DTYPE, 0](ctx, mf)
    var go = NOSE_GEOM_IDX * MODEL_GEOM_SIZE
    var nose_mj = Int(py=mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "nose"))
    var nose_pos = mj.geom_pos.tolist()
    assert_equal(
        Int(mf.geoms.data[go + GEOM_IDX_BODY]), HEAD_BODY_IDX,
        "NOSE_GEOM_IDX does not point at a head-body geom — our geom order is"
        " XML text order and MuJoCo's is body order, so this index is ours"
        " alone and has to be pinned",
    )
    assert_true(
        _close(
            Float64(mf.geoms.data[go + GEOM_IDX_POS_X]),
            Float64(py=nose_pos[nose_mj][0]),
        )
        and _close(
            Float64(mf.geoms.data[go + GEOM_IDX_POS_Y]),
            Float64(py=nose_pos[nose_mj][1]),
        )
        and _close(
            Float64(mf.geoms.data[go + GEOM_IDX_POS_Z]),
            Float64(py=nose_pos[nose_mj][2]),
        ),
        "NOSE_GEOM_IDX resolves to a geom whose local pos is not the nose's",
    )


def test_swimmer_invweight0_matches_mujoco() raises:
    """`body_invweight0` / `dof_invweight0` against MuJoCo's own arrays.

    Run for every newly ported model since finger (bug 20): these multiply
    EVERY constraint force, so an error here is a silent multiplicative error
    on all of them with no symptom until something constrains something.
    """
    comptime M = DMSwimmer6Model
    var mj = _ref(6)
    var ctx = DeviceContext()
    var mf = Model[
        DTYPE, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields[DTYPE, 0](ctx, mf)

    var biw = mj.body_invweight0.tolist()
    var diw = mj.dof_invweight0.tolist()
    var worst = Float64(0)
    for b in range(Int(py=mj.nbody)):
        for k in range(2):
            var ours = Float64(mf.body_invweight0.data[2 * b + k])
            var mref = Float64(py=biw[b][k])
            var rel = abs(ours - mref) / (1e-15 + abs(mref))
            if rel > worst:
                worst = rel
            assert_true(
                rel <= 1e-9,
                String("body_invweight0 mismatch on body ") + String(b),
            )
    for i in range(Int(py=mj.nv)):
        var o = Float64(mf.dof_invweight0.data[i])
        var r = Float64(py=diw[i])
        var rel = abs(o - r) / (1e-15 + abs(r))
        if rel > worst:
            worst = rel
        assert_true(rel <= 1e-9, "dof_invweight0 mismatch")
    print("  worst invweight0 rel err =", worst)


# ── fluid drag, in isolation ─────────────────────────────────────────────────


def test_swimmer_fluid_drag_matches_mujoco() raises:
    """Drag alone, with no actuation and no constraint active.

    Seeded with a pure translation of the whole body along its long axis and
    zero control, the ONLY force in the model is `mj_inertiaBoxFluidModel`'s
    quadratic pressure term — no gravity DOF, no contacts, no joint limit
    reached, no damping (`dof_damping` is 0 for every swimmer joint). So this
    isolates the force this domain rests on before the rollouts test it mixed
    in with everything else.
    """
    comptime M = DMSwimmer6Model
    comptime EnvT = Phyics3dEnv[
        DMSwimmer6Model, DMSwimmerConfig, DType.float64, False
    ]
    var mj = _ref(6)
    var mujoco = Python.import_module("mujoco")
    var dat = mujoco.MjData(mj)

    mujoco.mj_resetData(mj, dat)
    dat.qvel[1] = 1.0
    mujoco.mj_forward(mj, dat)

    var env = EnvT()
    _ = env.reset()
    var qs = List[Float64]()
    var vs = List[Float64]()
    for _ in range(M.NQ):
        qs.append(0.0)
        vs.append(0.0)
    vs[1] = 1.0
    env.set_state(qs, vs)

    var act = EnvT.ActionType()
    for k in range(M.ACTION_DIM):
        act.data[k] = 0.0

    var worst = Float64(0)
    var v_last = Float64(1.0)
    for s in range(10):
        for _ in range(FRAME_SKIP_S):
            mujoco.mj_step(mj, dat)
        _ = env.step(act)
        var a = Float64(py=dat.qvel[1])
        var b = Float64(env.d.qvel.data[1])
        var e = abs(a - b)
        if e > worst:
            worst = e
        v_last = a
        if s % 3 == 0:
            print("   step", s, " mj qvel_y", a, " ours", b, " err", e)
    print("  worst drag qvel err =", worst, " v after 10 steps =", v_last)
    assert_true(worst <= STATE_TOL, "fluid drag decay diverges from MuJoCo")
    # Non-vacuity: the drag has to have actually done something, or this is a
    # test that two zeros agree.
    assert_true(
        v_last < 0.7,
        "the seeded velocity barely decayed — drag is not being applied, and"
        " an agreeing pair of near-constant velocities proves nothing",
    )


# ── dynamics + observation + reward ──────────────────────────────────────────


def _rollout[
    ModelT: ModelDefLike, N_LINKS: Int, N_STEPS: Int
](
    amp: Float64, seed_scale: Float64, target_x: Float64, target_y: Float64
) raises -> List[Float64]:
    """One lockstep rollout against the reference.

    Returns [worst_state, worst_obs, worst_reward, max_limit_fraction,
             reward_min, reward_max, max_displacement].
    """
    comptime EnvT = Phyics3dEnv[
        ModelT, DMSwimmerConfig, DType.float64, False
    ]
    comptime NQ_ = ModelT.NQ
    var mj = _ref(N_LINKS)
    var mujoco = Python.import_module("mujoco")
    var dat = mujoco.MjData(mj)

    # A deterministic seed pose: the root translation stays at the origin (as
    # `randomize_limited_and_rotational_joints` leaves it), heading and every
    # internal hinge are set.
    var q0 = List[Float64]()
    for i in range(NQ_):
        q0.append(seed_scale * sin(0.9 * Float64(i)))
    q0[0] = 0.0
    q0[1] = 0.0

    # Reference: the target write `initialize_episode` performs, on the model.
    var tgt_gid = Int(
        py=mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "target")
    )
    mujoco.mj_resetData(mj, dat)
    mj.geom_pos[tgt_gid][0] = target_x
    mj.geom_pos[tgt_gid][1] = target_y
    for i in range(NQ_):
        dat.qpos[i] = q0[i]
    mujoco.mj_forward(mj, dat)

    # Ours: the same target through the per-env mocap path (gap G4).
    var env = EnvT()
    _ = env.reset()
    var qs = List[Float64]()
    var vs = List[Float64]()
    for i in range(NQ_):
        qs.append(q0[i])
        vs.append(0.0)
    env.set_state(qs, vs)
    env.d.mocap_pos.data[(ModelT.NBODY - 1) * 3 + 0] = target_x
    env.d.mocap_pos.data[(ModelT.NBODY - 1) * 3 + 1] = target_y
    env.d.mocap_pos.data[(ModelT.NBODY - 1) * 3 + 2] = TARGET_Z

    var sadr = mj.sensor_adr.tolist()
    var hv = Int(
        py=sadr[
            Int(py=mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SENSOR, "head_vel"))
        ]
    )
    var nose_id = Int(
        py=mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "nose")
    )
    var head_id = Int(
        py=mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "head")
    )
    var jrange = Float64(py=mj.jnt_range.tolist()[N_ROOT_DOF][1])
    # `target_size = physics.named.model.geom_size['target', 0]` — read from
    # the REFERENCE model, not from our `TARGET_SIZE` comptime. Using our own
    # constant on both sides would make the reward gate blind to exactly the
    # thing it is meant to catch: the XML and the constant drifting apart.
    var target_size = Float64(py=mj.geom_size.tolist()[tgt_gid][0])

    var worst_state = Float64(0)
    var worst_obs = Float64(0)
    var worst_rew = Float64(0)
    var max_limit_frac = Float64(0)
    var r_min = Float64(1e9)
    var r_max = Float64(-1e9)
    var max_disp = Float64(0)

    for step in range(N_STEPS):
        var act = EnvT.ActionType()
        for k in range(ModelT.ACTION_DIM):
            var a = amp * sin(0.23 * Float64(step) + 1.7 * Float64(k))
            dat.ctrl[k] = a
            act.data[k] = a
        for _ in range(FRAME_SKIP_S):
            mujoco.mj_step(mj, dat)
        mujoco.mj_forward(mj, dat)
        var out = env.step(act)
        var obs = out[0]

        var ds = Float64(0)
        for i in range(NQ_):
            var e = abs(Float64(py=dat.qpos[i]) - Float64(env.d.qpos.data[i]))
            if e > ds:
                ds = e
            var e2 = abs(Float64(py=dat.qvel[i]) - Float64(env.d.qvel.data[i]))
            if e2 > ds:
                ds = e2
        if ds > worst_state:
            worst_state = ds

        for i in range(N_ROOT_DOF, NQ_):
            var f = abs(Float64(py=dat.qpos[i])) / jrange
            if f > max_limit_frac:
                max_limit_frac = f
        var dsp = sqrt(
            Float64(py=dat.qpos[0]) ** 2 + Float64(py=dat.qpos[1]) ** 2
        )
        if dsp > max_disp:
            max_disp = dsp

        # Reference observation, exactly as `Swimmer.get_observation` builds it.
        var sd = dat.sensordata.tolist()
        var ref_obs = List[Float64]()
        for i in range(N_ROOT_DOF, NQ_):  # physics.joints()
            ref_obs.append(Float64(py=dat.qpos[i]))
        var gx = dat.geom_xpos.tolist()
        var dxv = List[Float64]()
        for c in range(3):
            dxv.append(
                Float64(py=gx[tgt_gid][c]) - Float64(py=gx[nose_id][c])
            )
        var xm = dat.xmat.tolist()
        var tt = List[Float64]()
        for c in range(2):  # nose_to_target.dot(head_orientation)[:2]
            var acc = Float64(0)
            for r in range(3):
                acc += dxv[r] * Float64(py=xm[head_id][r * 3 + c])
            tt.append(acc)
            ref_obs.append(acc)
        for r in range(N_LINKS):  # body_velocities()
            ref_obs.append(Float64(py=sd[hv + r * 6 + 0]))
            ref_obs.append(Float64(py=sd[hv + r * 6 + 1]))
            ref_obs.append(Float64(py=sd[hv + r * 6 + 5]))

        var do_ = Float64(0)
        for i in range(len(ref_obs)):
            var e = abs(ref_obs[i] - Float64(obs.data[i]))
            if e > do_:
                do_ = e
        if do_ > worst_obs:
            worst_obs = do_

        var dist = sqrt(tt[0] * tt[0] + tt[1] * tt[1])
        var ref_r = tolerance[SIGMOID_LONG_TAIL](
            dist, 0.0, target_size, 5.0 * target_size
        )
        var dr = abs(ref_r - Float64(out[1]))
        if dr > worst_rew:
            worst_rew = dr
        if ref_r < r_min:
            r_min = ref_r
        if ref_r > r_max:
            r_max = ref_r

    return [
        worst_state, worst_obs, worst_rew, max_limit_frac, r_min, r_max,
        max_disp,
    ]


def test_swimmer6_dynamics_obs_and_reward_match_mujoco() raises:
    """The real gate: fluid drag, joint limits, both sensor families and the
    `long_tail` reward, over 80 control steps (1200 integrator steps).

    Driven hard enough that every internal hinge crosses its +-60 degree range
    — those limits carry `solimplimit="0 .8 .1"`, so this is also the second
    model to exercise the impedance clamp on a `dmin = 0` pair.
    """
    var r = _rollout[DMSwimmer6Model, 6, 80](1.0, 1.0, 0.7, -0.4)
    print(
        "  worst state", r[0], " obs", r[1], " reward", r[2],
        "\n  max |q| / range", r[3], " reward in [", r[4], ",", r[5],
        "] displacement", r[6],
    )
    assert_true(r[0] <= STATE_TOL, "qpos/qvel diverge from MuJoCo")
    assert_true(r[1] <= OBS_TOL, "observation diverges from MuJoCo")
    assert_true(r[2] <= REWARD_TOL, "reward diverges from MuJoCo")

    # Non-vacuity, three ways.
    assert_true(
        r[3] > 1.0,
        "no internal hinge reached its range — the rollout never engages a"
        " limit constraint, so it says nothing about the limit path",
    )
    assert_true(
        r[5] - r[4] > 1e-3,
        "the reward never moved — a constant reward matches trivially",
    )
    assert_true(
        r[6] > 1e-3,
        "the swimmer never moved from the origin — drag is producing no net"
        " locomotion and the rollout is testing a stationary body",
    )


def test_swimmer15_dynamics_and_obs_match_mujoco() raises:
    """The 15-link model, whose only difference is what the generator emitted.

    Worth its own rollout rather than trusting the counts: the segment chain,
    the +-24 degree joint ranges and the 45-wide `body_velocities` block are
    all functions of `n_bodies`, and a generator that got the nesting subtly
    wrong would still produce the right counts.
    """
    var r = _rollout[DMSwimmer15Model, 15, 60](0.9, 0.4, 0.7, -0.4)
    print(
        "  worst state", r[0], " obs", r[1], " reward", r[2],
        " max |q| / range", r[3],
    )
    assert_true(r[0] <= STATE_TOL, "qpos/qvel diverge from MuJoCo")
    assert_true(r[1] <= OBS_TOL, "observation diverges from MuJoCo")
    assert_true(r[2] <= REWARD_TOL, "reward diverges from MuJoCo")
    assert_true(
        r[3] > 1.0,
        "no internal hinge reached its range — see the swimmer6 twin",
    )


def test_swimmer_reward_saturates_on_target() raises:
    """The in-bounds branch of `tolerance`, which the far-target rollouts never
    reach: with the target sitting on the nose, `dist < target_size` and the
    reward must be exactly 1.0, not merely close to it."""
    var r = _rollout[DMSwimmer6Model, 6, 20](0.3, 0.2, 0.0, -0.06)
    print("  reward in [", r[4], ",", r[5], "] worst err", r[2])
    assert_true(r[2] <= REWARD_TOL, "reward diverges from MuJoCo")
    assert_true(
        r[5] >= 1.0 - 1e-15,
        "the reward never reached 1.0 with the target on the nose — the"
        " in-bounds branch of tolerance() is untested",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
