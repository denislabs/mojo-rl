"""`<inertial fullinertia>` — eigendecomposition into diaginertia + iquat.

`full_parser` RAISED on `fullinertia`, which stopped ToddlerBot (45 bodies per
variant) and SO-ARM101 (7) at parse time, before anything downstream could be
measured. It was an honest refusal rather than a silent drop, which is the only
reason it surfaced the moment the real file was tried end to end.

MuJoCo's compiler diagonalises the 6-vector (`mjCBody::Compile` ->
`mjuu_fullInertia` -> `mjuu_eig3`) into `diaginertia` + `iquat`. `BodyData`
already stores exactly that pair, so this is a parser-side DECOMPOSITION into
existing fields, not a schema change.

⚠⚠ IT MUST BE *THE* EIGENSOLVER, NOT *AN* EIGENSOLVER, AND THE GATE HAS TO BE
TIGHT ENOUGH TO TELL THEM APART. MuJoCo's Jacobi forms the half-angle as
`sqrt(0.5 - 0.5c)`, which cancels catastrophically as it converges. Measured on
the 3.10.0 runtime against numpy's `eigh`:

    eigenVALUES  agree to  6.1e-13
    eigenVECTORS differ by ~8e-07   (column dot 0.9999999999996816)

So a perfectly correct independent eigensolver DISAGREES with `body_iquat` in
the seventh digit while looking valid from every angle. A gate written at
`1e-6` would pass one — and would therefore prove nothing about the port. The
tolerance below is `1e-12` for that reason, and it is met because
`eig3_symmetric` is a transcription of `mjuu_eig3` (landed for mesh inertia,
`feedback_a_valid_eigensolver_is_not_mujocos`), not an independent solver.

⚠ GATE THE EIGENVECTORS, NOT ONLY THE EIGENVALUES. A wrong `iquat` beside a
correct `diaginertia` leaves total mass and every scalar principal moment
right while silently rotating each body's inertia FRAME — the failure shape
that has already cost this arc a defect (weld orientation, #27). Both are
compared per body below.

MEASURED on the 3.10.0 runtime, and the whole reason the fixture looks like it
does:

  * eigenvalues come out in DECREASING order, always — `fullinertia="3 4 5"`
    gives `body_inertia = [5 4 3]`;
  * but `diaginertia="3 4 5"` stays `[3 4 5]`. The sort belongs to the
    fullinertia path ALONE, so body D below exists to catch a sort that
    leaked into the diagonal path;
  * `fullinertia` is MUTUALLY EXCLUSIVE with `diaginertia` and with EVERY
    inertial orientation spelling, INCLUDING a redundant `quat="1 0 0 0"`.
    MuJoCo raises rather than picking a winner.

Run with:
    pixi run mojo run -I . tests/physics3d/test_fullinertia_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.fields.model import Model
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    BODY_IDX_MASS,
    BODY_IDX_IXX,
    BODY_IDX_IQUAT_X,
    BODY_IDX_IQUAT_W,
)

comptime DTYPE = DType.float64

# EIGHT bodies, one per branch of `mjuu_eig3`, in ONE model — so the whole
# matrix costs one compile, and so a mistake keyed off body INDEX rather than
# the attributes cannot pass by accident.
#
# ⚠ NON-VACUITY: bodies A/B/C/H have genuinely off-diagonal inertia. A fixture
# whose `fullinertia` all happened to be diagonal would pass with the
# decomposition stubbed out entirely — the numbers would flow through
# unchanged and every assert would be green.
#
# The degenerate cases each take a DIFFERENT branch and are the ones where a
# "valid" solver diverges from MuJoCo's:
#   E  all three eigenvalues equal      -> `iquat` is wholly underdetermined
#   F  two equal, off-diagonal present  -> pivot choice decides the frame
#   G  eigenvalues 1e-7 apart           -> the kEigEPS-gated bubblesort swaps
#                                          or not, and the frame turns 90 deg
#   D  diaginertia, INCREASING          -> control: must NOT be sorted
#
# Every tensor satisfies A + B >= C; MuJoCo rejects the model outright
# otherwise, and most "obvious" test tensors (e.g. 3 2 1 with 0.1 0.2 0.3) do
# not — several of my first attempts were rejected for exactly that.
comptime XML = String(
    """<mujoco model="fullinertia_matrix">
  <option timestep="0.001" gravity="0 0 0"/>
  <worldbody>
    <body name="bA" pos="0 0 0">
      <joint name="jA" type="hinge" axis="0 0 1"/>
      <inertial pos="0 0 0" mass="1.5" fullinertia="3 4 5 0.1 0.2 0.3"/>
      <geom type="sphere" size=".05" contype="0" conaffinity="0"/>
    </body>
    <body name="bB" pos="0 1 0">
      <joint name="jB" type="hinge" axis="0 0 1"/>
      <inertial pos="0.01 -0.02 0.03" mass="2.5"
                fullinertia="5 4 3 0.1 0.2 0.3"/>
      <geom type="sphere" size=".05" contype="0" conaffinity="0"/>
    </body>
    <body name="bC" pos="0 2 0">
      <joint name="jC" type="hinge" axis="0 0 1"/>
      <inertial pos="0 0 0" mass="0.5"
                fullinertia="4 4 4 -0.3 0.2 -0.1"/>
      <geom type="sphere" size=".05" contype="0" conaffinity="0"/>
    </body>
    <body name="bD" pos="0 3 0">
      <joint name="jD" type="hinge" axis="0 0 1"/>
      <inertial pos="0 0 0" mass="1.0" diaginertia="3 4 5"/>
      <geom type="sphere" size=".05" contype="0" conaffinity="0"/>
    </body>
    <body name="bE" pos="0 4 0">
      <joint name="jE" type="hinge" axis="0 0 1"/>
      <inertial pos="0 0 0" mass="1.0" fullinertia="4 4 4 0 0 0"/>
      <geom type="sphere" size=".05" contype="0" conaffinity="0"/>
    </body>
    <body name="bF" pos="0 5 0">
      <joint name="jF" type="hinge" axis="0 0 1"/>
      <inertial pos="0 0 0" mass="1.0" fullinertia="4 4 3 0.1 0 0"/>
      <geom type="sphere" size=".05" contype="0" conaffinity="0"/>
    </body>
    <body name="bG" pos="0 6 0">
      <joint name="jG" type="hinge" axis="0 0 1"/>
      <inertial pos="0 0 0" mass="1.0"
                fullinertia="4 4.0000001 3 0 0 0"/>
      <geom type="sphere" size=".05" contype="0" conaffinity="0"/>
    </body>
    <body name="bH" pos="0 7 0">
      <joint name="jH" type="hinge" axis="0 0 1"/>
      <inertial pos="0 0 0" mass="0.0123"
                fullinertia="1.5e-05 1.9e-05 1.1e-05 -2.1e-07 3.3e-07 5.4e-08"/>
      <geom type="sphere" size=".05" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>"""
)

comptime pm = parse_xml(XML)
comptime M = ModelDefFromXML[
    xml=XML,
    nbody=pm.NBODY, njoint=pm.NJOINT, nq=pm.NQ, nv=pm.NV,
    ngeom=pm.NGEOM, nact=pm.NACT, ntex=pm.NTEX, nmat=pm.NMAT,
    nlight=pm.NLIGHT, ncam=pm.NCAM, nsite=pm.NSITE,
    max_contacts=8,
    obs_dim_override=1, obs_qpos_skip=0,
    timestep=pm.TIMESTEP,
]

# ⚠ Read into module-level comptime Ints before they reach a type parameter.
# Spelling `M.NPAIR` directly in the `Model` type folds it to `Int(0)` on one
# side while `init_fields` keeps the symbolic `parse_xml(XML).NPAIR` on the
# other, and the compiler will not unify the two — the same re-materialization
# trap as `Mdl._acd`. (`npair`/`nexclude` are left at their literal defaults
# above: this fixture has neither.)
comptime NV = M.NV
comptime NBODY = M.NBODY
comptime NJOINT = M.NJOINT
comptime NGEOM = M.NGEOM
comptime NEQ = M.MAX_EQUALITY
comptime NTD = M.MAX_TENDON
comptime NSITE = M.NSITE
comptime NEXCL = M.NEXCLUDE

# ⚠ NOT a placeholder. See the header: 1e-6 would pass an independently
# correct eigensolver that disagrees with MuJoCo's frame in the 7th digit,
# so it would not gate the thing this file exists to gate.
comptime TOL: Float64 = 1e-12

def _body_names() -> List[String]:
    """Built at runtime — a comptime `Array[String]` cannot be materialized."""
    return [
        String("bA"), String("bB"), String("bC"), String("bD"),
        String("bE"), String("bF"), String("bG"), String("bH"),
    ]

# ⚠ The `Model` parameter list must be spelled EXACTLY as `init_fields`
# declares it: mesh-verts is that method's own parameter (literal 0 here — no
# meshes in the fixture) while NPAIR comes from `M`. Writing a literal `0` for
# NPAIR instead leaves the compiler comparing `Int(0)` against the
# unmaterialized expression `parse_xml(XML).NPAIR`, which it will not unify.


def _build_model() raises -> Model[
    DTYPE, NV, NBODY, NJOINT, NGEOM, NEQ, NTD, NSITE, NEXCL, 0,
]:
    var ctx = DeviceContext()
    var mf = Model[
        DTYPE, M.NV, M.NBODY, M.NJOINT, M.NGEOM,
        M.MAX_EQUALITY, M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0,
    ]()
    M.init_fields[DTYPE, 0](ctx, mf)
    return mf^


def test_fullinertia_matches_mujoco() raises:
    """Per-body `diaginertia` AND `iquat` against `mjModel`.

    The eigenvalue leg alone is not enough: it is satisfied by any correct
    decomposition, including one that lands on a rotated frame."""
    print("=== fullinertia: diaginertia + iquat vs MuJoCo ===")
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(XML)

    var nbody = Int(py=m.nbody)
    assert_true(
        nbody == 9 and NBODY == 9,
        "the fixture must carry all eight bodies plus world (MuJoCo nbody="
        + String(nbody) + ", ours NBODY=" + String(NBODY) + ")",
    )

    var names = _body_names()
    var mf = _build_model()

    var worst_val = 0.0
    var worst_vec = 0.0
    var worst_mass = 0.0
    var worst_val_body = 0
    var worst_vec_body = 0
    # Sign-insensitive frame distance, reported ONLY to make a failure
    # readable: q and -q are the same rotation, so a large `worst_vec` beside
    # a tiny `worst_vec_pm` means the SIGN convention drifted, while both
    # large means the FRAME itself is wrong. Two very different bugs.
    var worst_vec_pm = 0.0

    for b in range(1, nbody):
        var base = b * MODEL_BODY_SIZE

        var dm = abs(
            mf.bodies.data[base + BODY_IDX_MASS] - Float64(py=m.body_mass[b])
        )
        if dm > worst_mass:
            worst_mass = dm

        var dv = 0.0
        for k in range(3):
            var d = abs(
                mf.bodies.data[base + BODY_IDX_IXX + k]
                - Float64(py=m.body_inertia[b][k])
            )
            if d > dv:
                dv = d
        if dv > worst_val:
            worst_val = dv
            worst_val_body = b

        # MuJoCo stores (w, x, y, z); we store (x, y, z, w).
        var mj_q = InlineArray[Float64, 4](fill=0.0)
        mj_q[0] = Float64(py=m.body_iquat[b][1])
        mj_q[1] = Float64(py=m.body_iquat[b][2])
        mj_q[2] = Float64(py=m.body_iquat[b][3])
        mj_q[3] = Float64(py=m.body_iquat[b][0])

        var dq = 0.0
        var dq_neg = 0.0
        for k in range(4):
            var ours = mf.bodies.data[base + BODY_IDX_IQUAT_X + k]
            var d = abs(ours - mj_q[k])
            var dn = abs(ours + mj_q[k])
            if d > dq:
                dq = d
            if dn > dq_neg:
                dq_neg = dn
        var dq_pm = dq if dq < dq_neg else dq_neg
        if dq > worst_vec:
            worst_vec = dq
            worst_vec_body = b
        if dq_pm > worst_vec_pm:
            worst_vec_pm = dq_pm

        print(
            "  ", names[b - 1],
            " inertia [",
            mf.bodies.data[base + BODY_IDX_IXX],
            mf.bodies.data[base + BODY_IDX_IXX + 1],
            mf.bodies.data[base + BODY_IDX_IXX + 2],
            "]  |d(val)| ", dv, "  |d(iquat)| ", dq,
        )

    print("  worst |d(mass)|       =", worst_mass)
    print("  worst |d(diaginertia)|=", worst_val, " (body", worst_val_body, ")")
    print("  worst |d(iquat)|      =", worst_vec, " (body", worst_vec_body, ")")
    print("  worst |d(iquat)| +/-  =", worst_vec_pm)

    assert_true(worst_mass <= TOL, "body mass differs from MuJoCo")
    assert_true(
        worst_val <= TOL,
        "diaginertia differs from MuJoCo by " + String(worst_val)
        + " at body " + String(worst_val_body)
        + " — ⚠ eigenvalues are sorted DECREASING on the fullinertia path and"
        " NOT sorted on the diaginertia path; check which body this is",
    )
    assert_true(
        worst_vec <= TOL,
        "iquat differs from MuJoCo by " + String(worst_vec) + " at body "
        + String(worst_vec_body) + " (sign-insensitive: "
        + String(worst_vec_pm) + ") — ⚠ if the sign-insensitive number is"
        " small the SIGN convention drifted; if both are large the inertia"
        " FRAME is wrong. ⚠⚠ a value near 1e-7 means an independently correct"
        " eigensolver was used instead of a transcription of `mjuu_eig3`:"
        " MuJoCo's own eigenvectors carry that much noise",
    )
    print("  PASS")


def test_diagonal_path_is_not_sorted() raises:
    """Body D control: `diaginertia="3 4 5"` must stay [3, 4, 5].

    The decreasing sort belongs to `mjuu_eig3` and therefore to the
    fullinertia path alone. Routing `diaginertia` through the same solver
    would be an easy simplification, would leave every OTHER assert in this
    file green, and would silently permute the inertia of every model in the
    tree that spells its inertia diagonally."""
    print("=== fullinertia: the diagonal path must NOT be sorted ===")
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(XML)
    var bd = Int(py=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bD"))

    var mf = _build_model()
    var base = bd * MODEL_BODY_SIZE

    var ours = InlineArray[Float64, 3](fill=0.0)
    for k in range(3):
        ours[k] = mf.bodies.data[base + BODY_IDX_IXX + k]

    print("   bD ours   [", ours[0], ours[1], ours[2], "]")
    print(
        "   bD MuJoCo [",
        Float64(py=m.body_inertia[bd][0]),
        Float64(py=m.body_inertia[bd][1]),
        Float64(py=m.body_inertia[bd][2]),
        "]",
    )

    # Vacuity: this only gates anything while MuJoCo leaves it ASCENDING.
    assert_true(
        Float64(py=m.body_inertia[bd][0]) < Float64(py=m.body_inertia[bd][2]),
        "bD stopped being ascending in MuJoCo — the control no longer"
        " distinguishes a sorted diagonal path from an unsorted one",
    )
    for k in range(3):
        assert_true(
            abs(ours[k] - Float64(py=m.body_inertia[bd][k])) <= TOL,
            "bD diaginertia was reordered — the decreasing sort leaked from"
            " the fullinertia path into the diagonal one",
        )
    print("  PASS")


def test_offdiagonal_actually_present() raises:
    """Non-vacuity: at least one fixture body must be genuinely off-diagonal.

    Stated as an assert rather than a comment because it is the single
    property that makes every other assert in this file meaningful. If the
    fixture drifts to all-diagonal tensors, the decomposition becomes an
    identity and a stubbed implementation passes."""
    print("=== fullinertia: fixture non-vacuity ===")
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(XML)

    var n_rotated = 0
    for b in range(1, Int(py=m.nbody)):
        var qw = Float64(py=m.body_iquat[b][0])
        if abs(abs(qw) - 1.0) > 1e-9:
            n_rotated += 1

    print("   bodies with a non-identity inertia frame:", n_rotated)
    assert_true(
        n_rotated >= 4,
        "only " + String(n_rotated) + " fixture bodies have a rotated inertia"
        " frame — the fixture no longer exercises the eigendecomposition",
    )
    print("  PASS")


def _rejects(body_inertial: String) -> Bool:
    """Does `parse_xml_full` refuse this `<inertial>`? (runtime, not comptime)."""
    var xml = String(
        "<mujoco><worldbody><body name=\"b\" pos=\"0 0 0\">"
        "<joint name=\"j\" type=\"hinge\" axis=\"0 0 1\"/>"
    ) + body_inertial + String(
        "<geom type=\"sphere\" size=\".05\" contype=\"0\" conaffinity=\"0\"/>"
        "</body></worldbody></mujoco>"
    )
    try:
        _ = parse_xml_full(xml)
        return False
    except:
        return True


def test_mutually_exclusive_spellings_are_rejected() raises:
    """`fullinertia` beside `diaginertia` or an orientation must RAISE.

    ⚠ MEASURED on the 3.10.0 runtime: MuJoCo rejects both combinations
    outright — including a redundant `quat="1 0 0 0"`, which looks harmless
    and is not. Accepting them would mean silently choosing a winner, and a
    model that loads in MuJoCo but means something different here is worse
    than one that refuses to load.

    ⚠ Vacuity guard: the last case must be ACCEPTED. Without it a parser that
    rejected every `<inertial>` unconditionally would pass this whole test."""
    print("=== fullinertia: mutually exclusive spellings ===")

    var full_ok = String(
        "<inertial pos=\"0 0 0\" mass=\"1\" fullinertia=\"3 4 5 .1 .2 .3\"/>"
    )
    var with_diag = String(
        "<inertial pos=\"0 0 0\" mass=\"1\" diaginertia=\"3 4 5\""
        " fullinertia=\"3 4 5 .1 .2 .3\"/>"
    )
    var with_quat = String(
        "<inertial pos=\"0 0 0\" mass=\"1\" quat=\"1 0 0 0\""
        " fullinertia=\"3 4 5 .1 .2 .3\"/>"
    )
    var with_euler = String(
        "<inertial pos=\"0 0 0\" mass=\"1\" euler=\"90 0 0\""
        " fullinertia=\"3 4 5 .1 .2 .3\"/>"
    )
    var non_psd = String(
        "<inertial pos=\"0 0 0\" mass=\"1\" fullinertia=\"1 1 1 5 0 0\"/>"
    )
    var wrong_count = String(
        "<inertial pos=\"0 0 0\" mass=\"1\" fullinertia=\"3 4 5\"/>"
    )

    print("   fullinertia + diaginertia rejected:", _rejects(with_diag))
    print("   fullinertia + quat        rejected:", _rejects(with_quat))
    print("   fullinertia + euler       rejected:", _rejects(with_euler))
    print("   non-PSD (neg eigenvalue)  rejected:", _rejects(non_psd))
    print("   only 3 values             rejected:", _rejects(wrong_count))
    print("   plain fullinertia         rejected:", _rejects(full_ok))

    assert_true(
        _rejects(with_diag),
        "fullinertia beside diaginertia was accepted; MuJoCo raises"
        " 'fullinertia and diagonal inertia cannot both be specified'",
    )
    assert_true(
        _rejects(with_quat),
        "fullinertia beside quat was accepted. ⚠ MuJoCo rejects this even for"
        " an IDENTITY quat — the decomposition overwrites iquat, so there is"
        " no way to honour both",
    )
    assert_true(
        _rejects(with_euler),
        "fullinertia beside euler was accepted; MuJoCo raises"
        " 'fullinertia and inertial orientation cannot both be specified'",
    )
    assert_true(
        _rejects(non_psd),
        "a non-PSD fullinertia was accepted. ⚠ its smallest eigenvalue is"
        " negative, so `body_inv_inertia = 1/eig` goes negative and the"
        " rollout is garbage with nothing raising anywhere",
    )
    assert_true(
        _rejects(wrong_count),
        "a 3-value fullinertia was accepted; it needs exactly 6",
    )
    assert_true(
        not _rejects(full_ok),
        "⚠ VACUITY: a plain valid fullinertia was REJECTED, so every assert"
        " above is satisfied by a parser that refuses everything",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
