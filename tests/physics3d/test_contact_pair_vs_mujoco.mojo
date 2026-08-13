"""`<contact><pair>` vs MuJoCo — unconditional collision + per-pair parameters.

WHY THIS EXISTS. `<pair>` was parsed by NOTHING until 2026-08-12: `full_parser`
handled `<exclude>` only, `merge_mjcf` carried the `<contact>` text without
anyone reading the pairs inside it, and there was no pair table in
`fields.Model`. A model that declares its collisions through pairs — which is
how most of Menagerie does it, ToddlerBot included — therefore loaded with a
plausible-looking body and NO ground contact at all.

WHAT IS ACTUALLY EASY TO GET WRONG HERE, and what each test pins:

  * **The parameters are MuJoCo's GLOBAL DEFAULTS, not values derived from the
    two geoms.** `mjCPair::Compile` reads as though an omitted attribute is
    filled in from geom1/geom2 (max margin, max gap, max condim, max friction,
    solmix-weighted solref/solimp) — and every one of those branches is DEAD on
    the XML path, because `mjs_defaultPair` has already written concrete
    defaults into the spec so `mjuu_defined()` is true throughout. Transcribing
    that function is the obvious mistake and it is invisible in the geometry:
    the contact still appears, in the right place, with the wrong friction.
    `test_pair_defaults_are_not_geom_derived` is the discriminator — the two
    geoms are given deliberately mismatched parameters so "defaults" and
    "derived from the geoms" cannot coincide.

  * **A predefined pair bypasses EVERY filter, not just one.** It collides
    through cleared contype/conaffinity, through `<exclude>`, and through the
    weld tests. Implementing only the mask skip still loses pairs between geoms
    on one body or on a welded parent/child.

  * **It must not DOUBLE-count.** A pair whose geoms also collide dynamically
    has to produce exactly one contact, carrying the pair's parameters.

  * **Both detection paths.** `contact_detection` and `broadphase_sap` are
    separate implementations and SAP carries the filter block TWICE (a plane
    pass and a sweep). Every test below runs through both. That is not
    paranoia: the SAP plane loop is where defect 24 lived, and the plane form
    is the ONLY form ToddlerBot's scene files use.

ALL SEMANTICS HERE WERE MEASURED AGAINST THE 3.10.0 RUNTIME, not read off a
reference tree, because the trees disagree with it. `<pair gap=>` is the
clearest case — `margin-gap` in 3.3.6/3.6.0/main, `includemargin == margin`
measured on 3.10.0, `margin + gap` in 3.11.0 — which is why the parser rejects
`gap` outright rather than picking one.

Run: pixi run mojo run -I . tests/physics3d/test_contact_pair_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.fields import Data, Model
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.collision.contact_detection import detect_contacts
from mojo_rl.physics3d.collision.broadphase_sap import detect_contacts_sap
from mojo_rl.physics3d.gpu.constants import (
    CONTACT_SIZE,
    META_IDX_NUM_CONTACTS,
    CONTACT_IDX_DIST,
    CONTACT_IDX_INCLUDEMARGIN,
    CONTACT_IDX_FRICTION,
    CONTACT_IDX_FRICTION_SPIN,
    CONTACT_IDX_FRICTION_ROLL,
    CONTACT_IDX_CONDIM,
    CONTACT_IDX_SOLREF_0,
    CONTACT_IDX_SOLREF_1,
    CONTACT_IDX_SOLIMP_0,
    CONTACT_IDX_SOLIMP_1,
    CONTACT_IDX_SOLIMP_2,
    CONTACT_IDX_SOLIMP_3,
    CONTACT_IDX_SOLIMP_4,
)


comptime DTYPE = DType.float64
comptime TOL: Float64 = 1e-12

# The two geoms carry DELIBERATELY MISMATCHED parameters in every fixture, so
# that "the pair's own values", "the geom mix" and "the global defaults" are
# three distinguishable answers. With matching geoms all three coincide and the
# gate would pass on an engine that ignored pairs entirely.
#
#   dynamic mix of g1/g2 -> condim 6, friction 1.5, solref (0.0125, 1.5)
#   MuJoCo defaults      -> condim 3, friction 1.0, solref (0.02,   1.0)
#
# ⚠ EACH FIXTURE IS ONE LITERAL. Building these by concatenating shared geom
# snippets is the obvious tidy-up and it does NOT compile: `ModelDefFromXML`
# takes the XML as a comptime PARAMETER, and a concatenated comptime String
# cannot be bound to one ("failed to infer parameter 'xml'"). The duplication
# below is the price of that.

# --- 1. explicit per-pair parameters -----------------------------------------
comptime XML_EXPLICIT = """
<mujoco model="pair_explicit">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="b1">
      <joint name="j1" type="slide" axis="0 0 1"/>
      <geom name="g1" type="sphere" size=".1" condim="3" friction=".5 .01 .0005" solref=".02 1" solimp=".9 .95 .001 .5 2"/>
    </body>
    <body name="b2" pos="0 0 .15">
      <joint name="j2" type="slide" axis="0 0 1"/>
      <geom name="g2" type="sphere" size=".1" condim="6" friction="1.5 .02 .001" solref=".005 2" solimp=".8 .99 .002 .4 3"/>
    </body>
  </worldbody>
  <contact>
    <pair geom1="g1" geom2="g2" condim="4" friction="0.7 0.7 0.02 0.003 0.003" solref="0.01 1" solimp="0.75 0.9 0.004 0.3 5"/>
  </contact>
</mujoco>
"""

# --- 2. attribute-less pair: defaults, NOT geom-derived ----------------------
comptime XML_DEFAULT = """
<mujoco model="pair_default">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="b1">
      <joint name="j1" type="slide" axis="0 0 1"/>
      <geom name="g1" type="sphere" size=".1" condim="3" friction=".5 .01 .0005" solref=".02 1" solimp=".9 .95 .001 .5 2"/>
    </body>
    <body name="b2" pos="0 0 .15">
      <joint name="j2" type="slide" axis="0 0 1"/>
      <geom name="g2" type="sphere" size=".1" condim="6" friction="1.5 .02 .001" solref=".005 2" solimp=".8 .99 .002 .4 3"/>
    </body>
  </worldbody>
  <contact>
    <pair geom1="g1" geom2="g2"/>
  </contact>
</mujoco>
"""

# --- 3. contact masks cleared: the pair must collide anyway ------------------
comptime XML_MASKED = """
<mujoco model="pair_masked">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="b1">
      <joint name="j1" type="slide" axis="0 0 1"/>
      <geom name="g1" type="sphere" size=".1" contype="0" conaffinity="0"/>
    </body>
    <body name="b2" pos="0 0 .15">
      <joint name="j2" type="slide" axis="0 0 1"/>
      <geom name="g2" type="sphere" size=".1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <contact>
    <pair geom1="g1" geom2="g2"/>
  </contact>
</mujoco>
"""

# --- 4. <exclude> on the same body pair: the pair still wins -----------------
comptime XML_EXCLUDED = """
<mujoco model="pair_excluded">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="b1">
      <joint name="j1" type="slide" axis="0 0 1"/>
      <geom name="g1" type="sphere" size=".1"/>
    </body>
    <body name="b2" pos="0 0 .15">
      <joint name="j2" type="slide" axis="0 0 1"/>
      <geom name="g2" type="sphere" size=".1"/>
    </body>
  </worldbody>
  <contact>
    <exclude body1="b1" body2="b2"/>
    <pair geom1="g1" geom2="g2"/>
  </contact>
</mujoco>
"""

# --- 5. a PLANE pair with the masks cleared ---------------------------------
# The plane form is the one ToddlerBot uses and the one SAP handles in a
# separate loop. The plane sits on the world body, so the `gj_body == 0` skip
# and the weld filter would both discard this without the pair bypass.
comptime XML_PLANE = """
<mujoco model="pair_plane">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1" contype="0" conaffinity="0"/>
    <body name="b1" pos="0 0 .05">
      <joint name="j1" type="slide" axis="0 0 1"/>
      <geom name="g1" type="sphere" size=".1" contype="0" conaffinity="0" friction="2 .02 .002"/>
    </body>
  </worldbody>
  <contact>
    <pair geom1="floor" geom2="g1" condim="6" friction="1.3 1.3 .01 .0002 .0002"/>
  </contact>
</mujoco>
"""

# --- 6. pair margin: a contact at POSITIVE separation ------------------------
# Centres 0.26 apart, radii 0.1 -> dist = 0.06, well outside the geometry but
# inside the pair's margin. This is what forces the bounding-sphere prefilter
# and SAP's AABBs to account for the PAIR's margin; with either one ignoring
# it the contact is pruned before the narrow phase ever runs.
comptime XML_MARGIN = """
<mujoco model="pair_margin">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="b1">
      <joint name="j1" type="slide" axis="0 0 1"/>
      <geom name="g1" type="sphere" size=".1"/>
    </body>
    <body name="b2" pos="0 0 .26">
      <joint name="j2" type="slide" axis="0 0 1"/>
      <geom name="g2" type="sphere" size=".1"/>
    </body>
  </worldbody>
  <contact>
    <pair geom1="g1" geom2="g2" margin="0.1"/>
  </contact>
</mujoco>
"""


# --- 7. plane margin: the plane-side bounding-sphere reject -----------------
# A sphere hovering 0.06 above the floor, inside a 0.1 pair margin. The
# plane-side reject compares `planeGeomDist` against `margin + rbound`; drop
# the margin term and this contact vanishes with no error anywhere, which is
# the ONLY failure mode that arm has (it cannot produce a wrong contact, only
# lose one). ⚠ The floor's masks are cleared so the pair is the sole reason
# the two are tested at all — the same trick fixture 5 uses.
comptime XML_PLANE_MARGIN = """
<mujoco model="plane_margin">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1" contype="0" conaffinity="0"/>
    <body name="b1" pos="0 0 .16">
      <joint name="j1" type="slide" axis="0 0 1"/>
      <geom name="g1" type="sphere" size=".1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <contact>
    <pair geom1="floor" geom2="g1" margin="0.1"/>
  </contact>
</mujoco>
"""


def _report(
    label: String,
    n_ours: Int,
    n_mj: Int,
    worst: Float64,
) raises:
    print(
        "  [" + label + "] ncon ours " + String(n_ours) + " MuJoCo "
        + String(n_mj) + "  worst |d| " + String(worst)
    )


def _gate[
    M: ModelDefFromXML
](
    xml: String, label: String, expect_ncon: Int, use_sap: Bool
) raises:
    """Build, detect on ONE path, and diff every contact parameter vs MuJoCo.

    Every
    fixture is built to produce exactly `expect_ncon` contacts, which keeps
    the pairing between our list and MuJoCo's positional rather than a
    matching problem — with one contact there is nothing to mis-match.
    """
    comptime Dat = Data[
        DTYPE, M.NQ, M.NV, M.NBODY, M.MAX_CONTACTS, M.NSITE, 1
    ]
    comptime Mod = Model[
        DTYPE, M.NV, M.NBODY, M.NJOINT, M.NGEOM, M.MAX_EQUALITY,
        M.MAX_TENDON, M.NSITE, M.NEXCLUDE, 0, M.NPAIR,
    ]

    var ctx = DeviceContext()
    var mf = Mod()
    M.init_fields[DTYPE, 0](ctx, mf)
    var d = Dat()
    M.reset_data(d)
    forward_kinematics["cpu"](d, mf)
    if use_sap:
        detect_contacts_sap["cpu"](d, mf)
    else:
        detect_contacts["cpu"](d, mf)

    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(xml)
    var md = mujoco.MjData(m)
    mujoco.mj_forward(m, md)

    var n_ours = Int(d.meta.data[META_IDX_NUM_CONTACTS])
    var n_mj = Int(py=md.ncon)

    # npair itself, before anything else — a pair table that failed to parse
    # makes every downstream assert vacuous in the direction of "no contact".
    var npair_mj = Int(py=m.npair)
    assert_true(
        M.NPAIR == npair_mj,
        label + ": npair ours " + String(M.NPAIR) + " != MuJoCo's "
        + String(npair_mj) + " — the <pair> records did not parse.",
    )
    assert_true(
        n_mj == expect_ncon,
        label + ": MuJoCo produced " + String(n_mj) + " contacts, the fixture"
        " expects " + String(expect_ncon) + " — the FIXTURE is wrong, not the"
        " engine.",
    )
    assert_true(
        n_ours == n_mj,
        label + ": ncon ours " + String(n_ours) + " != MuJoCo's "
        + String(n_mj) + ". For the masked/excluded fixtures a 0 here means"
        " the pair is not bypassing that filter; for the others it means the"
        " pair either vanished or was counted twice.",
    )

    var worst = Float64(0)
    for c in range(n_ours):
        var b = c * CONTACT_SIZE
        var cc = md.contact[c]

        var our_dim = Int(d.contacts.data[b + CONTACT_IDX_CONDIM])
        var mj_dim = Int(py=cc.dim)
        assert_true(
            our_dim == mj_dim,
            label + ": condim " + String(our_dim) + " != MuJoCo's "
            + String(mj_dim),
        )

        var fields = [
            (CONTACT_IDX_SOLREF_0, Float64(py=cc.solref[0]), String("solref0")),
            (CONTACT_IDX_SOLREF_1, Float64(py=cc.solref[1]), String("solref1")),
            (CONTACT_IDX_SOLIMP_0, Float64(py=cc.solimp[0]), String("solimp0")),
            (CONTACT_IDX_SOLIMP_1, Float64(py=cc.solimp[1]), String("solimp1")),
            (CONTACT_IDX_SOLIMP_2, Float64(py=cc.solimp[2]), String("solimp2")),
            (CONTACT_IDX_SOLIMP_3, Float64(py=cc.solimp[3]), String("solimp3")),
            (CONTACT_IDX_SOLIMP_4, Float64(py=cc.solimp[4]), String("solimp4")),
            (CONTACT_IDX_DIST, Float64(py=cc.dist), String("dist")),
            (
                CONTACT_IDX_INCLUDEMARGIN,
                Float64(py=cc.includemargin),
                String("includemargin"),
            ),
            # ⚠ MuJoCo's friction is 5-wide [slide, slide, spin, roll, roll];
            # ours is 3-wide. Index-for-index would look like a spin/roll swap.
            (CONTACT_IDX_FRICTION, Float64(py=cc.friction[0]), String("slide")),
            (
                CONTACT_IDX_FRICTION_SPIN,
                Float64(py=cc.friction[2]),
                String("spin"),
            ),
            (
                CONTACT_IDX_FRICTION_ROLL,
                Float64(py=cc.friction[3]),
                String("roll"),
            ),
        ]
        for f in fields:
            var got = Float64(d.contacts.data[b + f[0]])
            var e = abs(got - f[1])
            if e > worst:
                worst = e
            assert_true(
                e <= TOL,
                label + " " + f[2] + ": ours " + String(got)
                + " != MuJoCo's " + String(f[1]),
            )

    _report(label, n_ours, n_mj, worst)



comptime pe = parse_xml(XML_EXPLICIT)


def _m_e() -> ModelDefFromXML[
    xml = XML_EXPLICIT,
    nbody = pe.NBODY,
    njoint = pe.NJOINT,
    nq = pe.NQ,
    nv = pe.NV,
    ngeom = pe.NGEOM,
    nact = pe.NACT,
    ntex = pe.NTEX,
    nmat = pe.NMAT,
    nlight = pe.NLIGHT,
    ncam = pe.NCAM,
    nsite = pe.NSITE,
    max_tendon = pe.NTENDON,
    cone_type = ConeType.PYRAMIDAL,
    max_contacts=32,
    max_condim = pe.MAX_CONDIM,
    nexclude = pe.NEXCLUDE,
    npair = pe.NPAIR,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep = pe.TIMESTEP,
]:
    return {}


comptime ME = _m_e()

comptime pd = parse_xml(XML_DEFAULT)


def _m_d() -> ModelDefFromXML[
    xml = XML_DEFAULT,
    nbody = pd.NBODY,
    njoint = pd.NJOINT,
    nq = pd.NQ,
    nv = pd.NV,
    ngeom = pd.NGEOM,
    nact = pd.NACT,
    ntex = pd.NTEX,
    nmat = pd.NMAT,
    nlight = pd.NLIGHT,
    ncam = pd.NCAM,
    nsite = pd.NSITE,
    max_tendon = pd.NTENDON,
    cone_type = ConeType.PYRAMIDAL,
    max_contacts=32,
    max_condim = pd.MAX_CONDIM,
    nexclude = pd.NEXCLUDE,
    npair = pd.NPAIR,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep = pd.TIMESTEP,
]:
    return {}


comptime MD = _m_d()

comptime pk = parse_xml(XML_MASKED)


def _m_k() -> ModelDefFromXML[
    xml = XML_MASKED,
    nbody = pk.NBODY,
    njoint = pk.NJOINT,
    nq = pk.NQ,
    nv = pk.NV,
    ngeom = pk.NGEOM,
    nact = pk.NACT,
    ntex = pk.NTEX,
    nmat = pk.NMAT,
    nlight = pk.NLIGHT,
    ncam = pk.NCAM,
    nsite = pk.NSITE,
    max_tendon = pk.NTENDON,
    cone_type = ConeType.PYRAMIDAL,
    max_contacts=32,
    max_condim = pk.MAX_CONDIM,
    nexclude = pk.NEXCLUDE,
    npair = pk.NPAIR,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep = pk.TIMESTEP,
]:
    return {}


comptime MK = _m_k()

comptime px = parse_xml(XML_EXCLUDED)


def _m_x() -> ModelDefFromXML[
    xml = XML_EXCLUDED,
    nbody = px.NBODY,
    njoint = px.NJOINT,
    nq = px.NQ,
    nv = px.NV,
    ngeom = px.NGEOM,
    nact = px.NACT,
    ntex = px.NTEX,
    nmat = px.NMAT,
    nlight = px.NLIGHT,
    ncam = px.NCAM,
    nsite = px.NSITE,
    max_tendon = px.NTENDON,
    cone_type = ConeType.PYRAMIDAL,
    max_contacts=32,
    max_condim = px.MAX_CONDIM,
    nexclude = px.NEXCLUDE,
    npair = px.NPAIR,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep = px.TIMESTEP,
]:
    return {}


comptime MX = _m_x()

comptime pl = parse_xml(XML_PLANE)


def _m_l() -> ModelDefFromXML[
    xml = XML_PLANE,
    nbody = pl.NBODY,
    njoint = pl.NJOINT,
    nq = pl.NQ,
    nv = pl.NV,
    ngeom = pl.NGEOM,
    nact = pl.NACT,
    ntex = pl.NTEX,
    nmat = pl.NMAT,
    nlight = pl.NLIGHT,
    ncam = pl.NCAM,
    nsite = pl.NSITE,
    max_tendon = pl.NTENDON,
    cone_type = ConeType.PYRAMIDAL,
    max_contacts=32,
    max_condim = pl.MAX_CONDIM,
    nexclude = pl.NEXCLUDE,
    npair = pl.NPAIR,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep = pl.TIMESTEP,
]:
    return {}


comptime ML = _m_l()

comptime pg = parse_xml(XML_MARGIN)


def _m_g() -> ModelDefFromXML[
    xml = XML_MARGIN,
    nbody = pg.NBODY,
    njoint = pg.NJOINT,
    nq = pg.NQ,
    nv = pg.NV,
    ngeom = pg.NGEOM,
    nact = pg.NACT,
    ntex = pg.NTEX,
    nmat = pg.NMAT,
    nlight = pg.NLIGHT,
    ncam = pg.NCAM,
    nsite = pg.NSITE,
    max_tendon = pg.NTENDON,
    cone_type = ConeType.PYRAMIDAL,
    max_contacts=32,
    max_condim = pg.MAX_CONDIM,
    nexclude = pg.NEXCLUDE,
    npair = pg.NPAIR,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep = pg.TIMESTEP,
]:
    return {}


comptime MG = _m_g()


comptime ppm = parse_xml(XML_PLANE_MARGIN)


def _m_pm() -> ModelDefFromXML[
    xml = XML_PLANE_MARGIN,
    nbody = ppm.NBODY,
    njoint = ppm.NJOINT,
    nq = ppm.NQ,
    nv = ppm.NV,
    ngeom = ppm.NGEOM,
    nact = ppm.NACT,
    ntex = ppm.NTEX,
    nmat = ppm.NMAT,
    nlight = ppm.NLIGHT,
    ncam = ppm.NCAM,
    nsite = ppm.NSITE,
    max_tendon = ppm.NTENDON,
    cone_type = ConeType.PYRAMIDAL,
    max_contacts=32,
    max_condim = ppm.MAX_CONDIM,
    nexclude = ppm.NEXCLUDE,
    npair = ppm.NPAIR,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep = ppm.TIMESTEP,
]:
    return {}


comptime MPM = _m_pm()


def test_pair_explicit_params() raises:
    """A pair's own condim/friction/solref/solimp reach the contact.

    Non-vacuity is structural: the pair asks for condim 4 and friction 0.7,
    which is neither geom's value, neither the max, nor the default. An engine
    that ignored the pair would report condim 6 / friction 1.5 (the mix) and
    fail on the first assert.
    """
    print("--- pair: explicit parameters ---")
    _gate[ME](materialize[XML_EXPLICIT](), String("explicit/naive"), 1, False)
    _gate[ME](materialize[XML_EXPLICIT](), String("explicit/sap"), 1, True)


def test_pair_defaults_are_not_geom_derived() raises:
    """An attribute-less pair takes MuJoCo's GLOBAL defaults.

    ⚠ THIS IS THE TEST THAT CATCHES THE OBVIOUS PORT. `mjCPair::Compile`
    contains a full geom-derivation for every omitted attribute; it is dead
    code on the XML path. Here the two geoms differ in every parameter, so:

        derived-from-geoms -> condim 6, friction 1.5, solref (0.0125, 1.5)
        MuJoCo's actual    -> condim 3, friction 1.0, solref (0.02,   1.0)

    Both produce a contact in the same place. Only the parameters tell them
    apart, and only because the fixture refuses to let them coincide.
    """
    print("--- pair: attribute-less takes DEFAULTS, not the geom mix ---")
    _gate[MD](materialize[XML_DEFAULT](), String("default/naive"), 1, False)
    _gate[MD](materialize[XML_DEFAULT](), String("default/sap"), 1, True)


def test_pair_bypasses_contact_masks() raises:
    """`contype=0 conaffinity=0` on both geoms; the pair collides anyway."""
    print("--- pair: bypasses contype/conaffinity ---")
    _gate[MK](materialize[XML_MASKED](), String("masked/naive"), 1, False)
    _gate[MK](materialize[XML_MASKED](), String("masked/sap"), 1, True)


def test_pair_bypasses_exclude() raises:
    """`<exclude>` names the same body pair; the pair still collides.

    MuJoCo runs the predefined-pair merge BEFORE the exclude scan, so the
    exclusion never sees it.
    """
    print("--- pair: bypasses <contact><exclude> ---")
    _gate[MX](materialize[XML_EXCLUDED](), String("excluded/naive"), 1, False)
    _gate[MX](materialize[XML_EXCLUDED](), String("excluded/sap"), 1, True)


def test_pair_with_plane_and_masks_off() raises:
    """The plane form — SAP's separate plane loop, and ToddlerBot's spelling."""
    print("--- pair: plane vs sphere, masks off ---")
    _gate[ML](materialize[XML_PLANE](), String("plane/naive"), 1, False)
    _gate[ML](materialize[XML_PLANE](), String("plane/sap"), 1, True)


def test_pair_margin_survives_the_broadphase() raises:
    """A pair margin produces a contact at POSITIVE distance.

    The spheres are 0.06 apart. Both the bounding-sphere prefilter and SAP's
    AABBs would discard them on geometry alone, so this fails unless both
    account for the PAIR's margin rather than the geoms'.
    """
    print("--- pair: margin, contact at positive separation ---")
    _gate[MG](materialize[XML_MARGIN](), String("margin/naive"), 1, False)
    _gate[MG](materialize[XML_MARGIN](), String("margin/sap"), 1, True)


def test_plane_margin_survives_the_plane_reject() raises:
    """A geom hovering inside its margin above a plane still contacts it.

    ⚠⚠ THE PLANE ARM'S ONLY FAILURE MODE IS A LOST CONTACT, NOT A WRONG ONE.
    It decides whether narrow phase runs at all, so a bug there subtracts a
    contact and nothing downstream complains — no NaN, no count assertion, just
    a floor that stopped existing for geoms near it. Both paths are exercised
    because both carry their own copy of the reject.
    """
    print("--- plane: margin, contact at positive separation ---")
    _gate[MPM](materialize[XML_PLANE_MARGIN](), String("plane_margin/naive"), 1, False)
    _gate[MPM](materialize[XML_PLANE_MARGIN](), String("plane_margin/sap"), 1, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
