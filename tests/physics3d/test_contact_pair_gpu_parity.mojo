"""`<contact><pair>` on the GPU — CPU/GPU parity for both detection paths.

WHY THIS IS A SEPARATE FILE FROM `test_contact_pair_vs_mujoco.mojo`. That gate
runs at `float64` so it can diff MuJoCo to 1e-12, and **Float64 is banned on
the GPU path** — Metal rejects the kernel outright ("function's return type
'double' is not supported"), so a GPU leg cannot live inside it. This file
carries the same fixtures at `float32` and compares OUR CPU against OUR GPU.

WHAT IT IS ACTUALLY FOR. `pairs` is a NEW tensor on `fields.Model`, and a new
model tensor needs its own line in `upload_all`. Miss it and the device copy
stays zeroed: `find_predefined_pair` matches nothing, every filter it was
supposed to bypass applies again, and the masked fixture reports ZERO contacts
on GPU while every CPU leg stays green. That is why the masked fixture is here
— it is the one whose contact exists ONLY because the pair table was read, so
it fails loudly if the table did not reach the device, whereas the explicit
fixture would merely come back with different friction.

Both `detect_contacts` (O(N^2)) and `detect_contacts_sap` are covered: they are
separate kernels, and SAP carries the pair lookup in two more loops.

Run: pixi run mojo run -I . tests/physics3d/test_contact_pair_gpu_parity.mojo
"""

from std.math import abs
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
    MODEL_META_IDX_NPAIR,
    CONTACT_IDX_DIST,
    CONTACT_IDX_INCLUDEMARGIN,
    CONTACT_IDX_FRICTION,
    CONTACT_IDX_FRICTION_SPIN,
    CONTACT_IDX_FRICTION_ROLL,
    CONTACT_IDX_CONDIM,
    CONTACT_IDX_SOLREF_0,
    CONTACT_IDX_SOLREF_1,
    CONTACT_IDX_SOLIMP_0,
)


# ⚠ float32. See the module docstring — float64 is not runnable on Metal, and
# a "GPU test" written at float64 does not fail, it fails to BUILD A KERNEL.
comptime DTYPE = DType.float32
comptime TOL: Float64 = 1e-6

# Explicit per-pair parameters, on two geoms that also collide dynamically.
comptime XML_EXPLICIT = """
<mujoco model="pair_gpu_explicit">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="b1">
      <joint name="j1" type="slide" axis="0 0 1"/>
      <geom name="g1" type="sphere" size=".1" condim="3" friction=".5 .01 .0005" solref=".02 1"/>
    </body>
    <body name="b2" pos="0 0 .15">
      <joint name="j2" type="slide" axis="0 0 1"/>
      <geom name="g2" type="sphere" size=".1" condim="6" friction="1.5 .02 .001" solref=".005 2"/>
    </body>
  </worldbody>
  <contact>
    <pair geom1="g1" geom2="g2" condim="4" friction="0.7 0.7 0.02 0.003 0.003" solref="0.01 1"/>
  </contact>
</mujoco>
"""

# Masks cleared: the contact exists ONLY via the pair table, so a device copy
# that never got uploaded shows up as ncon 0 rather than as wrong parameters.
comptime XML_MASKED = """
<mujoco model="pair_gpu_masked">
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
    max_contacts=16,
    max_condim = pe.MAX_CONDIM,
    nexclude = pe.NEXCLUDE,
    npair = pe.NPAIR,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep = pe.TIMESTEP,
]:
    return {}


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
    max_contacts=16,
    max_condim = pk.MAX_CONDIM,
    nexclude = pk.NEXCLUDE,
    npair = pk.NPAIR,
    obs_dim_override=1,
    obs_qpos_skip=0,
    timestep = pk.TIMESTEP,
]:
    return {}


comptime ME = _m_e()
comptime MK = _m_k()


def _parity[M: ModelDefFromXML](label: String, use_sap: Bool) raises:
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

    # The pair table must have survived the build — a zero here makes every
    # comparison below a diff of two engines that both do nothing.
    assert_true(
        Int(mf.meta.data[MODEL_META_IDX_NPAIR]) == 1,
        label + ": model meta NPAIR != 1 — the <pair> was not serialized and"
        " this leg would be vacuous",
    )

    var dc = Dat()
    M.reset_data(dc)
    forward_kinematics["cpu"](dc, mf)
    if use_sap:
        detect_contacts_sap["cpu"](dc, mf)
    else:
        detect_contacts["cpu"](dc, mf)

    var dg = Dat()
    M.reset_data(dg)
    dg.upload_all(ctx)
    forward_kinematics["gpu"](dg, mf, ctx)
    if use_sap:
        detect_contacts_sap["gpu"](dg, mf, ctx)
    else:
        detect_contacts["gpu"](dg, mf, ctx)
    dg.contacts.download(ctx)
    dg.meta.download(ctx)

    var n_cpu = Int(dc.meta.data[META_IDX_NUM_CONTACTS])
    var n_gpu = Int(dg.meta.data[META_IDX_NUM_CONTACTS])
    assert_true(
        n_cpu == 1,
        label + ": CPU produced " + String(n_cpu) + " contacts, expected 1",
    )
    assert_true(
        n_gpu == n_cpu,
        label + ": GPU ncon " + String(n_gpu) + " != CPU " + String(n_cpu)
        + " — for the masked fixture a 0 means `m.pairs` never reached the"
        " device (check `Model.upload_all`).",
    )

    var worst = Float64(0)
    for c in range(n_cpu):
        var b = c * CONTACT_SIZE
        for slot in [
            CONTACT_IDX_CONDIM,
            CONTACT_IDX_SOLREF_0,
            CONTACT_IDX_SOLREF_1,
            CONTACT_IDX_SOLIMP_0,
            CONTACT_IDX_DIST,
            CONTACT_IDX_INCLUDEMARGIN,
            CONTACT_IDX_FRICTION,
            CONTACT_IDX_FRICTION_SPIN,
            CONTACT_IDX_FRICTION_ROLL,
        ]:
            var e = abs(
                Float64(dg.contacts.data[b + slot])
                - Float64(dc.contacts.data[b + slot])
            )
            if e > worst:
                worst = e
            assert_true(
                e <= TOL,
                label + ": GPU/CPU contact slot " + String(slot)
                + " differs by " + String(e),
            )
    print("  [" + label + "] ncon " + String(n_cpu) + "  worst |d| "
          + String(worst))


def test_pair_explicit_gpu_matches_cpu_naive() raises:
    print("--- pair GPU parity: explicit params, O(N^2) ---")
    _parity[ME](String("explicit/naive"), False)


def test_pair_explicit_gpu_matches_cpu_sap() raises:
    print("--- pair GPU parity: explicit params, SAP ---")
    _parity[ME](String("explicit/sap"), True)


def test_pair_masked_gpu_matches_cpu_naive() raises:
    """Masks cleared — the contact exists only if the device read the table."""
    print("--- pair GPU parity: masks off, O(N^2) ---")
    _parity[MK](String("masked/naive"), False)


def test_pair_masked_gpu_matches_cpu_sap() raises:
    print("--- pair GPU parity: masks off, SAP ---")
    _parity[MK](String("masked/sap"), True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
