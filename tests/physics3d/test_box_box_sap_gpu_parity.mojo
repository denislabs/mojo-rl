"""BOX/BOX CONTACTS ON THE GPU SAP PATH — CPU against GPU, both broadphases.

    pixi run -e apple  mojo run -I . tests/physics3d/test_box_box_sap_gpu_parity.mojo
    pixi run -e nvidia mojo run -I . tests/physics3d/test_box_box_sap_gpu_parity.mojo

## ⚠⚠ WHY THIS EXISTS — A MOKA POT FELL THROUGH THE TABLE (2026-09-14)

`examples/tasks/libero_family_batched.mojo` on `libero_kitchen_scene3`, M1 Pro:
the device reset matched the host to 8e-8, and then the moka pot fell 0.9 m
to the floor on every lane while the CPU leg rested it on the table. The
device listed ZERO contacts where the CPU listed 28 (table x moka pot). This
file is that scene reduced to what still fails: ONE free box resting on ONE
box, run through `detect_contacts_sap` and `detect_contacts` at float32.

MEASURED ON METAL, AT HEAD `a18f15da5`:

    fixture                 SAP cpu  SAP gpu   N^2 cpu  N^2 gpu
    box on a worldbody box     4        0         4        4
    box on a body's box        4        0         4        4
    moka pot on the table     28        0        28       28
    box on a plane (control)   4        4         4        4

and with `ccd_workspace.COLL_BLOCK_KERNEL = False` (the serial SAP kernel)
every row is 4 / 4. In the block kernel the pair IS a candidate (one, counted)
and reaches the box/box branch of `_sap_pair_narrow`, which stages nothing —
and adding an unrelated early `return` inside that branch made it stage the
right 4. A result that moves with an unrelated edit is the symptom
`feedback_metal_wide_per_thread_inlinearray_miscompute` records for this
kernel family, so this is filed as a Metal miscompute of the block kernel,
NOT a located logic defect.

⚠⚠ METAL ONLY — RTX 5090, 2026-09-15: every row 4/4, 4/4, 28/28, 4/4 on both
broadphases, each GPU leg 3-9 ms. So the block kernel is CORRECT on CUDA, and
a LIBERO scene on the Mac batch can drop a prop through a table while the same
build on NVIDIA does not. `libero_goal`'s props rested on Metal in
`libero_demo_batched`, which is consistent with that scene taking the serial
fallback (a lane the block kernel cannot finish is marked and re-run by the
serial kernel) — not measured.

⚠ ONE `DeviceContext` FOR THE RUN. The first NVIDIA run built a context per
leg and hung for good right after the CPU leg's `cuStreamDestroy`; sharing one
context, the same legs return in milliseconds.

The plane row is the CONTROL: box/plane goes through `_sap_plane_narrow`, not
the pair routine, and it agrees — so a failure below is the pair path, not
the harness.
"""

from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.fields import Data, Model, Dims
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.collision.contact_detection import detect_contacts
from mojo_rl.physics3d.collision.broadphase_sap import detect_contacts_sap
from mojo_rl.physics3d.gpu.constants import META_IDX_NUM_CONTACTS


# ⚠ float32: Metal has no double, and the batched env runs this precision.
comptime DTYPE = DType.float32

comptime XML_BOX_ON_WORLD_BOX = """
<mujoco model="bb_world">
  <option timestep="0.002"/>
  <worldbody>
    <geom pos="0 0 0.875" size="0.5 0.6 0.025" type="box"/>
    <body pos="0.05 -0.02 0.9195">
      <joint type="free"/>
      <geom type="box" size="0.025 0.025 0.02"/>
    </body>
  </worldbody>
</mujoco>
"""
comptime p0 = parse_xml(XML_BOX_ON_WORLD_BOX)


def _m0() -> ModelDefFromXML[
    xml = XML_BOX_ON_WORLD_BOX, nbody = p0.NBODY, njoint = p0.NJOINT, nq = p0.NQ,
    nv = p0.NV, ngeom = p0.NGEOM, nact = p0.NACT, ntex = p0.NTEX,
    nmat = p0.NMAT, nlight = p0.NLIGHT, ncam = p0.NCAM, nsite = p0.NSITE,
    max_tendon = p0.NTENDON, cone_type = ConeType.ELLIPTIC, max_contacts=64,
    max_condim = p0.MAX_CONDIM, nexclude = p0.NEXCLUDE, npair = p0.NPAIR,
    obs_dim_override=1, obs_qpos_skip=0, timestep = p0.TIMESTEP,
]:
    return {}


comptime M_BOX_ON_WORLD_BOX = _m0()


comptime XML_BOX_ON_BODY_BOX = """
<mujoco model="bb_body">
  <option timestep="0.002"/>
  <worldbody>
    <body pos="0 0 0.875">
      <geom size="0.05 0.06 0.025" type="box"/>
    </body>
    <body pos="0 0 0.9195">
      <joint type="free"/>
      <geom type="box" size="0.025 0.025 0.02"/>
    </body>
  </worldbody>
</mujoco>
"""
comptime p1 = parse_xml(XML_BOX_ON_BODY_BOX)


def _m1() -> ModelDefFromXML[
    xml = XML_BOX_ON_BODY_BOX, nbody = p1.NBODY, njoint = p1.NJOINT, nq = p1.NQ,
    nv = p1.NV, ngeom = p1.NGEOM, nact = p1.NACT, ntex = p1.NTEX,
    nmat = p1.NMAT, nlight = p1.NLIGHT, ncam = p1.NCAM, nsite = p1.NSITE,
    max_tendon = p1.NTENDON, cone_type = ConeType.ELLIPTIC, max_contacts=64,
    max_condim = p1.MAX_CONDIM, nexclude = p1.NEXCLUDE, npair = p1.NPAIR,
    obs_dim_override=1, obs_qpos_skip=0, timestep = p1.TIMESTEP,
]:
    return {}


comptime M_BOX_ON_BODY_BOX = _m1()


comptime XML_MOKA_POT_ON_TABLE = """
<mujoco model="bb_moka">
  <option timestep="0.002"/>
  <worldbody>
    <body pos="0 0 0.875">
      <geom size="0.5 0.6 0.025" type="box"/>
    </body>
    <body pos="0.05 -0.02 0.9656">
      <joint type="free"/>
      <geom friction="0.95 0.3 0.1" type="box" pos="-0.00000 -0.00000 0.00044" quat="0.00000 0.00000 0.00000 1.00000" size="0.02500 0.02500 0.06655" />
      <geom friction="0.95 0.3 0.1" type="box" pos="0.00000 -0.02594 0.00044" quat="0.00000 0.70711 0.70711 0.00000" size="0.00505 0.02500 0.06655" />
      <geom friction="0.95 0.3 0.1" type="box" pos="0.00000 0.02955 0.00044" quat="0.00000 0.70711 0.70711 0.00000" size="0.00505 0.02500 0.06655" />
      <geom friction="0.95 0.3 0.1" type="box" pos="0.03017 0.01254 0.00044" quat="0.00000 -0.19372 0.98106 0.00000" size="0.00420 0.01719 0.06655" />
      <geom friction="0.95 0.3 0.1" type="box" pos="-0.02934 -0.01403 0.00044" quat="-0.00000 -0.98106 -0.19372 -0.00000" size="0.00420 0.01719 0.06655" />
      <geom friction="0.95 0.3 0.1" type="box" pos="0.02962 -0.01403 0.00044" quat="0.00000 -0.97505 0.22201 0.00000" size="0.00420 0.01719 0.06655" />
      <geom friction="0.95 0.3 0.1" type="box" pos="-0.02934 0.01509 0.00044" quat="0.00000 -0.97505 0.22201 0.00000" size="0.00420 0.01719 0.06655" />
      <geom friction="0.95 0.3 0.1" type="box" pos="0.00000 -0.03591 0.04260" quat="0.00000 0.70711 0.70711 0.00000" size="0.00577 0.01762 0.02451" />
      <geom friction="0.95 0.3 0.1" type="box" pos="0.00000 -0.04910 0.04619" quat="0.00000 0.70711 0.70711 0.00000" size="0.00721 0.00802 0.01374" />
      <geom friction="0.95 0.3 0.1" type="box" pos="-0.00000 0.04159 0.05249" quat="0.00000 0.70711 0.70711 0.00000" size="0.00721 0.01021 0.01374" />
      <geom friction="0.95 0.3 0.1" type="box" pos="0.00000 0.05829 0.06027" quat="0.00000 -0.00000 0.70711 -0.70711" size="0.00655 0.00697 0.01409" />
      <geom friction="0.95 0.3 0.1" type="box" pos="-0.00000 0.07546 0.05391" quat="0.00000 0.00000 -0.90919 -0.41638" size="0.00655 0.00697 0.01409" />
      <geom friction="0.95 0.3 0.1" type="box" pos="-0.00000 0.07247 0.02824" quat="0.00000 -0.00000 -0.23224 -0.97266" size="0.00655 0.00710 0.02750" />
      <geom friction="0.95 0.3 0.1" type="box" pos="0.00013 0.05763 0.00230" quat="0.00000 -0.00000 0.92514 -0.37963" size="0.00655 0.00707 0.01007" />
      <geom friction="0.95 0.3 0.1" type="box" pos="-0.00000 -0.00134 0.07429" quat="0.00000 0.00000 1.00000 0.00000" size="0.00524 0.00598 0.01140" />
    </body>
  </worldbody>
</mujoco>
"""
comptime p2 = parse_xml(XML_MOKA_POT_ON_TABLE)


def _m2() -> ModelDefFromXML[
    xml = XML_MOKA_POT_ON_TABLE, nbody = p2.NBODY, njoint = p2.NJOINT, nq = p2.NQ,
    nv = p2.NV, ngeom = p2.NGEOM, nact = p2.NACT, ntex = p2.NTEX,
    nmat = p2.NMAT, nlight = p2.NLIGHT, ncam = p2.NCAM, nsite = p2.NSITE,
    max_tendon = p2.NTENDON, cone_type = ConeType.ELLIPTIC, max_contacts=64,
    max_condim = p2.MAX_CONDIM, nexclude = p2.NEXCLUDE, npair = p2.NPAIR,
    obs_dim_override=1, obs_qpos_skip=0, timestep = p2.TIMESTEP,
]:
    return {}


comptime M_MOKA_POT_ON_TABLE = _m2()


comptime XML_BOX_ON_PLANE = """
<mujoco model="bb_plane">
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="3 3 .1"/>
    <body pos="0.05 -0.02 0.0195">
      <joint type="free"/>
      <geom type="box" size="0.025 0.025 0.02"/>
    </body>
  </worldbody>
</mujoco>
"""
comptime p3 = parse_xml(XML_BOX_ON_PLANE)


def _m3() -> ModelDefFromXML[
    xml = XML_BOX_ON_PLANE, nbody = p3.NBODY, njoint = p3.NJOINT, nq = p3.NQ,
    nv = p3.NV, ngeom = p3.NGEOM, nact = p3.NACT, ntex = p3.NTEX,
    nmat = p3.NMAT, nlight = p3.NLIGHT, ncam = p3.NCAM, nsite = p3.NSITE,
    max_tendon = p3.NTENDON, cone_type = ConeType.ELLIPTIC, max_contacts=64,
    max_condim = p3.MAX_CONDIM, nexclude = p3.NEXCLUDE, npair = p3.NPAIR,
    obs_dim_override=1, obs_qpos_skip=0, timestep = p3.TIMESTEP,
]:
    return {}


comptime M_BOX_ON_PLANE = _m3()


def _ncon[M: ModelDefFromXML](
    ctx: DeviceContext, use_sap: Bool, on_gpu: Bool
) raises -> Int:
    comptime MD = Dims[
        nq=M.NQ, nv=M.NV, nbody=M.NBODY, njoint=M.NJOINT, ngeom=M.NGEOM,
        nsite=M.NSITE, max_contacts=M.MAX_CONTACTS, nequality=M.MAX_EQUALITY,
        ntendon=M.MAX_TENDON, nexclude=M.NEXCLUDE, nmesh_verts=0,
        npair=M.NPAIR, nact=M.NACT, nten=M.NTEN_F, nkey=M.NKEY,
    ]
    var sf = M.make_spec_fields[DTYPE]()
    var mf = Model[DTYPE, MD]()
    M.init_fields[DTYPE](ctx, mf)
    var d = Data[DTYPE, MD, 1]()
    M.reset_data(sf, d)
    if not on_gpu:
        forward_kinematics["cpu"](d, mf)
        if use_sap:
            detect_contacts_sap["cpu"](d, mf)
        else:
            detect_contacts["cpu"](d, mf)
        return Int(d.meta.data[META_IDX_NUM_CONTACTS])
    d.upload_all(ctx)
    forward_kinematics["gpu"](d, mf, ctx)
    if use_sap:
        detect_contacts_sap["gpu"](d, mf, ctx)
    else:
        detect_contacts["gpu"](d, mf, ctx)
    d.meta.download(ctx)
    ctx.synchronize()
    return Int(d.meta.data[META_IDX_NUM_CONTACTS])


def _leg[M: ModelDefFromXML](
    ctx: DeviceContext, label: String, use_sap: Bool, on_gpu: Bool
) raises -> Int:
    """One leg, announced BEFORE it runs and timed after — so a leg that never
    returns is named by the last line printed, not hidden in a silent row."""
    var what = (
        String("SAP " if use_sap else "N^2 ") + ("gpu" if on_gpu else "cpu")
    )
    print("    ...", what, "|", label, flush=True)
    var t0 = perf_counter_ns()
    var n = _ncon[M](ctx, use_sap, on_gpu)
    print("       ", what, "ncon", n, "in",
          Float64(perf_counter_ns() - t0) / 1e9, "s", flush=True)
    return n


def _row[M: ModelDefFromXML](
    ctx: DeviceContext, label: String, sap_only: Bool, mut failures: Int
) raises:
    var sc = _leg[M](ctx, label, True, False)
    var sg = _leg[M](ctx, label, True, True)
    var nc = sc
    var ng = sg
    if not sap_only:
        nc = _leg[M](ctx, label, False, False)
        ng = _leg[M](ctx, label, False, True)
    var ok = sc > 0 and sg == sc and nc == sc and ng == nc
    var n2 = (
        String("N^2 skipped") if sap_only
        else "N^2 cpu " + String(nc) + " gpu " + String(ng)
    )
    print("  " + ("ok  " if ok else "FAIL") + "  SAP cpu", sc, "gpu", sg,
          "|", n2, "|", label, flush=True)
    if not ok:
        failures += 1


def main() raises:
    # `--first`: only the first fixture; `--sap-only`: skip the N^2 control
    # legs. Together they are ONE GPU kernel — the quickest answer on NVIDIA.
    var first = False
    var sap_only = False
    var args = argv()
    for i in range(1, len(args)):
        var a = String(args[i])
        if a == "--first":
            first = True
        elif a == "--sap-only":
            sap_only = True
        else:
            raise Error("unknown argument '" + a + "' (--first, --sap-only)")
    print("=== box/box contacts: CPU vs GPU, SAP and N^2 ===", flush=True)
    # ⚠ ONE CONTEXT FOR THE WHOLE RUN, as the batched env holds one. A context
    # per leg destroyed a CUDA stream between the CPU leg and the GPU leg, and
    # the first NVIDIA run hung right after that `cuStreamDestroy`.
    var ctx = DeviceContext()
    var failures = 0
    _row[M_BOX_ON_WORLD_BOX](
        ctx, "box on a worldbody box (0.5 mm into it)", sap_only, failures
    )
    if not first:
        _row[M_BOX_ON_BODY_BOX](
            ctx, "box on a static body's box", sap_only, failures
        )
        _row[M_MOKA_POT_ON_TABLE](
            ctx, "LIBERO's moka pot (15 boxes) on the kitchen table", sap_only,
            failures,
        )
        _row[M_BOX_ON_PLANE](
            ctx, "CONTROL: box on a plane (the plane path, not the pair path)",
            sap_only, failures,
        )
    if failures > 0:
        raise Error(
            String(failures) + " fixture(s) disagree — see the header: the"
            " block SAP kernel drops box/box pairs on Metal"
        )
    print("=== PASS ===")
