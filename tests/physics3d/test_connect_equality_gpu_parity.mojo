"""`<equality><connect>` on a GPU — CPU/GPU parity, both solver paths.

WHY THIS FILE EXISTS. `test_connect_equality_vs_mujoco.mojo` gates the connect
rows against MuJoCo to ~1e-13, but every leg of it runs on the CPU in float64.
Nothing put a connect row through a GPU kernel, and in this tree "the shared
builder is already gated for weld" is not an argument — an ungated generic is
uncompiled code, and defect 27 was a Metal miscompute of a runtime-indexed
per-thread array that no amount of CPU parity could have seen.

⚠ float32, NOT float64. Float64 is banned on the GPU path here, and the CALL
SITE is half the fix — the model def, the `Data`, and the integrator all have
to be instantiated at `DType.float32` or the ban is enforced somewhere
unhelpful.

BOTH SOLVERS. A connect reaches the solver two different ways and the GPU
kernels are separate code:

  newton — the 3 rows are EDGE rows inside the Newton system (defect 29a)
  pgs    — the rows go through the `_equality_env` post-pass

`pgs` is also `EulerIntegrator`'s DEFAULT, so it is what a GPU model gets
unless it asks otherwise. Gating only `newton` would leave the common path
dark.

⚠ NON-VACUITY IS THE POINT OF THE LAST BLOCK. A CPU/GPU comparison passes
trivially if the connect never executes on either — that is exactly how a
whole class of gaps survived in this arc. Each leg re-runs one step with
`meta[NEQUALITY] = 0`, which short-circuits the row builder, and asserts the
answer MOVES. Without that, a GPU path that silently skipped the equality
would agree with a CPU path that did the same and report green.

Run with:
    pixi run -e apple mojo run -I . \\
        tests/physics3d/test_connect_equality_gpu_parity.mojo
"""

from std.math import abs
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.parser import parse_xml, ModelDefFromXML
from mojo_rl.physics3d.parser.xml_parser import merge_mjcf
from mojo_rl.physics3d.fields import Model, Data, Dims
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.gpu.constants import MODEL_META_IDX_NEQUALITY

# ⚠ float32 — Float64 is banned on the GPU path, and the call site is half
# the fix. Everything below is instantiated at this dtype.
comptime DTYPE = DType.float32
comptime BATCH = 1
comptime NSTEPS = 20

# Same geometry as the MuJoCo gate: a 45-degree quat on `arm` so the anchor
# derivation's transpose is exercised rather than degenerating to identity.
comptime _BODIES = """
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="arm" pos="0 0 1" quat="0.9238795325112867 0 0.3826834323650898 0">
      <joint name="arm_hinge" type="hinge" axis="0 1 0"/>
      <geom name="g_arm" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"
            mass="1" contype="0" conaffinity="0"/>
      <site name="s_arm" pos="0.4828326112068523 0.057 0.15656349186104051"
            size="0.01"/>
    </body>
    <body name="bob" pos="0.4 0 0.7">
      <joint name="bob_free" type="free"/>
      <geom name="g_bob" type="sphere" size="0.05" mass="1"
            contype="0" conaffinity="0"/>
      <site name="s_bob" pos="0.05 0.06 0.07" size="0.01"/>
    </body>
  </worldbody>
"""

comptime _RAW_BODY = (
    '<mujoco model="connect_gpu_body">'
    + _BODIES
    + """
  <equality>
    <connect body1="bob" body2="arm" anchor="0.05 0.06 0.07"/>
  </equality>
</mujoco>
"""
)

comptime _RAW_SITE = (
    '<mujoco model="connect_gpu_site">'
    + _BODIES
    + """
  <equality>
    <connect site1="s_bob" site2="s_arm" solref="0.004 1"
      solimp="0.9999 0.9999 0.001 0.5 2"/>
  </equality>
</mujoco>
"""
)

comptime XML_BODY = merge_mjcf(_RAW_BODY)
comptime XML_SITE = merge_mjcf(_RAW_SITE)
comptime pmb = parse_xml(XML_BODY)
comptime pms = parse_xml(XML_SITE)


def _model_body() -> ModelDefFromXML[
    xml=XML_BODY,
    nbody = pmb.NBODY, njoint = pmb.NJOINT, nq = pmb.NQ, nv = pmb.NV,
    ngeom = pmb.NGEOM, nact = pmb.NACT, ntex = pmb.NTEX, nmat = pmb.NMAT,
    nlight = pmb.NLIGHT, ncam = pmb.NCAM, nsite = pmb.NSITE,
    max_tendon = pmb.NTENDON, cone_type = ConeType.PYRAMIDAL,
    max_contacts=4, max_condim = pmb.MAX_CONDIM,
    neq = pmb.NEQ, max_equality = pmb.NEQ,
    nexclude = pmb.NEXCLUDE, timestep = pmb.TIMESTEP,
]:
    return {}


def _model_site() -> ModelDefFromXML[
    xml=XML_SITE,
    nbody = pms.NBODY, njoint = pms.NJOINT, nq = pms.NQ, nv = pms.NV,
    ngeom = pms.NGEOM, nact = pms.NACT, ntex = pms.NTEX, nmat = pms.NMAT,
    nlight = pms.NLIGHT, ncam = pms.NCAM, nsite = pms.NSITE,
    max_tendon = pms.NTENDON, cone_type = ConeType.PYRAMIDAL,
    max_contacts=4, max_condim = pms.MAX_CONDIM,
    neq = pms.NEQ, max_equality = pms.NEQ,
    nexclude = pms.NEXCLUDE, timestep = pms.TIMESTEP,
]:
    return {}


comptime MB = _model_body()
comptime MS = _model_site()


def _parity[M: ModelDefFromXML, SOLVER: StaticString](
    ctx: DeviceContext, label: String, tol: Float64
) raises:
    """Step CPU and GPU side by side, then prove the connect actually ran."""
    comptime MD = Dims[
        nq=M.NQ,
        nv=M.NV,
        nbody=M.NBODY,
        njoint=M.NJOINT,
        ngeom=M.NGEOM,
        nsite=M.NSITE,
        max_contacts=M.MAX_CONTACTS,
        nequality=M.MAX_EQUALITY,
        ntendon=M.MAX_TENDON,
        nexclude=M.NEXCLUDE,
        nmesh_verts=0,
        npair=M.NPAIR,
        nact=M.NACT,
        nten=M.NTEN_F,
        nkey=M.NKEY,
    ]
    var sf = M.make_spec_fields[DTYPE]()
    var mf = Model[DTYPE, MD]()
    M.init_fields[DTYPE](ctx, mf)

    # The equality must have survived serialization — MAX_EQUALITY sizes the
    # slab and meta carries the count. Zero here and everything below is a
    # comparison of two engines that both do nothing.
    assert_true(
        Int(mf.meta.data[MODEL_META_IDX_NEQUALITY]) == 1,
        label + ": model meta NEQUALITY != 1 — the connect was not"
        " serialized and this leg would be vacuous",
    )

    var dg = Data[DTYPE, MD, BATCH]()
    var dc = Data[DTYPE, MD, BATCH]()
    var doff = Data[DTYPE, MD, BATCH]()
    M.reset_data[DTYPE](sf, dg)
    M.reset_data[DTYPE](sf, dc)
    M.reset_data[DTYPE](sf, doff)
    dg.upload_all(ctx)
    doff.upload_all(ctx)

    var ig = EulerIntegrator[DTYPE, MD, M.CONE_TYPE, BATCH, SOLVER=SOLVER, MAX_CONDIM = M.MAX_CONDIM, NOSLIP_ITER = M.NOSLIP_ITER]()
    ig.prepare_gpu(ctx)
    var ic = EulerIntegrator[DTYPE, MD, M.CONE_TYPE, BATCH, SOLVER=SOLVER, MAX_CONDIM = M.MAX_CONDIM, NOSLIP_ITER = M.NOSLIP_ITER]()

    # Step-0 GPU qvel, kept for the non-vacuity comparison below.
    var qvel0 = List[Scalar[DTYPE]](capacity=BATCH * M.NV)
    for _ in range(BATCH * M.NV):
        qvel0.append(Scalar[DTYPE](0))

    for step in range(NSTEPS):
        ig.step["gpu"](dg, mf, ctx)
        ic.step["cpu"](dc, mf)
        if step == 0:
            dg.qvel.download(ctx)
            for i in range(BATCH * M.NV):
                qvel0[i] = dg.qvel.data[i]

    dg.qpos.download(ctx)
    dg.qvel.download(ctx)

    var wq = Float64(0)
    for i in range(BATCH * M.NQ):
        var e = abs(Float64(dc.qpos.data[i]) - Float64(dg.qpos.data[i]))
        if e > wq:
            wq = e
    var wv = Float64(0)
    for i in range(BATCH * M.NV):
        var e = abs(Float64(dc.qvel.data[i]) - Float64(dg.qvel.data[i]))
        if e > wv:
            wv = e
    print("  ", label, " worst |d(qpos)| =", wq, "  worst |d(qvel)| =", wv)
    assert_true(
        wq < tol,
        label + ": CPU/GPU qpos diverged by " + String(wq),
    )
    assert_true(
        wv < tol,
        label + ": CPU/GPU qvel diverged by " + String(wv),
    )

    # ── the connect must actually have executed on the GPU ────────────────
    # `meta[NEQUALITY] = 0` short-circuits the row builder. If the GPU step
    # is unchanged by that, the rows were never being built there and the
    # agreement above was two engines doing nothing in unison.
    mf.meta.data[MODEL_META_IDX_NEQUALITY] = Scalar[DTYPE](0)
    mf.meta.upload(ctx)
    ig.step["gpu"](doff, mf, ctx)
    doff.qvel.download(ctx)
    var ndiff = 0
    var wdiff = Float64(0)
    for i in range(BATCH * M.NV):
        var e = abs(Float64(doff.qvel.data[i]) - Float64(qvel0[i]))
        if e > wdiff:
            wdiff = e
        if doff.qvel.data[i] != qvel0[i]:
            ndiff += 1
    print(
        "  ", label, " connect-off differs in", ndiff, "qvel entries,"
        " worst |d| =", wdiff,
    )
    assert_true(
        ndiff > 0,
        label + ": turning the connect OFF changed nothing on the GPU — the"
        " rows are not being built there and this leg gates nothing",
    )
    # Restore, so a later leg reusing this model is not silently disabled.
    mf.meta.data[MODEL_META_IDX_NEQUALITY] = Scalar[DTYPE](1)
    mf.meta.upload(ctx)


def test_body_connect_cpu_vs_gpu_newton() raises:
    print("--- connect GPU parity: BODY semantics, newton ---")
    var ctx = DeviceContext()
    _parity[MB, "newton"](ctx, "body/newton", 1e-3)


def test_body_connect_cpu_vs_gpu_pgs() raises:
    print("--- connect GPU parity: BODY semantics, pgs (the DEFAULT) ---")
    var ctx = DeviceContext()
    _parity[MB, "pgs"](ctx, "body/pgs", 1e-3)


def test_site_connect_cpu_vs_gpu_newton() raises:
    print("--- connect GPU parity: SITE semantics, newton ---")
    var ctx = DeviceContext()
    _parity[MS, "newton"](ctx, "site/newton", 1e-3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
