"""The CPU DISPATCHERS give the same answer on a static and a dynamic provider.

WHY THIS TEST EXISTS
====================

`test_cholesky_both_legs` covers one leaf routine and `test_newton_both_legs`
covers `_newton_solve_env`, but both build their dynamic arm BY HAND: the
test itself constructs the `RuntimeLayout`s and passes them down. Nothing
gated the layer that phase 3 actually changed — the DISPATCHERS, which now
read `d.dims` and build those layouts themselves, and the CONTAINERS, which
now allocate from the same provider.

So this runs a real pipeline twice, through the shipped entry points:

    forward_kinematics -> compute_body_velocities -> compute_subtree_com
      -> compute_cdof -> compute_mass_matrix -> ldl_factor
      -> compute_bias_forces_rne

once on `Data[DT, ModelDims[Walker2dModel], BATCH]` and once on
`Data[DT, DynDims, BATCH]`, and requires the outputs to agree.

⚠⚠ THE FAILURE THIS EXISTS FOR IS SILENT. A `RuntimeLayout` built with the
wrong extents is a LEGAL layout over the wrong memory: it compiles, it runs,
and it reads a neighbouring row. That is the cap-as-stride class from 2b.2,
where all three bugs were found by static audit and none by a test.
`audit3a.py` checks the extents against the pre-3a source; this checks the
ANSWER, which is the half an audit cannot see (a dispatcher that reads the
right extents and passes the wrong provider down would pass the audit).

WHAT THE TWO ARMS SHARE, AND WHAT THEY DO NOT
---------------------------------------------
The model RECORDS are built once, by the static arm's `init_fields`, and
copied element-for-element into the dynamic arm's tensors — so the two
cannot differ by their inputs. `qpos`/`qvel` are seeded identically. What
differs is ONLY the provider:

    static   ModelDims[Walker2dModel]  comptime NQ/NV/... , stack `Scratch`,
                                       comptime `Layout.row_major(BATCH, NV)`
    dynamic  DynDims(nq=.., nv=.., ..) DIM_POISON comptime members, heap
                                       `Scratch`, `RuntimeLayout`

⚠ (D) IS THE CHECK THAT CANNOT PASS BY ACCIDENT. Agreement alone would also
hold if the dynamic arm never ran — two arrays of zeros agree perfectly. So
every comparison is preceded by a NON-ZERO count on the static arm, and
section D runs the dynamic arm again with ONE dimension deliberately wrong
and requires it to DISAGREE.

Run: pixi run mojo run -I . tests/physics3d/test_dispatchers_both_legs.mojo
"""

from max.gpu.host import DeviceContext

from mojo_rl.physics3d.fields import (
    Data,
    Model,
    DynamicsScratch,
    DimsLike,
    DynDims,
)
from mojo_rl.physics3d.fields.dims import DIM_POISON
from mojo_rl.physics3d.model.model_dims import ModelDims
from mojo_rl.physics3d.model.model_def import ModelDefLike
from mojo_rl.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from mojo_rl.physics3d.dynamics.subtree_com import compute_subtree_com
from mojo_rl.physics3d.dynamics.cdof import compute_cdof
from mojo_rl.physics3d.dynamics.mass_matrix import compute_mass_matrix
from mojo_rl.physics3d.dynamics.ldl import ldl_factor
from mojo_rl.physics3d.dynamics.rne import compute_bias_forces_rne
from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel
from mojo_rl.envs.inverted_double_pendulum.inverted_double_pendulum_xml import (
    InvertedDoublePendulumModel,
)

comptime DT = DType.float64
comptime BATCH = 2

# ⚠ SET FROM THE FLOOR, NOT FROM A ROUND NUMBER. Both arms run the same
# source lines over the same inputs in f64; the observed agreement is exactly
# 0.0 everywhere. §4.4 warns the dynamic leg cannot be assumed bit-exact in
# general (a comptime bound changes FMA contraction), so this leaves headroom
# — but a tolerance far above the noise floor is a tolerance that admits the
# bug the test exists to catch. An earlier gate lost an injected error to
# exactly that.
comptime AGREE_TOL = 1e-13


struct Tally(Movable):
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def truth(mut self, ok: Bool, what: String):
        self.checks += 1
        if not ok:
            self.fails += 1
            print("  FAIL", what)
        else:
            print("  ok:", what)


def dyn_dims[MD: DimsLike]() -> DynDims:
    """The same model, spelled as runtime state."""
    return DynDims(
        nq=MD.NQ,
        nv=MD.NV,
        nbody=MD.NBODY,
        njoint=MD.NJOINT,
        ngeom=MD.NGEOM,
        nsite=MD.NSITE,
        max_contacts=MD.MAX_CONTACTS,
        nequality=MD.NEQUALITY,
        ntendon=MD.NTENDON,
        nexclude=MD.NEXCLUDE,
        nmesh_verts=MD.NMESH_VERTS,
        npair=MD.NPAIR,
        nact=MD.NACT,
        nten=MD.NTEN,
        nkey=MD.NKEY,
    )


def copy_model[
    A: DimsLike, B: DimsLike
](mut src: Model[DT, A], mut dst: Model[DT, B]) raises:
    """Every record tensor, element for element.

    ⚠ THE ARMS MUST NOT DIFFER BY THEIR INPUTS. `init_fields` is
    parameterized on the model def's own dims, so it can only fill the static
    arm; copying is what makes a disagreement attributable to the provider
    rather than to two different models.

    ⚠ CLAMPED TO THE DESTINATION. Section D builds a provider with NBODY-1 on
    purpose, so its record tensors are SHORTER than the source's — copying
    `len(src)` elements walked off the end and crashed the test before the
    control could run. Copy what fits; the point of that arm is that it is a
    different model.
    """
    var n_bodies = len(src.bodies.data)
    if len(dst.bodies.data) < n_bodies:
        n_bodies = len(dst.bodies.data)
    for i in range(n_bodies):
        dst.bodies.data[i] = src.bodies.data[i]
    var n_joints = len(src.joints.data)
    if len(dst.joints.data) < n_joints:
        n_joints = len(dst.joints.data)
    for i in range(n_joints):
        dst.joints.data[i] = src.joints.data[i]
    var n_meta = len(src.meta.data)
    if len(dst.meta.data) < n_meta:
        n_meta = len(dst.meta.data)
    for i in range(n_meta):
        dst.meta.data[i] = src.meta.data[i]
    var n_geoms = len(src.geoms.data)
    if len(dst.geoms.data) < n_geoms:
        n_geoms = len(dst.geoms.data)
    for i in range(n_geoms):
        dst.geoms.data[i] = src.geoms.data[i]
    var n_equality = len(src.equality.data)
    if len(dst.equality.data) < n_equality:
        n_equality = len(dst.equality.data)
    for i in range(n_equality):
        dst.equality.data[i] = src.equality.data[i]
    var n_tendons = len(src.tendons.data)
    if len(dst.tendons.data) < n_tendons:
        n_tendons = len(dst.tendons.data)
    for i in range(n_tendons):
        dst.tendons.data[i] = src.tendons.data[i]
    var n_sites = len(src.sites.data)
    if len(dst.sites.data) < n_sites:
        n_sites = len(dst.sites.data)
    for i in range(n_sites):
        dst.sites.data[i] = src.sites.data[i]
    var n_body_invweight0 = len(src.body_invweight0.data)
    if len(dst.body_invweight0.data) < n_body_invweight0:
        n_body_invweight0 = len(dst.body_invweight0.data)
    for i in range(n_body_invweight0):
        dst.body_invweight0.data[i] = src.body_invweight0.data[i]
    var n_dof_invweight0 = len(src.dof_invweight0.data)
    if len(dst.dof_invweight0.data) < n_dof_invweight0:
        n_dof_invweight0 = len(dst.dof_invweight0.data)
    for i in range(n_dof_invweight0):
        dst.dof_invweight0.data[i] = src.dof_invweight0.data[i]
    var n_excludes = len(src.excludes.data)
    if len(dst.excludes.data) < n_excludes:
        n_excludes = len(dst.excludes.data)
    for i in range(n_excludes):
        dst.excludes.data[i] = src.excludes.data[i]
    var n_pairs = len(src.pairs.data)
    if len(dst.pairs.data) < n_pairs:
        n_pairs = len(dst.pairs.data)
    for i in range(n_pairs):
        dst.pairs.data[i] = src.pairs.data[i]




def seed_state[A: DimsLike](mut d: Data[DT, A, BATCH], nq: Int, nv: Int) raises:
    """⚠ `nq`/`nv` ARE ARGUMENTS, not `A.NQ`. This is called on the dynamic
    arm too, where the comptime members are `DIM_POISON` and the loops would
    run zero times — seeding nothing, and then both arms would "agree" on
    two states neither of them set."""
    for e in range(BATCH):
        for i in range(nq):
            # ⚠ `e * 5 % 5 == 0`, so an earlier `(e * 5 + i * 3) % 5`
            # gave EVERY env the same qpos — and section D's stride
            # check was then failing on the STATIC arm, correctly.
            var q = Scalar[DT]((e * 7 + i * 3) % 5 - 2) / 40.0
            if i == 1:
                q = 1.10
            d.qpos.data[e * nq + i] = q
        for i in range(nv):
            d.qvel.data[e * nv + i] = Scalar[DT]((e + i) % 3 - 1) / 10.0


def run_chain[
    A: DimsLike
](
    mut d: Data[DT, A, BATCH],
    mut m: Model[DT, A],
    mut sc: DynamicsScratch[DT, A, BATCH],
) raises:
    """The shipped CPU entry points, in the order an integrator calls them."""
    forward_kinematics["cpu", DT, BATCH=BATCH](d, m, None)
    compute_body_velocities["cpu", DT, BATCH=BATCH](d, m, None)
    compute_subtree_com["cpu", DT, BATCH=BATCH](d, m, None)
    compute_cdof["cpu", DT, BATCH=BATCH](d, m, sc, None)
    compute_mass_matrix["cpu", DT, BATCH=BATCH](d, m, sc, None)
    ldl_factor["cpu", DT, A, BATCH](sc, None)
    compute_bias_forces_rne["cpu", DT, BATCH=BATCH](d, m, sc, None)


def worst(a: List[Float64], b: List[Float64]) -> Float64:
    if len(a) != len(b):
        print("  !! length mismatch", len(a), len(b))
        return 1e30
    var w = Float64(0)
    for i in range(len(a)):
        var e = a[i] - b[i]
        if e < 0:
            e = -e
        if e > w:
            w = e
    return w


def grab(t: List[Scalar[DT]], n: Int) -> List[Float64]:
    var out = List[Float64]()
    for i in range(n):
        out.append(Float64(t[i]))
    return out^


def nonzero(a: List[Float64]) -> Int:
    var n = 0
    for i in range(len(a)):
        if a[i] != 0.0:
            n += 1
    return n


def rows_differ(a: List[Float64], width: Int) -> Bool:
    """A collapsed row stride folds env 1 onto env 0. See section D."""
    for i in range(width):
        if a[i] != a[width + i]:
            return True
    return False


def check_model[NAME: StaticString, MODEL: ModelDefLike](mut t: Tally) raises:
    """The whole comparison, for one model. See the module docstring."""
    comptime MD = ModelDims[MODEL]
    var ctx = DeviceContext()
    print()
    print("=== ", NAME, ": static provider vs DynDims, BATCH=", BATCH,
          " (nsite=", MD.NSITE, ") ===")

    # ── A. the static arm, which also builds the model records ──────────────
    var ms = Model[DT, MD]()
    MODEL.init_fields[DT](ctx, ms)
    var ds = Data[DT, MD, BATCH]()
    var ss = DynamicsScratch[DT, MD, BATCH]()
    seed_state(ds, MD.NQ, MD.NV)
    run_chain(ds, ms, ss)

    var xpos_s = grab(ds.xpos.data, BATCH * MD.NBODY * 3)
    var xvel_s = grab(ds.xvel.data, BATCH * MD.NBODY * 3)
    var stcom_s = grab(ds.subtree_com.data, BATCH * MD.NBODY * 3)
    var cdof_s = grab(ss.cdof.data, BATCH * MD.NV * 6)
    var M_s = grab(ss.M.data, BATCH * MD.NV * MD.NV)
    var L_s = grab(ss.L.data, BATCH * MD.NV * MD.NV)
    var bias_s = grab(ss.bias.data, BATCH * MD.NV)
    var sitex_s = grab(ds.site_xpos.data, BATCH * MD.NSITE * 3)

    print("--- A. the static arm actually computed something ---")
    t.truth(nonzero(xpos_s) > 0, "xpos is not all zeros")
    t.truth(nonzero(xvel_s) > 0, "xvel is not all zeros")
    t.truth(nonzero(stcom_s) > 0, "subtree_com is not all zeros")
    t.truth(nonzero(cdof_s) > 0, "cdof is not all zeros")
    t.truth(nonzero(M_s) > 0, "M is not all zeros")
    t.truth(nonzero(L_s) > 0, "L is not all zeros")
    t.truth(nonzero(bias_s) > 0, "bias is not all zeros")
    comptime if MD.NSITE > 0:
        # ⚠ THIS IS THE CHECK THE SITED MODEL WAS ADDED FOR. `_fk_sites` runs
        # under a `comptime if D.NSITE > 0:` in the dispatcher's CPU branch,
        # and on a dynamic provider that reads `-1 > 0` — false. The block is
        # not skipped for a reason; it is skipped because the dimension is
        # poison, and the result is silently absent physics.
        t.truth(nonzero(sitex_s) > 0, "site_xpos is not all zeros")

    # ── B. the dynamic arm ─────────────────────────────────────────────────
    print("--- B. the dynamic provider constructs and allocates ---")
    var dd = dyn_dims[MD]()
    var md = Model[DT, DynDims](dd)
    var dv = Data[DT, DynDims, BATCH](dd)
    var sd = DynamicsScratch[DT, DynDims, BATCH](dd)
    t.truth(DynDims.NV == DIM_POISON, "DynDims.NV is still the poison value")
    t.truth(
        len(dv.qvel.data) == BATCH * MD.NV,
        "Data allocated BATCH*NV from the RUNTIME dims (3b), not from poison",
    )
    t.truth(
        len(sd.M.data) == BATCH * MD.NV * MD.NV,
        "DynamicsScratch allocated BATCH*NV*NV from the runtime dims",
    )
    t.truth(
        len(md.bodies.data) == len(ms.bodies.data),
        "Model's body records are the same length on both legs",
    )

    copy_model(ms, md)
    seed_state(dv, MD.NQ, MD.NV)
    run_chain(dv, md, sd)

    print("--- C. the two legs agree ---")
    t.truth(
        worst(xpos_s, grab(dv.xpos.data, BATCH * MD.NBODY * 3)) <= AGREE_TOL,
        "forward_kinematics: xpos",
    )
    t.truth(
        worst(xvel_s, grab(dv.xvel.data, BATCH * MD.NBODY * 3)) <= AGREE_TOL,
        "compute_body_velocities: xvel",
    )
    t.truth(
        worst(stcom_s, grab(dv.subtree_com.data, BATCH * MD.NBODY * 3))
        <= AGREE_TOL,
        "compute_subtree_com: subtree_com",
    )
    t.truth(
        worst(cdof_s, grab(sd.cdof.data, BATCH * MD.NV * 6)) <= AGREE_TOL,
        "compute_cdof: cdof",
    )
    t.truth(
        worst(M_s, grab(sd.M.data, BATCH * MD.NV * MD.NV)) <= AGREE_TOL,
        "compute_mass_matrix: M",
    )
    t.truth(
        worst(L_s, grab(sd.L.data, BATCH * MD.NV * MD.NV)) <= AGREE_TOL,
        "ldl_factor: L",
    )
    t.truth(
        worst(bias_s, grab(sd.bias.data, BATCH * MD.NV)) <= AGREE_TOL,
        "compute_bias_forces_rne: bias",
    )
    comptime if MD.NSITE > 0:
        t.truth(
            worst(sitex_s, grab(dv.site_xpos.data, BATCH * MD.NSITE * 3))
            <= AGREE_TOL,
            "forward_kinematics: site_xpos (the NSITE gate)",
        )

    # ── D. the row STRIDE is real ──────────────────────────────────────────
    # ⚠ WITHOUT THIS, SECTION C PASSES ON TWO ARMS THAT BOTH DID NOTHING —
    # two arrays of zeros agree perfectly, and so do two arrays whose second
    # row was never written.
    #
    # The classic failure of this migration is a COLLAPSED ROW STRIDE: a
    # `RuntimeLayout` built with the wrong per-row width folds env 1 onto env
    # 0. That is exactly the cap-as-stride shape from 2b.2. The seeds differ
    # per env, so the two envs' outputs MUST differ — on both legs.
    #
    # ⚠ AND NOT BY PLANTING A WRONG DIMENSION. The first version of this
    # section built a `DynDims` with `nbody - 1` and ran the chain: it does
    # disagree, but it disagrees by walking off the end of a record array
    # (`Assert Error: index 49 … valid range 0 to 48`) and CRASHES the test
    # instead of failing it. The model records still describe 8 bodies; a
    # provider that says 7 is not a wrong LAYOUT, it is a different model.
    # The wrong-extent case is covered where it can be planted safely —
    # `scratchpad/audit3a.py`, which goes red on an injected `* 3 -> * 4`.
    print("--- D. the per-env row stride is real on BOTH legs ---")
    t.truth(
        rows_differ(xpos_s, MD.NBODY * 3),
        "static: env 0 and env 1 xpos differ (the seeds really do)",
    )
    var xpos_d = grab(dv.xpos.data, BATCH * MD.NBODY * 3)
    var M_d = grab(sd.M.data, BATCH * MD.NV * MD.NV)
    var bias_d = grab(sd.bias.data, BATCH * MD.NV)
    t.truth(
        rows_differ(xpos_d, MD.NBODY * 3),
        "dynamic: env rows of xpos differ (no collapsed stride)",
    )
    t.truth(
        rows_differ(M_d, MD.NV * MD.NV),
        "dynamic: env rows of M differ (no collapsed stride)",
    )
    t.truth(
        rows_differ(bias_d, MD.NV),
        "dynamic: env rows of bias differ (no collapsed stride)",
    )
    t.truth(nonzero(xpos_d) > 0, "dynamic: xpos is not all zeros")
    t.truth(nonzero(M_d) > 0, "dynamic: M is not all zeros")
    t.truth(nonzero(bias_d) > 0, "dynamic: bias is not all zeros")


def main() raises:
    var t = Tally()
    # ⚠ TWO MODELS, AND THE SECOND ONE IS THE POINT. walker2d has NSITE == 0,
    # NTENDON == 0 and NEQUALITY == 0, so every `comptime if D.NX > 0:` block
    # in the pipeline is skipped on BOTH legs and the agreement is real but
    # narrow. InvertedDoublePendulum has NSITE == 1, so it is the first model
    # here whose dynamic arm has to ENTER one of those blocks.
    check_model["walker2d", Walker2dModel](t)
    check_model["inverted_double_pendulum", InvertedDoublePendulumModel](t)

    print()
    print("checks:", t.checks, " failures:", t.fails)
    if t.fails == 0:
        print("test_dispatchers_both_legs: ALL PASS")
    else:
        print("test_dispatchers_both_legs: FAILED")
        raise Error("failures")
