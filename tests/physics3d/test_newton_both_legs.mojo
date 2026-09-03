"""`_newton_solve_env` gives the same solve on the static leg and the heap leg.

WHY THIS TEST EXISTS
====================

Phase 2b.2 converted the whole solver to runtime dims: caps size containers,
and every stride, loop bound and capacity guard reads the live `nv` /
`max_contacts` / `me`. Three real bugs turned up during that conversion — a
cap left as a loop bound in `island_pgs_solve`, the same in `rne`, and two
NBODY arrays sized by `_max_one` — and **every one of them was found by
static audit, not by a test.** No gate in the tree could see them, because on
the static leg a cap and the dimension it caps are THE SAME INTEGER.

`test_cholesky_both_legs` closed that hole for one leaf routine. This closes
it for the solver itself: `_newton_solve_env` is ~1800 lines, holds 72
`Scratch` buffers, and drives `contact_solve`, `scalar_rows`, `primal`,
`elliptic_cone`, `tendon_limit` and `noslip` underneath it. It is where a
surviving cap-as-stride would actually live.

WHAT THE TWO ARMS SHARE, AND WHAT THEY DO NOT
---------------------------------------------
Both arms run the SAME source lines of `_newton_solve_env` over the SAME
input buffers — the prep (FK, mass matrix, bias forces, contacts) runs ONCE,
on the static provider, so the two solves cannot differ by their inputs. What
differs is only the provider and the layouts:

    static   Dims[nq=.., nv=..]   + comptime Layout.row_major(BATCH, D.NV)
    dynamic  DynDims(nq=.., ..)   + RuntimeLayout over Layout.row_major[2]()

and, through the caps, which container every `Scratch` picks: stack on one
side, heap on the other. A cap used where a stride was meant collapses rows
onto row 0 on the dynamic arm; a cap used as a loop bound iterates zero
times. Both show up here as a numerical disagreement instead of silence.

⚠ (C) IS THE CLAIM THAT CANNOT PASS BY ACCIDENT. "The legs agree" and "the
heap leg compiles" would BOTH hold if the compiler quietly specialised the
dynamic arm on a constant — which is exactly how §12.3's first layout probe
returned a meaningless 1.000. So the dynamic arm runs TWO DIFFERENT MODELS
(walker2d and hopper, different NQ/NV/NBODY/NJOINT) through ONE `@no_inline`
body, and each must match its own static counterpart.

⚠ `@no_inline` IS PART OF THE ASSERTION, NOT AN OPTIMISATION HINT. Without
it the compiler may inline the dynamic body at both call sites and constant-
fold the dimensions into each copy, at which point "one body, two models" is
true of the source and false of the machine code.

Tolerance: f64, same source lines, same inputs. §4.4 warns the dynamic leg
cannot be assumed bit-exact in general (a comptime bound changes FMA
contraction), so the gate is `AGREE_TOL` on qacc and the worst error is
PRINTED. See that constant for why it is 1e-13 and not something rounder.

Run: pixi run mojo run -I . tests/physics3d/test_newton_both_legs.mojo
"""

from std.utils import IndexList
from layout import Layout, LayoutTensor, RuntimeLayout
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.fields import (
    Data,
    Model,
    DynamicsScratch,
    ContactScratch,
    Dims,
    DimsLike,
    DynDims,
    AsStatic,
    Scratch,
)
from mojo_rl.physics3d.fields.dims import DIM_POISON
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.model.model_dims import ModelDims
from mojo_rl.physics3d.model.model_def import ModelDefLike
from mojo_rl.physics3d.solver.newton_solve import _newton_solve_env
from mojo_rl.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from mojo_rl.physics3d.dynamics.subtree_com import compute_subtree_com
from mojo_rl.physics3d.dynamics.cdof import compute_cdof
from mojo_rl.physics3d.dynamics.mass_matrix import compute_mass_matrix
from mojo_rl.physics3d.dynamics.ldl import ldl_factor, ldl_solve, compute_m_inv
from mojo_rl.physics3d.dynamics.rne import compute_bias_forces_rne
from mojo_rl.physics3d.collision.contact_detection import detect_contacts
from mojo_rl.physics3d.integrator.euler import (
    _armature_env,
    _fnet_passive_env,
    _qacc_writeback_env,
)
from mojo_rl.physics3d.gpu.constants import (
    CONTACT_SIZE,
    METADATA_SIZE,
    META_IDX_NUM_CONTACTS,
    MODEL_JOINT_SIZE,
    MODEL_BODY_SIZE,
    MODEL_META_SIZE,
    MODEL_TREE_SIZE,
    MODEL_EQ_SIZE,
    MODEL_TENDON_SIZE,
    MODEL_SITE_SIZE,
    MODEL_GEOM_SIZE,
)
from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel
from mojo_rl.envs.hopper.hopper_xml import HopperModel

comptime DT = DType.float64
comptime BATCH = 2
# ⚠ SET FROM THE MEASURED FLOOR, NOT FROM A ROUND NUMBER. Both arms run the
# same source over the same inputs in f64, and the observed agreement is
# EXACTLY 0.0 on all three configurations. §4.4 warns the dynamic leg cannot
# be assumed bit-exact in general (a comptime bound changes FMA contraction),
# so this leaves headroom — but only a little. An earlier 1e-10 was ~1000x
# looser than anything ever observed, and it MASKED an injected cap-as-stride
# in the elliptic path that surfaced as 5.0e-13. A tolerance far above the
# noise floor is a tolerance that admits the bug the test exists to catch.
comptime AGREE_TOL = 1e-13

comptime DYN1 = Layout.row_major[1]()
comptime DYN2 = Layout.row_major[2]()


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

    def close(mut self, got: Float64, tol: Float64, what: String):
        self.checks += 1
        var e = got if got >= 0 else -got
        if not (e <= tol):
            self.fails += 1
            print("  FAIL", what, "err", e)
        else:
            print("  ok:", what, "(worst", String(e) + ")")


def worst(a: List[Float64], b: List[Float64]) -> Float64:
    if len(a) != len(b):
        print("  !! length mismatch", len(a), len(b))
        return 1e30
    var w = 0.0
    for i in range(len(a)):
        var e = a[i] - b[i]
        if e < 0:
            e = -e
        if e > w:
            w = e
    return w


def nonzero(a: List[Float64]) -> Int:
    """How many entries actually moved. A solve that wrote nothing agrees
    with another solve that wrote nothing — see the vacuity guard in main."""
    var n = 0
    for i in range(len(a)):
        if a[i] != 0.0:
            n += 1
    return n


def prep[
    MD: DimsLike
](
    mut d: Data[DT, MD, BATCH],
    mut mf: Model[DT, MD],
    mut sc: DynamicsScratch[DT, MD, BATCH],
    ctx: Optional[DeviceContext],
) raises:
    """Smooth-dynamics prep up to the constraint seam, on the STATIC provider.

    ⚠ RUN ONCE, FOR BOTH ARMS. The two solves must differ ONLY in their
    provider, so their inputs come from the same buffers produced here. If
    each arm ran its own prep, a disagreement could not be attributed.
    """
    comptime NQ = MD.NQ
    comptime NV = MD.NV
    comptime NJOINT = MD.NJOINT
    comptime L_JOINT = Layout.row_major(NJOINT, MODEL_JOINT_SIZE)
    comptime L_M = Layout.row_major(BATCH, NV * NV)
    comptime L_NV = Layout.row_major(BATCH, NV)
    comptime L_QPOS = Layout.row_major(BATCH, NQ)

    forward_kinematics["cpu", DT, BATCH=BATCH](d, mf, ctx)
    compute_body_velocities["cpu", DT, BATCH=BATCH](d, mf, ctx)
    compute_subtree_com["cpu", DT, BATCH=BATCH](d, mf, ctx)
    compute_cdof["cpu", DT, BATCH=BATCH](d, mf, sc, ctx)
    compute_mass_matrix["cpu", DT, BATCH=BATCH](d, mf, sc, ctx)

    var joints_v = mf.joints.lt["cpu", L_JOINT]()
    var M_v = sc.M.lt["cpu", L_M]()
    for e in range(BATCH):
        _armature_env[DT](e, AsStatic[MD](), joints_v, M_v)
    ldl_factor["cpu", DT, BATCH=BATCH](mf, sc, ctx)
    compute_m_inv["cpu", DT, BATCH=BATCH](mf, sc, ctx)
    compute_bias_forces_rne["cpu", DT, BATCH=BATCH](d, mf, sc, ctx)

    var qpos_v = d.qpos.lt["cpu", L_QPOS]()
    var qvel_v = d.qvel.lt["cpu", L_NV]()
    var qfrc_v = d.qfrc.lt["cpu", L_NV]()
    var bias_v = sc.bias.lt["cpu", L_NV]()
    var fnet_v = sc.fnet.lt["cpu", L_NV]()
    for e in range(BATCH):
        _fnet_passive_env[DT](
            e, AsStatic[MD](), qpos_v, qvel_v, qfrc_v, joints_v, bias_v, fnet_v
        )
    ldl_solve["cpu", DT, BATCH=BATCH](mf, sc, ctx)
    var qacc_ws_v = sc.qacc_ws.lt["cpu", L_NV]()
    var qacc_v = d.qacc.lt["cpu", L_NV]()
    var qacc_c_v = sc.qacc_constrained.lt["cpu", L_NV]()
    for e in range(BATCH):
        _qacc_writeback_env[DT](e, AsStatic[MD](), qacc_ws_v, qacc_v, qacc_c_v)
    detect_contacts["cpu", DT, BATCH=BATCH](d, mf, ctx)


def solve_static[
    MD: DimsLike, SOLVER_WS: Int, CONE: Int
](
    mut d: Data[DT, MD, BATCH],
    mut mf: Model[DT, MD],
    mut sc: DynamicsScratch[DT, MD, BATCH],
    mut cs: ContactScratch[DT, MD, BATCH, SOLVER_WS],
) raises:
    """The shipped leg: comptime layouts, `Dims`, every `Scratch` on the stack.
    """
    comptime L_QPOS = Layout.row_major(BATCH, MD.NQ)
    comptime L_NV = Layout.row_major(BATCH, MD.NV)
    comptime L_B3 = Layout.row_major(BATCH, MD.NBODY * 3)
    comptime L_B4 = Layout.row_major(BATCH, MD.NBODY * 4)
    comptime L_CON = Layout.row_major(BATCH, MD.MAX_CONTACTS * CONTACT_SIZE)
    comptime L_SMETA = Layout.row_major(BATCH, METADATA_SIZE)
    comptime L_JOINT = Layout.row_major(MD.NJOINT, MODEL_JOINT_SIZE)
    comptime L_BODY = Layout.row_major(MD.NBODY, MODEL_BODY_SIZE)
    comptime L_MMETA = Layout.row_major(MODEL_META_SIZE)
    comptime L_TREES = Layout.row_major(MD.NV * MODEL_TREE_SIZE)
    comptime L_EQ = Layout.row_major(MD.NEQUALITY, MODEL_EQ_SIZE)
    comptime L_TEN = Layout.row_major(MD.NTENDON, MODEL_TENDON_SIZE)
    comptime L_SITE = Layout.row_major(MD.NSITE, MODEL_SITE_SIZE)
    comptime L_GEOM_W = Layout.row_major(MD.NGEOM, MODEL_GEOM_SIZE)
    comptime L_BW = Layout.row_major(MD.NBODY, 2)
    comptime L_DW = Layout.row_major(MD.NV)
    comptime L_CDOF = Layout.row_major(BATCH, MD.NV * 6)
    comptime L_M = Layout.row_major(BATCH, MD.NV * MD.NV)
    comptime L_SOLVER = Layout.row_major(BATCH, SOLVER_WS)

    for e in range(BATCH):
        _newton_solve_env[DT, CONE, BATCH, SOLVER_WS](
            e,
            Dims[
                nq=MD.NQ,
                nv=MD.NV,
                nbody=MD.NBODY,
                njoint=MD.NJOINT,
                max_contacts=MD.MAX_CONTACTS,
                ngeom=MD.NGEOM,
                nequality=MD.NEQUALITY,
                ntendon=MD.NTENDON,
                nsite=MD.NSITE,
            ](),
            d.qpos.lt["cpu", L_QPOS](),
            d.qvel.lt["cpu", L_NV](),
            d.xpos.lt["cpu", L_B3](),
            d.xquat.lt["cpu", L_B4](),
            d.subtree_com.lt["cpu", L_B3](),
            d.contacts.lt["cpu", L_CON](),
            d.meta.lt["cpu", L_SMETA](),
            mf.joints.lt["cpu", L_JOINT](),
            mf.bodies.lt["cpu", L_BODY](),
            mf.meta.lt["cpu", L_MMETA](),
            mf.trees.lt["cpu", L_TREES](),
            mf.equality.lt["cpu", L_EQ](),
            mf.tendons.lt["cpu", L_TEN](),
            mf.sites.lt["cpu", L_SITE](),
            mf.geoms.lt["cpu", L_GEOM_W](),
            mf.body_invweight0.lt["cpu", L_BW](),
            mf.dof_invweight0.lt["cpu", L_DW](),
            sc.cdof.lt["cpu", L_CDOF](),
            sc.M.lt["cpu", L_M](),
            sc.m_inv.lt["cpu", L_M](),
            sc.qacc_constrained.lt["cpu", L_NV](),
            d.qacc_warmstart.lt["cpu", L_NV](),
            cs.solver.lt["cpu", L_SOLVER](),
        )


@no_inline
def solve_dynamic[
    MD: DimsLike, SOLVER_WS: Int, CONE: Int
](
    mut d: Data[DT, MD, BATCH],
    mut mf: Model[DT, MD],
    mut sc: DynamicsScratch[DT, MD, BATCH],
    mut cs: ContactScratch[DT, MD, BATCH, SOLVER_WS],
    nq: Int,
    nv: Int,
    nbody: Int,
    njoint: Int,
    max_contacts: Int,
    ngeom: Int,
    nequality: Int,
    ntendon: Int,
    nsite: Int,
) raises:
    """The dynamic leg. EVERY dimension is an ARGUMENT.

    ⚠ `MD` is still a parameter, but ONLY because `Data`/`Model` are typed by
    it — it is never read for a bound, a stride or a size inside the solve.
    The dimensions the solve sees come from `DynDims`, built from the runtime
    arguments above, and every layout below is a `RuntimeLayout`. That is what
    makes the (C) check meaningful: two models, one body.

    ⚠ `@no_inline` PINS IT TO ONE COMPILED BODY. Without it the compiler may
    inline this at both call sites and constant-fold the dimensions into each
    copy — "one body, two models" would then be true of the source and false
    of the machine code.
    """
    var rl_qpos = RuntimeLayout[DYN2].row_major(IndexList[2](BATCH, nq))
    var rl_nv = RuntimeLayout[DYN2].row_major(IndexList[2](BATCH, nv))
    var rl_b3 = RuntimeLayout[DYN2].row_major(IndexList[2](BATCH, nbody * 3))
    var rl_b4 = RuntimeLayout[DYN2].row_major(IndexList[2](BATCH, nbody * 4))
    var rl_con = RuntimeLayout[DYN2].row_major(
        IndexList[2](BATCH, max_contacts * CONTACT_SIZE)
    )
    var rl_smeta = RuntimeLayout[DYN2].row_major(
        IndexList[2](BATCH, METADATA_SIZE)
    )
    var rl_joint = RuntimeLayout[DYN2].row_major(
        IndexList[2](njoint, MODEL_JOINT_SIZE)
    )
    var rl_body = RuntimeLayout[DYN2].row_major(
        IndexList[2](nbody, MODEL_BODY_SIZE)
    )
    var rl_mmeta = RuntimeLayout[DYN1].row_major(IndexList[1](MODEL_META_SIZE))
    var rl_trees = RuntimeLayout[DYN1].row_major(
        IndexList[1](nv * MODEL_TREE_SIZE)
    )
    var rl_eq = RuntimeLayout[DYN2].row_major(
        IndexList[2](nequality, MODEL_EQ_SIZE)
    )
    var rl_ten = RuntimeLayout[DYN2].row_major(
        IndexList[2](ntendon, MODEL_TENDON_SIZE)
    )
    var rl_site = RuntimeLayout[DYN2].row_major(
        IndexList[2](nsite, MODEL_SITE_SIZE)
    )
    var rl_geom_w = RuntimeLayout[DYN2].row_major(
        IndexList[2](ngeom, MODEL_GEOM_SIZE)
    )
    var rl_bw = RuntimeLayout[DYN2].row_major(IndexList[2](nbody, 2))
    var rl_dw = RuntimeLayout[DYN1].row_major(IndexList[1](nv))
    var rl_cdof = RuntimeLayout[DYN2].row_major(IndexList[2](BATCH, nv * 6))
    var rl_m = RuntimeLayout[DYN2].row_major(IndexList[2](BATCH, nv * nv))
    var rl_solver = RuntimeLayout[DYN2].row_major(IndexList[2](BATCH, SOLVER_WS))

    var dims = DynDims(
        nq=nq,
        nv=nv,
        nbody=nbody,
        njoint=njoint,
        max_contacts=max_contacts,
        ngeom=ngeom,
        nequality=nequality,
        ntendon=ntendon,
        nsite=nsite,
    )
    for e in range(BATCH):
        _newton_solve_env[DT, CONE, BATCH, SOLVER_WS](
            e,
            dims,
            d.qpos.lt_dyn["cpu", DYN2](rl_qpos),
            d.qvel.lt_dyn["cpu", DYN2](rl_nv),
            d.xpos.lt_dyn["cpu", DYN2](rl_b3),
            d.xquat.lt_dyn["cpu", DYN2](rl_b4),
            d.subtree_com.lt_dyn["cpu", DYN2](rl_b3),
            d.contacts.lt_dyn["cpu", DYN2](rl_con),
            d.meta.lt_dyn["cpu", DYN2](rl_smeta),
            mf.joints.lt_dyn["cpu", DYN2](rl_joint),
            mf.bodies.lt_dyn["cpu", DYN2](rl_body),
            mf.meta.lt_dyn["cpu", DYN1](rl_mmeta),
            mf.trees.lt_dyn["cpu", DYN1](rl_trees),
            mf.equality.lt_dyn["cpu", DYN2](rl_eq),
            mf.tendons.lt_dyn["cpu", DYN2](rl_ten),
            mf.sites.lt_dyn["cpu", DYN2](rl_site),
            mf.geoms.lt_dyn["cpu", DYN2](rl_geom_w),
            mf.body_invweight0.lt_dyn["cpu", DYN2](rl_bw),
            mf.dof_invweight0.lt_dyn["cpu", DYN1](rl_dw),
            sc.cdof.lt_dyn["cpu", DYN2](rl_cdof),
            sc.M.lt_dyn["cpu", DYN2](rl_m),
            sc.m_inv.lt_dyn["cpu", DYN2](rl_m),
            sc.qacc_constrained.lt_dyn["cpu", DYN2](rl_nv),
            d.qacc_warmstart.lt_dyn["cpu", DYN2](rl_nv),
            cs.solver.lt_dyn["cpu", DYN2](rl_solver),
        )


def run_model[
    MODEL: ModelDefLike, CONE: Int, name: StaticString
](mut t: Tally) raises -> Tuple[List[Float64], List[Float64], Int]:
    """Prep once, then solve TWICE over the same buffers — static, then heap.

    Returns (static qacc, dynamic qacc, contacts found).
    """
    # ⚠ `ModelDims[MODEL]` is an ALIAS for `Dims[...]`; it does not carry the
    # model def, so the def and the dims are two separate parameters here.
    comptime MD = ModelDims[MODEL]
    comptime MC = MD.MAX_CONTACTS
    comptime SOLVER_WS = 81 * MC + 12 * MC * MD.NV
    var ctx = DeviceContext()
    var octx = Optional[DeviceContext](ctx)

    var mf = Model[DT, MD]()
    MODEL.init_fields[DT](ctx, mf)
    var d = Data[DT, MD, BATCH]()
    var sc = DynamicsScratch[DT, MD, BATCH]()
    var cs = ContactScratch[DT, MD, BATCH, SOLVER_WS]()

    # A pose that PENETRATES the floor, so the solve has real contact rows to
    # work on. A solve with nothing active would let the two arms agree on
    # having done nothing — the `nonzero` guard in main checks this held.
    for e in range(BATCH):
        for i in range(MD.NQ):
            var qp = Scalar[DT]((e * 5 + i * 3) % 5 - 2) / 40.0
            if i == 1:
                qp = 1.10
            d.qpos.data[e * MD.NQ + i] = qp
        for i in range(MD.NV):
            d.qvel.data[e * MD.NV + i] = Scalar[DT]((i % 3) - 1) * 0.05

    prep[MD](d, mf, sc, octx)

    var ncon = Int(d.meta.data[META_IDX_NUM_CONTACTS])

    # Snapshot the SOLVE INPUTS that the solve also writes, so arm two starts
    # exactly where arm one did.
    var qacc_in = List[Float64]()
    for i in range(BATCH * MD.NV):
        qacc_in.append(Float64(sc.qacc_constrained.data[i]))

    solve_static[MD, SOLVER_WS, CONE](d, mf, sc, cs)
    var out_s = List[Float64]()
    for i in range(BATCH * MD.NV):
        out_s.append(Float64(sc.qacc_constrained.data[i]))

    # restore, and clear the workspace the solve builds into
    for i in range(BATCH * MD.NV):
        sc.qacc_constrained.data[i] = Scalar[DT](qacc_in[i])
    for i in range(BATCH * SOLVER_WS):
        cs.solver.data[i] = Scalar[DT](0)

    solve_dynamic[MD, SOLVER_WS, CONE](
        d,
        mf,
        sc,
        cs,
        MD.NQ,
        MD.NV,
        MD.NBODY,
        MD.NJOINT,
        MD.MAX_CONTACTS,
        MD.NGEOM,
        MD.NEQUALITY,
        MD.NTENDON,
        MD.NSITE,
    )
    var out_d = List[Float64]()
    for i in range(BATCH * MD.NV):
        out_d.append(Float64(sc.qacc_constrained.data[i]))

    print("  [" + String(name) + "] nv =", MD.NV, " contacts =", ncon)
    return (out_s^, out_d^, ncon)


def main() raises:
    print("=== _newton_solve_env: static leg vs heap leg ===")
    var t = Tally()

    # ---- VACUITY GUARD: the two arms really are two different containers --
    # Everything below compares a "static" arm against a "dynamic" one. That
    # comparison is worthless if both ran the same leg, and `STATIC` is the
    # flag that picks the container — so assert it rather than infer it.
    print("\n--- the arms are genuinely two legs ---")
    t.truth(Scratch[Float64, 9].STATIC, "a positive cap selects the stack leg")
    t.truth(not Scratch[Float64, 0].STATIC, "cap 0 selects the heap leg")
    t.truth(DynDims.CAP_NV == 0, "a dynamic provider's scratch cap IS 0")
    t.truth(DynDims.NV == DIM_POISON, "and its comptime dimension is poison")

    # ---- (A) the legs agree, on a model with live contacts ---------------
    # ⚠ BOTH CONES. An earlier version of this file ran PYRAMIDAL only, and
    # an injected cap-as-stride in the ELLIPTIC region went UNDETECTED — the
    # branch simply never executed. `ell_state_force`,
    # `ell_add_contact_hessian` and `noslip_elliptic` are among the most
    # heavily converted routines in 2b.2, so a gate that skips them is a gate
    # aimed away from the risk. Found by injection, not by reading.
    print("\n--- A. walker2d, PYRAMIDAL: the two legs agree ---")
    var w = run_model[Walker2dModel, ConeType.PYRAMIDAL, "walker2d/pyr"](t)
    t.truth(w[2] > 0, "walker2d actually produced contacts (not a null solve)")
    t.truth(
        nonzero(w[0]) > 0, "and the static solve wrote a non-zero qacc"
    )
    t.close(worst(w[0], w[1]), AGREE_TOL, "walker2d static vs dynamic qacc")

    # ---- (C) ONE dynamic body, a DIFFERENT model -------------------------
    # This is the claim that cannot be faked by the compiler specialising the
    # dynamic arm: `solve_dynamic` is `@no_inline` and takes every dimension
    # as an argument, so hopper and walker2d share one compiled body.
    print("\n--- A2. walker2d, ELLIPTIC: the other cone ---")
    var we = run_model[Walker2dModel, ConeType.ELLIPTIC, "walker2d/ell"](t)
    t.truth(we[2] > 0, "elliptic leg produced contacts")
    t.truth(nonzero(we[0]) > 0, "and wrote a non-zero qacc")
    t.close(worst(we[0], we[1]), AGREE_TOL, "walker2d ELLIPTIC static vs dynamic")
    t.truth(
        worst(w[0], we[0]) > 1e-9,
        "the two cones give DIFFERENT answers (both were really run)",
    )

    print("\n--- C. hopper through the SAME dynamic body ---")
    var h = run_model[HopperModel, ConeType.PYRAMIDAL, "hopper"](t)
    t.truth(h[2] > 0, "hopper actually produced contacts")
    t.close(worst(h[0], h[1]), AGREE_TOL, "hopper static vs dynamic qacc")
    t.truth(
        len(w[0]) != len(h[0]),
        "the two models really are different sizes (one body served both)",
    )

    # ---- NEGATIVE CONTROL -------------------------------------------------
    # Every check above is "two lists are close". If `worst` were broken the
    # run would report a clean sweep of nothing. Plant the two failures this
    # test exists to catch — a perturbed value, and a truncated result (what a
    # cap-as-loop-bound produces) — and require both to be caught.
    print("\n--- negative control ---")
    var probe = Tally()
    var perturbed = w[0].copy()
    perturbed[0] += 1e-6
    probe.close(worst(w[0], perturbed), AGREE_TOL, "planted: perturbed qacc")
    var truncated = List[Float64]()
    for i in range(len(w[0]) - 1):
        truncated.append(w[0][i])
    probe.close(worst(w[0], truncated), AGREE_TOL, "planted: truncated result")
    if probe.fails != 2:
        print("!! THE CHECKER DOES NOT FAIL ON WRONG INPUT — run is VOID")
        t.fails += 1
    else:
        print("  negative control: 2/2 planted errors caught")

    print("\nchecks:", t.checks, " failures:", t.fails)
    if t.fails == 0:
        print("test_newton_both_legs: ALL PASS")
    else:
        raise Error(
            "test_newton_both_legs: " + String(t.fails) + " failure(s)"
        )
