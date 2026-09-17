"""ONE LIBERO STATE, ONE ELLIPTIC SOLVE, BOTH NEWTON LEGS — where do they part?

    # the state comes from the family driver:
    #   libero_family_batched --dump-at-step 1 --dump-state build/diag/lr3_step1.txt
    pixi run mojo run -I . tools/tasks/solve_at_pose.mojo build/diag/lr3_step1.txt 0

## ⚠⚠ WHY A STEPPING RUN CANNOT ANSWER THIS

`libero_family_batched --solver-log` shows the two legs BIT-IDENTICAL through
step 1 of libero_living_room_scene3 and diverging at step 2, the line-search
work first (cumulative evaluations 2997 against 3006) while the state still
matches to six digits. After that every difference is downstream of the first,
so a rollout says only THAT they part. This replays ONE state through both,
with identical inputs, and prints where the two answers differ.

`tests/physics3d/test_newton_blocked_elliptic.mojo` does exactly this for a
slam chain of capsules and asserts bit equality — and PASSES. LIBERO is boxes
resting at the contact margin, 20+ contacts a lane, which that fixture never
produces. The pre-solve pipeline below is its `_prep`, verbatim, for that
reason: the two files must feed their solves the same way.

## ⚠ max_contacts IS LOWERED TO FIT METAL

`libero_living_room_scene3` at 64 asks for 32 852 B of threadgroup memory and
Metal's limit is 32 768 — 84 bytes over. At 56 it fits, and the family's
measured contact peak is 28 (`contact_budget.kv`), so nothing is truncated.
Both legs run at the SAME cap, which is what the comparison needs.
"""

from std.math import abs
from std.sys import argv
from max.gpu.host import DeviceContext
from layout import Layout

from mojo_rl.physics3d.fields import (
    AsStatic, Data, Model, DynamicsScratch, ContactScratch, Dims,
)
from mojo_rl.physics3d.types import ConeType
from mojo_rl.physics3d.solver.je_budget import je_ws_size
from mojo_rl.physics3d.kinematics.forward_kinematics import (
    forward_kinematics, compute_body_velocities,
)
from mojo_rl.physics3d.dynamics.subtree_com import compute_subtree_com
from mojo_rl.physics3d.dynamics.cdof import compute_cdof
from mojo_rl.physics3d.dynamics.mass_matrix import compute_mass_matrix
from mojo_rl.physics3d.dynamics.ldl import ldl_factor, ldl_solve, compute_m_inv
from mojo_rl.physics3d.dynamics.rne import compute_bias_forces_rne
from mojo_rl.physics3d.integrator.euler import (
    _armature_kernel, _fnet_passive_kernel, _qacc_writeback_kernel,
)
from mojo_rl.physics3d.collision.broadphase_sap import detect_contacts_sap
from mojo_rl.physics3d.solver.newton_solve import (
    solve_newton_blocked, solve_newton,
)
from mojo_rl.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS, META_IDX_NEWTON_ITER, META_IDX_LS_EVAL,
    METADATA_SIZE, MODEL_JOINT_SIZE, CONTACT_SIZE, CONTACT_IDX_CONDIM,
    CONTACT_IDX_DIST, CONTACT_IDX_FORCE_N, CONTACT_IDX_FORCE_T1,
    CONTACT_IDX_FORCE_T2, CONTACT_IDX_FORCE_TORSION,
)
from mojo_rl.tasks.libero_envs.libero_living_room_scene3_xml import (
    LiberoLivingRoomScene3Model,
)


comptime DTYPE = DType.float32
comptime BATCH = 2
"""⚠⚠ TWO LANES, AND THAT IS DELIBERATE. The blocked kernel is one BLOCK PER
ENV; a per-env indexing mistake cannot show at BATCH 1, which is exactly the
shape a single-lane harness would call "bit-identical" forever. The two lanes
are filled with DIFFERENT dumped states."""
comptime M = LiberoLivingRoomScene3Model
"""⚠ ONE FAMILY PER BUILD — `sed` this line, as in the other family tools."""
comptime MC = 56
comptime NQ = M.NQ
comptime NV = M.NV
comptime NJOINT = M.NJOINT
comptime MDIMS = Dims[
    nq=M.NQ, nv=M.NV, nbody=M.NBODY, njoint=M.NJOINT, ngeom=M.NGEOM,
    nsite=M.NSITE, max_contacts=MC, nequality=M.MAX_EQUALITY,
    ntendon=M.MAX_TENDON, nexclude=M.NEXCLUDE, nmesh_verts=65536,
    npair=M.NPAIR, nact=M.NACT, nten=M.NTEN_F, nkey=M.NKEY,
]
comptime JE_WS = je_ws_size[
    DTYPE, MDIMS.NV, MDIMS.NJOINT, MDIMS.NTENDON, MDIMS.NEQUALITY,
    MDIMS.MAX_CONTACTS, M.MAX_CONDIM, CONE_TYPE=ConeType.ELLIPTIC,
]()


struct Solved(Movable):
    var qacc: List[Float64]
    var forces: List[Float64]
    var dist: List[Float64]
    var condim: List[Int]
    var ncon: Int
    var iters: Int
    var lsev: Int
    var per_lane_ncon: List[Int]
    var per_lane_iters: List[Int]
    var per_lane_lsev: List[Int]

    def __init__(out self):
        self.qacc = List[Float64]()
        self.forces = List[Float64]()
        self.dist = List[Float64]()
        self.condim = List[Int]()
        self.ncon = 0
        self.iters = 0
        self.lsev = 0
        self.per_lane_ncon = List[Int]()
        self.per_lane_iters = List[Int]()
        self.per_lane_lsev = List[Int]()


def _prep(
    mut d: Data[DTYPE, MDIMS, BATCH],
    mut mf: Model[DTYPE, MDIMS],
    mut scratch: DynamicsScratch[DTYPE, MDIMS, BATCH],
    ctx: DeviceContext,
) raises:
    """Smooth dynamics + detection up to the constraint seam — `test_newton_
    blocked_elliptic._prep`'s GPU branch, so both files feed the solve the
    same way. ⚠ SAP, not the O(N^2) loop: it is what the batched env runs."""
    comptime L_JOINT = Layout.row_major(NJOINT, MODEL_JOINT_SIZE)
    comptime L_M = Layout.row_major(BATCH, NV * NV)
    comptime L_NV = Layout.row_major(BATCH, NV)
    comptime L_QPOS = Layout.row_major(BATCH, NQ)
    forward_kinematics["gpu", DTYPE, BATCH=BATCH](d, mf, ctx)
    compute_body_velocities["gpu", DTYPE, BATCH=BATCH](d, mf, ctx)
    compute_subtree_com["gpu", DTYPE, BATCH=BATCH](d, mf, ctx)
    compute_cdof["gpu", DTYPE, BATCH=BATCH](d, mf, scratch, ctx)
    compute_mass_matrix["gpu", DTYPE, BATCH=BATCH](d, mf, scratch, ctx)
    ctx.enqueue_function[_armature_kernel[DTYPE, NV, NJOINT, BATCH]](
        mf.joints.lt["gpu", L_JOINT](), scratch.M.lt["gpu", L_M](),
        grid_dim=(BATCH,), block_dim=(1,),
    )
    ldl_factor["gpu", DTYPE, BATCH=BATCH](mf, scratch, ctx)
    compute_m_inv["gpu", DTYPE, BATCH=BATCH](mf, scratch, ctx)
    compute_bias_forces_rne["gpu", DTYPE, BATCH=BATCH](d, mf, scratch, ctx)
    ctx.enqueue_function[_fnet_passive_kernel[DTYPE, NQ, NV, NJOINT, BATCH]](
        d.qpos.lt["gpu", L_QPOS](), d.qvel.lt["gpu", L_NV](),
        d.qfrc.lt["gpu", L_NV](), mf.joints.lt["gpu", L_JOINT](),
        scratch.bias.lt["gpu", L_NV](), scratch.fnet.lt["gpu", L_NV](),
        grid_dim=(BATCH,), block_dim=(1,),
    )
    ldl_solve["gpu", DTYPE, BATCH=BATCH](mf, scratch, ctx)
    ctx.enqueue_function[_qacc_writeback_kernel[DTYPE, NV, BATCH]](
        scratch.qacc_ws.lt["gpu", L_NV](), d.qacc.lt["gpu", L_NV](),
        scratch.qacc_constrained.lt["gpu", L_NV](),
        grid_dim=(BATCH,), block_dim=(1,),
    )
    detect_contacts_sap["gpu", DTYPE, BATCH=BATCH](d, mf, ctx)


def _solve(
    ctx: DeviceContext, q: List[Float64], v: List[Float64], blocked: Bool
) raises -> Solved:
    """`q`/`v` hold BATCH lanes' words, lane-major."""
    var mf = Model[DTYPE, MDIMS]()
    M.init_fields[DTYPE](ctx, mf)
    # ⚠ NO `reset_data`: it is a single-env helper (its `Data` parameter is
    # batch 1) and every word it would set is overwritten below or starts at
    # the zero a fresh `Data` already has.
    var d = Data[DTYPE, MDIMS, BATCH]()
    for e in range(BATCH):
        for i in range(NQ):
            d.qpos.data[e * NQ + i] = Scalar[DTYPE](q[e * NQ + i])
        for i in range(NV):
            d.qvel.data[e * NV + i] = Scalar[DTYPE](v[e * NV + i])
            d.qfrc.data[e * NV + i] = 0
    var scratch = DynamicsScratch[DTYPE, MDIMS, BATCH]()
    var cscratch = ContactScratch[DTYPE, MDIMS, BATCH, JE_WS]()
    d.upload_all(ctx)
    scratch.upload_all(ctx)
    cscratch.upload_all(ctx)
    _prep(d, mf, scratch, ctx)
    if blocked:
        solve_newton_blocked[
            "gpu", DTYPE, CONE_TYPE=ConeType.ELLIPTIC, BATCH=BATCH,
            MAX_CONDIM = M.MAX_CONDIM, NOSLIP_ITER=0, JE_WS=JE_WS,
        ](d, mf, scratch, cscratch, ctx)
    else:
        # ⚠ `solve_newton` is the PER-ENV leg on Metal and the BLOCKED one on
        # NVIDIA. On a CUDA box run this arm with NEWTON_FORCE_PER_ENV = True,
        # or both arms are the same kernel and the comparison is an identity.
        solve_newton[
            "gpu", DTYPE, CONE_TYPE=ConeType.ELLIPTIC, BATCH=BATCH,
            MAX_CONDIM = M.MAX_CONDIM, NOSLIP_ITER=0, JE_WS=JE_WS,
        ](d, mf, scratch, cscratch, ctx)
    scratch.qacc_constrained.download(ctx)
    d.meta.download(ctx)
    d.contacts.download(ctx)
    ctx.synchronize()

    var out = Solved()
    for i in range(BATCH * NV):
        out.qacc.append(Float64(scratch.qacc_constrained.data[i]))
    var n = 0
    for e in range(BATCH):
        var ne = Int(d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
        if ne > MC:
            ne = MC
        out.per_lane_ncon.append(ne)
        out.per_lane_iters.append(
            Int(d.meta.data[e * METADATA_SIZE + META_IDX_NEWTON_ITER])
        )
        out.per_lane_lsev.append(
            Int(d.meta.data[e * METADATA_SIZE + META_IDX_LS_EVAL])
        )
        n += ne
    out.ncon = n
    out.iters = out.per_lane_iters[0]
    out.lsev = out.per_lane_lsev[0]
    for c in range(MC * BATCH):
        var cb = c * CONTACT_SIZE
        out.condim.append(Int(d.contacts.data[cb + CONTACT_IDX_CONDIM]))
        out.dist.append(Float64(d.contacts.data[cb + CONTACT_IDX_DIST]))
        out.forces.append(Float64(d.contacts.data[cb + CONTACT_IDX_FORCE_N]))
        out.forces.append(Float64(d.contacts.data[cb + CONTACT_IDX_FORCE_T1]))
        out.forces.append(Float64(d.contacts.data[cb + CONTACT_IDX_FORCE_T2]))
        out.forces.append(
            Float64(d.contacts.data[cb + CONTACT_IDX_FORCE_TORSION])
        )
    return out^


def main() raises:
    var a = argv()
    if len(a) < 2:
        raise Error(
            "usage: solve_at_pose.mojo <dumped-states.txt> [lane]   (the dump"
            " comes from libero_family_batched --dump-at-step N --dump-state)\n"
            "       solve_at_pose.mojo <states.txt> <lane> <step>"
        )
    var lane_a = Int(String(a[2])) if len(a) > 2 else 0
    var want_step = Int(String(a[3])) if len(a) > 3 else -1
    var lane_b = Int(String(a[4])) if len(a) > 4 else lane_a + 1
    var text: String
    with open(String(a[1]), "r") as fh:
        text = fh.read()
    var q = List[Float64]()
    var v = List[Float64]()
    var lines = text.split("\n")
    var lanes = List[Int]()
    lanes.append(lane_a)
    lanes.append(lane_b)
    for li in range(BATCH):
        var got_q = False
        var got_v = False
        for i in range(len(lines)):
            var l = String(String(lines[i]).strip())
            if l.byte_length() == 0:
                continue
            var t = l.split(" ")
            if len(t) < 5:
                continue
            if Int(String(t[2])) != lanes[li]:
                continue
            if want_step >= 0 and Int(String(t[4])) != want_step:
                continue
            if l.startswith("QPOS") and not got_q:
                got_q = True
                for k in range(5, len(t)):
                    q.append(Float64(String(t[k])))
            elif l.startswith("QVEL") and not got_v:
                got_v = True
                for k in range(5, len(t)):
                    v.append(Float64(String(t[k])))
    if len(q) != NQ * BATCH or len(v) != NV * BATCH:
        raise Error(
            "the dump gave " + String(len(q)) + " qpos and " + String(len(v))
            + " qvel words for lanes " + String(lane_a) + "," + String(lane_b)
            + "; this model is nq " + String(NQ) + " nv " + String(NV)
            + " over " + String(BATCH) + " lanes"
        )
    print("=" * 78)
    print("one elliptic solve, both legs — lanes", lane_a, lane_b, "step",
          want_step, "| nq", NQ, "nv", NV, "| max_contacts", MC)
    print("=" * 78)
    var ctx = DeviceContext()
    var per_env = _solve(ctx, q, v, False)
    var blocked = _solve(ctx, q, v, True)

    for e in range(BATCH):
        print("  lane", e, "per-env: ncon", per_env.per_lane_ncon[e], "iters",
              per_env.per_lane_iters[e], "lsev", per_env.per_lane_lsev[e],
              "| blocked: ncon", blocked.per_lane_ncon[e], "iters",
              blocked.per_lane_iters[e], "lsev", blocked.per_lane_lsev[e])
    if per_env.ncon != blocked.ncon:
        # ⚠ THE SOLVE DOES NOT CHANGE THE CONTACT SET: a difference here is the
        # detection, run twice on the same words, and would mean the two arms
        # were not fed the same thing.
        print("  ⚠ THE INPUTS DIFFER — the contact sets are not the same;"
              " nothing below is a solver comparison")
    var worst_qacc = 0.0
    var worst_i = -1
    for i in range(BATCH * NV):
        var dv = abs(per_env.qacc[i] - blocked.qacc[i])
        if dv > worst_qacc:
            worst_qacc = dv
            worst_i = i
    print("  worst |qacc| difference", worst_qacc, "at dof", worst_i,
          "(lane", worst_i // NV, ")")
    var nf = len(per_env.forces) if len(per_env.forces) < len(blocked.forces) else len(blocked.forces)
    var worst_f = 0.0
    var worst_c = -1
    for i in range(nf):
        var dv = abs(per_env.forces[i] - blocked.forces[i])
        if dv > worst_f:
            worst_f = dv
            worst_c = i // 4
    print("  worst |contact force| difference", worst_f, "at contact", worst_c)
    if worst_c >= 0:
        print("     that contact: dist", per_env.dist[worst_c], " condim",
              per_env.condim[worst_c], " forces per-env",
              per_env.forces[worst_c * 4], per_env.forces[worst_c * 4 + 1],
              per_env.forces[worst_c * 4 + 2], per_env.forces[worst_c * 4 + 3],
              " blocked", blocked.forces[worst_c * 4],
              blocked.forces[worst_c * 4 + 1], blocked.forces[worst_c * 4 + 2],
              blocked.forces[worst_c * 4 + 3])
    if worst_qacc == 0.0 and worst_f == 0.0:
        print()
        print("=== the two legs agree BIT FOR BIT on this state ===")
    else:
        print()
        print("=== they differ — this state is the reproduction ===")
