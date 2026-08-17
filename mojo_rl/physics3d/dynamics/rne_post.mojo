"""`mj_rnePostConstraint` over per-field tensors — `cacc` and `cfrc_int`.

MuJoCo runs this once per step, inside `mj_sensorAcc`, whenever the model
declares an acceleration-stage sensor (`engine_sensor.c:908` gates it on
`accelerometer` / `force` / `torque` / `framelinacc` / `frameangacc` /
`subtreeangmom`). It is NOT part of the dynamics: nothing it writes feeds
back into `qacc`. That is why this stage is comptime-gated on the
integrator's `RNE_POST` parameter — only quadruped pays for it.

    cacc[0]      = (0, -gravity)                      (world)
    cacc[b]      = cacc[parent] + cdof_dot*qvel + cdof*qacc
    cfrc_body[b] = cinert[b]*cacc[b] + cvel[b] x* (cinert[b]*cvel[b])
    cfrc_int[b]  = cfrc_body[b] - cfrc_ext[b]
    cfrc_int[parent] += cfrc_int[b]                   (leaves -> root)

Every quantity is world-oriented torque:force, referenced at the subtree
CoM of the body's kinematic root — the same convention as `Data.cfrc_ext`.

WHY THIS IS NOT A COPY OF THE RNE BIAS PASS. It is the same recursion with
one extra term, and the recursion is LINEAR in it, so the `cdof*qacc`
contribution can be accumulated in its own forward sweep afterwards:

    extra[b] = extra[parent] + sum_{dof in b} cdof[dof] * qacc[dof]
    cacc[b] += extra[b]

That lets `_rne_fwd_body` / `_rne_cinert_body` / `_rne_cfrc_body` /
`_rne_backward_env` be reused VERBATIM from `dynamics/rne.mojo` — the bias
pass and this one cannot drift apart, and the hot RNE path is untouched.
`_rne_fwd_body` also refills the `crb` scratch with per-body `cvel`, so
this stage does not depend on what the constraint solvers left behind.

WHICH `qacc`. MuJoCo's `mj_sensorAcc` reads `d->qacc` as written by
`mj_fwdConstraint` — BEFORE the Euler integrator's implicit-damping
re-solve. Ours is `scratch.qacc_constrained`; `Data.qacc` is the damped one
(`euler._finalize_env` overwrites it). Passing `d.qacc` here would be
silently wrong on any model with joint damping — which is every model that
wants these sensors.

EQUALITY CONSTRAINTS ARE NOT MAPPED INTO `cfrc_ext`. MuJoCo walks the
equality rows and adds `connect`/`weld` forces (`mjEQ_JOINT` and
`mjEQ_TENDON` contribute nothing — they only advance the row cursor).
quadruped's four equalities are all `<equality><tendon>`, so the walk is a
no-op for it; a model with `connect`/`weld` equalities plus a force/torque
sensor would read low here. `compute_rne_post` raises on that combination
rather than returning a plausible wrong number.

`xfrc_applied` is likewise absent: we have no such field.
"""

from std.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from ..fields import (
    Data,
    Model,
    DynamicsScratch,
    Dims,
    DimsLike,
    AsStatic,
    Scratch,
    cap,
    DYN1,
    DYN2,
    rl1,
    rl2,
)
from ..joint_types import JNT_FREE, JNT_BALL
from ..collision.contact_frame import contact_tangent_frame
from .rne import (
    _max_one,
    _rne_fwd_body,
    _rne_cinert_body,
    _rne_cfrc_body,
    _rne_backward_env,
)
from ..gpu.constants import (
    CONTACT_SIZE,
    METADATA_SIZE,
    META_IDX_NUM_CONTACTS,
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    MODEL_META_IDX_GRAVITY_X,
    MODEL_META_IDX_GRAVITY_Y,
    MODEL_META_IDX_GRAVITY_Z,
    BODY_IDX_PARENT,
    BODY_IDX_ROOTID,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_DOF_ADR,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_POS_X,
    CONTACT_IDX_POS_Y,
    CONTACT_IDX_POS_Z,
    CONTACT_IDX_NX,
    CONTACT_IDX_NY,
    CONTACT_IDX_NZ,
    CONTACT_IDX_FRAME_T1_X,
    CONTACT_IDX_FRAME_T1_Y,
    CONTACT_IDX_FRAME_T1_Z,
    CONTACT_IDX_FORCE_N,
    CONTACT_IDX_FORCE_T1,
    CONTACT_IDX_FORCE_T2,
    CONTACT_IDX_FORCE_TORSION,
    CONTACT_IDX_FORCE_ROLL1,
    CONTACT_IDX_FORCE_ROLL2,
)

comptime RNE_POST_TPB: Int = 64


@always_inline
def _cfrc_ext_env[
    DTYPE: DType,
    D: DimsLike,
    L_CONTACTS: Layout,
    L_DMETA: Layout,
    L_SUBTREE_COM: Layout,
    L_BODIES: Layout,
    L_CFRC_EXT: Layout,
](
    env: Int,
    dims: D,
    contacts: LayoutTensor[
        DTYPE, L_CONTACTS,
        MutAnyOrigin,
    ],
    dmeta: LayoutTensor[
        DTYPE, L_DMETA, MutAnyOrigin
    ],
    subtree_com: LayoutTensor[
        DTYPE, L_SUBTREE_COM, MutAnyOrigin
    ],
    bodies: LayoutTensor[
        DTYPE, L_BODIES, MutAnyOrigin
    ],
    cfrc_ext: LayoutTensor[
        DTYPE, L_CFRC_EXT, MutAnyOrigin
    ],
):
    """Contact forces accumulated per body at the root's subtree CoM.

    Same arithmetic as `gpu/cfrc_ext_gpu.mojo`, but reading the already
    computed `Data.subtree_com` and the `BODY_IDX_ROOTID` column instead of
    rebuilding both from body masses and parents.

    SIGN. `contacts[BODY_A]` is the body of geom[0] — MuJoCo's "body 1",
    which it SUBTRACTS (`mju_subFrom`), adding to body 2. Our stored contact
    force is the force on A, so A adds and B subtracts. (The one existing
    consumer, Ant's contact_cost, takes a norm and could not have caught a
    flipped sign; the quadruped force-sensor gate can.)
    """
    var nbody = dims.get_nbody()
    var max_contacts = dims.get_max_contacts()
    for i in range(nbody * 6):
        cfrc_ext[env, i] = Scalar[DTYPE](0)

    var ncon = Int(rebind[Scalar[DTYPE]](dmeta[env, META_IDX_NUM_CONTACTS]))

    for ci in range(max_contacts):
        if ci >= ncon:
            break
        var cb = ci * CONTACT_SIZE

        var nx = rebind[Scalar[DTYPE]](contacts[env, cb + CONTACT_IDX_NX])
        var ny = rebind[Scalar[DTYPE]](contacts[env, cb + CONTACT_IDX_NY])
        var nz = rebind[Scalar[DTYPE]](contacts[env, cb + CONTACT_IDX_NZ])
        # FRAME_T1 is a HINT, not a tangent — it has had no fallback, no
        # Gram-Schmidt and no normalization applied, and non-capsule pairs
        # never write it at all. Reading it raw gave the tangential force a
        # garbage direction while the normal component stayed right, because
        # that one only needs `n`. See collision/contact_frame.mojo.
        var frame = contact_tangent_frame[DTYPE](
            nx,
            ny,
            nz,
            rebind[Scalar[DTYPE]](contacts[env, cb + CONTACT_IDX_FRAME_T1_X]),
            rebind[Scalar[DTYPE]](contacts[env, cb + CONTACT_IDX_FRAME_T1_Y]),
            rebind[Scalar[DTYPE]](contacts[env, cb + CONTACT_IDX_FRAME_T1_Z]),
        )
        var t1x = frame[0]
        var t1y = frame[1]
        var t1z = frame[2]
        var t2x = frame[3]
        var t2y = frame[4]
        var t2z = frame[5]

        var f_n = rebind[Scalar[DTYPE]](contacts[env, cb + CONTACT_IDX_FORCE_N])
        var f_t1 = rebind[Scalar[DTYPE]](
            contacts[env, cb + CONTACT_IDX_FORCE_T1]
        )
        var f_t2 = rebind[Scalar[DTYPE]](
            contacts[env, cb + CONTACT_IDX_FORCE_T2]
        )
        var f_tors = rebind[Scalar[DTYPE]](
            contacts[env, cb + CONTACT_IDX_FORCE_TORSION]
        )
        var f_r1 = rebind[Scalar[DTYPE]](
            contacts[env, cb + CONTACT_IDX_FORCE_ROLL1]
        )
        var f_r2 = rebind[Scalar[DTYPE]](
            contacts[env, cb + CONTACT_IDX_FORCE_ROLL2]
        )

        var fw_x = f_n * nx + f_t1 * t1x + f_t2 * t2x
        var fw_y = f_n * ny + f_t1 * t1y + f_t2 * t2y
        var fw_z = f_n * nz + f_t1 * t1z + f_t2 * t2z
        var tw_x = f_tors * nx + f_r1 * t1x + f_r2 * t2x
        var tw_y = f_tors * ny + f_r1 * t1y + f_r2 * t2y
        var tw_z = f_tors * nz + f_r1 * t1z + f_r2 * t2z

        var px = rebind[Scalar[DTYPE]](contacts[env, cb + CONTACT_IDX_POS_X])
        var py = rebind[Scalar[DTYPE]](contacts[env, cb + CONTACT_IDX_POS_Y])
        var pz = rebind[Scalar[DTYPE]](contacts[env, cb + CONTACT_IDX_POS_Z])

        var ka = Int(rebind[Scalar[DTYPE]](contacts[env, cb + CONTACT_IDX_BODY_A]))
        var kb = Int(rebind[Scalar[DTYPE]](contacts[env, cb + CONTACT_IDX_BODY_B]))

        for side in range(2):
            var k = ka if side == 0 else kb
            if k <= 0:
                continue
            var rid = Int(rebind[Scalar[DTYPE]](bodies[k, BODY_IDX_ROOTID]))
            var dx = rebind[Scalar[DTYPE]](subtree_com[env, rid * 3 + 0]) - px
            var dy = rebind[Scalar[DTYPE]](subtree_com[env, rid * 3 + 1]) - py
            var dz = rebind[Scalar[DTYPE]](subtree_com[env, rid * 3 + 2]) - pz
            # transformSpatial(flg_force=1): torque -= (newpos-oldpos) x force
            var mx = tw_x - (dy * fw_z - dz * fw_y)
            var my = tw_y - (dz * fw_x - dx * fw_z)
            var mz = tw_z - (dx * fw_y - dy * fw_x)

            var s = Scalar[DTYPE](1) if side == 0 else Scalar[DTYPE](-1)
            var o = k * 6
            cfrc_ext[env, o + 0] = (
                rebind[Scalar[DTYPE]](cfrc_ext[env, o + 0]) + s * mx
            )
            cfrc_ext[env, o + 1] = (
                rebind[Scalar[DTYPE]](cfrc_ext[env, o + 1]) + s * my
            )
            cfrc_ext[env, o + 2] = (
                rebind[Scalar[DTYPE]](cfrc_ext[env, o + 2]) + s * mz
            )
            cfrc_ext[env, o + 3] = (
                rebind[Scalar[DTYPE]](cfrc_ext[env, o + 3]) + s * fw_x
            )
            cfrc_ext[env, o + 4] = (
                rebind[Scalar[DTYPE]](cfrc_ext[env, o + 4]) + s * fw_y
            )
            cfrc_ext[env, o + 5] = (
                rebind[Scalar[DTYPE]](cfrc_ext[env, o + 5]) + s * fw_z
            )


@always_inline
def _rne_post_env[
    DTYPE: DType,
    D: DimsLike,
    L_QVEL: Layout,
    L_XQUAT: Layout,
    L_XIPOS: Layout,
    L_CONTACTS: Layout,
    L_DMETA: Layout,
    L_BODIES: Layout,
    L_JOINTS: Layout,
    L_MMETA: Layout,
    L_CDOF: Layout,
    L_CRB: Layout,
    L_CVEL: Layout,
](
    env: Int,
    dims: D,
    qvel: LayoutTensor[DTYPE, L_QVEL, MutAnyOrigin],
    qacc: LayoutTensor[DTYPE, L_QVEL, MutAnyOrigin],
    xquat: LayoutTensor[
        DTYPE, L_XQUAT, MutAnyOrigin
    ],
    xipos: LayoutTensor[
        DTYPE, L_XIPOS, MutAnyOrigin
    ],
    subtree_com: LayoutTensor[
        DTYPE, L_XIPOS, MutAnyOrigin
    ],
    contacts: LayoutTensor[
        DTYPE, L_CONTACTS,
        MutAnyOrigin,
    ],
    dmeta: LayoutTensor[
        DTYPE, L_DMETA, MutAnyOrigin
    ],
    bodies: LayoutTensor[
        DTYPE, L_BODIES, MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, L_JOINTS, MutAnyOrigin
    ],
    mmeta: LayoutTensor[
        DTYPE, L_MMETA, MutAnyOrigin
    ],
    cdof: LayoutTensor[DTYPE, L_CDOF, MutAnyOrigin],
    crb: LayoutTensor[
        DTYPE, L_CRB, MutAnyOrigin
    ],
    cvel: LayoutTensor[
        DTYPE, L_CVEL, MutAnyOrigin
    ],
    cacc: LayoutTensor[
        DTYPE, L_CVEL, MutAnyOrigin
    ],
    cfrc_ext: LayoutTensor[
        DTYPE, L_CVEL, MutAnyOrigin
    ],
    cfrc_int: LayoutTensor[
        DTYPE, L_CVEL, MutAnyOrigin
    ],
):
    """One env's `mj_rnePostConstraint`. See the module docstring."""
    var nbody = dims.get_nbody()
    var njoint = dims.get_njoint()
    var gx = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_GRAVITY_X])
    var gy = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_GRAVITY_Y])
    var gz = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_GRAVITY_Z])

    comptime B6 = cap[D.NBODY]() * 6
    # ⚠ `nbody * 6`, NOT `B6`. `B6` is a CAP — 0 on a dynamic provider — so
    # bounding this loop with it would leave `cacc` holding the previous
    # step's values on that leg, silently. A cap belongs in a `Scratch[...]`
    # size and nowhere else; `scratchpad/p2b2/audit_caps.py` checks that.
    for i in range(nbody * 6):
        cacc[env, i] = Scalar[DTYPE](0)
    # World acceleration = -gravity. `_rne_fwd_body` writes this into every
    # body whose parent is 0 rather than reading it from here, so body 0's own
    # row is never used by the recursion — but MuJoCo sets it, and leaving it
    # zero would make `d.cacc` disagree with `mjData.cacc` on the one row a
    # parity gate is most likely to print first.
    cacc[env, 3] = -gx
    cacc[env, 4] = -gy
    cacc[env, 5] = -gz
    for i in range(nbody * 6):
        crb[env, i] = Scalar[DTYPE](0)

    comptime CIN = cap[D.NBODY]() * 10
    var cinert_g = Scratch[Scalar[DTYPE], CIN](nbody * 10, uninitialized=0)
    for i in range(nbody * 10):
        cinert_g[i] = Scalar[DTYPE](0)
    for b in range(nbody):
        _rne_cinert_body[DTYPE](
            env, b, xquat, xipos, subtree_com, bodies, cinert_g
        )

    # 1. cvel (into crb) + the qacc-free part of cacc, verbatim from RNE.
    for b in range(1, nbody):
        _rne_fwd_body[DTYPE](
            env, b, gx, gy, gz, dims, qvel, bodies, joints, cdof, crb, cacc
        )

    # 2. The cdof*qacc term, as its own forward sweep (see docstring).
    var extra = Scratch[Scalar[DTYPE], B6](nbody * 6, uninitialized=0)
    for i in range(nbody * 6):
        extra[i] = Scalar[DTYPE](0)
    for b in range(1, nbody):
        var parent = Int(rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_PARENT]))
        for k in range(6):
            extra[b * 6 + k] = extra[parent * 6 + k]
        for j in range(njoint):
            if Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_BODY_ID])) != b:
                continue
            var jt = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
            var adr = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR]))
            var ndof = 1
            if jt == JNT_FREE:
                ndof = 6
            elif jt == JNT_BALL:
                ndof = 3
            for dd in range(ndof):
                var dof = adr + dd
                var a = rebind[Scalar[DTYPE]](qacc[env, dof])
                for k in range(6):
                    extra[b * 6 + k] = extra[b * 6 + k] + rebind[
                        Scalar[DTYPE]
                    ](cdof[env, dof * 6 + k]) * a
        for k in range(6):
            cacc[env, b * 6 + k] = (
                rebind[Scalar[DTYPE]](cacc[env, b * 6 + k]) + extra[b * 6 + k]
            )

    # 3. External (contact) forces per body.
    _cfrc_ext_env[DTYPE](
        env, dims, contacts, dmeta, subtree_com, bodies, cfrc_ext
    )

    # 4. cfrc_int = cfrc_body - cfrc_ext, then accumulate leaves -> root.
    for b in range(nbody):
        _rne_cfrc_body[DTYPE](
            env, b, cinert_g, crb, cacc, cfrc_int
        )
    for i in range(nbody * 6):
        cfrc_int[env, i] = rebind[Scalar[DTYPE]](
            cfrc_int[env, i]
        ) - rebind[Scalar[DTYPE]](cfrc_ext[env, i])
    # ⚠ `_rne_backward_env` stops at parent > 0, where MuJoCo accumulates into
    # body 0 as well. `cfrc_int[0]` is therefore ours-only; no sensor reads it
    # (`site_bodyid` is never 0), and for a system with no external wrench the
    # two agree anyway, because the sum it would hold is the net wrench.
    _rne_backward_env[DTYPE](env, dims, bodies, cfrc_int)

    # 5. Publish cvel out of the crb scratch (the accelerometer needs it).
    for b in range(nbody):
        for k in range(6):
            cvel[env, b * 6 + k] = rebind[Scalar[DTYPE]](
                crb[env, b * 6 + k]
            )


def _rne_post_kernel[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    BATCH: Int,
](
    qvel: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    qacc: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV), MutAnyOrigin],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
    xipos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    subtree_com: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    contacts: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE),
        MutAnyOrigin,
    ],
    dmeta: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, METADATA_SIZE), MutAnyOrigin
    ],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    joints: LayoutTensor[
        DTYPE, Layout.row_major(NJOINT, MODEL_JOINT_SIZE), MutAnyOrigin
    ],
    mmeta: LayoutTensor[
        DTYPE, Layout.row_major(MODEL_META_SIZE), MutAnyOrigin
    ],
    cdof: LayoutTensor[DTYPE, Layout.row_major(BATCH, NV * 6), MutAnyOrigin],
    crb: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 10), MutAnyOrigin
    ],
    cvel: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 6), MutAnyOrigin
    ],
    cacc: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 6), MutAnyOrigin
    ],
    cfrc_ext: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 6), MutAnyOrigin
    ],
    cfrc_int: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 6), MutAnyOrigin
    ],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _rne_post_env[DTYPE](
        env, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, max_contacts=MAX_CONTACTS](), qvel, qacc, xquat, xipos, subtree_com, contacts, dmeta, bodies,
        joints, mmeta, cdof, crb, cvel, cacc, cfrc_ext, cfrc_int,
    )


def compute_rne_post[

    target: StaticString,
    DTYPE: DType,
    D: DimsLike,
    BATCH: Int = 1,
    # Appended, not grouped with NEXCLUDE — see `fields.Model`.
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    mut scratch: DynamicsScratch[DTYPE, D, BATCH],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Fill `d.cacc` / `d.cfrc_int` (and `d.cvel` / `d.cfrc_ext`) for the
    CURRENT state. Run between the constraint solve and the integration —
    that is where MuJoCo's `mj_sensorAcc` sits, and the FK products, the
    contact forces and `scratch.qacc_constrained` are all valid there."""
    comptime L_NV = Layout.row_major(BATCH, D.NV)
    comptime L_B3 = Layout.row_major(BATCH, D.NBODY * 3)
    comptime L_B4 = Layout.row_major(BATCH, D.NBODY * 4)
    comptime L_B6 = Layout.row_major(BATCH, D.NBODY * 6)
    comptime L_B10 = Layout.row_major(BATCH, D.NBODY * 10)
    comptime L_CON = Layout.row_major(BATCH, D.MAX_CONTACTS * CONTACT_SIZE)
    comptime L_DMETA = Layout.row_major(BATCH, METADATA_SIZE)
    comptime L_BODY = Layout.row_major(D.NBODY, MODEL_BODY_SIZE)
    comptime L_JOINT = Layout.row_major(D.NJOINT, MODEL_JOINT_SIZE)
    comptime L_MMETA = Layout.row_major(MODEL_META_SIZE)
    comptime L_CDOF = Layout.row_major(BATCH, D.NV * 6)

    comptime if target == "cpu":
        var dm = d.dims
        var rl_NV = rl2(BATCH, dm.get_nv())
        var rl_B4 = rl2(BATCH, dm.get_nbody() * 4)
        var rl_B3 = rl2(BATCH, dm.get_nbody() * 3)
        var rl_CON = rl2(BATCH, dm.get_max_contacts() * CONTACT_SIZE)
        var rl_DMETA = rl2(BATCH, METADATA_SIZE)
        var rl_BODY = rl2(dm.get_nbody(), MODEL_BODY_SIZE)
        var rl_JOINT = rl2(dm.get_njoint(), MODEL_JOINT_SIZE)
        var rl_MMETA = rl1(MODEL_META_SIZE)
        var rl_CDOF = rl2(BATCH, dm.get_nv() * 6)
        var rl_B10 = rl2(BATCH, dm.get_nbody() * 10)
        var rl_B6 = rl2(BATCH, dm.get_nbody() * 6)
        var qvel_v = d.qvel.lt_dyn["cpu", DYN2](rl_NV)
        var qacc_v = scratch.qacc_constrained.lt_dyn["cpu", DYN2](rl_NV)
        var xquat_v = d.xquat.lt_dyn["cpu", DYN2](rl_B4)
        var xipos_v = d.xipos.lt_dyn["cpu", DYN2](rl_B3)
        var stcom_v = d.subtree_com.lt_dyn["cpu", DYN2](rl_B3)
        var con_v = d.contacts.lt_dyn["cpu", DYN2](rl_CON)
        var dmeta_v = d.meta.lt_dyn["cpu", DYN2](rl_DMETA)
        var bodies_v = m.bodies.lt_dyn["cpu", DYN2](rl_BODY)
        var joints_v = m.joints.lt_dyn["cpu", DYN2](rl_JOINT)
        var mmeta_v = m.meta.lt_dyn["cpu", DYN1](rl_MMETA)
        var cdof_v = scratch.cdof.lt_dyn["cpu", DYN2](rl_CDOF)
        var crb_v = scratch.crb.lt_dyn["cpu", DYN2](rl_B10)
        var cvel_v = d.cvel.lt_dyn["cpu", DYN2](rl_B6)
        var cacc_v = d.cacc.lt_dyn["cpu", DYN2](rl_B6)
        var cfrc_ext_v = d.cfrc_ext.lt_dyn["cpu", DYN2](rl_B6)
        var cfrc_int_v = d.cfrc_int.lt_dyn["cpu", DYN2](rl_B6)
        for e in range(BATCH):
            _rne_post_env[DTYPE](
                e, AsStatic[D](), qvel_v, qacc_v, xquat_v, xipos_v, stcom_v, con_v, dmeta_v,
                bodies_v, joints_v, mmeta_v, cdof_v, crb_v, cvel_v, cacc_v,
                cfrc_ext_v, cfrc_int_v,
            )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + RNE_POST_TPB - 1) // RNE_POST_TPB
        c.enqueue_function[
            _rne_post_kernel[
                DTYPE, D.NV, D.NBODY, D.NJOINT, D.MAX_CONTACTS, BATCH
            ]
        ](
            d.qvel.lt["gpu", L_NV](),
            scratch.qacc_constrained.lt["gpu", L_NV](),
            d.xquat.lt["gpu", L_B4](),
            d.xipos.lt["gpu", L_B3](),
            d.subtree_com.lt["gpu", L_B3](),
            d.contacts.lt["gpu", L_CON](),
            d.meta.lt["gpu", L_DMETA](),
            m.bodies.lt["gpu", L_BODY](),
            m.joints.lt["gpu", L_JOINT](),
            m.meta.lt["gpu", L_MMETA](),
            scratch.cdof.lt["gpu", L_CDOF](),
            scratch.crb.lt["gpu", L_B10](),
            d.cvel.lt["gpu", L_B6](),
            d.cacc.lt["gpu", L_B6](),
            d.cfrc_ext.lt["gpu", L_B6](),
            d.cfrc_int.lt["gpu", L_B6](),
            grid_dim=(BLOCKS,),
            block_dim=(RNE_POST_TPB,),
        )
