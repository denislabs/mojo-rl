"""External contact force accumulation per body (cfrc_ext).

Computes cfrc_ext: the net external contact force on each body,
expressed in subtree CoM-based world-oriented 6D spatial form.

Matches MuJoCo's mj_rnePostConstraint contact force contribution:
- For each active contact, extract the world-frame force/torque wrench.
- Transform it from the contact point to the subtree CoM of each body.
- Accumulate: body_a gets -cfrc (reaction), body_b gets +cfrc.

Layout of cfrc_ext[body * 6 + 0..5]:
  [0..2] = torque (Nm) in world frame at subtree CoM
  [3..5] = force (N)  in world frame

Reference: MuJoCo engine_core_smooth.c: mj_rnePostConstraint (contact section).
"""

from ..types import Model, Data, _max_one, ConeType


fn compute_cfrc_ext[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int = 0,
    MAX_EQUALITY: Int = 0,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    MAX_TENDON: Int = 0,
    NSITE: Int = 0,
](
    model: Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        MAX_EQUALITY,
        CONE_TYPE,
        MAX_TENDON,
    NSITE,
    ],
    mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
):
    """Compute cfrc_ext: contact forces per body in subtree CoM-based frame.

    Accumulates constraint (contact) forces into cfrc_ext[body][6]:
      [torque_x, torque_y, torque_z, force_x, force_y, force_z]
    expressed in world orientation at the subtree CoM of each body's root.

    Matches MuJoCo mj_rnePostConstraint (contact force section only).
    xfrc_applied forces and equality-constraint forces are not included
    (they are zero in standard RL environments).

    Args:
        model: Static model configuration.
        data: Mutable simulation state. cfrc_ext is written here.
    """
    comptime ZERO = Scalar[DTYPE](0)
    comptime EPS = Scalar[DTYPE](1e-10)

    # -------------------------------------------------------------------------
    # 1. Zero cfrc_ext
    # -------------------------------------------------------------------------
    for i in range(NBODY * 6):
        data.cfrc_ext[i] = ZERO

    # -------------------------------------------------------------------------
    # 2. Compute subtree_com for each body (backward pass over kinematic tree).
    #    subtree_com[i] = mass-weighted CoM of body i and all its descendants.
    #    This matches MuJoCo's subtree_com computation in mj_comPos.
    # -------------------------------------------------------------------------
    # Accumulate mass-weighted positions
    var stmass = InlineArray[Scalar[DTYPE], NBODY](uninitialized=True)
    var stcom = InlineArray[Scalar[DTYPE], NBODY * 3](uninitialized=True)

    for i in range(NBODY):
        var m = model.body_mass[i]
        stmass[i] = m
        stcom[i * 3 + 0] = m * data.xipos[i * 3 + 0]
        stcom[i * 3 + 1] = m * data.xipos[i * 3 + 1]
        stcom[i * 3 + 2] = m * data.xipos[i * 3 + 2]

    # Backward sweep: add child contribution to parent
    for i in range(NBODY - 1, 0, -1):  # skip worldbody (index 0)
        var p = model.body_parent[i]
        stmass[p] += stmass[i]
        stcom[p * 3 + 0] += stcom[i * 3 + 0]
        stcom[p * 3 + 1] += stcom[i * 3 + 1]
        stcom[p * 3 + 2] += stcom[i * 3 + 2]

    # Normalize to get CoM position
    for i in range(NBODY):
        var sm = stmass[i]
        if sm > EPS:
            stcom[i * 3 + 0] = stcom[i * 3 + 0] / sm
            stcom[i * 3 + 1] = stcom[i * 3 + 1] / sm
            stcom[i * 3 + 2] = stcom[i * 3 + 2] / sm
        else:
            # Fall back to body CoM (xipos) for zero-mass bodies
            stcom[i * 3 + 0] = data.xipos[i * 3 + 0]
            stcom[i * 3 + 1] = data.xipos[i * 3 + 1]
            stcom[i * 3 + 2] = data.xipos[i * 3 + 2]

    # -------------------------------------------------------------------------
    # 3. Compute body_rootid for each body.
    #    body_rootid[k] = the first non-worldbody ancestor of k (or k itself
    #    if k's parent is worldbody).  MuJoCo uses subtree_com[rootid[k]] as
    #    the spatial reference for forces on body k.
    # -------------------------------------------------------------------------
    var rootid = InlineArray[Int, NBODY](uninitialized=True)
    rootid[0] = 0
    for i in range(1, NBODY):
        var p = model.body_parent[i]
        if p == 0:
            rootid[i] = i  # immediate child of worldbody: is its own root
        else:
            rootid[i] = rootid[p]

    # -------------------------------------------------------------------------
    # 4. Accumulate contact forces into cfrc_ext.
    #    For each contact:
    #      a) Convert contact-local force/torque to world frame.
    #      b) Transform from contact point to subtree_com[rootid[body]]
    #         using the spatial force transform (moment-arm correction).
    #      c) body_a (geom[0]) gets -cfrc (Newton's 3rd law reaction).
    #         body_b (geom[1]) gets +cfrc (direct force).
    # -------------------------------------------------------------------------
    for ci in range(data.num_contacts):
        var con = data.contacts[ci]

        # Contact frame axes
        var nx = con.normal_x
        var ny = con.normal_y
        var nz = con.normal_z
        var t1x = con.frame_t1_x
        var t1y = con.frame_t1_y
        var t1z = con.frame_t1_z
        # T2 = normal × T1
        var t2x = ny * t1z - nz * t1y
        var t2y = nz * t1x - nx * t1z
        var t2z = nx * t1y - ny * t1x

        # Contact forces in contact-local frame:
        #   result[0] = force_n  (along normal)
        #   result[1] = force_t1 (along T1)
        #   result[2] = force_t2 (along T2)
        #   result[3] = force_torsion (about normal, condim>=4)
        #   result[4] = force_roll1  (about T1, condim>=6)
        #   result[5] = force_roll2  (about T2, condim>=6)
        var f_n = con.force_n
        var f_t1 = con.force_t1
        var f_t2 = con.force_t2
        var f_tors = con.force_torsion
        var f_roll1 = con.force_roll1
        var f_roll2 = con.force_roll2

        # World-frame force: frame^T * [f_n, f_t1, f_t2]
        # = f_n*normal + f_t1*T1 + f_t2*T2
        var fw_x = f_n * nx + f_t1 * t1x + f_t2 * t2x
        var fw_y = f_n * ny + f_t1 * t1y + f_t2 * t2y
        var fw_z = f_n * nz + f_t1 * t1z + f_t2 * t2z

        # World-frame torque: frame^T * [f_tors, f_roll1, f_roll2]
        # = f_tors*normal + f_roll1*T1 + f_roll2*T2
        var tw_x = f_tors * nx + f_roll1 * t1x + f_roll2 * t2x
        var tw_y = f_tors * ny + f_roll1 * t1y + f_roll2 * t2y
        var tw_z = f_tors * nz + f_roll1 * t1z + f_roll2 * t2z

        # Contact point (world)
        var cx = con.pos_x
        var cy = con.pos_y
        var cz = con.pos_z

        # body_a (robot body, maps to MuJoCo geom[1]) → add (direct force)
        var ka = con.body_a
        if ka > 0:
            var rid = rootid[ka]
            var scx = stcom[rid * 3 + 0]
            var scy = stcom[rid * 3 + 1]
            var scz = stcom[rid * 3 + 2]

            # Spatial force transform (flg_force=1):
            # dif = newpos - oldpos = subtree_com - contact_pos
            # cfrc_com[torque] = torque - dif × force
            # cfrc_com[force]  = force  (unchanged)
            var dx = scx - cx
            var dy = scy - cy
            var dz = scz - cz
            var cx_ = dy * fw_z - dz * fw_y
            var cy_ = dz * fw_x - dx * fw_z
            var cz_ = dx * fw_y - dy * fw_x

            data.cfrc_ext[ka * 6 + 0] += tw_x - cx_
            data.cfrc_ext[ka * 6 + 1] += tw_y - cy_
            data.cfrc_ext[ka * 6 + 2] += tw_z - cz_
            data.cfrc_ext[ka * 6 + 3] += fw_x
            data.cfrc_ext[ka * 6 + 4] += fw_y
            data.cfrc_ext[ka * 6 + 5] += fw_z

        # body_b (typically worldbody=0, maps to MuJoCo geom[0]) → subtract (reaction)
        var kb = con.body_b
        if kb > 0:
            var rid = rootid[kb]
            var scx = stcom[rid * 3 + 0]
            var scy = stcom[rid * 3 + 1]
            var scz = stcom[rid * 3 + 2]

            var dx = scx - cx
            var dy = scy - cy
            var dz = scz - cz
            var cx_ = dy * fw_z - dz * fw_y
            var cy_ = dz * fw_x - dx * fw_z
            var cz_ = dx * fw_y - dy * fw_x

            data.cfrc_ext[kb * 6 + 0] -= tw_x - cx_
            data.cfrc_ext[kb * 6 + 1] -= tw_y - cy_
            data.cfrc_ext[kb * 6 + 2] -= tw_z - cz_
            data.cfrc_ext[kb * 6 + 3] -= fw_x
            data.cfrc_ext[kb * 6 + 4] -= fw_y
            data.cfrc_ext[kb * 6 + 5] -= fw_z
