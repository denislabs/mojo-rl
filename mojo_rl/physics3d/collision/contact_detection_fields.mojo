"""Contact detection over per-field tensors (migration P2, single-source).

Per-field port of `_geom_world_pos_gpu` + `detect_contacts_gpu`
(collision/contact_detection.mojo) — arithmetic verbatim. Reads FK products
(`d.xpos`, `d.xquat`) + geom/body records + model meta + exclude pairs;
writes packed contact records into `d.contacts` and the contact count into
`d.meta` (META_IDX_NUM_CONTACTS).

Operands (8): xpos, xquat (data) + geoms, bodies, mmeta, excludes (model)
+ contacts, smeta (data outputs). Mesh collision (GJK/EPA + plane-mesh
vertex scans) is NOT ported: the dispatcher gates on NMESH_VERTS == 0 and
mesh branches degrade to `continue`."""

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from ..kinematics.quat_math import gpu_quat_rotate, gpu_quat_mul
from ..constants import (
    GEOM_SPHERE,
    GEOM_CAPSULE,
    GEOM_BOX,
    GEOM_PLANE,
    GEOM_CYLINDER,
    GEOM_MESH,
)
from ..fields import DataFields, ModelFields
from ..gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_GEOM_SIZE,
    MODEL_META_SIZE,
    METADATA_SIZE,
    MODEL_META_IDX_NEXCLUDE,
    BODY_IDX_PARENT,
    BODY_IDX_WELDID,
    META_IDX_NUM_CONTACTS,
    CONTACT_SIZE,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_POS_X,
    CONTACT_IDX_POS_Y,
    CONTACT_IDX_POS_Z,
    CONTACT_IDX_NX,
    CONTACT_IDX_NY,
    CONTACT_IDX_NZ,
    CONTACT_IDX_DIST,
    CONTACT_IDX_INCLUDEMARGIN,
    CONTACT_IDX_FRICTION,
    CONTACT_IDX_FRICTION_SPIN,
    CONTACT_IDX_FRICTION_ROLL,
    CONTACT_IDX_CONDIM,
    CONTACT_IDX_FRAME_T1_X,
    CONTACT_IDX_FRAME_T1_Y,
    CONTACT_IDX_FRAME_T1_Z,
    GEOM_IDX_TYPE,
    GEOM_IDX_BODY,
    GEOM_IDX_POS_X,
    GEOM_IDX_POS_Y,
    GEOM_IDX_POS_Z,
    GEOM_IDX_QUAT_X,
    GEOM_IDX_QUAT_Y,
    GEOM_IDX_QUAT_Z,
    GEOM_IDX_QUAT_W,
    GEOM_IDX_RADIUS,
    GEOM_IDX_HALF_LENGTH,
    GEOM_IDX_HALF_X,
    GEOM_IDX_HALF_Y,
    GEOM_IDX_HALF_Z,
    GEOM_IDX_FRICTION,
    GEOM_IDX_CONTYPE,
    GEOM_IDX_CONAFFINITY,
    GEOM_IDX_CONDIM,
    GEOM_IDX_FRICTION_SPIN,
    GEOM_IDX_FRICTION_ROLL,
    GEOM_IDX_RBOUND,
    GEOM_IDX_MARGIN,
)
from .collision_primitives import (
    sphere_sphere,
    capsule_sphere,
    capsule_capsule,
    box_sphere,
    box_capsule,
    box_box,
    box_plane,
    cylinder_plane,
    cylinder_sphere,
    cylinder_capsule,
    cylinder_cylinder,
    cylinder_box,
)

comptime CD_TPB: Int = 64


@always_inline
def _geom_world_pos_fields[
    DTYPE: DType,
    NBODY: Int,
    NGEOM: Int,
    BATCH: Int,
](
    env: Int,
    g: Int,
    geoms: LayoutTensor[
        DTYPE, Layout.row_major(NGEOM, MODEL_GEOM_SIZE), MutAnyOrigin
    ],
    xpos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
    mut out_px: Scalar[DTYPE],
    mut out_py: Scalar[DTYPE],
    mut out_pz: Scalar[DTYPE],
    mut out_qx: Scalar[DTYPE],
    mut out_qy: Scalar[DTYPE],
    mut out_qz: Scalar[DTYPE],
    mut out_qw: Scalar[DTYPE],
):
    """Compute geom world pos/quat (verbatim from _geom_world_pos_gpu)."""
    var body_idx = Int(rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_BODY]))
    var lx = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_POS_X])
    var ly = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_POS_Y])
    var lz = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_POS_Z])
    var lqx = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_QUAT_X])
    var lqy = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_QUAT_Y])
    var lqz = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_QUAT_Z])
    var lqw = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_QUAT_W])
    if body_idx == 0:
        out_px = lx
        out_py = ly
        out_pz = lz
        out_qx = lqx
        out_qy = lqy
        out_qz = lqz
        out_qw = lqw
        return
    var bpx = rebind[Scalar[DTYPE]](xpos[env, body_idx * 3 + 0])
    var bpy = rebind[Scalar[DTYPE]](xpos[env, body_idx * 3 + 1])
    var bpz = rebind[Scalar[DTYPE]](xpos[env, body_idx * 3 + 2])
    var bqx = rebind[Scalar[DTYPE]](xquat[env, body_idx * 4 + 0])
    var bqy = rebind[Scalar[DTYPE]](xquat[env, body_idx * 4 + 1])
    var bqz = rebind[Scalar[DTYPE]](xquat[env, body_idx * 4 + 2])
    var bqw = rebind[Scalar[DTYPE]](xquat[env, body_idx * 4 + 3])
    if (
        lx == Scalar[DTYPE](0)
        and ly == Scalar[DTYPE](0)
        and lz == Scalar[DTYPE](0)
        and lqx == Scalar[DTYPE](0)
        and lqy == Scalar[DTYPE](0)
        and lqz == Scalar[DTYPE](0)
        and lqw == Scalar[DTYPE](1)
    ):
        out_px = bpx
        out_py = bpy
        out_pz = bpz
        out_qx = bqx
        out_qy = bqy
        out_qz = bqz
        out_qw = bqw
        return
    var rotated = gpu_quat_rotate(bqx, bqy, bqz, bqw, lx, ly, lz)
    out_px = bpx + rotated[0]
    out_py = bpy + rotated[1]
    out_pz = bpz + rotated[2]
    var wq = gpu_quat_mul(bqx, bqy, bqz, bqw, lqx, lqy, lqz, lqw)
    out_qx = wq[0]
    out_qy = wq[1]
    out_qz = wq[2]
    out_qw = wq[3]


@always_inline
def _detect_contacts_env_fields[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int,
    NEXCLUDE: Int,
    BATCH: Int,
](
    env: Int,
    xpos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
    geoms: LayoutTensor[
        DTYPE, Layout.row_major(NGEOM, MODEL_GEOM_SIZE), MutAnyOrigin
    ],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    mmeta: LayoutTensor[
        DTYPE, Layout.row_major(MODEL_META_SIZE), MutAnyOrigin
    ],
    excludes: LayoutTensor[
        DTYPE, Layout.row_major(NEXCLUDE, 2), MutAnyOrigin
    ],
    contacts: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE),
        MutAnyOrigin,
    ],
    smeta: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, METADATA_SIZE), MutAnyOrigin
    ],
):
    """Unified contact detection for one env (verbatim from
    detect_contacts_gpu; mesh branches degrade to `continue`)."""
    var num_contacts = 0

    for gi in range(NGEOM):
        var gi_type = Int(
            rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_TYPE])
        )
        var gi_body = Int(
            rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_BODY])
        )
        var gi_contype = Int(
            rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_CONTYPE])
        )
        var gi_conaffinity = Int(
            rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_CONAFFINITY])
        )
        for gj in range(gi + 1, NGEOM):
            if num_contacts >= MAX_CONTACTS:
                smeta[env, META_IDX_NUM_CONTACTS] = Scalar[DTYPE](
                    num_contacts
                )
                return
            var gj_type = Int(
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_TYPE])
            )
            var gj_body = Int(
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_BODY])
            )
            if gi_type == GEOM_PLANE and gj_body == 0:
                continue
            if gj_type == GEOM_PLANE and gi_body == 0:
                continue
            # MuJoCo-style weld body filtering (GPU)
            var weld_i = Int(
                rebind[Scalar[DTYPE]](bodies[gi_body, BODY_IDX_WELDID])
            )
            var weld_j = Int(
                rebind[Scalar[DTYPE]](bodies[gj_body, BODY_IDX_WELDID])
            )
            # Same weld body → filter
            if weld_i == weld_j:
                continue
            # Weld parent check
            if weld_i != 0 and weld_j != 0:
                var wp_i = Int(
                    rebind[Scalar[DTYPE]](bodies[weld_i, BODY_IDX_PARENT])
                )
                var wp_j = Int(
                    rebind[Scalar[DTYPE]](bodies[weld_j, BODY_IDX_PARENT])
                )
                var weld_parent_i = Int(
                    rebind[Scalar[DTYPE]](bodies[wp_i, BODY_IDX_WELDID])
                )
                var weld_parent_j = Int(
                    rebind[Scalar[DTYPE]](bodies[wp_j, BODY_IDX_WELDID])
                )
                if weld_i == weld_parent_j or weld_j == weld_parent_i:
                    continue
                # Check contact exclusion pairs
                var n_ex = Int(
                    rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NEXCLUDE])
                )
                if n_ex > 0:
                    var ba = gi_body if gi_body <= gj_body else gj_body
                    var bb = gj_body if gi_body <= gj_body else gi_body
                    var excluded = False
                    for ex in range(n_ex):
                        var eb1 = Int(
                            rebind[Scalar[DTYPE]](excludes[ex, 0])
                        )
                        var eb2 = Int(
                            rebind[Scalar[DTYPE]](excludes[ex, 1])
                        )
                        if eb1 == ba and eb2 == bb:
                            excluded = True
                            break
                    if excluded:
                        continue
            var gj_contype = Int(
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_CONTYPE])
            )
            var gj_conaffinity = Int(
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_CONAFFINITY])
            )
            if (gi_contype & gj_conaffinity) == 0 and (
                gj_contype & gi_conaffinity
            ) == 0:
                continue

            var pi_x: Scalar[DTYPE] = 0
            var pi_y: Scalar[DTYPE] = 0
            var pi_z: Scalar[DTYPE] = 0
            var qi_x: Scalar[DTYPE] = 0
            var qi_y: Scalar[DTYPE] = 0
            var qi_z: Scalar[DTYPE] = 0
            var qi_w: Scalar[DTYPE] = 1
            _geom_world_pos_fields[DTYPE, NBODY, NGEOM, BATCH](
                env,
                gi,
                geoms,
                xpos,
                xquat,
                pi_x,
                pi_y,
                pi_z,
                qi_x,
                qi_y,
                qi_z,
                qi_w,
            )
            var pj_x: Scalar[DTYPE] = 0
            var pj_y: Scalar[DTYPE] = 0
            var pj_z: Scalar[DTYPE] = 0
            var qj_x: Scalar[DTYPE] = 0
            var qj_y: Scalar[DTYPE] = 0
            var qj_z: Scalar[DTYPE] = 0
            var qj_w: Scalar[DTYPE] = 1
            _geom_world_pos_fields[DTYPE, NBODY, NGEOM, BATCH](
                env,
                gj,
                geoms,
                xpos,
                xquat,
                pj_x,
                pj_y,
                pj_z,
                qj_x,
                qj_y,
                qj_z,
                qj_w,
            )

            # Broadphase bounding sphere check (skip for plane geoms — they're infinite)
            if gi_type != GEOM_PLANE and gj_type != GEOM_PLANE:
                var dx = pi_x - pj_x
                var dy = pi_y - pj_y
                var dz = pi_z - pj_z
                var dist_sq = dx * dx + dy * dy + dz * dz
                var ri_bound = rebind[Scalar[DTYPE]](
                    geoms[gi, GEOM_IDX_RBOUND]
                )
                var rj_bound = rebind[Scalar[DTYPE]](
                    geoms[gj, GEOM_IDX_RBOUND]
                )
                var bound = ri_bound + rj_bound
                if dist_sq > bound * bound:
                    continue

            var ri = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_RADIUS])
            var rj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_RADIUS])
            var hli = rebind[Scalar[DTYPE]](
                geoms[gi, GEOM_IDX_HALF_LENGTH]
            )
            var hlj = rebind[Scalar[DTYPE]](
                geoms[gj, GEOM_IDX_HALF_LENGTH]
            )
            var hxi = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_HALF_X])
            var hyi = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_HALF_Y])
            var hzi = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_HALF_Z])
            var hxj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_HALF_X])
            var hyj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_HALF_Y])
            var hzj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_HALF_Z])
            # Friction combination: max per element (MuJoCo convention)
            var fi = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_FRICTION])
            var fj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_FRICTION])
            var contact_friction = fi
            if fj > fi:
                contact_friction = fj
            var fsi = rebind[Scalar[DTYPE]](
                geoms[gi, GEOM_IDX_FRICTION_SPIN]
            )
            var fsj = rebind[Scalar[DTYPE]](
                geoms[gj, GEOM_IDX_FRICTION_SPIN]
            )
            var contact_friction_spin = fsi
            if fsj > fsi:
                contact_friction_spin = fsj
            var fri = rebind[Scalar[DTYPE]](
                geoms[gi, GEOM_IDX_FRICTION_ROLL]
            )
            var frj = rebind[Scalar[DTYPE]](
                geoms[gj, GEOM_IDX_FRICTION_ROLL]
            )
            var contact_friction_roll = fri
            if frj > fri:
                contact_friction_roll = frj
            var ci = Int(
                rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_CONDIM])
            )
            var cj = Int(
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_CONDIM])
            )
            var contact_condim = ci
            if cj > ci:
                contact_condim = cj

            # Margin combination: max of both geoms (MuJoCo convention)
            var margin_gi = rebind[Scalar[DTYPE]](
                geoms[gi, GEOM_IDX_MARGIN]
            )
            var margin_gj = rebind[Scalar[DTYPE]](
                geoms[gj, GEOM_IDX_MARGIN]
            )
            # MuJoCo 3.5+ convention: margin = sum of both geoms
            var contact_margin = margin_gi + margin_gj

            # --- Plane handling ---
            if gi_type == GEOM_PLANE:
                var ground_z = pi_z
                if gj_type == GEOM_CAPSULE:
                    # MuJoCo mjc_PlaneCapsule: test BOTH endpoints, up to 2 contacts
                    var axis_w = gpu_quat_rotate(
                        qj_x,
                        qj_y,
                        qj_z,
                        qj_w,
                        Scalar[DTYPE](0),
                        Scalar[DTYPE](0),
                        Scalar[DTYPE](1),
                    )
                    # Endpoint 1: center + half_length * axis
                    var e1_x = pj_x + hlj * axis_w[0]
                    var e1_y = pj_y + hlj * axis_w[1]
                    var e1_z = pj_z + hlj * axis_w[2]
                    var dist1 = e1_z - rj - ground_z
                    if dist1 < contact_margin and num_contacts < MAX_CONTACTS:
                        var c_off = num_contacts * CONTACT_SIZE
                        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[
                            DTYPE
                        ](gj_body)
                        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[
                            DTYPE
                        ](0)
                        contacts[env, c_off + CONTACT_IDX_POS_X] = e1_x
                        contacts[env, c_off + CONTACT_IDX_POS_Y] = e1_y
                        contacts[
                            env, c_off + CONTACT_IDX_POS_Z
                        ] = ground_z + dist1 * Scalar[DTYPE](0.5)
                        contacts[env, c_off + CONTACT_IDX_NX] = Scalar[DTYPE](
                            0
                        )
                        contacts[env, c_off + CONTACT_IDX_NY] = Scalar[DTYPE](
                            0
                        )
                        contacts[env, c_off + CONTACT_IDX_NZ] = Scalar[DTYPE](
                            1
                        )
                        contacts[env, c_off + CONTACT_IDX_DIST] = dist1
                        contacts[
                            env, c_off + CONTACT_IDX_INCLUDEMARGIN
                        ] = contact_margin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[
                            DTYPE
                        ](contact_condim)
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_X] = axis_w[
                            0
                        ]
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_Y] = axis_w[
                            1
                        ]
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_Z] = axis_w[
                            2
                        ]
                        num_contacts += 1
                    # Endpoint 2: center - half_length * axis
                    var e2_x = pj_x - hlj * axis_w[0]
                    var e2_y = pj_y - hlj * axis_w[1]
                    var e2_z = pj_z - hlj * axis_w[2]
                    var dist2 = e2_z - rj - ground_z
                    if dist2 < contact_margin and num_contacts < MAX_CONTACTS:
                        var c_off = num_contacts * CONTACT_SIZE
                        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[
                            DTYPE
                        ](gj_body)
                        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[
                            DTYPE
                        ](0)
                        contacts[env, c_off + CONTACT_IDX_POS_X] = e2_x
                        contacts[env, c_off + CONTACT_IDX_POS_Y] = e2_y
                        contacts[
                            env, c_off + CONTACT_IDX_POS_Z
                        ] = ground_z + dist2 * Scalar[DTYPE](0.5)
                        contacts[env, c_off + CONTACT_IDX_NX] = Scalar[DTYPE](
                            0
                        )
                        contacts[env, c_off + CONTACT_IDX_NY] = Scalar[DTYPE](
                            0
                        )
                        contacts[env, c_off + CONTACT_IDX_NZ] = Scalar[DTYPE](
                            1
                        )
                        contacts[env, c_off + CONTACT_IDX_DIST] = dist2
                        contacts[
                            env, c_off + CONTACT_IDX_INCLUDEMARGIN
                        ] = contact_margin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[
                            DTYPE
                        ](contact_condim)
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_X] = axis_w[
                            0
                        ]
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_Y] = axis_w[
                            1
                        ]
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_Z] = axis_w[
                            2
                        ]
                        num_contacts += 1
                elif gj_type == GEOM_CYLINDER:
                    # Cylinder-plane: single contact at lowest rim point
                    var cp = cylinder_plane[DTYPE](
                        pj_x,
                        pj_y,
                        pj_z,
                        qj_x,
                        qj_y,
                        qj_z,
                        qj_w,
                        hlj,
                        rj,
                        ground_z,
                    )
                    var dist = cp[0]
                    if dist < contact_margin and num_contacts < MAX_CONTACTS:
                        var c_off = num_contacts * CONTACT_SIZE
                        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[
                            DTYPE
                        ](gj_body)
                        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[
                            DTYPE
                        ](0)
                        contacts[env, c_off + CONTACT_IDX_POS_X] = cp[1]
                        contacts[env, c_off + CONTACT_IDX_POS_Y] = cp[2]
                        contacts[env, c_off + CONTACT_IDX_POS_Z] = cp[3]
                        contacts[env, c_off + CONTACT_IDX_NX] = Scalar[DTYPE](
                            0
                        )
                        contacts[env, c_off + CONTACT_IDX_NY] = Scalar[DTYPE](
                            0
                        )
                        contacts[env, c_off + CONTACT_IDX_NZ] = Scalar[DTYPE](
                            1
                        )
                        contacts[env, c_off + CONTACT_IDX_DIST] = dist
                        contacts[
                            env, c_off + CONTACT_IDX_INCLUDEMARGIN
                        ] = contact_margin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[
                            DTYPE
                        ](contact_condim)
                        num_contacts += 1
                elif gj_type == GEOM_SPHERE:
                    var dist = pj_z - rj - ground_z
                    if dist < contact_margin and num_contacts < MAX_CONTACTS:
                        var c_off = num_contacts * CONTACT_SIZE
                        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[
                            DTYPE
                        ](gj_body)
                        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[
                            DTYPE
                        ](0)
                        contacts[env, c_off + CONTACT_IDX_POS_X] = pj_x
                        contacts[env, c_off + CONTACT_IDX_POS_Y] = pj_y
                        contacts[
                            env, c_off + CONTACT_IDX_POS_Z
                        ] = ground_z + dist * Scalar[DTYPE](0.5)
                        contacts[env, c_off + CONTACT_IDX_NX] = Scalar[DTYPE](
                            0
                        )
                        contacts[env, c_off + CONTACT_IDX_NY] = Scalar[DTYPE](
                            0
                        )
                        contacts[env, c_off + CONTACT_IDX_NZ] = Scalar[DTYPE](
                            1
                        )
                        contacts[env, c_off + CONTACT_IDX_DIST] = dist
                        contacts[
                            env, c_off + CONTACT_IDX_INCLUDEMARGIN
                        ] = contact_margin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[
                            DTYPE
                        ](contact_condim)
                        num_contacts += 1
                elif gj_type == GEOM_BOX:
                    var bp = box_plane[DTYPE](
                        pj_x, pj_y, pj_z,
                        qj_x, qj_y, qj_z, qj_w,
                        hxj, hyj, hzj,
                        ground_z,
                    )
                    var dist = bp[0]
                    if dist < contact_margin and num_contacts < MAX_CONTACTS:
                        var c_off = num_contacts * CONTACT_SIZE
                        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[
                            DTYPE
                        ](gj_body)
                        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[
                            DTYPE
                        ](0)
                        contacts[env, c_off + CONTACT_IDX_POS_X] = bp[1]
                        contacts[env, c_off + CONTACT_IDX_POS_Y] = bp[2]
                        contacts[env, c_off + CONTACT_IDX_POS_Z] = bp[3]
                        contacts[env, c_off + CONTACT_IDX_NX] = Scalar[DTYPE](
                            0
                        )
                        contacts[env, c_off + CONTACT_IDX_NY] = Scalar[DTYPE](
                            0
                        )
                        contacts[env, c_off + CONTACT_IDX_NZ] = Scalar[DTYPE](
                            1
                        )
                        contacts[env, c_off + CONTACT_IDX_DIST] = dist
                        contacts[
                            env, c_off + CONTACT_IDX_INCLUDEMARGIN
                        ] = contact_margin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[
                            DTYPE
                        ](contact_condim)
                        num_contacts += 1
                elif gj_type == GEOM_MESH:
                    continue  # mesh collision not ported (guarded at dispatcher)
                continue

            if gj_type == GEOM_PLANE:
                var ground_z = pj_z
                if gi_type == GEOM_CAPSULE:
                    # MuJoCo mjc_PlaneCapsule: test BOTH endpoints, up to 2 contacts
                    var axis_w = gpu_quat_rotate(
                        qi_x,
                        qi_y,
                        qi_z,
                        qi_w,
                        Scalar[DTYPE](0),
                        Scalar[DTYPE](0),
                        Scalar[DTYPE](1),
                    )
                    # Endpoint 1: center + half_length * axis
                    var e1_x = pi_x + hli * axis_w[0]
                    var e1_y = pi_y + hli * axis_w[1]
                    var e1_z = pi_z + hli * axis_w[2]
                    var dist1 = e1_z - ri - ground_z
                    if dist1 < contact_margin and num_contacts < MAX_CONTACTS:
                        var c_off = num_contacts * CONTACT_SIZE
                        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[
                            DTYPE
                        ](gi_body)
                        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[
                            DTYPE
                        ](0)
                        contacts[env, c_off + CONTACT_IDX_POS_X] = e1_x
                        contacts[env, c_off + CONTACT_IDX_POS_Y] = e1_y
                        contacts[
                            env, c_off + CONTACT_IDX_POS_Z
                        ] = ground_z + dist1 * Scalar[DTYPE](0.5)
                        contacts[env, c_off + CONTACT_IDX_NX] = Scalar[DTYPE](
                            0
                        )
                        contacts[env, c_off + CONTACT_IDX_NY] = Scalar[DTYPE](
                            0
                        )
                        contacts[env, c_off + CONTACT_IDX_NZ] = Scalar[DTYPE](
                            1
                        )
                        contacts[env, c_off + CONTACT_IDX_DIST] = dist1
                        contacts[
                            env, c_off + CONTACT_IDX_INCLUDEMARGIN
                        ] = contact_margin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[
                            DTYPE
                        ](contact_condim)
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_X] = axis_w[
                            0
                        ]
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_Y] = axis_w[
                            1
                        ]
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_Z] = axis_w[
                            2
                        ]
                        num_contacts += 1
                    # Endpoint 2: center - half_length * axis
                    var e2_x = pi_x - hli * axis_w[0]
                    var e2_y = pi_y - hli * axis_w[1]
                    var e2_z = pi_z - hli * axis_w[2]
                    var dist2 = e2_z - ri - ground_z
                    if dist2 < contact_margin and num_contacts < MAX_CONTACTS:
                        var c_off = num_contacts * CONTACT_SIZE
                        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[
                            DTYPE
                        ](gi_body)
                        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[
                            DTYPE
                        ](0)
                        contacts[env, c_off + CONTACT_IDX_POS_X] = e2_x
                        contacts[env, c_off + CONTACT_IDX_POS_Y] = e2_y
                        contacts[
                            env, c_off + CONTACT_IDX_POS_Z
                        ] = ground_z + dist2 * Scalar[DTYPE](0.5)
                        contacts[env, c_off + CONTACT_IDX_NX] = Scalar[DTYPE](
                            0
                        )
                        contacts[env, c_off + CONTACT_IDX_NY] = Scalar[DTYPE](
                            0
                        )
                        contacts[env, c_off + CONTACT_IDX_NZ] = Scalar[DTYPE](
                            1
                        )
                        contacts[env, c_off + CONTACT_IDX_DIST] = dist2
                        contacts[
                            env, c_off + CONTACT_IDX_INCLUDEMARGIN
                        ] = contact_margin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[
                            DTYPE
                        ](contact_condim)
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_X] = axis_w[
                            0
                        ]
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_Y] = axis_w[
                            1
                        ]
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_Z] = axis_w[
                            2
                        ]
                        num_contacts += 1
                elif gi_type == GEOM_CYLINDER:
                    # Cylinder-plane: single contact at lowest rim point
                    var cp = cylinder_plane[DTYPE](
                        pi_x,
                        pi_y,
                        pi_z,
                        qi_x,
                        qi_y,
                        qi_z,
                        qi_w,
                        hli,
                        ri,
                        ground_z,
                    )
                    var dist = cp[0]
                    if dist < contact_margin and num_contacts < MAX_CONTACTS:
                        var c_off = num_contacts * CONTACT_SIZE
                        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[
                            DTYPE
                        ](gi_body)
                        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[
                            DTYPE
                        ](0)
                        contacts[env, c_off + CONTACT_IDX_POS_X] = cp[1]
                        contacts[env, c_off + CONTACT_IDX_POS_Y] = cp[2]
                        contacts[env, c_off + CONTACT_IDX_POS_Z] = cp[3]
                        contacts[env, c_off + CONTACT_IDX_NX] = Scalar[DTYPE](
                            0
                        )
                        contacts[env, c_off + CONTACT_IDX_NY] = Scalar[DTYPE](
                            0
                        )
                        contacts[env, c_off + CONTACT_IDX_NZ] = Scalar[DTYPE](
                            1
                        )
                        contacts[env, c_off + CONTACT_IDX_DIST] = dist
                        contacts[
                            env, c_off + CONTACT_IDX_INCLUDEMARGIN
                        ] = contact_margin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[
                            DTYPE
                        ](contact_condim)
                        num_contacts += 1
                elif gi_type == GEOM_SPHERE:
                    var dist = pi_z - ri - ground_z
                    if dist < contact_margin and num_contacts < MAX_CONTACTS:
                        var c_off = num_contacts * CONTACT_SIZE
                        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[
                            DTYPE
                        ](gi_body)
                        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[
                            DTYPE
                        ](0)
                        contacts[env, c_off + CONTACT_IDX_POS_X] = pi_x
                        contacts[env, c_off + CONTACT_IDX_POS_Y] = pi_y
                        contacts[
                            env, c_off + CONTACT_IDX_POS_Z
                        ] = ground_z + dist * Scalar[DTYPE](0.5)
                        contacts[env, c_off + CONTACT_IDX_NX] = Scalar[DTYPE](
                            0
                        )
                        contacts[env, c_off + CONTACT_IDX_NY] = Scalar[DTYPE](
                            0
                        )
                        contacts[env, c_off + CONTACT_IDX_NZ] = Scalar[DTYPE](
                            1
                        )
                        contacts[env, c_off + CONTACT_IDX_DIST] = dist
                        contacts[
                            env, c_off + CONTACT_IDX_INCLUDEMARGIN
                        ] = contact_margin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[
                            DTYPE
                        ](contact_condim)
                        num_contacts += 1
                elif gi_type == GEOM_BOX:
                    var bp = box_plane[DTYPE](
                        pi_x, pi_y, pi_z,
                        qi_x, qi_y, qi_z, qi_w,
                        hxi, hyi, hzi,
                        ground_z,
                    )
                    var dist = bp[0]
                    if dist < contact_margin and num_contacts < MAX_CONTACTS:
                        var c_off = num_contacts * CONTACT_SIZE
                        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[
                            DTYPE
                        ](gi_body)
                        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[
                            DTYPE
                        ](0)
                        contacts[env, c_off + CONTACT_IDX_POS_X] = bp[1]
                        contacts[env, c_off + CONTACT_IDX_POS_Y] = bp[2]
                        contacts[env, c_off + CONTACT_IDX_POS_Z] = bp[3]
                        contacts[env, c_off + CONTACT_IDX_NX] = Scalar[DTYPE](
                            0
                        )
                        contacts[env, c_off + CONTACT_IDX_NY] = Scalar[DTYPE](
                            0
                        )
                        contacts[env, c_off + CONTACT_IDX_NZ] = Scalar[DTYPE](
                            1
                        )
                        contacts[env, c_off + CONTACT_IDX_DIST] = dist
                        contacts[
                            env, c_off + CONTACT_IDX_INCLUDEMARGIN
                        ] = contact_margin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[
                            DTYPE
                        ](contact_condim)
                        num_contacts += 1
                elif gi_type == GEOM_MESH:
                    continue  # mesh collision not ported (guarded at dispatcher)
                continue

            # --- Non-plane geom pair ---
            var dist: Scalar[DTYPE] = 1.0
            var cx: Scalar[DTYPE] = 0
            var cy: Scalar[DTYPE] = 0
            var cz: Scalar[DTYPE] = 0
            var nx: Scalar[DTYPE] = 0
            var ny: Scalar[DTYPE] = 0
            var nz: Scalar[DTYPE] = 1
            var body_a = gi_body
            var body_b = gj_body

            if gi_type == GEOM_SPHERE and gj_type == GEOM_SPHERE:
                var r = sphere_sphere[DTYPE](
                    pi_x, pi_y, pi_z, ri, pj_x, pj_y, pj_z, rj
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = r[4]
                ny = r[5]
                nz = r[6]
            elif gi_type == GEOM_CAPSULE and gj_type == GEOM_SPHERE:
                var r = capsule_sphere[DTYPE](
                    pi_x,
                    pi_y,
                    pi_z,
                    qi_x,
                    qi_y,
                    qi_z,
                    qi_w,
                    hli,
                    ri,
                    pj_x,
                    pj_y,
                    pj_z,
                    rj,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = r[4]
                ny = r[5]
                nz = r[6]
            elif gi_type == GEOM_SPHERE and gj_type == GEOM_CAPSULE:
                var r = capsule_sphere[DTYPE](
                    pj_x,
                    pj_y,
                    pj_z,
                    qj_x,
                    qj_y,
                    qj_z,
                    qj_w,
                    hlj,
                    rj,
                    pi_x,
                    pi_y,
                    pi_z,
                    ri,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = -r[4]
                ny = -r[5]
                nz = -r[6]
                body_a = gj_body
                body_b = gi_body
            elif gi_type == GEOM_CAPSULE and gj_type == GEOM_CAPSULE:
                var r = capsule_capsule[DTYPE](
                    pi_x,
                    pi_y,
                    pi_z,
                    qi_x,
                    qi_y,
                    qi_z,
                    qi_w,
                    hli,
                    ri,
                    pj_x,
                    pj_y,
                    pj_z,
                    qj_x,
                    qj_y,
                    qj_z,
                    qj_w,
                    hlj,
                    rj,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = r[4]
                ny = r[5]
                nz = r[6]
            elif gi_type == GEOM_BOX and gj_type == GEOM_SPHERE:
                var r = box_sphere[DTYPE](
                    pi_x,
                    pi_y,
                    pi_z,
                    qi_x,
                    qi_y,
                    qi_z,
                    qi_w,
                    hxi,
                    hyi,
                    hzi,
                    pj_x,
                    pj_y,
                    pj_z,
                    rj,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = r[4]
                ny = r[5]
                nz = r[6]
            elif gi_type == GEOM_SPHERE and gj_type == GEOM_BOX:
                var r = box_sphere[DTYPE](
                    pj_x,
                    pj_y,
                    pj_z,
                    qj_x,
                    qj_y,
                    qj_z,
                    qj_w,
                    hxj,
                    hyj,
                    hzj,
                    pi_x,
                    pi_y,
                    pi_z,
                    ri,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = -r[4]
                ny = -r[5]
                nz = -r[6]
                body_a = gj_body
                body_b = gi_body
            elif gi_type == GEOM_BOX and gj_type == GEOM_CAPSULE:
                var r = box_capsule[DTYPE](
                    pi_x,
                    pi_y,
                    pi_z,
                    qi_x,
                    qi_y,
                    qi_z,
                    qi_w,
                    hxi,
                    hyi,
                    hzi,
                    pj_x,
                    pj_y,
                    pj_z,
                    qj_x,
                    qj_y,
                    qj_z,
                    qj_w,
                    hlj,
                    rj,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = r[4]
                ny = r[5]
                nz = r[6]
            elif gi_type == GEOM_CAPSULE and gj_type == GEOM_BOX:
                var r = box_capsule[DTYPE](
                    pj_x,
                    pj_y,
                    pj_z,
                    qj_x,
                    qj_y,
                    qj_z,
                    qj_w,
                    hxj,
                    hyj,
                    hzj,
                    pi_x,
                    pi_y,
                    pi_z,
                    qi_x,
                    qi_y,
                    qi_z,
                    qi_w,
                    hli,
                    ri,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = -r[4]
                ny = -r[5]
                nz = -r[6]
                body_a = gj_body
                body_b = gi_body
            elif gi_type == GEOM_BOX and gj_type == GEOM_BOX:
                var r = box_box[DTYPE](
                    pi_x,
                    pi_y,
                    pi_z,
                    qi_x,
                    qi_y,
                    qi_z,
                    qi_w,
                    hxi,
                    hyi,
                    hzi,
                    pj_x,
                    pj_y,
                    pj_z,
                    qj_x,
                    qj_y,
                    qj_z,
                    qj_w,
                    hxj,
                    hyj,
                    hzj,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = r[4]
                ny = r[5]
                nz = r[6]
            elif gi_type == GEOM_CYLINDER and gj_type == GEOM_SPHERE:
                var r = cylinder_sphere[DTYPE](
                    pi_x,
                    pi_y,
                    pi_z,
                    qi_x,
                    qi_y,
                    qi_z,
                    qi_w,
                    hli,
                    ri,
                    pj_x,
                    pj_y,
                    pj_z,
                    rj,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = r[4]
                ny = r[5]
                nz = r[6]
            elif gi_type == GEOM_SPHERE and gj_type == GEOM_CYLINDER:
                var r = cylinder_sphere[DTYPE](
                    pj_x,
                    pj_y,
                    pj_z,
                    qj_x,
                    qj_y,
                    qj_z,
                    qj_w,
                    hlj,
                    rj,
                    pi_x,
                    pi_y,
                    pi_z,
                    ri,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = -r[4]
                ny = -r[5]
                nz = -r[6]
                body_a = gj_body
                body_b = gi_body
            elif gi_type == GEOM_CYLINDER and gj_type == GEOM_CAPSULE:
                var r = cylinder_capsule[DTYPE](
                    pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w, hli, ri,
                    pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w, hlj, rj,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = r[4]
                ny = r[5]
                nz = r[6]
            elif gi_type == GEOM_CAPSULE and gj_type == GEOM_CYLINDER:
                var r = cylinder_capsule[DTYPE](
                    pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w, hlj, rj,
                    pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w, hli, ri,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = -r[4]
                ny = -r[5]
                nz = -r[6]
                body_a = gj_body
                body_b = gi_body
            elif gi_type == GEOM_CYLINDER and gj_type == GEOM_CYLINDER:
                var r = cylinder_cylinder[DTYPE](
                    pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w, hli, ri,
                    pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w, hlj, rj,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = r[4]
                ny = r[5]
                nz = r[6]
            elif gi_type == GEOM_CYLINDER and gj_type == GEOM_BOX:
                var r = cylinder_box[DTYPE](
                    pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w, hli, ri,
                    pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w, hxj, hyj, hzj,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = r[4]
                ny = r[5]
                nz = r[6]
            elif gi_type == GEOM_BOX and gj_type == GEOM_CYLINDER:
                var r = cylinder_box[DTYPE](
                    pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w, hlj, rj,
                    pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w, hxi, hyi, hzi,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = -r[4]
                ny = -r[5]
                nz = -r[6]
                body_a = gj_body
                body_b = gi_body

            # GJK/EPA fallback for any pair involving a mesh geom
            elif gi_type == GEOM_MESH or gj_type == GEOM_MESH:
                continue  # mesh collision not ported (guarded at dispatcher)

            if dist < contact_margin and num_contacts < MAX_CONTACTS:
                var c_off = num_contacts * CONTACT_SIZE
                contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](
                    body_a
                )
                contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](
                    body_b
                )
                contacts[env, c_off + CONTACT_IDX_POS_X] = cx
                contacts[env, c_off + CONTACT_IDX_POS_Y] = cy
                contacts[env, c_off + CONTACT_IDX_POS_Z] = cz
                # Negate normal for body-body contacts (same fix as CPU path)
                if body_b > 0:
                    nx = -nx
                    ny = -ny
                    nz = -nz
                contacts[env, c_off + CONTACT_IDX_NX] = nx
                contacts[env, c_off + CONTACT_IDX_NY] = ny
                contacts[env, c_off + CONTACT_IDX_NZ] = nz
                contacts[env, c_off + CONTACT_IDX_DIST] = dist
                contacts[
                    env, c_off + CONTACT_IDX_INCLUDEMARGIN
                ] = contact_margin
                contacts[env, c_off + CONTACT_IDX_FRICTION] = contact_friction
                contacts[
                    env, c_off + CONTACT_IDX_FRICTION_SPIN
                ] = contact_friction_spin
                contacts[
                    env, c_off + CONTACT_IDX_FRICTION_ROLL
                ] = contact_friction_roll
                contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
                    contact_condim
                )
                num_contacts += 1

    smeta[env, META_IDX_NUM_CONTACTS] = Scalar[DTYPE](num_contacts)


def _detect_contacts_fields_kernel[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int,
    NEXCLUDE: Int,
    BATCH: Int,
](
    xpos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
    geoms: LayoutTensor[
        DTYPE, Layout.row_major(NGEOM, MODEL_GEOM_SIZE), MutAnyOrigin
    ],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    mmeta: LayoutTensor[
        DTYPE, Layout.row_major(MODEL_META_SIZE), MutAnyOrigin
    ],
    excludes: LayoutTensor[
        DTYPE, Layout.row_major(NEXCLUDE, 2), MutAnyOrigin
    ],
    contacts: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE),
        MutAnyOrigin,
    ],
    smeta: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, METADATA_SIZE), MutAnyOrigin
    ],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _detect_contacts_env_fields[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, NEXCLUDE, BATCH
    ](env, xpos, xquat, geoms, bodies, mmeta, excludes, contacts, smeta)


def detect_contacts_fields[
    target: StaticString,
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int = 0,
    NEQUALITY: Int = 0,
    NTENDON: Int = 0,
    NSITE: Int = 0,
    NEXCLUDE: Int = 0,
    NMESH_VERTS: Int = 0,
    BATCH: Int = 1,
](
    mut d: DataFields[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, BATCH],
    mut m: ModelFields[
        DTYPE,
        NV,
        NBODY,
        NJOINT,
        NGEOM,
        NEQUALITY,
        NTENDON,
        NSITE,
        NEXCLUDE,
        NMESH_VERTS,
    ],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Unified geom contact detection from FK products, both targets, one
    body. Reads `d.xpos`/`d.xquat` + geom/body/meta/exclude records; writes
    `d.contacts` + the ncon slot of `d.meta`."""
    comptime assert NMESH_VERTS == 0, "mesh collision not ported to fields yet"

    comptime L_B3 = Layout.row_major(BATCH, NBODY * 3)
    comptime L_B4 = Layout.row_major(BATCH, NBODY * 4)
    comptime L_GEOM = Layout.row_major(NGEOM, MODEL_GEOM_SIZE)
    comptime L_BODY = Layout.row_major(NBODY, MODEL_BODY_SIZE)
    comptime L_MMETA = Layout.row_major(MODEL_META_SIZE)
    comptime L_EXCLUDE = Layout.row_major(NEXCLUDE, 2)
    comptime L_CONTACTS = Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE)
    comptime L_SMETA = Layout.row_major(BATCH, METADATA_SIZE)

    comptime if target == "cpu":
        var xpos_v = d.xpos.lt["cpu", L_B3]()
        var xquat_v = d.xquat.lt["cpu", L_B4]()
        var geoms_v = m.geoms.lt["cpu", L_GEOM]()
        var bodies_v = m.bodies.lt["cpu", L_BODY]()
        var mmeta_v = m.meta.lt["cpu", L_MMETA]()
        var excludes_v = m.excludes.lt["cpu", L_EXCLUDE]()
        var contacts_v = d.contacts.lt["cpu", L_CONTACTS]()
        var smeta_v = d.meta.lt["cpu", L_SMETA]()
        for e in range(BATCH):
            _detect_contacts_env_fields[
                DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
                NEXCLUDE, BATCH,
            ](
                e, xpos_v, xquat_v, geoms_v, bodies_v, mmeta_v,
                excludes_v, contacts_v, smeta_v,
            )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + CD_TPB - 1) // CD_TPB
        c.enqueue_function[
            _detect_contacts_fields_kernel[
                DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
                NEXCLUDE, BATCH,
            ]
        ](
            d.xpos.lt["gpu", L_B3](),
            d.xquat.lt["gpu", L_B4](),
            m.geoms.lt["gpu", L_GEOM](),
            m.bodies.lt["gpu", L_BODY](),
            m.meta.lt["gpu", L_MMETA](),
            m.excludes.lt["gpu", L_EXCLUDE](),
            d.contacts.lt["gpu", L_CONTACTS](),
            d.meta.lt["gpu", L_SMETA](),
            grid_dim=(BLOCKS,),
            block_dim=(CD_TPB,),
        )
