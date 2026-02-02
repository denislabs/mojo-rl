"""Physics3D v2 multi-body collision detection.

Phase 3: Detects all contacts (sphere-plane + sphere-sphere) for multi-body systems.
Phase 8: Added capsule collision (capsule-plane, capsule-sphere, capsule-capsule).
Phase 9: Added box collision (box-plane, box-sphere, box-capsule, box-box).
Uses the pure collision primitives from collision_primitives.mojo.
"""

from .collision_primitives import (
    sphere_sphere,
    sphere_plane,
    capsule_plane,
    capsule_sphere,
    capsule_capsule,
    box_plane,
    box_sphere,
    box_capsule,
    box_box,
)
from ..types import Model, Data
from ..traits import CollisionSystem
from layout import LayoutTensor, Layout
from ..gpu.constants import (
    BODY_IDX_PX,
    BODY_IDX_PY,
    BODY_IDX_PZ,
    BODY_IDX_QX,
    BODY_IDX_QY,
    BODY_IDX_QZ,
    BODY_IDX_QW,
    body_offset,
    contact_offset,
    metadata_offset,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_POS_X,
    CONTACT_IDX_POS_Y,
    CONTACT_IDX_POS_Z,
    CONTACT_IDX_NX,
    CONTACT_IDX_NY,
    CONTACT_IDX_NZ,
    CONTACT_IDX_DIST,
    META_IDX_NUM_CONTACTS,
    MODEL_BODY_SIZE,
    MODEL_IDX_RADIUS,
    MODEL_IDX_GEOM_TYPE,
    MODEL_IDX_HALF_LENGTH,
    MODEL_IDX_HALF_X,
    MODEL_IDX_HALF_Y,
    MODEL_IDX_HALF_Z,
    GEOM_SPHERE,
    GEOM_CAPSULE,
    GEOM_BOX,
)


struct CollisionDetector(CollisionSystem):
    @always_inline
    @staticmethod
    fn detect_all_contacts[
        DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int, MAX_JOINTS: Int = 0
    ](
        model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS],
        mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS, MAX_JOINTS],
    ):
        """Detect all contacts (body-plane + body-body) with geometry dispatch.

        Phase 9: Supports sphere, capsule, and box geometries.
        O(N) for ground contacts, O(N²) for body-body.
        Fills the data.contacts buffer and sets data.num_contacts.

        Args:
            model: Static model configuration.
            data: Mutable simulation state.
        """
        data.num_contacts = 0

        # Phase 1: Body-plane contacts (each body vs ground)
        for i in range(NUM_BODIES):
            var px = data.positions[i * 3 + 0]
            var py = data.positions[i * 3 + 1]
            var pz = data.positions[i * 3 + 2]
            var radius = model.radii[i]
            var geom_type = model.geom_types[i]

            var dist: Scalar[DTYPE]
            var contact_x: Scalar[DTYPE]
            var contact_y: Scalar[DTYPE]
            var contact_z: Scalar[DTYPE]

            if geom_type == GEOM_BOX:
                # Box-plane collision
                var qx = data.quaternions[i * 4 + 0]
                var qy = data.quaternions[i * 4 + 1]
                var qz = data.quaternions[i * 4 + 2]
                var qw = data.quaternions[i * 4 + 3]

                var result = box_plane(
                    px, py, pz, qx, qy, qz, qw,
                    model.half_x[i], model.half_y[i], model.half_z[i],
                    model.ground_z
                )
                dist = result[0]
                contact_x = result[1]
                contact_y = result[2]
                contact_z = result[3]
            elif geom_type == GEOM_CAPSULE:
                # Capsule-plane collision
                var qx = data.quaternions[i * 4 + 0]
                var qy = data.quaternions[i * 4 + 1]
                var qz = data.quaternions[i * 4 + 2]
                var qw = data.quaternions[i * 4 + 3]
                var half_len = model.half_lengths[i]

                var result = capsule_plane(
                    px, py, pz, qx, qy, qz, qw, half_len, radius, model.ground_z
                )
                dist = result[0]
                contact_x = result[1]
                contact_y = result[2]
                contact_z = result[3]
            else:
                # Default: Sphere-plane collision
                var result = sphere_plane(px, py, pz, radius, model.ground_z)
                dist = result[0]
                contact_x = result[1]
                contact_y = result[2]
                contact_z = result[3]

            # Contact if penetrating (dist < 0)
            if dist < Scalar[DTYPE](0) and data.num_contacts < MAX_CONTACTS:
                data.contacts[data.num_contacts].set(
                    i,
                    -1,  # -1 indicates ground
                    contact_x,
                    contact_y,
                    contact_z,  # Contact position
                    Scalar[DTYPE](0),
                    Scalar[DTYPE](0),
                    Scalar[DTYPE](1),  # Normal up
                    dist,
                )
                data.num_contacts += 1

        # Phase 2: Body-body contacts (all pairs)
        for i in range(NUM_BODIES):
            for j in range(i + 1, NUM_BODIES):
                var px_i = data.positions[i * 3 + 0]
                var py_i = data.positions[i * 3 + 1]
                var pz_i = data.positions[i * 3 + 2]
                var px_j = data.positions[j * 3 + 0]
                var py_j = data.positions[j * 3 + 1]
                var pz_j = data.positions[j * 3 + 2]

                var geom_i = model.geom_types[i]
                var geom_j = model.geom_types[j]

                var dist: Scalar[DTYPE]
                var contact_x: Scalar[DTYPE]
                var contact_y: Scalar[DTYPE]
                var contact_z: Scalar[DTYPE]
                var nx: Scalar[DTYPE]
                var ny: Scalar[DTYPE]
                var nz: Scalar[DTYPE]

                # Get quaternions (needed for capsules and boxes)
                var qx_i = data.quaternions[i * 4 + 0]
                var qy_i = data.quaternions[i * 4 + 1]
                var qz_i = data.quaternions[i * 4 + 2]
                var qw_i = data.quaternions[i * 4 + 3]
                var qx_j = data.quaternions[j * 4 + 0]
                var qy_j = data.quaternions[j * 4 + 1]
                var qz_j = data.quaternions[j * 4 + 2]
                var qw_j = data.quaternions[j * 4 + 3]

                # Dispatch based on geometry types
                if geom_i == GEOM_SPHERE and geom_j == GEOM_SPHERE:
                    var result = sphere_sphere(
                        px_i, py_i, pz_i, model.radii[i],
                        px_j, py_j, pz_j, model.radii[j],
                    )
                    dist = result[0]
                    contact_x = result[1]
                    contact_y = result[2]
                    contact_z = result[3]
                    nx = result[4]
                    ny = result[5]
                    nz = result[6]
                elif geom_i == GEOM_CAPSULE and geom_j == GEOM_SPHERE:
                    var result = capsule_sphere(
                        px_i, py_i, pz_i,
                        qx_i, qy_i, qz_i, qw_i,
                        model.half_lengths[i], model.radii[i],
                        px_j, py_j, pz_j, model.radii[j],
                    )
                    dist = result[0]
                    contact_x = result[1]
                    contact_y = result[2]
                    contact_z = result[3]
                    nx = result[4]
                    ny = result[5]
                    nz = result[6]
                elif geom_i == GEOM_SPHERE and geom_j == GEOM_CAPSULE:
                    # Swap order: capsule first, then negate normal
                    var result = capsule_sphere(
                        px_j, py_j, pz_j,
                        qx_j, qy_j, qz_j, qw_j,
                        model.half_lengths[j], model.radii[j],
                        px_i, py_i, pz_i, model.radii[i],
                    )
                    dist = result[0]
                    contact_x = result[1]
                    contact_y = result[2]
                    contact_z = result[3]
                    nx = -result[4]
                    ny = -result[5]
                    nz = -result[6]
                elif geom_i == GEOM_CAPSULE and geom_j == GEOM_CAPSULE:
                    var result = capsule_capsule(
                        px_i, py_i, pz_i,
                        qx_i, qy_i, qz_i, qw_i,
                        model.half_lengths[i], model.radii[i],
                        px_j, py_j, pz_j,
                        qx_j, qy_j, qz_j, qw_j,
                        model.half_lengths[j], model.radii[j],
                    )
                    dist = result[0]
                    contact_x = result[1]
                    contact_y = result[2]
                    contact_z = result[3]
                    nx = result[4]
                    ny = result[5]
                    nz = result[6]
                # Box collision cases (Phase 9)
                elif geom_i == GEOM_BOX and geom_j == GEOM_SPHERE:
                    var result = box_sphere(
                        px_i, py_i, pz_i,
                        qx_i, qy_i, qz_i, qw_i,
                        model.half_x[i], model.half_y[i], model.half_z[i],
                        px_j, py_j, pz_j, model.radii[j],
                    )
                    dist = result[0]
                    contact_x = result[1]
                    contact_y = result[2]
                    contact_z = result[3]
                    nx = result[4]
                    ny = result[5]
                    nz = result[6]
                elif geom_i == GEOM_SPHERE and geom_j == GEOM_BOX:
                    # Swap order: box first, then negate normal
                    var result = box_sphere(
                        px_j, py_j, pz_j,
                        qx_j, qy_j, qz_j, qw_j,
                        model.half_x[j], model.half_y[j], model.half_z[j],
                        px_i, py_i, pz_i, model.radii[i],
                    )
                    dist = result[0]
                    contact_x = result[1]
                    contact_y = result[2]
                    contact_z = result[3]
                    nx = -result[4]
                    ny = -result[5]
                    nz = -result[6]
                elif geom_i == GEOM_BOX and geom_j == GEOM_CAPSULE:
                    var result = box_capsule(
                        px_i, py_i, pz_i,
                        qx_i, qy_i, qz_i, qw_i,
                        model.half_x[i], model.half_y[i], model.half_z[i],
                        px_j, py_j, pz_j,
                        qx_j, qy_j, qz_j, qw_j,
                        model.half_lengths[j], model.radii[j],
                    )
                    dist = result[0]
                    contact_x = result[1]
                    contact_y = result[2]
                    contact_z = result[3]
                    nx = result[4]
                    ny = result[5]
                    nz = result[6]
                elif geom_i == GEOM_CAPSULE and geom_j == GEOM_BOX:
                    # Swap order: box first, then negate normal
                    var result = box_capsule(
                        px_j, py_j, pz_j,
                        qx_j, qy_j, qz_j, qw_j,
                        model.half_x[j], model.half_y[j], model.half_z[j],
                        px_i, py_i, pz_i,
                        qx_i, qy_i, qz_i, qw_i,
                        model.half_lengths[i], model.radii[i],
                    )
                    dist = result[0]
                    contact_x = result[1]
                    contact_y = result[2]
                    contact_z = result[3]
                    nx = -result[4]
                    ny = -result[5]
                    nz = -result[6]
                elif geom_i == GEOM_BOX and geom_j == GEOM_BOX:
                    var result = box_box(
                        px_i, py_i, pz_i,
                        qx_i, qy_i, qz_i, qw_i,
                        model.half_x[i], model.half_y[i], model.half_z[i],
                        px_j, py_j, pz_j,
                        qx_j, qy_j, qz_j, qw_j,
                        model.half_x[j], model.half_y[j], model.half_z[j],
                    )
                    dist = result[0]
                    contact_x = result[1]
                    contact_y = result[2]
                    contact_z = result[3]
                    nx = result[4]
                    ny = result[5]
                    nz = result[6]
                else:
                    # Default: Sphere-sphere (fallback)
                    var result = sphere_sphere(
                        px_i, py_i, pz_i, model.radii[i],
                        px_j, py_j, pz_j, model.radii[j],
                    )
                    dist = result[0]
                    contact_x = result[1]
                    contact_y = result[2]
                    contact_z = result[3]
                    nx = result[4]
                    ny = result[5]
                    nz = result[6]

                # Contact if penetrating (dist < 0)
                if dist < Scalar[DTYPE](0) and data.num_contacts < MAX_CONTACTS:
                    data.contacts[data.num_contacts].set(
                        i, j,
                        contact_x, contact_y, contact_z,
                        nx, ny, nz,
                        dist,
                    )
                    data.num_contacts += 1

    @always_inline
    @staticmethod
    fn detect_all_contacts_gpu[
        DTYPE: DType,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
        MAX_JOINTS: Int,
        STATE_SIZE: Int,
        BATCH: Int,
    ](
        env: Int,
        state: LayoutTensor[
            DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        model: LayoutTensor[
            DTYPE, Layout.row_major(NUM_BODIES, MODEL_BODY_SIZE), MutAnyOrigin
        ],
        ground_z: Scalar[DTYPE],
    ):
        """Detect all contacts for one environment (GPU version).

        Phase 9: Supports sphere, capsule, and box geometries.
        """
        var num_contacts = 0

        # Phase 1: Body-plane contacts
        for i in range(NUM_BODIES):
            if num_contacts >= MAX_CONTACTS:
                break

            var b_off = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](i)
            var px = rebind[Scalar[DTYPE]](state[env, b_off + BODY_IDX_PX])
            var py = rebind[Scalar[DTYPE]](state[env, b_off + BODY_IDX_PY])
            var pz = rebind[Scalar[DTYPE]](state[env, b_off + BODY_IDX_PZ])
            var radius = rebind[Scalar[DTYPE]](model[i, MODEL_IDX_RADIUS])
            var geom_type = Int(rebind[Scalar[DTYPE]](model[i, MODEL_IDX_GEOM_TYPE]))

            var dist: Scalar[DTYPE]
            var contact_x: Scalar[DTYPE]
            var contact_y: Scalar[DTYPE]
            var contact_z: Scalar[DTYPE]

            if geom_type == GEOM_BOX:
                # Box-plane collision
                var qx = rebind[Scalar[DTYPE]](state[env, b_off + BODY_IDX_QX])
                var qy = rebind[Scalar[DTYPE]](state[env, b_off + BODY_IDX_QY])
                var qz = rebind[Scalar[DTYPE]](state[env, b_off + BODY_IDX_QZ])
                var qw = rebind[Scalar[DTYPE]](state[env, b_off + BODY_IDX_QW])
                var hx = rebind[Scalar[DTYPE]](model[i, MODEL_IDX_HALF_X])
                var hy = rebind[Scalar[DTYPE]](model[i, MODEL_IDX_HALF_Y])
                var hz = rebind[Scalar[DTYPE]](model[i, MODEL_IDX_HALF_Z])

                var result = box_plane(
                    px, py, pz, qx, qy, qz, qw, hx, hy, hz, ground_z
                )
                dist = result[0]
                contact_x = result[1]
                contact_y = result[2]
                contact_z = result[3]
            elif geom_type == GEOM_CAPSULE:
                # Capsule-plane collision
                var qx = rebind[Scalar[DTYPE]](state[env, b_off + BODY_IDX_QX])
                var qy = rebind[Scalar[DTYPE]](state[env, b_off + BODY_IDX_QY])
                var qz = rebind[Scalar[DTYPE]](state[env, b_off + BODY_IDX_QZ])
                var qw = rebind[Scalar[DTYPE]](state[env, b_off + BODY_IDX_QW])
                var half_len = rebind[Scalar[DTYPE]](model[i, MODEL_IDX_HALF_LENGTH])

                var result = capsule_plane(
                    px, py, pz, qx, qy, qz, qw, half_len, radius, ground_z
                )
                dist = result[0]
                contact_x = result[1]
                contact_y = result[2]
                contact_z = result[3]
            else:
                # Default: Sphere-plane collision
                var result = sphere_plane(px, py, pz, radius, ground_z)
                dist = result[0]
                contact_x = result[1]
                contact_y = result[2]
                contact_z = result[3]

            if dist < Scalar[DTYPE](0):
                var c_off = contact_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](
                    num_contacts
                )
                state[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](i)
                state[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](-1)
                state[env, c_off + CONTACT_IDX_POS_X] = contact_x
                state[env, c_off + CONTACT_IDX_POS_Y] = contact_y
                state[env, c_off + CONTACT_IDX_POS_Z] = contact_z
                state[env, c_off + CONTACT_IDX_NX] = Scalar[DTYPE](0)
                state[env, c_off + CONTACT_IDX_NY] = Scalar[DTYPE](0)
                state[env, c_off + CONTACT_IDX_NZ] = Scalar[DTYPE](1)
                state[env, c_off + CONTACT_IDX_DIST] = dist
                num_contacts += 1

        # Phase 2: Body-body contacts
        for i in range(NUM_BODIES):
            for j in range(i + 1, NUM_BODIES):
                if num_contacts >= MAX_CONTACTS:
                    break

                var b_off_i = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](i)
                var b_off_j = body_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](j)

                var px_i = rebind[Scalar[DTYPE]](state[env, b_off_i + BODY_IDX_PX])
                var py_i = rebind[Scalar[DTYPE]](state[env, b_off_i + BODY_IDX_PY])
                var pz_i = rebind[Scalar[DTYPE]](state[env, b_off_i + BODY_IDX_PZ])
                var r_i = rebind[Scalar[DTYPE]](model[i, MODEL_IDX_RADIUS])
                var geom_i = Int(rebind[Scalar[DTYPE]](model[i, MODEL_IDX_GEOM_TYPE]))

                var px_j = rebind[Scalar[DTYPE]](state[env, b_off_j + BODY_IDX_PX])
                var py_j = rebind[Scalar[DTYPE]](state[env, b_off_j + BODY_IDX_PY])
                var pz_j = rebind[Scalar[DTYPE]](state[env, b_off_j + BODY_IDX_PZ])
                var r_j = rebind[Scalar[DTYPE]](model[j, MODEL_IDX_RADIUS])
                var geom_j = Int(rebind[Scalar[DTYPE]](model[j, MODEL_IDX_GEOM_TYPE]))

                # Get quaternions (needed for capsules and boxes)
                var qx_i = rebind[Scalar[DTYPE]](state[env, b_off_i + BODY_IDX_QX])
                var qy_i = rebind[Scalar[DTYPE]](state[env, b_off_i + BODY_IDX_QY])
                var qz_i = rebind[Scalar[DTYPE]](state[env, b_off_i + BODY_IDX_QZ])
                var qw_i = rebind[Scalar[DTYPE]](state[env, b_off_i + BODY_IDX_QW])
                var qx_j = rebind[Scalar[DTYPE]](state[env, b_off_j + BODY_IDX_QX])
                var qy_j = rebind[Scalar[DTYPE]](state[env, b_off_j + BODY_IDX_QY])
                var qz_j = rebind[Scalar[DTYPE]](state[env, b_off_j + BODY_IDX_QZ])
                var qw_j = rebind[Scalar[DTYPE]](state[env, b_off_j + BODY_IDX_QW])
                var hl_i = rebind[Scalar[DTYPE]](model[i, MODEL_IDX_HALF_LENGTH])
                var hl_j = rebind[Scalar[DTYPE]](model[j, MODEL_IDX_HALF_LENGTH])
                var hx_i = rebind[Scalar[DTYPE]](model[i, MODEL_IDX_HALF_X])
                var hy_i = rebind[Scalar[DTYPE]](model[i, MODEL_IDX_HALF_Y])
                var hz_i = rebind[Scalar[DTYPE]](model[i, MODEL_IDX_HALF_Z])
                var hx_j = rebind[Scalar[DTYPE]](model[j, MODEL_IDX_HALF_X])
                var hy_j = rebind[Scalar[DTYPE]](model[j, MODEL_IDX_HALF_Y])
                var hz_j = rebind[Scalar[DTYPE]](model[j, MODEL_IDX_HALF_Z])

                var dist: Scalar[DTYPE]
                var contact_x: Scalar[DTYPE]
                var contact_y: Scalar[DTYPE]
                var contact_z: Scalar[DTYPE]
                var nx: Scalar[DTYPE]
                var ny: Scalar[DTYPE]
                var nz: Scalar[DTYPE]

                # Dispatch based on geometry types
                if geom_i == GEOM_SPHERE and geom_j == GEOM_SPHERE:
                    var result = sphere_sphere(
                        px_i, py_i, pz_i, r_i, px_j, py_j, pz_j, r_j
                    )
                    dist = result[0]
                    contact_x = result[1]
                    contact_y = result[2]
                    contact_z = result[3]
                    nx = result[4]
                    ny = result[5]
                    nz = result[6]
                elif geom_i == GEOM_CAPSULE and geom_j == GEOM_SPHERE:
                    var result = capsule_sphere(
                        px_i, py_i, pz_i, qx_i, qy_i, qz_i, qw_i, hl_i, r_i,
                        px_j, py_j, pz_j, r_j,
                    )
                    dist = result[0]
                    contact_x = result[1]
                    contact_y = result[2]
                    contact_z = result[3]
                    nx = result[4]
                    ny = result[5]
                    nz = result[6]
                elif geom_i == GEOM_SPHERE and geom_j == GEOM_CAPSULE:
                    var result = capsule_sphere(
                        px_j, py_j, pz_j, qx_j, qy_j, qz_j, qw_j, hl_j, r_j,
                        px_i, py_i, pz_i, r_i,
                    )
                    dist = result[0]
                    contact_x = result[1]
                    contact_y = result[2]
                    contact_z = result[3]
                    nx = -result[4]
                    ny = -result[5]
                    nz = -result[6]
                elif geom_i == GEOM_CAPSULE and geom_j == GEOM_CAPSULE:
                    var result = capsule_capsule(
                        px_i, py_i, pz_i, qx_i, qy_i, qz_i, qw_i, hl_i, r_i,
                        px_j, py_j, pz_j, qx_j, qy_j, qz_j, qw_j, hl_j, r_j,
                    )
                    dist = result[0]
                    contact_x = result[1]
                    contact_y = result[2]
                    contact_z = result[3]
                    nx = result[4]
                    ny = result[5]
                    nz = result[6]
                # Box collision cases (Phase 9)
                elif geom_i == GEOM_BOX and geom_j == GEOM_SPHERE:
                    var result = box_sphere(
                        px_i, py_i, pz_i, qx_i, qy_i, qz_i, qw_i,
                        hx_i, hy_i, hz_i,
                        px_j, py_j, pz_j, r_j,
                    )
                    dist = result[0]
                    contact_x = result[1]
                    contact_y = result[2]
                    contact_z = result[3]
                    nx = result[4]
                    ny = result[5]
                    nz = result[6]
                elif geom_i == GEOM_SPHERE and geom_j == GEOM_BOX:
                    var result = box_sphere(
                        px_j, py_j, pz_j, qx_j, qy_j, qz_j, qw_j,
                        hx_j, hy_j, hz_j,
                        px_i, py_i, pz_i, r_i,
                    )
                    dist = result[0]
                    contact_x = result[1]
                    contact_y = result[2]
                    contact_z = result[3]
                    nx = -result[4]
                    ny = -result[5]
                    nz = -result[6]
                elif geom_i == GEOM_BOX and geom_j == GEOM_CAPSULE:
                    var result = box_capsule(
                        px_i, py_i, pz_i, qx_i, qy_i, qz_i, qw_i,
                        hx_i, hy_i, hz_i,
                        px_j, py_j, pz_j, qx_j, qy_j, qz_j, qw_j,
                        hl_j, r_j,
                    )
                    dist = result[0]
                    contact_x = result[1]
                    contact_y = result[2]
                    contact_z = result[3]
                    nx = result[4]
                    ny = result[5]
                    nz = result[6]
                elif geom_i == GEOM_CAPSULE and geom_j == GEOM_BOX:
                    var result = box_capsule(
                        px_j, py_j, pz_j, qx_j, qy_j, qz_j, qw_j,
                        hx_j, hy_j, hz_j,
                        px_i, py_i, pz_i, qx_i, qy_i, qz_i, qw_i,
                        hl_i, r_i,
                    )
                    dist = result[0]
                    contact_x = result[1]
                    contact_y = result[2]
                    contact_z = result[3]
                    nx = -result[4]
                    ny = -result[5]
                    nz = -result[6]
                elif geom_i == GEOM_BOX and geom_j == GEOM_BOX:
                    var result = box_box(
                        px_i, py_i, pz_i, qx_i, qy_i, qz_i, qw_i,
                        hx_i, hy_i, hz_i,
                        px_j, py_j, pz_j, qx_j, qy_j, qz_j, qw_j,
                        hx_j, hy_j, hz_j,
                    )
                    dist = result[0]
                    contact_x = result[1]
                    contact_y = result[2]
                    contact_z = result[3]
                    nx = result[4]
                    ny = result[5]
                    nz = result[6]
                else:
                    # Default: Sphere-sphere (fallback)
                    var result = sphere_sphere(
                        px_i, py_i, pz_i, r_i, px_j, py_j, pz_j, r_j
                    )
                    dist = result[0]
                    contact_x = result[1]
                    contact_y = result[2]
                    contact_z = result[3]
                    nx = result[4]
                    ny = result[5]
                    nz = result[6]

                if dist < Scalar[DTYPE](0):
                    var c_off = contact_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS](
                        num_contacts
                    )
                    state[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](i)
                    state[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](j)
                    state[env, c_off + CONTACT_IDX_POS_X] = contact_x
                    state[env, c_off + CONTACT_IDX_POS_Y] = contact_y
                    state[env, c_off + CONTACT_IDX_POS_Z] = contact_z
                    state[env, c_off + CONTACT_IDX_NX] = nx
                    state[env, c_off + CONTACT_IDX_NY] = ny
                    state[env, c_off + CONTACT_IDX_NZ] = nz
                    state[env, c_off + CONTACT_IDX_DIST] = dist
                    num_contacts += 1

        # Store contact count in metadata
        var meta_off = metadata_offset[NUM_BODIES, MAX_CONTACTS, MAX_JOINTS]()
        state[env, meta_off + META_IDX_NUM_CONTACTS] = Scalar[DTYPE](num_contacts)
