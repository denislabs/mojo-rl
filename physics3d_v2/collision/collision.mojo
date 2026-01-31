"""Physics3D v2 multi-body collision detection.

Phase 3: Detects all contacts (sphere-plane + sphere-sphere) for multi-body systems.
Uses the pure collision primitives from collision_primitives.mojo.
"""

from .collision_primitives import sphere_sphere, sphere_plane
from ..types import Model, Data
from ..traits import CollisionSystem
from layout import LayoutTensor, Layout
from ..gpu.constants import (
    BODY_IDX_PX,
    BODY_IDX_PY,
    BODY_IDX_PZ,
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
)


struct CollisionDetector(CollisionSystem):
    @always_inline
    @staticmethod
    fn detect_all_contacts[
        DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int
    ](
        model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS],
        mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS],
    ):
        """Detect all contacts (sphere-plane + sphere-sphere).

        O(N) for ground contacts, O(N²) for sphere-sphere.
        Fills the data.contacts buffer and sets data.num_contacts.

        Args:
            model: Static model configuration.
            data: Mutable simulation state.
        """
        data.num_contacts = 0

        # Phase 1: Sphere-plane contacts (each body vs ground)
        for i in range(NUM_BODIES):
            var px = data.positions[i * 3 + 0]
            var py = data.positions[i * 3 + 1]
            var pz = data.positions[i * 3 + 2]
            var radius = model.radii[i]

            var result = sphere_plane(px, py, pz, radius, model.ground_z)
            var dist = result[0]

            # Contact if penetrating (dist < 0)
            if dist < Scalar[DTYPE](0) and data.num_contacts < MAX_CONTACTS:
                data.contacts[data.num_contacts].set(
                    i,
                    -1,  # -1 indicates ground
                    result[1],
                    result[2],
                    result[3],  # Contact position
                    Scalar[DTYPE](0),
                    Scalar[DTYPE](0),
                    Scalar[DTYPE](1),  # Normal up
                    dist,
                )
                data.num_contacts += 1

        # Phase 2: Sphere-sphere contacts (all pairs)
        for i in range(NUM_BODIES):
            for j in range(i + 1, NUM_BODIES):
                var px_i = data.positions[i * 3 + 0]
                var py_i = data.positions[i * 3 + 1]
                var pz_i = data.positions[i * 3 + 2]
                var px_j = data.positions[j * 3 + 0]
                var py_j = data.positions[j * 3 + 1]
                var pz_j = data.positions[j * 3 + 2]

                var result = sphere_sphere(
                    px_i,
                    py_i,
                    pz_i,
                    model.radii[i],
                    px_j,
                    py_j,
                    pz_j,
                    model.radii[j],
                )
                var dist = result[0]

                # Contact if penetrating (dist < 0)
                if dist < Scalar[DTYPE](0) and data.num_contacts < MAX_CONTACTS:
                    data.contacts[data.num_contacts].set(
                        i,
                        j,
                        result[1],
                        result[2],
                        result[3],  # Contact position
                        result[4],
                        result[5],
                        result[6],  # Normal
                        dist,
                    )
                    data.num_contacts += 1

    @always_inline
    @staticmethod
    fn detect_all_contacts_gpu[
        DTYPE: DType,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
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
        """Detect all contacts for one environment."""
        var num_contacts = 0

        # Phase 1: Sphere-plane contacts
        for i in range(NUM_BODIES):
            if num_contacts >= MAX_CONTACTS:
                break

            var b_off = body_offset[NUM_BODIES, MAX_CONTACTS](i)
            var px = rebind[Scalar[DTYPE]](state[env, b_off + BODY_IDX_PX])
            var py = rebind[Scalar[DTYPE]](state[env, b_off + BODY_IDX_PY])
            var pz = rebind[Scalar[DTYPE]](state[env, b_off + BODY_IDX_PZ])
            var radius = rebind[Scalar[DTYPE]](model[i, MODEL_IDX_RADIUS])

            var result = sphere_plane(px, py, pz, radius, ground_z)
            var dist = result[0]

            if dist < Scalar[DTYPE](0):
                var c_off = contact_offset[NUM_BODIES, MAX_CONTACTS](
                    num_contacts
                )
                state[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](i)
                state[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](-1)
                state[env, c_off + CONTACT_IDX_POS_X] = result[1]
                state[env, c_off + CONTACT_IDX_POS_Y] = result[2]
                state[env, c_off + CONTACT_IDX_POS_Z] = result[3]
                state[env, c_off + CONTACT_IDX_NX] = Scalar[DTYPE](0)
                state[env, c_off + CONTACT_IDX_NY] = Scalar[DTYPE](0)
                state[env, c_off + CONTACT_IDX_NZ] = Scalar[DTYPE](1)
                state[env, c_off + CONTACT_IDX_DIST] = dist
                num_contacts += 1

        # Phase 2: Sphere-sphere contacts
        for i in range(NUM_BODIES):
            for j in range(i + 1, NUM_BODIES):
                if num_contacts >= MAX_CONTACTS:
                    break

                var b_off_i = body_offset[NUM_BODIES, MAX_CONTACTS](i)
                var b_off_j = body_offset[NUM_BODIES, MAX_CONTACTS](j)

                var px_i = rebind[Scalar[DTYPE]](
                    state[env, b_off_i + BODY_IDX_PX]
                )
                var py_i = rebind[Scalar[DTYPE]](
                    state[env, b_off_i + BODY_IDX_PY]
                )
                var pz_i = rebind[Scalar[DTYPE]](
                    state[env, b_off_i + BODY_IDX_PZ]
                )
                var r_i = rebind[Scalar[DTYPE]](model[i, MODEL_IDX_RADIUS])

                var px_j = rebind[Scalar[DTYPE]](
                    state[env, b_off_j + BODY_IDX_PX]
                )
                var py_j = rebind[Scalar[DTYPE]](
                    state[env, b_off_j + BODY_IDX_PY]
                )
                var pz_j = rebind[Scalar[DTYPE]](
                    state[env, b_off_j + BODY_IDX_PZ]
                )
                var r_j = rebind[Scalar[DTYPE]](model[j, MODEL_IDX_RADIUS])

                var result = sphere_sphere(
                    px_i, py_i, pz_i, r_i, px_j, py_j, pz_j, r_j
                )
                var dist = result[0]

                if dist < Scalar[DTYPE](0):
                    var c_off = contact_offset[NUM_BODIES, MAX_CONTACTS](
                        num_contacts
                    )
                    state[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](i)
                    state[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](j)
                    state[env, c_off + CONTACT_IDX_POS_X] = result[1]
                    state[env, c_off + CONTACT_IDX_POS_Y] = result[2]
                    state[env, c_off + CONTACT_IDX_POS_Z] = result[3]
                    state[env, c_off + CONTACT_IDX_NX] = result[4]
                    state[env, c_off + CONTACT_IDX_NY] = result[5]
                    state[env, c_off + CONTACT_IDX_NZ] = result[6]
                    state[env, c_off + CONTACT_IDX_DIST] = dist
                    num_contacts += 1

        # Store contact count in metadata
        var meta_off = metadata_offset[NUM_BODIES, MAX_CONTACTS]()
        state[env, meta_off + META_IDX_NUM_CONTACTS] = Scalar[DTYPE](
            num_contacts
        )
