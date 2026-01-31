"""Physics3D v2 multi-body collision detection.

Phase 3: Detects all contacts (sphere-plane + sphere-sphere) for multi-body systems.
Uses the pure collision primitives from collision_primitives.mojo.
"""

from .collision_primitives import sphere_sphere, sphere_plane
from ..types import Model, Data
from ..traits import CollisionSystem


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
