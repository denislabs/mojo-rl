"""CollisionSystem trait for physics simulation.

CollisionSystem handles broad-phase and narrow-phase collision detection,
generating contact manifolds for the constraint solver.
"""

from ..constants import dtype, BODY_STATE_SIZE, SHAPE_MAX_SIZE, CONTACT_DATA_SIZE
from layout import LayoutTensor, Layout
from gpu.host import DeviceContext, DeviceBuffer


trait CollisionSystem(Movable & ImplicitlyCopyable):
    """Trait for collision detection systems.

    CollisionSystems detect overlapping bodies and generate contact manifolds.
    Different implementations can use different algorithms:
    - FlatTerrainCollision: Optimized for flat ground (LunarLander)
    - PolygonSATCollision: General polygon-polygon using SAT
    - SpatialHashCollision: Efficient for many bodies

    Layout:
    - bodies: [BATCH, NUM_BODIES, BODY_STATE_SIZE]
    - shapes: [NUM_SHAPES, SHAPE_MAX_SIZE]
    - contacts: [BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE]
    - contact_counts: [BATCH] - number of active contacts per environment

    Note: MAX_CONTACTS is passed as a template parameter to detect() rather
    than as a comptime constant, allowing flexibility in buffer sizing.
    """

    # =========================================================================
    # CPU Methods
    # =========================================================================

    fn detect[
        BATCH: Int,
        NUM_BODIES: Int,
        NUM_SHAPES: Int,
        MAX_CONTACTS: Int = 16,
    ](
        self,
        bodies: LayoutTensor[
            dtype, Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE), MutAnyOrigin
        ],
        shapes: LayoutTensor[
            dtype, Layout.row_major(NUM_SHAPES, SHAPE_MAX_SIZE), MutAnyOrigin
        ],
        mut contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ],
        mut contact_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ],
    ):
        """Detect collisions and generate contact manifolds.

        Args:
            bodies: Body state tensor [BATCH, NUM_BODIES, BODY_STATE_SIZE].
            shapes: Shape definitions [NUM_SHAPES, SHAPE_MAX_SIZE].
            contacts: Contact output tensor [BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE].
            contact_counts: Number of contacts per env [BATCH].
        """
        ...

    # =========================================================================
    # GPU Methods
    # =========================================================================
    # Note: GPU methods are not defined in the trait because different
    # collision systems require different parameters (e.g., ground_y for
    # flat terrain). Each implementation provides its own detect_gpu method.
    # The PhysicsWorld orchestrator calls these methods directly.
