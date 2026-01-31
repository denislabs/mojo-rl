from ..types import Model, Data
from layout import LayoutTensor, Layout
from ..gpu.constants import MODEL_BODY_SIZE

trait CollisionSystem(Movable & ImplicitlyCopyable):
    """Trait for collision detection systems.

    CollisionSystems detect overlapping bodies
    and generate contact manifolds for the constraint solver.
    """

    @staticmethod
    fn detect_all_contacts[
        DTYPE: DType, NUM_BODIES: Int, MAX_CONTACTS: Int
    ](
        model: Model[DTYPE, NUM_BODIES, MAX_CONTACTS],
        mut data: Data[DTYPE, NUM_BODIES, MAX_CONTACTS],
    ):
        """Detect all contacts (sphere-plane + sphere-sphere)."""
        ...

    ...

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