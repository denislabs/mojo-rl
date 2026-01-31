from ..types import Model, Data


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
