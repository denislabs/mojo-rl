from core import Action


@fieldwise_init
struct BipedalWalkerAction[DTYPE: DType](
    Action, Copyable, ImplicitlyCopyable, Movable
):
    """4D continuous action for BipedalWalker."""

    var hip1: Scalar[Self.DTYPE]
    var knee1: Scalar[Self.DTYPE]
    var hip2: Scalar[Self.DTYPE]
    var knee2: Scalar[Self.DTYPE]

    fn __init__(out self):
        self.hip1 = 0.0
        self.knee1 = 0.0
        self.hip2 = 0.0
        self.knee2 = 0.0

    fn __copyinit__(out self, existing: Self):
        self.hip1 = existing.hip1
        self.knee1 = existing.knee1
        self.hip2 = existing.hip2
        self.knee2 = existing.knee2

    fn __moveinit__(out self, deinit existing: Self):
        self.hip1 = existing.hip1
        self.knee1 = existing.knee1
        self.hip2 = existing.hip2
        self.knee2 = existing.knee2
