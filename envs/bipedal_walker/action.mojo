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

    fn __init__(out self, *, copy: Self):
        self.hip1 = copy.hip1
        self.knee1 = copy.knee1
        self.hip2 = copy.hip2
        self.knee2 = copy.knee2

    fn __init__(out self, *, deinit take: Self):
        self.hip1 = take.hip1
        self.knee1 = take.knee1
        self.hip2 = take.hip2
        self.knee2 = take.knee2
