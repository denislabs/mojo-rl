from mojo_rl.core import Action


@fieldwise_init
struct BipedalWalkerAction[DTYPE: DType](
    Action, Copyable, ImplicitlyCopyable, Movable
):
    """4D continuous action for BipedalWalker."""

    var hip1: Scalar[Self.DTYPE]
    var knee1: Scalar[Self.DTYPE]
    var hip2: Scalar[Self.DTYPE]
    var knee2: Scalar[Self.DTYPE]

    def __init__(out self):
        self.hip1 = 0.0
        self.knee1 = 0.0
        self.hip2 = 0.0
        self.knee2 = 0.0

    def __init__(out self, *, copy: Self):
        self.hip1 = copy.hip1
        self.knee1 = copy.knee1
        self.hip2 = copy.hip2
        self.knee2 = copy.knee2

    def __init__(out self, *, deinit move: Self):
        self.hip1 = move.hip1
        self.knee1 = move.knee1
        self.hip2 = move.hip2
        self.knee2 = move.knee2
