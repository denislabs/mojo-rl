"""PushT action: 2-D continuous target position in [0, 512]²."""

from mojo_rl.core import Action
from .constants import PConstants


struct PushTAction[DTYPE: DType](
    Action, Copyable, ImplicitlyCopyable, Movable
):
    """Target position the PD controller will drive the agent toward."""

    var target_x: Scalar[Self.DTYPE]
    var target_y: Scalar[Self.DTYPE]

    def __init__(out self):
        self.target_x = Scalar[Self.DTYPE](256.0)
        self.target_y = Scalar[Self.DTYPE](256.0)

    def __init__(
        out self,
        target_x: Scalar[Self.DTYPE],
        target_y: Scalar[Self.DTYPE],
    ):
        self.target_x = target_x
        self.target_y = target_y

    def __init__(out self, *, copy: Self):
        self.target_x = copy.target_x
        self.target_y = copy.target_y

    def __init__(out self, *, deinit move: Self):
        self.target_x = move.target_x
        self.target_y = move.target_y

    @staticmethod
    def from_list(
        values: List[Scalar[Self.DTYPE]],
    ) -> PushTAction[Self.DTYPE]:
        var a = PushTAction[Self.DTYPE]()
        if len(values) > 0:
            a.target_x = values[0]
        if len(values) > 1:
            a.target_y = values[1]
        return a^

    def to_list(self) -> List[Scalar[Self.DTYPE]]:
        var out = List[Scalar[Self.DTYPE]](capacity=PConstants.ACTION_DIM)
        out.append(self.target_x)
        out.append(self.target_y)
        return out^

    def clamp(self) -> PushTAction[Self.DTYPE]:
        var lo = Scalar[Self.DTYPE](PConstants.ACTION_LOW)
        var hi = Scalar[Self.DTYPE](PConstants.ACTION_HIGH)
        var tx = self.target_x
        var ty = self.target_y
        if tx < lo:
            tx = lo
        elif tx > hi:
            tx = hi
        if ty < lo:
            ty = lo
        elif ty > hi:
            ty = hi
        return PushTAction[Self.DTYPE](target_x=tx, target_y=ty)
