from mojo_rl.core import Action


@fieldwise_init
struct LunarLanderAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Action for LunarLander: 0=nop, 1=left, 2=main, 3=right."""

    var action_idx: Int

    def __init__(out self, *, copy: Self):
        self.action_idx = copy.action_idx

    def __init__(out self, *, deinit move: Self):
        self.action_idx = move.action_idx

    @staticmethod
    def nop() -> Self:
        """Do nothing."""
        return Self(action_idx=0)

    @staticmethod
    def left_engine() -> Self:
        """Fire left engine."""
        return Self(action_idx=1)

    @staticmethod
    def main_engine() -> Self:
        """Fire main engine."""
        return Self(action_idx=2)

    @staticmethod
    def right_engine() -> Self:
        """Fire right engine."""
        return Self(action_idx=3)
