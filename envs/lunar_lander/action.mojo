from core import Action


@fieldwise_init
struct LunarLanderAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Action for LunarLander: 0=nop, 1=left, 2=main, 3=right."""

    var action_idx: Int

    fn __copyinit__(out self, existing: Self):
        self.action_idx = existing.action_idx

    fn __moveinit__(out self, deinit existing: Self):
        self.action_idx = existing.action_idx

    @staticmethod
    fn nop() -> Self:
        """Do nothing."""
        return Self(action_idx=0)

    @staticmethod
    fn left_engine() -> Self:
        """Fire left engine."""
        return Self(action_idx=1)

    @staticmethod
    fn main_engine() -> Self:
        """Fire main engine."""
        return Self(action_idx=2)

    @staticmethod
    fn right_engine() -> Self:
        """Fire right engine."""
        return Self(action_idx=3)
