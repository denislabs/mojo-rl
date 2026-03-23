"""Shared types for board game environments."""

from mojo_rl.core import State, Action

comptime board_dtype = DType.float32


@fieldwise_init
struct BoardGameState(Copyable, ImplicitlyCopyable, Movable, State):
    """Generic state wrapper for board games (needed for Env trait)."""

    var index: Int

    def __init__(out self, *, copy: Self):
        self.index = copy.index

    def __init__(out self, *, deinit take: Self):
        self.index = take.index

    def __eq__(self, other: Self) -> Bool:
        return self.index == other.index


@fieldwise_init
struct BoardGameAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Generic discrete action wrapper for board games."""

    var value: Int

    def __init__(out self, *, copy: Self):
        self.value = copy.value

    def __init__(out self, *, deinit take: Self):
        self.value = take.value
