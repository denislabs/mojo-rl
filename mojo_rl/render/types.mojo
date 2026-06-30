"""Common data types for the render package.

Simple value types with no SDL library dependency.
"""

from std.memory import UnsafePointer


struct SDLHandle(ImplicitlyCopyable, Movable):
    """Generic opaque handle for SDL objects."""

    var ptr: Optional[UnsafePointer[UInt8, MutUntrackedOrigin]]

    def __init__(out self):
        self.ptr = None

    def __init__[o: Origin](out self, ptr: UnsafePointer[UInt8, o]):
        self.ptr = rebind[UnsafePointer[UInt8, MutUntrackedOrigin]](ptr)

    def __init__(
        out self, ptr: Optional[UnsafePointer[UInt8, MutUntrackedOrigin]]
    ):
        self.ptr = ptr

    def __bool__(self) -> Bool:
        return Bool(self.ptr)

    def copy(self) -> Self:
        return Self(self.ptr)


@fieldwise_init
struct SDL_Rect(ImplicitlyCopyable, Movable):
    """SDL rectangle structure."""

    var x: Int32
    var y: Int32
    var w: Int32
    var h: Int32


@fieldwise_init
struct SDL_Point(ImplicitlyCopyable, Movable):
    """SDL point structure."""

    var x: Int32
    var y: Int32


@fieldwise_init
struct Color(ImplicitlyCopyable, Movable):
    """RGBA color (0-255 per component)."""

    var r: UInt8
    var g: UInt8
    var b: UInt8
    var a: UInt8


comptime SDL_Color = Color
