"""Common data types for the render package.

Simple value types with no SDL library dependency.
"""

from memory import UnsafePointer


struct SDLHandle(ImplicitlyCopyable, Movable):
    """Generic opaque handle for SDL objects."""

    var ptr: UnsafePointer[UInt8, MutAnyOrigin]

    fn __init__(out self):
        self.ptr = UnsafePointer[UInt8, MutAnyOrigin]()

    fn __init__(out self, ptr: UnsafePointer[UInt8, MutAnyOrigin]):
        self.ptr = ptr

    fn __bool__(self) -> Bool:
        return self.ptr.__bool__()

    fn copy(self) -> Self:
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
