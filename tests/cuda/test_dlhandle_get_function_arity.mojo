"""`OwnedDLHandle.get_function` takes the RETURN type — guard the spelling.

`mojo_rl/cuda/graph.mojo` resolves eleven interceptor symbols. Before Mojo 1.0
the idiom was

    var f = lib.get_function[def (A, B) thin -> R](name)()   # trailing () unwraps
    var r = f(a, b)

where the parameter was the whole function type and the trailing `()` unwrapped
a `_DLCallable` into a raw function pointer. In 1.0 the signature is

    def get_function[return_type: RegisterPassable = NoneType](
        ref self, var name: String
    ) -> _DLCallable[return_type, origin_of(self)]

so the parameter is the symbol's RETURN type and the returned `_DLCallable` is
already the callable. The old spelling still COMPILES — a function type is a
valid `RegisterPassable` return type — but it now means "call the C function
with no arguments and interpret its return value as a function pointer". The
next call then jumps into that value. On NVIDIA that presented as a segfault at
an address equal to the CUDA stream handle, because `intercept_get_mojo_stream`
was invoked by the unwrap and its returned stream became the "function".

This test needs no GPU: the failure is in the FFI layer, so libc reproduces it.
It pins the correct spelling for both arities used by `graph.mojo` — a
zero-argument symbol (like `intercept_get_mojo_stream`) and one taking
arguments (like `intercept_stream_end_capture`).
"""

from std.ffi import OwnedDLHandle, c_int
from std.sys import CompilationTarget
from std.testing import assert_equal, assert_true


def main() raises:
    comptime libc = "libSystem.B.dylib" if CompilationTarget.is_macos() else "libc.so.6"
    var lib = OwnedDLHandle(String(libc))

    # --- zero-argument symbol -------------------------------------------
    # The shape that broke CUDA graphs: for a 0-arg symbol the old "unwrap"
    # and a real call are spelled identically, so the unwrap silently became
    # the call and the result was typed as code.
    var getpid = lib.get_function[c_int]("getpid")
    var pid = Int(getpid())
    assert_true(pid > 0, "getpid() should return a positive pid, got " + String(pid))
    # Same handle, resolved again: a stable value, not a fresh call into data.
    assert_equal(pid, Int(lib.get_function[c_int]("getpid")()))
    print("  0-arg  getpid() =", pid, " OK")

    # --- symbol taking arguments ----------------------------------------
    # Guards the other half: arguments must reach the callee. Under the old
    # spelling the C function ran with NO arguments and this value would be
    # whatever happened to be in the argument register.
    var abs_fn = lib.get_function[c_int]("abs")
    assert_equal(5, Int(abs_fn(c_int(-5))))
    assert_equal(0, Int(abs_fn(c_int(0))))
    assert_equal(7, Int(abs_fn(c_int(7))))
    print("  1-arg  abs(-5) =", Int(abs_fn(c_int(-5))), " OK")

    print("DLHANDLE get_function ARITY OK")
