# +--------------------------------------------------------------------------+ #
# | mojo-rl serial — the C shim, and why it exists
# +--------------------------------------------------------------------------+ #
"""Resolution of the ONE call the serial layer cannot make from Mojo.

`ioctl` is C-variadic. Mojo's `external_call` emits a *fixed* prototype, so
the third argument is passed in a register — while on Apple arm64 a variadic
callee reads its arguments from the **stack**. `ioctl(fd, IOSSIOSPEED, &speed)`
therefore returns `-1` with `errno == EFAULT`, silently, at runtime.

⚠ This is not a Mojo defect. The identical C call declared non-variadically
(`extern int f(int, unsigned long, void*) __asm__("_ioctl")`) fails the same
way — measured 2026-08-25. `std/ffi` says as much itself: *"Mojo function type
syntax has no variadic parameter form"* (MOCO-3692).

The shim is resolved **RTLD_DEFAULT first**, so:

* `mojo build ... -Xlinker mojo_rl/io/serial/mrl_serial.o -Xlinker -u
  -Xlinker _mrl_serial_set_speed` produces a single self-contained binary that
  never opens a dylib — which is the point, since the deployable artifact for
  the arm is supposed to be one native binary. ⚠ The `-u` is REQUIRED: nothing
  references the symbol at link time (that is the whole idea), so without it
  the linker dead-strips the object and the binary quietly falls back to the
  dylib;
* `mojo run`, whose JIT does not honour `-Xlinker` for objects, falls back to
  `libmrl_serial.dylib` beside this file.

Build both with `pixi run build-serial`.
"""

from std.ffi import external_call
from std.os import getenv
from std.pathlib import Path
from std.sys import CompilationTarget

comptime _RTLD_NOW = Int32(2)
# dlsym(RTLD_DEFAULT, ...) searches every image already loaded, the main
# executable included. That is what makes the statically linked build work.
comptime _RTLD_DEFAULT = -2


def _lib_name() -> String:
    comptime if CompilationTarget.is_macos():
        return String("libmrl_serial.dylib")
    else:
        return String("libmrl_serial.so")


def _candidates() -> List[String]:
    """Where to look for the shim, most explicit first — the same shape as
    `render/imgui`'s loader, so both answer to the same conventions."""
    var name = _lib_name()
    var out = List[String]()
    var override = getenv("MOJO_RL_SERIAL_LIB")
    if override.byte_length() > 0:
        out.append(override)
    var root = getenv("PIXI_PROJECT_ROOT")
    if root.byte_length() > 0:
        out.append(root + "/mojo_rl/io/serial/" + name)
    # Relative to CWD, which for this project is the repo root.
    out.append("mojo_rl/io/serial/" + name)
    out.append(name)
    return out^


def _dlsym(handle: Int, mut name: String) -> Int:
    return Int(
        external_call["dlsym", OpaquePointer[MutAnyOrigin]](
            handle, name.as_c_string_slice().unsafe_ptr()
        )
    )


def _resolve_handle() raises -> Int:
    """A dl handle whose `dlsym` finds `mrl_serial_set_speed`.

    ⚠ Deliberately NOT `OwnedDLHandle.check_symbol` — that returns `False` for
    symbols `dlsym` and `ctypes` both resolve, and `DLHandle.call` asserts on
    it, so a working shim aborts the process. Measured 2026-08-25.
    """
    var name = String("mrl_serial_set_speed")

    # 1. Already in the process? (a `-Xlinker …/mrl_serial.o` build)
    if _dlsym(_RTLD_DEFAULT, name) != 0:
        return _RTLD_DEFAULT

    # 2. Otherwise the dylib beside this package.
    var candidates = _candidates()
    for i in range(len(candidates)):
        var path = candidates[i]
        if not Path(path).exists():
            continue
        var h = Int(
            external_call["dlopen", OpaquePointer[MutAnyOrigin]](
                path.as_c_string_slice().unsafe_ptr(), _RTLD_NOW
            )
        )
        if h != 0 and _dlsym(h, name) != 0:
            return h

    var tried = String("")
    for i in range(len(candidates)):
        tried += "\n  - " + candidates[i]
    raise Error(
        "serial shim not found (mrl_serial_set_speed). Tried RTLD_DEFAULT,"
        " then:"
        + tried
        + "\nBuild it with `pixi run build-serial`, or set MOJO_RL_SERIAL_LIB."
    )


def set_speed(fd: Int32, baud: Int) raises -> Int32:
    """`ioctl(fd, IOSSIOSPEED, &baud)` on macOS; a no-op elsewhere.

    Resolved on every call: this runs once per port at open time, never in a
    control loop, so caching it would buy nothing and add a global.
    """
    comptime if not CompilationTarget.is_macos():
        return 0

    var name = String("mrl_serial_set_speed")
    var f = external_call["dlsym", def (Int32, UInt64) thin -> Int32](
        _resolve_handle(), name.as_c_string_slice().unsafe_ptr()
    )
    return f(fd, UInt64(baud))
