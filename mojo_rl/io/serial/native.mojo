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

## The shim is a dylib, and it has to be

An earlier version resolved the symbol with a hand-rolled `dlopen`/`dlsym`,
which let `mojo build … -Xlinker mrl_serial.o -Xlinker -u -Xlinker
_mrl_serial_set_speed` produce ONE self-contained binary. That worked for the
hardware teleop and **stopped working the moment the renderer shared the
binary**, for two compounding reasons:

* ⚠ **`external_call` re-declares a C symbol per module, and two declarations
  of one symbol with different signatures fail at LLVM LOWERING** — "existing
  function with conflicting signature", not a parse error. The stdlib already
  declares `dlsym` (`std/sys/_libc`), and the viewer pulls it in. The same
  trap hit `write`, `read` and `open`; `port.mojo` records how each was
  matched or side-stepped.
* ⚠ **`mojo run`'s JIT does not honour `-Xlinker` at all** — object or dylib,
  the symbol simply fails to materialise. So a direct
  `external_call["mrl_serial_set_speed"]` makes the module un-runnable under
  `mojo run` whether or not the call is ever reached.

Together those rule out every combination but this one: resolve through
`_get_dylib_function`, the same path `render/imgui` uses. It calls the
stdlib's own `dlsym`, so it declares nothing and collides with nothing.

**Consequence, stated plainly: a built binary needs `libmrl_serial.dylib`
beside it** (or at `MOJO_RL_SERIAL_LIB`), exactly as the imgui viewers need
`libmojo_imgui.dylib`. Build it with `pixi run build-serial`.
"""

from std.ffi import OwnedDLHandle, _Global, _get_dylib_function
from std.os import abort, getenv
from std.pathlib import Path
from std.sys import CompilationTarget


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


def serial_shim_available() -> Bool:
    """True when the shim can be found WITHOUT dlopening it.

    `_Global` aborts the process on a missing library — right for a hard
    dependency, wrong as a first impression. Call this before opening a port
    to print "run `pixi run build-serial`" instead of dying in the loader.
    """
    var candidates = _candidates()
    for i in range(len(candidates)):
        if Path(candidates[i]).exists():
            return True
    return False


def _init_serial_handle() -> OwnedDLHandle:
    """Non-raising, as `_Global` demands; aborts with the paths it tried."""
    var candidates = _candidates()
    for i in range(len(candidates)):
        try:
            return OwnedDLHandle(candidates[i])
        except:
            pass

    var tried = String("")
    for i in range(len(candidates)):
        tried += "\n  - " + candidates[i]
    abort(
        "serial shim not found. Tried:"
        + tried
        + "\nBuild it with `pixi run build-serial`, or set"
        + " MOJO_RL_SERIAL_LIB=/path/to/"
        + _lib_name()
    )


comptime lib = _Global["MOJO_RL_SERIAL", _init_serial_handle]()


def set_speed(fd: Int32, baud: Int) raises -> Int32:
    """`ioctl(fd, IOSSIOSPEED, &baud)` on macOS; a no-op elsewhere.

    Runs once per port at open time, never in a control loop, so the lookup's
    cost is irrelevant — and `_get_dylib_function` caches it anyway.
    """
    comptime if not CompilationTarget.is_macos():
        return 0

    return _get_dylib_function[
        lib, "mrl_serial_set_speed", def (Int32, UInt64) thin -> Int32
    ]()(fd, UInt64(baud))
