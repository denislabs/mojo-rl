"""OpenCV bindings — camera capture today, detection and calibration next.

    pixi run build-opencv       # ONCE, builds the C++ shim this binds to
    from mojo_rl.vision.opencv import VideoCapture, opencv_shim_available

Every function here is a thin call into `opencv_shim.cpp`, a flat C API over
OpenCV 5. `docs/OPENCV_SHIM_SCOPE.md` carries the full surface and the reasons;
what matters at the call site:

⚠ REQUIRES A BUILT SHIM. `pixi run build-opencv` produces `libmojo_cv.dylib`
beside this file. It is NOT tracked in git, so a fresh clone has to build it,
and the failure mode is a dlopen ABORT at the first call rather than a compile
error. `opencv_shim_available()` answers the question without touching FFI.

⚠⚠ FRAMES COME BACK BGR AND HWC. The rest of this project — the ACT store
above all — holds images CHW and RGB. A silent channel swap raises no error
anywhere and simply produces wrong answers, which is the exact class of defect
this tree keeps recording. `bgr_hwc_to_rgb_chw` is the ONE place that
conversion is allowed to happen, so that it has a fixed point to be gated at.

⚠ NOTHING IS VENDORED. Unlike Dear ImGui, OpenCV is already in every pixi env
as a full C++ build, so the shim compiles against the env and nothing is
cloned. That also means the dylib carries an `-rpath` into the env: it is a
development artifact, not a redistributable.
"""

from std.os import abort, getenv
from std.sys import CompilationTarget
from std.pathlib import Path
from std.ffi import _Global, OwnedDLHandle, _get_dylib_function, c_char

comptime Ptr = Pointer


def untracked[
    T: AnyType, o: Origin
](p: Pointer[T, o]) -> Pointer[T, MutUntrackedOrigin]:
    """Re-key a pointer's origin for an FFI call. Same helper as
    `render/sdl`, duplicated rather than imported so `vision` does not depend
    on `render` for three lines."""
    return rebind[Pointer[T, MutUntrackedOrigin]](p)


# ═══════════════════════════════════════════════════════════════════════════
# status codes — mirror `opencv_shim.cpp`
# ═══════════════════════════════════════════════════════════════════════════
#
# ⚠ NEGATIVE IS AN ERROR, ZERO IS OK, and positive is never a status. Anything
# a caller wants beyond success comes out through a pointer, so a status can
# never be confused with a value.

comptime CV_OK: Int32 = 0
comptime CV_ERR_CV: Int32 = -1
comptime CV_ERR_STD: Int32 = -2
comptime CV_ERR_UNKNOWN: Int32 = -3
comptime CV_ERR_ARG: Int32 = -4
comptime CV_ERR_CAPACITY: Int32 = -5
comptime CV_ERR_NO_FRAME: Int32 = -6


# ═══════════════════════════════════════════════════════════════════════════
# dylib loading — the `render/imgui` and `io/serial` pattern
# ═══════════════════════════════════════════════════════════════════════════


def _lib_name() -> String:
    comptime if CompilationTarget.is_macos():
        return String("libmojo_cv.dylib")
    elif CompilationTarget.is_linux():
        return String("libmojo_cv.so")
    else:
        comptime assert False, "OS is not supported"


def _candidates() -> List[String]:
    """Where to look, most explicit first.

    Kept separate from `_init_handle` so `opencv_shim_available()` can check
    the SAME list without dlopening anything — a probe consulting a different
    list would answer a different question than the loader asks.
    """
    var name = _lib_name()
    var out = List[String]()
    var override = getenv("MOJO_RL_OPENCV_LIB")
    if override.byte_length() > 0:
        out.append(override)
    var root = getenv("PIXI_PROJECT_ROOT")
    if root.byte_length() > 0:
        out.append(root + "/mojo_rl/vision/opencv/" + name)
    out.append("mojo_rl/vision/opencv/" + name)
    out.append(name)
    return out^


def opencv_shim_available() -> Bool:
    """True when the shim can be found WITHOUT dlopening it.

    `_Global` aborts the process on a missing library, which is right for a
    hard dependency and the wrong first impression for an optional one.
    """
    var c = _candidates()
    for i in range(len(c)):
        if Path(c[i]).exists():
            return True
    return False


def _init_handle() -> OwnedDLHandle:
    """Non-raising, as `_Global` demands; aborts with the paths it tried
    rather than returning an uninitialised handle, which would turn a missing
    library into a segfault at the first unrelated call."""
    var c = _candidates()
    for i in range(len(c)):
        try:
            return OwnedDLHandle(c[i])
        except:
            pass
    var tried = String("")
    for i in range(len(c)):
        tried += "\n  - " + c[i]
    abort(
        "OpenCV shim not found. Tried:"
        + tried
        + "\nBuild it with `pixi run build-opencv`, or set"
        + " MOJO_RL_OPENCV_LIB=/path/to/"
        + _lib_name()
    )


comptime lib = _Global["MOJO_RL_OPENCV", _init_handle]()


# ═══════════════════════════════════════════════════════════════════════════
# A — lifecycle
# ═══════════════════════════════════════════════════════════════════════════


def cv_last_error() raises -> String:
    """The last message the shim recorded, or "" if it has recorded none.

    ⚠ THREAD-LOCAL ON THE C SIDE, so this reports what THIS thread's last call
    failed with. A shared buffer would be a data race whose symptom is a wrong
    message rather than a crash.
    """
    var p = _get_dylib_function[
        lib, "mrl_cv_last_error", def() thin -> Ptr[c_char, MutUntrackedOrigin]
    ]()()
    # ⚠ NEVER NULL by construction — the shim returns "" when it has nothing
    # to say, so a caller can print this unconditionally.
    return String(unsafe_from_utf8_ptr=p)


def _check(status: Int32, what: String) raises:
    """Turn a status code into an exception carrying the C-side message."""
    if status == CV_OK:
        return
    var msg = String("opencv: ") + what + " failed (" + String(status) + ")"
    var detail = cv_last_error()
    if detail.byte_length() > 0:
        msg += ": " + detail
    raise msg


def cv_version() raises -> Tuple[Int, Int]:
    """`(major, minor)` of the OpenCV the shim was LINKED against.

    ⚠ A REAL COMPATIBILITY QUESTION, not a banner. OpenCV 5 moved `solvePnP`
    and `calibrateCamera` between headers and DELETED
    `estimatePoseSingleMarkers`, `calibrateCameraCharuco` and
    `calibrateHandEye`; a 4.x here would compile and then behave differently.
    Doubles as the cheapest possible "did the dylib actually load" probe.
    """
    var major = Int32(0)
    var minor = Int32(0)
    var st = _get_dylib_function[
        lib,
        "mrl_cv_version",
        def(
            Ptr[Int32, MutUntrackedOrigin], Ptr[Int32, MutUntrackedOrigin]
        ) thin -> Int32,
    ]()(untracked(Ptr(to=major)), untracked(Ptr(to=minor)))
    _check(st, "version")
    return (Int(major), Int(minor))


def cv_set_num_threads(n: Int) raises:
    """⚠ NOT A PERFORMANCE KNOB — A GATE REQUIREMENT.

    `calibrateCamera` is an iterative LM fit and OpenCV's `parallel_for_` can
    change reduction order with the thread count. Bit-equality against Python
    `cv2` is a claim about identical inputs AND identical scheduling, so both
    sides of a parity gate call this with 1.
    """
    _check(
        _get_dylib_function[
            lib, "mrl_cv_set_num_threads", def(Int32) thin -> Int32
        ]()(Int32(n)),
        "set_num_threads",
    )


# ═══════════════════════════════════════════════════════════════════════════
# B — capture
# ═══════════════════════════════════════════════════════════════════════════


struct VideoCapture(Movable):
    """A camera or a video file, yielding BGR HWC frames.

    ⚠ CLOSE IT EXPLICITLY. There is no destructor here: a capture left open
    holds a camera device against the whole machine, and `close()` is
    idempotent so a `try/finally` around a control loop is the right shape.
    (`examples/so101/deploy_reach_real.mojo` records what happens when a
    `finally` does NOT run on an abort — the same hazard, one device over.)
    """

    var _h: Int
    """The C `void*`, as an address.

    ⚠ AN `Int`, NOT A `Pointer`, AND THAT IS DELIBERATE. Mojo's `Pointer` is a
    SAFE type: it carries an origin and promises a dereferenceable pointee. We
    never dereference this — it is an opaque token minted and freed by
    `opencv_shim.cpp` — so modelling it as a `Pointer` would claim something
    false, and Mojo 1.0 gives it neither a null value nor a raw-address
    constructor anyway. `0` means closed. A `void*` and an `Int` are both one
    64-bit register in the C ABI, so the FFI signatures below are exact."""
    var width: Int
    var height: Int
    var channels: Int
    var fps: Float64
    var frame_count: Int
    """0 for a live device; only meaningful for a file."""

    def __init__(out self, _h: Int) raises:
        """Private — use `from_file` or `device`."""
        self._h = _h
        self.width = 0
        self.height = 0
        # ⚠ ASSUMED, THEN CONFIRMED. The channel count is not a capture
        # property in OpenCV; it is a property of the frame that comes back, so
        # `read` overwrites this with what actually arrived.
        self.channels = 3
        self.fps = 0.0
        self.frame_count = 0
        self._refresh_props()

    @staticmethod
    def from_file(path: String) raises -> Self:
        """Open a video file.

        ⚠ THIS IS WHAT THE GATE USES. A capture path tested only against a live
        camera has no gate at all — the frames are never the same twice, so
        there is nothing to compare against. Decoding a committed file through
        the same dylib on both sides IS comparable, bit for bit.
        """
        var p = path
        var h = _get_dylib_function[
            lib,
            "mrl_cv_cap_open_file",
            def(Ptr[c_char, MutUntrackedOrigin]) thin -> Int,
        ]()(untracked(p.as_c_string_slice().unsafe_ptr()))
        if h == 0:
            raise String("opencv: cannot open ") + path + ": " + cv_last_error()
        return Self(h)

    @staticmethod
    def device(
        index: Int, width: Int = 0, height: Int = 0, fps: Float64 = 0.0
    ) raises -> Self:
        """Open a live camera.

        ⚠ THE SIZE AND RATE ARE REQUESTS, NOT SETTINGS. A camera is free to
        ignore them, and OpenCV reports no error when it does — which is why
        `width`/`height`/`fps` on the returned struct are read back from the
        device rather than echoed from these arguments. A buffer sized from
        what you ASKED for, filled by a device that gave something else, is a
        defect that surfaces much later as a wrong answer.
        """
        var h = _get_dylib_function[
            lib,
            "mrl_cv_cap_open",
            def(Int32, Int32, Int32, Float64) thin -> Int,
        ]()(Int32(index), Int32(width), Int32(height), fps)
        if h == 0:
            raise String("opencv: cannot open device ") + String(
                index
            ) + ": " + cv_last_error()
        return Self(h)

    def _refresh_props(mut self) raises:
        var w = Int32(0)
        var h = Int32(0)
        var f = Float64(0.0)
        var n = Int32(0)
        var st = _get_dylib_function[
            lib,
            "mrl_cv_cap_props",
            def(
                Int,
                Ptr[Int32, MutUntrackedOrigin],
                Ptr[Int32, MutUntrackedOrigin],
                Ptr[Float64, MutUntrackedOrigin],
                Ptr[Int32, MutUntrackedOrigin],
            ) thin -> Int32,
        ]()(
            self._h,
            untracked(Ptr(to=w)),
            untracked(Ptr(to=h)),
            untracked(Ptr(to=f)),
            untracked(Ptr(to=n)),
        )
        _check(st, "cap_props")
        self.width = Int(w)
        self.height = Int(h)
        self.fps = f
        self.frame_count = Int(n)

    def frame_bytes(self) -> Int:
        """Bytes one frame needs, from the CURRENT reported geometry."""
        return self.width * self.height * self.channels

    def read(mut self, mut out: List[UInt8]) raises -> Bool:
        """Read one frame into `out`, resizing it to fit. BGR, HWC.

        Returns False at end of stream — which is an outcome, not an error, and
        the only non-exceptional way this returns False. Anything else raises.
        """
        var need = self.frame_bytes()
        if need <= 0:
            # A device that reported no geometry: ask for a generous buffer and
            # let the first frame tell us the truth.
            need = 1920 * 1080 * 3
        if len(out) < need:
            out.resize(need, 0)

        var w = Int32(0)
        var h = Int32(0)
        var c = Int32(0)
        var st = _get_dylib_function[
            lib,
            "mrl_cv_cap_read",
            def(
                Int,
                Ptr[UInt8, MutUntrackedOrigin],
                Int32,
                Ptr[Int32, MutUntrackedOrigin],
                Ptr[Int32, MutUntrackedOrigin],
                Ptr[Int32, MutUntrackedOrigin],
            ) thin -> Int32,
        ]()(
            self._h,
            untracked(Ptr(to=out[0])),
            Int32(len(out)),
            untracked(Ptr(to=w)),
            untracked(Ptr(to=h)),
            untracked(Ptr(to=c)),
        )
        if st == CV_ERR_NO_FRAME:
            return False
        # ⚠ THE SHIM REPORTS THE GEOMETRY BEFORE IT CHECKS CAPACITY, so a
        # too-small buffer is recoverable: adopt what it told us and retry once.
        # This is the live-camera case where the negotiated size differed from
        # the requested one.
        if st == CV_ERR_CAPACITY:
            self.width = Int(w)
            self.height = Int(h)
            self.channels = Int(c)
            out.resize(self.frame_bytes(), 0)
            return self.read(out)
        _check(st, "cap_read")
        self.width = Int(w)
        self.height = Int(h)
        self.channels = Int(c)
        return True

    def close(mut self):
        """Release the device. Idempotent — the shim ignores a stale handle."""
        if self._h == 0:
            return
        try:
            _get_dylib_function[
                lib,
                "mrl_cv_cap_close",
                def(Int) thin -> None,
            ]()(self._h)
        except:
            # Nothing to report to and nothing to retry; dropping the device
            # would be worse than dropping the message.
            pass
        self._h = Int()


# ═══════════════════════════════════════════════════════════════════════════
# the ONE conversion
# ═══════════════════════════════════════════════════════════════════════════


def bgr_hwc_to_rgb_chw(
    src: List[UInt8], mut dst: List[UInt8], width: Int, height: Int
) raises:
    """OpenCV's frame layout to this project's. BGR HWC -> RGB CHW.

    ⚠⚠ ONE FUNCTION, ON PURPOSE. Both halves of this — the channel ORDER and
    the memory LAYOUT — are silent when wrong: no size changes, no error is
    raised, and the image still looks like an image. Inlining it at each call
    site is how two call sites end up disagreeing, so every consumer of a
    captured frame goes through here and this is what the gate pins.
    """
    var n = width * height
    if len(src) < n * 3:
        raise String("bgr_hwc_to_rgb_chw: source holds ") + String(
            len(src)
        ) + " bytes, need " + String(n * 3)
    if len(dst) < n * 3:
        dst.resize(n * 3, 0)
    for i in range(n):
        # src is B,G,R interleaved; dst is R plane, G plane, B plane.
        dst[i] = src[i * 3 + 2]
        dst[n + i] = src[i * 3 + 1]
        dst[2 * n + i] = src[i * 3]
