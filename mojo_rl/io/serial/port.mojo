# +--------------------------------------------------------------------------+ #
# | mojo-rl serial — a raw-mode tty over libc
# +--------------------------------------------------------------------------+ #
"""`SerialPort`: open a tty, put it in 8N1 raw mode at an arbitrary baud, and
read/write bytes against a deadline.

Deliberately knows nothing about servos. The Feetech packet layer sits on top
in `mojo_rl/robot/feetech/`, and the same fd plumbing is what a native socket
sink would reuse (`docs/RERUN_ASSESSMENT.md` §5.5).

Two facts that cost an afternoon each, both measured 2026-08-25 on Darwin
25.5 / arm64 against a real SO-101 bus:

* **Use `/dev/cu.*`, not `/dev/tty.*`.** The `tty.` callin device blocks in
  `open` waiting for carrier detect. `cu.` is the callout device and does not.
  (pyserial hides this by always passing `O_NONBLOCK`; so do we, but the
  device choice still matters for anything that later clears the flag.)
* **`tcsetattr` REJECTS a literal 1 000 000 with EINVAL** on macOS, even
  though BSD's `Bxxx` constants *are* the baud numbers (`B230400 == 230400`).
  Only `IOSSIOSPEED` gets there — see `native.mojo` for why that needs C.
"""

from std.ffi import external_call
from std.sys import CompilationTarget
from std.time import perf_counter_ns

from mojo_rl.io.serial.native import set_speed

# ═══════════════════════════════════════════════════════════════════════════
# libc constants — Darwin/arm64
# ═══════════════════════════════════════════════════════════════════════════

comptime AT_FDCWD = -2
comptime O_RDWR = 2
comptime O_NOCTTY = 131072
comptime O_NONBLOCK = 4

comptime TCSANOW = 0
comptime TCIFLUSH = 1
comptime TCOFLUSH = 2
comptime TCIOFLUSH = 3

# `struct termios`, verified by a C `offsetof` probe rather than assumed:
#   4 x tcflag_t(8) | cc_t[NCCS=20] | 4 pad | speed_t c_ispeed | c_ospeed
# = 72 bytes. A wrong offset here is silent — it writes into c_cc and the
# port merely misbehaves — so the layout is asserted against a live tty by
# `tests/robot/test_serial_termios_layout.mojo`.
comptime TERMIOS_SIZE = 72
comptime OFF_IFLAG = 0
comptime OFF_OFLAG = 8
comptime OFF_CFLAG = 16
comptime OFF_LFLAG = 24
comptime OFF_CC = 32
comptime OFF_ISPEED = 56
comptime OFF_OSPEED = 64
comptime VMIN = 16
comptime VTIME = 17

comptime CSIZE = 0x300
comptime CS8 = 0x300
comptime CLOCAL = 0x8000
comptime CREAD = 0x800
comptime PARENB = 0x1000
comptime CSTOPB = 0x400
comptime CRTSCTS = 0x30000

comptime EINTR = 4
comptime EAGAIN = 35


def errno() -> Int32:
    return external_call["__error", Pointer[Int32, MutAnyOrigin]]()[]


# ═══════════════════════════════════════════════════════════════════════════
# SerialPort
# ═══════════════════════════════════════════════════════════════════════════


struct SerialPort(Movable):
    """An open tty in raw mode. Closes itself when it goes out of scope."""

    var fd: Int32
    var baud: Int
    var _path: String

    def __init__(out self, var path: String, baud: Int = 1000000) raises:
        """Open and configure. Raises rather than returning a bad fd, so a
        caller that got a `SerialPort` has a usable one."""
        comptime assert CompilationTarget.is_macos(), (
            "SerialPort's `struct termios` offsets are Darwin/arm64 only."
            " Linux has a DIFFERENT layout (32-bit tcflag_t, NCCS=32) and"
            " spells 1 Mbaud B1000000 in termios with no ioctl at all."
            " Fill in the layout and drop this assert when a Linux box is"
            " available to verify it against — do not guess the offsets."
        )

        self._path = path^
        self.baud = baud
        # ⚠ `openat`, NOT `open`, and the reason is a Mojo linking trap.
        # `external_call` re-declares a C symbol per module, and a SECOND
        # declaration of the same symbol with a different signature fails at
        # LLVM lowering — "existing function with conflicting signature". The
        # stdlib already declares `open` (`std/io/file.mojo:141`), and this
        # module lands in the same binary as it whenever anything touches the
        # filesystem, which the viewer does. Matching its signature exactly
        # did NOT resolve it. `openat` is the same call with an explicit
        # directory fd, nothing else declares it, and AT_FDCWD makes it
        # identical to `open` for an absolute path.
        #
        # The mode argument is harmless whichever way it travels: C `openat`
        # is variadic, so on Apple arm64 it lands in a register the callee
        # will not read — and the kernel only reads `mode` under O_CREAT,
        # which is never set here.
        self.fd = external_call["openat", Int32](
            Int32(AT_FDCWD),
            self._path.as_c_string_slice().unsafe_ptr(),
            Int32(O_RDWR | O_NOCTTY | O_NONBLOCK),
            Int32(0),
        )
        if self.fd < 0:
            var e = errno()
            raise Error(
                "serial: open("
                + self._path
                + ") failed, errno="
                + String(e)
                + (
                    "  (is the arm plugged in? prefer /dev/cu.* over"
                    " /dev/tty.*)"
                )
            )
        try:
            self._configure()
        except e:
            _ = external_call["close", Int32](self.fd)
            self.fd = -1
            raise e

    def __deinit__(deinit self):
        if self.fd >= 0:
            _ = external_call["close", Int32](self.fd)

    def _configure(mut self) raises:
        var tio = InlineArray[UInt8, TERMIOS_SIZE](fill=0)
        var p = tio.unsafe_ptr()
        if external_call["tcgetattr", Int32](self.fd, p) != 0:
            raise Error("serial: tcgetattr failed, errno=" + String(errno()))

        external_call["cfmakeraw", NoneType](p)
        var cflag = p.unsafe_offset(OFF_CFLAG).unsafe_bitcast[UInt64]()
        cflag[] = (cflag[] & ~UInt64(CSIZE)) | UInt64(CS8 | CLOCAL | CREAD)
        cflag[] = cflag[] & ~UInt64(PARENB | CSTOPB | CRTSCTS)
        # Fully non-blocking: `read` returns whatever is there, right now.
        # The deadline lives in `read_bytes`, not in the driver, so one
        # timeout policy governs the whole stack.
        p[unsafe_offset = OFF_CC + VMIN] = 0
        p[unsafe_offset = OFF_CC + VTIME] = 0

        # Park at a speed termios accepts; IOSSIOSPEED sets the real one.
        _ = external_call["cfsetispeed", Int32](p, UInt64(9600))
        _ = external_call["cfsetospeed", Int32](p, UInt64(9600))
        if external_call["tcsetattr", Int32](self.fd, Int32(TCSANOW), p) != 0:
            raise Error("serial: tcsetattr failed, errno=" + String(errno()))

        if set_speed(self.fd, self.baud) != 0:
            raise Error(
                "serial: could not set "
                + String(self.baud)
                + " baud, errno="
                + String(errno())
            )

        # Non-vacuity: confirm the driver took the speed rather than trusting
        # a 0 return. A shim that silently no-ops would otherwise leave the
        # port at 9600 and every read would time out with no explanation.
        if self.speed() != self.baud:
            raise Error(
                "serial: baud readback is "
                + String(self.speed())
                + ", asked for "
                + String(self.baud)
            )

    def speed(mut self) -> Int:
        """`c_ospeed` as the driver currently reports it."""
        var tio = InlineArray[UInt8, TERMIOS_SIZE](fill=0)
        var p = tio.unsafe_ptr()
        _ = external_call["tcgetattr", Int32](self.fd, p)
        return Int(p.unsafe_offset(OFF_OSPEED).unsafe_bitcast[UInt64]()[])

    def flush(mut self):
        """Discard everything queued in BOTH directions.

        Call before a request whose reply you intend to parse: a half-read
        packet from a previous timeout would otherwise be mistaken for this
        one's header.
        """
        _ = external_call["tcflush", Int32](self.fd, Int32(TCIOFLUSH))

    def write_bytes[
        mut: Bool, //, o: Origin[mut=mut]
    ](mut self, buf: Span[UInt8, o]) raises -> Int:
        var n = external_call["write", Int](
            Int(self.fd), buf.unsafe_ptr(), len(buf)
        )
        if n < 0:
            raise Error("serial: write failed, errno=" + String(errno()))
        return n

    def read_bytes[
        o: MutOrigin
    ](
        mut self, buf: Span[UInt8, o], want: Int, timeout_ms: Int
    ) raises -> Int:
        """Fill up to `want` bytes, returning early only on the deadline.

        Returns how many arrived — a short read is normal and is the caller's
        to interpret, because only the packet layer knows how long the reply
        should have been.
        """
        if want > len(buf):
            raise Error(
                "serial: read_bytes wants "
                + String(want)
                + " into a "
                + String(len(buf))
                + "-byte span"
            )
        var deadline = perf_counter_ns() + timeout_ms * 1_000_000
        var got = 0
        while got < want:
            # ⚠ `Int(self.fd)`, not `self.fd` — same collision as `write`
            # above: the stdlib declares `read` with an `index` fd
            # (`std/io/file_descriptor.mojo:117`) and a second declaration
            # with `si32` fails at LLVM lowering, not at parse.
            var n = external_call["read", Int](
                Int(self.fd), buf.unsafe_ptr().unsafe_offset(got), want - got
            )
            if n > 0:
                got += n
                continue
            if n < 0:
                var e = errno()
                if e != EAGAIN and e != EINTR:
                    raise Error("serial: read failed, errno=" + String(e))
            if perf_counter_ns() >= deadline:
                break
            # 100 us: two orders of magnitude under a 1 Mbaud packet's own
            # transit time, so the poll never dominates the round trip.
            _ = external_call["usleep", Int32](UInt32(100))
        return got
