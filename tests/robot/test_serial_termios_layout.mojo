# +--------------------------------------------------------------------------+ #
# | `struct termios` offsets, checked against libc — with no hardware
# +--------------------------------------------------------------------------+ #
"""`SerialPort` hardcodes each platform's `struct termios` layout. A wrong
offset is SILENT: it writes into `c_cc` instead of `c_cflag`, the port still
opens, and the failure surfaces hundreds of lines later as garbled bytes.

⚠ RUNS ON BOTH PLATFORMS NOW. It was Darwin-only because `port.mojo` was; the
Linux table (32-bit `tcflag_t`, NCCS=32, VMIN/VTIME swapped, Bxxx as ordinals)
was written from the glibc headers and has to be gated on a Linux box, which
is exactly what this does when run there.

This gates the layout without an arm, a USB adapter, or even a serial device,
by opening a pseudo-terminal (`openpty`, which is not variadic and so is
reachable from `external_call`) and cross-checking our offsets against libc's
own accessors. ⚠ It is the SLAVE fd that is the tty — `tcgetattr` on a pty
master returns ENOTTY on macOS.

* a **canary** past the end of the buffer proves `TERMIOS_SIZE` is not too
  small for what `tcgetattr` writes;
* `cfgetospeed` / `cfgetispeed` — libc reading the struct its own way — must
  agree with the value we read at `OFF_OSPEED` / `OFF_ISPEED`. If those
  offsets were wrong, the two would disagree.

The second check is what makes this non-vacuous: a purely self-consistent
test (write at offset X, read back at offset X) passes for EVERY offset.

Run: pixi run mojo run -I . tests/robot/test_serial_termios_layout.mojo
"""

from std.ffi import external_call
from std.sys import CompilationTarget
from std.testing import assert_equal, assert_true, TestSuite

from noeira.io.serial.native import (
    baud_constant, layout_from_headers, layout_names,
)
from noeira.io.serial.port import (
    OFF_CC,
    OFF_ISPEED,
    OFF_OSPEED,
    O_NOCTTY,
    O_RDWR,
    TCSANOW,
    TERMIOS_SIZE,
    VMIN,
    VTIME,
    assert_layout,
    expected_layout,
    raw_speed_at,
)


def test_constants_match_this_platforms_headers() raises:
    """⚠⚠ THE ONE THAT MAKES THE LINUX PORT HONEST. Every offset and flag in
    `port.mojo` against what `<termios.h>` says on the machine running this,
    through a C `offsetof` probe in the shim. No pty, no tty, no hardware.
    """
    var actual = layout_from_headers()
    var want = expected_layout()
    var names = layout_names()
    for i in range(len(want)):
        assert_equal(actual[i], want[i], names[i])
    # And the aggregate, which is what `SerialPort.__init__` actually calls.
    assert_layout()

comptime CANARY = UInt8(0xA5)


def _open_pty(mut master: Int32, mut slave: Int32) -> Int32:
    """A pseudo-terminal pair — a real tty with no device attached.

    `openpty(&m, &s, NULL, NULL, NULL)`; the SLAVE is the tty that answers
    `tcgetattr`.
    """
    return external_call["openpty", Int32](
        Pointer(to=master),
        Pointer(to=slave),
        # Mojo's `Pointer` is non-nullable by construction, so the three
        # optional `openpty` arguments are passed as a plain zero word —
        # which is what a NULL pointer is in the C ABI.
        Int(0),
        Int(0),
        Int(0),
    )


def test_termios_size_and_speed_offsets_agree_with_libc() raises:
    var master = Int32(-1)
    var fd = Int32(-1)
    assert_equal(Int(_open_pty(master, fd)), 0, "openpty")
    assert_true(fd >= 0, "openpty gave a slave fd")

    # 16 canary bytes past the struct: tcgetattr must not touch them.
    var buf = Array[UInt8, TERMIOS_SIZE + 16](fill=CANARY)
    var p = buf.unsafe_ptr()
    assert_equal(
        Int(external_call["tcgetattr", Int32](fd, p)), 0, "tcgetattr on pty"
    )
    for i in range(TERMIOS_SIZE, TERMIOS_SIZE + 16):
        assert_equal(
            Int(buf[i]),
            Int(CANARY),
            (
                "tcgetattr wrote past TERMIOS_SIZE — the struct is LARGER than"
                " "
                + String(TERMIOS_SIZE)
                + " bytes on this platform"
            ),
        )

    # Set a speed through libc's accessor, read it back through OUR offset.
    #
    # ⚠ THE ARGUMENT IS THE Bxxx CONSTANT, NOT THE BAUD. On BSD they are the
    # same number; on glibc `cfsetospeed(&t, 9600)` is EINVAL because B9600 is
    # 13. Asking the shim keeps one spelling for both.
    for baud in [Int(9600), Int(19200), Int(115200)]:
        var code = baud_constant(baud)
        assert_true(code >= 0, "this libc spells " + String(baud) + " baud")
        _ = external_call["cfsetospeed", Int32](p, UInt32(code))
        _ = external_call["cfsetispeed", Int32](p, UInt32(code))
        assert_equal(
            Int(external_call["tcsetattr", Int32](fd, Int32(TCSANOW), p)),
            0,
            "tcsetattr at " + String(baud),
        )
        _ = external_call["tcgetattr", Int32](fd, p)

        var ours_out = raw_speed_at(p.as_unsafe_any_origin(), OFF_OSPEED)
        var ours_in = raw_speed_at(p.as_unsafe_any_origin(), OFF_ISPEED)
        var libc_out = Int(external_call["cfgetospeed", UInt32](p))
        var libc_in = Int(external_call["cfgetispeed", UInt32](p))

        assert_equal(ours_out, libc_out, "OFF_OSPEED vs cfgetospeed")
        assert_equal(ours_in, libc_in, "OFF_ISPEED vs cfgetispeed")
        assert_equal(libc_out, code, "the pty took the speed we asked for")

    _ = external_call["close", Int32](fd)
    _ = external_call["close", Int32](master)


def test_c_cc_offset_round_trips_through_the_driver() raises:
    """VMIN/VTIME survive a tcsetattr/tcgetattr round trip at our offsets.

    Weaker than the speed check above — it cannot tell OFF_CC from any other
    writable offset in the struct — but it does catch an offset that lands
    outside the part of the struct the driver preserves.
    """
    var master = Int32(-1)
    var fd = Int32(-1)
    assert_equal(Int(_open_pty(master, fd)), 0, "openpty")
    var buf = Array[UInt8, TERMIOS_SIZE](fill=0)
    var p = buf.unsafe_ptr()
    _ = external_call["tcgetattr", Int32](fd, p)

    p[unsafe_offset = OFF_CC + VMIN] = 7
    p[unsafe_offset = OFF_CC + VTIME] = 3
    assert_equal(
        Int(external_call["tcsetattr", Int32](fd, Int32(TCSANOW), p)), 0
    )

    var back = Array[UInt8, TERMIOS_SIZE](fill=0)
    var q = back.unsafe_ptr()
    _ = external_call["tcgetattr", Int32](fd, q)
    assert_equal(Int(q[unsafe_offset = OFF_CC + VMIN]), 7, "c_cc[VMIN]")
    assert_equal(Int(q[unsafe_offset = OFF_CC + VTIME]), 3, "c_cc[VTIME]")

    _ = external_call["close", Int32](fd)
    _ = external_call["close", Int32](master)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
