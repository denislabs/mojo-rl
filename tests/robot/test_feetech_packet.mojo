# +--------------------------------------------------------------------------+ #
# | Feetech packet codec, gated against the reference SDK's own bytes
# +--------------------------------------------------------------------------+ #
"""Every expected byte string below was CAPTURED from `scservo_sdk` — the same
Python SDK lerobot drives the SO-101 with — by handing its `PacketHandler` a
fake port that records `writePort`. So this is a comparison against the
reference implementation's output, not against my reading of the protocol.

Reproduce the vectors with `tools/soarm/dump_feetech_vectors.py`.

No hardware, no serial port, no shim: this runs in CI. The layer that needs an
arm on the desk is `tools/soarm/so101_mojo_diag.mojo`, gated separately
against `so101_diag.py`.

Run: pixi run mojo run -I . tests/robot/test_feetech_packet.mojo
"""

from std.testing import assert_equal, assert_true, assert_false, TestSuite

from mojo_rl.robot.feetech.packet import (
    build_ping,
    build_read,
    build_write,
    build_sync_read,
    build_sync_write,
    parse_status,
    decode_le,
    encode_le,
    decode_sign_magnitude,
    encode_sign_magnitude,
    error_names,
)


def _assert_bytes[
    mut: Bool, //, o: Origin[mut=mut]
](name: String, got: Span[UInt8, o], n: Int, want: List[Int]) raises:
    assert_equal(n, len(want), name + ": packet length")
    for i in range(n):
        assert_equal(
            Int(got[i]),
            want[i],
            name + ": byte " + String(i),
        )


def test_ping_matches_reference_sdk() raises:
    """The reference SDK's own `PacketHandler(0).ping(port, 1)` bytes."""
    var buf = InlineArray[UInt8, 32](fill=0)
    var s = Span(buf)
    var n = build_ping(1, s)
    _assert_bytes("ping", s, n, [0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB])


def test_read_matches_reference_sdk() raises:
    """Reference `readTxRx(port, id=1, addr=56 Present_Position, len=2)`."""
    var buf = InlineArray[UInt8, 32](fill=0)
    var s = Span(buf)
    var n = build_read(1, 56, 2, s)
    _assert_bytes(
        "read", s, n, [0xFF, 0xFF, 0x01, 0x04, 0x02, 0x38, 0x02, 0xBE]
    )


def test_write_matches_reference_sdk() raises:
    """Reference `write2ByteTxRx(id=3, addr=42 Goal_Position, 2048)` — and
    the 1-byte form, which is the one that arms torque."""
    var buf = InlineArray[UInt8, 32](fill=0)
    var s = Span(buf)
    var n = build_write(3, 42, 2048, 2, s)
    _assert_bytes(
        "write2", s, n, [0xFF, 0xFF, 0x03, 0x05, 0x03, 0x2A, 0x00, 0x08, 0xC2]
    )

    var buf1 = InlineArray[UInt8, 32](fill=0)
    var s1 = Span(buf1)
    var n1 = build_write(2, 40, 1, 1, s1)
    _assert_bytes(
        "write1", s1, n1, [0xFF, 0xFF, 0x02, 0x04, 0x03, 0x28, 0x01, 0xCD]
    )


def test_sync_read_matches_reference_sdk() raises:
    """GroupSyncRead(56, 2) over ids 1..6 — the packet the control loop sends
    every tick."""
    var ids: List[UInt8] = [1, 2, 3, 4, 5, 6]
    var buf = InlineArray[UInt8, 32](fill=0)
    var s = Span(buf)
    var n = build_sync_read(56, 2, Span(ids), s)
    _assert_bytes(
        "sync_read",
        s,
        n,
        [
            0xFF, 0xFF, 0xFE, 0x0A, 0x82, 0x38, 0x02,
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x26,
        ],
    )


def test_sync_write_matches_reference_sdk() raises:
    """GroupSyncWrite(42, 2) over ids 1..6 with distinct values, so a swapped
    id/value pair or a big-endian split cannot pass."""
    var ids: List[UInt8] = [1, 2, 3, 4, 5, 6]
    var vals: List[Int32] = [100, 200, 300, 400, 500, 600]
    var buf = InlineArray[UInt8, 64](fill=0)
    var s = Span(buf)
    var n = build_sync_write(42, 2, Span(ids), Span(vals), s)
    _assert_bytes(
        "sync_write",
        s,
        n,
        [
            0xFF, 0xFF, 0xFE, 0x16, 0x83, 0x2A, 0x02,
            0x01, 0x64, 0x00,
            0x02, 0xC8, 0x00,
            0x03, 0x2C, 0x01,
            0x04, 0x90, 0x01,
            0x05, 0xF4, 0x01,
            0x06, 0x58, 0x02,
            0xEE,
        ],
    )


def test_parse_status_of_a_real_reply() raises:
    """The exact 6 bytes an STS3215 answered a ping with on 2026-08-25."""
    var rx: List[UInt8] = [0xFF, 0xFF, 0x01, 0x02, 0x00, 0xFC]
    var st = parse_status(Span(rx), 0)
    assert_true(st.ok, "ping reply parses")
    assert_equal(Int(st.id), 1)
    assert_equal(Int(st.err), 0)
    assert_equal(st.param_len, 0)
    assert_equal(st.end, 6)


def test_parse_status_resyncs_past_leading_garbage() raises:
    """A stale byte before the header must not shift the whole packet. Two
    junk bytes, then a real 2-param reply carrying 1931 little-endian."""
    var rx: List[UInt8] = [
        0x00, 0x7E, 0xFF, 0xFF, 0x01, 0x04, 0x00, 0x8B, 0x07, 0x68
    ]
    var st = parse_status(Span(rx), 0)
    assert_true(st.ok, "resynced reply parses")
    assert_equal(Int(st.id), 1)
    # 2 junk bytes + FF FF id len err  ->  params start at 7, not 5.
    assert_equal(st.param_at, 7)
    assert_equal(st.param_len, 2)
    assert_equal(decode_le(Span(rx), st.param_at, 2), 1931)
    assert_equal(st.end, 10)


def test_parse_status_rejects_a_bad_checksum() raises:
    """Non-vacuity for the two tests above: the SAME packet with one byte of
    payload flipped must NOT parse. Without this, a parser that ignored the
    checksum would pass every other case here."""
    var rx: List[UInt8] = [0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFC]
    var st = parse_status(Span(rx), 0)
    assert_false(st.ok, "corrupt reply must be rejected")


def test_parse_status_walks_six_concatenated_replies() raises:
    """A sync-read answer is N separate status packets back to back. Walking
    them is where an off-by-one in `end` shows up, so all six ids and all six
    positions are checked."""
    var rx = InlineArray[UInt8, 48](fill=0)
    var s = Span(rx)
    var positions: List[Int] = [1931, 812, 3125, 2901, 2102, 2559]
    for m in range(6):
        var at = m * 8
        s[at] = 0xFF
        s[at + 1] = 0xFF
        s[at + 2] = UInt8(m + 1)
        s[at + 3] = 4
        s[at + 4] = 0
        encode_le(positions[m], 2, s, at + 5)
        var sum = UInt32(0)
        for k in range(at + 2, at + 7):
            sum += UInt32(s[k])
        s[at + 7] = UInt8(~sum & 0xFF)

    var at = 0
    var seen = 0
    while at < len(s):
        var st = parse_status(s, at)
        if not st.ok:
            break
        assert_equal(Int(st.id), seen + 1, "id order")
        assert_equal(
            decode_le(s, st.param_at, 2), positions[seen], "position value"
        )
        at = st.end
        seen += 1
    assert_equal(seen, 6, "all six replies walked")


def test_sign_magnitude_is_not_twos_complement() raises:
    """`Homing_Offset` is 12-bit magnitude + a direction bit at 11. The
    follower's real offsets on 2026-08-25 included -430 and -485; read as
    two's complement those are 3666 and 3611, and every joint angle derived
    from them would be wrong by ~360 degrees."""
    assert_equal(encode_sign_magnitude(-430, 11), 2048 + 430)
    assert_equal(decode_sign_magnitude(2048 + 430, 11), -430)
    assert_equal(encode_sign_magnitude(563, 11), 563)
    assert_equal(decode_sign_magnitude(563, 11), 563)
    assert_equal(decode_sign_magnitude(encode_sign_magnitude(-1, 11), 11), -1)
    assert_equal(decode_sign_magnitude(encode_sign_magnitude(0, 11), 11), 0)


def test_error_names() raises:
    assert_equal(error_names(0), String("ok"))
    assert_equal(error_names(0x24), String("TEMPERATURE|OVERLOAD"))
    assert_equal(error_names(0x01), String("VOLTAGE"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
