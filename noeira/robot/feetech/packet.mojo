# +--------------------------------------------------------------------------+ #
# | Feetech SCS/STS packet codec
# +--------------------------------------------------------------------------+ #
"""The wire format of a Feetech bus servo, and nothing else.

No I/O, no globals, no hardware: every function here maps bytes to bytes, so
the whole codec is gated in CI against byte strings captured from the
reference `scservo_sdk` (`tests/robot/test_feetech_packet.mojo`). That is
deliberate — the layer above needs an arm on the desk to test, this one never
does.

Instruction packet:

    FF FF | id | len | inst | params… | ~checksum

where `len` counts everything after itself *including* the checksum, and the
checksum is the low byte of the one's complement of the sum from `id` through
the last param.

Status packet is the same shape with `inst` replaced by an error byte.

⚠ **STS/SMS are little-endian** (`SCS_END = 0`); the SCS series is big-endian.
Only STS3215 is in scope here, so LE is hard-coded rather than made a
parameter that nothing varies — see `MODEL_PROTOCOL` in lerobot's
`motors/feetech/tables.py`.
"""

comptime HEADER = UInt8(0xFF)
comptime BROADCAST_ID = UInt8(0xFE)
comptime MAX_ID = UInt8(0xFC)

comptime INST_PING = UInt8(0x01)
comptime INST_READ = UInt8(0x02)
comptime INST_WRITE = UInt8(0x03)
comptime INST_REG_WRITE = UInt8(0x04)
comptime INST_ACTION = UInt8(0x05)
comptime INST_SYNC_READ = UInt8(0x82)
comptime INST_SYNC_WRITE = UInt8(0x83)

# Status-byte bits, shared by Status(65), Unloading_Condition(19) and
# LED_Alarm_Condition(20).
comptime ERR_VOLTAGE = UInt8(0x01)
comptime ERR_SENSOR = UInt8(0x02)
comptime ERR_TEMPERATURE = UInt8(0x04)
comptime ERR_CURRENT = UInt8(0x08)
comptime ERR_ANGLE = UInt8(0x10)
comptime ERR_OVERLOAD = UInt8(0x20)


def error_names(err: UInt8) -> String:
    """`0x24 -> "TEMPERATURE|OVERLOAD"`, `0 -> "ok"`."""
    var out = String("")
    if err & ERR_VOLTAGE:
        out += "VOLTAGE|"
    if err & ERR_SENSOR:
        out += "SENSOR|"
    if err & ERR_TEMPERATURE:
        out += "TEMPERATURE|"
    if err & ERR_CURRENT:
        out += "CURRENT|"
    if err & ERR_ANGLE:
        out += "ANGLE|"
    if err & ERR_OVERLOAD:
        out += "OVERLOAD|"
    if out.byte_length() == 0:
        return String("ok")
    return String(out[byte = 0 : out.byte_length() - 1])


# ═══════════════════════════════════════════════════════════════════════════
# scalars
# ═══════════════════════════════════════════════════════════════════════════


def encode_le[o: MutOrigin](
    value: Int, size: Int, buf: Span[UInt8, o], at: Int
) raises:
    """Little-endian, `size` in {1, 2, 4}."""
    if size != 1 and size != 2 and size != 4:
        raise Error("feetech: register width must be 1, 2 or 4")
    for i in range(size):
        buf[at + i] = UInt8((value >> (8 * i)) & 0xFF)


def decode_le[mut: Bool, //, o: Origin[mut=mut]](
    buf: Span[UInt8, o], at: Int, size: Int
) raises -> Int:
    if size != 1 and size != 2 and size != 4:
        raise Error("feetech: register width must be 1, 2 or 4")
    var v = 0
    for i in range(size):
        v |= Int(buf[at + i]) << (8 * i)
    return v


def encode_sign_magnitude(value: Int, sign_bit: Int) raises -> Int:
    """⚠ Feetech signs `Homing_Offset` and `Goal_Speed` with a DIRECTION BIT,
    not two's complement. `Homing_Offset` is 12-bit magnitude + bit 11; read
    it as two's complement and a negative offset becomes a huge positive one.
    """
    var max_mag = (1 << sign_bit) - 1
    var mag = abs(value)
    if mag > max_mag:
        raise Error(
            "feetech: magnitude "
            + String(mag)
            + " exceeds "
            + String(max_mag)
            + " for sign bit "
            + String(sign_bit)
        )
    var dir = 1 if value < 0 else 0
    return (dir << sign_bit) | mag


def decode_sign_magnitude(encoded: Int, sign_bit: Int) -> Int:
    var mag = encoded & ((1 << sign_bit) - 1)
    return -mag if (encoded >> sign_bit) & 1 else mag


# ═══════════════════════════════════════════════════════════════════════════
# instruction packets
# ═══════════════════════════════════════════════════════════════════════════


def _finish[o: MutOrigin](buf: Span[UInt8, o], total: Int) -> Int:
    """Stamp header and checksum on a packet whose body is already written.

    `total` counts the checksum byte, so the sum runs over `[2, total-1)`.
    """
    buf[0] = HEADER
    buf[1] = HEADER
    var s = UInt32(0)
    for i in range(2, total - 1):
        s += UInt32(buf[i])
    buf[total - 1] = UInt8(~s & 0xFF)
    return total


def _require[mut: Bool, //, o: Origin[mut=mut]](
    buf: Span[UInt8, o], need: Int
) raises:
    if len(buf) < need:
        raise Error(
            "feetech: packet needs "
            + String(need)
            + " bytes, buffer holds "
            + String(len(buf))
        )


def build_ping[o: MutOrigin](id: UInt8, buf: Span[UInt8, o]) raises -> Int:
    _require(buf, 6)
    buf[2] = id
    buf[3] = 2
    buf[4] = INST_PING
    return _finish(buf, 6)


def build_read[o: MutOrigin](
    id: UInt8, addr: Int, size: Int, buf: Span[UInt8, o]
) raises -> Int:
    _require(buf, 8)
    buf[2] = id
    buf[3] = 4
    buf[4] = INST_READ
    buf[5] = UInt8(addr)
    buf[6] = UInt8(size)
    return _finish(buf, 8)


def build_write[o: MutOrigin](
    id: UInt8, addr: Int, value: Int, size: Int, buf: Span[UInt8, o]
) raises -> Int:
    var total = 7 + size
    _require(buf, total)
    buf[2] = id
    buf[3] = UInt8(size + 3)
    buf[4] = INST_WRITE
    buf[5] = UInt8(addr)
    encode_le(value, size, buf, 6)
    return _finish(buf, total)


def build_sync_read[
    mut: Bool, //, oi: Origin[mut=mut], ob: MutOrigin
](
    addr: Int, size: Int, ids: Span[UInt8, oi], buf: Span[UInt8, ob]
) raises -> Int:
    """One request, one reply per servo — the whole reason a 6-DoF arm can be
    read at 700+ Hz instead of six round trips' worth."""
    var n = len(ids)
    if n == 0:
        raise Error("feetech: sync_read with no ids")
    var total = 8 + n
    _require(buf, total)
    buf[2] = BROADCAST_ID
    buf[3] = UInt8(n + 4)
    buf[4] = INST_SYNC_READ
    buf[5] = UInt8(addr)
    buf[6] = UInt8(size)
    for i in range(n):
        buf[7 + i] = ids[i]
    return _finish(buf, total)


def build_sync_write[
    mi: Bool, mv: Bool, //, oi: Origin[mut=mi], ov: Origin[mut=mv], ob: MutOrigin
](
    addr: Int,
    size: Int,
    ids: Span[UInt8, oi],
    values: Span[Int32, ov],
    buf: Span[UInt8, ob],
) raises -> Int:
    """Every goal position on the bus in ONE packet. No status replies."""
    var n = len(ids)
    if n == 0:
        raise Error("feetech: sync_write with no ids")
    if len(values) != n:
        raise Error(
            "feetech: sync_write has "
            + String(n)
            + " ids but "
            + String(len(values))
            + " values"
        )
    var total = 8 + n * (size + 1)
    _require(buf, total)
    buf[2] = BROADCAST_ID
    buf[3] = UInt8(n * (size + 1) + 4)
    buf[4] = INST_SYNC_WRITE
    buf[5] = UInt8(addr)
    buf[6] = UInt8(size)
    var at = 7
    for i in range(n):
        buf[at] = ids[i]
        encode_le(Int(values[i]), size, buf, at + 1)
        at += size + 1
    return _finish(buf, total)


# ═══════════════════════════════════════════════════════════════════════════
# status packets
# ═══════════════════════════════════════════════════════════════════════════


@fieldwise_init
struct Status(Copyable, Movable):
    """One parsed status packet, as a view into the caller's receive buffer."""

    var ok: Bool
    var id: UInt8
    var err: UInt8
    var param_at: Int
    """Index of the first parameter byte, or -1."""
    var param_len: Int
    var end: Int
    """Index just past this packet — where to resume scanning."""

    @staticmethod
    def none(at: Int) -> Self:
        return Self(False, 0, 0, -1, 0, at)


def parse_status[mut: Bool, //, o: Origin[mut=mut]](
    buf: Span[UInt8, o], start: Int
) -> Status:
    """Find the next well-formed status packet at or after `start`.

    Resyncs on `FF FF` rather than assuming the reply begins at index 0. Half-
    duplex buses echo the transmitted packet on some driver boards (not on the
    SO-101's, measured — but a stale byte from a previous timeout has the same
    effect), and a parser that trusts the first byte reads the request back as
    a corrupt reply.
    """
    var i = start
    while i + 6 <= len(buf):
        if buf[i] != HEADER or buf[i + 1] != HEADER:
            i += 1
            continue
        var length = Int(buf[i + 3])
        if length < 2:
            i += 1
            continue
        var total = 4 + length
        if i + total > len(buf):
            return Status.none(i)
        var s = UInt32(0)
        for k in range(i + 2, i + total - 1):
            s += UInt32(buf[k])
        if buf[i + total - 1] != UInt8(~s & 0xFF):
            i += 1
            continue
        return Status(
            True,
            buf[i + 2],
            buf[i + 4],
            i + 5,
            length - 2,
            i + total,
        )
    return Status.none(len(buf))


def status_size(param_len: Int) -> Int:
    """Bytes a status packet carrying `param_len` params occupies on the wire.
    """
    return 6 + param_len
