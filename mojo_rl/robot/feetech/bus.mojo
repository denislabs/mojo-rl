# +--------------------------------------------------------------------------+ #
# | Feetech bus — packets over a serial port
# +--------------------------------------------------------------------------+ #
"""`FeetechBus`: request/reply over a half-duplex servo bus.

Owns a `SerialPort` and two fixed buffers, so a control loop that only calls
`sync_read` / `sync_write` allocates nothing per tick.

Measured on the SO-101 follower, 2026-08-25: `sync_read` of all six positions
round-trips in **1.3 ms** (766 Hz), against a control loop that wants 30–50 Hz.
The headroom is what pays for the retries below.
"""

from std.time import perf_counter_ns

from mojo_rl.io.serial import SerialPort
from mojo_rl.robot.feetech.control_table import sign_bit_for
from mojo_rl.robot.feetech.packet import (
    Status,
    build_ping,
    build_read,
    build_sync_read,
    build_sync_write,
    build_write,
    decode_le,
    decode_sign_magnitude,
    encode_sign_magnitude,
    error_names,
    parse_status,
    status_size,
)

comptime MAX_MOTORS = 32
comptime TX_CAP = 8 + MAX_MOTORS * 5
comptime RX_CAP = MAX_MOTORS * 10 + 16

comptime DEFAULT_TIMEOUT_MS = 50
"""`lerobot` patches the PyPI SDK's timeout to `transit + 3 bytes + 50 ms`
(`feetech.py:76`, working around gitee IBY2S6) because the SDK's own formula
is wrong. At 1 Mbaud transit is ~0.01 ms/byte, so 50 ms IS the timeout. It
only costs anything when a packet is genuinely lost — see
`FeetechBus.timeout_ms` for the control loop's tighter setting."""


struct FeetechBus(Movable):
    var port: SerialPort
    var timeout_ms: Int
    """Per-transaction reply deadline. Lower it (5 ms is ample at 1 Mbaud)
    in a control loop, where dropping a tick beats stalling one."""
    var retries: Int
    var _tx: InlineArray[UInt8, TX_CAP]
    var _rx: InlineArray[UInt8, RX_CAP]

    def __init__(
        out self,
        var path: String,
        baud: Int = 1000000,
        timeout_ms: Int = DEFAULT_TIMEOUT_MS,
        retries: Int = 2,
    ) raises:
        self.port = SerialPort(path^, baud)
        self.timeout_ms = timeout_ms
        self.retries = retries
        self._tx = InlineArray[UInt8, TX_CAP](fill=0)
        self._rx = InlineArray[UInt8, RX_CAP](fill=0)

    # ── one servo at a time ────────────────────────────────────────────────

    def _txrx(mut self, n_tx: Int, want_params: Int) raises -> Status:
        """Send `n_tx` bytes, wait for one status packet, return it parsed."""
        self.port.flush()
        var tx = Span(self._tx)
        _ = self.port.write_bytes(tx[0:n_tx])

        var want = status_size(want_params)
        var rx = Span(self._rx)
        var got = self.port.read_bytes(rx, want, self.timeout_ms)
        if got == 0:
            return Status.none(0)
        return parse_status(rx[0:got], 0)

    def ping(mut self, id: UInt8) raises -> Bool:
        for _ in range(self.retries + 1):
            var tx = Span(self._tx)
            var n = build_ping(id, tx)
            var st = self._txrx(n, 0)
            if st.ok and st.id == id:
                return True
        return False

    def read_register(mut self, id: UInt8, addr: Int, size: Int) raises -> Int:
        """Read one register, sign-decoded per `sign_bit_for(addr)`."""
        var raw = self.read_raw(id, addr, size)
        var bit = sign_bit_for(addr)
        return decode_sign_magnitude(raw, bit) if bit != 0 else raw

    def read_raw(mut self, id: UInt8, addr: Int, size: Int) raises -> Int:
        """Read one register with NO sign decoding — what the servo holds.

        Kept separate so a diagnostic can print the stored bits beside the
        interpreted value; a single sign-decoding reader makes a wrong
        `sign_bit_for` entry invisible.
        """
        for _ in range(self.retries + 1):
            var tx = Span(self._tx)
            var n = build_read(id, addr, size, tx)
            var st = self._txrx(n, size)
            if st.ok and st.id == id and st.param_len >= size:
                if st.err != 0:
                    raise Error(
                        "feetech: servo "
                        + String(Int(id))
                        + " reports "
                        + error_names(st.err)
                        + " while reading register "
                        + String(addr)
                    )
                return decode_le(Span(self._rx), st.param_at, size)
        raise Error(
            "feetech: no reply from servo "
            + String(Int(id))
            + " reading register "
            + String(addr)
            + " ("
            + String(size)
            + "B)"
        )

    def write_register(
        mut self, id: UInt8, addr: Int, value: Int, size: Int
    ) raises:
        """Write one register, sign-encoding per `sign_bit_for(addr)`."""
        var bit = sign_bit_for(addr)
        var raw = encode_sign_magnitude(value, bit) if bit != 0 else value
        for _ in range(self.retries + 1):
            var tx = Span(self._tx)
            var n = build_write(id, addr, raw, size, tx)
            var st = self._txrx(n, 0)
            if st.ok and st.id == id:
                if st.err != 0:
                    raise Error(
                        "feetech: servo "
                        + String(Int(id))
                        + " reports "
                        + error_names(st.err)
                        + " while writing register "
                        + String(addr)
                    )
                return
        raise Error(
            "feetech: no reply from servo "
            + String(Int(id))
            + " writing register "
            + String(addr)
        )

    # ── the whole bus at once ──────────────────────────────────────────────

    def sync_read[
        mut: Bool, //, oi: Origin[mut=mut], oo: MutOrigin
    ](
        mut self,
        addr: Int,
        size: Int,
        ids: Span[UInt8, oi],
        dest: Span[Int32, oo],
    ) raises -> Int:
        """Read one register from every id in ONE round trip.

        `dest[i]` is filled for `ids[i]`; entries whose servo did not answer
        are left untouched. Returns how many DID answer, so the caller can
        tell a partial read from a complete one — a loop that silently reused
        last tick's value for a dropped motor would command a stale goal.
        """
        var n = len(ids)
        if n > MAX_MOTORS:
            raise Error(
                "feetech: sync_read over "
                + String(n)
                + " motors exceeds MAX_MOTORS="
                + String(MAX_MOTORS)
            )
        if len(dest) < n:
            raise Error("feetech: sync_read output span is shorter than ids")

        var tx = Span(self._tx)
        var n_tx = build_sync_read(addr, size, ids, tx)
        self.port.flush()
        _ = self.port.write_bytes(tx[0:n_tx])

        var want = n * status_size(size)
        var rx = Span(self._rx)
        var got = self.port.read_bytes(rx, want, self.timeout_ms)
        if got == 0:
            return 0

        var bit = sign_bit_for(addr)
        var at = 0
        var ok = 0
        while at < got:
            var st = parse_status(rx[0:got], at)
            if not st.ok:
                break
            at = st.end
            if st.param_len < size:
                continue
            var raw = decode_le(rx, st.param_at, size)
            var value = decode_sign_magnitude(raw, bit) if bit != 0 else raw
            for i in range(n):
                if ids[i] == st.id:
                    dest[i] = Int32(value)
                    ok += 1
                    break
        return ok

    def sync_write[
        mi: Bool, mv: Bool, //, oi: Origin[mut=mi], ov: Origin[mut=mv]
    ](
        mut self,
        addr: Int,
        size: Int,
        ids: Span[UInt8, oi],
        values: Span[Int32, ov],
    ) raises:
        """Write one register on every id in ONE packet.

        ⚠ Sync-write is FIRE AND FORGET — the servos send no status packet, so
        there is nothing to check and nothing to retry. A goal that never
        arrives is invisible here; confirm motion with a `sync_read`, not with
        this call's return.
        """
        var n = len(ids)
        if n > MAX_MOTORS:
            raise Error(
                "feetech: sync_write over "
                + String(n)
                + " motors exceeds MAX_MOTORS="
                + String(MAX_MOTORS)
            )
        var bit = sign_bit_for(addr)
        var encoded = InlineArray[Int32, MAX_MOTORS](fill=0)
        for i in range(n):
            var v = Int(values[i])
            encoded[i] = Int32(
                encode_sign_magnitude(v, bit) if bit != 0 else v
            )
        var tx = Span(self._tx)
        var n_tx = build_sync_write(
            addr, size, ids, Span(encoded)[0:n], tx
        )
        _ = self.port.write_bytes(tx[0:n_tx])
