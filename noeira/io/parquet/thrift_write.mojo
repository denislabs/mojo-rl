# +--------------------------------------------------------------------------+ #
# | Thrift compact protocol — the WRITE side
# +--------------------------------------------------------------------------+ #
"""The mirror of `thrift.mojo`'s reader: build a Thrift compact struct.

    var w = ThriftWriter()
    w.field_struct(2)          #   struct field id 2
    w.field_i32(1, 5)          #     an i32 at id 1
    w.field_binary(4, name)    #     a binary at id 4
    w.struct_end()             #   the 0x00 that closes it
    var bytes = w.take()

The wire format is documented in full at the top of `thrift.mojo` and is not
repeated here. What matters on this side are the two pieces of STATE the
reader gets for free and a writer has to maintain.

## Field ids are DELTAS, and the deltas are per struct

A field header packs `(delta << 4) | type`, where `delta` is the id minus the
PREVIOUS field's id in the same struct. So a writer has to remember the last id
it wrote — and reset that memory on entering a nested struct, then restore it
on leaving. `_last` is a stack for exactly that.

⚠ **FIELDS MUST BE WRITTEN IN ASCENDING ID ORDER.** A delta is 4 bits and
unsigned; writing id 3 after id 7 cannot be encoded and produces a header that
decodes as some other field entirely. `_header` raises rather than emitting
one, because the failure is otherwise silent and lands in the reader as a
mis-typed field several structs later.

⚠ **A DELTA OF ZERO IS THE ESCAPE, NOT AN ERROR.** Ids more than 15 apart, and
id 0 itself, are written as a `0x0<type>` byte followed by the id as a zigzag
varint. Parquet's `FileMetaData` has fields at id 1..7 and `ColumnMetaData`
reaches id 15, so both paths are exercised by any real file.

⚠ **A BOOL CARRIES NO PAYLOAD.** Its value lives in the type nibble —
`T_BOOL_TRUE` or `T_BOOL_FALSE`. Writing a body byte after it desynchronises
the struct.
"""

from std.memory import bitcast

from .thrift import (
    T_BINARY, T_BOOL_FALSE, T_BOOL_TRUE, T_DOUBLE, T_I32, T_I64, T_LIST,
    T_STRUCT,
)


struct ThriftWriter(Movable):
    """A growable byte buffer that speaks Thrift compact."""

    var buf: List[UInt8]
    var _last: List[Int]
    """Stack of "last field id written", one entry per open struct. The
    outermost struct is entry 0, pushed at construction."""

    def __init__(out self):
        self.buf = List[UInt8]()
        self._last = List[Int]()
        self._last.append(0)

    def __init__(out self, *, deinit move: Self):
        self.buf = move.buf^
        self._last = move._last^

    def take(deinit self) -> List[UInt8]:
        return self.buf^

    def size(self) -> Int:
        return len(self.buf)

    # ── primitives ────────────────────────────────────────────────────

    def u8(mut self, v: Int):
        self.buf.append(UInt8(v & 0xFF))

    def raw(mut self, ref b: List[UInt8]):
        for i in range(len(b)):
            self.buf.append(b[i])

    def uvarint(mut self, v: Int):
        var u = v
        while True:
            var c = u & 0x7F
            u >>= 7
            if u == 0:
                self.buf.append(UInt8(c))
                return
            self.buf.append(UInt8(c | 0x80))

    def zigzag(mut self, v: Int):
        # ⚠ ARITHMETIC shift on the right operand: `v >> 63` must propagate
        # the sign bit, which is what makes -1 encode as 1 rather than as a
        # ten-byte varint of all ones.
        self.uvarint((v << 1) ^ (v >> 63))

    def le_f64(mut self, v: Float64):
        var bits = bitcast[DType.uint64](v)
        for k in range(8):
            self.buf.append(UInt8(Int((bits >> UInt64(8 * k)) & 0xFF)))

    def le_u32(mut self, v: Int):
        for k in range(4):
            self.buf.append(UInt8((v >> (8 * k)) & 0xFF))

    # ── field headers ─────────────────────────────────────────────────

    def _header(mut self, fid: Int, t: Int) raises:
        var last = self._last[len(self._last) - 1]
        if fid <= last and fid != 0:
            raise Error(
                "thrift: field id " + String(fid) + " written after "
                + String(last) + " — compact deltas are unsigned, so fields"
                " must be emitted in ascending id order"
            )
        var delta = fid - last
        if delta >= 1 and delta <= 15:
            self.u8((delta << 4) | t)
        else:
            self.u8(t)
            self.zigzag(fid)
        self._last[len(self._last) - 1] = fid

    def field_bool(mut self, fid: Int, v: Bool) raises:
        self._header(fid, T_BOOL_TRUE if v else T_BOOL_FALSE)

    def field_i32(mut self, fid: Int, v: Int) raises:
        self._header(fid, T_I32)
        self.zigzag(v)

    def field_i64(mut self, fid: Int, v: Int) raises:
        self._header(fid, T_I64)
        self.zigzag(v)

    def field_double(mut self, fid: Int, v: Float64) raises:
        self._header(fid, T_DOUBLE)
        self.le_f64(v)

    def field_binary(mut self, fid: Int, s: String) raises:
        self._header(fid, T_BINARY)
        self.uvarint(s.byte_length())
        var b = s.as_bytes()
        for i in range(s.byte_length()):
            self.buf.append(b[i])

    def field_binary_bytes(mut self, fid: Int, ref b: List[UInt8]) raises:
        self._header(fid, T_BINARY)
        self.uvarint(len(b))
        for i in range(len(b)):
            self.buf.append(b[i])

    def field_list(mut self, fid: Int, elem_type: Int, size: Int) raises:
        """A list header. The elements follow, written by the caller."""
        self._header(fid, T_LIST)
        if size < 15:
            self.u8((size << 4) | elem_type)
        else:
            self.u8((15 << 4) | elem_type)
            self.uvarint(size)

    # ── structs ───────────────────────────────────────────────────────

    def field_struct(mut self, fid: Int) raises:
        """Open a struct-typed FIELD. Close with `struct_end`."""
        self._header(fid, T_STRUCT)
        self._last.append(0)

    def struct_begin(mut self):
        """Open a struct that is a LIST ELEMENT — it has no field header of
        its own, only its own field-id numbering."""
        self._last.append(0)

    def struct_end(mut self) raises:
        if len(self._last) <= 1:
            raise Error("thrift: struct_end with no open struct")
        self.u8(0)
        _ = self._last.pop()

    def finish(mut self) raises:
        """Close the OUTERMOST struct. Every nested one must already be
        closed — an unbalanced writer otherwise emits a footer that parses as
        a shorter, wrong struct."""
        if len(self._last) != 1:
            raise Error(
                "thrift: " + String(len(self._last) - 1)
                + " struct(s) still open at finish()"
            )
        self.u8(0)
