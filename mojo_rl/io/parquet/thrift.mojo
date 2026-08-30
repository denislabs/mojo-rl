# +--------------------------------------------------------------------------+ #
# | Thrift compact protocol — the encoding Parquet uses for ALL its metadata
# +--------------------------------------------------------------------------+ #
"""A byte cursor plus just enough of the Thrift *compact* protocol to walk a
Parquet footer and its page headers.

Parquet stores every structural fact — the schema, the row groups, each
column chunk's offsets, each page's header — as a Thrift compact struct. There
is no way to read a Parquet file without this, and it is about 150 lines, which
is why this package exists at all rather than binding `libparquet` (C++, name
mangled, no stable ABI).

## The wire format, in full

* **varint** — 7 bits per byte, low bits first, high bit = "more follows".
* **zigzag** — signed integers are varint-encoded as `(n << 1) ^ (n >> 63)`, so
  small negatives stay small. Decode with `(u >> 1) ^ -(u & 1)`.
* **field header** — one byte, `(delta << 4) | type`. A non-zero `delta` is
  added to the previous field id (fields are written in ascending id order); a
  zero `delta` means the id follows as a zigzag varint. A header byte of `0x00`
  ends the struct.
* **types** — 1 `BOOL_TRUE`, 2 `BOOL_FALSE` (the value is IN the type nibble —
  a bool field carries no payload), 3 `I8`, 4 `I16`, 5 `I32`, 6 `I64`,
  7 `DOUBLE`, 8 `BINARY`, 9 `LIST`, 10 `SET`, 11 `MAP`, 12 `STRUCT`.
* **list header** — one byte `(size << 4) | elem_type`; a size of 15 means the
  real size follows as a varint.

⚠ `I8`/`I16`/`I32`/`I64` are ALL zigzag varints in the compact protocol — the
width in the type nibble says how to interpret the decoded value, not how many
bytes to read. Reading a fixed 4 bytes for an `I32` is the classic way to get
this wrong, and it desynchronises the whole rest of the struct rather than
failing where the mistake was made.

`DOUBLE` and the length prefix of `BINARY` are the exceptions: `DOUBLE` is 8
raw little-endian bytes, and `BINARY`'s length is an UNSIGNED varint.
"""

from std.memory import Pointer, bitcast


comptime BPtr = Pointer[Scalar[DType.uint8], MutUntrackedOrigin]
"""Origin-erased byte pointer. `MutUntrackedOrigin` rather than `MutAnyOrigin`
because these are stored in struct fields, where `AnyOrigin` is banned as of
Mojo 1.0 — the owner (`ParquetFile`, which holds the `List[UInt8]`) manages the
lifetime."""


@always_inline
def byte_ptr(mut lst: List[UInt8]) -> BPtr:
    """Erased base pointer of a host byte `List`."""
    return rebind[BPtr](
        lst.unsafe_ptr().unsafe_bitcast[Scalar[DType.uint8]]()
        .as_unsafe_any_origin()
    )


# Thrift compact type nibbles.
comptime T_BOOL_TRUE = 1
comptime T_BOOL_FALSE = 2
comptime T_I8 = 3
comptime T_I16 = 4
comptime T_I32 = 5
comptime T_I64 = 6
comptime T_DOUBLE = 7
comptime T_BINARY = 8
comptime T_LIST = 9
comptime T_SET = 10
comptime T_MAP = 11
comptime T_STRUCT = 12


struct ByteCursor(Copyable, ImplicitlyCopyable, Movable):
    """A bounds-checked read head over a byte buffer.

    Every accessor raises rather than reading out of bounds: a truncated or
    mis-parsed Parquet file otherwise walks off the end of the mapping and the
    process dies with no message worth reading.
    """

    var p: BPtr
    var n: Int
    var pos: Int

    def __init__(out self, p: BPtr, n: Int, pos: Int = 0):
        self.p = p
        self.n = n
        self.pos = pos

    @always_inline
    def _need(self, k: Int) raises:
        if self.pos + k > self.n or self.pos < 0:
            raise Error(
                "parquet: read of " + String(k) + " bytes at offset "
                + String(self.pos) + " runs past the end of a "
                + String(self.n) + "-byte buffer"
            )

    @always_inline
    def u8(mut self) raises -> Int:
        self._need(1)
        var v = Int(self.p[unsafe_offset = self.pos])
        self.pos += 1
        return v

    @always_inline
    def at(self, i: Int) raises -> Int:
        if i < 0 or i >= self.n:
            raise Error("parquet: index " + String(i) + " out of bounds")
        return Int(self.p[unsafe_offset=i])

    def skip_bytes(mut self, k: Int) raises:
        self._need(k)
        self.pos += k

    def uvarint(mut self) raises -> Int:
        """Unsigned LEB128. Capped at 10 bytes — a runaway high bit otherwise
        scans the whole file before failing."""
        var r = 0
        var shift = 0
        for _ in range(10):
            var c = self.u8()
            r |= (c & 0x7F) << shift
            if c < 0x80:
                return r
            shift += 7
        raise Error("parquet: varint longer than 10 bytes (corrupt metadata)")

    @always_inline
    def zigzag(mut self) raises -> Int:
        var u = self.uvarint()
        return (u >> 1) ^ (-(u & 1))

    def le_u32(mut self) raises -> Int:
        self._need(4)
        var r = 0
        for k in range(4):
            r |= Int(self.p[unsafe_offset = self.pos + k]) << (8 * k)
        self.pos += 4
        return r

    def le_f32(mut self) raises -> Float64:
        var bits = UInt32(self.le_u32())
        return Float64(bitcast[DType.float32](bits))

    def le_f64(mut self) raises -> Float64:
        self._need(8)
        var bits = UInt64(0)
        for k in range(8):
            bits |= UInt64(Int(self.p[unsafe_offset = self.pos + k])) << UInt64(8 * k)
        self.pos += 8
        return bitcast[DType.float64](bits)

    def le_i64(mut self) raises -> Int64:
        self._need(8)
        var bits = UInt64(0)
        for k in range(8):
            bits |= UInt64(Int(self.p[unsafe_offset = self.pos + k])) << UInt64(8 * k)
        self.pos += 8
        return bitcast[DType.int64](bits)

    def le_i32(mut self) raises -> Int64:
        return Int64(bitcast[DType.int32](UInt32(self.le_u32())))

    def binary(mut self) raises -> String:
        """A length-prefixed byte string, materialised as a `String`.

        Used only for names (`SchemaElement.name`, `path_in_schema`) which are
        ASCII in every Parquet writer; the bytes are copied verbatim.
        """
        var k = self.uvarint()
        self._need(k)
        var out = String("")
        for i in range(k):
            out += chr(Int(self.p[unsafe_offset = self.pos + i]))
        self.pos += k
        return out^


@fieldwise_init
struct FieldHeader(Copyable, ImplicitlyCopyable, Movable):
    var id: Int
    var type: Int
    var stop: Bool


def read_field_header(mut c: ByteCursor, last_id: Int) raises -> FieldHeader:
    """One field header. `stop` marks the `0x00` byte that ends a struct."""
    var h = c.u8()
    if h == 0:
        return FieldHeader(last_id, 0, True)
    var t = h & 0x0F
    var delta = (h & 0xF0) >> 4
    var fid: Int
    if delta == 0:
        fid = c.zigzag()
    else:
        fid = last_id + delta
    return FieldHeader(fid, t, False)


def skip_field(mut c: ByteCursor, t: Int) raises:
    """Skip a field of type `t` whose header has already been consumed.

    Every struct parser in `metadata.mojo` reads the handful of fields it
    needs and routes everything else here. That is what makes this reader
    tolerant of the fields Parquet keeps adding (column indexes, bloom filter
    offsets, geospatial statistics) instead of failing on a file written by a
    newer Arrow.
    """
    if t == T_BOOL_TRUE or t == T_BOOL_FALSE:
        return  # the value is the type nibble; no payload
    elif t == T_I8 or t == T_I16 or t == T_I32 or t == T_I64:
        _ = c.zigzag()
    elif t == T_DOUBLE:
        c.skip_bytes(8)
    elif t == T_BINARY:
        var k = c.uvarint()
        c.skip_bytes(k)
    elif t == T_STRUCT:
        skip_struct(c)
    elif t == T_LIST or t == T_SET:
        var lh = read_list_header(c)
        for _ in range(lh.size):
            skip_field(c, lh.elem_type)
    elif t == T_MAP:
        var n = c.uvarint()
        if n > 0:
            var kv = c.u8()
            for _ in range(n):
                skip_field(c, (kv & 0xF0) >> 4)
                skip_field(c, kv & 0x0F)
    else:
        raise Error("parquet: unknown thrift type " + String(t))


def skip_struct(mut c: ByteCursor) raises:
    var fid = 0
    while True:
        var h = read_field_header(c, fid)
        if h.stop:
            return
        fid = h.id
        skip_field(c, h.type)


@fieldwise_init
struct ListHeader(Copyable, ImplicitlyCopyable, Movable):
    var elem_type: Int
    var size: Int


def read_list_header(mut c: ByteCursor) raises -> ListHeader:
    """`(size << 4) | elem_type`, with size 15 escaping to a varint."""
    var b = c.u8()
    var et = b & 0x0F
    var n = (b & 0xF0) >> 4
    if n == 15:
        n = c.uvarint()
    return ListHeader(et, n)
