# +--------------------------------------------------------------------------+ #
# | Writing Parquet — the inverse of `reader.mojo`
# +--------------------------------------------------------------------------+ #
"""Write the column shapes a LeRobot v3 dataset is made of.

    var cols = List[PqColumn]()
    cols.append(pq_list(String("action"), PQ_F32, fixed_len=6))
    cols.append(pq_scalar(String("episode_index"), PQ_I64))

    var w = ParquetWriter(cols^)
    w.add_metadata(String("huggingface"), hf_json)
    for each episode:
        var vals = w.new_values()
        vals[0].push_f64(...)      # 6 per row
        vals[1].push_i64(ep)       # 1 per row
        w.write_row_group(vals, n_rows)
    w.close(path)

## What it supports, and why exactly that

Three column shapes, because a LeRobot v3 dataset contains three (measured
from a real recording with `reader.mojo`, not assumed):

| shape | example | max_def | max_rep |
|---|---|---|---|
| scalar, optional | `episode_index` INT64 | 1 | 0 |
| `list<T>` | `action` FLOAT, `tasks` BYTE_ARRAY | 3 | 1 |
| `list<list<list<T>>>` | `stats/<camera>/mean` DOUBLE | 7 | 3 |

Arrow's `fixed_size_list<float>[6]` is written as an ordinary LIST group — the
fixed width lives in the Arrow schema metadata, not in the Parquet tree — so
`action` and a variable-length `tasks` share one code path.

## Two deliberate simplifications

⚠ **UNCOMPRESSED, NOT SNAPPY.** `snappy.mojo` has a decompressor only, and the
format permits an uncompressed column chunk. The cost is file size — a data
file goes from roughly 814 KB to 1.6 MB — which is nothing beside the 676 MB
of video in the same dataset. Adding a compressor later changes one field.

⚠ **ONE DATA PAGE PER COLUMN CHUNK.** A row group is one episode (a few
hundred rows), so the largest page here is tens of kilobytes; Parquet's usual
~1 MB page target exists to bound the read granularity, and a reader that
seeks to a row group already has it. `_page` would need to loop to change this
and nothing else would.

⚠ **NO NULLS.** Every value is present. The levels still have to be written —
an OPTIONAL column has `max_def == 1` whether or not it uses it — but nothing
here can express a null, and `reader.mojo` refuses to read one back, so the
pair is consistent. A column that needs nulls needs a design, not a flag.

## The part that is genuinely new

The reader DISCARDS repetition levels (`reader.mojo:533`) because it reshapes
by a width it already knows. A writer cannot: the levels are the only record
of where one row's list ends and the next begins. `_levels_for` builds both
streams, and `_rle` encodes them as the RLE/bit-packed hybrid — runs only,
which is always valid and costs a few bytes per row against bit-packing.

⚠ **A REPETITION LEVEL IS "AT WHICH DEPTH DID A NEW ELEMENT START", NOT AN
INDEX.** For `list<list<list<T>>>` iterating `(i, j, k)`: the very first value
is 0, a new `k` is 3, a new `j` is 2, a new `i` is 1. Getting this wrong does
not fail — it produces a file whose values are all present and whose GROUPING
is wrong, so a reader hands back the right numbers in the wrong rows.
"""

from std.memory import bitcast

from .metadata import (
    CODEC_UNCOMPRESSED, PT_BYTE_ARRAY, PT_DOUBLE, PT_FLOAT, PT_INT64,
    REP_OPTIONAL, REP_REPEATED,
)
from .reader import ENC_PLAIN, ENC_RLE, PAGE_DATA
from .rle import bit_width_for
from .thrift import T_BINARY, T_I32, T_STRUCT
from .thrift_write import ThriftWriter
from ..fileio import write_file_atomic


# Column value kinds.
comptime PQ_I64 = 0
comptime PQ_F32 = 1
comptime PQ_F64 = 2
comptime PQ_STR = 3

# parquet.thrift ConvertedType
comptime CT_UTF8 = 0
comptime CT_LIST = 3

comptime PARQUET_VERSION = 2
comptime CREATED_BY = "mojo-rl parquet writer"


def _physical(kind: Int) raises -> Int:
    if kind == PQ_I64:
        return PT_INT64
    if kind == PQ_F32:
        return PT_FLOAT
    if kind == PQ_F64:
        return PT_DOUBLE
    if kind == PQ_STR:
        return PT_BYTE_ARRAY
    raise Error("parquet/write: unknown column kind " + String(kind))


@fieldwise_init
struct PqColumn(Copyable, ImplicitlyCopyable, Movable):
    """One top-level column of the schema."""

    var name: String
    """May contain `/` — LeRobot names columns `stats/action/min`. A slash is
    an ordinary character in a Parquet name; the path separator on the wire is
    the list<string> `path_in_schema`, not a character."""
    var kind: Int
    var depth: Int
    """0 scalar, 1 `list<T>`, 3 `list<list<list<T>>>`."""
    var d0: Int
    """depth 1: elements per row, or 0 for variable (use `counts`).
    depth 3: the outermost fixed dimension."""
    var d1: Int
    var d2: Int

    def max_def(self) -> Int:
        return 1 if self.depth == 0 else (1 + 2 * self.depth)

    def max_rep(self) -> Int:
        return self.depth

    def per_row(self) -> Int:
        """Values per row, or 0 when it varies."""
        if self.depth == 0:
            return 1
        if self.depth == 1:
            return self.d0
        return self.d0 * self.d1 * self.d2


def pq_scalar(var name: String, kind: Int) -> PqColumn:
    return PqColumn(name^, kind, 0, 0, 0, 0)


def pq_list(var name: String, kind: Int, fixed_len: Int = 0) -> PqColumn:
    return PqColumn(name^, kind, 1, fixed_len, 0, 0)


def pq_list3(
    var name: String, kind: Int, d0: Int, d1: Int, d2: Int
) -> PqColumn:
    return PqColumn(name^, kind, 3, d0, d1, d2)


struct PqValues(Movable):
    """Staged values for ONE column of ONE row group."""

    var f: List[Float64]
    var i: List[Int64]
    var b: List[UInt8]
    var off: List[Int]
    """BYTE_ARRAY offsets; `len(off)` is the string count + 1."""
    var counts: List[Int]
    """Per-row element count, for a depth-1 column of variable width."""

    def __init__(out self):
        self.f = List[Float64]()
        self.i = List[Int64]()
        self.b = List[UInt8]()
        self.off = List[Int]()
        self.off.append(0)
        self.counts = List[Int]()

    def __init__(out self, *, deinit move: Self):
        self.f = move.f^
        self.i = move.i^
        self.b = move.b^
        self.off = move.off^
        self.counts = move.counts^

    def push_f64(mut self, v: Float64):
        self.f.append(v)

    def push_i64(mut self, v: Int):
        self.i.append(Int64(v))

    def push_str(mut self, s: String):
        var by = s.as_bytes()
        for k in range(s.byte_length()):
            self.b.append(by[k])
        self.off.append(len(self.b))

    def push_count(mut self, n: Int):
        self.counts.append(n)

    def n_values(self, kind: Int) -> Int:
        if kind == PQ_STR:
            return len(self.off) - 1
        if kind == PQ_I64:
            return len(self.i)
        return len(self.f)


def _rle(ref levels: List[Int32], bit_width: Int) -> List[UInt8]:
    """The RLE/bit-packed hybrid, using RLE runs only.

    Runs only is always a legal encoding of the stream — the decoder chooses
    per run — and level streams here are extremely runny: every definition
    level is the maximum, so a whole page collapses to one run.
    """
    var out = List[UInt8]()
    if bit_width == 0 or len(levels) == 0:
        return out^
    var nbytes = (bit_width + 7) // 8
    var i = 0
    while i < len(levels):
        var v = levels[i]
        var run = 1
        while i + run < len(levels) and levels[i + run] == v:
            run += 1
        # header: uvarint(run << 1), low bit 0 marks an RLE run
        var h = run << 1
        while True:
            var c = h & 0x7F
            h >>= 7
            if h == 0:
                out.append(UInt8(c))
                break
            out.append(UInt8(c | 0x80))
        for k in range(nbytes):
            out.append(UInt8((Int(v) >> (8 * k)) & 0xFF))
        i += run
    return out^


def _levels_for(
    col: PqColumn, ref vals: PqValues, n_rows: Int
) raises -> Tuple[List[Int32], List[Int32]]:
    """Build (repetition, definition) level streams for one column chunk."""
    var rep = List[Int32]()
    var deff = List[Int32]()
    var maxd = Int32(col.max_def())

    if col.depth == 0:
        for _ in range(n_rows):
            deff.append(maxd)
        return (rep^, deff^)

    if col.depth == 1:
        for r in range(n_rows):
            var k = col.d0
            if k == 0:
                if r >= len(vals.counts):
                    raise Error(
                        "parquet/write: column '" + col.name + "' is variable"
                        " width but has no count for row " + String(r)
                    )
                k = vals.counts[r]
            if k == 0:
                # An EMPTY list is not a null: the outer group is present and
                # the repeated group has no elements, which is def == 1.
                rep.append(Int32(0))
                deff.append(Int32(1))
                continue
            for j in range(k):
                rep.append(Int32(0) if j == 0 else Int32(1))
                deff.append(maxd)
        return (rep^, deff^)

    if col.depth == 3:
        if col.d0 <= 0 or col.d1 <= 0 or col.d2 <= 0:
            raise Error(
                "parquet/write: column '" + col.name + "' is depth 3 but has"
                " shape [" + String(col.d0) + "," + String(col.d1) + ","
                + String(col.d2) + "]; every dimension must be positive"
            )
        for _ in range(n_rows):
            for i0 in range(col.d0):
                for i1 in range(col.d1):
                    for i2 in range(col.d2):
                        var r: Int32
                        if i0 == 0 and i1 == 0 and i2 == 0:
                            r = Int32(0)
                        elif i2 > 0:
                            r = Int32(3)
                        elif i1 > 0:
                            r = Int32(2)
                        else:
                            r = Int32(1)
                        rep.append(r)
                        deff.append(maxd)
        return (rep^, deff^)

    raise Error(
        "parquet/write: column '" + col.name + "' has nesting depth "
        + String(col.depth) + "; only 0, 1 and 3 are implemented"
    )


def _plain(col: PqColumn, ref vals: PqValues) raises -> List[UInt8]:
    """PLAIN-encode a column chunk's values."""
    var out = List[UInt8]()
    if col.kind == PQ_I64:
        for i in range(len(vals.i)):
            var bits = bitcast[DType.uint64](vals.i[i])
            for k in range(8):
                out.append(UInt8(Int((bits >> UInt64(8 * k)) & 0xFF)))
    elif col.kind == PQ_F32:
        for i in range(len(vals.f)):
            var bits = bitcast[DType.uint32](Float32(vals.f[i]))
            for k in range(4):
                out.append(UInt8(Int((bits >> UInt32(8 * k)) & 0xFF)))
    elif col.kind == PQ_F64:
        for i in range(len(vals.f)):
            var bits = bitcast[DType.uint64](vals.f[i])
            for k in range(8):
                out.append(UInt8(Int((bits >> UInt64(8 * k)) & 0xFF)))
    elif col.kind == PQ_STR:
        for s in range(len(vals.off) - 1):
            var a = vals.off[s]
            var b = vals.off[s + 1]
            var n = b - a
            for k in range(4):
                out.append(UInt8((n >> (8 * k)) & 0xFF))
            for k in range(a, b):
                out.append(vals.b[k])
    else:
        raise Error("parquet/write: unknown kind " + String(col.kind))
    return out^


@fieldwise_init
struct _ChunkMeta(Copyable, ImplicitlyCopyable, Movable):
    var offset: Int
    var num_values: Int
    var total_size: Int


struct ParquetWriter(Movable):
    """Builds a whole file in memory, then writes it atomically."""

    var cols: List[PqColumn]
    var out: List[UInt8]
    var created_by: String
    var kv_keys: List[String]
    var kv_values: List[String]
    var _rg_rows: List[Int]
    var _chunks: List[_ChunkMeta]
    """Flattened `[row_group][column]`, row-group major."""
    var total_rows: Int
    var closed: Bool

    def __init__(out self, var cols: List[PqColumn]) raises:
        if len(cols) == 0:
            raise Error("parquet/write: a file needs at least one column")
        self.cols = cols^
        self.out = List[UInt8]()
        self.created_by = String(CREATED_BY)
        self.kv_keys = List[String]()
        self.kv_values = List[String]()
        self._rg_rows = List[Int]()
        self._chunks = List[_ChunkMeta]()
        self.total_rows = 0
        self.closed = False
        # The file magic. Every offset recorded below is absolute, so this
        # has to be in the buffer before the first page is appended.
        self.out.append(UInt8(ord("P")))
        self.out.append(UInt8(ord("A")))
        self.out.append(UInt8(ord("R")))
        self.out.append(UInt8(ord("1")))

    def __init__(out self, *, deinit move: Self):
        self.cols = move.cols^
        self.out = move.out^
        self.created_by = move.created_by^
        self.kv_keys = move.kv_keys^
        self.kv_values = move.kv_values^
        self._rg_rows = move._rg_rows^
        self._chunks = move._chunks^
        self.total_rows = move.total_rows
        self.closed = move.closed

    def add_metadata(mut self, var key: String, var value: String):
        """A file-level key/value pair. The Hub's dataset viewer reads the
        Arrow `huggingface` entry out of here."""
        self.kv_keys.append(key^)
        self.kv_values.append(value^)

    def new_values(self) -> List[PqValues]:
        """One empty staging buffer per column, in schema order."""
        var v = List[PqValues]()
        for _ in range(len(self.cols)):
            v.append(PqValues())
        return v^

    # ── one row group ─────────────────────────────────────────────────

    def write_row_group(
        mut self, ref vals: List[PqValues], n_rows: Int
    ) raises:
        if self.closed:
            raise Error("parquet/write: the file is already closed")
        if len(vals) != len(self.cols):
            raise Error(
                "parquet/write: " + String(len(vals)) + " value buffers for "
                + String(len(self.cols)) + " columns"
            )
        if n_rows <= 0:
            raise Error(
                "parquet/write: a row group needs at least one row, got "
                + String(n_rows)
            )

        for c in range(len(self.cols)):
            # ⚠ A COPY, NOT A `ref`. `_page` takes `self` mutably, and a
            # borrow of `self.cols[c]` alive across that call is an alias the
            # compiler rejects. `PqColumn` is six scalars and a name.
            var col = self.cols[c].copy()
            # ⚠ CHECK THE COUNT BEFORE ENCODING. A column with the wrong
            # number of values still produces a structurally valid file whose
            # levels and values disagree, and the reader then reports a
            # confusing "declared N values" much later.
            var expect = col.per_row() * n_rows
            var got = vals[c].n_values(col.kind)
            if col.depth == 1 and col.d0 == 0:
                expect = 0
                for r in range(n_rows):
                    if r >= len(vals[c].counts):
                        raise Error(
                            "parquet/write: column '" + col.name + "' has "
                            + String(len(vals[c].counts)) + " row counts for "
                            + String(n_rows) + " rows"
                        )
                    expect += vals[c].counts[r]
            if got != expect:
                raise Error(
                    "parquet/write: column '" + col.name + "' has "
                    + String(got) + " values, the shape says " + String(expect)
                    + " for " + String(n_rows) + " rows"
                )
            var meta = self._page(col, vals[c], n_rows)
            self._chunks.append(meta)

        self._rg_rows.append(n_rows)
        self.total_rows += n_rows

    def _page(
        mut self, col: PqColumn, ref vals: PqValues, n_rows: Int
    ) raises -> _ChunkMeta:
        """Append one data page (header + body) and report where it landed."""
        var levels = _levels_for(col, vals, n_rows)
        var rep = levels[0].copy()
        var deff = levels[1].copy()
        var values = _plain(col, vals)

        # The page body: rep levels, then def levels, then values. Each level
        # stream on a v1 page is prefixed with its own 4-byte LE length.
        var body = List[UInt8]()
        if col.max_rep() > 0:
            var enc = _rle(rep, bit_width_for(col.max_rep()))
            for k in range(4):
                body.append(UInt8((len(enc) >> (8 * k)) & 0xFF))
            for i in range(len(enc)):
                body.append(enc[i])
        if col.max_def() > 0:
            var enc = _rle(deff, bit_width_for(col.max_def()))
            for k in range(4):
                body.append(UInt8((len(enc) >> (8 * k)) & 0xFF))
            for i in range(len(enc)):
                body.append(enc[i])
        for i in range(len(values)):
            body.append(values[i])

        # ⚠ `num_values` IS THE LEVEL COUNT, NOT THE ROW COUNT. For a list
        # column of width 6 over 528 rows it is 3168. The reader compares it
        # against the values it decoded, so a row count here reads as a
        # truncated column.
        var n_levels = len(deff)

        var h = ThriftWriter()
        h.field_i32(1, PAGE_DATA)
        h.field_i32(2, len(body))
        h.field_i32(3, len(body))  # uncompressed == compressed
        h.field_struct(5)
        h.field_i32(1, n_levels)
        h.field_i32(2, ENC_PLAIN)
        h.field_i32(3, ENC_RLE)  # definition_level_encoding
        h.field_i32(4, ENC_RLE)  # repetition_level_encoding
        h.struct_end()
        h.finish()
        var hdr = h^.take()

        var start = len(self.out)
        for i in range(len(hdr)):
            self.out.append(hdr[i])
        for i in range(len(body)):
            self.out.append(body[i])
        return _ChunkMeta(start, n_levels, len(self.out) - start)

    # ── the footer ────────────────────────────────────────────────────

    def _write_schema(mut self, mut w: ThriftWriter) raises:
        """`FileMetaData.schema`: the tree, flattened depth-first."""
        var n = 1  # the root
        for c in range(len(self.cols)):
            ref col = self.cols[c]
            if col.depth == 0:
                n += 1
            else:
                # ⚠ TWO GROUPS PER LEVEL PLUS ONE LEAF, i.e. `2*depth + 1`.
                # `3*depth` is the same number at depth 1 and wrong at depth 3
                # — it passed every list<T> test and produced a footer that
                # neither this reader nor pyarrow could deserialize, because
                # the declared element count ran the list header past the end
                # of the struct.
                n += 2 * col.depth + 1

        w.field_list(2, T_STRUCT, n)

        # root
        w.struct_begin()
        w.field_binary(4, String("schema"))
        w.field_i32(5, len(self.cols))
        w.struct_end()

        for c in range(len(self.cols)):
            ref col = self.cols[c]
            if col.depth == 0:
                w.struct_begin()
                w.field_i32(1, _physical(col.kind))
                w.field_i32(3, REP_OPTIONAL)
                w.field_binary(4, col.name)
                if col.kind == PQ_STR:
                    w.field_i32(6, CT_UTF8)
                    w.field_struct(10)  # LogicalType
                    w.field_struct(1)  # StringType
                    w.struct_end()
                    w.struct_end()
                w.struct_end()
                continue

            # `d` nested LIST groups, then the leaf.
            for d in range(col.depth):
                # the annotated group: OPTIONAL, one child, ConvertedType LIST
                w.struct_begin()
                w.field_i32(3, REP_OPTIONAL)
                w.field_binary(4, col.name if d == 0 else String("element"))
                w.field_i32(5, 1)
                w.field_i32(6, CT_LIST)
                w.field_struct(10)
                w.field_struct(3)  # ListType
                w.struct_end()
                w.struct_end()
                w.struct_end()

                # the REPEATED `list` group
                w.struct_begin()
                w.field_i32(3, REP_REPEATED)
                w.field_binary(4, String("list"))
                w.field_i32(5, 1)
                w.struct_end()

                if d + 1 < col.depth:
                    continue

                # the leaf `element`
                w.struct_begin()
                w.field_i32(1, _physical(col.kind))
                w.field_i32(3, REP_OPTIONAL)
                w.field_binary(4, String("element"))
                if col.kind == PQ_STR:
                    w.field_i32(6, CT_UTF8)
                    w.field_struct(10)
                    w.field_struct(1)
                    w.struct_end()
                    w.struct_end()
                w.struct_end()

    def _write_row_groups(mut self, mut w: ThriftWriter) raises:
        var ncols = len(self.cols)
        w.field_list(4, T_STRUCT, len(self._rg_rows))
        for g in range(len(self._rg_rows)):
            var total = 0
            for c in range(ncols):
                total += self._chunks[g * ncols + c].total_size

            w.struct_begin()
            w.field_list(1, T_STRUCT, ncols)
            for c in range(ncols):
                ref col = self.cols[c]
                ref ch = self._chunks[g * ncols + c]
                w.struct_begin()
                w.field_i64(2, ch.offset)  # ColumnChunk.file_offset
                w.field_struct(3)  # ColumnMetaData
                w.field_i32(1, _physical(col.kind))
                # encodings: the level encoding and the value encoding
                w.field_list(2, T_I32, 2)
                w.zigzag(ENC_RLE)
                w.zigzag(ENC_PLAIN)
                # ⚠ path_in_schema is a LIST OF NAMES, not a dotted string.
                # `stats/action/min` is ONE name containing slashes.
                var depth = col.depth
                w.field_list(3, T_BINARY, 1 + 2 * depth)
                w.uvarint(col.name.byte_length())
                var nb = col.name.as_bytes()
                for i in range(col.name.byte_length()):
                    w.u8(Int(nb[i]))
                for d in range(depth):
                    w.uvarint(4)
                    w.u8(ord("l"))
                    w.u8(ord("i"))
                    w.u8(ord("s"))
                    w.u8(ord("t"))
                    _ = d
                    var el = String("element")
                    w.uvarint(el.byte_length())
                    var eb = el.as_bytes()
                    for i in range(el.byte_length()):
                        w.u8(Int(eb[i]))
                w.field_i32(4, CODEC_UNCOMPRESSED)
                w.field_i64(5, ch.num_values)
                w.field_i64(6, ch.total_size)
                w.field_i64(7, ch.total_size)
                w.field_i64(9, ch.offset)  # data_page_offset
                w.struct_end()
                w.struct_end()
            w.field_i64(2, total)
            w.field_i64(3, self._rg_rows[g])
            w.struct_end()

    def close(mut self, var path: String) raises -> Int:
        """Write the footer and the file. Returns the byte count."""
        if self.closed:
            raise Error("parquet/write: already closed")
        if len(self._rg_rows) == 0:
            raise Error(
                "parquet/write: refusing to write '" + path + "' with no row"
                " groups — an empty Parquet file reads back as zero rows and"
                " looks like a successful recording of nothing"
            )

        var w = ThriftWriter()
        w.field_i32(1, PARQUET_VERSION)
        self._write_schema(w)
        w.field_i64(3, self.total_rows)
        self._write_row_groups(w)
        if len(self.kv_keys) > 0:
            w.field_list(5, T_STRUCT, len(self.kv_keys))
            for i in range(len(self.kv_keys)):
                w.struct_begin()
                w.field_binary(1, self.kv_keys[i])
                w.field_binary(2, self.kv_values[i])
                w.struct_end()
        w.field_binary(6, self.created_by)
        w.finish()
        var footer = w^.take()

        for i in range(len(footer)):
            self.out.append(footer[i])
        for k in range(4):
            self.out.append(UInt8((len(footer) >> (8 * k)) & 0xFF))
        self.out.append(UInt8(ord("P")))
        self.out.append(UInt8(ord("A")))
        self.out.append(UInt8(ord("R")))
        self.out.append(UInt8(ord("1")))

        var n = len(self.out)
        write_file_atomic(path, self.out)
        self.closed = True
        return n
