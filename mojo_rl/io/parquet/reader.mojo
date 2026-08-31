# +--------------------------------------------------------------------------+ #
# | ParquetFile — open a .parquet and read whole leaf columns
# +--------------------------------------------------------------------------+ #
"""The reader proper: footer -> row groups -> pages -> values.

    var f = ParquetFile(path)
    var ts = f.read_f64("videos/observation.images.front/from_timestamp")
    var ep = f.read_i64("episode_index")
    var act = f.read_f64("action.list.element")   # flat; reshape by list width

## Deliberate scope

This reads what Arrow writes for a LeRobot dataset and refuses everything else
BY NAME. Supported: `PLAIN`, `PLAIN_DICTIONARY`, `RLE_DICTIONARY` values over
`FLOAT`/`DOUBLE`/`INT32`/`INT64`/`BOOLEAN`, `RLE` levels, data page v1 and v2,
`UNCOMPRESSED` and `SNAPPY`.

Not supported, and each raises a message naming the encoding rather than
returning wrong numbers: `DELTA_*`, `BYTE_STREAM_SPLIT`, the deprecated
MSB-first `BIT_PACKED` level encoding, `GZIP`/`ZSTD`/`LZ4`/`BROTLI`, `INT96`,
byte arrays, and null values.

⚠ **NULLS ARE REJECTED, NOT SKIPPED.** A column here yields a dense value
array with no null mask, so a page whose definition levels report a null has
no honest representation. Arrow marks these columns `optional` even when it
writes no nulls (`stats_null = 0` throughout a LeRobot file), so the common
case is fine — but a file that did carry a null would otherwise silently
shorten the column and shift every row after it. The check is per page.

⚠ THE WHOLE FILE IS READ INTO MEMORY. LeRobot's metadata parquets are under a
megabyte and its data parquets a few; `MAX_FILE_BYTES` turns anything larger
into an error naming the limit rather than a swap storm. This reader is for
metadata and small tables — bulk sample data belongs in a `TrajectoryStore`.
"""

from .metadata import (
    CODEC_SNAPPY, CODEC_UNCOMPRESSED, ColumnChunkMeta, FileMetaData, LeafInfo,
    PT_BYTE_ARRAY,
    PT_BOOLEAN,
    PT_DOUBLE, PT_FLOAT, PT_INT32, PT_INT64, codec_name, parse_file_metadata,
    physical_type_name,
)
from .rle import bit_width_for, rle_decode
from .snappy import snappy_decompress
from .thrift import (
    BPtr, ByteCursor, byte_ptr, read_field_header, skip_field,
)


comptime MAX_FILE_BYTES = 512 * 1024 * 1024

# parquet.thrift `PageType`
comptime PAGE_DATA = 0
comptime PAGE_INDEX = 1
comptime PAGE_DICTIONARY = 2
comptime PAGE_DATA_V2 = 3

# parquet.thrift `Encoding`
comptime ENC_PLAIN = 0
comptime ENC_PLAIN_DICTIONARY = 2
comptime ENC_RLE = 3
comptime ENC_BIT_PACKED = 4
comptime ENC_DELTA_BINARY_PACKED = 5
comptime ENC_DELTA_LENGTH_BYTE_ARRAY = 6
comptime ENC_DELTA_BYTE_ARRAY = 7
comptime ENC_RLE_DICTIONARY = 8
comptime ENC_BYTE_STREAM_SPLIT = 9


def encoding_name(e: Int) -> String:
    if e == ENC_PLAIN: return String("PLAIN")
    if e == ENC_PLAIN_DICTIONARY: return String("PLAIN_DICTIONARY")
    if e == ENC_RLE: return String("RLE")
    if e == ENC_BIT_PACKED: return String("BIT_PACKED")
    if e == ENC_DELTA_BINARY_PACKED: return String("DELTA_BINARY_PACKED")
    if e == ENC_DELTA_LENGTH_BYTE_ARRAY: return String("DELTA_LENGTH_BYTE_ARRAY")
    if e == ENC_DELTA_BYTE_ARRAY: return String("DELTA_BYTE_ARRAY")
    if e == ENC_RLE_DICTIONARY: return String("RLE_DICTIONARY")
    if e == ENC_BYTE_STREAM_SPLIT: return String("BYTE_STREAM_SPLIT")
    return String("encoding#") + String(e)


@fieldwise_init
struct PageHeader(Copyable, ImplicitlyCopyable, Movable):
    var page_type: Int
    var uncompressed_size: Int
    var compressed_size: Int
    var num_values: Int
    var encoding: Int
    var def_encoding: Int
    var rep_encoding: Int
    var num_nulls: Int
    """v2 only; -1 on a v1 page."""
    var def_bytes: Int
    """v2 only: uncompressed definition-level byte count."""
    var rep_bytes: Int
    """v2 only: uncompressed repetition-level byte count."""
    var v2_compressed: Bool


def _read_page_header(mut c: ByteCursor) raises -> PageHeader:
    var h = PageHeader(-1, 0, 0, 0, ENC_PLAIN, ENC_RLE, ENC_RLE, -1, 0, 0, True)
    var fid = 0
    while True:
        var f = read_field_header(c, fid)
        if f.stop:
            break
        fid = f.id
        if f.id == 1:
            h.page_type = c.zigzag()
        elif f.id == 2:
            h.uncompressed_size = c.zigzag()
        elif f.id == 3:
            h.compressed_size = c.zigzag()
        elif f.id == 5:
            # DataPageHeader (v1)
            var g = 0
            while True:
                var q = read_field_header(c, g)
                if q.stop:
                    break
                g = q.id
                if q.id == 1:
                    h.num_values = c.zigzag()
                elif q.id == 2:
                    h.encoding = c.zigzag()
                elif q.id == 3:
                    h.def_encoding = c.zigzag()
                elif q.id == 4:
                    h.rep_encoding = c.zigzag()
                else:
                    skip_field(c, q.type)
        elif f.id == 7:
            # DictionaryPageHeader
            var g2 = 0
            while True:
                var q = read_field_header(c, g2)
                if q.stop:
                    break
                g2 = q.id
                if q.id == 1:
                    h.num_values = c.zigzag()
                elif q.id == 2:
                    h.encoding = c.zigzag()
                else:
                    skip_field(c, q.type)
        elif f.id == 8:
            # DataPageHeaderV2
            var g3 = 0
            while True:
                var q = read_field_header(c, g3)
                if q.stop:
                    break
                g3 = q.id
                if q.id == 1:
                    h.num_values = c.zigzag()
                elif q.id == 2:
                    h.num_nulls = c.zigzag()
                elif q.id == 4:
                    h.encoding = c.zigzag()
                elif q.id == 5:
                    h.def_bytes = c.zigzag()
                elif q.id == 6:
                    h.rep_bytes = c.zigzag()
                elif q.id == 7:
                    # `is_compressed`, default TRUE — a bool field carries its
                    # value in the thrift type nibble (1 = true, 2 = false).
                    h.v2_compressed = q.type == 1
                else:
                    skip_field(c, q.type)
        else:
            skip_field(c, f.type)
    return h^


def _plain_values(
    mut c: ByteCursor,
    physical_type: Int,
    count: Int,
    mut out_f: List[Float64],
    mut out_i: List[Int64],
    mut out_b: List[UInt8],
    mut out_off: List[Int],
) raises:
    """PLAIN: fixed-width little-endian values, back to back.

    ⚠ BYTE_ARRAY IS THE EXCEPTION: each value is a 4-byte little-endian LENGTH
    followed by that many bytes, so the values are not fixed-width and cannot
    be indexed without walking them. They land concatenated in `out_b` with
    `out_off` marking the boundaries — value `k` is
    `out_b[out_off[k] : out_off[k + 1]]`, which is why `out_off` always holds
    one more entry than there are values.
    """
    if physical_type == PT_FLOAT:
        for _ in range(count):
            out_f.append(c.le_f32())
    elif physical_type == PT_DOUBLE:
        for _ in range(count):
            out_f.append(c.le_f64())
    elif physical_type == PT_INT32:
        for _ in range(count):
            out_i.append(c.le_i32())
    elif physical_type == PT_INT64:
        for _ in range(count):
            out_i.append(c.le_i64())
    elif physical_type == PT_BOOLEAN:
        # 1 bit per value, LSB-first within each byte.
        var bits = List[Int32]()
        rle_bits_plain(c, count, bits)
        for i in range(count):
            out_i.append(Int64(Int(bits[i])))
    elif physical_type == PT_BYTE_ARRAY:
        if len(out_off) == 0:
            out_off.append(0)
        for _ in range(count):
            var n = Int(c.le_u32())
            var start = c.pos
            c.skip_bytes(n)
            for k in range(n):
                out_b.append(UInt8(c.at(start + k)))
            out_off.append(len(out_b))
    else:
        raise Error(
            "parquet: PLAIN decoding of " + physical_type_name(physical_type)
            + " is not implemented"
        )


def rle_bits_plain(mut c: ByteCursor, count: Int, mut out: List[Int32]) raises:
    """PLAIN booleans: a raw LSB-first bitmap, NOT the RLE hybrid."""
    var nbytes = (count + 7) // 8
    var start = c.pos
    c.skip_bytes(nbytes)
    for i in range(count):
        var b = c.at(start + (i >> 3))
        out.append(Int32((b >> (i & 7)) & 1))


struct ParquetFile(Movable):
    """A whole `.parquet` resident in memory, with its footer parsed."""

    var bytes: List[UInt8]
    var meta: FileMetaData
    var path: String

    def __init__(out self, var path: String) raises:
        var f = open(path, "r")
        self.bytes = f.read_bytes()
        f.close()
        self.path = path^

        var n = len(self.bytes)
        if n > MAX_FILE_BYTES:
            raise Error(
                "parquet: '" + self.path + "' is " + String(n) + " bytes,"
                " over this reader's " + String(MAX_FILE_BYTES) + "-byte"
                " limit (it reads whole files into memory)"
            )
        if n < 12:
            raise Error("parquet: '" + self.path + "' is too short to be one")

        var p = byte_ptr(self.bytes)
        var magic = [0x50, 0x41, 0x52, 0x31]  # "PAR1"
        for k in range(4):
            if Int(p[unsafe_offset=k]) != magic[k]:
                raise Error("parquet: '" + self.path + "' has no PAR1 header")
            if Int(p[unsafe_offset = n - 4 + k]) != magic[k]:
                raise Error("parquet: '" + self.path + "' has no PAR1 footer")

        var lc = ByteCursor(p, n, n - 8)
        var flen = lc.le_u32()
        var fstart = n - 8 - flen
        if fstart < 4:
            raise Error(
                "parquet: footer length " + String(flen) + " does not fit in a "
                + String(n) + "-byte file"
            )
        var fc = ByteCursor(p, n - 8, fstart)
        self.meta = parse_file_metadata(fc)

    def __init__(out self, *, deinit move: Self):
        self.bytes = move.bytes^
        self.meta = move.meta^
        self.path = move.path^

    def num_rows(self) -> Int:
        return self.meta.num_rows

    def column_names(self) -> List[String]:
        var out = List[String]()
        for i in range(len(self.meta.leaves)):
            out.append(String(self.meta.leaves[i].path))
        return out^

    def has_column(self, path: String) -> Bool:
        return self.meta.has_leaf(path)

    # ── public typed reads ────────────────────────────────────────────

    def read_f64(mut self, path: String) raises -> List[Float64]:
        """Every value of a FLOAT or DOUBLE leaf, in file order.

        A `FLOAT` column widens exactly — every float32 is representable — so
        the caller can narrow back to `Float32` with no loss.
        """
        var leaf = self.meta.leaves[self.meta.leaf_index(path)]
        if leaf.physical_type != PT_FLOAT and leaf.physical_type != PT_DOUBLE:
            raise Error(
                "parquet: column '" + path + "' is "
                + physical_type_name(leaf.physical_type)
                + ", not FLOAT or DOUBLE — use read_i64"
            )
        var vf = List[Float64]()
        var vi = List[Int64]()
        var vb = List[UInt8]()
        var vo = List[Int]()
        var rep = List[Int32]()
        self._read_leaf(leaf, vf, vi, vb, vo, False, rep)
        return vf^

    def read_i64(mut self, path: String) raises -> List[Int64]:
        """Every value of an INT32, INT64 or BOOLEAN leaf, in file order."""
        var leaf = self.meta.leaves[self.meta.leaf_index(path)]
        if (
            leaf.physical_type != PT_INT32
            and leaf.physical_type != PT_INT64
            and leaf.physical_type != PT_BOOLEAN
        ):
            raise Error(
                "parquet: column '" + path + "' is "
                + physical_type_name(leaf.physical_type)
                + ", not an integer — use read_f64"
            )
        var vf = List[Float64]()
        var vi = List[Int64]()
        var vb = List[UInt8]()
        var vo = List[Int]()
        var rep = List[Int32]()
        self._read_leaf(leaf, vf, vi, vb, vo, False, rep)
        return vi^

    def read_byte_arrays(
        mut self, path: String, mut values: List[UInt8], mut offsets: List[Int]
    ) raises -> Int:
        """Every value of a BYTE_ARRAY leaf, concatenated. Returns the count.

        ⚠ NOT A `List[List[UInt8]]`. CIFAR-10's image column is 60,000
        PNG blobs; a list of lists is 60,000 heap allocations to hand back
        something the caller immediately walks once. `values` holds the bytes
        end to end and `offsets` their boundaries — value `k` spans
        `values[offsets[k] : offsets[k + 1]]`, so `offsets` has one entry more
        than there are values.
        """
        var leaf = self.meta.leaves[self.meta.leaf_index(path)]
        if leaf.physical_type != PT_BYTE_ARRAY:
            raise Error(
                "parquet: column '" + path + "' is "
                + physical_type_name(leaf.physical_type)
                + ", not BYTE_ARRAY"
            )
        var vf = List[Float64]()
        var vi = List[Int64]()
        var rep = List[Int32]()
        self._read_leaf(leaf, vf, vi, values, offsets, False, rep)
        if len(offsets) == 0:
            offsets.append(0)
        return len(offsets) - 1

    def read_rep_levels(mut self, path: String) raises -> List[Int32]:
        """The raw repetition levels of a LIST leaf, in file order.

        ⚠ THIS IS THE ONLY WAY TO SEE HOW VALUES ARE GROUPED INTO ROWS. The
        value readers above return a flat sequence that is IDENTICAL whether a
        column was grouped `1,2,1,3,...` or `38,0,0,...` — so a writer that
        emits the wrong repetition levels puts every value in the right order
        and the wrong row, and no amount of value comparison detects it.
        `tests/io/test_parquet_write.mojo` compares this stream against a
        file Arrow wrote, which is what makes that gate bite.

        Level `0` starts a new ROW; a higher level starts a new list at that
        depth. Reading it is off the hot path — nothing in the LeRobot import
        calls this.
        """
        var leaf = self.meta.leaves[self.meta.leaf_index(path)]
        if leaf.max_rep == 0:
            raise Error(
                "parquet: column '" + path + "' is not inside a list, so it"
                " has no repetition levels"
            )
        var vf = List[Float64]()
        var vi = List[Int64]()
        var vb = List[UInt8]()
        var vo = List[Int]()
        var rep = List[Int32]()
        self._read_leaf(leaf, vf, vi, vb, vo, True, rep)
        return rep^

    def read_list_counts(mut self, path: String) raises -> List[Int]:
        """Values per ROW for a LIST leaf, derived from its levels.

        For `list<T>` that is the row's element count. For a nested column it
        is the total leaf count in that row — `3` for a `[3,1,1]` statistic.

        ⚠ EXACT ONLY BECAUSE NULLS AND EMPTY LISTS CANNOT BE READ. An empty
        list occupies a level entry with no value, which would make entries
        and values disagree; `_decode_data_page` already refuses any page
        whose definition levels report one, so within this reader the two
        counts coincide.
        """
        var rep = self.read_rep_levels(path)
        var counts = List[Int]()
        for i in range(len(rep)):
            if rep[i] == 0:
                counts.append(1)
            else:
                if len(counts) == 0:
                    raise Error(
                        "parquet: column '" + path + "' starts with"
                        " repetition level " + String(Int(rep[i]))
                        + "; the first value of a column must start a row"
                    )
                counts[len(counts) - 1] += 1
        return counts^

    # ── decoding ──────────────────────────────────────────────────────

    def _read_leaf(
        mut self,
        leaf: LeafInfo,
        mut out_f: List[Float64],
        mut out_i: List[Int64],
        mut out_b: List[UInt8],
        mut out_off: List[Int],
        want_rep: Bool,
        mut out_rep: List[Int32],
    ) raises:
        var p = byte_ptr(self.bytes)
        var n = len(self.bytes)
        var scratch = List[UInt8]()

        for g in range(len(self.meta.row_groups)):
            var found = False
            for k in range(len(self.meta.row_groups[g].columns)):
                var ch = self.meta.row_groups[g].columns[k]
                if ch.path != leaf.path:
                    continue
                found = True
                self._read_chunk(
                    p, n, ch, leaf, scratch, out_f, out_i, out_b, out_off,
                    want_rep, out_rep,
                )
            if not found:
                raise Error(
                    "parquet: row group " + String(g) + " has no chunk for '"
                    + leaf.path + "'"
                )

    def _read_chunk(
        mut self,
        p: BPtr,
        n: Int,
        ch: ColumnChunkMeta,
        leaf: LeafInfo,
        mut scratch: List[UInt8],
        mut out_f: List[Float64],
        mut out_i: List[Int64],
        mut out_b: List[UInt8],
        mut out_off: List[Int],
        want_rep: Bool,
        mut out_rep: List[Int32],
    ) raises:
        # The dictionary page, when present, precedes the data pages; the
        # chunk's byte span starts at whichever comes first.
        var start = ch.data_page_offset
        if ch.dictionary_page_offset >= 0 and ch.dictionary_page_offset < start:
            start = ch.dictionary_page_offset
        var stop = start + ch.total_compressed_size
        if stop > n:
            raise Error(
                "parquet: column chunk for '" + leaf.path + "' ends at "
                + String(stop) + " past the " + String(n) + "-byte file"
            )

        var dict_f = List[Float64]()
        var dict_i = List[Int64]()
        var dict_b = List[UInt8]()
        var dict_off = List[Int]()
        var have_dict = False
        var seen = 0
        var pos = start

        while pos < stop and seen < ch.num_values:
            var hc = ByteCursor(p, n, pos)
            var hdr = _read_page_header(hc)
            var body = hc.pos
            if body + hdr.compressed_size > n:
                raise Error("parquet: page body runs past the end of the file")

            # Decompress (or alias) the page body. On a v2 page only the value
            # section is compressed — the levels sit in front of it, raw.
            var raw_prefix = 0
            var pc: ByteCursor
            if hdr.page_type == PAGE_DATA_V2:
                raw_prefix = hdr.rep_bytes + hdr.def_bytes
            var comp_off = body + raw_prefix
            var comp_len = hdr.compressed_size - raw_prefix
            var unc_len = hdr.uncompressed_size - raw_prefix

            var needs_codec = ch.codec != CODEC_UNCOMPRESSED and (
                hdr.page_type != PAGE_DATA_V2 or hdr.v2_compressed
            )
            if needs_codec:
                if ch.codec != CODEC_SNAPPY:
                    raise Error(
                        "parquet: codec " + codec_name(ch.codec) + " is not"
                        " supported (only UNCOMPRESSED and SNAPPY)"
                    )
                if len(scratch) < raw_prefix + unc_len:
                    scratch.resize(raw_prefix + unc_len, UInt8(0))
                var sp = byte_ptr(scratch)
                for k in range(raw_prefix):
                    sp[unsafe_offset=k] = p[unsafe_offset = body + k]
                var got = snappy_decompress(
                    p.unsafe_offset(comp_off), comp_len, sp.unsafe_offset(raw_prefix),
                    unc_len,
                )
                if got != unc_len:
                    raise Error(
                        "parquet: page decompressed to " + String(got)
                        + " bytes, header said " + String(unc_len)
                    )
                pc = ByteCursor(sp, raw_prefix + unc_len, 0)
            else:
                pc = ByteCursor(p, body + hdr.compressed_size, body)

            if hdr.page_type == PAGE_DICTIONARY:
                if hdr.encoding != ENC_PLAIN and hdr.encoding != ENC_PLAIN_DICTIONARY:
                    raise Error(
                        "parquet: dictionary page encoded "
                        + encoding_name(hdr.encoding) + ", expected PLAIN"
                    )
                _plain_values(
                    pc, leaf.physical_type, hdr.num_values, dict_f, dict_i,
                    dict_b, dict_off,
                )
                have_dict = True
            elif hdr.page_type == PAGE_DATA or hdr.page_type == PAGE_DATA_V2:
                self._decode_data_page(
                    pc, hdr, leaf, dict_f, dict_i, dict_b, dict_off,
                    have_dict, out_f, out_i, out_b, out_off, want_rep,
                    out_rep,
                )
                seen += hdr.num_values
            elif hdr.page_type != PAGE_INDEX:
                raise Error(
                    "parquet: unknown page type " + String(hdr.page_type)
                )

            pos = body + hdr.compressed_size

        if seen != ch.num_values:
            raise Error(
                "parquet: chunk for '" + leaf.path + "' yielded " + String(seen)
                + " values, its metadata declares " + String(ch.num_values)
            )

    def _decode_data_page(
        mut self,
        mut pc: ByteCursor,
        hdr: PageHeader,
        leaf: LeafInfo,
        dict_f: List[Float64],
        dict_i: List[Int64],
        dict_b: List[UInt8],
        dict_off: List[Int],
        have_dict: Bool,
        mut out_f: List[Float64],
        mut out_i: List[Int64],
        mut out_b: List[UInt8],
        mut out_off: List[Int],
        want_rep: Bool,
        mut out_rep: List[Int32],
    ) raises:
        var is_v2 = hdr.page_type == PAGE_DATA_V2

        if not is_v2:
            if hdr.rep_encoding != ENC_RLE and leaf.max_rep > 0:
                raise Error(
                    "parquet: repetition levels encoded "
                    + encoding_name(hdr.rep_encoding) + "; only RLE is"
                    " supported (the deprecated BIT_PACKED packs MSB-first)"
                )
            if hdr.def_encoding != ENC_RLE and leaf.max_def > 0:
                raise Error(
                    "parquet: definition levels encoded "
                    + encoding_name(hdr.def_encoding) + "; only RLE is"
                    " supported (the deprecated BIT_PACKED packs MSB-first)"
                )

        # ── repetition levels ─────────────────────────────────────────
        # A flat read of every value does not need them; the caller reshapes
        # by the list width it already knows. They still have to be CONSUMED,
        # because the definition levels sit immediately after them.
        #
        # ⚠ `want_rep` EXISTS BECAUSE DISCARDING THEM MAKES ONE CLASS OF BUG
        # INVISIBLE. The levels are the only record of where one row's list
        # ends and the next begins, so a writer that emits the wrong ones
        # produces a file whose values are all correct and whose ROWS are
        # wrong — and a reader that skips them cannot tell. It is off by
        # default: `read_list_counts` is a gate's tool, and collecting a level
        # per value would cost a LeRobot import millions of appends it has no
        # use for.
        if leaf.max_rep > 0:
            if is_v2:
                if want_rep:
                    var rstart = pc.pos
                    var rc = ByteCursor(pc.p, rstart + hdr.rep_bytes, rstart)
                    rle_decode(
                        rc, bit_width_for(leaf.max_rep), hdr.num_values,
                        out_rep,
                    )
                pc.skip_bytes(hdr.rep_bytes)
            else:
                var rlen = pc.le_u32()
                if want_rep:
                    var rstart = pc.pos
                    var rc = ByteCursor(pc.p, rstart + rlen, rstart)
                    rle_decode(
                        rc, bit_width_for(leaf.max_rep), hdr.num_values,
                        out_rep,
                    )
                pc.skip_bytes(rlen)

        # ── definition levels: how many values are actually present ───
        var present = hdr.num_values
        if leaf.max_def > 0:
            var levels = List[Int32]()
            if is_v2:
                # Bounded to the header's own byte count, not to the page: a
                # level stream that overruns its declared length is corrupt,
                # and letting it read on into the values would decode garbage
                # levels rather than say so.
                var dstart = pc.pos
                var lc = ByteCursor(pc.p, dstart + hdr.def_bytes, dstart)
                rle_decode(
                    lc, bit_width_for(leaf.max_def), hdr.num_values, levels
                )
                pc.skip_bytes(hdr.def_bytes)
            else:
                var dlen = pc.le_u32()
                var dstart = pc.pos
                var lc = ByteCursor(pc.p, dstart + dlen, dstart)
                rle_decode(
                    lc, bit_width_for(leaf.max_def), hdr.num_values, levels
                )
                pc.skip_bytes(dlen)
            var np = 0
            for i in range(len(levels)):
                if Int(levels[i]) == leaf.max_def:
                    np += 1
            present = np

        if present != hdr.num_values:
            raise Error(
                "parquet: column '" + leaf.path + "' has "
                + String(hdr.num_values - present) + " null(s) in a page; this"
                " reader returns dense columns and has no way to represent"
                " them"
            )

        # ── values ────────────────────────────────────────────────────
        if hdr.encoding == ENC_PLAIN:
            _plain_values(
                pc, leaf.physical_type, present, out_f, out_i, out_b, out_off
            )
        elif hdr.encoding == ENC_RLE:
            # BOOLEAN values, and only BOOLEAN values, may be RLE-encoded —
            # parquet-cpp writes them this way on a v2 page. The stream is the
            # same hybrid the levels use, at bit width 1, behind its own
            # 4-byte length prefix (`RleBooleanDecoder::SetData`).
            if leaf.physical_type != PT_BOOLEAN:
                raise Error(
                    "parquet: RLE value encoding on "
                    + physical_type_name(leaf.physical_type) + " column '"
                    + leaf.path + "'; RLE values are defined for BOOLEAN only"
                )
            var blen = pc.le_u32()
            var bstart = pc.pos
            var bc = ByteCursor(pc.p, bstart + blen, bstart)
            var bits = List[Int32]()
            rle_decode(bc, 1, present, bits)
            for i in range(present):
                out_i.append(Int64(Int(bits[i])))
            pc.skip_bytes(blen)
        elif (
            hdr.encoding == ENC_RLE_DICTIONARY
            or hdr.encoding == ENC_PLAIN_DICTIONARY
        ):
            if not have_dict:
                raise Error(
                    "parquet: dictionary-encoded page for '" + leaf.path
                    + "' with no dictionary page in the chunk"
                )
            var width = pc.u8()
            var idx = List[Int32]()
            rle_decode(pc, width, present, idx)
            var is_ba = leaf.physical_type == PT_BYTE_ARRAY
            var nd: Int
            if is_ba:
                nd = len(dict_off) - 1 if len(dict_off) > 0 else 0
            elif len(dict_f) > 0:
                nd = len(dict_f)
            else:
                nd = len(dict_i)
            if is_ba and len(out_off) == 0:
                out_off.append(0)
            for i in range(present):
                var k = Int(idx[i])
                if k < 0 or k >= nd:
                    raise Error(
                        "parquet: dictionary index " + String(k) + " outside a "
                        + String(nd) + "-entry dictionary"
                    )
                if is_ba:
                    # ⚠ Keyed on the LEAF TYPE, not on which dictionary list is
                    # non-empty: an image column whose first entries happen to
                    # be zero-length would otherwise look like an empty
                    # dictionary and fall through to the numeric branch.
                    for j in range(dict_off[k], dict_off[k + 1]):
                        out_b.append(dict_b[j])
                    out_off.append(len(out_b))
                elif len(dict_f) > 0:
                    out_f.append(dict_f[k])
                else:
                    out_i.append(dict_i[k])
        else:
            raise Error(
                "parquet: value encoding " + encoding_name(hdr.encoding)
                + " for column '" + leaf.path + "' is not supported"
            )
