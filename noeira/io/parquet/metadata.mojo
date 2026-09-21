# +--------------------------------------------------------------------------+ #
# | The Parquet footer: schema tree, row groups, column chunk offsets
# +--------------------------------------------------------------------------+ #
"""Parses `FileMetaData` out of the footer and flattens the schema into leaves.

A Parquet file ends with `<footer bytes> <4-byte LE length> "PAR1"`. The footer
is one Thrift compact `FileMetaData`; everything else in the file is located
relative to offsets it carries.

## The schema is a TREE, the data is a list of LEAVES

`FileMetaData.schema` is the tree flattened depth-first, each element declaring
its `num_children`. Only childless elements hold data. Walking the tree is what
produces the two numbers every page decode needs:

* **max definition level** — +1 for every `OPTIONAL` *or* `REPEATED` ancestor
  (and for the leaf itself). A value is present iff its definition level equals
  this maximum.
* **max repetition level** — +1 for every `REPEATED` ancestor. Non-zero means
  the column is inside a list.

For Arrow's `fixed_size_list<float>[6]` the tree is
`action` (OPTIONAL group) → `list` (REPEATED group) → `element` (OPTIONAL
float), giving the leaf `action.list.element` with max_def 3 and max_rep 1.
Those are exactly the levels a data page prefixes its values with, so getting
this walk wrong does not produce an error — it mis-slices the level stream and
the values come out shifted.
"""

from .thrift import (
    ByteCursor, FieldHeader, read_field_header, read_list_header, skip_field,
    skip_struct, T_STRUCT,
)


# parquet.thrift `Type`
comptime PT_BOOLEAN = 0
comptime PT_INT32 = 1
comptime PT_INT64 = 2
comptime PT_INT96 = 3
comptime PT_FLOAT = 4
comptime PT_DOUBLE = 5
comptime PT_BYTE_ARRAY = 6
comptime PT_FIXED_LEN_BYTE_ARRAY = 7

# parquet.thrift `FieldRepetitionType`
comptime REP_REQUIRED = 0
comptime REP_OPTIONAL = 1
comptime REP_REPEATED = 2

# parquet.thrift `CompressionCodec`
comptime CODEC_UNCOMPRESSED = 0
comptime CODEC_SNAPPY = 1
comptime CODEC_GZIP = 2
comptime CODEC_LZO = 3
comptime CODEC_BROTLI = 4
comptime CODEC_LZ4 = 5
comptime CODEC_ZSTD = 6
comptime CODEC_LZ4_RAW = 7


def physical_type_name(t: Int) -> String:
    if t == PT_BOOLEAN: return String("BOOLEAN")
    if t == PT_INT32: return String("INT32")
    if t == PT_INT64: return String("INT64")
    if t == PT_INT96: return String("INT96")
    if t == PT_FLOAT: return String("FLOAT")
    if t == PT_DOUBLE: return String("DOUBLE")
    if t == PT_BYTE_ARRAY: return String("BYTE_ARRAY")
    if t == PT_FIXED_LEN_BYTE_ARRAY: return String("FIXED_LEN_BYTE_ARRAY")
    return String("type#") + String(t)


def codec_name(c: Int) -> String:
    if c == CODEC_UNCOMPRESSED: return String("UNCOMPRESSED")
    if c == CODEC_SNAPPY: return String("SNAPPY")
    if c == CODEC_GZIP: return String("GZIP")
    if c == CODEC_LZO: return String("LZO")
    if c == CODEC_BROTLI: return String("BROTLI")
    if c == CODEC_LZ4: return String("LZ4")
    if c == CODEC_ZSTD: return String("ZSTD")
    if c == CODEC_LZ4_RAW: return String("LZ4_RAW")
    return String("codec#") + String(c)


@fieldwise_init
struct SchemaElement(Copyable, ImplicitlyCopyable, Movable):
    var physical_type: Int
    """-1 for a group (no `type` field on the wire)."""
    var repetition: Int
    var name: String
    var num_children: Int


@fieldwise_init
struct LeafInfo(Copyable, ImplicitlyCopyable, Movable):
    var path: String
    """Dotted `path_in_schema`, e.g. `action.list.element`."""
    var physical_type: Int
    var max_def: Int
    var max_rep: Int


@fieldwise_init
struct ColumnChunkMeta(Copyable, ImplicitlyCopyable, Movable):
    var path: String
    var physical_type: Int
    var codec: Int
    var num_values: Int
    var data_page_offset: Int
    var dictionary_page_offset: Int
    """-1 when the chunk has no dictionary page."""
    var total_compressed_size: Int


@fieldwise_init
struct RowGroupMeta(Copyable, Movable):
    var columns: List[ColumnChunkMeta]
    var num_rows: Int


struct FileMetaData(Copyable, Movable):
    var version: Int
    var num_rows: Int
    var created_by: String
    var leaves: List[LeafInfo]
    var row_groups: List[RowGroupMeta]

    def __init__(out self):
        self.version = 0
        self.num_rows = 0
        self.created_by = String("")
        self.leaves = List[LeafInfo]()
        self.row_groups = List[RowGroupMeta]()

    def leaf_index(self, path: String) raises -> Int:
        for i in range(len(self.leaves)):
            if self.leaves[i].path == path:
                return i
        var have = String("")
        for i in range(len(self.leaves)):
            if i > 0:
                have += ", "
            have += self.leaves[i].path
        raise Error(
            "parquet: no leaf column '" + path + "'; the file has: " + have
        )

    def has_leaf(self, path: String) -> Bool:
        for i in range(len(self.leaves)):
            if self.leaves[i].path == path:
                return True
        return False


# ══════════════════════════════════════════════════════════════════════════
# Thrift struct parsers
# ══════════════════════════════════════════════════════════════════════════

def _read_schema_element(mut c: ByteCursor) raises -> SchemaElement:
    var e = SchemaElement(-1, REP_REQUIRED, String(""), 0)
    var fid = 0
    while True:
        var h = read_field_header(c, fid)
        if h.stop:
            break
        fid = h.id
        if h.id == 1:
            e.physical_type = c.zigzag()
        elif h.id == 3:
            e.repetition = c.zigzag()
        elif h.id == 4:
            e.name = c.binary()
        elif h.id == 5:
            e.num_children = c.zigzag()
        else:
            skip_field(c, h.type)
    return e^


def _read_column_meta(mut c: ByteCursor) raises -> ColumnChunkMeta:
    var m = ColumnChunkMeta(String(""), -1, CODEC_UNCOMPRESSED, 0, 0, -1, 0)
    var fid = 0
    while True:
        var h = read_field_header(c, fid)
        if h.stop:
            break
        fid = h.id
        if h.id == 1:
            m.physical_type = c.zigzag()
        elif h.id == 3:
            # path_in_schema: list<string>, joined with '.' to match the
            # dotted leaf paths built by `_flatten_schema`.
            var lh = read_list_header(c)
            var p = String("")
            for i in range(lh.size):
                var part = c.binary()
                if i > 0:
                    p += "."
                p += part
            m.path = p^
        elif h.id == 4:
            m.codec = c.zigzag()
        elif h.id == 5:
            m.num_values = c.zigzag()
        elif h.id == 7:
            m.total_compressed_size = c.zigzag()
        elif h.id == 9:
            m.data_page_offset = c.zigzag()
        elif h.id == 11:
            m.dictionary_page_offset = c.zigzag()
        else:
            skip_field(c, h.type)
    return m^


def _read_column_chunk(mut c: ByteCursor) raises -> ColumnChunkMeta:
    var out = ColumnChunkMeta(String(""), -1, CODEC_UNCOMPRESSED, 0, 0, -1, 0)
    var got = False
    var fid = 0
    while True:
        var h = read_field_header(c, fid)
        if h.stop:
            break
        fid = h.id
        if h.id == 1:
            # `file_path` — a chunk stored in a SEPARATE file. Nothing writes
            # these any more, and following one would need a second open, so
            # it is rejected rather than silently read from the wrong bytes.
            var fp = c.binary()
            if fp != "":
                raise Error(
                    "parquet: column chunk lives in an external file ('" + fp
                    + "'); this reader only handles self-contained files"
                )
        elif h.id == 3:
            out = _read_column_meta(c)
            got = True
        else:
            skip_field(c, h.type)
    if not got:
        raise Error("parquet: column chunk has no meta_data")
    return out^


def _read_row_group(mut c: ByteCursor) raises -> RowGroupMeta:
    var cols = List[ColumnChunkMeta]()
    var nrows = 0
    var fid = 0
    while True:
        var h = read_field_header(c, fid)
        if h.stop:
            break
        fid = h.id
        if h.id == 1:
            var lh = read_list_header(c)
            if lh.elem_type != T_STRUCT:
                raise Error("parquet: RowGroup.columns is not a struct list")
            for _ in range(lh.size):
                cols.append(_read_column_chunk(c))
        elif h.id == 3:
            nrows = c.zigzag()
        else:
            skip_field(c, h.type)
    return RowGroupMeta(cols^, nrows)


def _flatten_schema(
    elems: List[SchemaElement], mut leaves: List[LeafInfo]
) raises:
    """Depth-first walk of the flattened schema, accumulating levels.

    Explicitly iterative with a stack rather than recursive: the walk carries
    three pieces of per-node state (path, max_def, max_rep) and a recursive
    helper in Mojo would need them all as arguments plus a cursor into the
    flat list, which is the same bookkeeping with a call frame around it.
    """
    if len(elems) == 0:
        raise Error("parquet: empty schema")

    # The root element names the file, not a column, and contributes no level.
    var stack_remaining = List[Int]()
    var stack_path = List[String]()
    var stack_def = List[Int]()
    var stack_rep = List[Int]()

    stack_remaining.append(elems[0].num_children)
    stack_path.append(String(""))
    stack_def.append(0)
    stack_rep.append(0)

    var i = 1
    while i < len(elems):
        while len(stack_remaining) > 0 and stack_remaining[len(stack_remaining) - 1] == 0:
            _ = stack_remaining.pop()
            _ = stack_path.pop()
            _ = stack_def.pop()
            _ = stack_rep.pop()
        if len(stack_remaining) == 0:
            raise Error(
                "parquet: schema has " + String(len(elems)) + " elements but"
                " the child counts describe fewer — the footer is corrupt"
            )

        var top = len(stack_remaining) - 1
        stack_remaining[top] -= 1

        var e = elems[i]
        var parent_path = stack_path[top]
        var path = e.name if parent_path == "" else parent_path + "." + e.name
        var d = stack_def[top]
        var r = stack_rep[top]
        if e.repetition == REP_OPTIONAL:
            d += 1
        elif e.repetition == REP_REPEATED:
            d += 1
            r += 1

        if e.num_children == 0:
            if e.physical_type < 0:
                raise Error(
                    "parquet: leaf '" + path + "' declares no physical type"
                )
            leaves.append(LeafInfo(path^, e.physical_type, d, r))
        else:
            stack_remaining.append(e.num_children)
            stack_path.append(path^)
            stack_def.append(d)
            stack_rep.append(r)
        i += 1


def parse_file_metadata(mut c: ByteCursor) raises -> FileMetaData:
    """Parse a `FileMetaData` struct sitting at the cursor."""
    var out = FileMetaData()
    var elems = List[SchemaElement]()
    var fid = 0
    while True:
        var h = read_field_header(c, fid)
        if h.stop:
            break
        fid = h.id
        if h.id == 1:
            out.version = c.zigzag()
        elif h.id == 2:
            var lh = read_list_header(c)
            if lh.elem_type != T_STRUCT:
                raise Error("parquet: schema is not a struct list")
            for _ in range(lh.size):
                elems.append(_read_schema_element(c))
        elif h.id == 3:
            out.num_rows = c.zigzag()
        elif h.id == 4:
            var lh2 = read_list_header(c)
            if lh2.elem_type != T_STRUCT:
                raise Error("parquet: row_groups is not a struct list")
            for _ in range(lh2.size):
                out.row_groups.append(_read_row_group(c))
        elif h.id == 6:
            out.created_by = c.binary()
        else:
            skip_field(c, h.type)

    _flatten_schema(elems, out.leaves)
    return out^
