# +--------------------------------------------------------------------------+ #
# | A native Parquet reader for Mojo
# +--------------------------------------------------------------------------+ #
"""Read `.parquet` metadata and small tables without Python.

    from mojo_rl.io.parquet import ParquetFile

    var f = ParquetFile(String("meta/episodes/chunk-000/file-000.parquet"))
    var lengths = f.read_i64(String("length"))

## Why native rather than a binding

`libparquet` ships in the pixi environment, but its API is C++ with no stable
ABI and mangled symbols — unusable from `external_call` without a compiled
shim, and a shim pins the build to one Arrow release. The subset Parquet
actually needs to read an Arrow-written file is small enough to own: a Thrift
compact reader, Snappy, the RLE/bit-packed hybrid, and PLAIN. That is what
these four files are, and they replace `pyarrow` in the LeRobot import path.

See `reader.mojo` for the exact supported subset and the encodings it rejects
by name.
"""

from .thrift import ByteCursor, BPtr, byte_ptr
from .metadata import (
    FileMetaData, LeafInfo, ColumnChunkMeta, RowGroupMeta,
    PT_BOOLEAN, PT_INT32, PT_INT64, PT_FLOAT, PT_DOUBLE, PT_BYTE_ARRAY,
    physical_type_name, codec_name,
)
from .reader import ParquetFile, encoding_name
from .rle import bit_width_for, rle_decode
from .snappy import snappy_decompress, snappy_uncompressed_length
