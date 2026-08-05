# +--------------------------------------------------------------------------+ #
# | ColumnSpec — one named field of a trajectory store
# +--------------------------------------------------------------------------+ #
"""A store is a set of named columns over a shared row axis.

This is the design decision the whole layer turns on: the axis that varies
between consumers is the COLUMN SET, not CPU-vs-GPU and not the sampling
policy. Every legacy replay buffer hardcoded `obs/action/reward/next_obs/done`,
which is why dm_control BFM data (`qpos`/`qvel` — its rewards read `Data`, not
the observation), PushT (`pixels`/`proprio`/`state`) and Atari (`ram`) each
needed a buffer of their own.

Columns carry a runtime `DType`. Reads and writes still take a COMPTIME dtype
parameter — Mojo needs one to type the buffer — and the store checks it against
the registered spec, so a mismatched read raises instead of reinterpreting
bytes.

`shape` holds the TRAILING dims only (everything after the row axis). An empty
shape is a scalar column stored as rank-1 `[N]`; `[9]` is `[N, 9]`; PushT's
pixels are `[H, W, 3]` → `[N, H, W, 3]`.
"""

from mojo_rl.io.hdf5.h5_types import H5T_FLOAT, H5T_INTEGER, H5T_SGN_2
from std.ffi import c_int


struct ColumnSpec(Copyable & ImplicitlyDeletable):
    var name: String
    var dtype: DType
    var shape: List[Int]
    """Trailing dims, excluding the row axis. Empty = scalar column."""

    def __init__(out self, var name: String, dtype: DType, var shape: List[Int]):
        self.name = name^
        self.dtype = dtype
        self.shape = shape^

    def __init__(out self, var name: String, dtype: DType, row_dim: Int):
        """Convenience for the common flat case. `row_dim=1` → scalar column."""
        self.name = name^
        self.dtype = dtype
        self.shape = List[Int]()
        if row_dim != 1:
            self.shape.append(row_dim)

    def __init__(out self, *, copy: Self):
        self.name = String(copy.name)
        self.dtype = copy.dtype
        self.shape = copy.shape.copy()

    def __init__(out self, *, deinit move: Self):
        self.name = move.name^
        self.dtype = move.dtype
        self.shape = move.shape^

    def row_dim(self) -> Int:
        """Elements per row = product of the trailing dims (1 if scalar)."""
        var n = 1
        for i in range(len(self.shape)):
            n *= self.shape[i]
        return n

    def rank(self) -> Int:
        """Full dataset rank including the row axis."""
        return 1 + len(self.shape)

    def shape_str(self) -> String:
        var s = String()
        for i in range(len(self.shape)):
            if i > 0:
                s += ","
            s += String(self.shape[i])
        return s^

    def describe(self) raises -> String:
        var s = String(self.name) + ":" + dtype_name(self.dtype)
        var sh = self.shape_str()
        if sh.byte_length() > 0:
            s += ":" + sh
        return s^


# ── DType ⇄ manifest name ────────────────────────────────────────────────

def dtype_name(dt: DType) raises -> String:
    """Stable, explicit spelling for the manifest. Deliberately NOT
    `String(dt)` — the manifest is a persisted format and must not drift with
    a stdlib repr change."""
    if dt == DType.float32:
        return String("float32")
    if dt == DType.float64:
        return String("float64")
    if dt == DType.int8:
        return String("int8")
    if dt == DType.uint8:
        return String("uint8")
    if dt == DType.int16:
        return String("int16")
    if dt == DType.uint16:
        return String("uint16")
    if dt == DType.int32:
        return String("int32")
    if dt == DType.uint32:
        return String("uint32")
    if dt == DType.int64:
        return String("int64")
    if dt == DType.uint64:
        return String("uint64")
    raise Error("data: unsupported column dtype")


def dtype_from_name(name: String) raises -> DType:
    if name == "float32":
        return DType.float32
    if name == "float64":
        return DType.float64
    if name == "int8":
        return DType.int8
    if name == "uint8":
        return DType.uint8
    if name == "int16":
        return DType.int16
    if name == "uint16":
        return DType.uint16
    if name == "int32":
        return DType.int32
    if name == "uint32":
        return DType.uint32
    if name == "int64":
        return DType.int64
    if name == "uint64":
        return DType.uint64
    raise Error("data: unknown dtype name in manifest: " + name)


def dtype_from_h5(cls: c_int, elem_size: Int, sign: c_int) raises -> DType:
    """Recover a `DType` from what `H5Dataset` introspection reports.

    Used when ingesting a file we did not write, which has no manifest.
    """
    if cls == H5T_FLOAT:
        if elem_size == 4:
            return DType.float32
        if elem_size == 8:
            return DType.float64
        raise Error(
            "data: unsupported float width " + String(elem_size) + " bytes"
        )
    if cls == H5T_INTEGER:
        var signed = sign == H5T_SGN_2
        if elem_size == 1:
            return DType.int8 if signed else DType.uint8
        if elem_size == 2:
            return DType.int16 if signed else DType.uint16
        if elem_size == 4:
            return DType.int32 if signed else DType.uint32
        if elem_size == 8:
            return DType.int64 if signed else DType.uint64
        raise Error(
            "data: unsupported integer width " + String(elem_size) + " bytes"
        )
    raise Error(
        "data: unsupported HDF5 type class " + String(Int(cls))
        + " (only INTEGER and FLOAT columns are supported)"
    )


def dtype_bytes(dt: DType) raises -> Int:
    """Bytes per element. Explicit, for the same reason `dtype_name` is:
    residency guards and buffer sizing must not drift with a stdlib rename."""
    if dt == DType.int8 or dt == DType.uint8:
        return 1
    if dt == DType.int16 or dt == DType.uint16:
        return 2
    if dt == DType.float32 or dt == DType.int32 or dt == DType.uint32:
        return 4
    if dt == DType.float64 or dt == DType.int64 or dt == DType.uint64:
        return 8
    raise Error("data: unsupported dtype width")
