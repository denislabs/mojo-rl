# +--------------------------------------------------------------------------+ #
# | libhdf5 type aliases & enum constants
# +--------------------------------------------------------------------------+ #
"""HDF5 ABI type aliases and enum constants used across the FFI.

HDF5 1.10+ defines ``hid_t`` as ``int64_t``, ``hsize_t`` as ``uint64_t``,
``herr_t`` as ``int``. These are stable across the 1.x line.

Enum values are pinned at the C API level — they are part of the ABI and
do not change between minor releases.
"""

from std.ffi import c_int, c_uint


# ── ABI types ────────────────────────────────────────────────────────────────

comptime hid_t = Int64
"""HDF5 identifier handle (file / dataset / dataspace / type, etc.).

A non-positive value (typically -1) indicates an error or invalid handle.
"""

comptime hsize_t = UInt64
"""HDF5 unsigned dimension/size type."""

comptime herr_t = c_int
"""HDF5 error code — negative = error, non-negative = success."""


# ── H5F access flags (bitfield, unsigned) ────────────────────────────────────

comptime H5F_ACC_RDONLY = UInt32(0x0000)
"""Open file read-only."""

comptime H5F_ACC_RDWR = UInt32(0x0001)
"""Open file read-write."""

comptime H5F_ACC_SWMR_READ = UInt32(0x0040)
"""Open file in single-writer-multiple-reader read mode (safe for
concurrent reads while another process writes)."""


# ── H5P default property list ────────────────────────────────────────────────

comptime H5P_DEFAULT = hid_t(0)
"""Sentinel hid_t for "use default property list" — accepted everywhere
the HDF5 C API takes a property list argument."""


# ── H5T_class_t (datatype class) ─────────────────────────────────────────────

comptime H5T_NO_CLASS = c_int(-1)
comptime H5T_INTEGER = c_int(0)
comptime H5T_FLOAT = c_int(1)
comptime H5T_TIME = c_int(2)
comptime H5T_STRING = c_int(3)
comptime H5T_BITFIELD = c_int(4)
comptime H5T_OPAQUE = c_int(5)
comptime H5T_COMPOUND = c_int(6)
comptime H5T_REFERENCE = c_int(7)
comptime H5T_ENUM = c_int(8)
comptime H5T_VLEN = c_int(9)
comptime H5T_ARRAY = c_int(10)


# ── H5T_sign_t (integer signedness) ──────────────────────────────────────────

comptime H5T_SGN_ERROR = c_int(-1)
comptime H5T_SGN_NONE = c_int(0)
"""Unsigned integer."""
comptime H5T_SGN_2 = c_int(1)
"""Two's complement signed integer."""


# ── H5T_direction_t (for H5Tget_native_type) ─────────────────────────────────

comptime H5T_DIR_DEFAULT = c_int(0)
comptime H5T_DIR_ASCEND = c_int(1)
comptime H5T_DIR_DESCEND = c_int(2)


# ── H5S_seloper_t (hyperslab selection operator) ─────────────────────────────

comptime H5S_SELECT_SET = c_int(0)
comptime H5S_SELECT_OR = c_int(1)
comptime H5S_SELECT_AND = c_int(2)
comptime H5S_SELECT_XOR = c_int(3)
comptime H5S_SELECT_NOTB = c_int(4)
comptime H5S_SELECT_NOTA = c_int(5)


# ── H5S_ALL sentinel ─────────────────────────────────────────────────────────

comptime H5S_ALL = hid_t(0)
"""Sentinel hid_t for "use the dataset's full dataspace" — accepted by
H5Dread/H5Dwrite where a memory or file space argument is expected."""
