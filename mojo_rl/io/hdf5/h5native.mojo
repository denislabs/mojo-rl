# +--------------------------------------------------------------------------+ #
# | libhdf5 — H5open + the H5T_NATIVE_* global datatype ids
# +--------------------------------------------------------------------------+ #
"""Resolution of libhdf5's predefined native datatypes.

The READ path never needed these: `H5Dget_type` + `H5Tget_native_type`
derive the memory type from the file itself (see `h5t.mojo`'s note). The
WRITE path has no file to derive from — `H5Dcreate2` must be told the
datatype up front — so we have to reach the predefined ids.

Two facts about those ids, both verified against libhdf5 2.1.0 rather than
assumed, because both fail *silently* if you get them wrong:

1. `H5T_NATIVE_FLOAT` is not a constant. It is the C macro
   `(H5OPEN H5T_NATIVE_FLOAT_g)` — a global `hid_t` **variable**, plus an
   `H5open()` call the preprocessor inserts for you. An FFI caller gets
   neither, so we must dlsym the variable *and* call `H5open()` ourselves.

2. **Before `H5open()` every one of these globals reads `-1`**, and passing
   -1 onward fails with "not a datatype" — HDF5's own diagnostic, but only
   at the point of use, far from the cause. `native_type()` therefore runs
   `H5open()` behind a `_Global` so it happens exactly once, before any
   lookup.

⚠ Deref footgun: `handle.get_symbol[hid_t](name)` returns a pointer *to* the
global; the value is `p[][]`. Writing `Int(p[])` compiles fine and yields the
symbol's ADDRESS — a plausible-looking large integer that is not the id.
"""

from std.ffi import _get_dylib_function, _Global

from . import lib
from .h5_types import hid_t, herr_t
from .h5e import h5e_set_auto2_off


def h5open() raises -> herr_t:
    """``H5open(void) -> herr_t``.

    Initialize the library. Idempotent. Must precede any read of an
    `H5T_NATIVE_*_g` / `H5P_CLS_*_g` global (see module docstring).
    """
    return _get_dylib_function[lib, "H5open", def() thin -> herr_t]()()


def _init_h5_library() -> Bool:
    """`_Global` initializer — runs `H5open()` once per process."""
    try:
        var rc = h5open()
        if rc < 0:
            print("H5open() failed: rc=", Int(rc))
            return False
        # Silence libhdf5's automatic stderr stack dumps; every wrapper here
        # raises a Mojo Error with better context, and link enumeration uses a
        # failing call as its normal terminator. See h5e.mojo.
        _ = h5e_set_auto2_off()
        return True
    except e:
        print("H5open() raised:", e)
        return False


comptime _H5_INITIALIZED = _Global["MOJO_RL_HDF5_H5OPEN", _init_h5_library]()
"""Process-wide `H5open()` latch. Touch it before any global lookup."""


def global_hid(name: StaticString) raises -> hid_t:
    """Read a libhdf5 global `hid_t` variable by symbol name.

    Ensures `H5open()` has run first. Raises if the symbol is absent or
    still reads as an invalid handle.
    """
    if not _H5_INITIALIZED.get_or_create_ptr()[]:
        raise Error("libhdf5: H5open() failed; cannot resolve " + String(name))

    var p = lib.get_or_create_ptr()[].get_symbol[hid_t](name)
    if not p:
        raise Error("libhdf5: symbol not found: " + String(name))
    var id = p[][]
    if id < 0:
        raise Error(
            "libhdf5: global " + String(name) + " reads "
            + String(Int(id)) + " (expected a valid hid_t; H5open() ran but"
            " the type is still uninitialized)"
        )
    return id


def native_type_symbol[dtype: DType]() -> StaticString:
    """Map a Mojo `DType` onto its `H5T_NATIVE_*_g` symbol name.

    Only the dtypes the trajectory store actually persists are mapped;
    anything else is a comptime error rather than a silent mis-typed write.
    """
    comptime assert (
        dtype == DType.float32 or dtype == DType.float64
        or dtype == DType.int8 or dtype == DType.uint8
        or dtype == DType.int16 or dtype == DType.uint16
        or dtype == DType.int32 or dtype == DType.uint32
        or dtype == DType.int64 or dtype == DType.uint64
    ), (
        "hdf5: no H5T_NATIVE_* mapping for this DType. Supported:"
        " float32/64, int8/16/32/64, uint8/16/32/64."
    )
    comptime if dtype == DType.float32:
        return "H5T_NATIVE_FLOAT_g"
    elif dtype == DType.float64:
        return "H5T_NATIVE_DOUBLE_g"
    elif dtype == DType.int8:
        return "H5T_NATIVE_INT8_g"
    elif dtype == DType.uint8:
        return "H5T_NATIVE_UINT8_g"
    elif dtype == DType.int16:
        return "H5T_NATIVE_INT16_g"
    elif dtype == DType.uint16:
        return "H5T_NATIVE_UINT16_g"
    elif dtype == DType.int32:
        return "H5T_NATIVE_INT32_g"
    elif dtype == DType.uint32:
        return "H5T_NATIVE_UINT32_g"
    elif dtype == DType.int64:
        return "H5T_NATIVE_INT64_g"
    elif dtype == DType.uint64:
        return "H5T_NATIVE_UINT64_g"
    else:
        # Excluded by the comptime assert above.
        return ""


def native_type[dtype: DType]() raises -> hid_t:
    """The `H5T_NATIVE_*` id matching `dtype`, for H5Dcreate2/H5Dwrite.

    NOT an owned handle — these are library-owned predefined ids. Do not
    close them.
    """
    return global_hid(native_type_symbol[dtype]())
