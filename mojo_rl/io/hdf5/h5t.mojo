# +--------------------------------------------------------------------------+ #
# | libhdf5 — H5T (datatype) API
# +--------------------------------------------------------------------------+ #
"""Datatype introspection.

We use ``H5Tget_native_type`` rather than the ``H5T_NATIVE_*`` extern
globals to identify a dataset's native dtype. This keeps the FFI to pure
function lookups and sidesteps the ``H5T_NATIVE_*_g`` symbol-address
problem.

Flow used by ``reader.mojo``:
    file_type = H5Dget_type(dset)
    native_type = H5Tget_native_type(file_type, H5T_DIR_DEFAULT)
    cls = H5Tget_class(native_type)        # INTEGER / FLOAT
    sz  = H5Tget_size(native_type)         # 1 (u8/i8) / 4 (i32/f32) / 8 (i64/f64)
    sgn = H5Tget_sign(native_type)         # only meaningful for INTEGER
"""


def h5t_get_class(type_id: hid_t) raises -> c_int:
    """``H5Tget_class(hid_t type_id) -> H5T_class_t``.

    Returns one of the ``H5T_*`` class constants (INTEGER, FLOAT, ...).
    """
    return _get_dylib_function[
        lib, "H5Tget_class", def(hid_t) thin -> c_int
    ]()(type_id)


def h5t_get_size(type_id: hid_t) raises -> c_size_t:
    """``H5Tget_size(hid_t type_id) -> size_t``.

    Returns the in-memory size of one element in bytes, or 0 on error.
    """
    return _get_dylib_function[
        lib, "H5Tget_size", def(hid_t) thin -> c_size_t
    ]()(type_id)


def h5t_get_sign(type_id: hid_t) raises -> c_int:
    """``H5Tget_sign(hid_t type_id) -> H5T_sign_t``.

    Returns ``H5T_SGN_NONE`` (unsigned) or ``H5T_SGN_2`` (two's-complement
    signed). Only meaningful for integer types.
    """
    return _get_dylib_function[
        lib, "H5Tget_sign", def(hid_t) thin -> c_int
    ]()(type_id)


def h5t_get_native_type(
    type_id: hid_t, direction: c_int
) raises -> hid_t:
    """``H5Tget_native_type(hid_t type_id, H5T_direction_t direction) -> hid_t``.

    Returns a *new* hid_t for the native (host-endian, native-aligned) type
    corresponding to ``type_id``. Caller owns the returned id and must
    close it with ``h5t_close``.

    ``direction`` controls width matching for compound/array types; for
    flat scalar types ``H5T_DIR_DEFAULT`` is correct.
    """
    return _get_dylib_function[
        lib, "H5Tget_native_type", def(hid_t, c_int) thin -> hid_t
    ]()(type_id, direction)


def h5t_close(type_id: hid_t) raises -> herr_t:
    """``H5Tclose(hid_t type_id) -> herr_t``."""
    return _get_dylib_function[
        lib, "H5Tclose", def(hid_t) thin -> herr_t
    ]()(type_id)
