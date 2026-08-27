# +--------------------------------------------------------------------------+ #
# | libhdf5 — H5F (file) API
# +--------------------------------------------------------------------------+ #
"""File-level HDF5 operations: create, open, close.

Property-list arguments are passed as ``H5P_DEFAULT`` — per-dataset
configuration (chunking, compression) lives on the DCPL instead, see
``h5p.mojo``.
"""

from . import _get_dylib_function, c_char, c_int, c_uint, lib, Ptr
from .h5_types import herr_t, hid_t


def h5f_open(
    var path: String, flags: c_uint, fapl_id: hid_t
) raises -> hid_t:
    """``H5Fopen(const char *name, unsigned flags, hid_t fapl_id) -> hid_t``.

    Args:
        path: File path on disk.
        flags: Bitfield from H5F_ACC_*. Use ``H5F_ACC_RDONLY`` for read-only.
        fapl_id: File access property list. Use ``H5P_DEFAULT``.

    Returns:
        Positive ``hid_t`` on success, negative on failure.
    """
    return _get_dylib_function[
        lib,
        "H5Fopen",
        def(Ptr[c_char, ImmOrigin(origin_of(path))], c_uint, hid_t) thin -> hid_t,
    ]()(path.as_c_string_slice().unsafe_ptr(), flags, fapl_id)


def h5f_create(
    var path: String, flags: c_uint, fcpl_id: hid_t, fapl_id: hid_t
) raises -> hid_t:
    """``H5Fcreate(const char *name, unsigned flags, hid_t fcpl, hid_t fapl)``.

    Args:
        path: File path on disk.
        flags: ``H5F_ACC_TRUNC`` (overwrite) or ``H5F_ACC_EXCL`` (fail if
            the file exists). Note these are CREATE flags — passing
            ``H5F_ACC_RDWR`` here is rejected by HDF5.
        fcpl_id: File creation property list. Use ``H5P_DEFAULT``.
        fapl_id: File access property list. Use ``H5P_DEFAULT``.

    Returns:
        Positive ``hid_t`` on success, negative on failure.
    """
    return _get_dylib_function[
        lib,
        "H5Fcreate",
        def(
            Ptr[c_char, ImmOrigin(origin_of(path))], c_uint, hid_t, hid_t
        ) thin -> hid_t,
    ]()(path.as_c_string_slice().unsafe_ptr(), flags, fcpl_id, fapl_id)


def h5f_flush(object_id: hid_t, scope: c_int) raises -> herr_t:
    """``H5Fflush(hid_t object_id, H5F_scope_t scope) -> herr_t``.

    Force buffered data to disk without closing. `scope` 0 = H5F_SCOPE_LOCAL,
    1 = H5F_SCOPE_GLOBAL. Used to make a long collection run crash-resistant.
    """
    return _get_dylib_function[
        lib, "H5Fflush", def(hid_t, c_int) thin -> herr_t
    ]()(object_id, scope)


def h5f_close(file_id: hid_t) raises -> herr_t:
    """``H5Fclose(hid_t file_id) -> herr_t``.

    Returns non-negative on success.
    """
    return _get_dylib_function[
        lib, "H5Fclose", def(hid_t) thin -> herr_t
    ]()(file_id)
