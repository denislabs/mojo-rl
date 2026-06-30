# +--------------------------------------------------------------------------+ #
# | libhdf5 — H5F (file) API
# +--------------------------------------------------------------------------+ #
"""File-level HDF5 operations: open, close.

Only the read-only subset is exposed since the LeWM PushT loader never
writes. Property-list arguments are always passed as ``H5P_DEFAULT``.
"""


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
        def(Ptr[c_char, ImmutOrigin(origin_of(path))], c_uint, hid_t) thin -> hid_t,
    ]()(path.as_c_string_slice().unsafe_ptr(), flags, fapl_id)


def h5f_close(file_id: hid_t) raises -> herr_t:
    """``H5Fclose(hid_t file_id) -> herr_t``.

    Returns non-negative on success.
    """
    return _get_dylib_function[
        lib, "H5Fclose", def(hid_t) thin -> herr_t
    ]()(file_id)
