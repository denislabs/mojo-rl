# +--------------------------------------------------------------------------+ #
# | libhdf5 — H5D (dataset) API
# +--------------------------------------------------------------------------+ #
"""Dataset-level HDF5 operations: open, close, read, query.

The HDF5 C API uses identifier handles (hid_t) for everything — datasets,
dataspaces, datatypes. Every ``H5*get_*`` call that returns an id
*allocates a new copy* the caller must close. Wrappers here preserve
that contract.
"""


def h5d_open2(
    loc_id: hid_t, var name: String, dapl_id: hid_t
) raises -> hid_t:
    """``H5Dopen2(hid_t loc_id, const char *name, hid_t dapl_id) -> hid_t``.

    Open an existing dataset by path within ``loc_id`` (typically a file id).
    """
    return _get_dylib_function[
        lib,
        "H5Dopen2",
        def(hid_t, Ptr[c_char, ImmutOrigin(origin_of(name))], hid_t) thin -> hid_t,
    ]()(loc_id, name.as_c_string_slice().unsafe_ptr(), dapl_id)


def h5d_close(dset_id: hid_t) raises -> herr_t:
    """``H5Dclose(hid_t dset_id) -> herr_t``."""
    return _get_dylib_function[
        lib, "H5Dclose", def(hid_t) thin -> herr_t
    ]()(dset_id)


def h5d_get_space(dset_id: hid_t) raises -> hid_t:
    """``H5Dget_space(hid_t dset_id) -> hid_t``.

    Returns a *new* dataspace id describing the dataset's on-disk extent.
    Caller owns the returned id and must close it with ``h5s_close``.
    """
    return _get_dylib_function[
        lib, "H5Dget_space", def(hid_t) thin -> hid_t
    ]()(dset_id)


def h5d_get_type(dset_id: hid_t) raises -> hid_t:
    """``H5Dget_type(hid_t dset_id) -> hid_t``.

    Returns a *new* datatype id describing the dataset's on-disk type.
    Caller owns the returned id and must close it with ``h5t_close``.
    """
    return _get_dylib_function[
        lib, "H5Dget_type", def(hid_t) thin -> hid_t
    ]()(dset_id)


def h5d_read(
    dset_id: hid_t,
    mem_type_id: hid_t,
    mem_space_id: hid_t,
    file_space_id: hid_t,
    dxpl_id: hid_t,
    buf: UnsafePointer[NoneType, MutAnyOrigin],
) raises -> herr_t:
    """``H5Dread(dset, mem_type, mem_space, file_space, dxpl, void *buf)``.

    Read data from a dataset into a host buffer.

    Args:
        dset_id: Dataset handle from ``h5d_open2``.
        mem_type_id: Memory datatype id (typically from ``h5t_get_native_type``).
        mem_space_id: Memory dataspace, or ``H5S_ALL`` for a contiguous buffer
            matching the file selection.
        file_space_id: File dataspace, or ``H5S_ALL`` for the full dataset.
        dxpl_id: Data transfer property list. Use ``H5P_DEFAULT``.
        buf: Host buffer, large enough to hold the selected elements.

    Returns non-negative on success.
    """
    return _get_dylib_function[
        lib,
        "H5Dread",
        def(
            hid_t,
            hid_t,
            hid_t,
            hid_t,
            hid_t,
            UnsafePointer[NoneType, MutAnyOrigin],
        ) thin -> herr_t,
    ]()(dset_id, mem_type_id, mem_space_id, file_space_id, dxpl_id, buf)
