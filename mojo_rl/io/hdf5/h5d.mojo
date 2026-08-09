# +--------------------------------------------------------------------------+ #
# | libhdf5 — H5D (dataset) API
# +--------------------------------------------------------------------------+ #
"""Dataset-level HDF5 operations: create, open, close, read, write, query.

The HDF5 C API uses identifier handles (hid_t) for everything — datasets,
dataspaces, datatypes. Every ``H5*get_*`` call that returns an id
*allocates a new copy* the caller must close. Wrappers here preserve
that contract.
"""

from . import _get_dylib_function, c_char, lib, Ptr
from .h5_types import herr_t, hid_t, hsize_t


def h5d_open2(
    loc_id: hid_t, var name: String, dapl_id: hid_t
) raises -> hid_t:
    """``H5Dopen2(hid_t loc_id, const char *name, hid_t dapl_id) -> hid_t``.

    Open an existing dataset by path within ``loc_id`` (typically a file id).
    """
    return _get_dylib_function[
        lib,
        "H5Dopen2",
        def(hid_t, Ptr[c_char, ImmOrigin(origin_of(name))], hid_t) thin -> hid_t,
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


def h5d_create2(
    loc_id: hid_t,
    var name: String,
    type_id: hid_t,
    space_id: hid_t,
    lcpl_id: hid_t,
    dcpl_id: hid_t,
    dapl_id: hid_t,
) raises -> hid_t:
    """``H5Dcreate2(loc, name, type_id, space_id, lcpl, dcpl, dapl) -> hid_t``.

    Args:
        loc_id: File (or group) handle.
        name: Dataset path within ``loc_id``.
        type_id: Datatype — see ``h5native.native_type[dtype]()``.
        space_id: Dataspace from ``h5s_create_simple``. If any maxdim is
            ``H5S_UNLIMITED``, ``dcpl_id`` MUST set a chunk shape.
        lcpl_id: Link creation property list. Use ``H5P_DEFAULT``.
        dcpl_id: Dataset creation property list — chunking/compression.
        dapl_id: Dataset access property list. Use ``H5P_DEFAULT``.

    Caller owns the returned id and must close it with ``h5d_close``.
    """
    return _get_dylib_function[
        lib,
        "H5Dcreate2",
        def(
            hid_t,
            Ptr[c_char, ImmOrigin(origin_of(name))],
            hid_t,
            hid_t,
            hid_t,
            hid_t,
            hid_t,
        ) thin -> hid_t,
    ]()(
        loc_id,
        name.as_c_string_slice().unsafe_ptr(),
        type_id,
        space_id,
        lcpl_id,
        dcpl_id,
        dapl_id,
    )


def h5d_write(
    dset_id: hid_t,
    mem_type_id: hid_t,
    mem_space_id: hid_t,
    file_space_id: hid_t,
    dxpl_id: hid_t,
    buf: Pointer[NoneType, MutAnyOrigin],
) raises -> herr_t:
    """``H5Dwrite(dset, mem_type, mem_space, file_space, dxpl, const void *buf)``.

    Mirror of ``h5d_read``. ``file_space_id`` carries the hyperslab that
    selects where in the (possibly just-extended) dataset the rows land.
    """
    return _get_dylib_function[
        lib,
        "H5Dwrite",
        def(
            hid_t,
            hid_t,
            hid_t,
            hid_t,
            hid_t,
            Pointer[NoneType, MutAnyOrigin],
        ) thin -> herr_t,
    ]()(dset_id, mem_type_id, mem_space_id, file_space_id, dxpl_id, buf)


def h5d_set_extent(
    dset_id: hid_t, size: Pointer[hsize_t, MutUntrackedOrigin]
) raises -> herr_t:
    """``H5Dset_extent(hid_t dset_id, const hsize_t size[]) -> herr_t``.

    Grow (or shrink) a chunked dataset in place. This is what makes append
    possible: extend dim-0 by the batch, then write into the new tail rows.
    Any previously-obtained dataspace id is stale afterwards — re-fetch with
    ``h5d_get_space``.
    """
    return _get_dylib_function[
        lib,
        "H5Dset_extent",
        def(hid_t, Pointer[hsize_t, MutUntrackedOrigin]) thin -> herr_t,
    ]()(dset_id, size)


def h5d_read(
    dset_id: hid_t,
    mem_type_id: hid_t,
    mem_space_id: hid_t,
    file_space_id: hid_t,
    dxpl_id: hid_t,
    buf: Pointer[NoneType, MutAnyOrigin],
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
            Pointer[NoneType, MutAnyOrigin],
        ) thin -> herr_t,
    ]()(dset_id, mem_type_id, mem_space_id, file_space_id, dxpl_id, buf)
