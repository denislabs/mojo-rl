# +--------------------------------------------------------------------------+ #
# | libhdf5 — H5S (dataspace) API
# +--------------------------------------------------------------------------+ #
"""Dataspace operations: shape introspection and hyperslab selection.

For LeWM PushT we need just enough to read either an entire flat column
or a contiguous row range ``[start:end, ...]`` from a chunked dataset.
"""


def h5s_create_simple(
    rank: c_int,
    dims: UnsafePointer[hsize_t, MutAnyOrigin],
    maxdims: UnsafePointer[hsize_t, MutAnyOrigin],
) raises -> hid_t:
    """``H5Screate_simple(int rank, hsize_t *dims, hsize_t *maxdims) -> hid_t``.

    Create a new simple dataspace. ``maxdims`` may be a null pointer to
    mean "same as dims".

    Caller owns the returned id and must close it with ``h5s_close``.
    """
    return _get_dylib_function[
        lib,
        "H5Screate_simple",
        def(
            c_int,
            UnsafePointer[hsize_t, MutAnyOrigin],
            UnsafePointer[hsize_t, MutAnyOrigin],
        ) thin -> hid_t,
    ]()(rank, dims, maxdims)


def h5s_close(space_id: hid_t) raises -> herr_t:
    """``H5Sclose(hid_t space_id) -> herr_t``."""
    return _get_dylib_function[
        lib, "H5Sclose", def(hid_t) thin -> herr_t
    ]()(space_id)


def h5s_get_simple_extent_ndims(space_id: hid_t) raises -> c_int:
    """``H5Sget_simple_extent_ndims(hid_t space_id) -> int``.

    Returns the rank of the dataspace, or a negative value on error.
    """
    return _get_dylib_function[
        lib, "H5Sget_simple_extent_ndims", def(hid_t) thin -> c_int
    ]()(space_id)


def h5s_get_simple_extent_dims(
    space_id: hid_t,
    dims: UnsafePointer[hsize_t, MutAnyOrigin],
    maxdims: UnsafePointer[hsize_t, MutAnyOrigin],
) raises -> c_int:
    """``H5Sget_simple_extent_dims(hid_t, hsize_t *dims, hsize_t *maxdims)``.

    Fills ``dims`` (and optionally ``maxdims``) with the dataspace shape.
    The caller is responsible for allocating buffers of size ``ndims``.
    Returns ndims on success, negative on error.
    """
    return _get_dylib_function[
        lib,
        "H5Sget_simple_extent_dims",
        def(
            hid_t,
            UnsafePointer[hsize_t, MutAnyOrigin],
            UnsafePointer[hsize_t, MutAnyOrigin],
        ) thin -> c_int,
    ]()(space_id, dims, maxdims)


def h5s_select_hyperslab(
    space_id: hid_t,
    op: c_int,
    start: UnsafePointer[hsize_t, MutAnyOrigin],
    stride: UnsafePointer[hsize_t, MutAnyOrigin],
    count: UnsafePointer[hsize_t, MutAnyOrigin],
    block: UnsafePointer[hsize_t, MutAnyOrigin],
) raises -> herr_t:
    """``H5Sselect_hyperslab(hid_t, H5S_seloper_t op, const hsize_t *start,
    *stride, *count, *block) -> herr_t``.

    Select a strided hyperslab on the dataspace. ``stride`` and ``block``
    may be null for contiguous selections.
    """
    return _get_dylib_function[
        lib,
        "H5Sselect_hyperslab",
        def(
            hid_t,
            c_int,
            UnsafePointer[hsize_t, MutAnyOrigin],
            UnsafePointer[hsize_t, MutAnyOrigin],
            UnsafePointer[hsize_t, MutAnyOrigin],
            UnsafePointer[hsize_t, MutAnyOrigin],
        ) thin -> herr_t,
    ]()(space_id, op, start, stride, count, block)
