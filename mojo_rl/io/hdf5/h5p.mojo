# +--------------------------------------------------------------------------+ #
# | libhdf5 — H5P (property list) API
# +--------------------------------------------------------------------------+ #
"""Property lists — the write path's configuration surface.

Distinct from `h5pl.mojo`, which wraps H5PL (the *plugin* API used to
register hdf5plugin's filter directory).

A dataset creation property list (DCPL) is what carries chunking,
compression and fill-value settings into `H5Dcreate2`. Chunking is not
optional for us: an extendable dataset (any `H5S_UNLIMITED` axis) is
rejected by HDF5 unless the DCPL sets a chunk shape.

Like the native datatypes, the property-list *class* ids are global
variables (`H5P_CLS_DATASET_CREATE_ID_g`), not constants — see
`h5native.mojo` for why that matters and how it is resolved.
"""

from std.memory import alloc

from . import _get_dylib_function, c_int, c_uint, lib
from .h5_types import herr_t, hid_t, hsize_t
from .h5native import global_hid


def h5p_dataset_create_class() raises -> hid_t:
    """The DCPL class id (`H5P_DATASET_CREATE` in C)."""
    return global_hid("H5P_CLS_DATASET_CREATE_ID_g")


def h5p_create(cls_id: hid_t) raises -> hid_t:
    """``H5Pcreate(hid_t cls_id) -> hid_t``.

    Caller owns the returned id and must close it with `h5p_close`.
    """
    return _get_dylib_function[
        lib, "H5Pcreate", def(hid_t) thin -> hid_t
    ]()(cls_id)


def h5p_close(plist_id: hid_t) raises -> herr_t:
    """``H5Pclose(hid_t plist_id) -> herr_t``."""
    return _get_dylib_function[
        lib, "H5Pclose", def(hid_t) thin -> herr_t
    ]()(plist_id)


def h5p_set_chunk(
    plist_id: hid_t,
    ndims: c_int,
    dim: Pointer[mut=False, hsize_t, _],
) raises -> herr_t:
    """``H5Pset_chunk(hid_t plist_id, int ndims, const hsize_t *dim)``.

    Sets the chunk shape on a DCPL. Required for any dataset with an
    unlimited axis or any compression filter.

    ⚠ The array parameters are immutable and generic over the caller's origin rather than
    fixed at `MutUntrackedOrigin`. Fixing them forced every caller to write
    `.unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()`, and that cast SEVERS the
    borrow — Mojo then destroys the caller's buffer at its last mention, which
    is the cast, and this call reads freed memory. Keeping the origin generic
    lets the caller pass a `List`'s pointer directly and keeps the list alive
    for the duration of the call; the cast to the C signature happens here,
    where the tracked parameter is still live.
    """
    return _get_dylib_function[
        lib,
        "H5Pset_chunk",
        def(
            hid_t, c_int, Pointer[hsize_t, MutUntrackedOrigin]
        ) thin -> herr_t,
    ]()(plist_id, ndims, dim.unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]())


def h5p_set_deflate(plist_id: hid_t, level: c_uint) raises -> herr_t:
    """``H5Pset_deflate(hid_t plist_id, unsigned level) -> herr_t``.

    gzip/zlib compression, level 0-9. Always available (built into
    libhdf5), unlike the hdf5plugin filters, so it is the portable choice
    for a file we expect other tools to read.
    """
    return _get_dylib_function[
        lib, "H5Pset_deflate", def(hid_t, c_uint) thin -> herr_t
    ]()(plist_id, level)


def h5p_set_shuffle(plist_id: hid_t) raises -> herr_t:
    """``H5Pset_shuffle(hid_t plist_id) -> herr_t``.

    Byte-shuffle filter. Materially improves deflate ratios on arrays of
    same-typed numbers; must be set BEFORE the compression filter to run
    ahead of it in the pipeline.
    """
    return _get_dylib_function[
        lib, "H5Pset_shuffle", def(hid_t) thin -> herr_t
    ]()(plist_id)
