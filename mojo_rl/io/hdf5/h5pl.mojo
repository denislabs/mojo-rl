# +--------------------------------------------------------------------------+ #
# | libhdf5 — H5PL (plugin path) API
# +--------------------------------------------------------------------------+ #
"""Plugin path management.

Compression filters (Blosc/LZ4/ZSTD/BSHUF/...) are distributed by the
``hdf5plugin`` Python package as ``.dylib`` / ``.so`` files. ``H5PLprepend``
inserts a directory at the head of libhdf5's plugin search path at runtime,
which means we don't need to set ``HDF5_PLUGIN_PATH`` in the shell env.

Called once from ``H5File.__init__`` (see ``reader.mojo``).
"""


def h5pl_prepend(var path: String) raises -> herr_t:
    """``H5PLprepend(const char *plugin_path) -> herr_t``.

    Prepend a directory to libhdf5's plugin search path. Returns
    non-negative on success.
    """
    return _get_dylib_function[
        lib,
        "H5PLprepend",
        def(Ptr[c_char, ImmutAnyOrigin]) thin -> herr_t,
    ]()(path.as_c_string_slice().unsafe_ptr())
