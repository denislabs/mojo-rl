# +--------------------------------------------------------------------------+ #
# | libhdf5 — H5E (error stack) API
# +--------------------------------------------------------------------------+ #
"""Suppress libhdf5's automatic error printing.

By default libhdf5 dumps a multi-frame C error stack to stderr for every
failed call, *in addition to* returning a negative code. Every wrapper here
already turns that code into a Mojo `Error` carrying its own context, so the
dump is pure noise — and actively misleading where a negative return is the
NORMAL control flow, as in `list_link_names`, which walks link indices until
one fails to learn where the list ends. Without this, opening any store
printed ten lines of "link not found" and looked broken.

This is what h5py does too: turn off auto-printing, surface errors through the
host language.

`H5E_DEFAULT` is `(hid_t)0` in the C headers — a plain constant, not one of
the global variables that `h5native.mojo` has to dlsym.
"""

from std.ffi import _get_dylib_function

from . import lib
from .h5_types import herr_t, hid_t


comptime H5E_DEFAULT = hid_t(0)


def h5e_set_auto2_off() raises -> herr_t:
    """``H5Eset_auto2(hid_t estack_id, H5E_auto2_t func, void *client_data)``
    with a NULL callback — disables automatic stack printing.
    """
    # `UnsafePointer` is non-nullable at comptime; the runtime-Int overload
    # still yields a real NULL. Same trick as `reader.mojo::_null_ptr`.
    var addr: Int = 0
    var null_fn = UnsafePointer[NoneType, MutAnyOrigin](unsafe_from_address=addr)
    var null_data = UnsafePointer[NoneType, MutAnyOrigin](unsafe_from_address=addr)
    return _get_dylib_function[
        lib,
        "H5Eset_auto2",
        def(
            hid_t,
            UnsafePointer[NoneType, MutAnyOrigin],
            UnsafePointer[NoneType, MutAnyOrigin],
        ) thin -> herr_t,
    ]()(H5E_DEFAULT, null_fn, null_data)
