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
    # `Pointer` is non-nullable at comptime; the runtime-Int overload
    # still yields a real NULL. Same trick as `reader.mojo::_null_ptr`.
    var addr: Int = 0
    var null_fn = Pointer[NoneType, MutAnyOrigin](unsafe_from_address=addr)
    var null_data = Pointer[NoneType, MutAnyOrigin](unsafe_from_address=addr)
    return _get_dylib_function[
        lib,
        "H5Eset_auto2",
        def(
            hid_t,
            Pointer[NoneType, MutAnyOrigin],
            Pointer[NoneType, MutAnyOrigin],
        ) thin -> herr_t,
    ]()(H5E_DEFAULT, null_fn, null_data)



def h5e_print_stack(context: String):
    """Print libhdf5's CURRENT error stack to stderr, under a heading.

    ⚠⚠ FOR WRITE FAILURES ONLY. Auto-printing is off (see the module header)
    because many failed calls are normal control flow. A failed WRITE never
    is, and its return code alone says nothing: the first 50-episode store
    died on a rented box as "H5Fflush failed: ret=-1", with 11 GB free and the
    real cause sitting in a stack nobody printed. The writer calls this
    before raising, so the next failure names itself.

    `H5Eprint2(H5E_DEFAULT, NULL)` prints to stderr. Never raises: a
    diagnostic that fails must not replace the error it was describing.
    """
    try:
        print("── libhdf5 error stack: " + context + " ──")
        var addr: Int = 0
        var null_stream = Pointer[NoneType, MutAnyOrigin](unsafe_from_address=addr)
        _ = _get_dylib_function[
            lib,
            "H5Eprint2",
            def(hid_t, Pointer[NoneType, MutAnyOrigin]) thin -> herr_t,
        ]()(H5E_DEFAULT, null_stream)
    except:
        pass
