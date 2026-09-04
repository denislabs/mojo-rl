# +--------------------------------------------------------------------------+ #
# | libhdf5 — H5L (link) API
# +--------------------------------------------------------------------------+ #
"""Link enumeration — how we discover what is *in* a file.

Needed to ingest FOREIGN files (a HuggingFace dataset we did not write, which
carries no manifest of ours): open it, list the top-level datasets, introspect
each one's shape and dtype, and reconstruct the column set.

`H5Lget_name_by_idx` is used rather than `H5Literate`, because the latter takes
a C callback and `H5Gget_info` takes an out-struct — both awkward across this
FFI. Index-based lookup needs neither: call with increasing `n` until it
returns negative.
"""

from mojo_rl.core.bytes import string_from_bytes


from . import _get_dylib_function, c_char, c_int, c_size_t, lib, Ptr
from .h5_types import H5P_DEFAULT, hid_t, hsize_t
from .h5native import global_hid


comptime H5_INDEX_NAME = c_int(0)
"""Index links by name (alphabetical)."""

comptime H5_ITER_INC = c_int(0)
"""Iterate in increasing index order."""

comptime _NAME_BUF = 512
"""Max link-name length we handle. HDF5 permits longer; a dataset name that
long in an RL store would be pathological, and we raise rather than truncate."""


def h5l_get_name_by_idx(
    loc_id: hid_t,
    var group_name: String,
    idx_type: c_int,
    order: c_int,
    n: hsize_t,
    name: Pointer[mut=True, c_char, _],
    size: c_size_t,
    lapl_id: hid_t,
) raises -> Int64:
    """``H5Lget_name_by_idx(loc, group, idx_type, order, n, char *name,
    size_t size, hid_t lapl) -> ssize_t``.

    Returns the name length (excluding NUL), or negative when `n` is past the
    end. Passing a null `name` returns the required buffer size.

    ⚠ The array parameters are generic over the caller's origin rather than
    fixed at `MutUntrackedOrigin`. Fixing them forced every caller to write
    `.unsafe_origin_cast[MutUntrackedOrigin]()`, and that cast SEVERS the
    borrow — Mojo then destroys the caller's buffer at its last mention, which
    is the cast, and this call reads freed memory. Keeping the origin generic
    lets the caller pass a `List`'s pointer directly and keeps the list alive
    for the duration of the call; the cast to the C signature happens here,
    where the tracked parameter is still live.
    """
    return _get_dylib_function[
        lib,
        "H5Lget_name_by_idx",
        def(
            hid_t,
            Ptr[c_char, ImmOrigin(origin_of(group_name))],
            c_int,
            c_int,
            hsize_t,
            Pointer[c_char, MutUntrackedOrigin],
            c_size_t,
            hid_t,
        ) thin -> Int64,
    ]()(
        loc_id,
        group_name.as_c_string_slice().unsafe_ptr(),
        idx_type,
        order,
        n,
        name.unsafe_origin_cast[MutUntrackedOrigin](),
        size,
        lapl_id,
    )


def list_link_names(loc_id: hid_t) raises -> List[String]:
    """Every top-level link name under `loc_id`, in name order.

    Stops at the first index that errors, which is how HDF5 signals "past the
    end" for index-based lookup.
    """
    # Touch the H5open/H5Eset_auto latch before probing, so the terminating
    # error below does not dump a C stack to stderr. Cheap and idempotent.
    _ = global_hid("H5T_NATIVE_FLOAT_g")

    var names = List[String]()
    # `List` rather than `alloc`: `h5l_get_name_by_idx` raises and the overflow
    # check below raises, so the manual free leaked the buffer on both paths.
    var buf = List[c_char](length=_NAME_BUF, fill=c_char(0))
    var i = 0
    while True:
        var n = h5l_get_name_by_idx(
            loc_id,
            String("."),
            H5_INDEX_NAME,
            H5_ITER_INC,
            hsize_t(i),
            buf.unsafe_ptr(),
            c_size_t(_NAME_BUF),
            H5P_DEFAULT,
        )
        if n < 0:
            break
        if Int(n) >= _NAME_BUF:
            raise Error(
                "hdf5: link name at index " + String(i) + " is "
                + String(Int(n)) + " bytes, exceeding the "
                + String(_NAME_BUF) + "-byte buffer"
            )
        # ⚠ BYTES, not `chr` per byte — see `core/bytes.mojo`. HDF5 link
        # names are arbitrary bytes as far as the library is concerned.
        var o = List[UInt8]()
        for k in range(Int(n)):
            # ⚠ `buf` is Int8 (a C `char*`); the byte value is the
            # same bit pattern, and `UInt8` is what a byte-safe
            # String build needs.
            o.append(UInt8(buf[k]))
        names.append(string_from_bytes(o))
        i += 1
    return names^
