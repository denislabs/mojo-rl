# +--------------------------------------------------------------------------+ #
# | libhdf5 — high-level write API
# +--------------------------------------------------------------------------+ #
"""High-level HDF5 writer: ``H5Writer`` and ``H5DatasetWriter``.

Mirror of ``reader.mojo``. Handles are owned and released on destruction;
Movable but not Copyable.

Every dataset is created CHUNKED with an UNLIMITED leading axis, so a
collection run streams to disk instead of being held in RAM and dumped at
the end. Append is: extend dim-0, select the new tail rows, write.

    var w = H5Writer("out.h5")
    var ds = w.create[DType.float32]("qpos", row_dim=9, chunk_rows=4096)
    ds.append[DType.float32](rows_ptr, n_rows=256)   # repeat…
    ds.flush()

Row-major convention throughout: a dataset is ``[N, row_dim]`` where dim-0
grows. ``row_dim = 1`` gives a flat ``[N]`` column (rank 1), which is what
scalar columns like ``reward`` / ``done`` want — rank 1 rather than
``[N, 1]`` so h5py and our own reader both see a plain vector.
"""

from std.memory import alloc, Pointer

from . import c_int, c_uint
from .h5_types import (
    H5F_ACC_TRUNC,
    H5P_DEFAULT,
    H5S_ALL,
    H5S_SELECT_SET,
    H5S_UNLIMITED,
    hid_t,
    hsize_t,
)
from .h5d import (
    h5d_close, h5d_create2, h5d_get_space, h5d_set_extent, h5d_write,
)
from .h5f import h5f_close, h5f_create, h5f_flush
from .h5p import (
    h5p_close, h5p_create, h5p_dataset_create_class, h5p_set_chunk,
    h5p_set_deflate, h5p_set_shuffle,
)
from .h5s import h5s_close, h5s_create_simple, h5s_select_hyperslab
from .h5native import native_type


struct H5DatasetWriter(Movable):
    """Owning handle to a growable 2-D (or flat 1-D) HDF5 dataset.

    ``n_rows`` tracks the current dim-0 extent so `append` knows where the
    tail is without re-querying the file each call.
    """

    var dset_id: hid_t
    var type_id: hid_t
    var row_dim: Int
    var n_rows: Int
    var rank: Int

    def __init__(
        out self, dset_id: hid_t, type_id: hid_t, row_dim: Int, rank: Int
    ):
        self.dset_id = dset_id
        self.type_id = type_id
        self.row_dim = row_dim
        self.n_rows = 0
        self.rank = rank

    def __deinit__(deinit self):
        if self.dset_id > 0:
            try:
                _ = h5d_close(self.dset_id)
            except:
                pass

    def append[
        dtype: DType
    ](
        mut self,
        buf: Pointer[Scalar[dtype], MutAnyOrigin],
        n_rows: Int,
    ) raises:
        """Append ``n_rows`` rows from ``buf`` to the end of the dataset.

        ``buf`` must hold ``n_rows * row_dim`` elements of ``dtype``, laid
        out row-major and contiguous.
        """
        if n_rows <= 0:
            return

        var old_n = self.n_rows
        var new_n = old_n + n_rows

        # 1. Grow dim-0. Any dataspace obtained before this is stale.
        var ext = alloc[hsize_t](self.rank)
        ext[unsafe_offset=0] = hsize_t(new_n)
        if self.rank > 1:
            ext[unsafe_offset=1] = hsize_t(self.row_dim)
        var ext_ret = h5d_set_extent(self.dset_id, ext)
        ext.unsafe_free()
        if ext_ret < 0:
            raise Error("H5Dset_extent failed: ret=" + String(Int(ext_ret)))

        # 2. Select the new tail rows [old_n, new_n) in the FILE space.
        var file_space = h5d_get_space(self.dset_id)
        if file_space < 0:
            raise Error("H5Dget_space failed after extent")

        var start = alloc[hsize_t](self.rank)
        var count = alloc[hsize_t](self.rank)
        var unit = alloc[hsize_t](self.rank)
        start[unsafe_offset=0] = hsize_t(old_n)
        count[unsafe_offset=0] = hsize_t(n_rows)
        unit[unsafe_offset=0] = hsize_t(1)
        if self.rank > 1:
            start[unsafe_offset=1] = hsize_t(0)
            count[unsafe_offset=1] = hsize_t(self.row_dim)
            unit[unsafe_offset=1] = hsize_t(1)

        var sel = h5s_select_hyperslab(
            file_space, H5S_SELECT_SET, start, unit, count, unit
        )
        if sel < 0:
            _ = h5s_close(file_space)
            start.unsafe_free(); count.unsafe_free(); unit.unsafe_free()
            raise Error("H5Sselect_hyperslab failed (append)")

        # 3. Memory space is the compact [n_rows, row_dim] block in `buf`.
        var mem_space = h5s_create_simple(c_int(self.rank), count, count)
        if mem_space < 0:
            _ = h5s_close(file_space)
            start.unsafe_free(); count.unsafe_free(); unit.unsafe_free()
            raise Error("H5Screate_simple failed (append)")

        var ret = h5d_write(
            self.dset_id,
            self.type_id,
            mem_space,
            file_space,
            H5P_DEFAULT,
            buf.unsafe_bitcast[NoneType](),
        )
        _ = h5s_close(mem_space)
        _ = h5s_close(file_space)
        start.unsafe_free(); count.unsafe_free(); unit.unsafe_free()

        if ret < 0:
            raise Error("H5Dwrite failed: ret=" + String(Int(ret)))

        self.n_rows = new_n


struct H5Writer(Movable):
    """Owning handle to an HDF5 file open for writing (created/truncated)."""

    var file_id: hid_t

    def __init__(out self, var path: String) raises:
        self.file_id = h5f_create(
            path^, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT
        )
        if self.file_id < 0:
            raise Error(
                "H5Fcreate failed: hid_t=" + String(Int(self.file_id))
            )

    def __deinit__(deinit self):
        if self.file_id > 0:
            try:
                _ = h5f_close(self.file_id)
            except:
                pass

    def create[
        dtype: DType
    ](
        self,
        var name: String,
        row_dim: Int = 1,
        chunk_rows: Int = 4096,
        deflate: Int = 0,
    ) raises -> H5DatasetWriter:
        """Create an empty, growable dataset.

        Args:
            name: Dataset path in the file (e.g. ``"qpos"``).
            row_dim: Elements per row. ``1`` produces a rank-1 ``[N]``
                dataset; anything larger a rank-2 ``[N, row_dim]``.
            chunk_rows: Rows per HDF5 chunk. This is the unit of I/O and
                of compression; it also sets the granularity at which a
                reader can pull a slab. 4096 rows is a reasonable default
                for state columns.
            deflate: Gzip level 0-9. ``0`` disables compression (plus the
                shuffle filter). Worth it for pixels/uint8; usually a net
                loss for fp32 state columns, where decompression costs
                more than the bytes saved.
        """
        if row_dim < 1:
            raise Error("H5Writer.create: row_dim must be >= 1")
        if chunk_rows < 1:
            raise Error("H5Writer.create: chunk_rows must be >= 1")

        var rank = 1 if row_dim == 1 else 2
        var type_id = native_type[dtype]()

        # Dataspace: starts empty, dim-0 unlimited.
        var dims = alloc[hsize_t](rank)
        var maxdims = alloc[hsize_t](rank)
        dims[unsafe_offset=0] = hsize_t(0)
        maxdims[unsafe_offset=0] = H5S_UNLIMITED
        if rank > 1:
            dims[unsafe_offset=1] = hsize_t(row_dim)
            maxdims[unsafe_offset=1] = hsize_t(row_dim)

        var space = h5s_create_simple(c_int(rank), dims, maxdims)
        dims.unsafe_free()
        maxdims.unsafe_free()
        if space < 0:
            raise Error("H5Screate_simple failed (create)")

        # DCPL: chunking is MANDATORY for an unlimited axis.
        var dcpl = h5p_create(h5p_dataset_create_class())
        if dcpl < 0:
            _ = h5s_close(space)
            raise Error("H5Pcreate failed")

        var chunk = alloc[hsize_t](rank)
        chunk[unsafe_offset=0] = hsize_t(chunk_rows)
        if rank > 1:
            chunk[unsafe_offset=1] = hsize_t(row_dim)
        var ck = h5p_set_chunk(dcpl, c_int(rank), chunk)
        chunk.unsafe_free()
        if ck < 0:
            _ = h5p_close(dcpl)
            _ = h5s_close(space)
            raise Error("H5Pset_chunk failed")

        if deflate > 0:
            # Shuffle must precede deflate so it runs first in the pipeline.
            _ = h5p_set_shuffle(dcpl)
            var dz = h5p_set_deflate(dcpl, c_uint(deflate))
            if dz < 0:
                _ = h5p_close(dcpl)
                _ = h5s_close(space)
                raise Error("H5Pset_deflate failed")

        var dset = h5d_create2(
            self.file_id, name^, type_id, space, H5P_DEFAULT, dcpl, H5P_DEFAULT
        )
        _ = h5p_close(dcpl)
        _ = h5s_close(space)
        if dset < 0:
            raise Error("H5Dcreate2 failed: hid_t=" + String(Int(dset)))

        return H5DatasetWriter(dset, type_id, row_dim, rank)

    def flush(self) raises:
        """Force buffered data to disk without closing the file."""
        var ret = h5f_flush(self.file_id, c_int(0))  # H5F_SCOPE_LOCAL
        if ret < 0:
            raise Error("H5Fflush failed: ret=" + String(Int(ret)))
