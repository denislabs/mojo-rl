# +--------------------------------------------------------------------------+ #
# | libhdf5 — high-level read API
# +--------------------------------------------------------------------------+ #
"""High-level HDF5 reader: ``H5File`` and ``H5Dataset``.

These structs own their underlying ``hid_t`` handles and release them
on destruction. Movable but not Copyable — moves transfer the handles,
copies are forbidden.

Read flow:
    var f = H5File("data.h5")
    var ds = f.open_dataset("pixels")
    print(ds.dims, ds.dtype_class, ds.elem_size)
    var buf = alloc[Scalar[DType.uint8]](ds.n_elements())
    ds.read_all(buf)
    ...
    buf.free()
"""

from std.memory import alloc


struct H5File(Movable):
    """Owning handle to an open HDF5 file (read-only)."""

    var file_id: hid_t

    def __init__(out self, var path: String) raises:
        # Register hdf5plugin's filter directory so compressed datasets
        # decode without HDF5_PLUGIN_PATH being set in the shell env.
        # H5PLprepend is idempotent enough — duplicate paths are harmless.
        try:
            _ = h5pl_prepend(String(HDF5_PLUGIN_PATH))
        except:
            pass

        self.file_id = h5f_open(path^, H5F_ACC_RDONLY, H5P_DEFAULT)
        if self.file_id < 0:
            raise Error(
                "H5Fopen failed: hid_t=" + String(Int(self.file_id))
            )

    def __del__(deinit self):
        if self.file_id > 0:
            try:
                _ = h5f_close(self.file_id)
            except:
                pass

    def open_dataset(self, var name: String) raises -> H5Dataset:
        """Open a dataset by path within the file (e.g. ``"pixels"``)."""
        var dset_id = h5d_open2(self.file_id, name^, H5P_DEFAULT)
        if dset_id < 0:
            raise Error(
                "H5Dopen2 failed: hid_t=" + String(Int(dset_id))
            )
        return H5Dataset(dset_id)


struct H5Dataset(Movable):
    """Owning handle to an open HDF5 dataset.

    On construction this reads the dataset's shape and computes the native
    type for in-memory reads. ``dims`` reports the full extent; ``elem_size``
    is bytes per element; ``dtype_class``/``signedness`` describe the dtype.
    """

    var dset_id: hid_t
    var native_type_id: hid_t
    var dims: List[hsize_t]
    var dtype_class: c_int
    """One of ``H5T_INTEGER`` / ``H5T_FLOAT`` / etc."""

    var elem_size: Int
    """Bytes per element (1 = u8/i8, 4 = i32/f32, 8 = i64/f64)."""

    var signedness: c_int
    """``H5T_SGN_NONE`` (unsigned) or ``H5T_SGN_2`` (signed). Only meaningful
    when ``dtype_class == H5T_INTEGER``."""

    def __init__(out self, dset_id: hid_t) raises:
        if dset_id < 0:
            raise Error("H5Dataset: invalid dset_id")
        self.dset_id = dset_id

        # ── shape ──────────────────────────────────────────────────────
        var space_id = h5d_get_space(dset_id)
        if space_id < 0:
            raise Error("H5Dget_space failed")
        var ndims_c = h5s_get_simple_extent_ndims(space_id)
        if ndims_c < 0:
            _ = h5s_close(space_id)
            raise Error("H5Sget_simple_extent_ndims failed")
        var ndims = Int(ndims_c)

        var dims_buf = alloc[hsize_t](ndims)
        for i in range(ndims):
            dims_buf[i] = 0
        _ = h5s_get_simple_extent_dims(
            space_id,
            dims_buf,
            UnsafePointer[hsize_t, MutAnyOrigin](unsafe_from_address=0),
        )
        _ = h5s_close(space_id)

        var dims = List[hsize_t](capacity=ndims)
        for i in range(ndims):
            dims.append(dims_buf[i])
        dims_buf.free()
        self.dims = dims^

        # ── dtype (file → native) ──────────────────────────────────────
        var file_type_id = h5d_get_type(dset_id)
        if file_type_id < 0:
            raise Error("H5Dget_type failed")
        var native_type_id = h5t_get_native_type(
            file_type_id, H5T_DIR_DEFAULT
        )
        _ = h5t_close(file_type_id)
        if native_type_id < 0:
            raise Error("H5Tget_native_type failed")

        self.native_type_id = native_type_id
        self.dtype_class = h5t_get_class(native_type_id)
        self.elem_size = Int(h5t_get_size(native_type_id))
        if self.dtype_class == H5T_INTEGER:
            self.signedness = h5t_get_sign(native_type_id)
        else:
            self.signedness = H5T_SGN_NONE

    def __del__(deinit self):
        if self.dset_id > 0:
            try:
                _ = h5d_close(self.dset_id)
            except:
                pass
        if self.native_type_id > 0:
            try:
                _ = h5t_close(self.native_type_id)
            except:
                pass

    def ndim(self) -> Int:
        """Rank of the dataset."""
        return len(self.dims)

    def n_elements(self) -> Int:
        """Total element count = product of all dims."""
        var n = 1
        for i in range(len(self.dims)):
            n *= Int(self.dims[i])
        return n

    def n_bytes(self) -> Int:
        """Total in-memory size in bytes = ``n_elements() * elem_size``."""
        return self.n_elements() * self.elem_size

    def read_all[
        dtype: DType
    ](self, buf: UnsafePointer[Scalar[dtype], MutAnyOrigin]) raises:
        """Read the entire dataset into ``buf``.

        ``buf`` must point to at least ``n_bytes()`` bytes. ``dtype`` must
        match the dataset's element size and class (caller's responsibility —
        check via ``dtype_class`` / ``elem_size`` / ``signedness``).
        """
        var ret = h5d_read(
            self.dset_id,
            self.native_type_id,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            buf.bitcast[NoneType](),
        )
        if ret < 0:
            raise Error("H5Dread failed: ret=" + String(Int(ret)))

    def read_range[
        dtype: DType
    ](
        self,
        start: Int,
        end: Int,
        buf: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ) raises:
        """Read rows ``[start:end, :, :, ...]`` into ``buf``.

        For a 2D dataset of shape ``[N, F]``, reads a contiguous block
        of ``(end-start) * F`` elements. ``buf`` must be large enough.
        """
        var ndims = len(self.dims)
        if ndims == 0:
            raise Error("read_range: dataset is scalar (rank 0)")
        if start < 0 or end > Int(self.dims[0]) or end < start:
            raise Error(
                "read_range: out of bounds (start=" + String(start)
                + " end=" + String(end)
                + " dim0=" + String(Int(self.dims[0])) + ")"
            )

        # start_arr = [start, 0, 0, ...]; count_arr = [end-start, dim1, dim2, ...]
        var start_arr = alloc[hsize_t](ndims)
        var count_arr = alloc[hsize_t](ndims)
        start_arr[0] = hsize_t(start)
        count_arr[0] = hsize_t(end - start)
        for i in range(1, ndims):
            start_arr[i] = 0
            count_arr[i] = self.dims[i]

        var file_space = h5d_get_space(self.dset_id)
        if file_space < 0:
            start_arr.free()
            count_arr.free()
            raise Error("H5Dget_space failed")

        var sel_ret = h5s_select_hyperslab(
            file_space,
            H5S_SELECT_SET,
            start_arr,
            UnsafePointer[hsize_t, MutAnyOrigin](unsafe_from_address=0),
            count_arr,
            UnsafePointer[hsize_t, MutAnyOrigin](unsafe_from_address=0),
        )
        if sel_ret < 0:
            _ = h5s_close(file_space)
            start_arr.free()
            count_arr.free()
            raise Error("H5Sselect_hyperslab failed")

        var mem_space = h5s_create_simple(
            c_int(ndims),
            count_arr,
            UnsafePointer[hsize_t, MutAnyOrigin](unsafe_from_address=0),
        )
        if mem_space < 0:
            _ = h5s_close(file_space)
            start_arr.free()
            count_arr.free()
            raise Error("H5Screate_simple failed")

        var read_ret = h5d_read(
            self.dset_id,
            self.native_type_id,
            mem_space,
            file_space,
            H5P_DEFAULT,
            buf.bitcast[NoneType](),
        )
        _ = h5s_close(mem_space)
        _ = h5s_close(file_space)
        start_arr.free()
        count_arr.free()

        if read_ret < 0:
            raise Error("H5Dread failed: ret=" + String(Int(read_ret)))

    def read_strided[
        dtype: DType
    ](
        self,
        start: Int,
        count: Int,
        stride: Int,
        buf: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ) raises:
        """Strided dim-0 read: select rows ``[start, start+stride, …,
        start+(count-1)*stride]`` and write them contiguously into ``buf``.

        Equivalent to ``dataset[start:start + count*stride : stride, …]``.
        For ``stride == 1`` use ``read_range`` (one hyperslab op vs two).

        ``buf`` must hold ``count * prod(dims[1:])`` elements of ``dtype``.
        """
        var ndims = len(self.dims)
        if ndims == 0:
            raise Error("read_strided: dataset is scalar")
        if stride <= 0:
            raise Error("read_strided: stride must be positive")
        if count <= 0:
            raise Error("read_strided: count must be positive")
        if start < 0 or start + (count - 1) * stride >= Int(self.dims[0]):
            raise Error(
                "read_strided: out of bounds (start=" + String(start)
                + " count=" + String(count)
                + " stride=" + String(stride)
                + " dim0=" + String(Int(self.dims[0])) + ")"
            )

        # Hyperslab: start=[start, 0, 0, ...], stride=[stride, 1, 1, ...],
        #            count=[count, 1, 1, ...], block=[1, dim1, dim2, ...]
        var start_arr = alloc[hsize_t](ndims)
        var stride_arr = alloc[hsize_t](ndims)
        var count_arr = alloc[hsize_t](ndims)
        var block_arr = alloc[hsize_t](ndims)
        start_arr[0] = hsize_t(start)
        stride_arr[0] = hsize_t(stride)
        count_arr[0] = hsize_t(count)
        block_arr[0] = 1
        # Memory-space shape: (count, dim1, dim2, ...) — the contiguous
        # destination layout after deinterleaving.
        var mem_dims = alloc[hsize_t](ndims)
        mem_dims[0] = hsize_t(count)
        for i in range(1, ndims):
            start_arr[i] = 0
            stride_arr[i] = 1
            count_arr[i] = 1
            block_arr[i] = self.dims[i]
            mem_dims[i] = self.dims[i]

        var file_space = h5d_get_space(self.dset_id)
        if file_space < 0:
            start_arr.free()
            stride_arr.free()
            count_arr.free()
            block_arr.free()
            mem_dims.free()
            raise Error("H5Dget_space failed")

        var sel_ret = h5s_select_hyperslab(
            file_space,
            H5S_SELECT_SET,
            start_arr,
            stride_arr,
            count_arr,
            block_arr,
        )
        if sel_ret < 0:
            _ = h5s_close(file_space)
            start_arr.free()
            stride_arr.free()
            count_arr.free()
            block_arr.free()
            mem_dims.free()
            raise Error("H5Sselect_hyperslab (strided) failed")

        var mem_space = h5s_create_simple(
            c_int(ndims),
            mem_dims,
            UnsafePointer[hsize_t, MutAnyOrigin](unsafe_from_address=0),
        )
        if mem_space < 0:
            _ = h5s_close(file_space)
            start_arr.free()
            stride_arr.free()
            count_arr.free()
            block_arr.free()
            mem_dims.free()
            raise Error("H5Screate_simple failed")

        var read_ret = h5d_read(
            self.dset_id,
            self.native_type_id,
            mem_space,
            file_space,
            H5P_DEFAULT,
            buf.bitcast[NoneType](),
        )
        _ = h5s_close(mem_space)
        _ = h5s_close(file_space)
        start_arr.free()
        stride_arr.free()
        count_arr.free()
        block_arr.free()
        mem_dims.free()

        if read_ret < 0:
            raise Error("H5Dread (strided) failed: ret=" + String(Int(read_ret)))
