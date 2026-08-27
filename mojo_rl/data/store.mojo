# +--------------------------------------------------------------------------+ #
# | TrajectoryStore — the shared on-disk trajectory format
# +--------------------------------------------------------------------------+ #
"""One store format for offline datasets, replay dumps, demonstrations and
BFM trajectory data. See `docs/DATA_PLATFORM_PLAN.md`.

Layout inside a single `.h5`:

    __manifest__  uint8   (M,)          UTF-8 key=value block (see manifest.mojo)
    ep_len        int64   (N_ep,)
    ep_offset     int64   (N_ep,)
    <column>      any     (N_rows, ...) one dataset per declared column

Writing streams: every dataset is chunked with an unlimited row axis, so a
collection run appends as it goes rather than buffering in RAM. `ep_len` /
`ep_offset` / `__manifest__` are written at `close()`.

Reading is RESIDENT by design. Measured cost of one 4096-row minibatch:

    scattered rows, one read per row, from disk    272 ms
    scattered rows in 64-row groups, from disk    2.16 ms
    contiguous slab from disk                    0.072 ms
    scattered rows from a resident host buffer   0.060 ms

Gathering a minibatch row-by-row from HDF5 is ~4500x worse than from RAM, so
`load_column` (whole-column residency) is the sampling path and `read_range`
(contiguous slab) is the streaming path. There is deliberately no
gather-by-index-from-disk API — it would be a performance trap wearing a
convenient name.

Ingest of FOREIGN files (a HuggingFace dataset with no manifest of ours) works
by enumerating the datasets and introspecting each one's shape and dtype.
"""

from std.memory import alloc, dealloc

from mojo_rl.io.hdf5 import H5File, H5Dataset, H5Writer, H5DatasetWriter
from .column import ColumnSpec, dtype_bytes, dtype_from_h5, dtype_name
from .episode_index import EpisodeIndex
from .manifest import (
    MANIFEST_DATASET, Manifest, SCHEMA_VERSION, parse_manifest,
)


comptime EP_LEN_DATASET = "ep_len"
comptime EP_OFFSET_DATASET = "ep_offset"

comptime DEFAULT_CHUNK_ROWS = 4096

comptime DEFAULT_MAX_RESIDENT_BYTES = 8 * 1024 * 1024 * 1024
"""Guard on `load_column`. Whole-column residency is right for state data
(walker 10 M transitions = 992 MiB) and wrong for a 44 GB pixel column; this
turns the latter into an error naming `read_range` rather than a swap storm."""


# ══════════════════════════════════════════════════════════════════════════
# Writer
# ══════════════════════════════════════════════════════════════════════════

struct TrajectoryStoreWriter(Movable):
    """Streaming writer. Declare columns up front, append rows, mark episode
    boundaries, close.

        var w = TrajectoryStoreWriter(path, columns, env_id="…", seed=7)
        w.append[DType.float32]("qpos", ptr, n)
        …
        w.end_episode()
        w.close()
    """

    var _file: H5Writer
    var _specs: List[ColumnSpec]
    var _dsets: List[H5DatasetWriter]
    var _ep_len: List[Int64]
    var _ep_offset: List[Int64]
    var _rows_committed: Int
    """Rows covered by closed episodes."""
    var _closed: Bool

    var env_id: String
    var seed: Int
    var source_commit: String

    def __init__(
        out self,
        var path: String,
        var columns: List[ColumnSpec],
        var env_id: String = String(""),
        seed: Int = 0,
        var source_commit: String = String(""),
        chunk_rows: Int = DEFAULT_CHUNK_ROWS,
        deflate: Int = 0,
    ) raises:
        if len(columns) == 0:
            raise Error("TrajectoryStoreWriter: no columns declared")
        for i in range(len(columns)):
            var nm = columns[i].name
            if nm == MANIFEST_DATASET or nm == EP_LEN_DATASET or nm == EP_OFFSET_DATASET:
                raise Error(
                    "TrajectoryStoreWriter: '" + nm + "' is a reserved"
                    " dataset name"
                )
            for j in range(i):
                if columns[j].name == nm:
                    raise Error(
                        "TrajectoryStoreWriter: duplicate column '" + nm + "'"
                    )

        self._file = H5Writer(path^)
        self._specs = columns^
        self._dsets = List[H5DatasetWriter]()
        self._ep_len = List[Int64]()
        self._ep_offset = List[Int64]()
        self._rows_committed = 0
        self._closed = False
        self.env_id = env_id^
        self.seed = seed
        self.source_commit = source_commit^

        for i in range(len(self._specs)):
            self._dsets.append(
                self._create_for(self._specs[i], chunk_rows, deflate)
            )

    def __init__(out self, *, deinit move: Self):
        self._file = move._file^
        self._specs = move._specs^
        self._dsets = move._dsets^
        self._ep_len = move._ep_len^
        self._ep_offset = move._ep_offset^
        self._rows_committed = move._rows_committed
        self._closed = move._closed
        self.env_id = move.env_id^
        self.seed = move.seed
        self.source_commit = move.source_commit^

    def _create_for(
        self, spec: ColumnSpec, chunk_rows: Int, deflate: Int
    ) raises -> H5DatasetWriter:
        """Dispatch the comptime dtype the writer needs from the runtime spec.

        Flattens trailing dims into `row_dim`: a `[N,H,W,3]` column is written
        as `[N, H*W*3]`, and the manifest carries the real shape so a reader
        restores it. Keeps the Stage-0 writer at rank <= 2 without losing
        fidelity.
        """
        var rd = spec.row_dim()
        var nm = String(spec.name)
        if spec.dtype == DType.float32:
            return self._file.create[DType.float32](nm^, rd, chunk_rows, deflate)
        if spec.dtype == DType.float64:
            return self._file.create[DType.float64](nm^, rd, chunk_rows, deflate)
        if spec.dtype == DType.uint8:
            return self._file.create[DType.uint8](nm^, rd, chunk_rows, deflate)
        if spec.dtype == DType.int8:
            return self._file.create[DType.int8](nm^, rd, chunk_rows, deflate)
        if spec.dtype == DType.int16:
            return self._file.create[DType.int16](nm^, rd, chunk_rows, deflate)
        if spec.dtype == DType.uint16:
            return self._file.create[DType.uint16](nm^, rd, chunk_rows, deflate)
        if spec.dtype == DType.int32:
            return self._file.create[DType.int32](nm^, rd, chunk_rows, deflate)
        if spec.dtype == DType.uint32:
            return self._file.create[DType.uint32](nm^, rd, chunk_rows, deflate)
        if spec.dtype == DType.int64:
            return self._file.create[DType.int64](nm^, rd, chunk_rows, deflate)
        if spec.dtype == DType.uint64:
            return self._file.create[DType.uint64](nm^, rd, chunk_rows, deflate)
        raise Error("TrajectoryStoreWriter: unsupported dtype for " + spec.name)

    def _index_of(self, name: String) raises -> Int:
        for i in range(len(self._specs)):
            if self._specs[i].name == name:
                return i
        raise Error("TrajectoryStoreWriter: no such column: " + name)

    def append[
        dtype: DType
    ](
        mut self,
        name: String,
        buf: Pointer[Scalar[dtype], MutAnyOrigin],
        n_rows: Int,
    ) raises:
        """Append `n_rows` rows to one column.

        `dtype` is checked against the declared spec — a mismatch raises
        rather than reinterpreting the bytes, which is the failure mode that
        would otherwise corrupt a store silently.
        """
        var i = self._index_of(name)
        if self._specs[i].dtype != dtype:
            raise Error(
                "TrajectoryStoreWriter: column '" + name + "' is declared "
                + dtype_name(self._specs[i].dtype) + " but append was called"
                " with " + dtype_name(dtype)
            )
        self._dsets[i].append[dtype](buf, n_rows)

    def rows_in(self, name: String) raises -> Int:
        return self._dsets[self._index_of(name)].n_rows

    def _check_columns_aligned(self) raises -> Int:
        """Every column must hold the same number of rows. They share one row
        axis; letting them drift would make the episode index meaningless."""
        var n = self._dsets[0].n_rows
        for i in range(1, len(self._dsets)):
            if self._dsets[i].n_rows != n:
                raise Error(
                    "TrajectoryStoreWriter: column '" + self._specs[i].name
                    + "' has " + String(self._dsets[i].n_rows) + " rows but '"
                    + self._specs[0].name + "' has " + String(n)
                    + " — every column must be appended in step"
                )
        return n

    def end_episode(mut self) raises:
        """Close the current episode at the current row count."""
        var n = self._check_columns_aligned()
        var length = n - self._rows_committed
        if length <= 0:
            raise Error(
                "TrajectoryStoreWriter.end_episode: no rows appended since the"
                " previous boundary (zero-length episodes are not storable)"
            )
        self._ep_offset.append(Int64(self._rows_committed))
        self._ep_len.append(Int64(length))
        self._rows_committed = n

    def close(mut self) raises:
        """Write the episode index and the manifest, then flush."""
        if self._closed:
            return
        var n = self._check_columns_aligned()
        if n != self._rows_committed:
            raise Error(
                "TrajectoryStoreWriter.close: " + String(n - self._rows_committed)
                + " rows are not inside any episode — call end_episode()"
                " before closing"
            )
        if len(self._ep_len) == 0:
            raise Error("TrajectoryStoreWriter.close: no episodes recorded")

        # ── episode index ──────────────────────────────────────────────
        var n_ep = len(self._ep_len)
        var lb_a = alloc[Scalar[DType.int64]]({count = n_ep})
        var lb = lb_a.unsafe_ptr().as_unsafe_any_origin()
        var ob_a = alloc[Scalar[DType.int64]]({count = n_ep})
        var ob = ob_a.unsafe_ptr().as_unsafe_any_origin()
        try:
            for i in range(n_ep):
                lb[unsafe_offset=i] = self._ep_len[i]
                ob[unsafe_offset=i] = self._ep_offset[i]
            var dl = self._file.create[DType.int64](String(EP_LEN_DATASET), 1, 4096, 0)
            dl.append[DType.int64](lb, n_ep)
            var do = self._file.create[DType.int64](String(EP_OFFSET_DATASET), 1, 4096, 0)
            do.append[DType.int64](ob, n_ep)
        finally:
            dealloc(lb_a^)
            dealloc(ob_a^)

        # ── manifest ───────────────────────────────────────────────────
        var m = Manifest()
        m.schema_version = SCHEMA_VERSION
        m.env_id = String(self.env_id)
        m.n_rows = n
        m.n_episodes = n_ep
        m.seed = self.seed
        m.source_commit = String(self.source_commit)
        for i in range(len(self._specs)):
            m.columns.append(ColumnSpec(copy=self._specs[i]))
        var text = m.encode()

        var tbytes = text.as_bytes()
        var nbytes = len(tbytes)
        var mb_a = alloc[Scalar[DType.uint8]]({count = nbytes})
        var mb = mb_a.unsafe_ptr().as_unsafe_any_origin()
        try:
            for i in range(nbytes):
                mb[unsafe_offset=i] = Scalar[DType.uint8](tbytes[i])
            var dm = self._file.create[DType.uint8](
                String(MANIFEST_DATASET), 1, 4096, 0
            )
            dm.append[DType.uint8](mb, nbytes)
        finally:
            dealloc(mb_a^)

        self._file.flush()
        self._closed = True


# ══════════════════════════════════════════════════════════════════════════
# Reader
# ══════════════════════════════════════════════════════════════════════════

struct TrajectoryStore(Movable):
    """Read side. Opens a store we wrote (manifest present) or a foreign file
    (schema recovered by introspection)."""

    var _file: H5File
    var manifest: Manifest
    var episodes: EpisodeIndex
    var path: String

    def __init__(out self, var path: String) raises:
        self.path = String(path)
        self._file = H5File(path^)
        # Every field must be initialized before any method call on `self`;
        # both are overwritten immediately below.
        self.manifest = Manifest()
        self.episodes = EpisodeIndex()

        var names = self._file.dataset_names()
        var has_manifest = False
        for i in range(len(names)):
            if names[i] == MANIFEST_DATASET:
                has_manifest = True

        if has_manifest:
            self.manifest = self._read_manifest()
        else:
            self.manifest = self._infer_manifest(names)

        self.episodes = self._read_episode_index()
        # A store whose index disagrees with its columns yields samplers that
        # silently read the wrong rows, so this is checked on open, not lazily.
        self.episodes.validate(self.manifest.n_rows)

    def __init__(out self, *, deinit move: Self):
        self._file = move._file^
        self.manifest = move.manifest^
        self.episodes = move.episodes^
        self.path = move.path^

    def _read_manifest(self) raises -> Manifest:
        var ds = self._file.open_dataset(String(MANIFEST_DATASET))
        var n = ds.n_elements()
        var buf_a = alloc[Scalar[DType.uint8]]({count = n})
        var buf = buf_a.unsafe_ptr().as_unsafe_any_origin()
        try:
            ds.read_all[DType.uint8](buf)
            var text = String()
            for i in range(n):
                text += chr(Int(buf[unsafe_offset=i]))
            return parse_manifest(text)

        finally:
            dealloc(buf_a^)
    def _infer_manifest(self, names: List[String]) raises -> Manifest:
        """Recover a schema from a file we did not write.

        Every non-reserved dataset becomes a column, with its dtype and
        trailing shape read from the file. `n_rows` comes from dim-0, which
        every column must agree on.
        """
        var m = Manifest()
        m.schema_version = SCHEMA_VERSION
        m.env_id = String("(foreign)")
        m.n_rows = -1

        for i in range(len(names)):
            var nm = names[i]
            if nm == EP_LEN_DATASET or nm == EP_OFFSET_DATASET:
                continue
            var ds = self._file.open_dataset(String(nm))
            if ds.ndim() < 1:
                continue  # scalar dataset: metadata, not a column
            var dt = dtype_from_h5(ds.dtype_class, ds.elem_size, ds.signedness)
            var shape = List[Int]()
            for k in range(1, ds.ndim()):
                shape.append(Int(ds.dims[k]))
            var rows = Int(ds.dims[0])
            if m.n_rows < 0:
                m.n_rows = rows
            elif rows != m.n_rows:
                raise Error(
                    "data: foreign file column '" + nm + "' has " + String(rows)
                    + " rows but a previous column has " + String(m.n_rows)
                    + " — columns must share one row axis"
                )
            m.columns.append(ColumnSpec(String(nm), dt, shape^))

        if m.n_rows < 0:
            raise Error("data: file contains no usable columns")
        return m^

    def _read_episode_index(mut self) raises -> EpisodeIndex:
        var lens = self._read_int_column(String(EP_LEN_DATASET))
        var offs = self._read_int_column(String(EP_OFFSET_DATASET))
        var idx = EpisodeIndex(lens^, offs^)
        self.manifest.n_episodes = idx.n_episodes()
        return idx^

    def _read_int_column(self, name: String) raises -> List[Int64]:
        """Read a 1-D integer dataset as Int64, accepting int32 or int64 on
        disk — PushT writes `ep_len` as int32 and `ep_offset` as int64."""
        var ds = self._file.open_dataset(String(name))
        var n = ds.n_elements()
        var out = List[Int64](capacity=n)
        if ds.elem_size == 8:
            var b = (
                alloc[Scalar[DType.int64]]({count = n})
                .unsafe_leak()
                .as_unsafe_any_origin()
            )
            ds.read_all[DType.int64](b)
            for i in range(n):
                out.append(b[unsafe_offset=i])
            b.unsafe_free()
        elif ds.elem_size == 4:
            var b = (
                alloc[Scalar[DType.int32]]({count = n})
                .unsafe_leak()
                .as_unsafe_any_origin()
            )
            ds.read_all[DType.int32](b)
            for i in range(n):
                out.append(Int64(b[unsafe_offset=i]))
            b.unsafe_free()
        else:
            raise Error(
                "data: '" + name + "' has " + String(ds.elem_size)
                + "-byte elements; expected int32 or int64"
            )
        return out^

    # ── accessors ──────────────────────────────────────────────────────

    def n_rows(self) -> Int:
        return self.manifest.n_rows

    def n_episodes(self) -> Int:
        return self.episodes.n_episodes()

    def column(self, name: String) raises -> ColumnSpec:
        return self.manifest.column(name)

    def column_names(self) -> List[String]:
        var out = List[String]()
        for i in range(len(self.manifest.columns)):
            out.append(String(self.manifest.columns[i].name))
        return out^

    def _checked_spec[dtype: DType](self, name: String) raises -> ColumnSpec:
        var spec = self.manifest.column(name)
        if spec.dtype != dtype:
            raise Error(
                "data: column '" + name + "' is " + dtype_name(spec.dtype)
                + " but was read as " + dtype_name(dtype)
            )
        return spec^

    def read_range[
        dtype: DType
    ](
        self,
        name: String,
        start: Int,
        end: Int,
        buf: Pointer[Scalar[dtype], MutAnyOrigin],
    ) raises:
        """Contiguous slab `[start, end)` of one column into `buf`.

        The streaming path: contiguous reads run at GiB/s. `buf` must hold
        `(end - start) * row_dim` elements.
        """
        var spec = self._checked_spec[dtype](name)
        if start < 0 or end > self.manifest.n_rows or end < start:
            raise Error(
                "data.read_range: [" + String(start) + "," + String(end)
                + ") out of bounds for " + String(self.manifest.n_rows)
                + " rows"
            )
        var ds = self._file.open_dataset(String(name))
        ds.read_range[dtype](start, end, buf)

    def load_column[
        dtype: DType
    ](
        self,
        name: String,
        max_bytes: Int = DEFAULT_MAX_RESIDENT_BYTES,
    ) raises -> List[Scalar[dtype]]:
        """Load a whole column into host memory — the SAMPLING path.

        Residency is not an optimisation here: gathering scattered rows
        straight from HDF5 measured ~4500x slower than from RAM. Every
        state-column dataset we generate fits (walker 10 M rows = 992 MiB).

        Raises past `max_bytes` rather than thrashing; use `read_range` to
        stream a column that does not fit.
        """
        var spec = self._checked_spec[dtype](name)
        var n_elems = self.manifest.n_rows * spec.row_dim()
        var nbytes = n_elems * dtype_bytes(dtype)
        if nbytes > max_bytes:
            raise Error(
                "data.load_column('" + name + "'): " + String(nbytes // (1 << 20))
                + " MiB exceeds the " + String(max_bytes // (1 << 20))
                + " MiB residency guard. Raise max_bytes if it really fits,"
                " or stream it with read_range."
            )
        var out = List[Scalar[dtype]](unsafe_uninit_length=n_elems)
        var ds = self._file.open_dataset(String(name))
        ds.read_all[dtype](out.unsafe_ptr().as_unsafe_any_origin())
        return out^
