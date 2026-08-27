"""End-to-end test for the HDF5 WRITE path.

Writes a file from Mojo, then verifies it three ways:

  1. shape/dtype/extent via our own reader (`H5File`/`H5Dataset`)
  2. VALUES via our own reader — full read and hyperslab read
  3. the same file re-read by **h5py**, so a self-consistent-but-wrong
     encoding cannot pass. A round-trip through one implementation proves
     nothing about the format; this is the discriminating half.

Multi-append is exercised deliberately (3 uneven batches, one of them
crossing the chunk boundary) because appending is the whole point of the
unlimited axis, and an off-by-one in the extent/hyperslab arithmetic would
otherwise only show up on a long collection run.

Run:
    pixi run mojo run -I . tests/io/hdf5/test_writer_roundtrip.mojo
"""

from std.memory import alloc
from std.python import Python
from std.testing import assert_equal, assert_true, assert_almost_equal

from mojo_rl.io.hdf5 import (
    H5File,
    H5Writer,
    H5T_FLOAT,
    H5T_INTEGER,
    H5T_SGN_2,
    H5T_SGN_NONE,
    native_type,
    h5t_get_size,
    h5t_get_class,
)


comptime OUT_PATH = "/tmp/mojo_rl_hdf5_writer_roundtrip.h5"

comptime ROW_DIM: Int = 9          # e.g. a walker qpos row
comptime CHUNK_ROWS: Int = 4       # deliberately tiny: forces chunk crossing
comptime B0: Int = 3               # append batch sizes, uneven on purpose
comptime B1: Int = 5
comptime B2: Int = 2
comptime N_TOTAL: Int = B0 + B1 + B2   # 10 rows


def expected_qpos(row: Int, col: Int) -> Float64:
    """Deterministic generator — distinct per (row, col), no symmetry that
    could hide a transposed write."""
    return Float64(row) * 100.0 + Float64(col) * 3.0 + 0.5


def expected_reward(row: Int) -> Float64:
    return Float64(row) * -0.25 + 1.0


def test_native_types_resolve() raises:
    print("[test] H5T_NATIVE_* globals resolve after H5open()...")
    var f32 = native_type[DType.float32]()
    var f64 = native_type[DType.float64]()
    var i32 = native_type[DType.int32]()
    var u8 = native_type[DType.uint8]()

    # A raw id proves nothing — check the SEMANTICS libhdf5 reports back.
    assert_equal(Int(h5t_get_size(f32)), 4, "native float32 size")
    assert_equal(Int(h5t_get_class(f32)), Int(H5T_FLOAT), "float32 class")
    assert_equal(Int(h5t_get_size(f64)), 8, "native float64 size")
    assert_equal(Int(h5t_get_class(f64)), Int(H5T_FLOAT), "float64 class")
    assert_equal(Int(h5t_get_size(i32)), 4, "native int32 size")
    assert_equal(Int(h5t_get_class(i32)), Int(H5T_INTEGER), "int32 class")
    assert_equal(Int(h5t_get_size(u8)), 1, "native uint8 size")
    print("       sizes/classes OK")


def write_fixture() raises:
    print("[test] write", N_TOTAL, "rows in 3 appends (chunk =",
          CHUNK_ROWS, "rows)...")
    var w = H5Writer(String(OUT_PATH))

    var qpos = w.create[DType.float32](
        String("qpos"), row_dim=ROW_DIM, chunk_rows=CHUNK_ROWS
    )
    var reward = w.create[DType.float32](
        String("reward"), row_dim=1, chunk_rows=CHUNK_ROWS
    )
    var ep_len = w.create[DType.int32](
        String("ep_len"), row_dim=1, chunk_rows=CHUNK_ROWS
    )

    var buf = alloc[Scalar[DType.float32]](N_TOTAL * ROW_DIM).as_unsafe_any_origin()
    var rbuf = alloc[Scalar[DType.float32]](N_TOTAL).as_unsafe_any_origin()

    var written = 0
    var batches = [B0, B1, B2]
    for bi in range(len(batches)):
        var n = batches[bi]
        for i in range(n):
            var row = written + i
            for c in range(ROW_DIM):
                buf[i * ROW_DIM + c] = Scalar[DType.float32](
                    expected_qpos(row, c)
                )
            rbuf[i] = Scalar[DType.float32](expected_reward(row))
        qpos.append[DType.float32](buf, n)
        reward.append[DType.float32](rbuf, n)
        written += n
        assert_equal(qpos.n_rows, written, "qpos n_rows after append")

    # int32 column, single append.
    var ibuf = alloc[Scalar[DType.int32]](3).as_unsafe_any_origin()
    ibuf[0] = 4
    ibuf[1] = 3
    ibuf[2] = 3
    ep_len.append[DType.int32](ibuf, 3)

    w.flush()
    buf.free()
    rbuf.free()
    ibuf.free()
    print("       wrote", written, "rows OK")


def test_readback_mojo() raises:
    print("[test] read back through our own reader...")
    var f = H5File(String(OUT_PATH))

    var ds = f.open_dataset(String("qpos"))
    assert_equal(ds.ndim(), 2, "qpos rank")
    assert_equal(Int(ds.dims[0]), N_TOTAL, "qpos rows")
    assert_equal(Int(ds.dims[1]), ROW_DIM, "qpos cols")
    assert_equal(ds.dtype_class, H5T_FLOAT, "qpos class")
    assert_equal(ds.elem_size, 4, "qpos elem_size")

    var buf = alloc[Scalar[DType.float32]](N_TOTAL * ROW_DIM).as_unsafe_any_origin()
    ds.read_all[DType.float32](buf)
    for r in range(N_TOTAL):
        for c in range(ROW_DIM):
            assert_almost_equal(
                Float64(buf[r * ROW_DIM + c]),
                expected_qpos(r, c),
                atol=1e-4,
                msg="qpos[" + String(r) + "," + String(c) + "]",
            )
    buf.free()

    # Hyperslab read of a range that straddles two appends AND two chunks.
    var sub = alloc[Scalar[DType.float32]](4 * ROW_DIM).as_unsafe_any_origin()
    ds.read_range[DType.float32](2, 6, sub)
    for k in range(4):
        for c in range(ROW_DIM):
            assert_almost_equal(
                Float64(sub[k * ROW_DIM + c]),
                expected_qpos(2 + k, c),
                atol=1e-4,
                msg="qpos range row " + String(2 + k),
            )
    sub.free()

    var rds = f.open_dataset(String("reward"))
    assert_equal(rds.ndim(), 1, "reward rank (row_dim=1 must be rank 1)")
    assert_equal(Int(rds.dims[0]), N_TOTAL, "reward rows")
    var rb = alloc[Scalar[DType.float32]](N_TOTAL).as_unsafe_any_origin()
    rds.read_all[DType.float32](rb)
    for r in range(N_TOTAL):
        assert_almost_equal(
            Float64(rb[r]), expected_reward(r), atol=1e-6,
            msg="reward[" + String(r) + "]",
        )
    rb.free()

    var eds = f.open_dataset(String("ep_len"))
    assert_equal(eds.dtype_class, H5T_INTEGER, "ep_len class")
    assert_equal(eds.signedness, H5T_SGN_2, "ep_len signed")
    assert_equal(eds.elem_size, 4, "ep_len elem_size")
    print("       values + shapes OK")


def test_readback_h5py() raises:
    """The half that actually gates the FORMAT: an independent reader."""
    print("[test] cross-check with h5py...")
    var h5py = Python.import_module("h5py")
    var np = Python.import_module("numpy")
    var f = h5py.File(OUT_PATH, "r")

    var keys = List[String]()
    for k in f.keys():
        keys.append(String(k))
    assert_equal(len(keys), 3, "dataset count seen by h5py")

    var q = f["qpos"]
    assert_equal(Int(py=q.shape[0]), N_TOTAL, "h5py qpos rows")
    assert_equal(Int(py=q.shape[1]), ROW_DIM, "h5py qpos cols")
    assert_true(
        String(q.dtype) == "float32", "h5py qpos dtype, got " + String(q.dtype)
    )
    # Chunking must have survived — without it the unlimited axis is a lie.
    assert_true(q.chunks is not None, "h5py: qpos must be chunked")
    assert_equal(Int(py=q.chunks[0]), CHUNK_ROWS, "h5py chunk rows")
    assert_true(
        q.maxshape[0] is None, "h5py: qpos dim-0 must be unlimited"
    )

    for r in range(N_TOTAL):
        for c in range(ROW_DIM):
            assert_almost_equal(
                Float64(py=q[r][c]), expected_qpos(r, c), atol=1e-4,
                msg="h5py qpos[" + String(r) + "," + String(c) + "]",
            )

    var rw = f["reward"]
    assert_equal(Int(py=rw.ndim), 1, "h5py reward rank")
    assert_equal(Int(py=rw.shape[0]), N_TOTAL, "h5py reward rows")

    var el = f["ep_len"]
    assert_true(
        String(el.dtype) == "int32", "h5py ep_len dtype, got " + String(el.dtype)
    )
    assert_equal(Int(py=el[0]), 4, "h5py ep_len[0]")
    assert_equal(Int(py=el[1]), 3, "h5py ep_len[1]")

    f.close()
    print("       h5py agrees on shape, dtype, chunking, maxshape, values OK")


def test_deflate_roundtrip() raises:
    """Compression path: shuffle+deflate must not change the values."""
    print("[test] shuffle+deflate round-trip...")
    var path = String("/tmp/mojo_rl_hdf5_writer_deflate.h5")
    var w = H5Writer(path)
    var ds = w.create[DType.float32](
        String("pixels"), row_dim=ROW_DIM, chunk_rows=CHUNK_ROWS, deflate=4
    )
    var buf = alloc[Scalar[DType.float32]](N_TOTAL * ROW_DIM).as_unsafe_any_origin()
    for r in range(N_TOTAL):
        for c in range(ROW_DIM):
            buf[r * ROW_DIM + c] = Scalar[DType.float32](expected_qpos(r, c))
    ds.append[DType.float32](buf, N_TOTAL)
    w.flush()
    buf.free()

    var f = H5File(path)
    var rds = f.open_dataset(String("pixels"))
    assert_equal(Int(rds.dims[0]), N_TOTAL, "deflate rows")
    var rb = alloc[Scalar[DType.float32]](N_TOTAL * ROW_DIM).as_unsafe_any_origin()
    rds.read_all[DType.float32](rb)
    for r in range(N_TOTAL):
        for c in range(ROW_DIM):
            assert_almost_equal(
                Float64(rb[r * ROW_DIM + c]), expected_qpos(r, c), atol=1e-4,
                msg="deflate qpos[" + String(r) + "," + String(c) + "]",
            )
    rb.free()
    print("       deflate values OK")


def main() raises:
    test_native_types_resolve()
    write_fixture()
    test_readback_mojo()
    test_readback_h5py()
    test_deflate_roundtrip()
    print("\n[PASS] hdf5 writer round-trip (mojo + h5py agree)")
