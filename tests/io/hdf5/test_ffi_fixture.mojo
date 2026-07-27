"""End-to-end test: open the synthetic fixture, read every dataset via
the Mojo FFI, verify values match the deterministic generator.

Prerequisite — first run:
    pixi run python tests/io/hdf5/make_fixture.py
    pixi run mojo run -I . tests/io/hdf5/test_ffi_fixture.mojo

This exercises:
    H5Fopen / H5Fclose
    H5Dopen2 / H5Dclose / H5Dread
    H5Dget_space / H5Sget_simple_extent_(ndims|dims) / H5Sclose
    H5Dget_type / H5Tget_native_type / H5Tget_class / H5Tget_size / H5Tget_sign
    H5Sselect_hyperslab / H5Screate_simple   (via read_range)
"""

from std.memory import alloc
from std.testing import assert_equal, assert_true
from mojo_rl.io.hdf5 import (
    H5File,
    H5Dataset,
    H5T_INTEGER,
    H5T_FLOAT,
    H5T_SGN_NONE,
    H5T_SGN_2,
)


comptime FIXTURE_PATH = "/tmp/mojo_rl_hdf5_fixture.h5"

# Must match tests/io/hdf5/make_fixture.py
comptime N_TOTAL: Int = 12
comptime H: Int = 8
comptime W: Int = 6
comptime ACTION_DIM: Int = 2
comptime PROPRIO_DIM: Int = 2
comptime STATE_DIM: Int = 5


def test_open_close() raises:
    print("[test] open + close...")
    var f = H5File(String(FIXTURE_PATH))
    assert_true(f.file_id > 0, "file_id should be positive after open")
    print("       file_id =", f.file_id, " OK")


def test_ep_len_int32() raises:
    print("[test] ep_len (int32, [3])...")
    var f = H5File(String(FIXTURE_PATH))
    var ds = f.open_dataset(String("ep_len"))

    assert_equal(ds.ndim(), 1, "ep_len rank")
    assert_equal(Int(ds.dims[0]), 3, "ep_len.shape[0]")
    assert_equal(ds.dtype_class, H5T_INTEGER, "ep_len class")
    assert_equal(ds.elem_size, 4, "ep_len elem_size")
    assert_equal(ds.signedness, H5T_SGN_2, "ep_len signedness")

    var buf = alloc[Scalar[DType.int32]](3).as_unsafe_any_origin()
    ds.read_all[DType.int32](buf)
    assert_equal(Int(buf[0]), 4, "ep_len[0]")
    assert_equal(Int(buf[1]), 3, "ep_len[1]")
    assert_equal(Int(buf[2]), 5, "ep_len[2]")
    buf.free()
    print("       OK")


def test_ep_offset_int64() raises:
    print("[test] ep_offset (int64, [3])...")
    var f = H5File(String(FIXTURE_PATH))
    var ds = f.open_dataset(String("ep_offset"))

    assert_equal(ds.dtype_class, H5T_INTEGER, "ep_offset class")
    assert_equal(ds.elem_size, 8, "ep_offset elem_size")
    assert_equal(ds.signedness, H5T_SGN_2, "ep_offset signedness")

    var buf = alloc[Scalar[DType.int64]](3).as_unsafe_any_origin()
    ds.read_all[DType.int64](buf)
    assert_equal(Int(buf[0]), 0, "ep_offset[0]")
    assert_equal(Int(buf[1]), 4, "ep_offset[1]")
    assert_equal(Int(buf[2]), 7, "ep_offset[2]")
    buf.free()
    print("       OK")


def test_action_float32() raises:
    print("[test] action (float32, [12, 2])...")
    var f = H5File(String(FIXTURE_PATH))
    var ds = f.open_dataset(String("action"))

    assert_equal(ds.ndim(), 2, "action rank")
    assert_equal(Int(ds.dims[0]), N_TOTAL, "action.shape[0]")
    assert_equal(Int(ds.dims[1]), ACTION_DIM, "action.shape[1]")
    assert_equal(ds.dtype_class, H5T_FLOAT, "action class")
    assert_equal(ds.elem_size, 4, "action elem_size")

    var n = ds.n_elements()
    var buf = alloc[Scalar[DType.float32]](n).as_unsafe_any_origin()
    ds.read_all[DType.float32](buf)
    # action[t] = [t, t + 0.5]
    for t in range(N_TOTAL):
        assert_equal(Float64(buf[t * ACTION_DIM + 0]), Float64(t), "act[t,0]")
        assert_equal(
            Float64(buf[t * ACTION_DIM + 1]), Float64(t) + 0.5, "act[t,1]"
        )
    buf.free()
    print("       OK (", n, " elements)")


def test_pixels_uint8() raises:
    print("[test] pixels (uint8, [12,8,6,3])...")
    var f = H5File(String(FIXTURE_PATH))
    var ds = f.open_dataset(String("pixels"))

    assert_equal(ds.ndim(), 4, "pixels rank")
    assert_equal(Int(ds.dims[0]), N_TOTAL, "pixels.shape[0]")
    assert_equal(Int(ds.dims[1]), H, "pixels.shape[1]")
    assert_equal(Int(ds.dims[2]), W, "pixels.shape[2]")
    assert_equal(Int(ds.dims[3]), 3, "pixels.shape[3]")
    assert_equal(ds.dtype_class, H5T_INTEGER, "pixels class")
    assert_equal(ds.elem_size, 1, "pixels elem_size")
    assert_equal(ds.signedness, H5T_SGN_NONE, "pixels unsigned")

    var n = ds.n_elements()
    var buf = alloc[Scalar[DType.uint8]](n).as_unsafe_any_origin()
    ds.read_all[DType.uint8](buf)
    # pixels[t,*,*,*] = (t*7) % 256
    var stride = H * W * 3
    for t in range(N_TOTAL):
        var expected = UInt8((t * 7) % 256)
        for i in range(stride):
            assert_equal(buf[t * stride + i], expected, "pixels[t, ...]")
    buf.free()
    print("       OK (", n, " bytes)")


def test_pixels_read_range() raises:
    """Hyperslab read of rows [1:4) — exercises H5Sselect_hyperslab."""
    print("[test] pixels read_range[1:4) ...")
    var f = H5File(String(FIXTURE_PATH))
    var ds = f.open_dataset(String("pixels"))

    var stride = H * W * 3
    var buf = alloc[Scalar[DType.uint8]](3 * stride).as_unsafe_any_origin()
    ds.read_range[DType.uint8](1, 4, buf)

    for k in range(3):
        var t = 1 + k
        var expected = UInt8((t * 7) % 256)
        for i in range(stride):
            assert_equal(buf[k * stride + i], expected, "pixels[t, ...]")
    buf.free()
    print("       OK")


def test_action_read_strided() raises:
    """Strided read: 3 rows starting at idx=1 with stride=2 → rows {1,3,5}."""
    print("[test] action read_strided(start=1, count=3, stride=2)...")
    var f = H5File(String(FIXTURE_PATH))
    var ds = f.open_dataset(String("action"))

    var buf = alloc[Scalar[DType.float32]](3 * ACTION_DIM).as_unsafe_any_origin()
    ds.read_strided[DType.float32](1, 3, 2, buf)
    # Expected: action[1] = [1, 1.5], action[3] = [3, 3.5], action[5] = [5, 5.5]
    var expected_t: List[Int] = [1, 3, 5]
    for k in range(3):
        var t = expected_t[k]
        assert_equal(
            Float64(buf[k * ACTION_DIM + 0]), Float64(t), "strided act[t,0]"
        )
        assert_equal(
            Float64(buf[k * ACTION_DIM + 1]),
            Float64(t) + 0.5,
            "strided act[t,1]",
        )
    buf.free()
    print("       OK")


def test_state_float32() raises:
    print("[test] state (float32, [12, 5])...")
    var f = H5File(String(FIXTURE_PATH))
    var ds = f.open_dataset(String("state"))

    assert_equal(Int(ds.dims[0]), N_TOTAL, "state.shape[0]")
    assert_equal(Int(ds.dims[1]), STATE_DIM, "state.shape[1]")

    var n = ds.n_elements()
    var buf = alloc[Scalar[DType.float32]](n).as_unsafe_any_origin()
    ds.read_all[DType.float32](buf)
    # state[t] = [t, 2t, 3t, 4t, 5t]
    for t in range(N_TOTAL):
        for j in range(STATE_DIM):
            assert_equal(
                Float64(buf[t * STATE_DIM + j]),
                Float64(t * (j + 1)),
                "state[t,j]",
            )
    buf.free()
    print("       OK")


def main() raises:
    test_open_close()
    test_ep_len_int32()
    test_ep_offset_int64()
    test_action_float32()
    test_pixels_uint8()
    test_pixels_read_range()
    test_action_read_strided()
    test_state_float32()
    print("[hdf5 FFI fixture test] all passing.")
