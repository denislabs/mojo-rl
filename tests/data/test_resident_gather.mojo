"""Gate for data-platform Stage 2 — residency + gather.

The stated gate is "bit-identical gather vs the host path on the same
indices", so the device comparison uses **exact equality**, not a tolerance.
A gather is a pure copy: there is no arithmetic to round, so any difference at
all is a bug, and a tolerance would hide exactly the class of error this is
meant to catch (a stride mistake that lands on a neighbouring element).

Covered:
  1. host gather values against the generator, including duplicate indices,
     first row, last row and reverse order
  2. bounds checking raises rather than reading arbitrary memory
  3. device gather == host gather, bit-exact, for float32 (state-shaped)
     and uint8 (pixel-shaped, wide rows)
  4. a scalar column (row_dim == 1), whose stride arithmetic is the easiest
     to get subtly wrong

Run:
    pixi run mojo run -I . tests/data/test_resident_gather.mojo
"""

from std.gpu.host import DeviceContext
from std.testing import assert_equal, assert_true

from mojo_rl.data import (
    ColumnSpec,
    IDX_DT,
    IndexBatch,
    ResidentColumn,
    TrajectoryStore,
    TrajectoryStoreWriter,
)


comptime OUT = "/tmp/mojo_rl_resident_gather.h5"

comptime N_ROWS: Int = 40
comptime QPOS: Int = 9
comptime PIX: Int = 48          # wide-ish row, uint8 (pixel-shaped)
comptime EP: Int = 10           # 4 episodes of 10


def expected_qpos(row: Int, col: Int) -> Float32:
    return Float32(row) * 10.0 + Float32(col) * 0.25


def expected_pix(row: Int, col: Int) -> UInt8:
    return UInt8((row * 7 + col * 3) % 251)


def expected_rew(row: Int) -> Float32:
    return Float32(row) * -0.5 + 1.0


def build_store() raises:
    print("[setup] write", N_ROWS, "rows ...")
    var cols = List[ColumnSpec]()
    cols.append(ColumnSpec(String("qpos"), DType.float32, QPOS))
    cols.append(ColumnSpec(String("pixels"), DType.uint8, PIX))
    cols.append(ColumnSpec(String("reward"), DType.float32, 1))

    var w = TrajectoryStoreWriter(
        String(OUT), cols^, env_id=String("gather-test"), chunk_rows=8
    )

    var bq = List[Scalar[DType.float32]](unsafe_uninit_length=EP * QPOS)
    var bp = List[Scalar[DType.uint8]](unsafe_uninit_length=EP * PIX)
    var br = List[Scalar[DType.float32]](unsafe_uninit_length=EP)

    var row = 0
    for _ep in range(N_ROWS // EP):
        for i in range(EP):
            for c in range(QPOS):
                bq[i * QPOS + c] = expected_qpos(row + i, c)
            for c in range(PIX):
                bp[i * PIX + c] = expected_pix(row + i, c)
            br[i] = expected_rew(row + i)
        w.append[DType.float32](
            String("qpos"), bq.unsafe_ptr().as_unsafe_any_origin(), EP
        )
        w.append[DType.uint8](
            String("pixels"), bp.unsafe_ptr().as_unsafe_any_origin(), EP
        )
        w.append[DType.float32](
            String("reward"), br.unsafe_ptr().as_unsafe_any_origin(), EP
        )
        w.end_episode()
        row += EP
    w.close()
    print("       OK")


def make_indices() -> IndexBatch:
    """Adversarial on purpose: duplicates (a real sampler draws with
    replacement), first and last row, and descending order so a gather that
    accidentally streams sequentially still fails."""
    var h = List[Scalar[IDX_DT]]()
    h.append(Scalar[IDX_DT](0))
    h.append(Scalar[IDX_DT](N_ROWS - 1))
    h.append(Scalar[IDX_DT](17))
    h.append(Scalar[IDX_DT](17))          # duplicate
    h.append(Scalar[IDX_DT](5))
    h.append(Scalar[IDX_DT](39))
    h.append(Scalar[IDX_DT](2))
    h.append(Scalar[IDX_DT](0))           # duplicate of the first
    h.append(Scalar[IDX_DT](31))
    h.append(Scalar[IDX_DT](10))
    return IndexBatch(h^)


def test_host_gather() raises:
    print("[1] host gather values ...")
    var s = TrajectoryStore(String(OUT))
    var col = ResidentColumn[DType.float32].load(s, String("qpos"))
    assert_equal(col.n_rows, N_ROWS, "resident n_rows")
    assert_equal(col.row_dim, QPOS, "resident row_dim")

    var idx = make_indices()
    var out = List[Scalar[DType.float32]]()
    col.gather_host(idx, out)
    assert_equal(len(out), idx.size() * QPOS, "gather output size")

    for i in range(idx.size()):
        var r = Int(idx.host[i])
        for c in range(QPOS):
            assert_equal(
                out[i * QPOS + c], expected_qpos(r, c),
                "qpos gather lane " + String(i) + " (row " + String(r)
                + ") col " + String(c),
            )
    print("    OK")


def test_scalar_column() raises:
    """Scalar column (row_dim == 1) — the stride arithmetic most easily broken."""
    print("[2] scalar column (row_dim=1) ...")
    var s = TrajectoryStore(String(OUT))
    var col = ResidentColumn[DType.float32].load(s, String("reward"))
    assert_equal(col.row_dim, 1, "reward row_dim")

    var idx = make_indices()
    var out = List[Scalar[DType.float32]]()
    col.gather_host(idx, out)
    assert_equal(len(out), idx.size(), "scalar gather size")
    for i in range(idx.size()):
        assert_equal(
            out[i], expected_rew(Int(idx.host[i])),
            "reward gather lane " + String(i),
        )
    print("    OK")


def test_bounds_check() raises:
    print("[3] out-of-range index must raise ...")
    var s = TrajectoryStore(String(OUT))
    var col = ResidentColumn[DType.float32].load(s, String("qpos"))

    var bad = List[Scalar[IDX_DT]]()
    bad.append(Scalar[IDX_DT](0))
    bad.append(Scalar[IDX_DT](N_ROWS))       # one past the end
    var ib = IndexBatch(bad^)
    var out = List[Scalar[DType.float32]]()
    var raised = False
    try:
        col.gather_host(ib, out)
    except:
        raised = True
    assert_true(raised, "index == n_rows must raise")

    var neg = List[Scalar[IDX_DT]]()
    neg.append(Scalar[IDX_DT](-1))
    var ib2 = IndexBatch(neg^)
    raised = False
    try:
        col.gather_host(ib2, out)
    except:
        raised = True
    assert_true(raised, "negative index must raise")
    print("    OK")


def test_device_matches_host_f32() raises:
    print("[4] device gather == host gather, BIT-EXACT (float32) ...")
    var ctx = DeviceContext()
    var s = TrajectoryStore(String(OUT))
    var col = ResidentColumn[DType.float32].load(s, String("qpos"))

    var idx = make_indices()
    var host_out = List[Scalar[DType.float32]]()
    col.gather_host(idx, host_out)

    var dev_out = List[Scalar[DType.float32]]()
    col.gather_device_to_host(ctx, idx, dev_out)

    assert_equal(len(dev_out), len(host_out), "device/host output size")
    for i in range(len(host_out)):
        assert_equal(
            dev_out[i], host_out[i],
            "float32 lane element " + String(i) + " differs device vs host",
        )
    print("    ", len(host_out), "elements identical  OK")


def test_device_matches_host_u8() raises:
    """Wide uint8 rows — the pixel shape, and a different element width so a
    stride bug that cancels at 4 bytes shows up."""
    print("[5] device gather == host gather, BIT-EXACT (uint8, wide) ...")
    var ctx = DeviceContext()
    var s = TrajectoryStore(String(OUT))
    var col = ResidentColumn[DType.uint8].load(s, String("pixels"))
    assert_equal(col.row_dim, PIX, "pixels row_dim")

    var idx = make_indices()
    var host_out = List[Scalar[DType.uint8]]()
    col.gather_host(idx, host_out)

    var dev_out = List[Scalar[DType.uint8]]()
    col.gather_device_to_host(ctx, idx, dev_out)

    assert_equal(len(dev_out), len(host_out), "device/host output size")
    for i in range(len(host_out)):
        assert_equal(
            dev_out[i], host_out[i],
            "uint8 lane element " + String(i) + " differs device vs host",
        )
    # And against the generator, so both paths agreeing on garbage fails too.
    for i in range(idx.size()):
        var r = Int(idx.host[i])
        for c in range(PIX):
            assert_equal(
                dev_out[i * PIX + c], expected_pix(r, c),
                "pixels lane " + String(i) + " col " + String(c),
            )
    print("    ", len(host_out), "elements identical  OK")


def test_device_scalar_column() raises:
    print("[6] device gather, scalar column ...")
    var ctx = DeviceContext()
    var s = TrajectoryStore(String(OUT))
    var col = ResidentColumn[DType.float32].load(s, String("reward"))

    var idx = make_indices()
    var host_out = List[Scalar[DType.float32]]()
    col.gather_host(idx, host_out)
    var dev_out = List[Scalar[DType.float32]]()
    col.gather_device_to_host(ctx, idx, dev_out)

    for i in range(len(host_out)):
        assert_equal(
            dev_out[i], host_out[i],
            "scalar lane " + String(i) + " differs device vs host",
        )
    print("    OK")


def test_device_large_batch() raises:
    """A batch big enough to span many thread blocks.

    The small cases above fit in one or two blocks, so a mistake in the
    grid/block arithmetic (or a missing `t >= batch*row_dim` guard) would pass
    them and fail only at real batch sizes. 1024 lanes x 9 = 9216 elements is
    ~36 blocks at TPB=256.
    """
    print("[7] device gather, multi-block batch (1024 lanes) ...")
    var ctx = DeviceContext()
    var s = TrajectoryStore(String(OUT))
    var col = ResidentColumn[DType.float32].load(s, String("qpos"))

    var h = List[Scalar[IDX_DT]]()
    var seed = 20260805
    for _ in range(1024):
        seed = (seed * 1103515245 + 12345) % 2147483647
        h.append(Scalar[IDX_DT](seed % N_ROWS))
    var idx = IndexBatch(h^)

    var host_out = List[Scalar[DType.float32]]()
    col.gather_host(idx, host_out)
    var dev_out = List[Scalar[DType.float32]]()
    col.gather_device_to_host(ctx, idx, dev_out)

    assert_equal(len(dev_out), 1024 * QPOS, "large batch size")
    for i in range(len(host_out)):
        assert_equal(
            dev_out[i], host_out[i],
            "large-batch element " + String(i) + " differs device vs host",
        )
    # Against the generator too, so both paths agreeing on garbage still fails.
    for i in range(idx.size()):
        var r = Int(idx.host[i])
        for c in range(QPOS):
            assert_equal(
                dev_out[i * QPOS + c], expected_qpos(r, c),
                "large-batch lane " + String(i) + " col " + String(c),
            )
    print("    ", len(host_out), "elements identical  OK")


def main() raises:
    build_store()
    test_host_gather()
    test_scalar_column()
    test_bounds_check()
    test_device_matches_host_f32()
    test_device_matches_host_u8()
    test_device_scalar_column()
    test_device_large_batch()
    print("\n[PASS] resident gather — Stage 2")
