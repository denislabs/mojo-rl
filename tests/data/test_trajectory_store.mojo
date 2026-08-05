"""Gate for data-platform Stage 1 — `mojo_rl/data/`.

Four layers:

  1. manifest encode → parse round-trip, including the shaped-column form
  2. write → read → verify a synthetic 3-episode store, values exact
  3. EpisodeIndex boundary logic (the guard that stops a sequence sampler
     from spanning two episodes)
  4. FOREIGN ingest — a file with no manifest of ours, schema recovered by
     introspection. Run against the h5py-written PushT fixture AND, when it
     is present, the real 44 GB `pusht_expert_train.h5`.

Prerequisite for layer 4a:
    pixi run python tests/io/hdf5/make_fixture.py

Run:
    pixi run mojo run -I . tests/data/test_trajectory_store.mojo
"""

from std.memory import alloc
from std.pathlib import Path
from std.testing import assert_almost_equal, assert_equal, assert_true

from mojo_rl.data import (
    ColumnSpec,
    EpisodeIndex,
    Manifest,
    TrajectoryStore,
    TrajectoryStoreWriter,
    parse_manifest,
)


comptime OUT = "/tmp/mojo_rl_store_roundtrip.h5"
comptime PUSHT_FIXTURE = "/tmp/mojo_rl_hdf5_fixture.h5"
comptime PUSHT_REAL = "pusht_expert_train.h5"

# Three uneven episodes; the middle one crosses the 4-row chunk boundary.
comptime E0: Int = 3
comptime E1: Int = 5
comptime E2: Int = 2
comptime N_TOTAL: Int = E0 + E1 + E2
comptime QPOS: Int = 9
comptime ACT: Int = 6
comptime CHUNK: Int = 4


def expected_qpos(row: Int, col: Int) -> Float64:
    return Float64(row) * 100.0 + Float64(col) * 3.0 + 0.5


def expected_act(row: Int, col: Int) -> Float64:
    return Float64(row) * -1.5 + Float64(col) * 0.25


def expected_reward(row: Int) -> Float64:
    return Float64(row) * 0.125 - 2.0


# ══════════════════════════════════════════════════════════════════════════
# 1. Manifest
# ══════════════════════════════════════════════════════════════════════════

def test_manifest_roundtrip() raises:
    print("[1] manifest encode → parse ...")
    var m = Manifest()
    m.env_id = String("dm_control/walker-walk")
    m.n_rows = 10_000_000
    m.n_episodes = 10_000
    m.seed = 12345
    m.source_commit = String("081b53c0")
    m.columns.append(ColumnSpec(String("qpos"), DType.float32, 9))
    m.columns.append(ColumnSpec(String("reward"), DType.float32, 1))
    var shape = List[Int]()
    shape.append(84)
    shape.append(84)
    shape.append(3)
    m.columns.append(ColumnSpec(String("pixels"), DType.uint8, shape^))

    var text = m.encode()
    var back = parse_manifest(text)

    assert_equal(back.schema_version, 1, "schema_version")
    assert_true(back.env_id == "dm_control/walker-walk", "env_id")
    assert_equal(back.n_rows, 10_000_000, "n_rows")
    assert_equal(back.seed, 12345, "seed")
    assert_true(back.source_commit == "081b53c0", "source_commit")
    assert_equal(len(back.columns), 3, "column count")

    var q = back.column(String("qpos"))
    assert_true(q.dtype == DType.float32, "qpos dtype")
    assert_equal(q.row_dim(), 9, "qpos row_dim")

    # Scalar column must survive as rank-1, not [N,1].
    var r = back.column(String("reward"))
    assert_equal(r.rank(), 1, "reward rank")
    assert_equal(r.row_dim(), 1, "reward row_dim")

    # Multi-dim shape must survive intact — flattening to row_dim on disk
    # must not lose the real shape.
    var p = back.column(String("pixels"))
    assert_true(p.dtype == DType.uint8, "pixels dtype")
    assert_equal(len(p.shape), 3, "pixels shape rank")
    assert_equal(p.shape[0], 84, "pixels H")
    assert_equal(p.shape[2], 3, "pixels C")
    assert_equal(p.row_dim(), 84 * 84 * 3, "pixels row_dim")

    # Unknown keys must be ignored, not fatal (forward compatibility).
    var extended = text + "future_key=whatever\n"
    var fwd = parse_manifest(extended)
    assert_equal(len(fwd.columns), 3, "unknown key must not break parsing")
    print("    OK")


# ══════════════════════════════════════════════════════════════════════════
# 2. Write → read → verify
# ══════════════════════════════════════════════════════════════════════════

def write_store() raises:
    print("[2] write 3 episodes (", E0, E1, E2, ") ...")
    var cols = List[ColumnSpec]()
    cols.append(ColumnSpec(String("qpos"), DType.float32, QPOS))
    cols.append(ColumnSpec(String("action"), DType.float32, ACT))
    cols.append(ColumnSpec(String("reward"), DType.float32, 1))

    var w = TrajectoryStoreWriter(
        String(OUT),
        cols^,
        env_id=String("dm_control/walker-walk"),
        seed=99,
        source_commit=String("stage1"),
        chunk_rows=CHUNK,
    )

    var bq = alloc[Scalar[DType.float32]](E1 * QPOS).as_unsafe_any_origin()
    var ba = alloc[Scalar[DType.float32]](E1 * ACT).as_unsafe_any_origin()
    var br = alloc[Scalar[DType.float32]](E1).as_unsafe_any_origin()

    var eps = [E0, E1, E2]
    var row = 0
    for ei in range(len(eps)):
        var n = eps[ei]
        for i in range(n):
            for c in range(QPOS):
                bq[i * QPOS + c] = Scalar[DType.float32](expected_qpos(row + i, c))
            for c in range(ACT):
                ba[i * ACT + c] = Scalar[DType.float32](expected_act(row + i, c))
            br[i] = Scalar[DType.float32](expected_reward(row + i))
        w.append[DType.float32](String("qpos"), bq, n)
        w.append[DType.float32](String("action"), ba, n)
        w.append[DType.float32](String("reward"), br, n)
        w.end_episode()
        row += n

    w.close()
    bq.free(); ba.free(); br.free()
    print("    wrote", row, "rows OK")


def test_readback() raises:
    print("[2b] read back ...")
    var s = TrajectoryStore(String(OUT))

    assert_equal(s.n_rows(), N_TOTAL, "n_rows")
    assert_equal(s.n_episodes(), 3, "n_episodes")
    assert_true(s.manifest.env_id == "dm_control/walker-walk", "env_id")
    assert_equal(s.manifest.seed, 99, "seed survived")
    assert_equal(len(s.manifest.columns), 3, "column count")

    var qpos = s.load_column[DType.float32](String("qpos"))
    assert_equal(len(qpos), N_TOTAL * QPOS, "qpos element count")
    for r in range(N_TOTAL):
        for c in range(QPOS):
            assert_almost_equal(
                Float64(qpos[r * QPOS + c]), expected_qpos(r, c), atol=1e-4,
                msg="qpos[" + String(r) + "," + String(c) + "]",
            )

    var rew = s.load_column[DType.float32](String("reward"))
    assert_equal(len(rew), N_TOTAL, "reward element count")
    for r in range(N_TOTAL):
        assert_almost_equal(
            Float64(rew[r]), expected_reward(r), atol=1e-6,
            msg="reward[" + String(r) + "]",
        )

    # Slab spanning episode 0→1 and a chunk boundary.
    var sub = alloc[Scalar[DType.float32]](4 * ACT).as_unsafe_any_origin()
    s.read_range[DType.float32](String("action"), 2, 6, sub)
    for k in range(4):
        for c in range(ACT):
            assert_almost_equal(
                Float64(sub[k * ACT + c]), expected_act(2 + k, c), atol=1e-4,
                msg="action range row " + String(2 + k),
            )
    sub.free()
    print("    OK")


def test_dtype_mismatch_raises() raises:
    """A mis-typed read must raise, not reinterpret bytes — this is the
    failure that would corrupt a store silently."""
    print("[2c] dtype mismatch must raise ...")
    var s = TrajectoryStore(String(OUT))
    var raised = False
    try:
        var _bad = s.load_column[DType.float64](String("qpos"))
    except:
        raised = True
    assert_true(raised, "reading a float32 column as float64 must raise")
    print("    OK")


# ══════════════════════════════════════════════════════════════════════════
# 3. Episode boundaries
# ══════════════════════════════════════════════════════════════════════════

def test_episode_index() raises:
    print("[3] episode boundary logic ...")
    var s = TrajectoryStore(String(OUT))
    ref ix = s.episodes

    assert_equal(ix.start_of(0), 0, "ep0 start")
    assert_equal(ix.length_of(0), E0, "ep0 len")
    assert_equal(ix.start_of(1), E0, "ep1 start")
    assert_equal(ix.end_of(2), N_TOTAL, "ep2 end")

    assert_equal(ix.episode_of(0), 0, "row 0 → ep0")
    assert_equal(ix.episode_of(E0 - 1), 0, "last row of ep0")
    assert_equal(ix.episode_of(E0), 1, "first row of ep1")
    assert_equal(ix.episode_of(N_TOTAL - 1), 2, "last row overall")

    # The guard a sequence-window sampler depends on.
    assert_true(ix.window_fits(0, E0), "window filling ep0 fits")
    assert_true(not ix.window_fits(E0 - 1, 2), "window spanning ep0→ep1 must NOT fit")
    assert_true(ix.window_fits(E0, E1), "window filling ep1 fits")
    assert_true(not ix.window_fits(N_TOTAL - 1, 2), "window past the end must NOT fit")

    var raised = False
    try:
        _ = ix.episode_of(N_TOTAL + 10)
    except:
        raised = True
    assert_true(raised, "out-of-range row must raise")
    print("    OK")


# ══════════════════════════════════════════════════════════════════════════
# 4. Foreign ingest
# ══════════════════════════════════════════════════════════════════════════

def test_foreign_fixture() raises:
    print("[4a] foreign ingest — h5py-written PushT fixture ...")
    if not Path(PUSHT_FIXTURE).exists():
        print("    SKIP (run: pixi run python tests/io/hdf5/make_fixture.py)")
        return

    var s = TrajectoryStore(String(PUSHT_FIXTURE))
    # make_fixture.py: EP_LENGTHS = [4, 3, 5], 12 frames.
    assert_equal(s.n_rows(), 12, "foreign n_rows")
    assert_equal(s.n_episodes(), 3, "foreign n_episodes")
    assert_equal(s.episodes.length_of(0), 4, "foreign ep0 len")
    assert_equal(s.episodes.length_of(2), 5, "foreign ep2 len")

    var names = s.column_names()
    assert_equal(len(names), 4, "foreign column count (pixels/action/proprio/state)")

    # Shape and dtype must be recovered without a manifest.
    var px = s.column(String("pixels"))
    assert_true(px.dtype == DType.uint8, "pixels dtype inferred")
    assert_equal(len(px.shape), 3, "pixels shape rank inferred")
    assert_equal(px.shape[0], 8, "pixels H")
    assert_equal(px.shape[1], 6, "pixels W")
    assert_equal(px.shape[2], 3, "pixels C")

    var st = s.column(String("state"))
    assert_true(st.dtype == DType.float32, "state dtype inferred")
    assert_equal(st.row_dim(), 5, "state row_dim inferred")

    # Values, against make_fixture.py's generator: state[t] = t * (1..5)
    var state = s.load_column[DType.float32](String("state"))
    for t in range(12):
        for c in range(5):
            assert_almost_equal(
                Float64(state[t * 5 + c]), Float64(t) * Float64(c + 1),
                atol=1e-4, msg="state[" + String(t) + "," + String(c) + "]",
            )
    print("    OK")


def test_foreign_real_pusht() raises:
    print("[4b] foreign ingest — REAL pusht_expert_train.h5 ...")
    var home = Path.home()
    var p = home / ".cache" / "mojo_rl" / "lewm_pusht" / PUSHT_REAL
    if not p.exists():
        print("    SKIP (not cached)")
        return

    var s = TrajectoryStore(String(p))
    print("        n_rows    =", s.n_rows())
    print("        n_episodes=", s.n_episodes())
    var names = s.column_names()
    for i in range(len(names)):
        var c = s.column(names[i])
        print("        column", names[i], ":", c.describe(), " row_dim=", c.row_dim())

    assert_true(s.n_rows() > 0, "real pusht n_rows")
    assert_true(s.n_episodes() > 0, "real pusht n_episodes")
    # The index/columns consistency check already ran in the constructor;
    # this asserts it covered a real multi-GB file, not just a fixture.
    assert_equal(
        s.episodes.total_rows(), s.n_rows(),
        "episode index must cover every row of the real file",
    )

    # A slab read from the middle of a 44 GB file.
    var act = s.column(String("action"))
    var rows = 256
    var mid = s.n_rows() // 2
    var buf = alloc[Scalar[DType.float32]](
        rows * act.row_dim()
    ).as_unsafe_any_origin()
    s.read_range[DType.float32](String("action"), mid, mid + rows, buf)
    var finite = 0
    for i in range(rows * act.row_dim()):
        var v = Float64(buf[i])
        if v == v:  # NaN check
            finite += 1
    assert_equal(finite, rows * act.row_dim(), "slab must be all finite")
    buf.free()

    # Episode lookup must work at scale.
    var ep = s.episodes.episode_of(mid)
    assert_true(
        s.episodes.start_of(ep) <= mid and mid < s.episodes.end_of(ep),
        "episode_of must bracket the row",
    )
    print("    OK")


def main() raises:
    test_manifest_roundtrip()
    write_store()
    test_readback()
    test_dtype_mismatch_raises()
    test_episode_index()
    test_foreign_fixture()
    test_foreign_real_pusht()
    print("\n[PASS] trajectory store — Stage 1")
