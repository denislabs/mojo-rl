"""Smoke test for the R2 dataset store (data-platform stage 6).

Round-trips a real `TrajectoryStore` through the monitor's `/datasets`
catalog and R2: write locally -> push -> pull to a fresh path -> reopen and
verify the contents survived.

Credentials come from `.env` exactly like the remote logger
(`examples/half_cheetah/sac_half_cheetah_training_gpu.mojo`): `RL_MONITOR_URL`
+ `RL_MONITOR_API_KEY`.

Run:
    pixi run mojo run -I . examples/data/dataset_remote_smoke.mojo
"""

from std.python import Python
from std.pathlib import Path

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.data import ColumnSpec, TrajectoryStore, TrajectoryStoreWriter
from mojo_rl.data.remote import RemoteCatalog


comptime LOCAL = "/tmp/mojo_rl_remote_smoke.h5"
comptime PULLED = "/tmp/mojo_rl_remote_smoke_pulled.h5"
comptime NAME = "smoke"
comptime VERSION = "v1"

comptime N_ROWS: Int = 24
comptime DIM: Int = 4


def expected(row: Int, col: Int) -> Float32:
    return Float32(row) * 3.0 + Float32(col) * 0.25


def build_local() raises:
    print("[1] writing a local store ...")
    var cols = List[ColumnSpec]()
    cols.append(ColumnSpec(String("state"), DType.float32, DIM))
    var w = TrajectoryStoreWriter(
        String(LOCAL), cols^, env_id=String("smoke-env"), seed=99,
        source_commit=String("smoke"), chunk_rows=8,
    )
    var buf = List[Scalar[DType.float32]](unsafe_uninit_length=N_ROWS * DIM)
    for r in range(N_ROWS):
        for c in range(DIM):
            buf[r * DIM + c] = expected(r, c)
    w.append[DType.float32](
        String("state"), buf.unsafe_ptr().as_unsafe_any_origin(), N_ROWS
    )
    w.end_episode()
    w.close()
    print("      wrote", N_ROWS, "rows ->", LOCAL)


def main() raises:
    # ── credentials, and a readable failure if they are absent ────────
    var env = load_dotenv()
    var url = env.get("RL_MONITOR_URL", "")
    var key = env.get("RL_MONITOR_API_KEY", "")
    print("[0] config")
    print("      RL_MONITOR_URL  =", url)
    print("      RL_MONITOR_API_KEY len =", key.byte_length())
    if url.byte_length() == 0 or key.byte_length() == 0:
        print("      MISSING — set both in .env; aborting")
        return
    # The catalog lives at <base>/datasets, NOT <base>/api/datasets: /api/* is
    # behind the dashboard's browser-session middleware.
    print("      catalog base    =", url.removesuffix("/") + "/datasets")

    var cat = RemoteCatalog(url, key)

    # ── list first: the cheapest call, and it isolates AUTH from R2 ───
    # If this 401s, the fault is auth/routing and nothing to do with the
    # bucket, presigning, or the S3 token.
    print("\n[2] GET /datasets (auth + routing check) ...")
    try:
        var rows = cat.list_datasets()
        print("      OK —", len(rows), "dataset(s) registered")
    except e:
        print("      FAILED:", String(e))
        print("      Read the error body above:")
        print("        'Unauthorized'    -> still hitting the /api/* session")
        print("                             middleware; routes must be at")
        print("                             /datasets (redeploy needed?)")
        print("        'Missing API key' -> no Authorization header arrived")
        print("        'Invalid API key' -> key did not verify (wrong/revoked")
        print("                             key, or wrong environment)")
        print("        404 / HTML        -> the static-asset handler answered;")
        print("                             /datasets needs run_worker_first")
        return

    build_local()

    print("\n[3] push ...")
    var id = cat.push(
        String(LOCAL), String(NAME), String(VERSION),
        env_id=String("smoke-env"), n_rows=N_ROWS, n_episodes=1, seed=99,
        source_commit=String("smoke"),
    )
    print("      pushed as", id)

    print("\n[4] describe ...")
    var meta = cat.describe(id)
    print("      status =", meta["status"], " sizeBytes =", meta["sizeBytes"])
    print("      sha256 =", meta["sha256"])

    print("\n[5] pull to a fresh path ...")
    var os = Python.import_module("os")
    try:
        _ = os.remove(PULLED)
    except:
        pass
    _ = cat.pull(id, String(PULLED))

    print("\n[6] reopen the pulled file and verify contents ...")
    var s = TrajectoryStore(String(PULLED))
    print("      rows =", s.n_rows(), " episodes =", s.n_episodes(),
          " env_id =", s.manifest.env_id, " seed =", s.manifest.seed)
    var col = s.load_column[DType.float32](String("state"))
    var bad = 0
    for r in range(N_ROWS):
        for c in range(DIM):
            if col[r * DIM + c] != expected(r, c):
                bad += 1
    if bad == 0:
        print("      all", N_ROWS * DIM, "elements survived the round trip")
    else:
        print("      MISMATCH in", bad, "elements")
        return

    print("\n[7] pull again — must skip the transfer (hash already matches)")
    _ = cat.pull(id, String(PULLED))

    print("\n[PASS] R2 dataset round trip")
