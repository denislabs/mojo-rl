# +--------------------------------------------------------------------------+ #
# | Import LAFAN1 for the G1 into a TrajectoryStore — no Python
# +--------------------------------------------------------------------------+ #
"""Download (if needed) `lafan_29dof.pkl` and convert its 40 clips to the
50 Hz store BFM-Zero's training and evaluation read — G1b of
`docs/BFM_ZERO_G1_REPRODUCTION.md`.

    pixi run mojo run -I . examples/g1/lafan_import.mojo
    pixi run mojo run -I . examples/g1/lafan_import.mojo --pkl /path/to/lafan_29dof.pkl --out lafan_g1_50hz.h5

Replaces `tools/g1/lafan_reference_dump.py` + `lafan_to_store.py`, which
ran the reference's own motion library (torch, joblib, scipy, h5py) and
are kept as the ORACLE `tests/robots/test_lafan_import_vs_oracle.mojo`
holds this path against. This needs nothing but the pickle: the joblib
reader (`io/pickle.mojo`), the transcribed motion pipeline
(`data/lafan.mojo`) and our G1 for the forward kinematics.

Output: `lafan_g1_50hz.h5` in the working directory (the path every G1
consumer defaults to), 441 k rows, one episode per clip, the columns of
`lafan_to_store.py` (qpos 36, qvel 35, state 64, privileged 463,
body_pos / body_quat / body_vel / body_ang_vel, motion_id), the clip
names as the store's task table, `default_dof_pos` and `env_dt` as side
vectors. ~8 s on an M1 for the conversion (1.3 M frames of FK through
the engine); the 200 MB download is what takes the time.

Options
-------
--pkl PATH         a local `lafan_29dof.pkl`; skips the download
--out PATH         output .h5 (default `lafan_g1_50hz.h5`)
--revision REV     Hub revision of `LeCAR-Lab/BFM-Zero` (default `main`)
--force            rebuild even if the output already exists
--no-download      fail rather than fetch anything over the network

The pickle is pinned by size and sha256 (`PKL_SHA256`): the Hub file that
is not this one is refused, because every number downstream — the
tracking numbers of G2, the RSI table — was gated on this dump.
"""

from std.os.path import exists
from std.sys import argv
from std.time import perf_counter_ns

from mojo_rl.data.column import ColumnSpec
from mojo_rl.data.lafan import (
    load_lafan_clips, convert_clip, LafanRows, LAFAN_NQ, LAFAN_NV,
    LAFAN_STATE_DIM, LAFAN_PRIV_DIM, LAFAN_N_BODIES, LAFAN_ENV_DT,
)
from mojo_rl.data.store import TrajectoryStoreWriter
from mojo_rl.envs.robots import UnitreeG1
from mojo_rl.envs.robots.unitree_g1_pd import G1_N_DOF, g1_default_pos
from mojo_rl.io.fetch import sha256_file
from mojo_rl.io.fileio import remove_file, rename_over
from mojo_rl.io.hf import hf_download_file, HF_MODEL


comptime REPO = "LeCAR-Lab/BFM-Zero"
comptime PKL_REL = "data/lafan_29dof.pkl"
comptime PKL_SHA256 = "f3a0c2810363f5c50bf4146fa2db33c1ff5b90d00cb7c0bc2aa4622696375e11"
comptime PKL_SIZE = 209659488
comptime ENV_ID = "unitree_g1_lafan1_50hz"


def _opt(args: List[String], name: String, fallback: String) raises -> String:
    for i in range(len(args) - 1):
        if args[i] == name:
            return String(args[i + 1])
    return fallback


def _flag(args: List[String], name: String) -> Bool:
    for i in range(len(args)):
        if args[i] == name:
            return True
    return False


@always_inline
def _fptr(mut lst: List[Float32]) -> Pointer[Scalar[DType.float32], MutAnyOrigin]:
    return lst.unsafe_ptr().unsafe_bitcast[Scalar[DType.float32]]().as_unsafe_any_origin()


@always_inline
def _iptr(mut lst: List[Int32]) -> Pointer[Scalar[DType.int32], MutAnyOrigin]:
    return lst.unsafe_ptr().unsafe_bitcast[Scalar[DType.int32]]().as_unsafe_any_origin()


@always_inline
def _dptr(mut lst: List[Float64]) -> Pointer[Scalar[DType.float64], MutAnyOrigin]:
    return lst.unsafe_ptr().unsafe_bitcast[Scalar[DType.float64]]().as_unsafe_any_origin()


def _columns() -> List[ColumnSpec]:
    var cols = List[ColumnSpec]()
    cols.append(ColumnSpec(String("qpos"), DType.float32, LAFAN_NQ))
    cols.append(ColumnSpec(String("qvel"), DType.float32, LAFAN_NV))
    cols.append(ColumnSpec(String("state"), DType.float32, LAFAN_STATE_DIM))
    cols.append(ColumnSpec(String("privileged"), DType.float32, LAFAN_PRIV_DIM))
    cols.append(ColumnSpec(String("body_pos"), DType.float32, LAFAN_N_BODIES * 3))
    cols.append(ColumnSpec(String("body_quat"), DType.float32, LAFAN_N_BODIES * 4))
    cols.append(ColumnSpec(String("body_vel"), DType.float32, LAFAN_N_BODIES * 3))
    cols.append(ColumnSpec(String("body_ang_vel"), DType.float32, LAFAN_N_BODIES * 3))
    cols.append(ColumnSpec(String("motion_id"), DType.int32, 1))
    return cols^


def _write_clip(mut w: TrajectoryStoreWriter, mut rows: LafanRows, motion_id: Int) raises:
    var n = rows.n_rows
    w.append[DType.float32](String("qpos"), _fptr(rows.qpos), n)
    w.append[DType.float32](String("qvel"), _fptr(rows.qvel), n)
    w.append[DType.float32](String("state"), _fptr(rows.state), n)
    w.append[DType.float32](String("privileged"), _fptr(rows.privileged), n)
    w.append[DType.float32](String("body_pos"), _fptr(rows.body_pos), n)
    w.append[DType.float32](String("body_quat"), _fptr(rows.body_quat), n)
    w.append[DType.float32](String("body_vel"), _fptr(rows.body_vel), n)
    w.append[DType.float32](String("body_ang_vel"), _fptr(rows.body_ang_vel), n)
    var ids = List[Int32](length=n, fill=Int32(motion_id))
    w.append[DType.int32](String("motion_id"), _iptr(ids), n)
    w.end_episode()


def main() raises:
    var raw = argv()
    var args = List[String]()
    for i in range(1, len(raw)):
        args.append(String(raw[i]))
    var pkl = _opt(args, String("--pkl"), String(""))
    var out = _opt(args, String("--out"), String("lafan_g1_50hz.h5"))
    var revision = _opt(args, String("--revision"), String("main"))
    var pinned = _opt(args, String("--revision"), String(""))
    var force = _flag(args, String("--force"))
    var download = not _flag(args, String("--no-download"))

    print("LAFAN1 (lafan_29dof.pkl) -> TrajectoryStore")
    print("  out: " + out)
    if exists(out) and not force:
        print("  already present — pass --force to rebuild")
        return

    # ── the pickle ────────────────────────────────────────────────────
    if pkl == "":
        if not download:
            raise Error("--no-download and no --pkl: nothing to convert")
        pkl = hf_download_file(String(REPO), String(PKL_REL), HF_MODEL, String(""), revision, True)
    print("  pickle: " + pkl)
    var digest = sha256_file(pkl)
    if digest != PKL_SHA256:
        raise Error(
            "the pickle's sha256 is " + digest + ", the pinned dump is "
            + String(PKL_SHA256) + " — a different LAFAN dump would move every"
            " gated number; pass the pinned file or re-gate"
        )
    print("  sha256 " + digest[byte=0:16] + " (pinned)")

    # ── convert ───────────────────────────────────────────────────────
    var t0 = perf_counter_ns()
    var clips = load_lafan_clips(pkl)
    print("  " + String(len(clips)) + " clips, read in", Float64(perf_counter_ns() - t0) * 1e-9, "s")
    var env = UnitreeG1[DType.float64]()
    _ = env.reset()

    var part = out + ".part"
    if exists(part):
        remove_file(part)
    var w = TrajectoryStoreWriter(part, _columns(), String(ENV_ID), 0, pinned)
    var total = 0
    for ci in range(len(clips)):
        var t1 = perf_counter_ns()
        var rows = convert_clip(clips[ci], env, ci)
        _write_clip(w, rows, ci)
        w.add_task(ci, clips[ci].name)
        total += rows.n_rows
        print(
            "  [" + String(ci + 1) + "/" + String(len(clips)) + "] " + clips[ci].name
            + ": " + String(clips[ci].n) + " frames @ " + String(clips[ci].fps)
            + " fps -> " + String(rows.n_rows) + " rows  ("
            + String(Float64(perf_counter_ns() - t1) * 1e-9)[byte=0:5] + " s)"
        )

    # the side vectors the observation was built with
    var default = List[Float32](length=G1_N_DOF, fill=Float32(0))
    for j in range(G1_N_DOF):
        default[j] = Float32(g1_default_pos(j))
    w.write_vector[DType.float32](String("default_dof_pos"), _fptr(default), G1_N_DOF)
    var dt = List[Float64](length=1, fill=LAFAN_ENV_DT)
    w.write_vector[DType.float64](String("env_dt"), _dptr(dt), 1)
    w.close()
    rename_over(part, String(out))
    print(
        "wrote " + out + ": " + String(total) + " rows, " + String(len(clips))
        + " episodes in", Float64(perf_counter_ns() - t0) * 1e-9, "s"
    )
