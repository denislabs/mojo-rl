# +--------------------------------------------------------------------------+ #
# | Write a dataset, push it, pull it back, and compare the two stores
# +--------------------------------------------------------------------------+ #
"""The whole path, through the real Hub, with nothing mocked.

    pixi run mojo run -I . tools/hf/dataset_roundtrip.mojo
    pixi run mojo run -I . tools/hf/dataset_roundtrip.mojo --keep

    LeRobotWriter  ->  directory  ->  import  ->  store A
                            |
                            +-- push -> Hub -> hf_download_dataset -> import -> store B

    A and B must be identical.

⚠ THIS IS NOT A GATE AND MUST NOT BECOME ONE. It needs the network, a write
token and about a minute; `pixi run test-io` is defined as offline and
deterministic. `tests/data/test_lerobot_write.mojo` covers the writer without
leaving the box. This covers the part a local gate cannot: that the bytes
survive an upload and a download.

⚠ IT CREATES AND DELETES A REAL PRIVATE REPO on the account the token belongs
to, named `mojo-rl-roundtrip`. `--keep` leaves it for inspection.

The frames are synthetic flat colours, so nothing recorded leaves the machine.
"""

from std.os import makedirs
from std.os.path import exists
from std.sys import argv

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.data.lerobot import hf_download_dataset, import_lerobot_v3
from mojo_rl.data.lerobot_push import dataset_files, push_lerobot_dataset
from mojo_rl.data.lerobot_write import LeRobotWriter
from mojo_rl.data.store import TrajectoryStore
from mojo_rl.io.hf_push import HubPush, hf_whoami


comptime H = 48
comptime W = 64
comptime FPS = 30
comptime EP_LENS: InlineArray[Int, 3] = [5, 3, 6]
comptime N_ROWS = 14


def _build(root: String) raises:
    var sn = List[String]()
    for n in [String("a.pos"), String("b.pos"), String("c.pos")]:
        sn.append(n)
    var an = List[String]()
    for n in [String("d.pos"), String("e.pos")]:
        an.append(n)
    var cams = List[String]()
    cams.append(String("observation.images.front"))
    cams.append(String("observation.images.side"))

    var w = LeRobotWriter(root, FPS, sn^, an^, cams^, H, W)
    var lens = materialize[EP_LENS]()
    for ep in range(3):
        w.begin_episode(String("roundtrip task"))
        for t in range(lens[ep]):
            var st = List[Float64]()
            var ac = List[Float64]()
            for d in range(3):
                st.append(Float64(ep * 16 + t * 4 + d) + 0.25)
            for d in range(2):
                ac.append(Float64(ep * 16 + t * 4 + d) * -1.0 - 0.5)
            var frames = List[List[UInt8]]()
            for cam in range(2):
                var f = List[UInt8](unsafe_uninit_length = W * H * 3)
                for p in range(W * H):
                    f[p * 3] = UInt8(20 + ep * 60)
                    f[p * 3 + 1] = UInt8(20 + t * 20)
                    f[p * 3 + 2] = UInt8(40 + cam * 120)
                frames.append(f^)
            w.add_frame(st, ac, frames)
        w.end_episode()
    w.close(verbose=False)


def _compare(a_path: String, b_path: String) raises -> Int:
    var a = TrajectoryStore(a_path)
    var b = TrajectoryStore(b_path)
    if a.n_rows() != b.n_rows() or a.n_episodes() != b.n_episodes():
        raise Error(
            "roundtrip: local store has " + String(a.n_rows()) + " rows / "
            + String(a.n_episodes()) + " episodes, the downloaded one has "
            + String(b.n_rows()) + " / " + String(b.n_episodes())
        )
    for e in range(a.n_episodes()):
        if (
            a.episodes.start_of(e) != b.episodes.start_of(e)
            or a.episodes.length_of(e) != b.episodes.length_of(e)
        ):
            raise Error("roundtrip: episode " + String(e) + " differs")

    var n = a.n_rows()
    var compared = 0

    var qa = List[Float32](unsafe_uninit_length = n * 3)
    var qb = List[Float32](unsafe_uninit_length = n * 3)
    a.read_range[DType.float32](
        String("qpos"), 0, n,
        qa.unsafe_ptr().unsafe_bitcast[Scalar[DType.float32]]()
        .as_unsafe_any_origin(),
    )
    b.read_range[DType.float32](
        String("qpos"), 0, n,
        qb.unsafe_ptr().unsafe_bitcast[Scalar[DType.float32]]()
        .as_unsafe_any_origin(),
    )
    for i in range(len(qa)):
        if qa[i] != qb[i]:
            raise Error("roundtrip: qpos[" + String(i) + "] differs")
        compared += 1

    var aa = List[Float32](unsafe_uninit_length = n * 2)
    var ab = List[Float32](unsafe_uninit_length = n * 2)
    a.read_range[DType.float32](
        String("action"), 0, n,
        aa.unsafe_ptr().unsafe_bitcast[Scalar[DType.float32]]()
        .as_unsafe_any_origin(),
    )
    b.read_range[DType.float32](
        String("action"), 0, n,
        ab.unsafe_ptr().unsafe_bitcast[Scalar[DType.float32]]()
        .as_unsafe_any_origin(),
    )
    for i in range(len(aa)):
        if aa[i] != ab[i]:
            raise Error("roundtrip: action[" + String(i) + "] differs")
        compared += 1

    # ⚠ THE IMAGES MUST BE BYTE-IDENTICAL, not merely close. The mp4 that came
    # back is the same LFS object that went up, so the decode is the same
    # decode; anything else means the transfer changed bytes.
    var per_row = 2 * 3 * H * W
    var ia = List[UInt8](unsafe_uninit_length = per_row)
    var ib = List[UInt8](unsafe_uninit_length = per_row)
    for r in range(n):
        a.read_range[DType.uint8](
            String("images"), r, r + 1,
            ia.unsafe_ptr().unsafe_bitcast[Scalar[DType.uint8]]()
            .as_unsafe_any_origin(),
        )
        b.read_range[DType.uint8](
            String("images"), r, r + 1,
            ib.unsafe_ptr().unsafe_bitcast[Scalar[DType.uint8]]()
            .as_unsafe_any_origin(),
        )
        for i in range(per_row):
            if ia[i] != ib[i]:
                raise Error(
                    "roundtrip: image byte " + String(i) + " of row "
                    + String(r) + " differs (" + String(Int(ia[i])) + " vs "
                    + String(Int(ib[i])) + ")"
                )
        compared += per_row
    return compared


def main() raises:
    print("=" * 72)
    print("LeRobot dataset round trip through the Hub")
    print("=" * 72)

    var keep = False
    var args = argv()
    for i in range(len(args)):
        if String(args[i]) == "--keep":
            keep = True

    var token = String("")
    try:
        var env = load_dotenv(String(".env"))
        if "HF_TOKEN" in env:
            token = env["HF_TOKEN"]
    except:
        pass

    var local = String("/tmp/mojo_rl_rt_local")
    var pulled = String("/tmp/mojo_rl_rt_pulled")

    print("\n── 1. write ────────────────────────────────────────────────")
    _build(local)
    var rels = dataset_files(local)
    print("  " + String(len(rels)) + " files:")
    for i in range(len(rels)):
        print("    " + rels[i])

    print("\n── 2. import locally ───────────────────────────────────────")
    import_lerobot_v3(local, local + ".h5", H, W, verbose=False)
    print("  " + local + ".h5")

    print("\n── 3. push ─────────────────────────────────────────────────")
    var repo = hf_whoami(token.copy()) + "/mojo-rl-roundtrip"
    print("  -> " + repo)
    _ = push_lerobot_dataset(
        local, repo.copy(), String("round trip"), True, token.copy()
    )

    print("\n── 4. pull ─────────────────────────────────────────────────")
    _ = hf_download_dataset(
        repo.copy(), pulled.copy(), String("main"), False, token.copy()
    )
    var back = dataset_files(pulled)
    # ⚠ RE-LIST THE LOCAL SIDE. `push_lerobot_dataset` writes `README.md` if
    # there is not one, so the listing taken before the push is one short —
    # and the count check would then fail on a round trip that worked.
    var sent = dataset_files(local)
    print("  " + String(len(back)) + " files back, " + String(len(sent)) + " sent")
    if len(back) != len(sent):
        raise Error(
            "roundtrip: pushed " + String(len(sent)) + " files, pulled "
            + String(len(back))
        )
    for i in range(len(sent)):
        if sent[i] != back[i]:
            raise Error(
                "roundtrip: file " + String(i) + " is '" + sent[i]
                + "' locally and '" + back[i] + "' on the Hub"
            )

    print("\n── 5. import the pulled copy and compare ───────────────────")
    import_lerobot_v3(pulled, pulled + ".h5", H, W, verbose=False)
    var n = _compare(local + ".h5", pulled + ".h5")
    print("  " + String(n) + " values identical across the round trip")

    if keep:
        print("\n--keep: leaving " + repo)
    else:
        var p = HubPush(repo.copy(), token=token.copy())
        _ = p.delete_repo()
    print("\n[ROUND TRIP OK]")
