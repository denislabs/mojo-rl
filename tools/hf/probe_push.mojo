# +--------------------------------------------------------------------------+ #
# | Phase 0 — what does the Hub's write API actually ask us for?
# +--------------------------------------------------------------------------+ #
"""Push a tiny dataset to a scratch repo and print every raw answer.

    pixi run mojo run -I . tools/hf/probe_push.mojo
    pixi run mojo run -I . tools/hf/probe_push.mojo --repo me/scratch --keep

`docs/SO101_RECORDING_PLAN.md` phase 0. Two assumptions in that plan are about
someone else's backend rather than about this repo, and both are cheap to
settle before the Parquet writer is written:

  1. **Multipart.** Above some size the Hub answers the LFS batch with a
     `chunk_size`, which is a different upload protocol. Where is the line?
  2. **Xet.** `huggingface_hub` 1.x prefers Xet transfer for new repos. Is the
     plain `basic` LFS path still honoured?

⚠ **QUESTION 1 IS ANSWERED WITHOUT UPLOADING ANYTHING.** The LFS batch request
carries only `{oid, size}` per object — no bytes. So this asks the Hub about
an object the size of a real 194 MB recording, reads whether it comes back
with a `chunk_size`, and never PUTs it. The whole probe moves about a megabyte.

⚠ THIS CREATES A REAL REPO ON A REAL ACCOUNT. It is PRIVATE, it is named
`mojo-rl-push-probe`, and it is DELETED at the end unless `--keep` is passed.
The namespace comes from `whoami`, not from a guess.
"""

from std.os import getenv, makedirs
from std.os.path import exists
from std.sys import argv

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.io.base64 import b64_encode_n
from mojo_rl.io.fileio import file_size, read_file_bytes, write_file_atomic
from mojo_rl.io.hf_push import (
    HF_ENDPOINT, HubPush, HubUpload, LFS_JSON, hf_token, hf_whoami,
)
from mojo_rl.io.http import HttpClient
from mojo_rl.io.json import JsonWriter, parse_json


comptime SCRATCH = "/tmp/mojo_rl_hf_probe"

comptime BIG_MB = 194
"""The size of `record-test_20260828_092736`'s first mp4, in MB — the real
question, asked about a fabricated object so no bytes move."""


def _bytes(s: String) -> List[UInt8]:
    var b = List[UInt8]()
    for i in range(s.byte_length()):
        b.append(s.as_bytes()[i])
    return b^


def _head(s: String, n: Int) -> String:
    """The first `n` bytes of `s`, for printing a truncated href."""
    if s.byte_length() <= n:
        return s.copy()
    return String(s[byte=0:n]) + "..."


def main() raises:
    print("=" * 72)
    print("HuggingFace push probe — docs/SO101_RECORDING_PLAN.md phase 0")
    print("=" * 72)

    var repo = String("")
    var keep = False
    # `--big N` also pushes an N MB SYNTHETIC file. The batch endpoint saying
    # `basic` for a large object is not the same as a large PUT succeeding,
    # and a dataset's mp4s are exactly the objects that matter.
    var big_mb = 0
    var args = argv()
    for i in range(len(args)):
        if String(args[i]) == "--repo" and i + 1 < len(args):
            repo = String(args[i + 1])
        elif String(args[i]) == "--keep":
            keep = True
        elif String(args[i]) == "--big" and i + 1 < len(args):
            big_mb = Int(String(args[i + 1]))

    # `.env` carries HF_TOKEN here, exactly as it carries RL_MONITOR_API_KEY
    # for `RemoteCatalog.from_env`. Read it directly rather than pushing it
    # into the environment — `hf_token()` is the fallback, not the only path.
    var token = String("")
    try:
        var env = load_dotenv(String(".env"))
        if "HF_TOKEN" in env:
            token = env["HF_TOKEN"]
    except:
        pass
    if token == "":
        token = hf_token()
    print(
        "token: " + String(token.byte_length()) + " chars, starts '"
        + _head(token, 4) + "'"
    )

    var probe_client = HttpClient(0, 30000)
    probe_client.bearer(token.copy())
    var who = hf_whoami(token.copy())
    print("whoami: " + who)
    if repo == "":
        repo = who + "/mojo-rl-push-probe"
    print("repo:   " + repo + "  (private)")
    print("")

    # ── the files ─────────────────────────────────────────────────────
    makedirs(String(SCRATCH) + "/meta", exist_ok=True)
    makedirs(String(SCRATCH) + "/data/chunk-000", exist_ok=True)

    var info = String(
        '{\n  "codebase_version": "v3.0",\n  "fps": 30,\n'
        '  "note": "mojo-rl push probe, safe to delete"\n}\n'
    )
    var info_path = String(SCRATCH) + "/meta/info.json"
    write_file_atomic(info_path, _bytes(info))

    # A real parquet, so `preupload` classifies something it would really see.
    var home = getenv("HOME")
    var real_pq = (
        home + "/.cache/huggingface/lerobot/DenisLabs/"
        "record-test_20260828_092736/data/chunk-000/file-000.parquet"
    )
    var pq_path = String(SCRATCH) + "/data/chunk-000/file-000.parquet"
    if exists(real_pq):
        var b = read_file_bytes(real_pq)
        write_file_atomic(pq_path, b)
        print(
            "using a REAL parquet for the classification test: "
            + String(len(b)) + " bytes"
        )
    else:
        # Still binary-looking, still `.parquet` — enough to see uploadMode.
        var b = List[UInt8]()
        b.append(UInt8(ord("P")))
        b.append(UInt8(ord("A")))
        b.append(UInt8(ord("R")))
        b.append(UInt8(ord("1")))
        for i in range(200000):
            b.append(UInt8((i * 31 + 7) & 255))
        write_file_atomic(pq_path, b)
        print("no cached dataset found; using a synthetic .parquet")
    print("")

    var big_path = String(SCRATCH) + "/videos/big.bin"
    if big_mb > 0:
        makedirs(String(SCRATCH) + "/videos", exist_ok=True)
        # Built in one buffer and written atomically: `io/fileio` is the
        # repo's whole-file path and already handles the ~2 GiB syscall cap.
        var blob = List[UInt8](unsafe_uninit_length = big_mb * 1000000)
        for i in range(len(blob)):
            blob[i] = UInt8((i * 31 + 7) & 255)
        write_file_atomic(big_path, blob)
        print(
            "synthetic large object: " + String(file_size(big_path))
            + " bytes  (NOT a real recording — no dataset bytes leave this box)"
        )
        print("")

    # ── 1. create ─────────────────────────────────────────────────────
    print("── 1. create_repo ──────────────────────────────────────────")
    var p = HubPush(repo.copy(), token=token.copy())
    _ = p.create_repo(private=True)
    print("")

    # ── 2. preupload ──────────────────────────────────────────────────
    print("── 2. preupload: which files go LFS? ───────────────────────")
    var files = List[HubUpload]()
    files.append(HubUpload(String("meta/info.json"), info_path.copy()))
    files.append(
        HubUpload(String("data/chunk-000/file-000.parquet"), pq_path.copy())
    )
    if big_mb > 0:
        files.append(
            HubUpload(String("videos/big.bin"), big_path.copy())
        )
    p.preupload(files)
    for i in range(len(files)):
        print(
            "    " + files[i].repo_path + "  " + String(files[i].size)
            + " B  -> " + files[i].mode
            + ("  (gitignored)" if files[i].ignored else "")
        )
    print("")

    # ── 3. the batch, asked about a BIG object without sending it ─────
    print("── 3. lfs batch: is a " + String(BIG_MB) + " MB object multipart? ──")
    var big = BIG_MB * 1000 * 1000
    var sizes = [10 * 1000 * 1000, 60 * 1000 * 1000, big, 3 * 1000 * 1000 * 1000]

    var w = JsonWriter()
    w.begin_object()
    w.member(String("operation"), String("upload"))
    w.key(String("transfers"))
    w.begin_array()
    w.string(String("basic"))
    w.string(String("multipart"))
    w.end_array()
    w.member(String("hash_algo"), String("sha256"))
    w.key(String("objects"))
    w.begin_array()
    for k in range(len(sizes)):
        # A fabricated oid: 64 hex characters the Hub has certainly never
        # seen, so it must answer with an upload action rather than "have it".
        var hexd = String("0123456789abcdef")
        var oid = String("")
        for j in range(64):
            oid += chr(Int(hexd.as_bytes()[(k * 7 + j) % 16]))
        w.begin_object()
        w.member(String("oid"), oid)
        w.member(String("size"), sizes[k])
        w.end_object()
    w.end_array()
    w.end_object()

    var bc = HttpClient(0, 30000)
    bc.bearer(token.copy())
    bc.header(String("Accept"), String(LFS_JSON))
    bc.max_body(1 << 24)
    var br = bc.request(
        String("POST"),
        String(HF_ENDPOINT) + "/datasets/" + repo
        + ".git/info/lfs/objects/batch",
        _bytes(w.done()),
        String(LFS_JSON),
    )
    print("    HTTP " + String(br.status))
    var btext = br.text()
    print("    transfer: " + ("multipart" if btext.find('"multipart"') >= 0 else "basic"))
    print("    chunk_size present: " + ("YES" if btext.find("chunk_size") >= 0 else "no"))
    print("    ── raw (first 1200 bytes) ──")
    print(_head(btext, 1200))
    print("")

    # ── 4 + 5. upload the small ones and commit ───────────────────────
    print("── 4+5. upload + commit ────────────────────────────────────")
    p.upload_lfs(files)
    var url = p.push(files, String("mojo-rl push probe"))
    print("    commit: " + url)
    print("")

    # ── verify by reading it back through io/hf.mojo ──────────────────
    print("── verify: the tree API sees the files ─────────────────────")
    var vr = probe_client.get(
        String(HF_ENDPOINT) + "/api/datasets/" + repo + "/tree/main?recursive=1"
    )
    print("    HTTP " + String(vr.status))
    print(_head(vr.text(), 800))
    print("")

    if keep:
        print("--keep: leaving " + repo + " in place")
    else:
        _ = p.delete_repo()

    print("[PROBE DONE]")

