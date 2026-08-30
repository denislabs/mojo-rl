# +--------------------------------------------------------------------------+ #
# | Import a LeRobot v3 dataset into a TrajectoryStore — no Python
# +--------------------------------------------------------------------------+ #
"""Download (if needed) and convert a LeRobot v3.0 dataset for ACT training.

    pixi run mojo run -I . examples/so101/act_so101_import_dataset.mojo \\
        --repo DenisLabs/record-test_20260828_092736

    pixi run mojo run -I . examples/so101/act_so101_import_dataset.mojo \\
        --root /path/to/snapshot --out /tmp/store.h5 --height 480 --width 640

Replaces `tools/act/lerobot_v3_to_store.py`. The Python script needed
`huggingface_hub`, `pyarrow`, `imageio`, `Pillow`, `numpy` and `h5py`; this
needs `curl` and `ffmpeg` on PATH, both of which the pixi environment provides.

Output goes to `~/.cache/mojo_rl/act_so101/<Org>__<name>_<H>x<W>.h5`, which is
the path `examples/so101/act_so101_train_gpu.mojo` expects in `ACT_STORE`.

Options
-------
--repo REPO        HuggingFace dataset repo id (downloaded if not cached)
--root DIR         a local dataset directory; skips resolution entirely
--out PATH         output .h5 (default: the cache path above)
--height / --width resize target (default 240x320; the recording is 480x640)
--revision REV     branch or commit (default `main`)
--force            rebuild even if the output already exists
--no-download      fail rather than fetch anything over the network

⚠ REBUILDING IS NOT FREE — the 50-episode recording is a 700 MB download and a
6.8 GB store, and every frame is H.264-decoded and resampled. Existing output
is left alone unless `--force` says otherwise.
"""

from std.os import makedirs
from std.os.path import exists
from std.sys import argv

from mojo_rl.data.lerobot import (
    import_lerobot_v3, mojo_rl_cache, repo_slug, resolve_dataset_root,
)


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


def main() raises:
    var raw = argv()
    var args = List[String]()
    for i in range(1, len(raw)):
        args.append(String(raw[i]))

    var repo = _opt(args, String("--repo"), String(""))
    var root = _opt(args, String("--root"), String(""))
    var revision = _opt(args, String("--revision"), String("main"))
    # `source_commit` records what was ASKED FOR, and only when it was asked
    # for: "main" is a moving reference, not a commit, so writing it into the
    # manifest would claim provenance the store does not have.
    var pinned = _opt(args, String("--revision"), String(""))
    var height = atol(_opt(args, String("--height"), String("240")))
    var width = atol(_opt(args, String("--width"), String("320")))
    var force = _flag(args, String("--force"))
    var download = not _flag(args, String("--no-download"))

    if repo == "" and root == "":
        raise Error(
            "need --repo <org/name> or --root <dir>; see the docstring at the"
            " top of this file"
        )

    var out = _opt(args, String("--out"), String(""))
    if out == "":
        if repo == "":
            raise Error("--out is required when only --root is given")
        var dir = mojo_rl_cache() + "/act_so101"
        makedirs(dir, exist_ok=True)
        out = (
            dir + "/" + repo_slug(repo) + "_" + String(height) + "x"
            + String(width) + ".h5"
        )

    print("LeRobot v3 -> TrajectoryStore")
    print("  out: " + out)
    if exists(out) and not force:
        print("  already present — pass --force to rebuild")
        return

    if root == "":
        root = resolve_dataset_root(repo, revision, download, True)
    print("  root: " + root)
    print("")

    import_lerobot_v3(
        root,
        out,
        height,
        width,
        String("lerobot/") + repo if repo != "" else String(""),
        pinned,
        True,
    )
