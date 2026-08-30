# +--------------------------------------------------------------------------+ #
# | The Mojo LeRobot v3 importer, against the Python converter
# +--------------------------------------------------------------------------+ #
"""Imports a synthetic v3.0 dataset and compares the store to the reference.

    pixi run python tools/act/make_synthetic_lerobot_v3.py --out /tmp/lerobot_synth
    pixi run mojo run -I . tests/data/test_lerobot_import.mojo /tmp/lerobot_synth

The oracle is `tools/act/lerobot_v3_to_store.py`, the converter that produced
the ACT store this port has been training against — so what is being asserted
is not "the Mojo importer is plausible" but "the Mojo importer and the Python
one are the same function".

The decoders underneath already have their own bit-exact gates (parquet vs
pyarrow, resize vs Pillow, JSON). ⚠ WHAT ONLY THIS FILE COVERS IS THE
BOOKKEEPING: episode ordering, `ep_offset` / `ep_len`, camera slot order, the
HWC->CHW transpose, and routing each video frame to its flat row. Every one of
those failures produces a store of the right SHAPE holding the wrong numbers,
which is why the synthetic dataset (see the generator's docstring) is built out
of unequal episode lengths, out-of-order episode rows, two video files per
camera, a frame gap inside a file, and per-frame pixel content that identifies
its own (camera, index).

Comparison is exact on every column and every byte. A resize that is one LSB
off is a different dataset, not a rounding difference.
"""

from std.os.path import exists
from std.sys import argv

from mojo_rl.data.lerobot import import_lerobot_v3
from mojo_rl.data.store import TrajectoryStore


comptime DEFAULT_REF = "/tmp/lerobot_synth"
comptime SENTINEL = "/case.txt"
comptime GEN = (
    "pixi run python tools/act/make_synthetic_lerobot_v3.py --out /tmp/lerobot_synth"
)

comptime USAGE = (
    "usage: mojo run -I . tests/data/test_lerobot_import.mojo <case_dir>\n"
    "  generate <case_dir> with:\n"
    "    pixi run python tools/act/make_synthetic_lerobot_v3.py"
    " --out <case_dir>"
)


def check(mut fails: Int, name: String, ok: Bool, detail: String = String("")):
    if ok:
        print("  PASS  " + name + ("  " + detail if detail else ""))
    else:
        fails += 1
        print("  FAIL  " + name + ("  " + detail if detail else ""))


def compare_f32(
    mut fails: Int,
    name: String,
    a: List[Scalar[DType.float32]],
    b: List[Scalar[DType.float32]],
) raises -> Int:
    if len(a) != len(b):
        fails += 1
        print(
            "  FAIL  " + name + "  length " + String(len(b)) + " vs reference "
            + String(len(a))
        )
        return 0
    var diff = 0
    var worst = 0.0
    for i in range(len(a)):
        if a[i] != b[i]:
            diff += 1
            var d = abs(Float64(a[i]) - Float64(b[i]))
            if d > worst:
                worst = d
    check(
        fails,
        name,
        diff == 0,
        String(len(a)) + " values compared"
        if diff == 0
        else String(diff) + "/" + String(len(a)) + " differ, max|delta| = "
        + String(worst),
    )
    return len(a)


def compare_u8(
    mut fails: Int,
    name: String,
    a: List[Scalar[DType.uint8]],
    b: List[Scalar[DType.uint8]],
) raises -> Int:
    if len(a) != len(b):
        fails += 1
        print(
            "  FAIL  " + name + "  length " + String(len(b)) + " vs reference "
            + String(len(a))
        )
        return 0
    var diff = 0
    var worst = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            diff += 1
            var d = Int(a[i]) - Int(b[i])
            if d < 0:
                d = -d
            if d > worst:
                worst = d
    check(
        fails,
        name,
        diff == 0,
        String(len(a)) + " bytes compared"
        if diff == 0
        else String(diff) + "/" + String(len(a)) + " differ, max|delta| = "
        + String(worst),
    )
    return len(a)


def read_case(path: String, mut out: List[String]) raises:
    var f = open(path, "r")
    var text = f.read()
    f.close()
    for line in text.splitlines():
        var s = String(line)
        if s != "":
            out.append(s^)
    if len(out) != 4:
        raise Error(
            "case.txt should hold <root> <reference.h5> <height> <width>;"
            " found " + String(len(out)) + " lines"
        )


def main() raises:
    var args = argv()
    var case_dir = String(DEFAULT_REF) if len(args) < 2 else String(args[1])
    if len(args) > 2:
        print(USAGE)
        raise Error("test_lerobot_import: expected at most one argument")
    if not exists(case_dir + SENTINEL):
        raise Error(
            "no synthetic dataset at " + case_dir + " — generate it with:\n    "
            + GEN
        )

    var spec = List[String]()
    read_case(case_dir + "/case.txt", spec)
    var root = String(spec[0])
    var ref_path = String(spec[1])
    var height = atol(spec[2])
    var width = atol(spec[3])
    var mine = case_dir + "/mojo.h5"

    print("LeRobot v3 import: Mojo vs the Python converter")
    print("  dataset:   " + root)
    print("  reference: " + ref_path)
    print("")

    import_lerobot_v3(
        root, mine, height, width, String(""), String(""), False
    )

    var a = TrajectoryStore(String(ref_path))
    var b = TrajectoryStore(String(mine))
    var fails = 0
    var compared = 0

    check(
        fails,
        "row count",
        a.n_rows() == b.n_rows(),
        String(b.n_rows()) + " rows",
    )
    check(
        fails,
        "episode count",
        a.n_episodes() == b.n_episodes(),
        String(b.n_episodes()) + " episodes",
    )
    if a.n_rows() != b.n_rows() or a.n_episodes() != b.n_episodes():
        raise Error("gate: the stores disagree on shape")

    # ── the episode index ─────────────────────────────────────────────
    var ep_ok = True
    var ep_detail = String("")
    for e in range(a.n_episodes()):
        if (
            a.episodes.start_of(e) != b.episodes.start_of(e)
            or a.episodes.length_of(e) != b.episodes.length_of(e)
        ):
            ep_ok = False
            ep_detail = (
                "episode " + String(e) + ": offset/len ("
                + String(b.episodes.start_of(e)) + ", "
                + String(b.episodes.length_of(e)) + ") vs reference ("
                + String(a.episodes.start_of(e)) + ", "
                + String(a.episodes.length_of(e)) + ")"
            )
            break
    check(
        fails,
        "ep_offset / ep_len",
        ep_ok,
        String(a.n_episodes()) + " episodes compared" if ep_ok else ep_detail,
    )

    # ── the columns ───────────────────────────────────────────────────
    compared += compare_f32(
        fails,
        "qpos",
        a.load_column[DType.float32](String("qpos")),
        b.load_column[DType.float32](String("qpos")),
    )
    compared += compare_f32(
        fails,
        "action",
        a.load_column[DType.float32](String("action")),
        b.load_column[DType.float32](String("action")),
    )
    compared += compare_u8(
        fails,
        "images",
        a.load_column[DType.uint8](String("images")),
        b.load_column[DType.uint8](String("images")),
    )

    print("")
    print("  " + String(compared) + " values compared")
    if compared == 0:
        raise Error("gate: zero values compared — this run proved nothing")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILURES")
        raise Error("lerobot import gate failed")
