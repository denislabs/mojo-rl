"""A demonstration's FIXTURE poses, written into a built `Model` — once.

    from noeira.tasks.libero_fixtures import patch_fixtures, fixtures_dump_path
    var placed = patch_fixtures(dump, demo, fmd.body_names, m)

LIBERO re-draws every fixture's xy inside a 19-20 mm band at each reset, and
the pose it drew lives in the `model_file` attribute of the recording, which
`io/hdf5` cannot read (no `H5A`). `tools/tasks/libero_demo_success.py` extracts
it with MuJoCo into `references/libero_demos/_dumps/<suite>/<task>.dump`, one
`DEMO demo_<k>` block per demonstration with a `FIX body px py pz qw qx qy qz`
line per fixture. This reads that file and writes the seven columns into
`bodies[b, BODY_IDX_POS_* / QUAT_*]` — what forward kinematics reads, so the
patch is complete without a rebuild.

⚠ WORTH 7 dB TO A CAMERA. `libero_camera_video.mojo` measured
`turn_on_the_stove` frame 78 at 32.39 dB with the demo's poses and 25.34 without:
a centimetre on the stove is a strip of wrong pixels along every edge of it.
The camera gate, the demo-success gate, the video tool and the re-render all
apply the same patch, so it lives here rather than in each of them.

⚠ A MISSING DUMP IS THE CALLER'S DECISION, NOT THIS FILE'S. `fixtures_dump_path`
names where the dump should be; a caller that can live with the scene's band
centre warns and goes on, one that cannot raises. This file only refuses a
dump that names a body the scene does not have, or a malformed line.
"""

from std.os.path import exists, dirname, basename

from noeira.physics3d.fields import Model
from noeira.physics3d.fields.dims import DimsLike
from noeira.physics3d.gpu.constants import (
    MODEL_BODY_SIZE, BODY_IDX_POS_X, BODY_IDX_QUAT_X, BODY_IDX_QUAT_W,
)


def fixtures_dump_path(demos_dir: String, suite: String, task: String) -> String:
    """`<demos_dir>/_dumps/<suite>/<task>.dump` — where `libero-demo-dump`
    writes it. `task` is the stem WITHOUT the `<suite>__` prefix."""
    return demos_dir + "/_dumps/" + suite + "/" + task + ".dump"


def patch_fixtures[DTYPE: DType, D: DimsLike](
    dump_path: String, demo: Int, body_names: List[String],
    mut m: Model[DTYPE, D],
) raises -> Int:
    """Write demo `demo`'s `FIX` poses into `m.bodies`; return how many were
    placed (0 when the dump has no block for this demo).

    ⚠ HOST RECORDS ONLY. `m.bodies.data` is the host copy; a device consumer
    that binds `bodies` (the physics kernels do, the camera kernel does not —
    it reads `xpos`/`xquat` that a host FK produced from these records) must
    upload it afterwards."""
    var text: String
    with open(dump_path, "r") as fh:
        text = fh.read()
    var want = String("demo_") + String(demo)
    var cur = String("")
    var placed = 0
    var lines = text.split("\n")
    for li in range(len(lines)):
        var l = String(String(lines[li]).strip())
        if l.startswith("DEMO "):
            var toks = l.split(" ")
            cur = String(toks[1])
        elif l.startswith("FIX ") and cur == want:
            var toks = l.split(" ")
            if len(toks) < 9:
                raise Error(dump_path + ": malformed FIX line: " + l)
            var name = String(toks[1])
            var bi = -1
            for b in range(len(body_names)):
                if String(body_names[b]) == name:
                    bi = b
            if bi <= 0:
                raise Error(dump_path + ": no body '" + name + "' in the scene")
            var o = bi * MODEL_BODY_SIZE
            m.bodies.data[o + BODY_IDX_POS_X + 0] = Scalar[DTYPE](Float64(String(toks[2])))
            m.bodies.data[o + BODY_IDX_POS_X + 1] = Scalar[DTYPE](Float64(String(toks[3])))
            m.bodies.data[o + BODY_IDX_POS_X + 2] = Scalar[DTYPE](Float64(String(toks[4])))
            m.bodies.data[o + BODY_IDX_QUAT_W] = Scalar[DTYPE](Float64(String(toks[5])))
            m.bodies.data[o + BODY_IDX_QUAT_X + 0] = Scalar[DTYPE](Float64(String(toks[6])))
            m.bodies.data[o + BODY_IDX_QUAT_X + 1] = Scalar[DTYPE](Float64(String(toks[7])))
            m.bodies.data[o + BODY_IDX_QUAT_X + 2] = Scalar[DTYPE](Float64(String(toks[8])))
            placed += 1
    return placed


def dump_path_from_index(index_path: String, entry: String) raises -> String:
    """The dump a line of an `index.txt` names, resolved NEXT TO THE INDEX.

    ⚠⚠ THE INDEXES USED TO CARRY ABSOLUTE PATHS, AND THE FIRST BOX RUN DIED ON
    THE MAC'S HOME DIRECTORY. `libero_demo_success.py`, `libero_init_table.py`
    and `libero_camera_gate.py` wrote `os.path.join(out_dir, ...)` from wherever
    they ran, so an index copied to another machine pointed back at the
    machine it was written on. The writers now emit basenames; this reads
    both: an entry that exists is taken as given, anything else is looked
    for beside the index — which is where every writer puts its dumps — and a
    dump missing from BOTH places is named with both paths tried.
    """
    if exists(entry):
        return entry
    var beside = dirname(index_path) + "/" + basename(entry)
    if exists(beside):
        return beside
    raise Error(
        "no dump at '" + entry + "' nor beside its index at '" + beside
        + "' — the index names a file this machine does not have; re-run"
        " the dump tool here, or copy the whole _dumps/<suite> directory"
    )
