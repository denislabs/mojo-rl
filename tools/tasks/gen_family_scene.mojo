"""Compose every checked-in `.family` into its scene — GENERATED artifacts.

    pixi run gen-family-scenes          # write
    pixi run gen-family-scenes --check  # CI: fail if stale

⚠ A GENERATOR, NOT A TEST. `noeira/tasks/scenes/*.xml` is checked in and read
by `tools/gen_model_dims.py` and by a `ModelDefFromXML`, so it must be produced
by something whose job is to produce it. An earlier draft had P1c's GATE write
the file, which makes a test the source of a build input — run the tests in a
different order, or not at all, and the model changes.

Same shape as `tools/tasks/gen_park_scenes.py`: regenerate, then

    pixi run gen-dims

so the family's dimensions stay a CI assertion against MuJoCo rather than a
comment (`TASK_LAYER_IMPLEMENTATION.md` Gap B).
"""

from std.os import listdir
from std.sys import argv
from noeira.tasks.spec import load_family
from noeira.tasks.family import compose_family, scene_path, scene_dir

comptime TASK_ROOTS = "noeira/tasks,noeira/envs/libero"
"""Every task root (a dir with `families/ tasks/ scenes/`). The generic layer
never names an env package; the generators that enumerate everything do."""


def families() raises -> List[String]:
    """Every checked-in `.family`, sorted.

    ⚠⚠ ENUMERATED, NOT LISTED. It was an explicit list of three so that adding
    a family was a deliberate act; `libero_10` + `libero_90` are TWENTY scene
    families at once (one compile unit per scene — see
    `gen_libero_family._scene_prefix`), and a hand-kept list of twenty-three is
    a list that goes stale silently. What made the explicit list safe survives
    anyway: this is a GENERATOR with a `--check`, so a family whose scene is
    missing or out of date fails CI either way.

    ⚠ SORTED, because `listdir` order is a filesystem detail and the printed
    report should not depend on it.
    """
    var out = List[String]()
    for root in String(TASK_ROOTS).split(","):
        var dir = String(root) + "/families"
        for e in listdir(dir):
            var n = String(e)
            if n.endswith(".family"):
                out.append(dir + "/" + n)
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            if out[j] < out[i]:
                out[i], out[j] = out[j], out[i]
    if len(out) == 0:
        raise Error("no .family under " + String(TASK_ROOTS))
    return out^


def main() raises:
    var args = argv()
    var check = False
    for i in range(len(args)):
        if String(args[i]) == "--check":
            check = True

    var stale = 0
    var fams = families()
    for i in range(len(fams)):
        var f = load_family(fams[i])
        var out = scene_path(f)
        var xml = compose_family(f, scene_dir(f))

        var old = String("")
        var have = True
        try:
            with open(out, "r") as fh:
                old = fh.read()
        except e:
            have = False

        if check:
            if not have or old != xml:
                print("  STALE:", out)
                stale += 1
            else:
                print("  up to date:", out)
        elif not have or old != xml:
            with open(out, "w") as fh:
                fh.write(xml)
            print("  wrote", out)
        else:
            print("  unchanged", out)

    if check and stale != 0:
        raise Error(
            "family scenes: " + String(stale) + " stale. Run"
            " `pixi run gen-family-scenes`, then `pixi run gen-dims`."
        )
