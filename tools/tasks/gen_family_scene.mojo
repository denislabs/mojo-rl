"""Compose every checked-in `.family` into its scene — GENERATED artifacts.

    pixi run gen-family-scenes          # write
    pixi run gen-family-scenes --check  # CI: fail if stale

⚠ A GENERATOR, NOT A TEST. `mojo_rl/tasks/scenes/*.xml` is checked in and read
by `tools/gen_model_dims.py` and by a `ModelDefFromXML`, so it must be produced
by something whose job is to produce it. An earlier draft had P1c's GATE write
the file, which makes a test the source of a build input — run the tests in a
different order, or not at all, and the model changes.

Same shape as `tools/tasks/gen_park_scenes.py`: regenerate, then

    pixi run gen-dims

so the family's dimensions stay a CI assertion against MuJoCo rather than a
comment (`TASK_LAYER_IMPLEMENTATION.md` Gap B).
"""

from std.sys import argv
from mojo_rl.tasks.spec import load_family
from mojo_rl.tasks.family import compose_family, scene_path, SCENE_DIR


def families() -> List[String]:
    """Every checked-in `.family`.

    ⚠ A FUNCTION, NOT A COMPTIME ARRAY. `Array[String, N]` is not
    `ImplicitlyCopyable`, so a comptime table cannot be indexed at runtime and
    the error names materialisation rather than the lookup —
    `studio/scene.mojo`'s `_prop_mjcf_type` records the same trap.
    """
    var out = List[String]()
    out.append(String("mojo_rl/tasks/families/so101_tabletop.family"))
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
        var xml = compose_family(f, SCENE_DIR)

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
