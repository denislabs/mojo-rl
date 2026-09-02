"""`expand_mjcf` is the IDENTITY on a model that composes nothing — P1a gate.

WHAT THIS PROTECTS
==================
P1a routed `ModelDefFromXML.xml_text()` through `expand_mjcf`, so the comptime
leg and `runtime_load` finally agree on what a model IS. Before it, they did
not:

    parse_model_runtime(path)      → expanded   (CPU only)
    ModelDefFromXML[xml_path=path] → raw        (the only leg with a GPU)

That change touches EVERY shipped model, and its safety rests on one claim:
expansion does nothing to a document with no `<include>`, `<attach>` or
`<frame>`. This asserts the claim rather than believing it.

⚠⚠ THE TEST HAS TWO HALVES AND NEEDS BOTH.

An identity assertion ALONE is vacuous in the most embarrassing possible way:
`def expand_mjcf(x, d): return x` passes it on every model in the tree. So the
POSITIVE CONTROL is not optional — `fixtures/attach/scene.xml` composes three
instances of two assets, and expansion MUST change it. One half proves the
expander is quiet where it should be; the other proves it is awake at all.
See `feedback_a_hit_count_is_not_coverage_of_the_branch`.

⚠ THE WALK IS THE COVERAGE, so it prints what it compared. A recursive walk
that silently finds nothing — a moved asset root, a permissions hole — would
report "0 mismatches" and look like a pass. `MIN_MODELS` is the floor that
turns that into a failure, and the count is printed beside the mismatch count
because "rows compared" is the number that makes "rows differing" mean
anything.

⚠ WHY `mojo_rl/` AND NOT A HAND-WRITTEN LIST. A list goes stale the first time
someone adds a model, and it goes stale SILENTLY — the new model is simply not
covered. The walk cannot.

Run: pixi run mojo run -I . tests/physics3d/test_expand_identity.mojo
"""

from std.os import listdir
from std.os.path import isdir

from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.model_def_from_xml import ModelDefFromXML


# The shipped asset tree. Every `.xml` under here is a model this repository
# builds, and every one of them must be unchanged by expansion today.
comptime ASSET_ROOT = String("mojo_rl")

# ⚠ A FLOOR, NOT A COUNT. 58 `.xml` files were under `mojo_rl/` on 2026-09-02;
# this is deliberately slack so that adding a model does not fail the gate,
# while a walk that collapses to nothing still does. Do not "fix" a failure
# here by lowering it — a drop means the walk broke or assets moved.
comptime MIN_MODELS = 50

# The positive control: a scene that DOES compose. Two assets, three
# instances, one rotated frame.
comptime COMPOSED = String("tests/physics3d/fixtures/attach/scene.xml")

# ⚠ THE ORACLE IS MuJoCo 3.10.0, read off the fixture on 2026-09-02:
#   nbody 5  njnt 3  nq 15  nv 13  ngeom 5  nu 1  nsite 3  nmat 2
# `nbody` COUNTS THE WORLD BODY and `FlatModelDef.bodies` does not, so the
# number this file compares against is 4. Getting that off by one would make
# the gate pass on a scene with one instance missing.
comptime COMPOSED_NBODY_NOWORLD = 4

# The same scene as a COMPTIME model def — the leg that could not read it
# before P1a. Dimensions hand-supplied from MuJoCo above, which is what a
# `.family` will do through `tools/gen_model_dims.py`.
comptime ComposedDef = ModelDefFromXML[
    nbody=5,
    njoint=3,
    nq=15,
    nv=13,
    ngeom=5,
    nact=1,
    nmat=2,
    nsite=3,
    timestep=0.002,
    xml_path=COMPOSED,
]


def _walk(dir: String, mut out: List[String]) raises:
    """Every `.xml` under `dir`, recursively, as full paths."""
    var names = listdir(dir)
    for i in range(len(names)):
        var name = String(names[i])
        if name.startswith("."):
            continue
        var full = dir + "/" + name
        if isdir(full):
            _walk(full, out)
        elif name.endswith(".xml"):
            out.append(full)


def _read(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def _dirname(path: String) -> String:
    """MuJoCo's asset-resolution base: the directory of the model file."""
    var cut = path.rfind("/")
    return String(path[byte=0:cut]) if cut > 0 else String("")


def main() raises:
    print("=== expand_mjcf is the identity on a flat model — P1a ===")

    # ── half 1: every shipped model is untouched ──────────────────────────
    var models = List[String]()
    _walk(ASSET_ROOT, models)

    var compared = 0
    var differing = 0
    var first_bad = String("")
    for i in range(len(models)):
        var path = models[i]
        var raw = _read(path)
        var expanded = expand_mjcf(raw, _dirname(path))
        compared += 1
        if expanded != raw:
            differing += 1
            if first_bad == "":
                first_bad = path
                print("  DIFFERS:", path)
                print("    raw     ", raw.byte_length(), "bytes")
                print("    expanded", expanded.byte_length(), "bytes")

    # ⚠ BOTH NUMBERS, ALWAYS. "0 differing" out of 0 compared is the failure
    # this line exists to make impossible to misread as a pass.
    print("--- compared", compared, "shipped models,", differing, "differ ---")

    if compared < MIN_MODELS:
        raise Error(
            "expand identity: walked only "
            + String(compared)
            + " models under '"
            + ASSET_ROOT
            + "', expected at least "
            + String(MIN_MODELS)
            + ". The walk found nothing — assets moved, or the root is wrong."
            + " This is NOT a pass."
        )
    if differing != 0:
        raise Error(
            "expand identity: "
            + String(differing)
            + " shipped model(s) are CHANGED by expansion, first '"
            + first_bad
            + "'. Either that model genuinely composes (in which case P1a"
            + " changed what it parses, and its dims must be re-checked"
            + " against MuJoCo), or the expander now matches something it"
            + " should not — `<framepos>` is a SENSOR, not a `<frame>`."
        )
    print("  OK: expansion changes no shipped model")

    # ── half 2: the positive control — it is awake ────────────────────────
    # Without this, `return xml` passes half 1 on every model in the tree.
    # ⚠ THROUGH `ComposedDef`, NOT `_read`. Reading the file here would test
    # a source selection the parser does not use; `raw_xml_text()` /
    # `xml_text()` are the two halves of the thing P1a split, so comparing
    # exactly those two is what makes this control a control.
    var scene_raw = ComposedDef.raw_xml_text()
    var scene_exp = ComposedDef.xml_text()
    print("--- positive control:", COMPOSED, "---")
    print("    raw     ", scene_raw.byte_length(), "bytes")
    print("    expanded", scene_exp.byte_length(), "bytes")
    if scene_exp == scene_raw:
        raise Error(
            "expand identity: the COMPOSING fixture came back unchanged, so"
            " half 1 above proves nothing — an expander that returns its"
            " input passes it on every model. `<attach>`/`<frame>` expansion"
            " is broken or was bypassed."
        )
    # The composed scene must have gained bodies and lost its composition tags.
    if scene_exp.find("<attach") != -1 or scene_exp.find("<frame ") != -1:
        raise Error(
            "expand identity: the expanded scene still contains `<attach>` or"
            " `<frame>` — expansion ran but did not finish."
        )
    print("  OK: the composing fixture IS expanded")

    # ── half 3: the COMPTIME leg reads it — what P1a was for ──────────────
    # ⚠ THIS IS THE HALF THAT WOULD HAVE FAILED BEFORE P1a, and it fails in
    # the way `runtime_load.mojo`'s header records: not with a diagnostic, but
    # with a model missing most of itself. `ModelDefFromXML.xml_text()` handed
    # `parse_xml_full` the RAW scene, whose `<worldbody>` holds one floor geom
    # and three `<attach/>` tags the parser does not know — so the four bodies
    # of `arm1_`/`cube1_`/`cube2_` simply were not there.
    #
    # Deliberately parsed through `ComposedDef.xml_text()` rather than
    # `expand_mjcf` directly: the point is what the COMPTIME PATH sees, and
    # calling the expander here would test the expander a third time instead.
    print("--- half 3: the comptime leg parses the composed scene ---")
    var fmd = parse_xml_full(
        ComposedDef.xml_text(), ComposedDef.asset_base_dir()
    )
    print("    bodies (world excluded):", len(fmd.bodies))
    print("    geoms                  :", len(fmd.geoms))
    print("    joints                 :", len(fmd.joints))
    if len(fmd.bodies) != COMPOSED_NBODY_NOWORLD:
        raise Error(
            "expand identity: the comptime leg sees "
            + String(len(fmd.bodies))
            + " bodies in the composed scene, MuJoCo says "
            + String(COMPOSED_NBODY_NOWORLD)
            + " (excluding world). `ModelDefFromXML.xml_text()` is not"
            + " expanding — this is the P1a regression, and it is SILENT"
            + " everywhere except here."
        )
    print("  OK: the comptime leg sees the whole composed model")

    print("=== PASS ===")
