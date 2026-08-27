"""A fast-path edit reaches the DOCUMENT too — the third copy — V2.4.

WHY THIS EXISTS
===============
`test_studio_edit_roundtrip` gates two copies of one fact: the live `Model` and
the `FlatModelDef`. There is a THIRD, and until V2.3 nothing could see it —
the studio's document (`Loaded.flat`), written once at load and never again.

That was invisible while the only way out was `writer.to_mjcf`, which
REGENERATES the file from the record. It became a silent loss of work the
moment `File > Save edited model` started writing the document verbatim:

    drag a geom's size  ->  Model: new.  record: new.  document: OLD.
    save                ->  the file has the size it had when it was opened.

Nothing raises. The sim shows the edit, the inspector shows the edit, and the
saved file does not have it.

THE GATE IS PLAN §4's, one copy further out:

> rebuilding the model from the re-parsed DOCUMENT must reproduce the live one

⚠ AND THE NEGATIVE CONTROL IS WHAT MAKES IT MEAN ANYTHING: re-parsing the
ORIGINAL document must NOT reproduce the live model. Without that, a gate
comparing two models that were never really edited passes trivially — which is
exactly the state this file was written to detect.

⚠ THE MuJoCo HALF. Our writer and our parser could agree on a wrong spelling
and cancel out perfectly. The edited document is written to
/tmp/physics3d_structural with the values we believe we wrote, and
`scripts/check_structural_edits_vs_mujoco.py` asks MuJoCo what it reads.

Run: pixi run mojo run -I . tests/physics3d/test_edit_reaches_the_document.mojo
     pixi run python scripts/check_structural_edits_vs_mujoco.py
"""

from mojo_rl.physics3d.fields import Model, DynDims
from mojo_rl.physics3d.gpu.constants import (
    MODEL_GEOM_SIZE, MODEL_BODY_SIZE, BODY_IDX_MASS,
)
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.flat_model import FlatModelDef
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime,
)
from mojo_rl.physics3d.studio.edit import (
    Edit, apply_edit, apply_edit_to_document,
    TARGET_GEOM, TARGET_BODY,
    F_POS_X, F_POS_Z, F_SIZE_0, F_SIZE_1, F_FRICTION, F_RGBA_R, F_MASS,
)


comptime DT = DType.float64
comptime MODEL = String("mojo_rl/envs/walker2d/assets/walker2d.xml")
comptime BASE = String("mojo_rl/envs/walker2d/assets")
comptime OUT = String("/tmp/physics3d_structural/doc_edit.xml")
comptime EXPECT = String("/tmp/physics3d_structural/doc_edit_expect.txt")


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


def _read(p: String) raises -> String:
    var f = open(p, "r")
    var s = f.read()
    f.close()
    return s^


def _geom_diff(a: Model[DT, DynDims], b: Model[DT, DynDims], n: Int) -> Int:
    """How many geom scalars differ. 0 means byte-identical."""
    var d = 0
    for i in range(n * MODEL_GEOM_SIZE):
        if Float64(a.geoms.data[i]) != Float64(b.geoms.data[i]):
            d += 1
    return d


def _mass_diff(
    a: Model[DT, DynDims], b: Model[DT, DynDims], nbody: Int
) -> Int:
    var d = 0
    for i in range(nbody):
        if (Float64(a.bodies.data[i * MODEL_BODY_SIZE + BODY_IDX_MASS])
                != Float64(b.bodies.data[i * MODEL_BODY_SIZE + BODY_IDX_MASS])):
            d += 1
    return d


def main() raises:
    var t = Tally()
    print("=== a fast-path edit reaches the document ===")

    var src = expand_mjcf(_read(MODEL), BASE)
    var fmd = parse_xml_full(src, BASE)
    var dims = dims_from_flat(fmd)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)

    # ⚠ THE FIXTURE NAMES EVERY GEOM, so this pass exercises only the by-NAME
    # half of the locator. `swimmer.xml` (3 of 4 geoms unnamed) is run below
    # for the POSITIONAL half, which is the harder one and the one most models
    # in the wild need.

    # ── a batch touching every writer ─────────────────────────────────────
    # ⚠ A NAMED GEOM AND AN UNNAMED ONE. The locator has two halves — by name
    # where there is one, by "the k-th geom of body X" where there is not —
    # and walker2d has both. Exercising only the named half would leave the
    # positional key, which is the harder one, untested.
    var doc = src
    var edits = List[Edit]()
    edits.append(Edit(TARGET_GEOM, 1, F_SIZE_0, 0.077))
    edits.append(Edit(TARGET_GEOM, 1, F_POS_Z, -0.31))
    edits.append(Edit(TARGET_GEOM, 2, F_SIZE_1, 0.23))
    edits.append(Edit(TARGET_GEOM, 3, F_FRICTION, 1.37))
    edits.append(Edit(TARGET_GEOM, 4, F_RGBA_R, 0.42))
    edits.append(Edit(TARGET_GEOM, 0, F_POS_X, 0.011))
    edits.append(Edit(TARGET_BODY, 2, F_MASS, 4.25))

    var named = 0
    for gi in range(len(fmd.geoms)):
        if fmd.geom_names[gi].byte_length() > 0:
            named += 1
    print("    geoms:", len(fmd.geoms), " named:", named,
          " unnamed:", len(fmd.geoms) - named)
    t.truth(named == len(fmd.geoms),
            "walker2d names every geom — the by-NAME half of the locator")

    for e in edits:
        apply_edit(fmd, m, e)
        doc = apply_edit_to_document(fmd, m, doc, e)
    # A mass edit changes derived inertia, so the live model is rebuilt from
    # the record — the same thing the studio does for `needs_rebuild`.
    build_model_runtime[DT](fmd, dims, m)

    # ── the negative control, FIRST ───────────────────────────────────────
    # ⚠ WITHOUT THIS the comparison below could be true because nothing moved.
    print("--- the original document does NOT reproduce the live model ---")
    var fmd0 = parse_xml_full(src, BASE)
    var dims0 = dims_from_flat(fmd0)
    var m0 = Model[DT, DynDims](dims0)
    build_model_runtime[DT](fmd0, dims0, m0)
    var moved = _geom_diff(m, m0, dims.get_ngeom())
    t.truth(moved > 0,
            String("the edits moved ", moved, " geom scalar(s) away from the"
                   " file as loaded"))
    t.truth(_mass_diff(m, m0, dims.get_nbody()) > 0,
            "and at least one body mass")

    # ── the gate ──────────────────────────────────────────────────────────
    print("--- the EDITED document rebuilds the live model ---")
    var fmd2 = parse_xml_full(doc, BASE)
    var dims2 = dims_from_flat(fmd2)
    t.truth(dims2.get_ngeom() == dims.get_ngeom()
            and dims2.get_nbody() == dims.get_nbody(),
            "a numeric edit is structurally the identity")
    var m2 = Model[DT, DynDims](dims2)
    build_model_runtime[DT](fmd2, dims2, m2)
    var d = _geom_diff(m, m2, dims.get_ngeom())
    t.truth(d == 0,
            String("every geom scalar is IDENTICAL after the round trip (",
                   d, " differ)"))
    for i in range(dims.get_nbody()):
        var a = Float64(m.bodies.data[i * MODEL_BODY_SIZE + BODY_IDX_MASS])
        var b2 = Float64(m2.bodies.data[i * MODEL_BODY_SIZE + BODY_IDX_MASS])
        if a != b2:
            print("       body", i, fmd.body_names[i], "live", a, "doc", b2)
    var dm = _mass_diff(m, m2, dims.get_nbody())
    t.truth(dm == 0,
            String("every body mass is identical too (", dm, " differ)"))

    # ── the POSITIONAL locator, on a model that names almost nothing ──────
    # ⚠⚠ MOST GEOMS IN THE WILD HAVE NO NAME, and there is nothing to look
    # one up by. The key is "the k-th geom of body X", read from the same
    # body-grouped order `FlatModelDef` stores — and naming the element to
    # make it addressable would be an edit the user never asked for, in a file
    # they may be reading.
    print("--- an UNNAMED geom, located by position ---")
    var sw_base = String("mojo_rl/envs/swimmer/assets")
    var sw_src = expand_mjcf(
        _read(String("mojo_rl/envs/swimmer/assets/swimmer.xml")), sw_base
    )
    var sw = parse_xml_full(sw_src, sw_base)
    var sw_named = 0
    for gi in range(len(sw.geoms)):
        if sw.geom_names[gi].byte_length() > 0:
            sw_named += 1
    t.truth(sw_named < len(sw.geoms),
            String("swimmer leaves ", len(sw.geoms) - sw_named,
                   " geom(s) unnamed (that is the point of this arm)"))
    var sw_dims = dims_from_flat(sw)
    var sw_m = Model[DT, DynDims](sw_dims)
    build_model_runtime[DT](sw, sw_dims, sw_m)
    var sw_doc = sw_src
    # Pick an UNNAMED geom, so the by-name half cannot answer.
    var target = -1
    for gi in range(len(sw.geoms)):
        if sw.geom_names[gi].byte_length() == 0:
            target = gi
            break
    t.truth(target >= 0, String("editing unnamed geom ", target))
    var se = Edit(TARGET_GEOM, target, F_SIZE_0, 0.133)
    apply_edit(sw, sw_m, se)
    sw_doc = apply_edit_to_document(sw, sw_m, sw_doc, se)
    var sw2 = parse_xml_full(sw_doc, sw_base)
    var sw_dims2 = dims_from_flat(sw2)
    var sw_m2 = Model[DT, DynDims](sw_dims2)
    build_model_runtime[DT](sw2, sw_dims2, sw_m2)
    t.truth(_geom_diff(sw_m, sw_m2, sw_dims.get_ngeom()) == 0,
            "the unnamed geom's edit round-trips through the document")
    # ⚠ AND IT LANDED ON THE RIGHT ONE. A locator that always returned the
    # FIRST geom would round-trip perfectly and edit the wrong element.
    t.truth(abs(Float64(sw2.geoms[target].radius) - 0.133) < 1e-12,
            String("and on geom ", target, " specifically (radius ",
                   sw2.geoms[target].radius, ")"))

    # ── hand it to MuJoCo ─────────────────────────────────────────────────
    # ⚠ OUR WRITER AND OUR PARSER COULD AGREE ON A WRONG SPELLING. Only the
    # reference can see that, and it cannot be called from here.
    var wf = open(OUT, "w")
    wf.write(doc)
    wf.close()
    # geom index 1's size[0] and pos[2], and body 2's mass, as WE believe we
    # wrote them.
    var ef = open(EXPECT, "w")
    ef.write(
        String(fmd.geom_names[1]) + " " + String(fmd.geoms[1].radius) + " "
        + String(fmd.geoms[1].pos_z) + " "
        + String(fmd.body_names[2]) + " "
        # ⚠ THE **BUILT** MASS, NOT THE RECORD'S. On an inertiafromgeom model
        # the body's mass is derived from its geoms and `BodyData.mass` is
        # whatever was last written there; asking MuJoCo about a number our
        # own simulator does not use would gate nothing.
        + String(Float64(m.bodies.data[2 * MODEL_BODY_SIZE + BODY_IDX_MASS]))
        + "\n"
    )
    ef.close()
    t.truth(fmd.geom_names[1].byte_length() > 0,
            "the MuJoCo-checked geom is a NAMED one (it has to be findable)")

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    print("    wrote", OUT)
    print("    pixi run python scripts/check_structural_edits_vs_mujoco.py")
    if t.fails != 0:
        raise Error(
            "test_edit_reaches_the_document: " + String(t.fails) + " failed"
        )
