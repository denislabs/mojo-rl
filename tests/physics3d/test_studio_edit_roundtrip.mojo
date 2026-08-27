"""A fast-path edit must leave the record and the live model AGREEING — S3.

WHY THIS EXISTS
===============
An edit writes to TWO places: the live `Model` (so the sim shows it now) and
the `FlatModelDef` it was built from (so a save, an undo or any structural
edit keeps it). Those are two copies of one fact, and this tree loses more to
two spellings of one quantity drifting apart than to anything else.

Neither half-failure raises:

* **record only** — the sim keeps stepping with the OLD value while the
  inspector shows the new one;
* **`Model` only** — the sim shows the change and the next rebuild silently
  reverts it, so the user's work disappears on undo or on adding a prop.

⇒ **the gate is byte-identity**: rebuild the `Model` from the edited record
and require it to equal the live one, element for element. Same shape as
`test_runtime_model_load` — compare RECORDS, not trajectories.

⚠ NON-VACUITY IS THE WHOLE RISK HERE. A comparison of two models that were
never edited passes trivially, and so does one where the edit changed nothing
observable. So the file asserts, separately, that the edits MOVED the model
away from its loaded state — and it does that by fingerprinting before and
after.

Run: pixi run mojo run -I . tests/physics3d/test_studio_edit_roundtrip.mojo
"""

from mojo_rl.physics3d.fields import Model, DynDims
from mojo_rl.physics3d.gpu.constants import MODEL_GEOM_SIZE
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, read_model_source,
)
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.studio.edit import (
    Edit, EditLog, apply_edit, needs_rebuild,
    TARGET_GEOM, TARGET_BODY,
    F_POS_X, F_POS_Z, F_SIZE_0, F_SIZE_1, F_FRICTION, F_RGBA_R, F_MASS,
)

comptime DT = DType.float64
comptime MODEL = String("mojo_rl/envs/walker2d/assets/walker2d.xml")


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


def edits() -> List[Edit]:
    """A batch that touches each dispatch branch `apply_edit` has.

    ⚠ ONE PER BRANCH ON PURPOSE. `size` alone routes through three different
    packed columns depending on the geom's TYPE, so an edit set that only ever
    moved a sphere would leave the box and capsule paths untested while
    reporting full agreement.
    """
    var e = List[Edit]()
    e.append(Edit(TARGET_GEOM, 1, F_POS_X, 0.123))
    e.append(Edit(TARGET_GEOM, 1, F_POS_Z, -0.4))      # negative, deliberately
    e.append(Edit(TARGET_GEOM, 2, F_SIZE_0, 0.077))    # capsule radius
    e.append(Edit(TARGET_GEOM, 2, F_SIZE_1, 0.31))     # capsule half-length
    e.append(Edit(TARGET_GEOM, 3, F_FRICTION, 1.7))
    e.append(Edit(TARGET_GEOM, 4, F_RGBA_R, 0.25))     # VISUAL only
    return e^


def fingerprint(m: Model[DT, DynDims], ngeom: Int) -> Float64:
    var s = 0.0
    for i in range(ngeom * MODEL_GEOM_SIZE):
        s += Float64(m.geoms.data[i]) * Float64(i + 1)
    return s


def main() raises:
    var t = Tally()
    print("=== a fast-path edit reaches BOTH the record and the model ===")

    var src = read_model_source(MODEL)
    var flat = expand_mjcf(src[0], src[1])
    var fmd = parse_xml_full(flat, src[1])
    var dims = dims_from_flat(fmd, max_contacts=64)
    var live = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, live)
    var ngeom = dims.get_ngeom()
    var before = fingerprint(live, ngeom)

    var log = EditLog()
    for e in edits():
        log.push(e)
        apply_edit(fmd, live, e)

    var after = fingerprint(live, ngeom)
    # ⚠ NON-VACUITY. Two unedited models agree perfectly; require the batch to
    # have MOVED the live model before believing any agreement below.
    t.truth(after != before,
            String("the edits changed the live model (", before, " -> ",
                   after, ")"))

    # ── the gate ──────────────────────────────────────────────────────────
    # Rebuild from the EDITED record. If a field was written to only one side,
    # this is where it shows.
    var rebuilt = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, rebuilt)
    var diff = 0
    var first = -1
    for i in range(ngeom * MODEL_GEOM_SIZE):
        if Float64(live.geoms.data[i]) != Float64(rebuilt.geoms.data[i]):
            diff += 1
            if first < 0:
                first = i
    if diff != 0:
        print("      first differing geom slot:", first, " geom",
              first // MODEL_GEOM_SIZE, " column", first % MODEL_GEOM_SIZE,
              " live", Float64(live.geoms.data[first]),
              " rebuilt", Float64(rebuilt.geoms.data[first]))
    t.truth(diff == 0,
            String("rebuilding from the edited record reproduces the live",
                   " model byte-identically (", ngeom * MODEL_GEOM_SIZE,
                   " slots compared, ", diff, " differ)"))

    # ⚠ A VISUAL-ONLY FIELD MUST **NOT** REACH THE PHYSICS MODEL. `rgba` lives
    # in the record and in `RenderFields`; writing a colour into a packed
    # geom column would silently change a contact parameter, since the slot
    # it would land in belongs to something else.
    t.truth(fmd.geoms[4].rgba_r == 0.25,
            "an rgba edit reached the RECORD (the renderer's source)")

    # ── mass is honestly slow-path ────────────────────────────────────────
    # ⚠ The inertia tensor and dof_invweight0 are DERIVED from mass during the
    # build. A live mass beside stale derived values is a model that steps
    # with one mass and solves with another, so `needs_rebuild` says so
    # instead of the loop pretending to be fast.
    var me = Edit(TARGET_BODY, 2, F_MASS, 7.5)
    t.truth(needs_rebuild(me), "a MASS edit reports that it needs a rebuild")
    t.truth(not needs_rebuild(edits()[0]),
            "a pos edit does NOT (it is genuinely dims-preserving)")
    apply_edit(fmd, live, me)
    t.truth(fmd.bodies[1].mass == 7.5, "the mass edit reached the record")

    # ── undo replays, it does not invert ──────────────────────────────────
    print("--- undo ---")
    t.truth(log.can_undo() and not log.can_redo(),
            "after six edits: undo available, redo not")
    log.undo()
    log.undo()
    t.truth(log.cursor == 4 and log.can_redo(),
            "two undos leave four live edits and a redo tail")

    # Replay onto a FRESH load — the document is file + log, so this is the
    # snapshot the plan asks for, without making `FlatModelDef` copyable.
    var fmd2 = parse_xml_full(flat, src[1])
    var m2 = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd2, dims, m2)
    log.replay(fmd2, m2)
    t.truth(fmd2.geoms[1].pos_x == 0.123,
            "replay reproduces an edit inside the live prefix")
    t.truth(fmd2.geoms[3].friction != 1.7,
            "replay does NOT reproduce an edit past the cursor (the undo"
            " actually undid something)")

    var e2 = Edit(TARGET_GEOM, 5, F_POS_X, 0.9)
    log.push(e2)
    t.truth(not log.can_redo(),
            "a new edit after an undo discards the redo tail")

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error(
            "test_studio_edit_roundtrip: " + String(t.fails) + " failed"
        )
