"""Editing a live model — the two-tier loop, and what keeps it honest — S3.

## The two tiers

* **Fast path — DIMS-PRESERVING edits.** A geom's `pos`/`size`/`rgba`, a
  body's `mass`, a joint's `damping`/`range`. Nothing about the model's SHAPE
  changes, so the edit is written straight into the live `Model` and the sim
  keeps running. Measured cost: a couple of tensor stores.
* **Slow path — STRUCTURAL edits.** Adding or deleting an instance, changing
  a mesh. The scene document is regenerated and everything is rebuilt.
  Measured at 0.2-14 ms per asset (`bench_runtime_load`), which is a click,
  not a drag.

## ⚠⚠ WHAT KEEPS THE TWO REPRESENTATIONS FROM DRIFTING

A fast-path edit writes to the live `Model` AND to the `FlatModelDef` it was
built from. Those are two copies of one fact, and this tree loses more to two
spellings of one quantity than to anything else. So:

> **GATE: rebuilding the `Model` from the edited `FlatModelDef` must reproduce
> the live one BYTE-IDENTICALLY.**

`test_studio_edit_roundtrip` does exactly that after a batch of edits. It is
the cheapest possible guard, and it is the same gate shape as
`test_runtime_model_load` — compare RECORDS, not trajectories.

⚠ A FIELD WRITTEN TO ONLY ONE SIDE IS INVISIBLE UNTIL A REBUILD. The sim
shows the edit (the `Model` has it) and the document does not, so a save, an
undo, or any structural edit silently reverts it. That is why every writer
below touches both, and why the gate exists rather than a comment saying they
should.

## Undo is a REPLAY, not an inverse

The plan calls for snapshotting the document rather than command/inverse
pairs, because command/inverse is where editors accumulate their subtlest
bugs. `FlatModelDef` is `Movable` and not `Copyable`, so the snapshot here is
the EDIT LOG over the loaded file — which IS the document: file + edits.
Undo truncates the log and replays from a fresh parse.

⚠ REPLAY COSTS A PARSE (0.2-14 ms). That is a click. The alternative — making
`FlatModelDef` copyable — is a real change to a hot type and belongs on its
own justification, not smuggled in under undo.
"""

from ..fields import Model, DynDims
from ..parser.flat_model import FlatModelDef
from ..gpu.constants import (
    MODEL_GEOM_SIZE, MODEL_BODY_SIZE, MODEL_JOINT_SIZE,
    GEOM_IDX_POS_X, GEOM_IDX_POS_Y, GEOM_IDX_POS_Z,
    GEOM_IDX_RADIUS, GEOM_IDX_HALF_LENGTH,
    GEOM_IDX_HALF_X, GEOM_IDX_HALF_Y, GEOM_IDX_HALF_Z,
    GEOM_IDX_FRICTION, GEOM_IDX_RBOUND,
)
from ..parser.fields_build import rbound_of

comptime DT = DType.float64

comptime TARGET_GEOM: Int = 0
comptime TARGET_BODY: Int = 1

# Fields a fast-path edit may touch. ⚠ EVERY ONE OF THESE IS DIMS-PRESERVING
# by construction — none changes a count, an address or a type — which is what
# makes writing straight into the live `Model` safe. Adding a field that is
# NOT (a geom's `type`, a mesh id) belongs on the slow path, because the
# containers were sized from the old value.
comptime F_POS_X: Int = 0
comptime F_POS_Y: Int = 1
comptime F_POS_Z: Int = 2
comptime F_SIZE_0: Int = 3
comptime F_SIZE_1: Int = 4
comptime F_SIZE_2: Int = 5
comptime F_RGBA_R: Int = 6
comptime F_RGBA_G: Int = 7
comptime F_RGBA_B: Int = 8
comptime F_RGBA_A: Int = 9
comptime F_FRICTION: Int = 10
comptime F_MASS: Int = 11


def field_name(f: Int) -> String:
    if f == F_POS_X:
        return String("pos[0]")
    if f == F_POS_Y:
        return String("pos[1]")
    if f == F_POS_Z:
        return String("pos[2]")
    if f == F_SIZE_0:
        return String("size[0]")
    if f == F_SIZE_1:
        return String("size[1]")
    if f == F_SIZE_2:
        return String("size[2]")
    if f == F_RGBA_R:
        return String("rgba[0]")
    if f == F_RGBA_G:
        return String("rgba[1]")
    if f == F_RGBA_B:
        return String("rgba[2]")
    if f == F_RGBA_A:
        return String("rgba[3]")
    if f == F_FRICTION:
        return String("friction")
    return String("mass")


@fieldwise_init
struct Edit(Copyable, ImplicitlyCopyable, Movable):
    """One dims-preserving change. The log of these IS the document's delta."""

    var target: Int
    var index: Int
    var field: Int
    var value: Float64


def apply_edit(mut fmd: FlatModelDef, mut m: Model[DT, DynDims], e: Edit):
    """Write ONE edit to BOTH the record and the live model.

    ⚠⚠ BOTH, ALWAYS. Writing only the `Model` shows the change on screen and
    loses it on the next rebuild — a save or an undo silently reverts what the
    user is looking at. Writing only the record leaves the sim showing the old
    value. Neither failure raises, and `test_studio_edit_roundtrip` is what
    catches a field added to one side and not the other.

    ⚠ A GEOM'S SIZE IS PER-TYPE, and the packed columns are not one array:
    box half-extents, capsule radius+half-length and sphere radius live in
    DIFFERENT slots (`build_render_fields`' mapping 4 documents the same
    hazard from the render side). Dispatching on the type here is what keeps
    "size[0]" meaning the same thing in the inspector as in the model.
    """
    if e.target == TARGET_GEOM:
        if e.index < 0 or e.index >= len(fmd.geoms):
            return
        var o = e.index * MODEL_GEOM_SIZE
        if e.field == F_SIZE_0 or e.field == F_SIZE_1 or e.field == F_SIZE_2:
            # ⚠ BEFORE the `ref` below: `_set_size` needs `fmd` itself, and
            # holding a reference into it across the call would alias.
            _set_size(fmd, m, e.index, e.field - F_SIZE_0, e.value)
            return
        ref g = fmd.geoms[e.index]
        if e.field == F_POS_X:
            g.pos_x = e.value
            m.geoms.data[o + GEOM_IDX_POS_X] = Scalar[DT](e.value)
        elif e.field == F_POS_Y:
            g.pos_y = e.value
            m.geoms.data[o + GEOM_IDX_POS_Y] = Scalar[DT](e.value)
        elif e.field == F_POS_Z:
            g.pos_z = e.value
            m.geoms.data[o + GEOM_IDX_POS_Z] = Scalar[DT](e.value)
        elif e.field == F_RGBA_R:
            g.rgba_r = e.value
        elif e.field == F_RGBA_G:
            g.rgba_g = e.value
        elif e.field == F_RGBA_B:
            g.rgba_b = e.value
        elif e.field == F_RGBA_A:
            g.rgba_a = e.value
        elif e.field == F_FRICTION:
            g.friction = e.value
            m.geoms.data[o + GEOM_IDX_FRICTION] = Scalar[DT](e.value)
        # ⚠ RGBA HAS NO `Model` COLUMN — it is VISUAL ONLY, and lives in
        # `RenderFields`. The record is the single source, and the renderer
        # is refreshed by rebuilding `rf`, not by touching `Model`. Writing a
        # colour into a physics tensor is how a "harmless" edit becomes a
        # changed contact parameter.
    elif e.target == TARGET_BODY:
        # Body 0 is the worldbody and is absent from `fmd.bodies`.
        var bi = e.index - 1
        if bi < 0 or bi >= len(fmd.bodies):
            return
        ref b = fmd.bodies[bi]
        if e.field == F_MASS:
            b.mass = e.value
            # ⚠ MASS IS *NOT* WRITTEN TO THE LIVE MODEL, deliberately. The
            # packed body record carries the INERTIA TENSOR and invweight0,
            # both DERIVED from mass by `compute_invweight0` at build time;
            # storing a new mass beside stale derived values gives a model
            # that is internally inconsistent in a way no gate can name. Mass
            # is therefore a SLOW-PATH field — `needs_rebuild` reports it.


def needs_rebuild(e: Edit) -> Bool:
    """Does this edit require the model to be rebuilt to take effect?

    ⚠ THE HONEST ANSWER FOR MASS IS YES, and saying so is the point. It is
    tempting to write mass into the live record and call the loop "fast",
    but the inertia tensor and `dof_invweight0` are derived from it during
    the build; a live mass with stale derived values is a model that steps
    with one mass and solves with another.
    """
    return e.target == TARGET_BODY and e.field == F_MASS


def _set_size(
    mut fmd: FlatModelDef, mut m: Model[DT, DynDims], gi: Int, slot: Int,
    v: Float64,
):
    """`size[slot]` for this geom's TYPE — see `apply_edit`'s note."""
    var o = gi * MODEL_GEOM_SIZE
    ref g = fmd.geoms[gi]
    var gt = g.geom_type
    if gt == 3:  # BOX
        if slot == 0:
            g.half_x = v
            m.geoms.data[o + GEOM_IDX_HALF_X] = Scalar[DT](v)
        elif slot == 1:
            g.half_y = v
            m.geoms.data[o + GEOM_IDX_HALF_Y] = Scalar[DT](v)
        else:
            g.half_z = v
            m.geoms.data[o + GEOM_IDX_HALF_Z] = Scalar[DT](v)
    elif gt == 2 or gt == 4:  # CAPSULE / CYLINDER
        if slot == 0:
            g.radius = v
            m.geoms.data[o + GEOM_IDX_RADIUS] = Scalar[DT](v)
        elif slot == 1:
            g.half_length = v
            m.geoms.data[o + GEOM_IDX_HALF_LENGTH] = Scalar[DT](v)
    else:  # SPHERE and everything bounded by a radius
        if slot == 0:
            g.radius = v
            m.geoms.data[o + GEOM_IDX_RADIUS] = Scalar[DT](v)

    # ⚠⚠ `rbound` IS DERIVED FROM THE SIZE AND MUST FOLLOW IT. The first run
    # of `test_studio_edit_roundtrip` failed here: resizing a capsule left the
    # broadphase bound at its OLD value, so the pair is rejected before the
    # narrow phase ever runs — MISSED CONTACTS, with nothing to see. It is the
    # same shape as mass/inertia, and the difference is that `rbound` is a
    # pure function of type and size, so it can be recomputed here instead of
    # forcing a rebuild.
    #
    # ⚠ VIA `rbound_of`, THE BUILDER'S OWN FUNCTION. Re-deriving the five
    # per-type formulas here would be a second spelling of one quantity, and
    # a divergence between them would be invisible for the same reason the
    # staleness was.
    #
    # ⚠ A MESH KEEPS ITS MEASURED BOUND. The builder overwrites `rbound` from
    # the loaded hull, and no size slot the studio can edit affects it.
    if gt != 5:
        m.geoms.data[o + GEOM_IDX_RBOUND] = Scalar[DT](
            rbound_of(gt, g.radius, g.half_length, g.half_x, g.half_y,
                      g.half_z)
        )


struct EditLog(Movable):
    """The document's delta over the loaded file. Undo TRUNCATES and replays.

    ⚠ NOT COMMAND/INVERSE. An inverse has to reconstruct the previous value,
    and every editor that stores one eventually stores a stale one — the
    subtlest class of undo bug, and the reason the plan chose snapshots. The
    log plus a fresh parse IS the snapshot; replay is deterministic because
    every `Edit` is an absolute value, never a delta.
    """

    var edits: List[Edit]
    var cursor: Int
    """How many edits are LIVE. Redo is possible while it is below `len`."""

    def __init__(out self):
        self.edits = List[Edit]()
        self.cursor = 0

    def push(mut self, e: Edit):
        # A new edit after an undo discards the redo tail, as every editor does.
        var keep = List[Edit]()
        for i in range(self.cursor):
            keep.append(self.edits[i])
        keep.append(e)
        self.edits = keep^
        self.cursor = len(self.edits)

    def can_undo(self) -> Bool:
        return self.cursor > 0

    def can_redo(self) -> Bool:
        return self.cursor < len(self.edits)

    def undo(mut self):
        if self.cursor > 0:
            self.cursor -= 1

    def redo(mut self):
        if self.cursor < len(self.edits):
            self.cursor += 1

    def replay(self, mut fmd: FlatModelDef, mut m: Model[DT, DynDims]):
        """Re-apply the live prefix onto a freshly loaded model."""
        for i in range(self.cursor):
            apply_edit(fmd, m, self.edits[i])
