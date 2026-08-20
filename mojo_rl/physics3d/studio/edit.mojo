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
    GEOM_IDX_QUAT_W, GEOM_IDX_QUAT_X, GEOM_IDX_QUAT_Y, GEOM_IDX_QUAT_Z,
    GEOM_IDX_RADIUS, GEOM_IDX_HALF_LENGTH,
    GEOM_IDX_HALF_X, GEOM_IDX_HALF_Y, GEOM_IDX_HALF_Z,
    GEOM_IDX_FRICTION, GEOM_IDX_RBOUND,
)
from ..parser.fields_build import rbound_of
from ..model.inertia_from_geom import geom_volume
from ..parser.flat_model import GeomData
from ..parser.expander import element_end, _find_tag
from ..gpu.constants import (
    BODY_IDX_IXX, BODY_IDX_IYY, BODY_IDX_IZZ, BODY_IDX_MASS,
    BODY_IDX_IPOS_X, BODY_IDX_IPOS_Y, BODY_IDX_IPOS_Z,
    BODY_IDX_IQUAT_X, BODY_IDX_IQUAT_Y, BODY_IDX_IQUAT_Z, BODY_IDX_IQUAT_W,
    BODY_IDX_POS_X, BODY_IDX_POS_Y, BODY_IDX_POS_Z,
    BODY_IDX_QUAT_W, BODY_IDX_QUAT_X, BODY_IDX_QUAT_Y, BODY_IDX_QUAT_Z,
)
from .structure import set_attribute_at, find_child, find_named

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
# ⚠ ORIENTATION — ADDED FOR THE GIZMO (V2.10). Four components, not three:
# an Euler triple would have to pick a convention, and MJCF has FIVE
# spellings for the same rotation (`quat`, `euler`, `axisangle`, `xyaxes`,
# `zaxis`). The record carries a quaternion, so that is what an edit carries;
# `apply_edit_to_document` is where the spelling question is answered, and it
# answers it by writing `quat` and REMOVING the other four.
comptime F_QUAT_W: Int = 12
comptime F_QUAT_X: Int = 13
comptime F_QUAT_Y: Int = 14
comptime F_QUAT_Z: Int = 15


def is_pos_field(f: Int) -> Bool:
    return f == F_POS_X or f == F_POS_Y or f == F_POS_Z


def is_quat_field(f: Int) -> Bool:
    return f == F_QUAT_W or f == F_QUAT_X or f == F_QUAT_Y or f == F_QUAT_Z


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
    if f == F_QUAT_W:
        return String("quat[0]")
    if f == F_QUAT_X:
        return String("quat[1]")
    if f == F_QUAT_Y:
        return String("quat[2]")
    if f == F_QUAT_Z:
        return String("quat[3]")
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
        elif e.field == F_QUAT_W:
            g.quat_w = e.value
            m.geoms.data[o + GEOM_IDX_QUAT_W] = Scalar[DT](e.value)
        elif e.field == F_QUAT_X:
            g.quat_x = e.value
            m.geoms.data[o + GEOM_IDX_QUAT_X] = Scalar[DT](e.value)
        elif e.field == F_QUAT_Y:
            g.quat_y = e.value
            m.geoms.data[o + GEOM_IDX_QUAT_Y] = Scalar[DT](e.value)
        elif e.field == F_QUAT_Z:
            g.quat_z = e.value
            m.geoms.data[o + GEOM_IDX_QUAT_Z] = Scalar[DT](e.value)
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
        # ⚠⚠ A BODY'S OWN FRAME — NEW IN V2.10, AND IT HAD NO PATH AT ALL.
        # `F_POS_*` existed and only geoms honoured it, so a body pos edit
        # was accepted and silently dropped. It is live because forward
        # kinematics reads these columns every step; it is nonetheless a
        # SLOW-PATH field (see `needs_rebuild`), because `dof_invweight0` and
        # a free joint's `qpos0` are DERIVED from the body frame at build
        # time and go stale the moment it moves.
        var bo = e.index * MODEL_BODY_SIZE
        if e.field == F_POS_X:
            b.pos_x = e.value
            m.bodies.data[bo + BODY_IDX_POS_X] = Scalar[DT](e.value)
            return
        if e.field == F_POS_Y:
            b.pos_y = e.value
            m.bodies.data[bo + BODY_IDX_POS_Y] = Scalar[DT](e.value)
            return
        if e.field == F_POS_Z:
            b.pos_z = e.value
            m.bodies.data[bo + BODY_IDX_POS_Z] = Scalar[DT](e.value)
            return
        if e.field == F_QUAT_W:
            b.quat_w = e.value
            m.bodies.data[bo + BODY_IDX_QUAT_W] = Scalar[DT](e.value)
            return
        if e.field == F_QUAT_X:
            b.quat_x = e.value
            m.bodies.data[bo + BODY_IDX_QUAT_X] = Scalar[DT](e.value)
            return
        if e.field == F_QUAT_Y:
            b.quat_y = e.value
            m.bodies.data[bo + BODY_IDX_QUAT_Y] = Scalar[DT](e.value)
            return
        if e.field == F_QUAT_Z:
            b.quat_z = e.value
            m.bodies.data[bo + BODY_IDX_QUAT_Z] = Scalar[DT](e.value)
            return
        if e.field == F_MASS:
            # ⚠⚠ SETTING A MASS MUST ALSO MAKE THE INERTIAL EXPLICIT, and
            # until now it did not — so the slider DID NOTHING on any body
            # without an `<inertial>`, which is most of them. Measured on
            # walker2d: record 1.0 -> 4.25, built model 4.0578 both before and
            # after. `inertiafromgeom` defaults to "auto" (derive unless the
            # body says otherwise), so the rebuild re-derived from the geoms
            # and threw the edit away, silently, every time.
            #
            # MJCF has no `mass=` on `<body>`; the only spelling for an
            # override IS an explicit inertial frame. So the edit materialises
            # one, taking the pose and the principal axes from the BUILT model
            # — which is where the derivation ran — rather than from the
            # record, whose defaults were never the real values.
            #
            # ⚠ AND THE INERTIA IS SCALED BY THE MASS RATIO. Freezing the old
            # tensor beside a new mass describes a body of the same shape with
            # a different density in one place and not the other; scaling is
            # what `<compiler settotalmass>` does for the same reason.
            var o = e.index * MODEL_BODY_SIZE
            var old_mass = Float64(m.bodies.data[o + BODY_IDX_MASS])

            # ⚠⚠ AND ON A `inertiafromgeom="true"` MODEL AN `<inertial>` IS
            # IGNORED — by MuJoCo as well. That mode means "always derive from
            # the geoms", so the ONLY expressible override is on the geoms
            # themselves. walker2d, humanoid and ant all set it, so this is
            # not the rare branch. Scaling every contributing geom by the same
            # ratio moves the body's mass to the target and leaves its
            # inertia consistent with its shape, which is what the derivation
            # would have produced for a denser body.
            if fmd.inertiafromgeom == 1:
                # ⚠⚠ THE RATIO COMES FROM THE **RECORD**, NOT THE LIVE MODEL.
                # `m`'s body mass is only refreshed by a rebuild, so after a
                # size edit it is one edit stale — and the mass the user typed
                # then landed off by the volume change. Measured: asked for
                # 4.25 on walker2d's thigh after resizing it, got 4.332, and
                # MuJoCo agreed with the 4.332. Summing the geoms is what the
                # builder will do, so it is the number to divide by.
                #
                # ⚠ AND WITH `inertiagrouprange`, because the builder filters
                # by it: a geom outside the range contributes nothing, and
                # scaling it would move a mass nobody is looking at.
                var sum_mass = 0.0
                for gi in range(len(fmd.geoms)):
                    if fmd.geoms[gi].body_id != e.index:
                        continue
                    if fmd.geoms[gi].group < fmd.inertiagrouprange_min \
                            or fmd.geoms[gi].group > fmd.inertiagrouprange_max:
                        continue
                    if fmd.geoms[gi].mass > 0.0:
                        sum_mass += fmd.geoms[gi].mass
                var gr = (e.value / sum_mass) if sum_mass > 1e-15 else 1.0
                for gi in range(len(fmd.geoms)):
                    if fmd.geoms[gi].body_id != e.index:
                        continue
                    if fmd.geoms[gi].group < fmd.inertiagrouprange_min \
                            or fmd.geoms[gi].group > fmd.inertiagrouprange_max:
                        continue
                    # ⚠⚠ THE MASS IS RE-DERIVED FROM THE SCALED DENSITY, not
                    # scaled alongside it. The document can only carry ONE of
                    # the two — `density=` — so a re-parse recomputes
                    # density x volume, and multiplying `mass` separately left
                    # the record 1 ULP from what any reload produces
                    # (4.25 vs 4.249999999999999). Deriving it the way the
                    # parser will is what makes the round trip exact.
                    #
                    # ⚠ AN EXPLICIT `mass=` IS NOT DERIVED and is scaled
                    # directly — the document carries that one verbatim.
                    if fmd.geoms[gi].has_explicit_mass:
                        fmd.geoms[gi].mass *= gr
                        fmd.geoms[gi].density *= gr
                        continue
                    fmd.geoms[gi].density *= gr
                    var v = Float64(geom_volume[DT](
                        fmd.geoms[gi].geom_type, fmd.geoms[gi].radius,
                        fmd.geoms[gi].half_length, fmd.geoms[gi].half_x,
                        fmd.geoms[gi].half_y, fmd.geoms[gi].half_z,
                    ))
                    if v > 0.0:
                        fmd.geoms[gi].mass = fmd.geoms[gi].density * v
                b.mass = e.value
                return
            var ratio = (e.value / old_mass) if old_mass > 1e-15 else 1.0
            b.ipos_x = Float64(m.bodies.data[o + BODY_IDX_IPOS_X])
            b.ipos_y = Float64(m.bodies.data[o + BODY_IDX_IPOS_Y])
            b.ipos_z = Float64(m.bodies.data[o + BODY_IDX_IPOS_Z])
            b.iquat_x = Float64(m.bodies.data[o + BODY_IDX_IQUAT_X])
            b.iquat_y = Float64(m.bodies.data[o + BODY_IDX_IQUAT_Y])
            b.iquat_z = Float64(m.bodies.data[o + BODY_IDX_IQUAT_Z])
            b.iquat_w = Float64(m.bodies.data[o + BODY_IDX_IQUAT_W])
            b.ixx = Float64(m.bodies.data[o + BODY_IDX_IXX]) * ratio
            b.iyy = Float64(m.bodies.data[o + BODY_IDX_IYY]) * ratio
            b.izz = Float64(m.bodies.data[o + BODY_IDX_IZZ]) * ratio
            b.has_explicit_inertia = True
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

    # ⚠⚠ AND THE MASS FOLLOWS THE SHAPE. `gd.mass` is resolved AT PARSE TIME
    # as density x volume, so resizing a geom left the old volume's mass in
    # the record — the sim kept a thigh's original mass however thin it was
    # made, and a rebuild kept it too, because the rebuild reads `gd.mass`.
    # Re-parsing the same file gave a DIFFERENT model, which is how the
    # document round-trip gate found this.
    #
    # ⚠ ONLY WHEN THE SOURCE DID NOT WRITE `mass=`. An explicit mass is an
    # override of exactly this derivation and must survive a resize, the same
    # way MuJoCo's compiler leaves it alone.
    if not g.has_explicit_mass:
        var vol = Float64(geom_volume[DT](
            g.geom_type, g.radius, g.half_length, g.half_x, g.half_y, g.half_z
        ))
        if vol > 0.0:
            g.mass = g.density * vol



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

    def replay_all(
        self, mut fmd: FlatModelDef, mut m: Model[DT, DynDims], xml: String
    ) raises -> String:
        """Replay onto the model AND the document, in step.

        ⚠ INTERLEAVED, NOT TWO PASSES. `apply_edit_to_document` reads the
        record the edit just produced — a mass edit, in particular, writes an
        `<inertial>` (or a scaled density) computed from it. Running every
        model edit first and then every document edit would write each one
        from the FINAL record rather than from the one it belonged to.
        """
        var doc = xml
        for i in range(self.cursor):
            apply_edit(fmd, m, self.edits[i])
            doc = apply_edit_to_document(fmd, m, doc, self.edits[i])
        return doc^


# =============================================================================
# The THIRD copy — the document
# =============================================================================


def _n(v: Float64) -> String:
    """FULL precision. The document IS the model; a rounded number is an edit."""
    return String(v)


def _geom_size_attr(g: GeomData) -> String:
    """MJCF `size=` for this geom's TYPE, with the component count it needs.

    ⚠ THE COUNT IS PER TYPE AND MuJoCo ENFORCES IT — a sphere with three
    numbers is a load error. `writer._geom_size` is the other place that knows
    this, and `scene.Prop.size_attr` is the third; each names the others.
    """
    var t = g.geom_type
    if t == 0:
        return _n(g.half_x) + " " + _n(g.half_y) + " " + _n(g.half_z)
    if t == 1:
        return _n(g.radius)
    if t == 2 or t == 4:
        return _n(g.radius) + " " + _n(g.half_length)
    if t == 3 or t == 6:
        return _n(g.half_x) + " " + _n(g.half_y) + " " + _n(g.half_z)
    return String("")


def _geom_ordinal(fmd: FlatModelDef, gi: Int) -> Int:
    """How many geoms of the same body come before `gi`."""
    var n = 0
    for k in range(gi):
        if fmd.geoms[k].body_id == fmd.geoms[gi].body_id:
            n += 1
    return n


def _write_quat_at(
    xml: String, at: Int, w: Float64, x: Float64, y: Float64, z: Float64
) -> String:
    """Write `quat=` on the open tag at `at` and REMOVE every rival spelling.

    ⚠⚠ MJCF HAS FIVE SPELLINGS FOR ONE ROTATION — `quat`, `euler`,
    `axisangle`, `xyaxes`, `zaxis` — AND MuJoCo REFUSES A TAG CARRYING TWO.
    So writing `quat` beside an existing `euler` does not produce a model
    with the new rotation; it produces a file that will not load, from a
    studio that shows the new rotation happily. Dropping the rivals is not
    tidying, it is what makes the edit expressible.

    ⚠ AND `euler` IS UNIT-DEPENDENT: `<compiler angle="degree">` is MJCF's
    DEFAULT, so the same triple means two different rotations in two files.
    `quat` is the one spelling with no such dependency, which is why it is
    the one we write rather than the one the file happened to use.
    """
    var out = set_attribute_at(
        xml, at, String("quat"),
        _n(w) + " " + _n(x) + " " + _n(y) + " " + _n(z),
    )
    out = _drop_attribute_at(out, at, String("euler"))
    out = _drop_attribute_at(out, at, String("axisangle"))
    out = _drop_attribute_at(out, at, String("xyaxes"))
    out = _drop_attribute_at(out, at, String("zaxis"))
    return out^


def apply_edit_to_document(
    fmd: FlatModelDef, m: Model[DT, DynDims], xml: String, e: Edit
) raises -> String:
    """Write the same edit into the DOCUMENT, so all three copies agree.

    ⚠⚠ THIS IS THE THIRD COPY, AND IT WAS THE ONE LOSING WORK. `apply_edit`
    keeps the live `Model` and the `FlatModelDef` in step; the studio's
    document (`Loaded.flat`) was written once at load and never again. That
    was invisible while the only way out was `writer.to_mjcf`, which
    regenerates from the record — and became a silent loss the moment "Save
    edited model" started writing the document verbatim: drag a geom's size,
    save, and the size is the one the file had when it was opened.

    ⚠ THE ELEMENT IS FOUND BY NAME WHERE IT HAS ONE AND BY POSITION WHERE IT
    DOES NOT — "the third geom of body `thigh`". Most geoms in this tree are
    unnamed, and naming one to make it addressable would be an edit the user
    did not ask for, in a file they may be reading.
    """
    if e.target == TARGET_GEOM:
        if e.index < 0 or e.index >= len(fmd.geoms):
            return xml
        ref g = fmd.geoms[e.index]
        var at = -1
        if e.index < len(fmd.geom_names) \
                and fmd.geom_names[e.index].byte_length() > 0:
            at = find_named(xml, String("geom"), fmd.geom_names[e.index])
        if at == -1:
            var parent = String("")
            if g.body_id > 0 and g.body_id < len(fmd.body_names):
                parent = fmd.body_names[g.body_id]
            at = find_child(xml, parent, String("geom"),
                            _geom_ordinal(fmd, e.index))
        if at == -1:
            raise Error(
                "cannot locate this geom in the document — it has no name and"
                " no body to count within, so the edit cannot be saved"
            )
        if e.field == F_POS_X or e.field == F_POS_Y or e.field == F_POS_Z:
            return _materialise_fromto(
                set_attribute_at(
                    xml, at, String("pos"),
                    _n(g.pos_x) + " " + _n(g.pos_y) + " " + _n(g.pos_z),
                ),
                at, g,
            )
        if is_quat_field(e.field):
            # ⚠ `_materialise_fromto` SECOND, and it writes the quat AGAIN
            # from the record — deliberately. A `fromto` capsule carries its
            # own orientation and overrides anything written beside it, so
            # the rotation only survives once `fromto` is gone.
            return _materialise_fromto(
                _write_quat_at(xml, at, g.quat_w, g.quat_x, g.quat_y,
                               g.quat_z),
                at, g,
            )
        if e.field == F_SIZE_0 or e.field == F_SIZE_1 or e.field == F_SIZE_2:
            return _materialise_fromto(
                set_attribute_at(xml, at, String("size"), _geom_size_attr(g)),
                at, g,
            )
        if e.field == F_RGBA_R or e.field == F_RGBA_G \
                or e.field == F_RGBA_B or e.field == F_RGBA_A:
            return set_attribute_at(
                xml, at, String("rgba"),
                _n(g.rgba_r) + " " + _n(g.rgba_g) + " " + _n(g.rgba_b) + " "
                + _n(g.rgba_a),
            )
        if e.field == F_FRICTION:
            return set_attribute_at(
                xml, at, String("friction"),
                _n(g.friction) + " " + _n(g.friction_spin) + " "
                + _n(g.friction_roll),
            )
        return xml

    if e.target == TARGET_BODY \
            and (is_pos_field(e.field) or is_quat_field(e.field)):
        var bpi = e.index - 1
        if bpi < 0 or bpi >= len(fmd.bodies):
            return xml
        ref bd = fmd.bodies[bpi]
        # ⚠ BY NAME, AND ONLY BY NAME. A geom can be found positionally
        # ("the third geom of body `thigh`") because its parent is named;
        # a body's own position in the document is a path through the tree,
        # and `find_child` counts SIBLINGS, which is not the same thing on a
        # tree three deep. An unnamed body is refused rather than guessed at.
        var bname = fmd.body_names[e.index] if e.index < len(fmd.body_names) \
            else String("")
        if bname.byte_length() == 0:
            raise Error(
                "cannot save a transform edit on an unnamed body — there is"
                " nothing in the document to find it by"
            )
        var bat = find_named(xml, String("body"), bname)
        if bat == -1:
            raise Error("body '" + bname + "' is not in the document")
        if is_quat_field(e.field):
            return _write_quat_at(xml, bat, bd.quat_w, bd.quat_x, bd.quat_y,
                                  bd.quat_z)
        return set_attribute_at(
            xml, bat, String("pos"),
            _n(bd.pos_x) + " " + _n(bd.pos_y) + " " + _n(bd.pos_z),
        )

    if e.target == TARGET_BODY and e.field == F_MASS:
        var bi = e.index - 1
        if bi < 0 or bi >= len(fmd.bodies):
            return xml

        # ⚠ THE SAME BRANCH `apply_edit` TOOK. On an `inertiafromgeom="true"`
        # model the override lives on the GEOMS, so that is what the document
        # has to say — writing an `<inertial>` there would produce a file
        # MuJoCo reads with the ORIGINAL mass while the studio shows the new
        # one, which is precisely the drift this function exists to close.
        if fmd.inertiafromgeom == 1:
            var out_g = xml
            for gi in range(len(fmd.geoms)):
                if fmd.geoms[gi].body_id != e.index:
                    continue
                var gat = -1
                if gi < len(fmd.geom_names) \
                        and fmd.geom_names[gi].byte_length() > 0:
                    gat = find_named(out_g, String("geom"), fmd.geom_names[gi])
                if gat == -1:
                    var pn = String("")
                    if e.index < len(fmd.body_names):
                        pn = fmd.body_names[e.index]
                    gat = find_child(out_g, pn, String("geom"),
                                     _geom_ordinal(fmd, gi))
                if gat == -1:
                    raise Error(
                        "cannot locate a geom of this body in the document,"
                        " so the mass edit cannot be saved"
                    )
                # ⚠ WHICH ATTRIBUTE DEPENDS ON WHICH THE FILE USED. An
                # existing `mass=` silently overrides `density=`, so writing
                # the density on such a geom would change nothing and say
                # nothing.
                if fmd.geoms[gi].has_explicit_mass:
                    out_g = set_attribute_at(out_g, gat, String("mass"),
                                             _n(fmd.geoms[gi].mass))
                else:
                    out_g = set_attribute_at(out_g, gat, String("density"),
                                             _n(fmd.geoms[gi].density))
            return out_g^
        var name = fmd.body_names[e.index] if e.index < len(fmd.body_names) \
            else String("")
        if name.byte_length() == 0:
            raise Error(
                "cannot save a mass edit on an unnamed body — <inertial> has"
                " to be written INTO that body, and there is nothing to find"
                " it by"
            )
        var at = find_named(xml, String("body"), name)
        if at == -1:
            raise Error("body '" + name + "' is not in the document")
        # ⚠⚠ A MASS EDIT MATERIALISES AN `<inertial>`, AND THAT IS A REAL
        # SEMANTIC CHANGE, not a serialisation detail. MJCF has no `mass=` on
        # `<body>`; the only spelling is an explicit inertial frame — which
        # FREEZES the inertia that was until now DERIVED from the geoms, so a
        # later size edit stops changing this body's dynamics. `writer`'s
        # `_body_xml` refuses to write one unprompted for exactly this reason.
        # Setting a mass by hand IS that override, so the note says so and the
        # values come from the BUILT model, which is where the derivation ran.
        var o = e.index * MODEL_BODY_SIZE
        var inert = String('<inertial pos="')
        inert += _n(fmd.bodies[bi].ipos_x) + " " + _n(fmd.bodies[bi].ipos_y)
        inert += " " + _n(fmd.bodies[bi].ipos_z) + '" mass="'
        inert += _n(fmd.bodies[bi].mass) + '" diaginertia="'
        inert += _n(Float64(m.bodies.data[o + BODY_IDX_IXX])) + " "
        inert += _n(Float64(m.bodies.data[o + BODY_IDX_IYY])) + " "
        inert += _n(Float64(m.bodies.data[o + BODY_IDX_IZZ])) + '"/>'
        return _replace_or_insert_inertial(xml, at, inert)

    return xml


def _materialise_fromto(xml: String, at: Int, g: GeomData) -> String:
    """Turn a `fromto` capsule into explicit pos/quat/size, if it is one.

    ⚠⚠ `fromto` OVERRIDES BOTH pos AND size[1], so a `size=` or `pos=` written
    beside it is simply ignored — by MuJoCo and by our parser. Editing such a
    geom and leaving `fromto` in place produces a saved file that reads back
    with the OLD length and the OLD position while the inspector shows the new
    ones: the exact silent loss this whole write-through exists to close, and
    the reason swimmer's capsules failed the round trip when only `size` was
    written.

    ⚠ SO THE QUAT GOES IN TOO. `fromto` carries the ORIENTATION as well as
    the length; dropping it without writing the quat leaves the segment
    axis-aligned, which looks like the model falling apart.
    """
    if xml.find("fromto=", at) == -1:
        return xml
    var e = xml.find(">", at)
    if e == -1 or xml.find("fromto=", at) > e:
        return xml
    var out = set_attribute_at(
        xml, at, String("pos"),
        _n(g.pos_x) + " " + _n(g.pos_y) + " " + _n(g.pos_z),
    )
    out = set_attribute_at(
        out, at, String("quat"),
        _n(g.quat_w) + " " + _n(g.quat_x) + " " + _n(g.quat_y) + " "
        + _n(g.quat_z),
    )
    out = set_attribute_at(out, at, String("size"), _geom_size_attr(g))
    return _drop_attribute_at(out, at, String("fromto"))


def _drop_attribute_at(xml: String, at: Int, attr: String) -> String:
    """Remove `attr="…"` from the open tag at `at`, if present."""
    var e = xml.find(">", at)
    if e == -1:
        return xml
    var head = String(xml[byte=at:e])
    var needle = attr + '="'
    var k = head.find(needle)
    if k <= 0:
        return xml
    var ve = head.find('"', k + needle.byte_length())
    if ve == -1:
        return xml
    var cut_from = k - 1 if String(head[byte = k - 1 : k]) == " " else k
    var new_head = String(head[byte=0:cut_from]) + String(
        head[byte = ve + 1 : head.byte_length()]
    )
    return (
        String(xml[byte=0:at]) + new_head
        + String(xml[byte = e : xml.byte_length()])
    )


def _replace_or_insert_inertial(
    xml: String, body_at: Int, inert: String
) raises -> String:
    """Put `inert` in as the body's `<inertial>`, replacing an existing one."""
    var open_end = xml.find(">", body_at)
    if open_end == -1:
        return xml
    var span_end = element_end(xml, String("body"), body_at)
    var existing = _find_tag(xml, String("<inertial"), open_end)
    if existing != -1 and existing < span_end:
        var ee = element_end(xml, String("inertial"), existing)
        return (
            String(xml[byte=0:existing]) + inert
            + String(xml[byte = ee : xml.byte_length()])
        )
    if String(xml[byte = open_end - 1 : open_end]) == "/":
        return xml
    return (
        String(xml[byte = 0 : open_end + 1]) + "\n      " + inert
        + String(xml[byte = open_end + 1 : xml.byte_length()])
    )
