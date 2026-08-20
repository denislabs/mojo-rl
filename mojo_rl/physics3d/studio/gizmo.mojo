"""The transform gizmo's arithmetic — everything about it that is not C++.

WHY THIS IS A SEPARATE FILE. `mojo_rl/render/imgui`'s `gz_*` bindings draw a
gizmo and hand back a 4x4. Turning that 4x4 into an `Edit` on a `pos=`/`quat=`
the document can carry is the part that can be WRONG QUIETLY, and it is the
part a headless test can reach. So the FFI lives there, the frame algebra
lives here, and `physics_studio.mojo` only wires the two together.

─────────────────────────────────────────────────────────────────────────────
THE THREE THINGS THAT MAKE A GIZMO SUBTLY WRONG, AND WHAT IS DONE ABOUT EACH
─────────────────────────────────────────────────────────────────────────────

⚠⚠ 1. THE GIZMO WORKS IN WORLD SPACE; MJCF STORES A LOCAL FRAME. A geom's
`pos`/`quat` are relative to its BODY, a body's to its PARENT. Handing the
gizmo the local numbers would draw it near the origin on any articulated
model and move the part by the wrong amount as soon as the parent is rotated.
So the element's world frame is COMPOSED for the gizmo and the parent frame's
inverse is applied to whatever comes back. `parent_frame` and `local_frame`
are one function each, so the compose and the un-compose cannot disagree.

⚠⚠ 2. THE MATRIX LAYOUT IS COLUMN-MAJOR AND THIS PROJECT'S `Mat4` IS NOT.
ImGuizmo takes OpenGL-style `float[16]` — translation at [12][13][14]. A
row-major matrix passed straight through does not fail; it draws a plausible
gizmo in the wrong place, rotated by the transpose. `frame_to_cm` /
`cm_to_frame` are the only two places that know the layout, and
`test_gizmo_math` round-trips them against each other AND against an
independently-built `Mat4`.

⚠⚠ 3. `float32` ROUND-TRIP NOISE IS AN EDIT. The matrix crosses the FFI as
`float32`, so merely GRABBING a handle re-quantises every component of a
`float64` record — and each changed component would be written into the
document, logged as an undo step and re-parsed. Two guards: only the fields
the OPERATION can change are emitted (a translate never touches `quat`), and
a component must move by more than `_noise_floor` of the model's own scale to
count. Without the second, a rotation drag rewrites `pos` with
`0.10000000149011612`.

⚠ AND THE BODY CASE IS NOT THE GEOM CASE. A geom's world frame is exactly
`body_world ∘ geom_local` — geoms have no degrees of freedom, so what is
drawn is what is edited. A BODY's rendered pose additionally carries its
JOINTS: `xpos` is the definition frame with the joint transform applied on
top. The gizmo is placed on the frame it EDITS (the definition frame), which
coincides with the rendered body at the reference pose and separates from it
once the sim has moved. `frame_drift` measures the gap so the caller can say
so rather than leave the user looking at a gizmo floating off the part.
"""

from std.math import sqrt

from mojo_rl.math3d import Vec3 as Vec3G, Quat as QuatG, Mat3 as Mat3G
from mojo_rl.math3d import Mat4 as Mat4G
from ..parser.flat_model import FlatModelDef
from .edit import (
    Edit, TARGET_GEOM, TARGET_BODY,
    F_POS_X, F_POS_Y, F_POS_Z, F_QUAT_W, F_QUAT_X, F_QUAT_Y, F_QUAT_Z,
)

comptime DT = DType.float64
comptime Vec3 = Vec3G[DT]
comptime Quat = QuatG[DT]
comptime Mat3 = Mat3G[DT]
comptime Mat4 = Mat4G[DT]


# ── the studio's gizmo modes, which are NOT ImGuizmo's OPERATION bits ───────
# ⚠ DELIBERATELY A SEPARATE ENUM. `render.imgui`'s `GZ_TRANSLATE` is a
# bitmask that has to match a C++ header; this is what the panel toggles and
# what the undo key is built from. Collapsing them would put an FFI constant
# in the panel's state and make a shim change a UI change.
comptime GIZMO_OFF: Int = 0
comptime GIZMO_MOVE: Int = 1
comptime GIZMO_TURN: Int = 2


def gizmo_mode_name(m: Int) -> String:
    if m == GIZMO_MOVE:
        return String("move")
    if m == GIZMO_TURN:
        return String("turn")
    return String("off")


@fieldwise_init
struct Frame(Copyable, ImplicitlyCopyable, Movable):
    """A rigid pose. The only thing a gizmo may produce and MJCF may store."""

    var pos: Vec3
    var quat: Quat

    @staticmethod
    def identity() -> Self:
        return Self(Vec3(0.0, 0.0, 0.0), Quat(1.0, 0.0, 0.0, 0.0))

    def compose(self, inner: Self) -> Self:
        """`self ∘ inner` — `inner` expressed in `self`'s parent."""
        return Self(
            self.pos + self.quat.rotate_vec(inner.pos),
            (self.quat * inner.quat).normalized(),
        )

    def inverse(self) -> Self:
        var qi = self.quat.inverse()
        return Self(-qi.rotate_vec(self.pos), qi)


def frame_to_cm(f: Frame) -> List[Float32]:
    """The pose as ImGuizmo's COLUMN-MAJOR `float[16]`.

    ⚠ INDEX = col * 4 + row. Translation therefore lands at 12/13/14, not at
    3/7/11. This project's `Mat4` is ROW-major, which is why this is written
    out by hand rather than copied off one — the two conventions differ by a
    transpose that is invisible on an identity rotation and on nothing else.
    """
    var r = Mat3.from_quat(f.quat)
    var m = List[Float32](length=16, fill=Float32(0))
    m[0] = Float32(r.m00); m[1] = Float32(r.m10); m[2] = Float32(r.m20)
    m[4] = Float32(r.m01); m[5] = Float32(r.m11); m[6] = Float32(r.m21)
    m[8] = Float32(r.m02); m[9] = Float32(r.m12); m[10] = Float32(r.m22)
    m[12] = Float32(f.pos.x)
    m[13] = Float32(f.pos.y)
    m[14] = Float32(f.pos.z)
    m[15] = Float32(1)
    return m^


def mat4_to_cm(m: Mat4) -> List[Float32]:
    """A ROW-major `Mat4` as ImGuizmo's COLUMN-major `float[16]` — a transpose.

    ⚠ THIS IS THE SAME TRANSPOSE `render.gpu_types.mat4_to_gpu_f32` DOES, and
    it is spelled out again rather than imported so that the studio does not
    take a dependency on the GPU uniform layout for its UI. If the two ever
    disagree the gizmo lands somewhere the scene is not, which is the one
    failure this file exists to make impossible; `test_gizmo_math` compares
    them element for element for that reason.
    """
    var o = List[Float32](length=16, fill=Float32(0))
    o[0] = Float32(m.m00); o[1] = Float32(m.m10)
    o[2] = Float32(m.m20); o[3] = Float32(m.m30)
    o[4] = Float32(m.m01); o[5] = Float32(m.m11)
    o[6] = Float32(m.m21); o[7] = Float32(m.m31)
    o[8] = Float32(m.m02); o[9] = Float32(m.m12)
    o[10] = Float32(m.m22); o[11] = Float32(m.m32)
    o[12] = Float32(m.m03); o[13] = Float32(m.m13)
    o[14] = Float32(m.m23); o[15] = Float32(m.m33)
    return o^


def cm_to_frame(m: List[Float32]) -> Frame:
    """The inverse of `frame_to_cm`, with any scale divided back out.

    ⚠ THE COLUMNS ARE RE-NORMALISED, and that is not defensive tidying. Even
    a pure rotation comes back with column lengths a few ULPs off 1 after a
    `float32` round trip, and `Mat3.to_quat` reads the trace — so an
    un-normalised matrix yields a quaternion that is not unit, which the
    physics then propagates into every child frame. A DEGENERATE column
    (length 0, which a scale gizmo can produce) falls back to identity rather
    than dividing by zero.
    """
    if len(m) < 16:
        return Frame.identity()
    var c0 = Vec3(Float64(m[0]), Float64(m[1]), Float64(m[2]))
    var c1 = Vec3(Float64(m[4]), Float64(m[5]), Float64(m[6]))
    var c2 = Vec3(Float64(m[8]), Float64(m[9]), Float64(m[10]))
    var l0 = c0.length()
    var l1 = c1.length()
    var l2 = c2.length()
    var p = Vec3(Float64(m[12]), Float64(m[13]), Float64(m[14]))
    if l0 < 1e-12 or l1 < 1e-12 or l2 < 1e-12:
        return Frame(p, Quat(1.0, 0.0, 0.0, 0.0))
    var q = Mat3.from_cols(c0 / l0, c1 / l1, c2 / l2).to_quat().normalized()
    # ⚠ THE SIGN IS PINNED TO w >= 0. `q` and `-q` are the same rotation, and
    # `to_quat` picks whichever the trace branch lands on — so a gizmo held
    # still could flip all four components, which reads downstream as "every
    # quat component changed" and writes an edit that rotates nothing.
    if q.w < 0.0:
        q = -q
    return Frame(p, q)


# ═══════════════════════════════════════════════════════════════════════════
# where the gizmo goes
# ═══════════════════════════════════════════════════════════════════════════


def parent_frame(
    fmd: FlatModelDef, positions: List[Vec3], quats: List[Quat],
    body_parent: List[Int], target: Int, index: Int,
) -> Frame:
    """The WORLD frame the element's stored `pos`/`quat` are relative to.

    A geom's is its own body's; a body's is its PARENT body's. `positions` /
    `quats` are this frame's forward-kinematics output — the same arrays the
    picker, the outline and the renderer were handed, so the gizmo cannot sit
    on a pose one step old.
    """
    var b = -1
    if target == TARGET_GEOM:
        if index < 0 or index >= len(fmd.geoms):
            return Frame.identity()
        b = fmd.geoms[index].body_id
    else:
        # ⚠ body 0 IS THE WORLDBODY and is absent from `fmd.bodies`, so the
        # parent table is indexed by `index - 1`. The worldbody itself has no
        # parent and no editable frame.
        if index <= 0 or index - 1 >= len(body_parent):
            return Frame.identity()
        b = body_parent[index - 1]
    if b < 0 or b >= len(positions) or b >= len(quats):
        return Frame.identity()
    return Frame(positions[b], quats[b])


def local_frame(fmd: FlatModelDef, target: Int, index: Int) -> Frame:
    """The element's stored frame, exactly as the document spells it."""
    if target == TARGET_GEOM:
        if index < 0 or index >= len(fmd.geoms):
            return Frame.identity()
        ref g = fmd.geoms[index]
        return Frame(
            Vec3(g.pos_x, g.pos_y, g.pos_z),
            Quat(g.quat_w, g.quat_x, g.quat_y, g.quat_z),
        )
    var bi = index - 1
    if bi < 0 or bi >= len(fmd.bodies):
        return Frame.identity()
    ref b = fmd.bodies[bi]
    return Frame(
        Vec3(b.pos_x, b.pos_y, b.pos_z),
        Quat(b.quat_w, b.quat_x, b.quat_y, b.quat_z),
    )


def edit_frame(
    fmd: FlatModelDef, positions: List[Vec3], quats: List[Quat],
    body_parent: List[Int], target: Int, index: Int,
) -> Frame:
    """Where the gizmo is drawn: the world pose of the frame being edited."""
    return parent_frame(
        fmd, positions, quats, body_parent, target, index
    ).compose(local_frame(fmd, target, index))


def frame_drift(
    fmd: FlatModelDef, positions: List[Vec3], quats: List[Quat],
    body_parent: List[Int], target: Int, index: Int,
) -> Float64:
    """Metres between the frame the gizmo edits and the pose on screen.

    ⚠⚠ ZERO FOR EVERY GEOM, BY CONSTRUCTION, AND NOT FOR A JOINTED BODY.
    `xpos` carries the joint transform on top of the definition frame, so a
    body whose hinge has swung is DRAWN somewhere its `pos=` does not
    describe. The gizmo stays on the frame it edits — moving the drawn pose
    would mean writing an edit the document cannot express — and this is the
    number that lets the caller SAY so instead of leaving the user staring at
    a gizmo floating beside the part.
    """
    var b = index
    if target == TARGET_GEOM:
        if index < 0 or index >= len(fmd.geoms):
            return 0.0
        b = fmd.geoms[index].body_id
        # A geom's drawn pose IS `body_world ∘ geom_local`; there is nothing
        # for it to drift from.
        return 0.0
    if b <= 0 or b >= len(positions):
        return 0.0
    return (edit_frame(
        fmd, positions, quats, body_parent, target, index
    ).pos - positions[b]).length()


# ═══════════════════════════════════════════════════════════════════════════
# what comes back
# ═══════════════════════════════════════════════════════════════════════════


def _noise_floor(scale: Float64) -> Float64:
    """Below this, a component "changed" only because `float32` was involved.

    `float32` carries ~7 decimal digits, so a value of magnitude `s` comes
    back perturbed by up to ~1e-7 * s. Ten times that is comfortably above
    the noise and far below a drag, which moves at least a pixel — on a
    1 m model with a 900 px viewport that is ~1e-3 m, four decades clear.
    """
    var s = scale if scale > 1.0 else 1.0
    return 1e-6 * s


def edits_from_frame(
    target: Int, index: Int, op: Int, before: Frame, after: Frame,
    scale: Float64 = 1.0,
) -> List[Edit]:
    """The `Edit`s that turn `before` into `after`. Empty when nothing moved.

    ⚠⚠ THE OPERATION GATES WHICH FIELDS MAY BE WRITTEN AT ALL. A translate
    that also emitted `quat` would rewrite a rotation the user did not touch
    with its own `float32` round trip — and on a geom whose orientation came
    from `euler` or `fromto`, that rewrite CHANGES THE DOCUMENT'S SPELLING
    (see `_write_quat_at`). Restricting by operation is what keeps a drag on
    the move gizmo from silently rewriting how the file expresses rotation.

    ⚠ AND THE THRESHOLD IS THE SECOND GUARD, for the fields the operation
    does own: a grab with no motion must produce NO edit, or the undo stack
    fills with steps that do nothing and the document is rewritten in
    `float32`.
    """
    var out = List[Edit]()
    var eps = _noise_floor(scale)
    if op == GIZMO_MOVE:
        if abs(after.pos.x - before.pos.x) > eps:
            out.append(Edit(target, index, F_POS_X, after.pos.x))
        if abs(after.pos.y - before.pos.y) > eps:
            out.append(Edit(target, index, F_POS_Y, after.pos.y))
        if abs(after.pos.z - before.pos.z) > eps:
            out.append(Edit(target, index, F_POS_Z, after.pos.z))
        return out^
    if op != GIZMO_TURN:
        return out^
    # ⚠ ONE COMPONENT MOVING MEANS ALL FOUR ARE WRITTEN. A quaternion is not
    # four independent numbers: writing `w` and leaving `x` at its old value
    # is a different rotation AND is not unit. The threshold decides WHETHER
    # the rotation changed, never WHICH parts of it to write.
    #
    # ⚠ THE COMPONENT THRESHOLD IS ABSOLUTE, not scaled — a quaternion's
    # components are bounded by 1 regardless of how big the model is.
    var qeps = _noise_floor(1.0)
    var moved = (
        abs(after.quat.w - before.quat.w) > qeps
        or abs(after.quat.x - before.quat.x) > qeps
        or abs(after.quat.y - before.quat.y) > qeps
        or abs(after.quat.z - before.quat.z) > qeps
    )
    if not moved:
        return out^
    var q = after.quat.normalized()
    out.append(Edit(target, index, F_QUAT_W, q.w))
    out.append(Edit(target, index, F_QUAT_X, q.x))
    out.append(Edit(target, index, F_QUAT_Y, q.y))
    out.append(Edit(target, index, F_QUAT_Z, q.z))
    return out^


def gizmo_edits(
    fmd: FlatModelDef, positions: List[Vec3], quats: List[Quat],
    body_parent: List[Int], target: Int, index: Int, op: Int,
    world_after: List[Float32], scale: Float64 = 1.0,
) -> List[Edit]:
    """The whole round trip: gizmo matrix in, `Edit`s on a LOCAL frame out.

    This is the function the studio calls and the function the gate drives.
    Splitting the world→local step out of `edits_from_frame` would let the
    two disagree about which frame the threshold is measured in.
    """
    var par = parent_frame(fmd, positions, quats, body_parent, target, index)
    var before = local_frame(fmd, target, index)
    var after = par.inverse().compose(cm_to_frame(world_after))
    return edits_from_frame(target, index, op, before, after, scale)
