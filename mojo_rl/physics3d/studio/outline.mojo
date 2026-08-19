"""The yellow wireframe around the selected geom — studio S1.

MuJoCo's `simulate` outlines the selected element, and it is not decoration:
without it the panel names something and the user has to guess which shape on
screen that is. On a 128-geom model the guess is wrong most of the time.

⚠ IT IS LINE GEOMETRY, NOT A SHADER. `Renderer3D.draw_line_3d` and its
`LINELIST` pipeline already exist, so this file only has to emit segments.
That is the whole reason the highlight was cheap; a silhouette/stencil pass
would have been Metal work.

⚠⚠ THE SHAPE COMES FROM `RenderFields`, MATCHING THE PICKER AND THE DRAW. All
three read the same records, so the outline lands on the geom the ray hit and
the renderer drew. Sourcing it from `Model.geoms` instead would put the box in
the right place for a *different* set of geoms — see `pick.mojo`'s header.

The wireframes are deliberately COARSE (a box is 12 edges, a sphere 3 rings of
16 segments): they mark a selection, they do not describe a surface. A mesh
gets its bounding box, which is honest — the picker uses a bound for meshes
too, so outlining the true hull would promise a precision the selection does
not have.
"""

from std.math import sin, cos, pi

from mojo_rl.math3d import Vec3 as Vec3G, Quat as QuatG
from mojo_rl.render import Color
from ..model.model_renderer import OverlayLine
from ..parser.render_fields import RenderFields

comptime Vec3 = Vec3G[DType.float64]
comptime Quat = QuatG[DType.float64]

comptime RF_PLANE: Int = 0
comptime RF_SPHERE: Int = 1
comptime RF_CAPSULE: Int = 2
comptime RF_BOX: Int = 3
comptime RF_CYLINDER: Int = 4

comptime SELECT_COLOR = Color(255, 230, 40, 255)
"""MuJoCo's selection yellow. Bright, and not a colour any model's material
palette lands on by accident — a highlight the same colour as the thing it
highlights is not a highlight."""

comptime RING_SEGS: Int = 20


def _seg(mut out: List[OverlayLine], c: Vec3, q: Quat, a: Vec3, b: Vec3):
    """One segment, given in the geom's LOCAL frame."""
    out.append(OverlayLine(c + q.rotate_vec(a), c + q.rotate_vec(b),
                           SELECT_COLOR))


def _ring(mut out: List[OverlayLine], c: Vec3, q: Quat, r: Float64,
          axis: Int, offset: Float64 = 0.0):
    """A circle of radius `r` in the plane normal to `axis`, at `offset`."""
    var prev = Vec3(0.0, 0.0, 0.0)
    for i in range(RING_SEGS + 1):
        var th = 2.0 * pi * Float64(i) / Float64(RING_SEGS)
        var u = r * cos(th)
        var v = r * sin(th)
        var p = Vec3(offset, u, v)
        if axis == 1:
            p = Vec3(u, offset, v)
        elif axis == 2:
            p = Vec3(u, v, offset)
        if i > 0:
            _seg(out, c, q, prev, p)
        prev = p


def _box_edges(mut out: List[OverlayLine], c: Vec3, q: Quat,
               hx: Float64, hy: Float64, hz: Float64):
    """The 12 edges of an axis-aligned box in the geom's own frame."""
    for sx in range(2):
        for sy in range(2):
            for sz in range(2):
                var x = hx if sx == 1 else -hx
                var y = hy if sy == 1 else -hy
                var z = hz if sz == 1 else -hz
                var p = Vec3(x, y, z)
                # ⚠ ONE EDGE PER AXIS PER CORNER, ONLY IN THE + DIRECTION.
                # Emitting both directions draws all 12 edges TWICE, which is
                # invisible on screen and doubles the segment count for every
                # frame the selection is up.
                if sx == 0:
                    _seg(out, c, q, p, Vec3(hx, y, z))
                if sy == 0:
                    _seg(out, c, q, p, Vec3(x, hy, z))
                if sz == 0:
                    _seg(out, c, q, p, Vec3(x, y, hz))


def outline_geom(
    rf: RenderFields,
    geom: Int,
    positions: List[Vec3],
    quats: List[Quat],
    visual_radius_scale: Float64 = 1.0,
) -> List[OverlayLine]:
    """The selected geom's wireframe, in world space. Empty for no selection.

    `positions`/`quats` are the BODY poses this frame — the same arrays the
    picker and the renderer were handed, so the outline cannot lag the shape.
    """
    var out = List[OverlayLine]()
    if geom < 0 or geom >= len(rf.geom_type):
        return out^
    var bid = rf.geom_body_id[geom]
    if bid < 0 or bid >= len(positions):
        return out^

    var lp = Vec3(rf.geom_pos_x[geom], rf.geom_pos_y[geom],
                  rf.geom_pos_z[geom])
    var lq = Quat(rf.geom_quat_w[geom], rf.geom_quat_x[geom],
                  rf.geom_quat_y[geom], rf.geom_quat_z[geom])
    var c = positions[bid] + quats[bid].rotate_vec(lp)
    var q = quats[bid] * lq

    var gt = rf.geom_type[geom]
    var r = rf.geom_radius[geom] * visual_radius_scale
    if gt == RF_BOX:
        _box_edges(out, c, q, rf.geom_half_x[geom], rf.geom_half_y[geom],
                   rf.geom_half_z[geom])
    elif gt == RF_SPHERE:
        # Three great circles read as a sphere from any angle; one reads as a
        # ring the moment the camera is edge-on to it.
        _ring(out, c, q, r, 0)
        _ring(out, c, q, r, 1)
        _ring(out, c, q, r, 2)
    elif gt == RF_CAPSULE or gt == RF_CYLINDER:
        # Local Z is the axis — the convention `render_body_geoms` draws with.
        var hl = rf.geom_half_length[geom]
        _ring(out, c, q, r, 2, hl)
        _ring(out, c, q, r, 2, -hl)
        for k in range(4):
            var th = 2.0 * pi * Float64(k) / 4.0
            var u = r * cos(th)
            var v = r * sin(th)
            _seg(out, c, q, Vec3(u, v, -hl), Vec3(u, v, hl))
        if gt == RF_CAPSULE:
            # The caps, as two half-rings each, so the outline ends where the
            # drawn shape does rather than at the cylinder's flat end.
            _ring(out, c, q, r, 0, 0.0)
            _ring(out, c, q, r, 1, 0.0)
    else:
        # MESH / ELLIPSOID / anything new: the bounding box. ⚠ HONEST RATHER
        # THAN PRETTY — the PICKER uses a bounding sphere for these, so an
        # outline tracing the true hull would promise a precision the
        # selection itself does not have.
        var hx = rf.geom_half_x[geom]
        var hy = rf.geom_half_y[geom]
        var hz = rf.geom_half_z[geom]
        if hx <= 0.0 and hy <= 0.0 and hz <= 0.0:
            hx = r
            hy = r
            hz = r
        _box_edges(out, c, q, hx, hy, hz)
    return out^


def outline_body(
    rf: RenderFields,
    body: Int,
    positions: List[Vec3],
    quats: List[Quat],
    visual_radius_scale: Float64 = 1.0,
) -> List[OverlayLine]:
    """Every visible geom of a body, outlined — what selecting a BODY means.

    ⚠ THE VISIBILITY SKIPS ARE THE PICKER'S, again. Outlining a body's
    collision proxies would draw a yellow cage around a robot whose visible
    shape is its skin — dog has 123 hidden geoms and 5 shown.
    """
    var out = List[OverlayLine]()
    for g in range(len(rf.geom_type)):
        if rf.geom_body_id[g] != body:
            continue
        if rf.geom_type[g] == RF_PLANE:
            continue
        if rf.geom_group[g] >= 3 or rf.geom_rgba_a[g] < 1.0:
            continue
        var part = outline_geom(rf, g, positions, quats, visual_radius_scale)
        for i in range(len(part)):
            out.append(part[i].copy())
    return out^
