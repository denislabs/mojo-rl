"""Ray-pick against the geoms the renderer DRAWS — studio S1.

Click a thing, get its geom index. The index is directly a `FlatModelDef`
index (`build_render_fields` walks `fmd.geoms` in order), so the caller reads
the name straight out of `fmd.geom_names`.

## ⚠⚠ IT PICKS FROM `RenderFields`, NOT FROM THE PHYSICS MODEL

`Model.geoms` and `RenderFields` describe the same geoms, and picking from the
wrong one is a bug you cannot see: a collision-only geom is INVISIBLE (group
>= 3, or fully transparent) and picking it would select something the user
cannot see, while a visual-only geom is absent from the narrow phase and is
UNPICKABLE though it is the thing on screen. dm_control's dog is the extreme
case — 123 of its 128 geoms are hidden by group, and the 5 you see are the
skin.

⇒ **this sweep SHARES `render_body_geoms`' skips**: `body_geom_visible` in
`parser/render_fields.mojo`, plus the body -1 bound. "You pick what you see"
is the contract, and it is only true if the rule is one rule — restating it
here is what let the two drift (`alpha < 1.0` here against `alpha < 0.99`
there, so a geom at 0.995 drew and could not be picked).

## What is exact and what is a bound

| geom | test | exact? |
|---|---|---|
| SPHERE | ray-sphere | yes |
| BOX | ray-OBB slabs in the geom's own frame | yes |
| CAPSULE / CYLINDER | ray-vs-SEGMENT distance <= radius | **hit test exact for a capsule; `t` is the ray's closest approach, not the surface entry** |
| MESH / ELLIPSOID | bounding sphere | **no — a bound** |
| PLANE | not picked | — |

A picker wants "which of these did I click", and the depth ordering that
answers it is stable under those approximations for every case where two
geoms are not nearly coincident. A cylinder is treated as a capsule, so its
corners over-select slightly. Written down rather than hidden because the day
someone reuses this for a rangefinder SENSOR, the `t` values are the output
and these approximations stop being acceptable.
"""

from std.math import tan, sqrt

from mojo_rl.math3d import Vec3 as Vec3G, Quat as QuatG
from ..parser.render_fields import RenderFields, body_geom_visible

comptime Vec3 = Vec3G[DType.float64]
comptime Quat = QuatG[DType.float64]

comptime RF_PLANE: Int = 0
comptime RF_SPHERE: Int = 1
comptime RF_CAPSULE: Int = 2
comptime RF_BOX: Int = 3
comptime RF_CYLINDER: Int = 4
comptime RF_MESH: Int = 5
comptime RF_ELLIPSOID: Int = 6


@fieldwise_init
struct Ray(Copyable, Movable):
    var origin: Vec3
    var dir: Vec3
    """UNIT length. Several tests below assume it, and a picker fed a
    non-normalised direction returns `t` in units of that direction — which
    compares fine within one call and is meaningless across calls."""


def ray_through_pixel(
    mouse_x: Float64,
    mouse_y: Float64,
    viewport_x0: Float64,
    viewport_w: Float64,
    viewport_h: Float64,
    eye: Vec3,
    target: Vec3,
    up: Vec3,
    fov_y: Float64,
) -> Ray:
    """The world-space ray under a window pixel.

    ⚠ `viewport_x0` IS NOT ZERO WHEN A SIDEBAR IS UP. `Renderer3D` reserves a
    strip on the LEFT (`ui_sidebar_width`) and renders the scene into what is
    left, so a ray built from the raw window coordinate is off by the strip's
    width — which reads as "picking is systematically biased to the right",
    a symptom that looks like a projection bug rather than an offset.

    ⚠ NDC y IS FLIPPED. Mouse y grows downward, clip space grows upward.
    """
    var nx = (2.0 * (mouse_x - viewport_x0) / viewport_w) - 1.0
    var ny = 1.0 - (2.0 * mouse_y / viewport_h)

    var fwd = (target - eye).normalized()
    var right = fwd.cross(up).normalized()
    # ⚠ RE-DERIVED, NOT THE CAMERA'S `up`. The stored `up` is a HINT and need
    # not be perpendicular to `fwd`; using it directly skews the ray as the
    # camera orbits toward the pole. This is the same orthonormalisation the
    # view matrix does, and picking has to match the view matrix or the ray
    # will not land where the pixel was drawn.
    var upv = right.cross(fwd).normalized()

    var half_h = tan(fov_y * 0.5)
    var half_w = half_h * (viewport_w / viewport_h)
    var d = fwd + right * (nx * half_w) + upv * (ny * half_h)
    return Ray(eye, d.normalized())


@fieldwise_init
struct Hit(Copyable, Movable):
    var geom: Int
    """Index into `rf.geom_*` AND into `fmd.geoms` / `fmd.geom_names` — the
    two are built in the same order by `build_render_fields`. -1 for a miss."""
    var t: Float64
    var point: Vec3


def _hit_sphere(r: Ray, c: Vec3, radius: Float64) -> Float64:
    """Nearest positive `t`, or -1."""
    var oc = r.origin - c
    var b = oc.dot(r.dir)
    var cc = oc.dot(oc) - radius * radius
    var disc = b * b - cc
    if disc < 0.0:
        return -1.0
    var sd = sqrt(disc)
    var t0 = -b - sd
    if t0 > 1e-6:
        return t0
    var t1 = -b + sd
    # ⚠ THE FAR ROOT IS KEPT ON PURPOSE. The camera can be INSIDE a geom —
    # zoom into a torso and every click would otherwise miss it — and a picker
    # that silently stops working when you fly inside the robot is worse than
    # one that picks the thing you are inside.
    if t1 > 1e-6:
        return t1
    return -1.0


def _hit_box(r: Ray, c: Vec3, q: Quat, hx: Float64, hy: Float64,
             hz: Float64) -> Float64:
    """Exact ray-OBB, by taking the ray into the box's own frame."""
    var lo = q.rotate_vec_inverse(r.origin - c)
    var ld = q.rotate_vec_inverse(r.dir)
    var tmin = -1.0e30
    var tmax = 1.0e30
    for a in range(3):
        var o = lo[a]
        var d = ld[a]
        var h = hx if a == 0 else (hy if a == 1 else hz)
        if d > -1e-12 and d < 1e-12:
            # Parallel to this slab: a miss only if the origin is outside it.
            if o < -h or o > h:
                return -1.0
            continue
        var t1 = (-h - o) / d
        var t2 = (h - o) / d
        if t1 > t2:
            var s = t1
            t1 = t2
            t2 = s
        if t1 > tmin:
            tmin = t1
        if t2 < tmax:
            tmax = t2
        if tmin > tmax:
            return -1.0
    if tmin > 1e-6:
        return tmin
    if tmax > 1e-6:
        return tmax  # origin inside — same reason as the sphere's far root
    return -1.0


def _hit_capsule(r: Ray, c: Vec3, q: Quat, radius: Float64,
                 half_len: Float64) -> Float64:
    """Closest approach of the ray to the capsule's SEGMENT, if within radius.

    ⚠ THE `t` IS THE CLOSEST APPROACH, NOT THE SURFACE ENTRY, and for picking
    that is the right trade: the hit TEST is exact for a capsule (a point is
    inside iff its distance to the segment is <= radius), only the depth is
    approximate, and depth is used here solely to order two hits. See the
    module docstring before reusing this as a sensor.
    """
    # The capsule's axis is local Z — the same convention `render_body_geoms`
    # draws with (`axis=2`).
    var ax = q.rotate_vec(Vec3(0.0, 0.0, 1.0))
    var p0 = c - ax * half_len
    var seg = ax * (2.0 * half_len)

    var w0 = r.origin - p0
    var a = r.dir.dot(r.dir)
    var b = r.dir.dot(seg)
    var cc = seg.dot(seg)
    var d = r.dir.dot(w0)
    var e = seg.dot(w0)
    var den = a * cc - b * b

    var t_ray = 0.0
    var s_seg = 0.0
    if den > 1e-12 or den < -1e-12:
        t_ray = (b * e - cc * d) / den
        s_seg = (a * e - b * d) / den
    else:
        # ⚠⚠ RAY PARALLEL TO THE AXIS — AND THE NEAR CAP IS NOT ALWAYS `s=0`.
        # `den` vanishes here, so there is no closest-approach solution and an
        # endpoint has to be chosen. Choosing `s = 0` unconditionally picks
        # the cap the ray reaches LAST whenever the capsule points toward the
        # camera: a ray straight down a capsule's axis then reported the FAR
        # cap, `t` too large by the capsule's whole length. Two coaxial
        # capsules would select in the wrong order, and a lone one would
        # simply seem slightly deep — which is why this needed a gate with an
        # answer known on paper (`test_studio_ray_pick`, 11.5 vs 7.5) rather
        # than a look at the screen.
        var p1 = p0 + seg
        s_seg = 0.0 if (p0 - r.origin).dot(r.dir) \
            <= (p1 - r.origin).dot(r.dir) else 1.0
    if s_seg < 0.0:
        s_seg = 0.0
    elif s_seg > 1.0:
        s_seg = 1.0
    # Re-solve the ray parameter against the clamped segment point, or a click
    # near either cap reports a distance from the wrong place.
    var sp = p0 + seg * s_seg
    t_ray = (sp - r.origin).dot(r.dir)
    if t_ray < 1e-6:
        return -1.0
    var rp = r.origin + r.dir * t_ray
    if (rp - sp).length() > radius:
        return -1.0
    # Step back to roughly the surface so ordering against a sphere or box,
    # which report entry points, is not biased by the capsule's radius.
    var t_surface = t_ray - radius
    return t_surface if t_surface > 1e-6 else t_ray


def pick_geom(
    r: Ray,
    rf: RenderFields,
    positions: List[Vec3],
    quats: List[Quat],
    visual_radius_scale: Float64 = 1.0,
) -> Hit:
    """Nearest visible geom under the ray, or `Hit(-1, …)`.

    `positions` / `quats` are the BODY poses the renderer was handed this
    frame — the same arrays, so a pick cannot select a geom at a stale pose.
    """
    var best = Hit(-1, 1.0e30, Vec3(0.0, 0.0, 0.0))
    for i in range(len(rf.geom_type)):
        var bid = rf.geom_body_id[i]
        # ── the skips, IDENTICAL to `render_body_geoms` — now by SHARING the
        # predicate rather than by restating it. The comment claimed identity
        # while the alpha thresholds differed (`< 1.0` here, `< 0.99` there);
        # "you pick what you see" cannot survive two copies of the rule.
        if bid < 0 or bid >= len(positions):
            continue
        var gt = rf.geom_type[i]
        if not body_geom_visible(rf, i):
            continue

        var bp = positions[bid]
        var bq = quats[bid]
        var lp = Vec3(rf.geom_pos_x[i], rf.geom_pos_y[i], rf.geom_pos_z[i])
        var lq = Quat(
            rf.geom_quat_w[i], rf.geom_quat_x[i],
            rf.geom_quat_y[i], rf.geom_quat_z[i],
        )
        var wc = bp + bq.rotate_vec(lp)
        var wq = bq * lq

        var t = -1.0
        var rad = rf.geom_radius[i] * visual_radius_scale
        if gt == RF_SPHERE:
            t = _hit_sphere(r, wc, rad)
        elif gt == RF_BOX:
            t = _hit_box(
                r, wc, wq,
                rf.geom_half_x[i], rf.geom_half_y[i], rf.geom_half_z[i],
            )
        elif gt == RF_CAPSULE or gt == RF_CYLINDER:
            t = _hit_capsule(r, wc, wq, rad, rf.geom_half_length[i])
        else:
            # MESH / ELLIPSOID — a bounding sphere. `geom_radius` is the box
            # diagonal for a box and the hull bound for a mesh; see mapping 4
            # in `build_render_fields`.
            var br = rad
            if br <= 0.0:
                var hx = rf.geom_half_x[i]
                var hy = rf.geom_half_y[i]
                var hz = rf.geom_half_z[i]
                br = sqrt(hx * hx + hy * hy + hz * hz)
            if br <= 0.0:
                continue
            t = _hit_sphere(r, wc, br)

        if t > 0.0 and t < best.t:
            best = Hit(i, t, r.origin + r.dir * t)
    return best^
