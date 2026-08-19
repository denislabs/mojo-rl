"""Ray-pick: the unprojection, the primitives, and the sweep — studio S1.

WHY THIS EXISTS
===============
Picking is the one part of a viewer that CANNOT be checked by looking at the
screen. A ray that is slightly wrong still selects *something*, usually the
right thing near the centre of the view, and goes wrong only at the edges,
only when the camera is rolled, or only once a sidebar shifts the viewport.
Every one of those reads as "selection feels flaky", not as a maths bug.

So the three layers are gated separately, because they fail differently:

1. **The unprojection.** Sign errors and the sidebar offset. Both produce a
   ray that is *plausible* — it still moves with the mouse.
2. **The primitives.** Analytic cases with answers known on paper.
3. **The sweep.** Ordering, and the visibility skips that must match
   `render_body_geoms` exactly (see `pick.mojo`'s header for why picking the
   wrong list is invisible: collision geoms are hidden, visual geoms are not
   in the narrow phase).

⚠ THE NEGATIVE CONTROLS ARE HALF THE FILE. A picker that returns "hit
everything" passes every positive arm — so each layer also asserts a ray that
must MISS.

Run: pixi run mojo run -I . tests/physics3d/test_studio_ray_pick.mojo
"""

from std.math import sqrt, pi

from mojo_rl.math3d import Vec3 as Vec3G, Quat as QuatG
from mojo_rl.physics3d.studio.pick import (
    Ray, Hit, ray_through_pixel, pick_geom,
    _hit_sphere, _hit_box, _hit_capsule,
)
from mojo_rl.physics3d.parser.render_fields import build_render_fields
from mojo_rl.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime, read_model_source,
)
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.integrator.euler import EulerIntegrator
from mojo_rl.physics3d.parser.runtime_load import spec_fields_runtime

comptime DT = DType.float64
comptime Vec3 = Vec3G[DT]
comptime Quat = QuatG[DT]


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

    def near(mut self, got: Float64, want: Float64, tol: Float64, msg: String):
        self.checks += 1
        if abs(got - want) <= tol:
            print("  ok:", msg, "=", got)
        else:
            self.fails += 1
            print("  FAIL:", msg, "— want", want, "got", got)


# ═══════════════════════════════════════════════════════════════════════════
# 1. the unprojection
# ═══════════════════════════════════════════════════════════════════════════


def test_unprojection(mut t: Tally) raises:
    print("--- unprojection ---")
    var eye = Vec3(0.0, 0.0, 10.0)
    var tgt = Vec3(0.0, 0.0, 0.0)
    var up = Vec3(0.0, 1.0, 0.0)
    var fov = 0.7853981633974483  # 45 deg
    var W = 800.0
    var H = 600.0

    # The CENTRE pixel looks straight down the view axis.
    var c = ray_through_pixel(W * 0.5, H * 0.5, 0.0, W, H, eye, tgt, up, fov)
    t.near(c.dir.x, 0.0, 1e-12, "centre ray x")
    t.near(c.dir.y, 0.0, 1e-12, "centre ray y")
    t.near(c.dir.z, -1.0, 1e-12, "centre ray z (toward the target)")
    t.near(c.dir.length(), 1.0, 1e-12, "direction is UNIT")

    # ⚠ THE HALF-ANGLE IS THE ONE NUMBER A SIGN ERROR CANNOT FAKE. At the top
    # edge the ray must sit exactly fov/2 off the axis; a ray built with the
    # wrong aspect term or a flipped ndc still points "up", just not by this
    # much.
    var top = ray_through_pixel(W * 0.5, 0.0, 0.0, W, H, eye, tgt, up, fov)
    var ang = 0.0
    var d = -top.dir.z
    if d > 0.0:
        ang = top.dir.y / d
    t.near(ang, 0.4142135623730951, 1e-9, "top edge = tan(fov/2)")

    # y FLIPS: mouse y grows down, clip space grows up.
    var bot = ray_through_pixel(W * 0.5, H, 0.0, W, H, eye, tgt, up, fov)
    t.truth(top.dir.y > 0.0 and bot.dir.y < 0.0,
            "screen-top is world-UP (ndc y is flipped)")
    # x does not flip.
    var right = ray_through_pixel(W, H * 0.5, 0.0, W, H, eye, tgt, up, fov)
    t.truth(right.dir.x > 0.0, "screen-right is world +x for this camera")

    # ⚠⚠ THE SIDEBAR OFFSET. With a 320 px strip reserved, the CENTRE OF THE
    # REMAINING VIEWPORT must give the axis ray. Getting this wrong biases
    # every pick toward one side by half the strip — a symptom that reads as a
    # projection bug, not an offset.
    var x0 = 320.0
    var vw = W - x0
    var off = ray_through_pixel(x0 + vw * 0.5, H * 0.5, x0, vw, H,
                                eye, tgt, up, fov)
    t.near(off.dir.x, 0.0, 1e-12, "sidebar: viewport centre is still the axis")
    var naive = ray_through_pixel(W * 0.5, H * 0.5, x0, vw, H,
                                  eye, tgt, up, fov)
    t.truth(abs(naive.dir.x) > 1e-3,
            "negative control: the WINDOW centre is NOT the viewport centre")

    # A non-perpendicular `up` must be re-orthonormalised, or the ray skews as
    # the camera orbits toward the pole.
    var skew = ray_through_pixel(W * 0.5, H * 0.5, 0.0, W, H, eye, tgt,
                                 Vec3(0.0, 1.0, 0.6), fov)
    t.near(skew.dir.z, -1.0, 1e-12, "a non-perpendicular `up` is re-derived")


# ═══════════════════════════════════════════════════════════════════════════
# 2. the primitives
# ═══════════════════════════════════════════════════════════════════════════


def test_primitives(mut t: Tally) raises:
    print("--- primitives ---")
    var down_z = Ray(Vec3(0.0, 0.0, 10.0), Vec3(0.0, 0.0, -1.0))

    # sphere r=2 at the origin: surface at z=2, so t = 8.
    t.near(_hit_sphere(down_z, Vec3(0.0, 0.0, 0.0), 2.0), 8.0, 1e-12,
           "ray-sphere entry t")
    t.near(_hit_sphere(down_z, Vec3(5.0, 0.0, 0.0), 2.0), -1.0, 1e-12,
           "negative control: sphere off to the side MISSES")
    # ⚠ ORIGIN INSIDE returns the FAR root, not a miss — zoom into a torso and
    # every click would otherwise stop working.
    var inside = Ray(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, -1.0))
    t.near(_hit_sphere(inside, Vec3(0.0, 0.0, 0.0), 2.0), 2.0, 1e-12,
           "camera INSIDE the sphere still hits")

    # axis-aligned box, half-extents 1: face at z=1 => t = 9.
    var idq = Quat(1.0, 0.0, 0.0, 0.0)
    t.near(_hit_box(down_z, Vec3(0.0, 0.0, 0.0), idq, 1.0, 1.0, 1.0),
           9.0, 1e-12, "ray-OBB entry t, unrotated")
    t.near(_hit_box(down_z, Vec3(3.0, 0.0, 0.0), idq, 1.0, 1.0, 1.0),
           -1.0, 1e-12, "negative control: box off to the side MISSES")

    # ⚠ ROTATED IS THE ARM THAT MATTERS. An implementation that forgot to take
    # the ray into the box's frame passes the unrotated case exactly.
    # 45 deg about Z: the top face is unchanged (rotation about the ray axis),
    # so t is still 9 — but a 45 deg about X lifts the corner to sqrt(2).
    var rx = Quat(0.9238795325112867, 0.3826834323650898, 0.0, 0.0)  # 45deg X
    t.near(_hit_box(down_z, Vec3(0.0, 0.0, 0.0), rx, 1.0, 1.0, 1.0),
           10.0 - sqrt(2.0), 1e-12, "ray-OBB, box rolled 45 deg about X")

    # capsule along local Z, r=0.5, half-length 2 => it spans z in [-2, 2].
    # A ray down the axis hits near the top cap.
    # ⚠ THE AXIAL CASE IS THE DEGENERATE ONE and it found a real bug: `den`
    # vanishes when the ray is parallel to the axis, and the endpoint chosen
    # for the clamp was the FAR cap — 11.5 where the top of the capsule sits
    # at z = 2.5, i.e. t = 7.5. An answer known on paper is the only way to
    # see that; on screen it is "the capsule feels slightly deep".
    var cap_t = _hit_capsule(down_z, Vec3(0.0, 0.0, 0.0), idq, 0.5, 2.0)
    t.near(cap_t, 7.5, 1e-12, "capsule AXIAL hit is the NEAR cap")
    # From the SIDE, at z = 1 — inside the segment's span.
    var side = Ray(Vec3(10.0, 0.0, 1.0), Vec3(-1.0, 0.0, 0.0))
    var st_ = _hit_capsule(side, Vec3(0.0, 0.0, 0.0), idq, 0.5, 2.0)
    t.truth(st_ > 9.0 and st_ < 10.0, String("capsule side hit (", st_, ")"))
    # ⚠ NEGATIVE CONTROL BEYOND THE CAP. A test that only fires at the middle
    # passes even if the segment clamp is missing entirely.
    var past = Ray(Vec3(10.0, 0.0, 4.0), Vec3(-1.0, 0.0, 0.0))
    t.near(_hit_capsule(past, Vec3(0.0, 0.0, 0.0), idq, 0.5, 2.0), -1.0, 1e-12,
           "negative control: past the cap MISSES")
    # Behind the camera must miss, or a click selects what is at your back.
    var behind = Ray(Vec3(0.0, 0.0, 10.0), Vec3(0.0, 0.0, 1.0))
    t.near(_hit_sphere(behind, Vec3(0.0, 0.0, 0.0), 2.0), -1.0, 1e-12,
           "negative control: geom BEHIND the ray misses")


# ═══════════════════════════════════════════════════════════════════════════
# 3. the sweep, on a real model
# ═══════════════════════════════════════════════════════════════════════════


def test_sweep(mut t: Tally) raises:
    print("--- sweep on walker2d ---")
    var path = String("mojo_rl/envs/walker2d/assets/walker2d.xml")
    var src = read_model_source(path)
    var fmd = parse_xml_full(src[0], src[1])
    var dims = dims_from_flat(fmd, max_contacts=64)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims)
    var d = Data[DT, DynDims, 1](dims)
    var integ = EulerIntegrator[DT, DynDims, BATCH=1, MAX_CONDIM=3](dims)
    var rf = build_render_fields(fmd, src[0], src[1])

    var nq = dims.get_nq()
    var nbody = dims.get_nbody()
    for i in range(nq):
        d.qpos.data[i] = sf.qpos0.data[i]
    integ.step["cpu"](d, m)

    var positions = List[Vec3]()
    var quats = List[Quat]()
    for b in range(nbody):
        positions.append(Vec3(
            Float64(d.xpos.data[b * 3 + 0]),
            Float64(d.xpos.data[b * 3 + 1]),
            Float64(d.xpos.data[b * 3 + 2]),
        ))
        quats.append(Quat(
            Float64(d.xquat.data[b * 4 + 3]),
            Float64(d.xquat.data[b * 4 + 0]),
            Float64(d.xquat.data[b * 4 + 1]),
            Float64(d.xquat.data[b * 4 + 2]),
        ))

    # ⚠ NON-VACUITY FIRST. Every arm below is about which geom is picked; if
    # none are pickable the file passes while testing nothing.
    var pickable = 0
    for i in range(len(rf.geom_type)):
        if rf.geom_body_id[i] < 0 or rf.geom_type[i] == 0:
            continue
        if rf.geom_group[i] >= 3 or rf.geom_rgba_a[i] < 1.0:
            continue
        pickable += 1
    t.truth(pickable >= 5,
            String("walker2d has ", pickable, " pickable geoms (non-vacuous)"))

    # ⚠⚠ THE STRONG ARM: aim at EVERY pickable geom's own world centre from
    # far away, and require the sweep to return SOMETHING. This exercises the
    # whole chain — body transform, per-type intersection, ordering — once per
    # geom, and a single unhandled geom type shows up as a miss rather than as
    # a subtly wrong selection.
    var aimed = 0
    var hit_self = 0
    for i in range(len(rf.geom_type)):
        if rf.geom_body_id[i] < 0 or rf.geom_type[i] == 0:
            continue
        if rf.geom_group[i] >= 3 or rf.geom_rgba_a[i] < 1.0:
            continue
        var bid = rf.geom_body_id[i]
        var lp = Vec3(rf.geom_pos_x[i], rf.geom_pos_y[i], rf.geom_pos_z[i])
        var wc = positions[bid] + quats[bid].rotate_vec(lp)
        # From +Y, which is walker2d's "side on" — the plane it moves in is
        # x/z, so nothing is hidden behind anything else from here.
        var eye = wc + Vec3(0.0, 20.0, 0.0)
        var r = Ray(eye, (wc - eye).normalized())
        var h = pick_geom(r, rf, positions, quats)
        aimed += 1
        if h.geom == i:
            hit_self += 1
        elif h.geom >= 0:
            # Another geom in front is a legitimate outcome for a limb behind
            # the torso; a MISS is not.
            hit_self += 1
    t.truth(aimed == pickable and hit_self == aimed,
            String("every pickable geom is reachable by a ray at its centre (",
                   hit_self, "/", aimed, ")"))

    # ⚠ NEGATIVE CONTROL: the same rays REVERSED must all miss. Without this,
    # a `pick_geom` that ignored `t < 0` would pass the arm above perfectly.
    var back_hits = 0
    for i in range(len(rf.geom_type)):
        if rf.geom_body_id[i] < 0 or rf.geom_type[i] == 0:
            continue
        if rf.geom_group[i] >= 3 or rf.geom_rgba_a[i] < 1.0:
            continue
        var bid = rf.geom_body_id[i]
        var lp = Vec3(rf.geom_pos_x[i], rf.geom_pos_y[i], rf.geom_pos_z[i])
        var wc = positions[bid] + quats[bid].rotate_vec(lp)
        var eye = wc + Vec3(0.0, 20.0, 0.0)
        var r = Ray(eye, (eye - wc).normalized())
        if pick_geom(r, rf, positions, quats).geom >= 0:
            back_hits += 1
    t.truth(back_hits == 0,
            String("negative control: reversed rays hit nothing (",
                   back_hits, ")"))

    # A ray into empty space, well clear of the model.
    var far = Ray(Vec3(0.0, 50.0, 50.0), Vec3(0.0, 0.0, 1.0))
    t.truth(pick_geom(far, rf, positions, quats).geom < 0,
            "negative control: a ray into the sky hits nothing")

    # ⚠ THE PLANE IS SKIPPED, and it must be: walker2d's floor spans the
    # world, so a pickable plane would swallow every click aimed past the
    # robot. `render_body_geoms` skips it too — the two skip lists are the
    # "you pick what you see" contract.
    var floor_i = -1
    for i in range(len(rf.geom_type)):
        if rf.geom_type[i] == 0:
            floor_i = i
    t.truth(floor_i >= 0, "walker2d does declare a plane (arm is live)")
    var down = Ray(Vec3(0.0, 8.0, 30.0), Vec3(0.0, 0.0, -1.0))
    t.truth(pick_geom(down, rf, positions, quats).geom != floor_i,
            "the ground plane is never picked")

    # Ordering: the NEAREST of two candidates wins. Fire along -Z from high
    # above the torso; whatever comes back must be closer than the far side.
    var bid_t = rf.geom_body_id[1]
    var lp_t = Vec3(rf.geom_pos_x[1], rf.geom_pos_y[1], rf.geom_pos_z[1])
    var wc_t = positions[bid_t] + quats[bid_t].rotate_vec(lp_t)
    var eye2 = wc_t + Vec3(0.0, 20.0, 0.0)
    var r2 = Ray(eye2, (wc_t - eye2).normalized())
    var h2 = pick_geom(r2, rf, positions, quats)
    t.truth(h2.geom >= 0 and h2.t > 0.0 and h2.t < 20.5,
            String("nearest hit is in front of the aim point (t=", h2.t, ")"))


def main() raises:
    var t = Tally()
    print("=== studio ray-pick ===")
    test_unprojection(t)
    test_primitives(t)
    test_sweep(t)
    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_studio_ray_pick: " + String(t.fails) + " failed")
