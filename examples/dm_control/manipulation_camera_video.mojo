"""What the batched camera tracer actually hands a manipulation policy.

    pixi run mojo run -I . examples/dm_control/manipulation_camera_video.mojo

Writes two files:

    manipulation_lift_brick_rgb.mp4      the 84x84 observation, upscaled 4x
    manipulation_lift_brick_panels.mp4   rgb | depth | segmentation

⚠⚠ THIS IS NOT `benchmarks/camera_tracer_lift_brick.mojo`. That file also emits
a clip, and its clip is nearly STILL: it steps with `E.ActionType()`, which is
an all-zero action, so the arm hangs where it was reset and a "manipulation
video" shows no manipulation. It is right for a benchmark — a fixed pose is a
reproducible one — and wrong for looking at. This file drives the actuators.

⚠ THE DRIVE IS THE VIEWER'S `sweep`, not a policy: a slow out-of-phase sine per
actuator (`viewer_core.mojo`, `DRIVE_SWEEP`). No policy is trained on this task
here, and a random walk mostly jitters; the sine sweeps every joint through its
range in a way an eye can follow, which is what a fidelity clip is for. It is
NOT a task demonstration and the arm is not trying to lift anything.

⚠⚠ 84x84 IS THE OBSERVATION AND THE UPSCALE IS NEAREST-NEIGHBOUR. The video is
4x bigger than the tensor, with no filtering, so what you see is the real pixel
grid magnified rather than a prettier render the agent never gets. Judging the
fidelity off a 336x336 render would be judging the wrong image.

⚠⚠ WHAT THE BVH CHANGED IN THESE PIXELS: NOTHING, BY CONSTRUCTION. The tree
culls only triangles the ray provably misses, so it returns the identical
distance and normal, and `tests/physics3d/test_ray_bvh_matches_linear.mojo`
asserts EXACT equality on colour, depth and geom id. What changed is the time —
21.97x on a 5090. If this clip looks different from an older one, the cause is
the PHYSICS moving underneath it, not the renderer.

THE THREE CHANNELS, AND WHY THE PANEL FILE EXISTS
=================================================
`depth` and `seg` are computed by the SAME ray as the colour and written to
their own buffers at no extra intersection cost — `render.mojo` gets them out
of one `RayHit`. Nothing is wired to consume them yet. The panel clip exists so
the question "is 84x84 enough for manipulation?" can be asked of the
observation an RGB-D policy WOULD get, rather than of the colour alone: the
shading is flat per geom (one `geom_rgba`, no texture, no specular), so a
brick reads as a flat patch in RGB while it is unambiguous in depth.

⚠ DEPTH IS NORMALISED OVER THE WHOLE CLIP, NOT PER FRAME. A per-frame stretch
makes the background pulse whenever the nearest surface moves, which reads as
the depth channel being noisy when it is not. The range used is printed.

⚠ 0 MEANS NO HIT in `depth`, and it is a SENTINEL, not "zero metres" — it is
painted black here and excluded from the range. Same contract as
`rangefinder`'s -1.
"""

from std.math import sin, pi
from std.random import seed
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from mojo_rl.envs.dm_control.manipulation_lift_brick import DMLiftBrick
from mojo_rl.physics3d.dynamics.subtree_com import compute_subtree_com
from mojo_rl.physics3d.raytrace import (
    BatchedCameraRenderer,
    RGB_CHANNELS,
    init_camera_reference,
)
from mojo_rl.render.video_recorder import VideoRecorder

comptime DT = DType.float32
comptime W = 84
comptime H = 84
comptime NPIX = W * H
comptime CAM = 0  # "front_close", the task's only camera
comptime UPSCALE = 4

comptime FRAMES = 240  # 8 s at 30 fps
comptime FPS = 30
comptime SWEEP_PERIOD = 120.0  # steps per full sine cycle
comptime SWEEP_SCALE = 1.0

comptime E = DMLiftBrick[DT]
comptime R = BatchedCameraRenderer[DT, E.MD, 1, W, H]
comptime ACT_DIM = E.MODEL_DEF.ACTION_DIM

comptime SEED: Int = 0


def _to_byte(v: Float64) -> Int:
    """[0, 1] to [0, 255], clamped. The renderer already clamps colour, but
    depth and segmentation arrive unnormalised and this is their only guard."""
    var x = Int(v * 255.0 + 0.5)
    if x < 0:
        return 0
    if x > 255:
        return 255
    return x


def _seg_colour(g: Int) -> Tuple[Int, Int, Int]:
    """A stable colour per geom id, black for the background.

    ⚠ A HASH, NOT THE GEOM'S OWN `rgba`. Painting the segmentation with the
    render colours would make the two panels agree by construction and show
    nothing — the point of the channel is that it separates geoms the SHADING
    does not, and two grey links of an arm are exactly that case.
    """
    if g < 0:
        return (0, 0, 0)
    var r = (g * 97 + 61) % 256
    var gr = (g * 151 + 23) % 256
    var b = (g * 211 + 137) % 256
    # Keep it out of the very dark corner so the background stays distinct.
    return (64 + r * 3 // 4, 64 + gr * 3 // 4, 64 + b * 3 // 4)


def main() raises:
    comptime if not has_accelerator():
        print("no accelerator — this exporter renders on the GPU")
        return

    seed(SEED)
    var ctx = DeviceContext()
    var env = E()
    _ = env.reset()
    compute_subtree_com["cpu", DT, E.MD, 1](env.d, env.mf)
    init_camera_reference(env.d, env.mf)
    env.mf.upload_all(ctx)
    env.d.upload_all(ctx)
    ctx.synchronize()

    print("scene    : dm_control manipulation/lift_brick")
    print("geoms    :", E.MD.NGEOM, " mesh tri:", E.MD.NMESH_TRI)
    print("actuators:", ACT_DIM, " drive: sweep, scale", SWEEP_SCALE)
    print("camera   :", CAM, " resolution", W, "x", H, " frames", FRAMES)

    var r = R(ctx, env.mf, CAM)

    # ── the whole clip is buffered, then written ──────────────────────────
    #
    # ⚠ SO THE DEPTH RANGE CAN BE A CLIP-WIDE ONE. The alternative is a second
    # pass over the physics to collect the range first, which would have to
    # reproduce the trajectory exactly to be the same clip. 240 frames of
    # three channels is ~24 MB; the second pass is a correctness risk for no
    # saving.
    var rgb_all = List[Scalar[DT]](
        length=FRAMES * NPIX * RGB_CHANNELS, fill=Scalar[DT](0)
    )
    var dep_all = List[Scalar[DT]](length=FRAMES * NPIX, fill=Scalar[DT](0))
    var seg_all = List[Scalar[DT]](length=FRAMES * NPIX, fill=Scalar[DT](0))

    var h_rgb = ctx.enqueue_create_host_buffer[DT](NPIX * RGB_CHANNELS)
    var h_dep = ctx.enqueue_create_host_buffer[DT](NPIX)
    var h_seg = ctx.enqueue_create_host_buffer[DT](NPIX)

    var act = E.ActionType()
    for f in range(FRAMES):
        var t = Float64(f) / SWEEP_PERIOD
        for a in range(ACT_DIM):
            var phase = Float64(a) / Float64(ACT_DIM if ACT_DIM > 0 else 1)
            act.data[a] = sin(2.0 * pi * (t + phase)) * SWEEP_SCALE

        _ = env.step(act)
        compute_subtree_com["cpu", DT, E.MD, 1](env.d, env.mf)
        env.d.upload_all(ctx)
        ctx.synchronize()

        r.render(ctx, env.d, env.mf)
        ctx.synchronize()
        ctx.enqueue_copy(h_rgb, r.rgb)
        ctx.enqueue_copy(h_dep, r.depth)
        ctx.enqueue_copy(h_seg, r.seg)
        ctx.synchronize()

        var cb = f * NPIX
        for p in range(NPIX):
            dep_all[cb + p] = h_dep[p]
            seg_all[cb + p] = h_seg[p]
            for c in range(RGB_CHANNELS):
                rgb_all[(cb + p) * RGB_CHANNELS + c] = (
                    h_rgb[p * RGB_CHANNELS + c]
                )

    # ── the clip-wide depth range, over HITS only ─────────────────────────
    var dmin = Float64(1e30)
    var dmax = Float64(0.0)
    var nhit = 0
    for i in range(FRAMES * NPIX):
        var d = Float64(dep_all[i])
        if d <= 0.0:
            continue
        nhit += 1
        if d < dmin:
            dmin = d
        if d > dmax:
            dmax = d
    if nhit == 0:
        print("!! no pixel hit anything in the whole clip — nothing to look at")
        return
    var dspan = dmax - dmin
    if dspan <= 0.0:
        dspan = 1.0
    print(
        "depth    :", dmin, "..", dmax, "m over", nhit, "of",
        FRAMES * NPIX, "pixels",
    )

    # ── file 1: the observation ──────────────────────────────────────────
    var ow = W * UPSCALE
    var oh = H * UPSCALE
    var frame = List[UInt8](length=ow * oh * 4, fill=UInt8(255))
    var rec = VideoRecorder()
    rec.start(String("manipulation_lift_brick_rgb.mp4"), fps=FPS)
    for f in range(FRAMES):
        var cb = f * NPIX
        for py in range(H):
            for px in range(W):
                var p = cb + py * W + px
                var b = _to_byte(Float64(rgb_all[p * RGB_CHANNELS + 2]))
                var g = _to_byte(Float64(rgb_all[p * RGB_CHANNELS + 1]))
                var rr = _to_byte(Float64(rgb_all[p * RGB_CHANNELS + 0]))
                for dy in range(UPSCALE):
                    for dx in range(UPSCALE):
                        var o = ((py * UPSCALE + dy) * ow
                                 + (px * UPSCALE + dx)) * 4
                        frame[o + 0] = UInt8(b)
                        frame[o + 1] = UInt8(g)
                        frame[o + 2] = UInt8(rr)
        rec.add_frame_bgra(Int(frame.unsafe_ptr()), ow, oh)
    rec.stop()

    # ── file 2: rgb | depth | segmentation, side by side ─────────────────
    var pw = ow * 3 + 2 * UPSCALE  # two 1-px (pre-upscale) separators
    var panel = List[UInt8](length=pw * oh * 4, fill=UInt8(0))
    var rec2 = VideoRecorder()
    rec2.start(String("manipulation_lift_brick_panels.mp4"), fps=FPS)
    for f in range(FRAMES):
        var cb = f * NPIX
        for py in range(H):
            for px in range(W):
                var p = cb + py * W + px

                var cb_ = _to_byte(Float64(rgb_all[p * RGB_CHANNELS + 2]))
                var cg = _to_byte(Float64(rgb_all[p * RGB_CHANNELS + 1]))
                var cr = _to_byte(Float64(rgb_all[p * RGB_CHANNELS + 0]))

                # NEAR is BRIGHT — the convention every depth viewer uses, and
                # the opposite of the raw metres.
                var dv = Float64(dep_all[p])
                var dg = 0
                if dv > 0.0:
                    dg = _to_byte(1.0 - (dv - dmin) / dspan)

                var sc = _seg_colour(Int(seg_all[p]))

                for dy in range(UPSCALE):
                    for dx in range(UPSCALE):
                        var row = (py * UPSCALE + dy) * pw
                        var col = px * UPSCALE + dx
                        var o0 = (row + col) * 4
                        panel[o0 + 0] = UInt8(cb_)
                        panel[o0 + 1] = UInt8(cg)
                        panel[o0 + 2] = UInt8(cr)
                        panel[o0 + 3] = UInt8(255)
                        var o1 = (row + ow + UPSCALE + col) * 4
                        panel[o1 + 0] = UInt8(dg)
                        panel[o1 + 1] = UInt8(dg)
                        panel[o1 + 2] = UInt8(dg)
                        panel[o1 + 3] = UInt8(255)
                        var o2 = (
                            row + 2 * (ow + UPSCALE) + col
                        ) * 4
                        panel[o2 + 0] = UInt8(sc[2])
                        panel[o2 + 1] = UInt8(sc[1])
                        panel[o2 + 2] = UInt8(sc[0])
                        panel[o2 + 3] = UInt8(255)
        rec2.add_frame_bgra(Int(panel.unsafe_ptr()), pw, oh)
    rec2.stop()

    print("wrote manipulation_lift_brick_rgb.mp4    (", ow, "x", oh, ")")
    print("wrote manipulation_lift_brick_panels.mp4 (", pw, "x", oh, ")")
