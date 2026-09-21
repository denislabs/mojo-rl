"""A LIBERO demonstration, re-rendered by OUR camera tracer, as an MP4.

    pixi run libero-camera-video put_the_bowl_on_the_plate
    pixi run mojo run -I . examples/libero/libero_camera_video.mojo put_the_bowl_on_the_plate
    pixi run mojo run -I . examples/libero/libero_camera_video.mojo turn_on_the_stove --demo 3
    pixi run mojo run -I . examples/libero/libero_camera_video.mojo turn_on_the_stove --camera eye_in_hand
    pixi run mojo run -I . examples/libero/libero_camera_video.mojo open_the_middle_drawer_of_the_cabinet \\
        --size 512 --samples 4 --scale 1 --out drawer_512.mp4

Reads `data/demo_<k>/states` from LIBERO's own HDF5, remaps every recorded
state into our scene's joint order (`envs/libero/state_remap`), renders each
one with the batched tracer's host leg at LIBERO's settings — 4x MSAA, the
visual geom group, the stove burner rule — and writes

    LEFT   LIBERO's recorded frame (`obs/agentview_rgb` or `eye_in_hand_rgb`)
    RIGHT  ours, from the same state

side by side, one frame per 20 Hz policy step, nearest-neighbour upscaled by
`--scale`. The mean PSNR of ours against the recording is printed at the end.
Needs `ffmpeg` on PATH and the gitignored demos under
`references/libero_demos/<suite>/` (`--demos-dir` to point elsewhere).

⚠⚠ THIS IS A STATE REPLAY, NOT A SIMULATION. Every frame is a recorded state
put into our engine and rendered; nothing is stepped. It shows what our camera
makes of LIBERO's own trajectory, so the two panels differ only by rendering
and scene — which is what a fidelity clip is for. `libero_demo_replay.mojo`
is the one that runs the demo's ACTIONS through our physics.

⚠ OBSERVATION `i` IS THE STATE AT `i + 1`, and it is stored upside down —
`tools/libero/libero_camera_gate.py`'s header measured both (6 dB and 26 dB
respectively). Both are applied here; a misaligned clip would look like a
one-frame lag and an unflipped one like a ceiling camera.

⚠⚠ THE FIXTURES ARE THE DEMO'S, READ FROM THE DEMO-SUCCESS DUMP — AND THEY ARE
WORTH 7 dB. LIBERO re-draws each fixture inside a 19-20 mm band per episode,
and the recorded pose lives in the `model_file` attribute, which `io/hdf5`
cannot read (no `H5A`). `tools/libero/libero_demo_success.py` extracts it with
MuJoCo into `references/libero_demos/_dumps/<suite>/<task>.dump` (`FIX` lines
per `DEMO` block), and this reads that file when it exists. Without it the
fixtures sit at the scene's band centre, up to ~1 cm off — measured on
`turn_on_the_stove` frame 78: 32.39 dB with the demo's poses, 25.34 without.
`--fixtures-dump FILE` points elsewhere; a missing dump is WARNED, not fatal.

⚠ `--size` OTHER THAN THE RECORDING'S 128 renders a sharper picture than a
policy ever gets. The left panel is then upscaled to match, nearest-neighbour,
so the comparison is honest about which side has the pixels.

⚠ HOST RENDERING, ONE CORE: ~0.3 s a frame at 128x128 with 4 samples on an
M1 Pro (`turn_on_the_stove`, 79 frames, 23 s), so `--size 512` is about
sixteen times that. `--stride K` renders every K-th step and `--frames N`
stops after N.

MEASURED, demo 0, whole clip against the recording: `turn_on_the_stove`
agentview 32.0 dB; `put_the_bowl_on_the_plate` eye_in_hand 29.7 dB (every
2nd step, 30 frames).
"""

from std.sys import argv
from std.math import log10
from std.memory.alloc import unsafe_alloc
from std.os.path import exists

from noeira.math3d import Vec3
from noeira.io.hdf5.reader import H5File
from noeira.physics3d.fields import Data, Model, DynDims
from noeira.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.raytrace.visual import build_visual_model
from noeira.physics3d.raytrace.host_render import render_lane_cpu
from noeira.render.video_recorder import VideoRecorder
from noeira.tasks.spec import load_family
from noeira.tasks.family import scene_path
from noeira.envs.libero.state_remap import load_state_remap
from noeira.envs.libero.visual import libero_site_conditions
from noeira.envs.libero.fixtures import patch_fixtures, fixtures_dump_path
from noeira.envs.libero.models.libero_goal_xml import LIBERO_GOAL_MAX_CONTACTS


comptime DT = DType.float64
comptime VISUAL_GROUP_MASK: Int = 1 << 1
"""robosuite renders with `render_collision_mesh=False` — group 1 only."""


def _to_byte(x: Float64) -> UInt8:
    var v = Int(x * 255.0 + 0.5)
    if v < 0:
        v = 0
    if v > 255:
        v = 255
    return UInt8(v)


def _usage() -> String:
    return String(
        "usage: libero_camera_video.mojo <task> [--suite libero_goal]"
        " [--demo 0] [--camera agentview|eye_in_hand] [--size 128]"
        " [--samples 4|1] [--scale 4] [--stride 1] [--frames 0]"
        " [--fps 20] [--no-compare] [--demos-dir DIR] [--fixtures-dump FILE]"
        " [--out FILE.mp4]"
    )


def main() raises:
    var args = argv()
    var task = String("")
    var suite = String("libero_goal")
    var demo = 0
    var camera = String("agentview")
    var size = 128
    var samples = 4
    var scale = 4
    var stride = 1
    var max_frames = 0
    var fps = 20
    var compare = True
    var demos_dir = String("references/libero_demos")
    var fixtures_dump = String("")
    var out = String("")
    var i = 1
    while i < len(args):
        var a = String(args[i])
        var has_val = i + 1 < len(args)
        if a == "--no-compare":
            compare = False
        elif a.startswith("--"):
            if not has_val:
                raise Error(a + " takes a value\n" + _usage())
            var v = String(args[i + 1])
            if a == "--suite":
                suite = v
            elif a == "--demo":
                demo = Int(v)
            elif a == "--camera":
                camera = v
            elif a == "--size":
                size = Int(v)
            elif a == "--samples":
                samples = Int(v)
            elif a == "--scale":
                scale = Int(v)
            elif a == "--stride":
                stride = Int(v)
            elif a == "--frames":
                max_frames = Int(v)
            elif a == "--fps":
                fps = Int(v)
            elif a == "--demos-dir":
                demos_dir = v
            elif a == "--fixtures-dump":
                fixtures_dump = v
            elif a == "--out":
                out = v
            else:
                # ⚠ REFUSED: a mistyped flag must not silently render the
                # wrong clip for minutes.
                raise Error("unknown option " + a + "\n" + _usage())
            i += 1
        elif task.byte_length() == 0:
            task = a
        else:
            raise Error("unexpected argument " + a + "\n" + _usage())
        i += 1
    if task.byte_length() == 0:
        raise Error(_usage())
    var cut = task.find("__")
    if cut >= 0:
        var t2 = String(task[byte = cut + 2 :])
        task = t2^
    if samples != 1 and samples != 4:
        raise Error("--samples must be 1 or 4 (MuJoCo's default is 4)")
    if scale < 1 or stride < 1 or size < 8:
        raise Error("--scale and --stride must be >= 1, --size >= 8")
    var scene_cam = String("arena_agentview")
    var obs_key = String("agentview_rgb")
    if camera == "eye_in_hand":
        scene_cam = String("robot_eye_in_hand")
        obs_key = String("eye_in_hand_rgb")
    elif camera != "agentview":
        raise Error("--camera must be agentview or eye_in_hand")
    if out.byte_length() == 0:
        out = (
            String("libero_") + suite + "__" + task + "_demo" + String(demo)
            + "_" + camera + ".mp4"
        )

    # ── the recording ─────────────────────────────────────────────────────
    var h5 = demos_dir + "/" + suite + "/" + task + "_demo.hdf5"
    if not exists(h5):
        raise Error(
            "no demo file " + h5 + " — the LIBERO HDF5s are gitignored; point"
            " --demos-dir at them"
        )
    var remap = load_state_remap(suite)
    var row_words = remap.row_words()
    var f5 = H5File(h5)
    var base = String("data/demo_") + String(demo) + "/"
    if not f5.has_dataset(base + "states"):
        raise Error(h5 + " has no " + base + "states")
    var d_st = f5.open_dataset(base + "states")
    var T = Int(d_st.dims[0])
    if Int(d_st.dims[1]) != row_words:
        raise Error(
            h5 + ": states are " + String(Int(d_st.dims[1])) + " wide, the "
            + suite + " remap expects " + String(row_words)
        )
    var states = unsafe_alloc[Scalar[DType.float64]](
        T * row_words
    ).as_unsafe_any_origin()
    d_st.read_all[DType.float64](states)

    var rec_h = 0
    var rec_w = 0
    var rec = unsafe_alloc[Scalar[DType.uint8]](1).as_unsafe_any_origin()
    if compare:
        var d_im = f5.open_dataset(base + "obs/" + obs_key)
        if d_im.ndim() != 4 or Int(d_im.dims[0]) != T or Int(d_im.dims[3]) != 3:
            raise Error(h5 + ": " + obs_key + " is not (T, H, W, 3)")
        rec_h = Int(d_im.dims[1])
        rec_w = Int(d_im.dims[2])
        if size % rec_h != 0 or rec_h != rec_w:
            raise Error(
                "--size " + String(size) + " is not a multiple of the recording's "
                + String(rec_h) + "; use --no-compare for an arbitrary size"
            )
        rec = unsafe_alloc[Scalar[DType.uint8]](
            T * rec_h * rec_w * 3
        ).as_unsafe_any_origin()
        d_im.read_all[DType.uint8](rec)

    # ── the scene and the camera ──────────────────────────────────────────
    var fam = load_family("noeira/envs/libero/families/" + suite + ".family")
    var fmd = parse_model_runtime(scene_path(fam))
    var verts = 32768
    var dims = dims_from_flat(
        fmd, max_contacts=LIBERO_GOAL_MAX_CONTACTS, nmesh_verts=verts
    )
    var m = Model[DT, DynDims](dims)
    while True:
        try:
            build_model_runtime[DT](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") < 0:
                raise e
            verts *= 2
            dims = dims_from_flat(
                fmd, max_contacts=LIBERO_GOAL_MAX_CONTACTS, nmesh_verts=verts
            )
            m = Model[DT, DynDims](dims)
    var d = Data[DT, DynDims, 1](dims)
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    if remap.nq != nq or remap.nv != nv:
        raise Error("the remap is for nq/nv " + String(remap.nq) + "/"
                    + String(remap.nv) + ", the scene has " + String(nq)
                    + "/" + String(nv))
    var vis = build_visual_model[DT, DynDims](
        fmd, m, group_mask=VISUAL_GROUP_MASK,
        conditions=libero_site_conditions(fam),
    )
    if fixtures_dump.byte_length() == 0:
        fixtures_dump = fixtures_dump_path(demos_dir, suite, task)
    var fix_note = String("")
    if exists(fixtures_dump):
        var placed = patch_fixtures[DT, DynDims](
            fixtures_dump, demo, fmd.body_names, m
        )
        if placed == 0:
            fix_note = String("  ⚠ ") + fixtures_dump + " has no FIX lines for demo_" + String(demo) + " — fixtures at the scene's poses (~1 cm off, ~7 dB)"
        else:
            fix_note = String("  fixtures: ") + String(placed) + " placed from " + fixtures_dump
    else:
        fix_note = (
            String("  ⚠ no fixture dump at ") + fixtures_dump
            + " — fixtures at the scene's band centre (~1 cm off, ~7 dB)."
            " Write it with `pixi run python tools/libero/libero_demo_success.py"
            " --suite " + suite + "`."
        )
    var cam = -1
    for k in range(len(fmd.camera_names)):
        if String(fmd.camera_names[k]) == scene_cam:
            cam = k
    if cam < 0:
        raise Error("no camera '" + scene_cam + "' in " + scene_path(fam))

    var nframes = (T - 1 + stride - 1) // stride
    if max_frames > 0 and nframes > max_frames:
        nframes = max_frames
    var up = size // rec_h if compare else 1
    var pw = size * scale
    var ph = size * scale
    var ow = pw * (2 if compare else 1)
    print("=" * 72)
    print("LIBERO camera video —", suite, "/", task, "demo", demo)
    print("=" * 72)
    print("  " + vis.describe())
    print(fix_note)
    print("  camera", scene_cam, "|", size, "x", size, "|", samples,
          "samples | scale", scale, "|", nframes, "frames of", T - 1,
          "| stride", stride)
    print("  ->", out, "(", ow, "x", ph, "@", fps, "fps )")

    var recorder = VideoRecorder()
    recorder.start(out, fps=fps)
    var frame = List[UInt8](length=ow * ph * 4, fill=UInt8(255))
    var row = List[Float64](length=row_words, fill=0.0)
    var qp = List[Float64](length=nq, fill=0.0)
    var qv = List[Float64](length=nv, fill=0.0)
    var rgb = List[Scalar[DT]]()
    var depth = List[Scalar[DT]]()
    var seg = List[Scalar[DT]]()
    var refl = List[Scalar[DT]]()
    var psnr_sum = 0.0
    var bg = Vec3[DT](0.60, 0.72, 0.90)

    for fi in range(nframes):
        var obs_i = fi * stride
        # ⚠ observation `i` shows the state AFTER step `i` — `states[i + 1]`.
        var st = obs_i + 1
        for k in range(row_words):
            row[k] = Float64(states[unsafe_offset = st * row_words + k])
        remap.convert_into(row, qp, qv)
        for k in range(nq):
            d.qpos.data[k] = Scalar[DT](qp[k])
        for k in range(nv):
            d.qvel.data[k] = Scalar[DT](qv[k])
        forward_kinematics["cpu", DT, DynDims, 1](d, m)
        if samples == 4:
            render_lane_cpu[DT, DynDims, 1, False, True, 4](
                d, m, vis, cam, 0, size, size, bg, rgb, depth, seg, refl,
            )
        else:
            render_lane_cpu[DT, DynDims, 1, False, True, 1](
                d, m, vis, cam, 0, size, size, bg, rgb, depth, seg, refl,
            )

        # ── compose BGRA: [recorded | ours], nearest-neighbour ────────────
        var se = 0.0
        var xo = pw if compare else 0
        for y in range(ph):
            var sy = y // scale
            for x in range(pw):
                var sx = x // scale
                var o = (y * ow + xo + x) * 4
                var s = (sy * size + sx) * 3
                var r = _to_byte(Float64(rgb[s]))
                var g = _to_byte(Float64(rgb[s + 1]))
                var b = _to_byte(Float64(rgb[s + 2]))
                frame[o] = b
                frame[o + 1] = g
                frame[o + 2] = r
                frame[o + 3] = UInt8(255)
                if compare:
                    # ⚠ STORED UPSIDE DOWN: recorded row 0 is the image bottom.
                    var ry = rec_h - 1 - sy // up
                    var rx = sx // up
                    var ro = ((obs_i * rec_h + ry) * rec_w + rx) * 3
                    var rr = rec[unsafe_offset = ro]
                    var rg = rec[unsafe_offset = ro + 1]
                    var rb = rec[unsafe_offset = ro + 2]
                    var lo = (y * ow + x) * 4
                    frame[lo] = rb
                    frame[lo + 1] = rg
                    frame[lo + 2] = rr
                    frame[lo + 3] = UInt8(255)
        if compare:
            # PSNR at the recording's own resolution, ours box-averaged down
            # when `--size` is larger.
            for ry in range(rec_h):
                for rx in range(rec_w):
                    for c in range(3):
                        var acc = 0.0
                        for yy in range(up):
                            for xx in range(up):
                                acc += Float64(
                                    _to_byte(Float64(rgb[((ry * up + yy) * size + rx * up + xx) * 3 + c]))
                                )
                        var ours = acc / Float64(up * up)
                        var theirs = Float64(
                            rec[unsafe_offset = ((obs_i * rec_h + (rec_h - 1 - ry)) * rec_w + rx) * 3 + c]
                        )
                        se += (ours - theirs) * (ours - theirs)
            var mse = se / Float64(rec_h * rec_w * 3)
            var p = 99.0 if mse <= 0.0 else 10.0 * log10(255.0 * 255.0 / mse)
            psnr_sum += p
        recorder.add_frame_bgra(Int(frame.unsafe_ptr()), ow, ph)
        if fi % 10 == 0 or fi == nframes - 1:
            print("  frame", fi + 1, "/", nframes)

    recorder.stop()
    states.free()
    rec.free()
    print("  wrote", out, "—", recorder.frames_written(), "frames")
    if compare and nframes > 0:
        print("  mean PSNR ours vs recording:", psnr_sum / Float64(nframes), "dB")
