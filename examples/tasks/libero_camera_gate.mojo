"""Our tracer against LIBERO's own pixels — the L5 gate's fourth column.

    pixi run python tools/tasks/libero_camera_gate.py --suite libero_goal
    pixi run mojo run -I . examples/tasks/libero_camera_gate.mojo \
        references/libero_demos/_camera/libero_goal/index.txt

The Python half writes three columns and this writes the fourth. Read its
header for what each one isolates; the short version is that
RECORDED-vs-THEIRS is the renderer-and-version floor no work of ours can
beat, THEIRS-vs-OURS-MJ is what our composed scene costs, and what this file
prints — OURS-RT against both — is what our SHADING costs with the scene held
fixed.

⚠⚠ IT RENDERS ON THE HOST, ONE LANE, AND THAT IS THE GATE'S SHAPE RATHER
THAN A LIMITATION. `render_lane_cpu` calls the same `render_pixel` the
batched kernel calls, so this measures the tracer and not a copy of it.
128x128 over a LIBERO scene takes a few seconds a frame.

⚠ GEOM GROUP 1 ONLY, and it is `build_visual_model`'s argument rather than a
runtime mask: robosuite renders with `render_collision_mesh=False`. Rendering
MuJoCo's default group set instead scores 10.8 dB against the recording where
the visual set scores 43.9.

⚠ SHADOWS ARE OFF, AND THE MODEL SAYS SO. Both LIBERO arena lights carry
`castshadow="false"` and a headlight never casts, so the shadow ray would be
compiled into every pixel and reached by none. `SHADOWS=False` is the
faithful setting here, not a shortcut.
"""

from std.sys import argv
from std.math import log10, sqrt

from mojo_rl.math3d import Vec3
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE, BODY_IDX_POS_X, BODY_IDX_QUAT_X, BODY_IDX_QUAT_W,
    MAX_GPU_CAMERAS, MODEL_CAM_SIZE, CAM_IDX_ACTIVE, CAM_IDX_FOVY,
    CAM_IDX_POS_X, CAM_IDX_QUAT_X, CAM_IDX_QUAT_W,
)
from mojo_rl.physics3d.raytrace.visual import build_visual_model
from mojo_rl.physics3d.raytrace.host_render import render_lane_cpu
from mojo_rl.physics3d.raytrace.camera import camera_world_frame, camera_pixel_ray
from mojo_rl.physics3d.fields.rt_layout import DYN1, DYN2, rl1, rl2
from mojo_rl.tasks.spec import load_family
from mojo_rl.tasks.family import scene_path
from mojo_rl.tasks.libero_goal_xml import LIBERO_GOAL_MAX_CONTACTS


comptime DT = DType.float64
comptime VISUAL_GROUP_MASK: Int = 1 << 1


def _floats(s: String) raises -> List[Float64]:
    var out = List[Float64]()
    var toks = s.split(" ")
    for k in range(len(toks)):
        var t = String(String(toks[k]).strip())
        if t.byte_length() > 0:
            out.append(Float64(t))
    return out^


def _read_bytes(path: String) raises -> List[UInt8]:
    with open(path, "r") as fh:
        return fh.read_bytes()


def _pad(s: String, n: Int) -> String:
    var out = s
    while out.byte_length() < n:
        out += " "
    return out^


def _to_byte(x: Float64) -> Int:
    var v = Int(x * 255.0 + 0.5)
    if v < 0:
        return 0
    return 255 if v > 255 else v


@fieldwise_init
struct Score(Copyable, ImplicitlyCopyable, Movable):
    var psnr: Float64
    var within8: Float64


def _score(
    ours: List[Scalar[DT]], want: List[UInt8], width: Int, height: Int
) -> Score:
    """PSNR in dB and the fraction of channels within 8 of 255.

    ⚠ THE COMPARISON IS IN BYTES, both sides. Ours is a float in [0, 1] and
    the reference is what a PNG held; converting the reference UP to float
    would compare our rounding against no rounding and flatter us by a
    fraction of a dB on every pixel.

    ⚠⚠ THE REFERENCE IS READ BOTTOM ROW FIRST, AND OUR IMAGE IS NOT FLIPPED.
    LIBERO's recorded frames are stored in OpenGL's own order — row 0 is the
    BOTTOM of the scene — because robosuite hands `mjr_readPixels`' buffer
    through unchanged, and nothing downstream turns it up the right way: the
    policies train on it as it lies. Our tracer is upright (row 0 the top,
    `camera_pixel_ray`), and MuJoCo's `Renderer.render()` is upright too —
    measured on `arena_frontview`, where the unflipped render stands the
    robot on the table and the flipped one hangs it from the ceiling. So the
    flip belongs HERE, at the comparison, and not in the renderer: writing it
    into `camera_pixel_ray` would turn every other camera in this tree upside
    down to make one dataset's storage order come out right.
    """
    var se = 0.0
    var near = 0
    var n = width * height
    for i in range(n * 3):
        var p = i // 3
        var c = i - p * 3
        var row = p // width
        var col = p - row * width
        var j = ((height - 1 - row) * width + col) * 3 + c
        var a = Float64(_to_byte(Float64(ours[i])))
        var b = Float64(Int(want[j]))
        var e = a - b
        se += e * e
        if e < 0:
            e = -e
        if e <= 8.0:
            near += 1
    var mse = se / Float64(n * 3)
    var p = 99.0 if mse == 0.0 else 10.0 * log10(255.0 * 255.0 / mse)
    return Score(p, Float64(near) / Float64(n * 3))


def main() raises:
    var a = argv()
    if len(a) < 2:
        raise Error(
            "usage: libero_camera_gate.mojo <index.txt>   (written by"
            " tools/tasks/libero_camera_gate.py)"
        )
    var index_path = String(a[1])
    var index_text: String
    try:
        with open(index_path, "r") as fh:
            index_text = fh.read()
    except e:
        # ⚠ SKIPPED, LOUDLY, AND NOT A PASS — the demonstrations are ~6 GB a
        # suite and gitignored.
        print("  SKIPPED: no index at", index_path)
        print("  Fetch the suite's demos into references/libero_demos/<suite>/")
        print("  and run  pixi run libero-camera-gate --suite <suite>")
        print("=== SKIPPED (no demonstrations — this is not a pass) ===")
        return

    var names = List[String]()
    var dumps = List[String]()
    var lines = index_text.split("\n")
    for i in range(len(lines)):
        var l = String(String(lines[i]).strip())
        if l.byte_length() == 0:
            continue
        var parts = l.split(" ")
        names.append(String(parts[0]))
        dumps.append(String(parts[1]))
    if len(names) == 0:
        raise Error("empty index: " + index_path)

    var suite = String(names[0])
    var cut = suite.find("__")
    if cut < 0:
        raise Error("task name without a suite prefix: " + suite)
    var suite_cut = String(suite[byte=0:cut])
    suite = suite_cut^

    var f = load_family("mojo_rl/tasks/families/" + suite + ".family")
    var fmd = parse_model_runtime(scene_path(f))
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

    print("=" * 78)
    print("OUR tracer against LIBERO's own pixels —", suite)
    print("=" * 78)
    print("  scene:", scene_path(f), "| nq", nq, "nv", nv,
          "ngeom", dims.get_ngeom())

    var vis = build_visual_model[DT, DynDims](
        fmd, m, group_mask=VISUAL_GROUP_MASK
    )
    print("  " + vis.describe())

    # The camera the family's arena declares under LIBERO's own name.
    var cam = -1
    for i in range(len(fmd.camera_names)):
        if String(fmd.camera_names[i]) == "arena_agentview":
            cam = i
    if cam < 0:
        raise Error(
            "no camera 'arena_agentview' in " + scene_path(f)
            + " — LIBERO's `_setup_camera` pose is written into the arena by"
            " tools/tasks/gen_libero_arenas.py; is the arena regenerated?"
        )
    var cb = cam * MODEL_CAM_SIZE
    print("  camera: arena_agentview (index", cam, ") fovy",
          Float64(m.cameras.data[cb + CAM_IDX_FOVY]),
          "pos", Float64(m.cameras.data[cb + CAM_IDX_POS_X]),
          Float64(m.cameras.data[cb + CAM_IDX_POS_X + 1]),
          Float64(m.cameras.data[cb + CAM_IDX_POS_X + 2]),
          "quat(xyzw)", Float64(m.cameras.data[cb + CAM_IDX_QUAT_X]),
          Float64(m.cameras.data[cb + CAM_IDX_QUAT_X + 1]),
          Float64(m.cameras.data[cb + CAM_IDX_QUAT_X + 2]),
          Float64(m.cameras.data[cb + CAM_IDX_QUAT_W]))
    print()
    print("  " + _pad(String("task"), 50)
          + "  OURS-RT|REC     OURS-RT|OURS-MJ")
    print("  " + _pad(String(""), 50) + "  psnr  within8   psnr  within8")

    var sum_rec = 0.0
    var sum_rec_w = 0.0
    var sum_mj = 0.0
    var sum_mj_w = 0.0
    var sum_seg = 0.0
    var sum_refl = 0.0
    var worst = 1e30
    var worst_name = String("")
    for ti in range(len(names)):
        var dump: String
        with open(dumps[ti], "r") as fh:
            dump = fh.read()
        var dl = dump.split("\n")
        var width = 0
        var height = 0
        var ref_path = String("")
        var mj_path = String("")
        var qp = List[Float64]()
        var qv = List[Float64]()
        for i in range(len(dl)):
            var l = String(String(dl[i]).strip())
            if l.startswith("WIDTH "):
                width = Int(String(l[byte=6:]))
            elif l.startswith("HEIGHT "):
                height = Int(String(l[byte=7:]))
            elif l.startswith("RECORDED "):
                ref_path = String(l[byte=9:])
            elif l.startswith("OURSMJ "):
                mj_path = String(l[byte=7:])
            elif l.startswith("QPOS "):
                qp = _floats(String(l[byte=5:]))
            elif l.startswith("QVEL "):
                qv = _floats(String(l[byte=5:]))
            elif l.startswith("FIX "):
                var toks = l.split(" ")
                var want = String(toks[1])
                var bi = -1
                for b in range(len(fmd.body_names)):
                    if String(fmd.body_names[b]) == want:
                        bi = b
                if bi <= 0:
                    raise Error("no body '" + want + "' in the scene")
                var o = bi * MODEL_BODY_SIZE
                m.bodies.data[o + BODY_IDX_POS_X + 0] = Scalar[DT](Float64(String(toks[2])))
                m.bodies.data[o + BODY_IDX_POS_X + 1] = Scalar[DT](Float64(String(toks[3])))
                m.bodies.data[o + BODY_IDX_POS_X + 2] = Scalar[DT](Float64(String(toks[4])))
                m.bodies.data[o + BODY_IDX_QUAT_W] = Scalar[DT](Float64(String(toks[5])))
                m.bodies.data[o + BODY_IDX_QUAT_X + 0] = Scalar[DT](Float64(String(toks[6])))
                m.bodies.data[o + BODY_IDX_QUAT_X + 1] = Scalar[DT](Float64(String(toks[7])))
                m.bodies.data[o + BODY_IDX_QUAT_X + 2] = Scalar[DT](Float64(String(toks[8])))
        if len(qp) != nq or len(qv) != nv:
            raise Error(
                dumps[ti] + ": " + String(len(qp)) + " qpos / "
                + String(len(qv)) + " qvel, scene wants " + String(nq)
                + " / " + String(nv)
            )
        for k in range(nq):
            d.qpos.data[k] = Scalar[DT](qp[k])
        for k in range(nv):
            d.qvel.data[k] = Scalar[DT](qv[k])
        forward_kinematics["cpu", DT, DynDims, 1](d, m)

        var rgb = List[Scalar[DT]]()
        var depth = List[Scalar[DT]]()
        var seg = List[Scalar[DT]]()
        var refl = List[Scalar[DT]]()
        render_lane_cpu[DT, DynDims, 1, False, True](
            d, m, vis, cam, 0, width, height,
            Vec3[DT](0.60, 0.72, 0.90), rgb, depth, seg, refl,
        )
        var npix = width * height
        var recorded = _read_bytes(ref_path)
        var mj = _read_bytes(mj_path)
        if len(recorded) != npix * 3 or len(mj) != npix * 3:
            raise Error(
                "reference " + ref_path + " is " + String(len(recorded))
                + " bytes; " + String(width) + "x" + String(height)
                + "x3 is " + String(npix * 3)
            )
        # ⚠ WRITTEN EVERY RUN, beside the dump the Python leg wrote. A PSNR
        # says how far off we are and never which way; `--png` on the Python
        # leg turns these three files into one strip you can look at, and
        # the first two defects L5 found (a missing robot, an unlit scene)
        # were both obvious there and invisible in the number.
        var raw = List[UInt8](length=npix * 3, fill=UInt8(0))
        for i in range(npix * 3):
            raw[i] = UInt8(_to_byte(Float64(rgb[i])))
        var stem = String(ref_path[byte = 0 : ref_path.byte_length() - 4])
        with open(stem + ".ours.rgb", "w") as fh:
            fh.write_bytes(raw)
        # ⚠ THE SEGMENTATION IS THE COLOUR-BLIND CHECK. A PSNR mixes geometry
        # and shading into one scalar; the geom id at each pixel says whether
        # the RAY is right independently of what colour we then painted. Three
        # of the five defects `ray_model` was falsified against left the
        # distance untouched and showed only as a different geom.
        var sraw = List[UInt8](length=npix, fill=UInt8(0))
        for i in range(npix):
            var g = Int(Float64(seg[i]))
            sraw[i] = UInt8(0 if g < 0 else (g + 1) % 256)
        with open(stem + ".ours.seg", "w") as fh:
            fh.write_bytes(sraw)
        # ⚠ WHAT THE MIRROR SHOWS, per pixel — the colour-blind check for the
        # SECOND ray. The reflection pass is additive, so a mirror that
        # reflects the wrong surface reads as a brightness error and nothing
        # else; this says which surface it was.
        var rraw = List[UInt8](length=npix, fill=UInt8(0))
        var filled = 0
        for i in range(npix):
            var g = Int(Float64(refl[i]))
            rraw[i] = UInt8(0 if g < 0 else (g + 1) % 256)
            if g >= 0:
                filled += 1
        sum_refl += Float64(filled)
        with open(stem + ".ours.refl", "w") as fh:
            fh.write_bytes(rraw)

        # ⚠⚠ THE COLOUR-BLIND COLUMN, AND THE ONE THAT DECIDES WHERE A GAP
        # LIVES. A PSNR folds geometry and shading into one scalar. The geom
        # id at each pixel does not: it says whether the RAY is right —
        # camera pose, mesh soup, BVH, group filter, fixture patch — with no
        # opinion about what colour was painted afterwards. On libero_goal it
        # is 0.9998, so every dB of the colour gap below is SHADING and none
        # of it is geometry.
        var mseg = _read_bytes(
            String(mj_path[byte = 0 : mj_path.byte_length() - 4]) + ".seg"
        )
        var agree = 0
        for i in range(npix):
            if Int(sraw[i]) == Int(mseg[i]):
                agree += 1
        sum_seg += Float64(agree) / Float64(npix)

        var s1 = _score(rgb, recorded, width, height)
        var s2 = _score(rgb, mj, width, height)
        sum_rec += s1.psnr
        sum_rec_w += s1.within8
        sum_mj += s2.psnr
        sum_mj_w += s2.within8
        if s1.psnr < worst:
            worst = s1.psnr
            worst_name = String(names[ti])
        var short = String(names[ti])
        var sc = short.find("__")
        if sc >= 0:
            var short_cut = String(short[byte = sc + 2 :])
            short = short_cut^
        print("  " + _pad(short, 50), s1.psnr, s1.within8, " ", s2.psnr,
              s2.within8)

    var n = Float64(len(names))
    print()
    print(_pad(String("  MEAN"), 52), sum_rec / n, sum_rec_w / n, " ",
          sum_mj / n, sum_mj_w / n)
    print("  worst vs the recording:", worst, "dB —", worst_name)
    print("  segmentation agreement vs MuJoCo on OUR scene:", sum_seg / n)
    print("  pixels the mirror fills, mean per frame:", sum_refl / n,
          "(mirror geom", vis.mirror, ")")
    if vis.mirror >= 0 and sum_refl <= 0.0:
        raise Error(
            "the scene declares a reflective geom (visual index "
            + String(vis.mirror) + ") and the mirror reflected NOTHING in any"
            " frame. The reflection pass is dead — a `REFLECT=False`"
            " instantiation, an `isBehind` test with the wrong sign, or a"
            " +Z face nothing hits. It would read as a uniformly dark geom"
            " and not as an error."
        )
    if sum_seg / n < 0.99:
        raise Error(
            "the tracer and MuJoCo disagree about WHICH GEOM is at "
            + String(100.0 * (1.0 - sum_seg / n))
            + "% of pixels. That is geometry, not shading: a camera pose, a"
            " mesh soup, a BVH, the group filter or the fixture patch. Fix it"
            " before reading the colour columns — a PSNR cannot tell you"
            " which of the two it is measuring."
        )
    print()
    print("=== the gap is a NUMBER; see the assessment's L5 for what it"
          " bounds ===")
