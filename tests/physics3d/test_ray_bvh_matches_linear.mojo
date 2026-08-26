"""The mesh BVH must not move the answer — only the time.

    pixi run mojo run -I . tests/physics3d/test_ray_bvh_matches_linear.mojo

⚠⚠ THIS IS THE WHOLE CONTRACT OF AN ACCELERATION STRUCTURE, AND IT IS THE ONLY
THING WORTH GATING. A BVH culls a node when the ray provably misses its box, so
the `ray_triangle` calls that survive are a SUBSET of the linear sweep's and the
nearest of a subset that still contains the winner is the same winner. Any
difference at all — one pixel, one ulp — means a box is too small, a slab test
rejects something it should not, or the escape indices are wrong. So the
assertion is EXACT EQUALITY, not a tolerance: there is no float error budget to
spend here, because both legs run the SAME `ray_triangle` on the SAME triangle.

⚠⚠ AND IT IS GATED ON THE DEVICE TOO, WHICH IS THE POINT OF THE STACKLESS WALK.
The reference walks its tree with `int stack[mjMAXTREEDEPTH]` — a per-thread
array indexed by a RUNTIME value, the read that has been silently wrong on Metal
four times in this engine. `ray_mesh_bvh` has no such array; it stores the tree
in pre-order and jumps to an ESCAPE index on a miss. `test_gpu_*` below is what
says the substitution actually holds on a GPU, and it compares GPU-BVH against
GPU-LINEAR so a float32-vs-float64 difference cannot be mistaken for a
traversal defect.

HOW THE CONTROL LEG IS SELECTED
===============================
`ray_model` dispatches on `MESH_META_IDX_BVHNUM`: non-zero takes the tree, zero
takes the sweep. So the control is the SAME BINARY with that one field zeroed —
no second code path to keep in step, no comptime flag, and the ablation is data
rather than a rebuild. The counts are restored afterwards, and
`test_the_ablation_is_real` checks the zeroing actually changes what runs, since
"both legs agree" and "the ablation did nothing" print the same 0.

⚠ THE SCENE CARRIES TWO MESHES ON PURPOSE, and one of them is a PRISM. A prism's
caps and side faces are FLAT — their triangle AABBs are degenerate on an axis,
which is the case MuJoCo inflates by `mjEPS` = 1e-14 and which at float32 is no
inflation at all. A flat box fails a STRICT `tmin < tmax`, so a ray in the plane
of such a face would be culled from a hit it makes. That is the defect this
scene is shaped to catch.
"""

from std.math import abs
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true, assert_equal, TestSuite

from mojo_rl.physics3d.fields import Data, Model, Dims, init_hfield_data
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.fields_build import build_model_fields_from_flat
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.dynamics.subtree_com import compute_subtree_com
from mojo_rl.physics3d.gpu.constants import (
    MAX_GPU_MESHES,
    MODEL_MESH_META_SIZE,
    MESH_META_IDX_TRINUM,
    MESH_META_IDX_BVHADR,
    MESH_META_IDX_BVHNUM,
)
from mojo_rl.physics3d.raytrace import (
    BatchedCameraRenderer,
    RGB_CHANNELS,
    init_camera_reference,
)

comptime GT = DType.float32
comptime BATCH = 2
comptime W = 64
comptime H = 48
comptime NPIX = W * H

comptime NQ = 7
comptime NV = 6
comptime NBODY = 2
comptime NJOINT = 1
comptime NGEOM = 8
# 396 (prism) + 28 (notch) triangles, so the deeper tree is 791 nodes at
# depth 9 — enough that a subtree's escape index is not trivially its
# neighbour, which is the arithmetic this file is really checking.
comptime NTRI = 512

comptime GD = Dims[
    nq=NQ,
    nv=NV,
    nbody=NBODY,
    njoint=NJOINT,
    ngeom=NGEOM,
    nsite=0,
    max_contacts=16,
    nmesh_verts=512,
    nmesh_tri=NTRI,
    nhfield_data=0,
]

comptime Renderer = BatchedCameraRenderer[GT, GD, BATCH, W, H]

comptime SCENE = String(
    """
<mujoco model="bvh parity gate">
  <asset>
    <!-- ⚠ SCALED x5. The assets are 10 cm across; at a camera distance that
         also keeps a floor and a couple of primitives in frame they would
         cover ~19 of 6 144 pixels, and this file's parity numbers would be
         about almost nothing. The vacuity guard below caught exactly that. -->
    <mesh name="prism" file="tests/physics3d/assets/ngon100_prism.stl" scale="5 5 5"/>
    <mesh name="notch" file="tests/physics3d/assets/notch.stl" scale="5 5 5"/>
  </asset>
  <worldbody>
    <camera name="cam" pos="0 -1.30 0.20" xyaxes="1 0 0 0 0.28 0.96" fovy="72"/>
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 -0.45" rgba="0.35 0.4 0.32 1"/>
    <geom name="ball" type="sphere" size="0.12" pos="0.62 0.30 -0.30" rgba="0.85 0.2 0.2 1"/>
    <geom name="brick" type="box" size="0.09 0.07 0.08" pos="-0.68 0.25 -0.34" euler="20 -35 15" rgba="0.2 0.4 0.85 1"/>
    <geom name="prism_flat" type="mesh" mesh="prism" pos="-0.30 0.10 -0.28" rgba="0.95 0.5 0.1 1"/>
    <geom name="prism_tilt" type="mesh" mesh="prism" pos="0.34 0.05 -0.10" euler="63 -17 29" rgba="0.1 0.8 0.5 1"/>
    <geom name="notch_a" type="mesh" mesh="notch" pos="-0.34 -0.30 0.14" euler="15 -25 40" rgba="0.7 0.3 0.8 1"/>
    <body name="rider" pos="0.06 -0.25 0.36">
      <freejoint/>
      <geom name="notch_b" type="mesh" mesh="notch" euler="-40 12 8" rgba="0.15 0.75 0.95 1"/>
    </body>
  </worldbody>
</mujoco>
"""
)

comptime CAM = 0


def _build(mut m: Model[GT, GD], mut d: Data[GT, GD, BATCH]) raises:
    """Parse, build, place the two lanes DIFFERENTLY, run FK + subtree CoM."""
    var fmd = parse_xml_full(SCENE, String("."))
    build_model_fields_from_flat[GT](fmd, m)
    init_hfield_data(d, m)
    d.qpos.data[0 * NQ + 0] = Scalar[GT](0.06)
    d.qpos.data[0 * NQ + 1] = Scalar[GT](-0.25)
    d.qpos.data[0 * NQ + 2] = Scalar[GT](0.36)
    d.qpos.data[0 * NQ + 3] = Scalar[GT](1.0)
    d.qpos.data[1 * NQ + 0] = Scalar[GT](-0.16)
    d.qpos.data[1 * NQ + 1] = Scalar[GT](-0.34)
    d.qpos.data[1 * NQ + 2] = Scalar[GT](0.20)
    d.qpos.data[1 * NQ + 3] = Scalar[GT](1.0)
    forward_kinematics["cpu", GT, GD, BATCH](d, m)
    compute_subtree_com["cpu", GT, GD, BATCH](d, m)
    init_camera_reference(d, m)


def _bvhnum_backup(m: Model[GT, GD]) -> List[Scalar[GT]]:
    var out = List[Scalar[GT]]()
    for i in range(MAX_GPU_MESHES):
        out.append(
            m.mesh_meta.data[i * MODEL_MESH_META_SIZE + MESH_META_IDX_BVHNUM]
        )
    return out.copy()


def _bvhnum_set(mut m: Model[GT, GD], vals: List[Scalar[GT]]):
    for i in range(MAX_GPU_MESHES):
        m.mesh_meta.data[
            i * MODEL_MESH_META_SIZE + MESH_META_IDX_BVHNUM
        ] = vals[i]


def _bvhnum_zero(mut m: Model[GT, GD]):
    for i in range(MAX_GPU_MESHES):
        m.mesh_meta.data[
            i * MODEL_MESH_META_SIZE + MESH_META_IDX_BVHNUM
        ] = Scalar[GT](0)


def _mesh_pixels(seg: List[Scalar[GT]], m: Model[GT, GD]) -> Int:
    """Pixels whose winning geom is one of the four MESH geoms (3..7).

    Counted from the scene's own declaration order rather than from the model,
    because the number this guards — "the meshes are actually in frame" — is a
    property of the SCENE and reading it back out of the thing under test is
    how a vacuous gate is written.
    """
    var n = 0
    for i in range(len(seg)):
        var g = Int(seg[i])
        if g >= 3 and g <= 6:
            n += 1
    return n


def _diff(
    a_rgb: List[Scalar[GT]], a_dep: List[Scalar[GT]], a_seg: List[Scalar[GT]],
    b_rgb: List[Scalar[GT]], b_dep: List[Scalar[GT]], b_seg: List[Scalar[GT]],
) -> Tuple[Int, Int, Int]:
    """(pixels differing in seg, in depth, in any colour channel) — EXACTLY."""
    var ds = 0
    var dd = 0
    var dc = 0
    for i in range(len(a_seg)):
        if Int(a_seg[i]) != Int(b_seg[i]):
            ds += 1
        if a_dep[i] != b_dep[i]:
            dd += 1
        var c = False
        for k in range(RGB_CHANNELS):
            if a_rgb[i * RGB_CHANNELS + k] != b_rgb[i * RGB_CHANNELS + k]:
                c = True
        if c:
            dc += 1
    return (ds, dd, dc)


def _render_cpu_both(
    mut m: Model[GT, GD], mut d: Data[GT, GD, BATCH], ctx: DeviceContext
) raises -> Tuple[Int, Int, Int, Int]:
    """Render with the tree, then with it ablated. Returns the three diff
    counts and the number of MESH pixels the tree leg produced."""
    var r = Renderer(ctx, m, CAM)

    var a_rgb = List[Scalar[GT]]()
    var a_dep = List[Scalar[GT]]()
    var a_seg = List[Scalar[GT]]()
    r.render_cpu(d, m, a_rgb, a_dep, a_seg)
    var mesh_px = _mesh_pixels(a_seg, m)

    var keep = _bvhnum_backup(m)
    _bvhnum_zero(m)
    var b_rgb = List[Scalar[GT]]()
    var b_dep = List[Scalar[GT]]()
    var b_seg = List[Scalar[GT]]()
    r.render_cpu(d, m, b_rgb, b_dep, b_seg)
    _bvhnum_set(m, keep)

    var dif = _diff(a_rgb, a_dep, a_seg, b_rgb, b_dep, b_seg)
    return (dif[0], dif[1], dif[2], mesh_px)


def _render_gpu(
    ctx: DeviceContext,
    mut r: Renderer,
    mut d: Data[GT, GD, BATCH],
    mut m: Model[GT, GD],
    mut rgb: List[Scalar[GT]],
    mut dep: List[Scalar[GT]],
    mut seg: List[Scalar[GT]],
) raises:
    """Launch, synchronise, and bring the three buffers back as plain lists."""
    r.render(ctx, d, m)
    ctx.synchronize()
    var n = BATCH * NPIX
    var h_rgb = ctx.enqueue_create_host_buffer[GT](n * RGB_CHANNELS)
    var h_dep = ctx.enqueue_create_host_buffer[GT](n)
    var h_seg = ctx.enqueue_create_host_buffer[GT](n)
    ctx.enqueue_copy(h_rgb, r.rgb)
    ctx.enqueue_copy(h_dep, r.depth)
    ctx.enqueue_copy(h_seg, r.seg)
    ctx.synchronize()
    rgb = List[Scalar[GT]](length=n * RGB_CHANNELS, fill=Scalar[GT](0))
    dep = List[Scalar[GT]](length=n, fill=Scalar[GT](0))
    seg = List[Scalar[GT]](length=n, fill=Scalar[GT](0))
    for i in range(n):
        dep[i] = h_dep[i]
        seg[i] = h_seg[i]
        for c in range(RGB_CHANNELS):
            rgb[i * RGB_CHANNELS + c] = h_rgb[i * RGB_CHANNELS + c]


def test_the_tree_was_built_at_all() raises:
    """`2n - 1` nodes per mesh, and a `BVHADR` past the triangles.

    ⚠ THE FIRST THING TO CHECK, BECAUSE ZERO IS THE FALLBACK. `ray_model`
    takes the linear leg on `BVHNUM == 0` and returns the right answer while
    doing so — so a parser that silently built no tree would pass every parity
    assertion in this file and every picture would be correct. This is the
    only test here that would notice.
    """
    var m = Model[GT, GD]()
    var d = Data[GT, GD, BATCH]()
    _build(m, d)

    var trees = 0
    var total_tri = 0
    var total_nodes = 0
    for i in range(MAX_GPU_MESHES):
        var b = i * MODEL_MESH_META_SIZE
        var ntri = Int(m.mesh_meta.data[b + MESH_META_IDX_TRINUM])
        var nn = Int(m.mesh_meta.data[b + MESH_META_IDX_BVHNUM])
        var adr = Int(m.mesh_meta.data[b + MESH_META_IDX_BVHADR])
        if ntri == 0:
            assert_equal(
                nn, 0,
                "a mesh with no triangles was given a tree — the traversal"
                " has no case for a node with no leaf under it",
            )
            continue
        trees += 1
        total_tri += ntri
        total_nodes += nn
        assert_equal(
            nn, 2 * ntri - 1,
            String("mesh ") + String(i) + " has " + String(ntri)
            + " triangles and " + String(nn) + " nodes; a"
            " one-triangle-per-leaf tree has exactly 2n-1",
        )
        assert_true(
            adr >= total_tri or adr >= ntri,
            "a BVH node overlaps the triangle region of the arena",
        )
    print(
        "  ", trees, " trees, ", total_tri, " triangles, ",
        total_nodes, " nodes",
    )
    assert_true(
        trees >= 2,
        "fewer than two meshes carry a tree — the scene declares two and this"
        " file's parity numbers would be about one of them",
    )
    assert_true(
        total_tri > 300,
        String("only ") + String(total_tri) + " triangles in the whole model;"
        " the prism should contribute 396 on its own, so the soup budget or"
        " the asset is not what this file assumes",
    )


def test_the_ablation_is_real() raises:
    """Zeroing `BVHNUM` must change what RUNS, or the parity test is vacuous.

    ⚠ "BOTH LEGS AGREE" AND "THE ABLATION DID NOTHING" PRINT THE SAME 0. The
    control here is the one thing the two legs cannot share: their SPEED. The
    linear sweep over 396 triangles against a depth-9 tree is not a subtle
    difference, so a wall-clock ratio well above 1 is evidence the second
    render really took the other branch.
    """
    from std.time import perf_counter_ns

    var m = Model[GT, GD]()
    var d = Data[GT, GD, BATCH]()
    _build(m, d)
    var ctx = DeviceContext()
    var r = Renderer(ctx, m, CAM)

    var rgb = List[Scalar[GT]]()
    var dep = List[Scalar[GT]]()
    var seg = List[Scalar[GT]]()

    r.render_cpu(d, m, rgb, dep, seg)  # warm
    var t0 = perf_counter_ns()
    r.render_cpu(d, m, rgb, dep, seg)
    var tree_ms = Float64(perf_counter_ns() - t0) / 1.0e6

    var keep = _bvhnum_backup(m)
    _bvhnum_zero(m)
    var t1 = perf_counter_ns()
    r.render_cpu(d, m, rgb, dep, seg)
    var lin_ms = Float64(perf_counter_ns() - t1) / 1.0e6
    _bvhnum_set(m, keep)

    print(
        "  tree ", Int(tree_ms), " ms vs linear ", Int(lin_ms),
        " ms — ", Float64(Int(lin_ms / tree_ms * 100.0)) / 100.0, "x",
    )
    assert_true(
        lin_ms > tree_ms * 1.3,
        "zeroing BVHNUM did not make the render measurably slower, so the"
        " control leg is probably running the same code as the test leg and"
        " the parity assertions below are vacuous",
    )


def test_cpu_bvh_matches_the_linear_sweep() raises:
    """The parity assertion, on the host, over every channel.

    Colour, planar depth and the segmentation id, compared EXACTLY. `seg` is
    the sharpest of the three — three of the five defects `ray_model` was
    falsified against left the distance untouched and showed only as a
    different geom.
    """
    var m = Model[GT, GD]()
    var d = Data[GT, GD, BATCH]()
    _build(m, d)
    var ctx = DeviceContext()
    var out = _render_cpu_both(m, d, ctx)
    var total = BATCH * NPIX
    print(
        "  compared ", total, " pixels (", out[3], " on meshes): ",
        out[0], " seg, ", out[1], " depth, ", out[2], " colour differ",
    )
    assert_true(
        out[3] > total // 40,
        String("only ") + String(out[3]) + " of " + String(total)
        + " pixels land on a mesh — the meshes are barely in frame, so a"
        " BVH defect could pass this file unnoticed",
    )
    assert_equal(out[0], 0, "the BVH and the linear sweep disagree on WHICH"
                            " geom a pixel sees")
    assert_equal(out[1], 0, "the BVH and the linear sweep disagree on DEPTH")
    assert_equal(out[2], 0, "the BVH and the linear sweep disagree on COLOUR")


def test_gpu_bvh_matches_the_linear_sweep() raises:
    """The same parity, on the device, where the stack would have lived.

    ⚠ GPU AGAINST GPU, NOT GPU AGAINST CPU. `test_camera_render_gpu_vs_cpu`
    already owns the cross-device comparison; running it again here would put
    a float32 rounding difference and a traversal defect in the same number.
    Both legs below are the same kernel on the same device with one `Model`
    field changed, so the ONLY thing that can differ is the walk.
    """
    comptime if not has_accelerator():
        print("  no accelerator — skipped")
        return

    var m = Model[GT, GD]()
    var d = Data[GT, GD, BATCH]()
    _build(m, d)
    var ctx = DeviceContext()
    m.upload_all(ctx)
    d.upload_all(ctx)
    ctx.synchronize()

    var r = Renderer(ctx, m, CAM)
    var a_rgb = List[Scalar[GT]]()
    var a_dep = List[Scalar[GT]]()
    var a_seg = List[Scalar[GT]]()
    _render_gpu(ctx, r, d, m, a_rgb, a_dep, a_seg)

    var keep = _bvhnum_backup(m)
    _bvhnum_zero(m)
    m.upload_all(ctx)
    ctx.synchronize()
    var b_rgb = List[Scalar[GT]]()
    var b_dep = List[Scalar[GT]]()
    var b_seg = List[Scalar[GT]]()
    _render_gpu(ctx, r, d, m, b_rgb, b_dep, b_seg)
    _bvhnum_set(m, keep)
    m.upload_all(ctx)
    ctx.synchronize()

    var mesh_px = _mesh_pixels(a_seg, m)
    var dif = _diff(a_rgb, a_dep, a_seg, b_rgb, b_dep, b_seg)
    var total = BATCH * NPIX
    print(
        "  compared ", total, " pixels (", mesh_px, " on meshes): ",
        dif[0], " seg, ", dif[1], " depth, ", dif[2], " colour differ",
    )
    assert_true(
        mesh_px > total // 40,
        "the meshes are barely in frame on the device leg either",
    )
    assert_equal(dif[0], 0, "the GPU BVH walk sees a DIFFERENT geom than the"
                            " GPU linear sweep — the classic shape of a"
                            " runtime-indexed per-thread read")
    assert_equal(dif[1], 0, "the GPU BVH walk disagrees on DEPTH")
    assert_equal(dif[2], 0, "the GPU BVH walk disagrees on COLOUR")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
