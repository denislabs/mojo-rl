"""One pixel of a batched camera observation — `render.py`'s megakernel body.

The whole of the tracer's per-pixel work, as a plain function over
`LayoutTensor`s and an `env` index, so the host leg and the device leg are the
SAME code. `physics3d/ray` is built on that discipline and
`tests/physics3d/test_ray_model_gpu_vs_cpu.mojo` exists to enforce it; this
module inherits both, and `test_camera_render_gpu_vs_cpu.mojo` is its half of
the bargain.

⚠⚠ NO PER-THREAD ARRAY APPEARS HERE, AND THAT IS A REQUIREMENT, NOT A HABIT.
An `InlineArray` indexed by a runtime value reads back silently wrong on Metal
and has done so four times in this engine (`87960e10` is the most recent). A
pixel's three colour channels are three named scalars in a `Vec3`, not a
three-element array, for exactly that reason.

WHY A TRACER AND NOT `Renderer3D`
=================================
`Renderer3D` is a rasteriser over CPU-built draw commands: N environments cost
N sequential scenes, a window, a swapchain and a shadow pass with its own
depth-map resolution and bias. This is one kernel over (env, pixel) that reads
the batched `Data` in place. The assessment
(`docs/DM_CONTROL_AND_CAMERA_ASSESSMENT_2026_08_24.md` §6) is blunt about which
one belongs where: **the SDL pipeline stays the VIEWER, the tracer is the
OBSERVATION path**, and MuJoCo itself keeps the same split. Per frame the
tracer is the more expensive of the two — it is chosen because it batches and
because it reuses collision geometry that is already on the device, not because
it is faster at one image.
"""

from layout import Layout, LayoutTensor

from mojo_rl.math3d import Vec3 as Vec3Generic

from ..gpu.constants import MODEL_GEOM_RGBA_SIZE
from ..ray.model import ray_model
from .camera import CameraFrame, camera_pixel_ray
from .shade import ambient_term, directional_light_term, _clamp01


@fieldwise_init
struct PixelHit[DTYPE: DType](Copyable, Movable):
    """What one pixel produced: colour, planar depth, and what was hit."""

    var rgb: Vec3Generic[Self.DTYPE]
    """Linear colour in [0, 1]. The background colour on a miss."""

    var depth: Scalar[Self.DTYPE]
    """PLANAR depth — the hit distance projected onto the optical axis, which
    is what a depth camera reports and what `render.py` writes.

    ⚠ NOT THE RAY PARAMETER. `dist * (-ray_dir_local.z)` differs from `dist`
    by `cos(theta)` off-axis; at a 90 deg fovy the corner pixels differ by
    ~30%. A policy trained on one and deployed against the other sees a
    barrel-distorted world.

    ⚠ **0 MEANS NO HIT**, not "zero metres away". Nothing can be at zero
    distance from a camera that is not inside a geom, so the sentinel is
    unambiguous — but it is a SENTINEL, and a consumer normalising depth must
    handle it before dividing. Same contract, same reason, as
    `rangefinder`'s -1."""

    var geom: Int
    """The geom the primary ray hit, or -1 for background.

    ⚠ THIS IS THE SEGMENTATION CHANNEL AND IT IS FREE. `render.py` writes a
    whole `seg_data` buffer for it; here it falls out of the same `RayHit`.
    It is also what makes a colour-blind gate possible: three of the five
    defects `ray_model` was falsified against left the distance untouched and
    showed only as a different geom."""


def render_pixel[
    DTYPE: DType,
    SHADOWS: Bool,
    L_GEOMS: Layout,
    L_RGBA: Layout,
    L_BODIES: Layout,
    L_XPOS: Layout,
    L_XQUAT: Layout,
    L_MESH_META: Layout,
    L_TRI: Layout,
    L_HF_META: Layout,
    L_HF: Layout,
](
    geoms: LayoutTensor[DTYPE, L_GEOMS, MutAnyOrigin],
    ngeom: Int,
    geom_rgba: LayoutTensor[DTYPE, L_RGBA, MutAnyOrigin],
    bodies: LayoutTensor[DTYPE, L_BODIES, MutAnyOrigin],
    xpos: LayoutTensor[DTYPE, L_XPOS, MutAnyOrigin],
    xquat: LayoutTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    env: Int,
    mesh_meta: LayoutTensor[DTYPE, L_MESH_META, MutAnyOrigin],
    mesh_tris: LayoutTensor[DTYPE, L_TRI, MutAnyOrigin],
    hfield_meta: LayoutTensor[DTYPE, L_HF_META, MutAnyOrigin],
    hfield_data: LayoutTensor[DTYPE, L_HF, MutAnyOrigin],
    hf_stride: Int,
    frame: CameraFrame[DTYPE],
    width: Int,
    height: Int,
    px: Int,
    py: Int,
    light_dir: Vec3Generic[DTYPE],
    background: Vec3Generic[DTYPE],
) -> PixelHit[DTYPE] where DTYPE.is_floating_point():
    """Primary ray, shade, and the two by-products.

    ⚠ `flg_static` IS LEFT AT ITS DEFAULT (statics INCLUDED) and no group is
    filtered. A camera sees the floor; a `geomgroup` mask is what MuJoCo's
    VIEWER uses to hide collision geometry from a HUMAN, and an observation
    should see what the robot's sensor would. Invisible geoms
    (`GEOM_IDX_RAY_VISIBLE`) are still skipped, which is the same rule that
    keeps a decoration out of a rangefinder.

    ⚠ NO `bodyexclude`. A wrist camera SHOULD see the gripper it is mounted
    on — that is most of what it is for. This is the opposite default from
    `rangefinder_site`, which excludes its own body because MuJoCo's sensor
    does.
    """
    var dir = camera_pixel_ray[DTYPE](frame, width, height, px, py)

    var hit = ray_model[DTYPE](
        geoms,
        ngeom,
        bodies,
        xpos,
        xquat,
        env,
        mesh_meta,
        mesh_tris,
        hfield_meta,
        hfield_data,
        hf_stride,
        frame.pos,
        dir,
    )

    if hit.geom < 0:
        return PixelHit[DTYPE](background, Scalar[DTYPE](0), -1)

    # PLANAR depth. `dir` is normalised and world-space, so the cosine to the
    # optical axis is `dot(dir, -zaxis)` — the same quantity the reference
    # spells `-ray_dir_local_cam[2]` in the camera's own frame.
    var cos_axis = -dir.dot(frame.zaxis)
    var depth = hit.t * cos_axis

    var gb = hit.geom * MODEL_GEOM_RGBA_SIZE
    var base = Vec3Generic[DTYPE](
        rebind[Scalar[DTYPE]](geom_rgba[gb + 0]),
        rebind[Scalar[DTYPE]](geom_rgba[gb + 1]),
        rebind[Scalar[DTYPE]](geom_rgba[gb + 2]),
    )

    var hitpoint = frame.pos + dir * hit.t

    var amb = ambient_term[DTYPE](hit.normal)
    var rgb = Vec3Generic[DTYPE](
        Scalar[DTYPE](0.5) * base.x * amb.x,
        Scalar[DTYPE](0.5) * base.y * amb.y,
        Scalar[DTYPE](0.5) * base.z * amb.z,
    )

    var lit = directional_light_term[DTYPE, SHADOWS](
        geoms,
        ngeom,
        bodies,
        xpos,
        xquat,
        env,
        mesh_meta,
        mesh_tris,
        hfield_meta,
        hfield_data,
        hf_stride,
        hit.normal,
        hitpoint,
        light_dir,
    )
    rgb = rgb + base * lit

    return PixelHit[DTYPE](
        Vec3Generic[DTYPE](
            _clamp01[DTYPE](rgb.x),
            _clamp01[DTYPE](rgb.y),
            _clamp01[DTYPE](rgb.z),
        ),
        depth,
        hit.geom,
    )
