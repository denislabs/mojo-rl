"""One pixel of a batched camera observation — `render.py`'s megakernel body.

The whole of the tracer's per-pixel work, as a plain function over
`LayoutTensor`s and an `env` index, so the host leg and the device leg are the
SAME code. `physics3d/ray` is built on that discipline and
`tests/physics3d/test_ray_model_gpu_vs_cpu.mojo` exists to enforce it; this
module inherits both, and `test_camera_render_gpu_vs_cpu.mojo` is its half of
the bargain.

⚠⚠ NO PER-THREAD ARRAY APPEARS HERE, AND THAT IS A REQUIREMENT, NOT A HABIT.
An `Array` indexed by a runtime value reads back silently wrong on Metal
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

from mojo_rl.math3d import Quat as QuatGeneric

from ..constants import GEOM_BOX
from ..gpu.constants import (
    GEOM_IDX_HALF_Z,
    GEOM_IDX_TYPE,
    GEOM_IDX_POS_X,
    GEOM_IDX_POS_Y,
    GEOM_IDX_POS_Z,
    GEOM_IDX_QUAT_X,
    GEOM_IDX_QUAT_Y,
    GEOM_IDX_QUAT_Z,
    GEOM_IDX_QUAT_W,
    GEOM_IDX_BODY,
)
from ..ray.model import RayHit, ray_model
from .camera import CameraFrame, camera_pixel_ray
from .appearance import (
    _clamp01,
    geom_uv,
    sample_texture,
    shade_lights,
    Texel,
)
from .visual_records import *


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

    var refl_geom: Int
    """What the MIRROR shows at this pixel, or -1 — no mirror here, or the
    reflected ray escaped.

    ⚠ THE COLOUR-BLIND CHECK FOR THE REFLECTION PASS, and it is not free
    decoration: our first reflection filled 100% of the mirror where the
    reference fills 76% of it, and the colour alone could not say whether
    that was a shading gain or a geometry error. `seg` answers that question
    for the primary ray; this answers it for the second one.
    """


@always_inline
def _geom_world_pose[
    DTYPE: DType, L_GEOMS: Layout, L_XPOS: Layout, L_XQUAT: Layout
](
    geoms: LayoutTensor[DTYPE, L_GEOMS, MutAnyOrigin],
    xpos: LayoutTensor[DTYPE, L_XPOS, MutAnyOrigin],
    xquat: LayoutTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    env: Int,
    g: Int,
) -> Tuple[Vec3Generic[DTYPE], QuatGeneric[DTYPE]] where (
    DTYPE.is_floating_point()
):
    """The geom's world pose, composed exactly as `ray_model` composes it.

    ⚠ RECOMPUTED RATHER THAN RETURNED BY THE HIT. A `RayHit` carrying a frame
    would carry it on every miss too, and a miss is the common case in a
    per-pixel loop over a whole scene.
    """
    var lp = Vec3Generic[DTYPE](
        rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_POS_X]),
        rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_POS_Y]),
        rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_POS_Z]),
    )
    var lq = QuatGeneric[DTYPE](
        rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_QUAT_W]),
        rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_QUAT_X]),
        rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_QUAT_Y]),
        rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_QUAT_Z]),
    )
    var body = Int(rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_BODY]))
    if body <= 0:
        return (lp, lq)
    # `Data.xquat` is packed (x, y, z, w); `Quat` takes (w, x, y, z).
    var bq = QuatGeneric[DTYPE](
        rebind[Scalar[DTYPE]](xquat[env, body * 4 + 3]),
        rebind[Scalar[DTYPE]](xquat[env, body * 4 + 0]),
        rebind[Scalar[DTYPE]](xquat[env, body * 4 + 1]),
        rebind[Scalar[DTYPE]](xquat[env, body * 4 + 2]),
    )
    var bp = Vec3Generic[DTYPE](
        rebind[Scalar[DTYPE]](xpos[env, body * 3 + 0]),
        rebind[Scalar[DTYPE]](xpos[env, body * 3 + 1]),
        rebind[Scalar[DTYPE]](xpos[env, body * 3 + 2]),
    )
    return (bp + bq.rotate_vec(lp), bq * lq)


def shade_hit[
    DTYPE: DType,
    SHADOWS: Bool,
    L_GEOMS: Layout,
    L_APP: Layout,
    L_BODIES: Layout,
    L_XPOS: Layout,
    L_XQUAT: Layout,
    L_MESH_META: Layout,
    L_TRI: Layout,
    L_UV: Layout,
    L_HF_META: Layout,
    L_HF: Layout,
    L_MAT: Layout,
    L_TEX: Layout,
    L_TEXELS: Layout,
    L_LIGHTS: Layout,
](
    geoms: LayoutTensor[DTYPE, L_GEOMS, MutAnyOrigin],
    ngeom: Int,
    appearance: LayoutTensor[DTYPE, L_APP, MutAnyOrigin],
    bodies: LayoutTensor[DTYPE, L_BODIES, MutAnyOrigin],
    xpos: LayoutTensor[DTYPE, L_XPOS, MutAnyOrigin],
    xquat: LayoutTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    env: Int,
    mesh_meta: LayoutTensor[DTYPE, L_MESH_META, MutAnyOrigin],
    mesh_tris: LayoutTensor[DTYPE, L_TRI, MutAnyOrigin],
    mesh_uv: LayoutTensor[DTYPE, L_UV, MutAnyOrigin],
    hfield_meta: LayoutTensor[DTYPE, L_HF_META, MutAnyOrigin],
    hfield_data: LayoutTensor[DTYPE, L_HF, MutAnyOrigin],
    hf_stride: Int,
    materials: LayoutTensor[DTYPE, L_MAT, MutAnyOrigin],
    textures: LayoutTensor[DTYPE, L_TEX, MutAnyOrigin],
    texels: LayoutTensor[DType.uint8, L_TEXELS, MutAnyOrigin],
    lights: LayoutTensor[DTYPE, L_LIGHTS, MutAnyOrigin],
    nlight: Int,
    hit: RayHit[DTYPE],
    hitpoint: Vec3Generic[DTYPE],
    eye: Vec3Generic[DTYPE],
    gaze: Vec3Generic[DTYPE],
    base_scale: Scalar[DTYPE],
) -> Vec3Generic[DTYPE] where DTYPE.is_floating_point():
    """The colour of one surface point: material, texel, then the lights.

    `base_scale` multiplies the geom's rgba BEFORE the shading, which is
    `renderGeomReflection`'s whole body:

        for (k) { old[k] = rgba[k]; rgba[k] *= reflectance; }
        renderGeom(...); restore

    ⚠ THAT IS NOT THE SAME AS SCALING THE RESULT. The ambient and diffuse
    terms are linear in the material colour and the SPECULAR term does not
    contain it at all (`GL_SPECULAR` is the material's own `specular`, not
    its colour), so a reflection at 0.5 keeps its highlights at full
    strength. Scaling the returned colour instead would dim them, which on a
    polished floor is most of what a reflection shows.
    """
    var g = hit.geom
    var ab = g * VIS_GEOM_APPEARANCE
    var base = Vec3Generic[DTYPE](
        rebind[Scalar[DTYPE]](appearance[ab + APP_IDX_R]) * base_scale,
        rebind[Scalar[DTYPE]](appearance[ab + APP_IDX_G]) * base_scale,
        rebind[Scalar[DTYPE]](appearance[ab + APP_IDX_B]) * base_scale,
    )
    var matid = Int(rebind[Scalar[DTYPE]](appearance[ab + APP_IDX_MATID]))
    var specular = Scalar[DTYPE](0.5)
    var shininess = Scalar[DTYPE](0.5)
    var texid = -1
    var repeat_u = Scalar[DTYPE](1)
    var repeat_v = Scalar[DTYPE](1)
    var texuniform = False
    var ttype = -1
    if matid >= 0 and matid < MAX_VIS_MATERIALS:
        var mb = matid * VIS_MAT_WORDS
        if rebind[Scalar[DTYPE]](materials[mb + MAT_IDX_ACTIVE]) != 0:
            specular = rebind[Scalar[DTYPE]](materials[mb + MAT_IDX_SPECULAR])
            shininess = rebind[Scalar[DTYPE]](
                materials[mb + MAT_IDX_SHININESS]
            )
            repeat_u = rebind[Scalar[DTYPE]](
                materials[mb + MAT_IDX_TEXREPEAT_U]
            )
            repeat_v = rebind[Scalar[DTYPE]](
                materials[mb + MAT_IDX_TEXREPEAT_V]
            )
            texuniform = (
                rebind[Scalar[DTYPE]](materials[mb + MAT_IDX_TEXUNIFORM]) != 0
            )
            texid = Int(rebind[Scalar[DTYPE]](materials[mb + MAT_IDX_TEXID]))
            if texid >= 0:
                var tb = texid * VIS_TEX_WORDS
                ttype = Int(
                    rebind[Scalar[DTYPE]](textures[tb + TEX_IDX_TYPE])
                )

    var tx = Texel[DTYPE](
        Scalar[DTYPE](1), Scalar[DTYPE](1), Scalar[DTYPE](1), False
    )
    if texid >= 0:
        # ⚠ THE HIT POINT AND THE NORMAL GO BACK INTO THE GEOM'S FRAME, which
        # is where every one of `settexture`'s texgen planes lives.
        var pose = _geom_world_pose[DTYPE](geoms, xpos, xquat, env, g)
        var inv = pose[1].conjugate()
        var lp = inv.rotate_vec(hitpoint - pose[0])
        var ln = inv.rotate_vec(hit.normal)
        var uv = geom_uv[DTYPE](
            geoms, mesh_uv, g, hit.tri, hit.bu, hit.bv,
            lp, ln, ttype, repeat_u, repeat_v, texuniform,
        )
        tx = sample_texture[DTYPE](textures, texels, texid, uv.u, uv.v)

    var lit = shade_lights[DTYPE, SHADOWS](
        lights, nlight, geoms, ngeom, bodies, xpos, xquat, env,
        mesh_meta, mesh_tris, hfield_meta, hfield_data, hf_stride,
        hitpoint, hit.normal, eye, gaze, base, specular, shininess,
        Scalar[DTYPE](0),
    )
    # ⚠⚠ `GL_MODULATE` MULTIPLIES THE *LIT* COLOUR, NOT THE MATERIAL COLOUR.
    # `render_gl3.c:699` sets it with the default single-colour specular, so
    # OpenGL lights the vertex colour (highlight included), clamps the sum to
    # [0, 1], and only then multiplies by the texel. Modulating `base` before
    # the lights instead added every textured surface's highlight on top of
    # the texture at full strength: LIBERO's wood table came out ~7% bright
    # and the metal stove base (shininess 1, specular .5) near white.
    if tx.hit:
        lit = Vec3Generic[DTYPE](
            _clamp01[DTYPE](lit.x) * tx.r,
            _clamp01[DTYPE](lit.y) * tx.g,
            _clamp01[DTYPE](lit.z) * tx.b,
        )
    return lit


def render_pixel[
    DTYPE: DType,
    SHADOWS: Bool,
    REFLECT: Bool,
    L_GEOMS: Layout,
    L_APP: Layout,
    L_BODIES: Layout,
    L_XPOS: Layout,
    L_XQUAT: Layout,
    L_MESH_META: Layout,
    L_TRI: Layout,
    L_UV: Layout,
    L_HF_META: Layout,
    L_HF: Layout,
    L_MAT: Layout,
    L_TEX: Layout,
    L_TEXELS: Layout,
    L_LIGHTS: Layout,
](
    geoms: LayoutTensor[DTYPE, L_GEOMS, MutAnyOrigin],
    ngeom: Int,
    appearance: LayoutTensor[DTYPE, L_APP, MutAnyOrigin],
    bodies: LayoutTensor[DTYPE, L_BODIES, MutAnyOrigin],
    xpos: LayoutTensor[DTYPE, L_XPOS, MutAnyOrigin],
    xquat: LayoutTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    env: Int,
    mesh_meta: LayoutTensor[DTYPE, L_MESH_META, MutAnyOrigin],
    mesh_tris: LayoutTensor[DTYPE, L_TRI, MutAnyOrigin],
    mesh_uv: LayoutTensor[DTYPE, L_UV, MutAnyOrigin],
    hfield_meta: LayoutTensor[DTYPE, L_HF_META, MutAnyOrigin],
    hfield_data: LayoutTensor[DTYPE, L_HF, MutAnyOrigin],
    hf_stride: Int,
    materials: LayoutTensor[DTYPE, L_MAT, MutAnyOrigin],
    textures: LayoutTensor[DTYPE, L_TEX, MutAnyOrigin],
    texels: LayoutTensor[DType.uint8, L_TEXELS, MutAnyOrigin],
    lights: LayoutTensor[DTYPE, L_LIGHTS, MutAnyOrigin],
    nlight: Int,
    frame: CameraFrame[DTYPE],
    width: Int,
    height: Int,
    px: Int,
    py: Int,
    background: Vec3Generic[DTYPE],
) -> PixelHit[DTYPE] where DTYPE.is_floating_point():
    """Primary ray, material, texel, lights, and MuJoCo's reflection pass.

    ⚠ EVERY GEOM IN `geoms` IS DRAWN. The group filter and the alpha-zero
    filter both ran when `VisualModel` was built, so there is no mask here and
    no `flg_static`: this table already IS the set of things a camera sees.
    That is also why the loop is 36 geoms on `libero_goal` rather than 240.

    ⚠ NO `bodyexclude`. A wrist camera SHOULD see the gripper it is mounted
    on — that is most of what it is for. The opposite default from
    `rangefinder_site`, which excludes its own body because MuJoCo's sensor
    does.

    ⚠⚠ THE REFLECTION IS **ADDED**, NOT BLENDED, AND THAT IS THE REFERENCE.
    `mjr_render` draws the mirrored scene first (each geom's rgb scaled by
    `reflectance`), then draws the reflective geom over it with
    `glBlendFunc(GL_ONE, GL_ONE)`. So the mirror's own colour is NOT
    attenuated by `1 - reflectance`; it is added at full strength on top of
    the reflection, and the sum is clamped on write. A lerp — the reflex a
    ray tracer reaches for — is DIMMER than the reference everywhere and is
    not what the recorded pixels show: on the stove it is the difference
    between 75 and 156 out of 255.

    ⚠ AND THE MIRROR SHOWS BLACK WHERE THE REFLECTED RAY MISSES.
    `glClearColor(0, 0, 0, 0)` is what the reflection pass draws over, and the
    skybox pass runs LATER and is depth-tested away inside the mirror. So a
    miss contributes nothing at all — not the background this function
    returns for a primary miss.
    """
    var dir = camera_pixel_ray[DTYPE](frame, width, height, px, py)

    var hit = ray_model[DTYPE](
        geoms, ngeom, bodies, xpos, xquat, env,
        mesh_meta, mesh_tris, hfield_meta, hfield_data, hf_stride,
        frame.pos, dir,
    )

    if hit.geom < 0:
        return PixelHit[DTYPE](background, Scalar[DTYPE](0), -1, -1)

    # PLANAR depth. `dir` is normalised and world-space, so the cosine to the
    # optical axis is `dot(dir, -zaxis)` — the same quantity the reference
    # spells `-ray_dir_local_cam[2]` in the camera's own frame.
    var cos_axis = -dir.dot(frame.zaxis)
    var depth = hit.t * cos_axis
    var hitpoint = frame.pos + dir * hit.t
    var gaze = frame.zaxis * Scalar[DTYPE](-1)

    var rgb = shade_hit[DTYPE, SHADOWS](
        geoms, ngeom, appearance, bodies, xpos, xquat, env,
        mesh_meta, mesh_tris, mesh_uv, hfield_meta, hfield_data, hf_stride,
        materials, textures, texels, lights, nlight,
        hit, hitpoint, frame.pos, gaze, Scalar[DTYPE](1),
    )

    var seen = -1
    comptime if REFLECT:
        var refl = rebind[Scalar[DTYPE]](
            appearance[hit.geom * VIS_GEOM_APPEARANCE + APP_IDX_REFLECT]
        )
        if refl > Scalar[DTYPE](0):
            var pose = _geom_world_pose[DTYPE](
                geoms, xpos, xquat, env, hit.geom
            )
            # ⚠⚠ THE MIRROR IS THE GEOM'S **+Z FACE**, NOT ITS SURFACE. For a
            # box the reference builds a temporary PLANE and pushes it out to
            # the +Z side (`pos += size[2] * mat[:,2]`); for a plane the geom
            # already is that plane. Reflecting about the surface the ray
            # actually hit would mirror the scene about a side face and put
            # the reflection somewhere else entirely.
            var n = pose[1].rotate_vec(Vec3Generic[DTYPE](0, 0, 1))
            var ppos = pose[0]
            var gtype = Int(
                rebind[Scalar[DTYPE]](geoms[hit.geom, GEOM_IDX_TYPE])
            )
            if gtype == GEOM_BOX:
                ppos = ppos + n * rebind[Scalar[DTYPE]](
                    geoms[hit.geom, GEOM_IDX_HALF_Z]
                )
            # `isBehind` — the camera on the far side of the plane sees the
            # back of the mirror and there is nothing to show.
            var side = (frame.pos - ppos).dot(n)
            var along = dir.dot(n)
            # ⚠⚠ ONLY WHERE THE +Z FACE ITSELF IS VISIBLE. The reference's
            # stencil is that temporary PLANE rendered into the stencil
            # buffer, so it is the +Z face's silhouette and NOT the box's: a
            # box seen from above shows its top face and one or two SIDES, and
            # the sides are outside the stencil and get no reflection at all.
            # Reflecting wherever the primary ray hits the mirror geom fills
            # 100% of the stove where the reference fills 76%, and those extra
            # pixels are its brightest — they are side faces catching a
            # reflection the reference does not draw there.
            var face = hit.normal.dot(n)
            if (
                side >= Scalar[DTYPE](0)
                and along < Scalar[DTYPE](-1e-9)
                and face > Scalar[DTYPE](0.5)
            ):
                # Where the PRIMARY ray crosses the mirror plane. For a hit on
                # the +Z face this is the hit point; for one on a side face it
                # is where the reflected image would have come from, which is
                # what the stencilled mirrored render draws there.
                var tp = side / (-along)
                var org = frame.pos + dir * tp
                var rdir = dir - n * (Scalar[DTYPE](2) * along)
                var rhit = ray_model[DTYPE](
                    geoms, ngeom, bodies, xpos, xquat, env,
                    mesh_meta, mesh_tris, hfield_meta, hfield_data, hf_stride,
                    org + n * Scalar[DTYPE](1.0e-6), rdir,
                )
                # `i != j` in the reference's loop: the mirror does not
                # reflect itself.
                if rhit.geom >= 0 and rhit.geom != hit.geom:
                    var rp = org + rdir * rhit.t
                    # ⚠⚠ THE EYE IS THE **MIRRORED** ONE, AND ONLY THE EYE.
                    # `mjr_render` pushes the reflection matrix and THEN calls
                    # `adjustLight`, so in its pass the geometry and every
                    # light are mirrored and the eye is not. Mapping that
                    # whole configuration back through the (isometric)
                    # reflection — which leaves every dot product alone —
                    # gives real geometry, real lights, the camera's own gaze
                    # for the headlight, and the eye reflected. Getting this
                    # wrong moves every specular highlight inside the mirror
                    # and nothing else, which reads as a shading bug rather
                    # than a frame bug.
                    #
                    # The mirrored eye is `tp` back along the reflected ray:
                    # the reflected ray leaves the plane at `org`, and the
                    # plane is equidistant from the eye and its image.
                    seen = rhit.geom
                    var meye = org - rdir * tp
                    # ⚠ AND NO SHADOWS IN THE REFLECTION. The reference's
                    # reflection pass enables the lights plainly and never
                    # binds the shadow map, so a mirrored scene is unshadowed
                    # even when the direct one is not.
                    rgb = rgb + shade_hit[DTYPE, False](
                        geoms, ngeom, appearance, bodies, xpos, xquat, env,
                        mesh_meta, mesh_tris, mesh_uv, hfield_meta,
                        hfield_data, hf_stride, materials, textures, texels,
                        lights, nlight, rhit, rp, meye, gaze, refl,
                    )

    return PixelHit[DTYPE](
        Vec3Generic[DTYPE](
            _clamp01[DTYPE](rgb.x),
            _clamp01[DTYPE](rgb.y),
            _clamp01[DTYPE](rgb.z),
        ),
        depth,
        hit.geom,
        seen,
    )
