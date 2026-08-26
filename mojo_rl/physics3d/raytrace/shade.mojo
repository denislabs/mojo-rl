"""Shading for the batched tracer — `render.py`'s ambient + light loop.

Deliberately SMALL, and the reason is in the assessment this work came from:
"MuJoCo's renderer has its own materials, lights and defaults. A `_vision`
task's pixels will not match ours to any tolerance"
(`docs/DM_CONTROL_AND_CAMERA_ASSESSMENT_2026_08_24.md` §3.2). The GEOMETRY here
is gateable — a hit distance and a geom id have a right answer, and
`ray/` is gated against MuJoCo four ways. The COLOUR is not. So this module
carries the reference's structure (a hemispheric ambient term plus a Lambert
term per light, shadowed by a second ray) and nothing that would only be
justified by a pixel comparison that cannot be run.

⚠ NOT PORTED, AND EACH FOR A REASON, NOT AN OVERSIGHT:
  * textures and the skybox — `sample_texture`/`sample_skybox` need
    `mat_texid`, `mesh_texcoord` and a `Texture2D` sampler, none of which
    reach `Model`. The background is a flat colour.
  * `<light>` — lights live in `RenderFields`, host-side, like cameras did
    before this landing. Until they get a `Model` table the tracer takes ONE
    directional light from its caller. Adding the table later changes this
    signature and nothing else: the loop the reference runs over `m.nlight` is
    this function called n times and summed.
  * specular / shininess / reflectance — material fields the ambient+Lambert
    model has no term for.
"""

from layout import Layout, LayoutTensor

from mojo_rl.math3d import Vec3 as Vec3Generic

from ..ray.model import ray_model


@always_inline
def _clamp01[
    DTYPE: DType
](x: Scalar[DTYPE]) -> Scalar[DTYPE] where DTYPE.is_floating_point():
    if x < Scalar[DTYPE](0):
        return Scalar[DTYPE](0)
    if x > Scalar[DTYPE](1):
        return Scalar[DTYPE](1)
    return x


@always_inline
def ambient_term[
    DTYPE: DType
](normal: Vec3Generic[DTYPE]) -> Vec3Generic[DTYPE] where (
    DTYPE.is_floating_point()
):
    """`render.py`'s hemispheric ambient, transcribed.

        hemispheric = 0.5 * (n.z + 1)
        ambient     = (0.4,0.4,0.45)*h + (0.1,0.1,0.12)*(1-h)
        result      = 0.5 * base * ambient

    The `0.5 *` and the multiply by `base_color` are the CALLER's, so this
    returns the ambient colour alone. A surface facing up is lit bluish-white,
    one facing down is lit dim blue — a cheap stand-in for sky and ground
    bounce that keeps an unlit face from being pure black.
    """
    var n = normal
    var l = n.length()
    if l > Scalar[DTYPE](0):
        n = n / l
    else:
        n = Vec3Generic[DTYPE](0, 0, 1)
    var h = Scalar[DTYPE](0.5) * (n.z + Scalar[DTYPE](1))
    var one_h = Scalar[DTYPE](1) - h
    return Vec3Generic[DTYPE](
        Scalar[DTYPE](0.40) * h + Scalar[DTYPE](0.10) * one_h,
        Scalar[DTYPE](0.40) * h + Scalar[DTYPE](0.10) * one_h,
        Scalar[DTYPE](0.45) * h + Scalar[DTYPE](0.12) * one_h,
    )


def directional_light_term[
    DTYPE: DType,
    L_GEOMS: Layout,
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
    bodies: LayoutTensor[DTYPE, L_BODIES, MutAnyOrigin],
    xpos: LayoutTensor[DTYPE, L_XPOS, MutAnyOrigin],
    xquat: LayoutTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    env: Int,
    mesh_meta: LayoutTensor[DTYPE, L_MESH_META, MutAnyOrigin],
    mesh_tris: LayoutTensor[DTYPE, L_TRI, MutAnyOrigin],
    hfield_meta: LayoutTensor[DTYPE, L_HF_META, MutAnyOrigin],
    hfield_data: LayoutTensor[DTYPE, L_HF, MutAnyOrigin],
    hf_stride: Int,
    normal: Vec3Generic[DTYPE],
    hitpoint: Vec3Generic[DTYPE],
    light_dir: Vec3Generic[DTYPE],
    use_shadows: Bool,
) -> Scalar[DTYPE] where DTYPE.is_floating_point():
    """`compute_lighting`'s directional branch — `ndotl * visible`.

    `light_dir` is the direction the light TRAVELS, as `mjModel.light_dir` is,
    so the vector towards the light is `-light_dir`. A directional light has no
    attenuation and no spot factor, which is the whole of the branch.

    ⚠ THE SHADOW RAY IS `ray_model` AGAIN, and that is the property that makes
    the tracer worth building: shadows are not a second pipeline with its own
    depth-map resolution, bias and frustum — the three knobs `fba3b30a` spent a
    session on for the SDL renderer — they are one more query against the same
    geometry the primary ray hit.

    ⚠ THE ORIGIN IS NUDGED ALONG THE NORMAL BY 1e-4, exactly as the reference
    does. Without it every lit surface shadows itself at t = 0 and the whole
    image goes to the ambient term, which reads as "shadows are broken" rather
    than "the ray started on the surface".

    ⚠ ANY HIT DARKENS TO 0.3 RATHER THAN 0. Also the reference's number. It
    keeps a shadowed surface from collapsing into the background, which for an
    OBSERVATION matters more than it does for a picture.
    """
    var l = -light_dir
    var ln = l.length()
    if ln <= Scalar[DTYPE](0):
        return Scalar[DTYPE](0)
    l = l / ln

    var n = normal
    var nl = n.length()
    if nl > Scalar[DTYPE](0):
        n = n / nl
    var ndotl = n.dot(l)
    if ndotl <= Scalar[DTYPE](0):
        return Scalar[DTYPE](0)

    if not use_shadows:
        return ndotl

    var origin = hitpoint + n * Scalar[DTYPE](1.0e-4)
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
        origin,
        l,
    )
    if hit.geom >= 0:
        return ndotl * Scalar[DTYPE](0.3)
    return ndotl
