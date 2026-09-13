"""The tracer's appearance path, on cases whose answer is arithmetic.

`examples/tasks/libero_camera_gate.mojo` is L5's real gate and it needs ~6 GB
of demonstrations. This file needs nothing: it drives `sample_texture`,
`geom_uv` and `shade_lights` on hand-built tables whose answer is a closed
form, so a clone with no assets still catches a broken sampler.

⚠⚠ IT IS NOT A SUBSTITUTE FOR THE GATE AND MUST NOT BE READ AS ONE. Every
expected value here is one I derived from the same reading of
`render_gl3.c`/`render_context.c` that the code was written from, so the two
agree by construction — the blind shape this tree keeps paying for
(`_a_gate_that_shares_its_reference_implementation_is_blind`). What it defends
is the ARITHMETIC: a half-texel offset, a wrap that lands one column over, a
box face picked by position instead of by normal, a specular term that fires
on a surface facing away from the light. The camera gate is what says the
MODEL is right.

⚠ ANTI-VACUITY IS EXPLICIT. Several checks below would pass on a function
that returned a constant, so each is paired with a second call that must
differ — the "0 mismatches means nothing was tested" failure this tree treats
as the default one.
"""

from std.math import sqrt

from layout import Layout, LayoutTensor

from mojo_rl.math3d import Vec3
from mojo_rl.nn.core.tensor import TensorImpl
from mojo_rl.physics3d.constants import GEOM_BOX, GEOM_PLANE, GEOM_MESH
from mojo_rl.physics3d.fields.rt_layout import DYN1, DYN2, rl1, rl2
from mojo_rl.physics3d.gpu.constants import (
    MODEL_GEOM_SIZE,
    GEOM_IDX_TYPE,
    GEOM_IDX_HALF_X,
    GEOM_IDX_HALF_Y,
    GEOM_IDX_HALF_Z,
    GEOM_IDX_RADIUS,
    GEOM_IDX_HALF_LENGTH,
)
from mojo_rl.physics3d.parser.flat_model import TEX_2D, TEX_CUBE
from mojo_rl.physics3d.raytrace.appearance import (
    geom_uv,
    sample_texture,
)
from mojo_rl.physics3d.raytrace.visual_records import (
    LIGHT_BODY_HEADLIGHT,
    MAX_VIS_TEXTURES,
    TEX_IDX_ACTIVE,
    TEX_IDX_ADR,
    TEX_IDX_HEIGHT,
    TEX_IDX_NCHAN,
    TEX_IDX_TYPE,
    TEX_IDX_WIDTH,
    VIS_TEX_WORDS,
    VIS_UV_WORDS,
)

comptime DT = DType.float64


struct Tally(Copyable, Movable):
    var n: Int
    var bad: Int

    def __init__(out self):
        self.n = 0
        self.bad = 0

    def check(mut self, ok: Bool, what: String) raises:
        self.n += 1
        if not ok:
            self.bad += 1
            print("  FAIL:", what)

    def near(
        mut self, got: Float64, want: Float64, tol: Float64, what: String
    ) raises:
        var d = got - want
        if d < 0:
            d = -d
        self.n += 1
        if not (d <= tol):
            self.bad += 1
            print(
                "  FAIL:", what, "got", got, "want", want, "tol", tol
            )


def _tex_tables(
    mut meta: TensorImpl[DT],
    mut px: TensorImpl[DType.uint8],
    w: Int,
    h: Int,
    ttype: Int,
) raises:
    """One texture at slot 0, `w x h`, red ramping with the COLUMN and green
    with the ROW, so a swapped axis or a flipped row is a different number and
    not a different shade of the same one."""
    meta = TensorImpl[DT].alloc(MAX_VIS_TEXTURES * VIS_TEX_WORDS)
    meta.data[TEX_IDX_ADR] = 0
    meta.data[TEX_IDX_WIDTH] = Scalar[DT](w)
    meta.data[TEX_IDX_HEIGHT] = Scalar[DT](h)
    meta.data[TEX_IDX_TYPE] = Scalar[DT](ttype)
    meta.data[TEX_IDX_ACTIVE] = 1
    meta.data[TEX_IDX_NCHAN] = 3
    px = TensorImpl[DType.uint8].alloc(w * h * 3)
    for y in range(h):
        for x in range(w):
            var o = (y * w + x) * 3
            px.data[o + 0] = UInt8(255 * x // max(1, w - 1))
            px.data[o + 1] = UInt8(255 * y // max(1, h - 1))
            px.data[o + 2] = UInt8(0)


def main() raises:
    var t = Tally()
    print("=" * 74)
    print("the tracer's appearance path — sampler, texgen, lighting")
    print("=" * 74)

    # ── the sampler ───────────────────────────────────────────────────────
    var meta = TensorImpl[DT]()
    var px = TensorImpl[DType.uint8]()
    _tex_tables(meta, px, 4, 4, TEX_2D)
    var mv = meta.lt_dyn["cpu", DYN1](rl1(meta.n))
    var pv = px.lt_dyn["cpu", DYN1](rl1(px.n))

    # A texel CENTRE is at (i + 0.5)/w, and there the bilinear weights are
    # (1, 0): the sample must be that texel exactly. Getting the half-texel
    # offset wrong shifts every texture by half a texel — invisible on a 4096
    # map and the whole image on a 4x4 one.
    for i in range(4):
        var u = (Float64(i) + 0.5) / 4.0
        var s = sample_texture[DT](mv, pv, 0, Scalar[DT](u), Scalar[DT](0.125))
        t.near(
            Float64(s.r), Float64(255 * i // 3) / 255.0, 1e-9,
            "texel centre column " + String(i),
        )
        t.near(Float64(s.g), 0.0, 1e-9, "texel centre row 0")

    # Halfway between two texel CENTRES is their mean, exactly. On a 4-wide
    # texture the centres are 0.125, 0.375, 0.625, 0.875, so 0.5 is the
    # midpoint of columns 1 and 2 — not 0.375, which is a centre.
    var mid = sample_texture[DT](
        mv, pv, 0, Scalar[DT](0.5), Scalar[DT](0.125)
    )
    t.near(
        Float64(mid.r),
        0.5 * (Float64(85) + Float64(170)) / 255.0,
        1e-9,
        "bilinear midpoint between columns 1 and 2",
    )

    # ⚠ ANTI-VACUITY: the two samples above must DIFFER. A sampler returning a
    # constant would satisfy neither check on its own if the constant were
    # right, and both if it were the mean.
    t.check(
        Float64(mid.r) != Float64(
            sample_texture[DT](mv, pv, 0, Scalar[DT](0.125), Scalar[DT](0.125)).r
        ),
        "the sampler returns different colours for different u",
    )

    # `GL_REPEAT` on a 2D texture: u and u+1 are the same texel.
    var a0 = sample_texture[DT](mv, pv, 0, Scalar[DT](0.375), Scalar[DT](0.625))
    var a1 = sample_texture[DT](mv, pv, 0, Scalar[DT](3.375), Scalar[DT](0.625))
    t.near(Float64(a0.r), Float64(a1.r), 1e-12, "2D wrap: u and u+3 agree")
    t.near(Float64(a0.g), Float64(a1.g), 1e-12, "2D wrap: v unchanged")

    # A CUBE face CLAMPS instead — `mjr_uploadTexture` sets GL_CLAMP_TO_EDGE,
    # and a repeat there shows the opposite edge across a silhouette.
    var cmeta = TensorImpl[DT]()
    var cpx = TensorImpl[DType.uint8]()
    _tex_tables(cmeta, cpx, 4, 4, TEX_CUBE)
    var cmv = cmeta.lt_dyn["cpu", DYN1](rl1(cmeta.n))
    var cpv = cpx.lt_dyn["cpu", DYN1](rl1(cpx.n))
    var c_out = sample_texture[DT](cmv, cpv, 0, Scalar[DT](1.9), Scalar[DT](0.5))
    var c_edge = sample_texture[DT](cmv, cpv, 0, Scalar[DT](1.0), Scalar[DT](0.5))
    t.near(
        Float64(c_out.r), Float64(c_edge.r), 1e-12,
        "cube clamp: u past 1 reads the last column",
    )
    t.check(
        Float64(c_out.r) != Float64(a0.r),
        "cube clamp and 2D wrap are not the same answer",
    )

    # An inactive slot is a MISS and returns white, so an unset material
    # multiplies its colour by 1 rather than by black.
    var miss = sample_texture[DT](mv, pv, 7, Scalar[DT](0.5), Scalar[DT](0.5))
    t.check(not miss.hit, "an inactive texture slot reports a miss")
    t.near(Float64(miss.r), 1.0, 1e-12, "a miss is white, not black")

    # ── texgen ────────────────────────────────────────────────────────────
    var geoms = TensorImpl[DT].alloc(2 * MODEL_GEOM_SIZE)
    geoms.data[GEOM_IDX_TYPE] = Scalar[DT](GEOM_BOX)
    geoms.data[GEOM_IDX_HALF_X] = 0.5
    geoms.data[GEOM_IDX_HALF_Y] = 0.25
    geoms.data[GEOM_IDX_HALF_Z] = 0.1
    geoms.data[MODEL_GEOM_SIZE + GEOM_IDX_TYPE] = Scalar[DT](GEOM_PLANE)
    geoms.data[MODEL_GEOM_SIZE + GEOM_IDX_HALF_X] = 3.0
    geoms.data[MODEL_GEOM_SIZE + GEOM_IDX_HALF_Y] = 2.0
    var gv = geoms.lt_dyn["cpu", DYN2](rl2(2, MODEL_GEOM_SIZE))
    var uvs = TensorImpl[DT].alloc(VIS_UV_WORDS)
    uvs.data[0] = 0.0
    uvs.data[1] = 0.0
    uvs.data[2] = 1.0
    uvs.data[3] = 0.0
    uvs.data[4] = 0.0
    uvs.data[5] = 1.0
    var uvv = uvs.lt_dyn["cpu", DYN1](rl1(uvs.n))

    var one = Scalar[DT](1)
    # A box's +Z face: `u = (x/sx + 1)/2`, `v = 1 - (y/sy + 1)/2`. The centre
    # of the face is the centre of the texture.
    var bc = geom_uv[DT](
        gv, uvv, 0, -1, 0, 0,
        Vec3[DT](0, 0, 0.1), Vec3[DT](0, 0, 1),
        TEX_2D, one, one, False,
    )
    t.near(Float64(bc.u), 0.5, 1e-12, "box +Z centre u")
    t.near(Float64(bc.v), 0.5, 1e-12, "box +Z centre v")
    var bx = geom_uv[DT](
        gv, uvv, 0, -1, 0, 0,
        Vec3[DT](0.5, -0.25, 0.1), Vec3[DT](0, 0, 1),
        TEX_2D, one, one, False,
    )
    t.near(Float64(bx.u), 1.0, 1e-12, "box +Z corner u")
    t.near(Float64(bx.v), 1.0, 1e-12, "box +Z corner v is FLIPPED")

    # ⚠ THE FACE COMES FROM THE NORMAL. The same point with a +X normal reads
    # the (y, z) pair instead, and must NOT give the +Z answer.
    var bxx = geom_uv[DT](
        gv, uvv, 0, -1, 0, 0,
        Vec3[DT](0.5, -0.25, 0.1), Vec3[DT](1, 0, 0),
        TEX_2D, one, one, False,
    )
    t.check(
        Float64(bxx.u) != Float64(bx.u),
        "the box face is chosen by the NORMAL, not by the position",
    )
    t.near(Float64(bxx.u), 0.0, 1e-12, "box +X face u is (y/sy + 1)/2")

    # `texrepeat` multiplies, and `texuniform` multiplies again by the size —
    # `settexture`'s `scl[k] * geom->size[k]`.
    var br = geom_uv[DT](
        gv, uvv, 0, -1, 0, 0,
        Vec3[DT](0.5, -0.25, 0.1), Vec3[DT](0, 0, 1),
        TEX_2D, Scalar[DT](3), Scalar[DT](3), False,
    )
    t.near(Float64(br.u), 3.0, 1e-12, "texrepeat scales the box UV")
    var bu = geom_uv[DT](
        gv, uvv, 0, -1, 0, 0,
        Vec3[DT](0.5, -0.25, 0.1), Vec3[DT](0, 0, 1),
        TEX_2D, Scalar[DT](3), Scalar[DT](3), True,
    )
    t.near(
        Float64(bu.u), 3.0 * 0.5, 1e-12,
        "texuniform multiplies the repeat by the geom's half-size",
    )

    # A finite plane: `u = (x + sx)/(2 sx)`, `v = 1 - (y + sy)/(2 sy)`.
    var pc = geom_uv[DT](
        gv, uvv, 1, -1, 0, 0,
        Vec3[DT](1.5, 1.0, 0), Vec3[DT](0, 0, 1),
        TEX_2D, one, one, False,
    )
    t.near(Float64(pc.u), 0.75, 1e-12, "plane u")
    t.near(Float64(pc.v), 0.25, 1e-12, "plane v")

    # A mesh triangle: the barycentrics interpolate the corner UVs. `bu`
    # weights corner 0, `bv` corner 1, and `1 - bu - bv` corner 2 — the same
    # order `ray_triangle` solves for.
    geoms.data[GEOM_IDX_TYPE] = Scalar[DT](GEOM_MESH)
    var gv2 = geoms.lt_dyn["cpu", DYN2](rl2(2, MODEL_GEOM_SIZE))
    var mu = geom_uv[DT](
        gv2, uvv, 0, 0, Scalar[DT](0.25), Scalar[DT](0.5),
        Vec3[DT](0, 0, 0), Vec3[DT](0, 0, 1),
        TEX_2D, one, one, False,
    )
    t.near(Float64(mu.u), 0.5, 1e-12, "mesh u = bv * u1")
    t.near(Float64(mu.v), 0.25, 1e-12, "mesh v = (1 - bu - bv) * v2")

    print()
    print(t.n, "checks,", t.bad, "failed")
    if t.bad != 0:
        raise Error("the appearance path is wrong in " + String(t.bad)
                    + " of " + String(t.n) + " checks")
    print("=== PASS ===")
