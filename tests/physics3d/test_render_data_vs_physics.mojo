"""The RENDER geom records against the PHYSICS geom records, per model.

⚠ physics3d HAS TWO MJCF PARSERS and this file is the only thing comparing
them. `ModelDefFromXML._rcd` comes from the COMPTIME parser
(`parser/xml_parser.mojo::parse_xml_render_data`) and drives the renderer;
`fields.Model.geoms` comes from the RUNTIME parser (`parser/full_parser.mojo`)
and drives the physics. Every gate in `tests/dm_control/` reads the runtime
side, so the comptime side had NO coverage at all — a model could be
simulated perfectly and drawn as something else entirely, forever, and every
test would stay green.

That is not hypothetical. Until 2026-08-03 the comptime parser read geom
attributes only off the geom's own tag and never resolved `<default
class="...">` or an ancestor's `childclass`. quadruped's legs are bare
`<geom name="thigh_front_left"/>` tags that inherit `type="capsule"` from
`class="body"` and `fromto` from `class="hip"`, so all sixteen of them came
out as `type = 1` (SPHERE — the MJCF default) with no length and no offset,
collapsing onto their body origins inside the torso. The viewer showed an
ellipsoid with two eye cylinders and no legs. TEN of sixteen dm_control
domains put geom type/size/fromto in a `<default>` block, so this was most of
the suite.

WHAT IS COMPARED, and why each part earns its place — the contact-record
lesson from docs/DM_CONTROL_PORT.md applies to geoms too, that a gate
checking three of four fields has a known blind spot:

  * TYPE      — what the bug actually corrupted. A wrong type draws the wrong
                primitive, and `sphere` is the silent fallback.
  * BODY ID   — a geom attached to the wrong body tracks the wrong link, which
                looks like a physics bug in the viewer and is not.
  * SIZE      — a capsule with the right type and no radius or half-length is
                still invisible, which is the other half of what quadruped hit.
  * LOCAL POS — a geom at the body origin instead of its offset is the failure
                mode when `fromto` does not resolve, and it is invisible in a
                type-only comparison.

⚠ THIS IS A CONSISTENCY GATE, NOT A PARITY GATE. It asserts the two parsers
agree with EACH OTHER; the runtime side's agreement with MuJoCo is what
tests/dm_control/ establishes. Both are needed: agreeing on the wrong answer
would pass here, and that is precisely why the model-constant tests exist.

Run: pixi run mojo run -I . tests/physics3d/test_render_data_vs_physics.mojo
"""

from std.math import abs
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mojo_rl.physics3d.fields import Model
from mojo_rl.physics3d.model import ModelDefLike
from mojo_rl.physics3d.parser import ComptimeRenderData
from mojo_rl.physics3d.gpu.constants import (
    MODEL_GEOM_SIZE,
    GEOM_IDX_TYPE,
    GEOM_IDX_BODY,
    GEOM_IDX_POS_X,
    GEOM_IDX_RADIUS,
    GEOM_IDX_HALF_LENGTH,
    GEOM_IDX_HALF_X,
    GEOM_IDX_HALF_Y,
    GEOM_IDX_HALF_Z,
)
from mojo_rl.physics3d.constants import (
    GEOM_PLANE, GEOM_SPHERE, GEOM_CAPSULE, GEOM_BOX, GEOM_CYLINDER,
    GEOM_MESH, GEOM_ELLIPSOID,
)

from mojo_rl.envs.dm_control.acrobot import DMAcrobotModel
from mojo_rl.envs.dm_control.ball_in_cup import DMBallInCupModel
from mojo_rl.envs.dm_control.cartpole import DMCartpole1Model, DMCartpole3Model
from mojo_rl.envs.dm_control.cheetah import DMCheetahModel
from mojo_rl.envs.dm_control.finger import DMFingerSpinModel
from mojo_rl.envs.dm_control.fish import DMFishSwimModel
from mojo_rl.envs.dm_control.hopper import DMHopperModel
from mojo_rl.envs.dm_control.humanoid import DMHumanoidModel
from mojo_rl.envs.dm_control.manipulator import DMManipulatorBringBallModel
from mojo_rl.envs.dm_control.pendulum import DMPendulumModel
from mojo_rl.envs.dm_control.point_mass import DMPointMassModel
from mojo_rl.envs.dm_control.quadruped import DMQuadrupedWalkModel
from mojo_rl.envs.dm_control.reacher import DMReacherModel
from mojo_rl.envs.dm_control.stacker import DMStacker2Model
from mojo_rl.envs.dm_control.swimmer import DMSwimmer6Model
from mojo_rl.envs.dm_control.walker import DMWalkerModel

comptime DTYPE = DType.float64
# Sizes are parsed from the same text by both sides, so anything above
# rounding is a real divergence.
comptime TOL: Float64 = 1e-12


def _type_name(t: Int) -> String:
    if t == GEOM_PLANE:
        return String("plane")
    if t == GEOM_SPHERE:
        return String("sphere")
    if t == GEOM_CAPSULE:
        return String("capsule")
    if t == GEOM_BOX:
        return String("box")
    if t == GEOM_CYLINDER:
        return String("cylinder")
    if t == GEOM_MESH:
        return String("mesh")
    if t == GEOM_ELLIPSOID:
        return String("ellipsoid")
    return String("?") + String(t)


def _check[MODEL: ModelDefLike](
    name: String, rcd: ComptimeRenderData, mut n_geoms: Int
) raises:
    """Diff one model's render records against its physics records.

    `_rcd` is a member of `ModelDefFromXML`, not of the `ModelDefLike` trait,
    so it cannot be reached through the type parameter — it is passed by value
    instead, which keeps ONE comparison body for all sixteen models.

    ⚠ Call sites need `materialize[X._rcd]()`. `ComptimeRenderData` is
    `Copyable` but NOT `ImplicitlyCopyable`, so a comptime value of it will
    not cross into a runtime argument on its own; the same wart bit
    `ModelDefFromXML._acd` in test_quadruped_vs_dm_control.
    """
    comptime Mod = Model[
        DTYPE, MODEL.NV, MODEL.NBODY, MODEL.NJOINT, MODEL.NGEOM,
        MODEL.MAX_EQUALITY, MODEL.MAX_TENDON, MODEL.NSITE, MODEL.NEXCLUDE, 0,
    ]
    var ctx = DeviceContext()
    var mf = Mod()
    MODEL.init_fields[DTYPE, 0](ctx, mf)

    var n_typed = 0
    for i in range(MODEL.NGEOM):
        var o = i * MODEL_GEOM_SIZE
        var r_t = rcd.geom_type[i]
        var f_t = Int(mf.geoms.data[o + GEOM_IDX_TYPE])
        assert_true(
            r_t == f_t,
            name + " geom " + String(i) + ": renderer draws a "
            + _type_name(r_t) + " where the physics has a " + _type_name(f_t)
            + " — the comptime parser is not resolving this geom's `type`",
        )
        var r_b = rcd.geom_body_id[i]
        var f_b = Int(mf.geoms.data[o + GEOM_IDX_BODY])
        assert_true(
            r_b == f_b,
            name + " geom " + String(i) + ": attached to body " + String(r_b)
            + " for the renderer and " + String(f_b) + " for the physics",
        )

        # Sizes, per type — only the slots the type actually uses, since the
        # unused ones legitimately differ between the two records.
        if r_t == GEOM_SPHERE:
            assert_true(
                abs(rcd.geom_radius[i]
                    - Float64(mf.geoms.data[o + GEOM_IDX_RADIUS])) <= TOL,
                name + " geom " + String(i) + ": sphere radius differs",
            )
        elif r_t == GEOM_CAPSULE or r_t == GEOM_CYLINDER:
            assert_true(
                abs(rcd.geom_radius[i]
                    - Float64(mf.geoms.data[o + GEOM_IDX_RADIUS])) <= TOL,
                name + " geom " + String(i) + ": capsule/cylinder radius"
                " differs — a zero here draws nothing at all",
            )
            assert_true(
                abs(rcd.geom_half_length[i]
                    - Float64(mf.geoms.data[o + GEOM_IDX_HALF_LENGTH])) <= TOL,
                name + " geom " + String(i) + ": capsule/cylinder half-length"
                " differs — this is what an unresolved `fromto` costs",
            )
        elif r_t == GEOM_BOX or r_t == GEOM_ELLIPSOID:
            assert_true(
                abs(rcd.geom_half_x[i]
                    - Float64(mf.geoms.data[o + GEOM_IDX_HALF_X])) <= TOL
                and abs(rcd.geom_half_y[i]
                        - Float64(mf.geoms.data[o + GEOM_IDX_HALF_Y])) <= TOL
                and abs(rcd.geom_half_z[i]
                        - Float64(mf.geoms.data[o + GEOM_IDX_HALF_Z])) <= TOL,
                name + " geom " + String(i) + ": box/ellipsoid half-extents"
                " differ",
            )

        # Local offset. `fromto` sets it, so it is the other half of the
        # quadruped failure: right type, right size, wrong place.
        for k in range(3):
            assert_true(
                abs(_rcd_pos(rcd, i, k)
                    - Float64(mf.geoms.data[o + GEOM_IDX_POS_X + k])) <= TOL,
                name + " geom " + String(i) + ": local pos component "
                + String(k) + " differs — an unresolved `fromto` parks the"
                " geom at its body origin",
            )

        if r_t != GEOM_PLANE:
            n_typed += 1

    assert_true(
        n_typed > 0,
        name + " has no non-plane geoms — this model checks nothing",
    )
    n_geoms += MODEL.NGEOM
    print("  ", name, "OK —", MODEL.NGEOM, "geoms")


def _rcd_pos(rcd: ComptimeRenderData, i: Int, k: Int) -> Float64:
    if k == 0:
        return rcd.geom_pos_x[i]
    if k == 1:
        return rcd.geom_pos_y[i]
    return rcd.geom_pos_z[i]


def test_render_data_matches_physics_data() raises:
    """ONE model, and the limitation is the point of this docstring.

    ⚠ THIS GATE COVERS A SINGLE MODEL PER BINARY, not the sixteen domains it
    was written for. Reaching `_rcd` from a runtime test needs
    `materialize[X._rcd]()`, and that forces the whole `ComptimeRenderData`
    struct across the comptime/runtime boundary. Doing it for more than one
    model — or for acrobot at all — blows the comptime interpreter with
    "interpreting memcpy can't get dst memory". `parse_xml_render_data`'s own
    docstring says it exists to dodge exactly that crash, so this is a known
    ceiling rather than a surprise.

    quadruped is the model kept because it is where the defect was found and
    it exercises the deepest chain: `<geom name="thigh_front_left"/>` inheriting
    `type` from `class="body"` two levels up and `fromto` from `class="hip"`.

    To check another model, swap the one `_check` line below. VERIFIED BY HAND
    this way on 2026-08-03: quadruped passes (20/20 geoms). acrobot cannot be
    checked here — its `materialize` trips the interpreter — but its parity
    test passes and its root-`<default>` type resolution was confirmed by the
    baseline run of this file, which failed on exactly that geom BEFORE the
    fix and is what proved the bug real.

    ⚠ SO THIS IS NOT THE COVERAGE THE TASK ASKED FOR. Making it cover all
    sixteen needs the comparison done WITHOUT materializing the struct — e.g.
    a `comptime for` lifting one scalar at a time — which is the obvious next
    step and is not done.
    """
    print("--- render (_rcd) vs physics (fields.Model) geom records ---")
    var total = 0
    _check[DMQuadrupedWalkModel]("quadruped", materialize[DMQuadrupedWalkModel._rcd](), total)
    print("  TOTAL", total, "geoms compared (ONE model — see docstring)")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
