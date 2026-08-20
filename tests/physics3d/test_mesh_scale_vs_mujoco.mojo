"""`<mesh scale>` — the attribute that decides whether a hull is millimetres.

    pixi run mojo run -I . tests/physics3d/test_mesh_scale_vs_mujoco.mojo

WHAT WAS MISSING. `<mesh scale>` was parsed NOWHERE in the tree, so every mesh
was loaded in whatever units its file happened to use.

⚠⚠ THE MODELS THAT NEED IT DECLARE IT ONCE, IN A `<default>` BLOCK, and that
is why reading the asset's own tag finds nothing. robotis_op3 ships its STLs
in MILLIMETRES and writes `<mesh scale="0.001 0.001 0.001"/>` inside
`<default>`; 48 of its 49 `<mesh file=.../>` assets carry no scale at all.
Its collision hulls therefore came out 1000x oversized — measured, contacts
10 to 20 m deep, at z = -4 to -20, from a robot whose root sits at 0.3 —
and the solver answered that by launching it:

    op3, 400 steps : zmax 77.8 m       ->  settles at 0.2791025 (MuJoCo 0.2791)
    step 0         : ncon 56, |qvel| clamped at 100
                                       ->  ncon 0, |qvel| 0.019620 (MuJoCo 0.0196)

⚠ THE SYMPTOM WAS INDISTINGUISHABLE FROM THE TWO INTEGRATOR BUGS FIXED IN
66230b9d — "opens fine, then flies" — and it is a completely different cause.
What separated them was `ncon` at step 0: spot had NO contacts and a velocity
already at 36 rad/s (an actuator/integrator problem), op3 had 56 contacts
before it had moved (a geometry problem). Print the contact count beside the
velocity; it costs one column and it picks the subsystem.

⚠ 19 MENAGERIE ROBOTS SET IT, and the size case is only half. Of the
declarations, 38 are `0.001 0.001 0.001` and 44 are a MIRROR — `1 -1 1` and
friends — which is how a model builds a left part and a right part from one
file. The mirror case does not explode; it silently produces a part that is
inside-out, which is why it needs a gate rather than a bug report.

⚠ A MIRROR REVERSES TRIANGLE WINDING. `load_stl` flips the winding and the
face normal when the scale's determinant is negative, and transforms normals
by the INVERSE TRANSPOSE (n_i / s_i, renormalised) rather than like positions
— a distinction that is invisible for the uniform 0.001 case and wrong for
`0.9 1 1`, which Menagerie also ships.

⚠ THE HULL CACHE KEY CARRIES THE SCALE. A hull cached before this existed is
UNSCALED, and without the scale in the key it would be served to a
`scale="0.001"` model straight from disk — reproducing the bug invisibly, on
a machine where the code is correct. `hull_cache.mojo` learned the same lesson
about DTYPE once already.
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
    read_model_source,
)
from mojo_rl.physics3d.studio.stepping import StudioIntegPyr
from mojo_rl.physics3d.dynamics.actuation import apply_actions_fields
from mojo_rl.physics3d.gpu.constants import META_IDX_NUM_CONTACTS
from mojo_rl.physics3d.parser.render_fields import build_render_fields

comptime DT = DType.float64
comptime OP3 = String(
    "references/mujoco_menagerie-main/robotis_op3/scene.xml"
)

# ⚠ THE SCALE LIVES IN A `<default>` AND THE ASSETS ARE BARE — the exact shape
# op3 uses, and the one that made reading the asset's own tag useless. The
# third asset states its own scale so the precedence is pinned, and the fourth
# is a MIRROR.
comptime XML = String(
    """<mujoco>
  <compiler angle="radian" meshdir="references/mujoco_menagerie-main/robotis_op3/assets"/>
  <default>
    <mesh scale="0.001 0.001 0.001"/>
  </default>
  <asset>
    <mesh name="a" file="ll1.stl"/>
    <mesh name="b" file="ll2.stl"/>
    <mesh name="c" file="ll3.stl" scale="1 1 1"/>
    <mesh name="d" file="ll4.stl" scale="0.001 -0.001 0.001"/>
  </asset>
  <worldbody>
    <body>
      <freejoint/>
      <geom type="mesh" mesh="a"/>
      <geom type="mesh" mesh="c"/>
      <geom type="mesh" mesh="d"/>
    </body>
  </worldbody>
</mujoco>"""
)


def test_default_block_scale_reaches_the_assets_and_geoms() raises:
    """The parse: a bare `<mesh>` inherits the default's scale."""
    print("=== <default><mesh scale> reaches the asset table ===")
    var fmd = parse_xml_full(XML, String(""))
    assert_true(
        fmd.num_mesh_assets == 4,
        "fixture did not parse four mesh assets — the gate would be vacuous;"
        " got " + String(fmd.num_mesh_assets),
    )
    for i in range(4):
        print("  asset", i, " scale",
              fmd.mesh_asset_scale[i * 3 + 0],
              fmd.mesh_asset_scale[i * 3 + 1],
              fmd.mesh_asset_scale[i * 3 + 2])
    # a and b: bare tags, so the DEFAULT applies. This is the whole feature.
    for i in range(2):
        assert_true(
            abs(fmd.mesh_asset_scale[i * 3 + 0] - 0.001) < 1e-15
            and abs(fmd.mesh_asset_scale[i * 3 + 1] - 0.001) < 1e-15
            and abs(fmd.mesh_asset_scale[i * 3 + 2] - 0.001) < 1e-15,
            "a bare <mesh> must inherit <default><mesh scale>; asset "
            + String(i) + " got " + String(fmd.mesh_asset_scale[i * 3]),
        )
    # ⚠ THE NEGATIVE CONTROL, and it is not redundant: without it this file
    # passes against an implementation that applies the default to EVERY
    # asset, ignoring an explicit scale.
    assert_true(
        fmd.mesh_asset_scale[2 * 3 + 0] == 1.0,
        "an explicit scale='1 1 1' must WIN over the default's 0.001, got "
        + String(fmd.mesh_asset_scale[2 * 3 + 0]),
    )
    assert_true(
        fmd.mesh_asset_scale[3 * 3 + 1] == -0.001,
        "a MIRROR component must survive the parse with its sign — 44"
        " Menagerie declarations rely on it to build a left part from a right"
        " one; got " + String(fmd.mesh_asset_scale[3 * 3 + 1]),
    )
    # And the geom must carry its asset's scale, since the loader is handed a
    # filename and nothing else.
    print("  geom0 scale", fmd.geoms[0].mesh_scale_x,
          " geom1 scale", fmd.geoms[1].mesh_scale_x,
          " geom2 scale_y", fmd.geoms[2].mesh_scale_y)
    assert_true(
        abs(fmd.geoms[0].mesh_scale_x - 0.001) < 1e-15
        and fmd.geoms[1].mesh_scale_x == 1.0
        and fmd.geoms[2].mesh_scale_y == -0.001,
        "each geom must carry the scale of the asset it names",
    )
    print("  PASS")


def _op3() raises -> Tuple[Float64, Float64, Int, Float64]:
    """op3 through the runtime path: (max|vert|, zfinal, ncon, qvel0)."""
    var src = read_model_source(OP3)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var verts = 262144
    var dims = dims_from_flat(fmd, max_contacts=128, nmesh_verts=verts)
    var m = Model[DT, DynDims](dims)
    var tries = 0
    while True:
        try:
            build_model_runtime[DT](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") == -1 or tries > 24:
                raise e
            tries += 1
            verts = verts * 2
            dims = dims_from_flat(fmd, max_contacts=128, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)
    # ⚠ THE HULL VERTICES THEMSELVES, which is the only place the bug lived.
    # Everything downstream (contacts, forces, the trajectory) is a
    # consequence; this is the cause, and it is one number.
    var vmaxabs = 0.0
    var nvert = dims.get_nmesh_verts()
    for i in range(nvert * 3):
        var v = abs(Float64(m.mesh_verts.data[i]))
        if v > vmaxabs:
            vmaxabs = v

    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, 1](dims)
    for i in range(dims.get_nq()):
        d.qpos.data[i] = sf.qpos0.data[i]
    for i in range(dims.get_nv()):
        d.qvel.data[i] = Scalar[DT](0)
    var integ = StudioIntegPyr(dims)
    # ⚠⚠ THE ACTUATORS RUN, WITH ctrl = 0, AND THAT IS NOT THE SAME AS NOT
    # RUNNING THEM. op3's 20 `<position kp="21.1">` servos hold every joint at
    # 0; omitting them lets the robot go limp and it settles at 0.2695 instead
    # of 0.2791 — a 1 cm gap that would have to be absorbed by a loose
    # tolerance, hiding exactly the kind of drift this gate exists to catch.
    # MuJoCo's `mj_step` always computes `qfrc_actuator`, so a comparison
    # against its number has to as well.
    var nact = dims.get_nact()
    var actions = List[Float64](length=nact, fill=0.0)
    var act = List[Scalar[DT]](length=nact, fill=Scalar[DT](0))
    var qvel0 = 0.0
    for t in range(400):
        apply_actions_fields[DT](sf, d, actions, act, fmd.timestep)
        integ.step["cpu"](d, m)
        if t == 0:
            for i in range(dims.get_nv()):
                var v = abs(Float64(d.qvel.data[i]))
                if v > qvel0:
                    qvel0 = v
    return (
        vmaxabs,
        Float64(d.qpos.data[2]),
        Int(Float64(d.meta.data[META_IDX_NUM_CONTACTS])),
        qvel0,
    )


def test_op3_hulls_are_metres_and_it_stands() raises:
    """op3, the model this was found on, against MuJoCo's own numbers.

    ⚠ MEASURED ON THE 3.10.0 RUNTIME: op3 settles at z = 0.2791015861 with
    ncon 16, and its first step gives |qvel|max 0.0196 with NO contacts.
    """
    print("=== op3: scaled hulls, and a robot that stands ===")
    var r = _op3()
    print("  max |hull vertex|", r[0], " (op3's largest part is ~0.17 m;"
          " unscaled this was ~5-20)")
    print("  step0 |qvel|max  ", r[3], " (MuJoCo 0.0196)")
    print("  final z          ", r[1], " ncon", r[2],
          " (MuJoCo 0.2791015861 / 16)")
    # ⚠ THE CAUSE, ASSERTED DIRECTLY. A 1000x error is not a tolerance
    # question, so the bound is loose on purpose: op3's biggest mesh is
    # 0.17 m and the bug produced metres.
    assert_true(
        r[0] < 0.5,
        "op3's hull vertices reach " + String(r[0])
        + " m from the mesh origin. Its largest part is 0.17 m, so <mesh"
        " scale='0.001'> is not reaching the vertices and every hull is"
        " 1000x oversized.",
    )
    assert_true(
        r[3] < 1.0,
        "op3's first step gives |qvel|max " + String(r[3])
        + " where MuJoCo gives 0.0196 — the robot is being pushed apart"
        " before it has moved, which means it starts out interpenetrating.",
    )
    assert_true(
        abs(r[1] - 0.2791015861) < 1e-3,
        "op3 should settle standing at MuJoCo's 0.2791 m; it ended at "
        + String(r[1]) + " (unscaled, it passed 77 m)",
    )
    assert_true(
        r[2] == 16,
        "op3 rests on the floor with 16 contacts in MuJoCo; we report "
        + String(r[2]) + " — with none, the gate is not measuring a rest pose",
    )
    print("  PASS")


def test_the_renderer_gets_the_scale_too() raises:
    """The hulls and the picture must agree, or the tool lies about the sim.

    ⚠ A SEPARATE PATH FROM THE COLLISION ONE, and it would have stayed broken
    on its own: the renderer loads the STL by FILENAME through its own cache,
    so scaling the collision hulls leaves it drawing the millimetre mesh at
    metre size. A studio that simulates a 0.17 m robot and draws a 170 m one
    is not showing you the model it is stepping.

    ⚠ PER-GEOM, because `geom_mesh_id` resolves against the asset table by
    filename and a mirrored pair is two assets over one file.
    """
    print("=== the render path carries the scale ===")
    var src = read_model_source(OP3)
    var flat = expand_mjcf(src[0], src[1])
    var fmd = parse_xml_full(flat, src[1])
    var rf = build_render_fields(fmd, flat, src[1])
    assert_true(
        len(rf.geom_mesh_scale) == len(fmd.geoms) * 3,
        "every geom needs three scale slots; got "
        + String(len(rf.geom_mesh_scale)) + " for "
        + String(len(fmd.geoms)) + " geoms",
    )
    var n_scaled = 0
    for g in range(len(fmd.geoms)):
        if abs(rf.geom_mesh_scale[g * 3 + 0] - 0.001) < 1e-15:
            n_scaled += 1
    print("  geoms carrying 0.001 into the renderer:", n_scaled, "of",
          len(fmd.geoms))
    assert_true(
        n_scaled > 40,
        "op3 declares <mesh scale='0.001'> in its <default> and nearly all of"
        " its 51 geoms are meshes, so the renderer should receive 0.001 for"
        " almost all of them — it got " + String(n_scaled)
        + ". At 1.0 the robot draws 1000x oversized.",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
