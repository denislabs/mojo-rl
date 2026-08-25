"""`<mesh maxhullvert>` — the qhull BUDGET, and what it does to the hull.

    pixi run mojo run -I . tests/physics3d/test_maxhullvert_decimates_the_hull.mojo

WHAT WAS MISSING. `mjCMesh::MakeGraph` (`user_mesh.cc:1732`) builds its qhull
option string as

    "qhull Qt"                                    maxhullvert == -1
    "qhull Qt Q9 TA<maxhullvert - 4>"             otherwise

`TAn` is "stop after adding n vertices to the initial simplex" and `Q9` is
"process the furthest of all furthest points", so a budgeted hull is *the hull
qhull would have built had it run out of budget* — NOT the full hull with
vertices deleted afterwards. We parsed the attribute nowhere and passed a hard
`-1`, so every budgeted mesh kept the vertices qhull's budget stopped it
adding. The shim (`native/mrl_qhull.c`) has taken the argument since it was
written; the gap was the parser and the two calls between them.

⚠⚠ THE ONLY SPELLING trossen_wxai USES IS A `<default>`. It writes
`<default><mesh maxhullvert="64"/></default>` once and leaves all 27 of its
`<mesh file=.../>` assets bare, so an element-only read finds NOTHING there.
robotstudio_so101 needs the other half of the same precedence: it sets 128 in a
`<default>` and overrides it to 64 on three meshes. `test_the_class_chain_is
_read` below is those two spellings.

⚠ THE DECLARATION COUNT IS NOT THE GATE. `FlatModelDef.unhonoured_maxhullvert`
counted these declarations correctly for months while every hull was wrong —
the same shape as `feedback_a_threshold_parsed_without_the_count_it_trims`.
`test_the_declarations_are_still_seen` is plumbing and says so; the gate is the
hull.

MEASURED against MuJoCo 3.10.0, the runtime version, on two so101 gripper
meshes (`mesh_graph[graphadr]`, `mesh_polynum`, widest `mesh_polyvertnum`):

    budget      moving_jaw_..._part1        wrist_roll_follower_..._part0
    (none)      3428 / 5351 / 13            739 / 1226 / 12
    64            64 /  122 /  4             64 /  122 /  4
    128          128 /  243 /  6            128 /  244 /  5

⚠ THE UNBUDGETED ROW IS THE CONTROL, and it is what makes the other two mean
something: our full hull already reproduces MuJoCo's 3428 and 739 exactly, so a
budgeted row that misses can only be the budget.

⚠ THE HULL CACHE KEY CARRIES THE BUDGET. One STL at two budgets is two
payloads; without it in the key the first arm below would serve its hull to the
second and this file would pass while proving nothing.
"""

from std.math import abs
from std.testing import assert_true, assert_equal, TestSuite

from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, read_model_source,
)
from mojo_rl.physics3d.fields import Model, DynDims
from mojo_rl.physics3d.gpu.constants import (
    MODEL_MESH_META_SIZE, MESH_META_IDX_VERTNUM,
    MESH_META_IDX_POLYADR, MESH_META_IDX_POLYNUM,
    MODEL_MESH_POLY_SIZE, MESH_POLY_IDX_VERTNUM,
    MODEL_GEOM_SIZE, GEOM_IDX_MESH_ID,
)

comptime DT = DType.float64

comptime SO101_DIR = String(
    "references/mujoco_menagerie-main/robotstudio_so101/assets"
)
comptime WXAI = String(
    "references/mujoco_menagerie-main/trossen_wxai/scene.xml"
)
comptime SO101 = String(
    "references/mujoco_menagerie-main/robotstudio_so101/scene.xml"
)
comptime DEXEE = String(
    "references/mujoco_menagerie-main/shadow_dexee/scene.xml"
)

# `mesh_graph[graphadr]` / `mesh_polynum` / max `mesh_polyvertnum`, MuJoCo
# 3.10.0. Mesh A is `moving_jaw_so101_gripper_part1_v1.stl`, B is
# `wrist_roll_follower_so101_gripper_part0_v1.stl`.
comptime A_FULL_V: Int = 3428
comptime A_FULL_P: Int = 5351
comptime A_FULL_W: Int = 13
comptime B_FULL_V: Int = 739
comptime B_FULL_P: Int = 1226
comptime B_FULL_W: Int = 12
comptime A64_V: Int = 64
comptime A64_P: Int = 122
comptime A64_W: Int = 4
comptime B64_V: Int = 64
comptime B64_P: Int = 122
comptime B64_W: Int = 4
comptime A128_V: Int = 128
comptime A128_P: Int = 243
comptime A128_W: Int = 6
comptime B128_V: Int = 128
comptime B128_P: Int = 244
comptime B128_W: Int = 5


def _xml(mesh_defs: String, default_sec: String) -> String:
    """Two so101 gripper meshes, collidable, with whatever `<mesh>` spelling.

    ⚠ BOTH GEOMS MUST BE COLLIDABLE. `load_mesh_hull` only runs for a mesh some
    geom actually collides with, so a `contype="0"` fixture would build no hull
    at all and every assertion below would read a default.
    """
    return String(
        "<mujoco><compiler meshdir=\"" + SO101_DIR + "\"/>"
        + default_sec
        + "<asset>" + mesh_defs + "</asset>"
        + "<worldbody><body><freejoint/>"
        + "<geom type=\"mesh\" mesh=\"a\"/>"
        + "<geom type=\"mesh\" mesh=\"b\"/>"
        + "</body></worldbody></mujoco>"
    )


@fieldwise_init
struct _Hull(Copyable, Movable):
    """One mesh's (hull vertices, polygons, widest polygon)."""
    var nvert: Int
    var npoly: Int
    var maxpv: Int


def _hulls(xml: String) raises -> List[_Hull]:
    """Build the model and read back each COLLIDABLE mesh's hull topology.

    Returned in geom order, which for this fixture is (a, b).
    """
    var fmd = parse_xml_full(xml, String("."))
    var verts = 32768
    var dims = dims_from_flat(fmd, max_contacts=8, nmesh_verts=verts)
    var m = Model[DT, DynDims](dims)
    while True:
        try:
            build_model_runtime[DT](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") == -1:
                raise e
            verts = verts * 2
            dims = dims_from_flat(fmd, max_contacts=8, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)

    var out = List[_Hull]()
    var seen = List[Int]()
    for g in range(dims.get_ngeom()):
        var mid = Int(
            Float64(m.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_MESH_ID])
        )
        if mid < 0:
            continue
        var have = False
        for k in range(len(seen)):
            if seen[k] == mid:
                have = True
        if have:
            continue
        seen.append(mid)
        var o = mid * MODEL_MESH_META_SIZE
        var nv = Int(Float64(m.mesh_meta.data[o + MESH_META_IDX_VERTNUM]))
        var pa = Int(Float64(m.mesh_meta.data[o + MESH_META_IDX_POLYADR]))
        var pn = Int(Float64(m.mesh_meta.data[o + MESH_META_IDX_POLYNUM]))
        var maxpv = 0
        for p in range(pn):
            var po = (pa + p) * MODEL_MESH_POLY_SIZE
            var pv = Int(
                Float64(m.mesh_polys.data[po + MESH_POLY_IDX_VERTNUM])
            )
            if pv > maxpv:
                maxpv = pv
        out.append(_Hull(nv, pn, maxpv))
    return out^


def _check(
    label: String, got: List[_Hull],
    av: Int, ap: Int, aw: Int, bv: Int, bp: Int, bw: Int,
) raises:
    assert_true(
        len(got) == 2,
        label + ": the fixture must build two collidable meshes; got "
        + String(len(got)) + ". A 0 means no geom collides with them and no"
        " hull was ever built.",
    )
    print(
        "  " + label + "  a", got[0].nvert, got[0].npoly, got[0].maxpv,
        " b", got[1].nvert, got[1].npoly, got[1].maxpv,
    )
    assert_true(
        got[0].nvert == av and got[0].npoly == ap and got[0].maxpv == aw,
        label + ": mesh a is " + String(got[0].nvert) + "/"
        + String(got[0].npoly) + "/" + String(got[0].maxpv)
        + " against MuJoCo's " + String(av) + "/" + String(ap) + "/"
        + String(aw) + ". A hull with MORE vertices than MuJoCo's means the"
        " budget never reached qhull; a DIFFERENT count at the same budget"
        " means the option string is not `Q9 TA<n-4>`.",
    )
    assert_true(
        got[1].nvert == bv and got[1].npoly == bp and got[1].maxpv == bw,
        label + ": mesh b is " + String(got[1].nvert) + "/"
        + String(got[1].npoly) + "/" + String(got[1].maxpv)
        + " against MuJoCo's " + String(bv) + "/" + String(bp) + "/"
        + String(bw),
    )


comptime _BARE = String(
    "<mesh name=\"a\" file=\"moving_jaw_so101_gripper_part1_v1.stl\"/>"
    "<mesh name=\"b\" file=\"wrist_roll_follower_so101_gripper_part0_v1.stl\"/>"
)
comptime _E64 = String(
    "<mesh name=\"a\" maxhullvert=\"64\""
    " file=\"moving_jaw_so101_gripper_part1_v1.stl\"/>"
    "<mesh name=\"b\" maxhullvert=\"64\""
    " file=\"wrist_roll_follower_so101_gripper_part0_v1.stl\"/>"
)
comptime _E128 = String(
    "<mesh name=\"a\" maxhullvert=\"128\""
    " file=\"moving_jaw_so101_gripper_part1_v1.stl\"/>"
    "<mesh name=\"b\" maxhullvert=\"128\""
    " file=\"wrist_roll_follower_so101_gripper_part0_v1.stl\"/>"
)


def test_the_unbudgeted_hull_is_the_control() raises:
    """No `maxhullvert` — our full hull is MuJoCo's, to the vertex.

    ⚠ THIS ROW IS WHY THE OTHERS MEAN ANYTHING. It shares the STL, the loader,
    the dedup, qhull and the polygon merge with every arm below; if it moves,
    the defect is not the budget.
    """
    print("=== no maxhullvert (the control) ===")
    _check(
        String("full   "), _hulls(_xml(_BARE, String(""))),
        A_FULL_V, A_FULL_P, A_FULL_W, B_FULL_V, B_FULL_P, B_FULL_W,
    )
    print("  PASS")


def test_the_budget_decimates_the_hull() raises:
    """`maxhullvert="64"` on the element — 3428 vertices become 64."""
    print("=== maxhullvert on the <mesh> element ===")
    _check(
        String("64     "), _hulls(_xml(_E64, String(""))),
        A64_V, A64_P, A64_W, B64_V, B64_P, B64_W,
    )
    _check(
        String("128    "), _hulls(_xml(_E128, String(""))),
        A128_V, A128_P, A128_W, B128_V, B128_P, B128_W,
    )
    print("  PASS")


def test_the_class_chain_is_read() raises:
    """`<default><mesh maxhullvert>` alone, and an element overriding it.

    ⚠ THE FIRST HALF IS trossen_wxai'S ONLY SPELLING and the second is
    robotstudio_so101's. An element-only read passes the second and fails the
    first; a default-only read does the reverse.
    """
    print("=== the <default> chain ===")
    _check(
        String("dflt 64"),
        _hulls(_xml(_BARE,
                    String("<default><mesh maxhullvert=\"64\"/></default>"))),
        A64_V, A64_P, A64_W, B64_V, B64_P, B64_W,
    )
    # The element wins over the class — so101's own arrangement, inverted so a
    # parser that took the default unconditionally would land on 128.
    _check(
        String("el>dflt"),
        _hulls(_xml(_E64,
                    String("<default><mesh maxhullvert=\"128\"/></default>"))),
        A64_V, A64_P, A64_W, B64_V, B64_P, B64_W,
    )
    print("  PASS")


def test_a_budget_below_four_is_rejected() raises:
    """`xml_native_reader.cc:1914` raises; so must we.

    A budget of 3 would ask qhull for `TA-1`, which is not "no budget" — it is
    an option qhull reads as its own default and the model would silently get
    an unlimited hull.
    """
    print("=== maxhullvert < 4 ===")
    var bad = _xml(
        String("<mesh name=\"a\" maxhullvert=\"3\""
               " file=\"moving_jaw_so101_gripper_part1_v1.stl\"/>"
               "<mesh name=\"b\""
               " file=\"wrist_roll_follower_so101_gripper_part0_v1.stl\"/>"),
        String(""),
    )
    var raised = False
    try:
        _ = parse_xml_full(bad, String("."))
    except:
        raised = True
    assert_true(
        raised,
        "`maxhullvert=\"3\"` must be rejected — MuJoCo raises \"maxhullvert"
        " must be larger than 3\" and a silent -1 hands qhull `TA-1`.",
    )
    print("  PASS")


def test_the_declarations_are_still_seen() raises:
    """Plumbing, NOT the gate — the count over the document.

    ⚠ THIS ASSERTION WAS TRUE FOR MONTHS WHILE EVERY BUDGETED HULL WAS WRONG.
    It is kept because it is the only thing that separates "this model never
    declared it" from "the `<default>` chain lost the declaration", and it is
    labelled so nobody reads a green row here as a working feature.
    """
    print("=== declarations counted (plumbing) ===")
    var t = _count(WXAI)
    var so = _count(SO101)
    var g = _count(DEXEE)
    print("  trossen_wxai", t, " robotstudio_so101", so,
          "  shadow_dexee (none)", g)
    assert_equal(t, 1, "trossen_wxai declares it once, in a <default>")
    assert_equal(so, 4, "robotstudio_so101: one default, three elements")
    assert_equal(g, 0, "shadow_dexee declares none")
    print("  PASS")


def _count(path: String) raises -> Int:
    var src = read_model_source(path)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    return fmd.unhonoured_maxhullvert


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
