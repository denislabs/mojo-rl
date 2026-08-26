"""`MakePolygons`' emission order IS MuJoCo's, because we call the same map.

    pixi run mojo run -I . tests/physics3d/test_polygon_order_is_mujocos.mojo

WHAT THIS PINS. `mjCMesh::MakePolygons` emits
`for (const auto& pair : mesh_polygons)` — an `std::unordered_map` ITERATION
ORDER. That order is OBSERVABLE: where `multicontact()` breaks a tie between two
coplanar candidate faces for one edge it takes the FIRST in polygon order, so
the choice lands in the contact POSITION. `hello_robot_stretch_3` and
`shadow_dexee` were each placing a manifold point millimetres off with the
depth and the normal EXACT — board rows #1 and #3, one mechanism.

⚠⚠ THE ORDER IS REPRODUCIBLE, AND THAT WAS MEASURED RATHER THAN ASSUMED. An
instrumented build of MuJoCo 3.10.0 (`MakePolygons` dumping its map's iteration
order) shows the real key order is exactly bucket-contiguous at the map's own
`bucket_count` — runs/distinct 1.000 on every mesh — i.e. it IS a libc++
iteration; and a locally built 3.10.0 agrees with the pixi wheel on the polygon
order of 84 of 85 `stretch_3` meshes, i.e. it is stable across builds.

So the faithful port is a CALL (`native/mrl_polyorder.cc`), exactly as it is for
the hull itself. Validated against that dump: our emitted key order matched
MuJoCo's 649/649 on `base_link_collision`.

⚠ WHAT THIS FILE CAN AND CANNOT SEE. It cannot re-run the instrumented build,
so it does NOT re-derive MuJoCo's order. It pins the two things that would
silently undo the work:

  1. the shim is REACHABLE and returns a permutation (not the fallback);
  2. that permutation is NOT the identity and NOT a plain reversal — if it were
     either, someone has replaced the map call with a rule, which is what this
     whole line of work established cannot be done.

⚠ THE FALLBACK IS THE FAILURE MODE TO FEAR. Without the dylib the engine emits
REVERSE first-seen order, which agrees with MuJoCo only ~70% of the time
(measured 68.7-73.8% over 5 meshes of 2 scenes, against 26.7-31.5% forward).
Nothing crashes; two Menagerie scenes just place a contact a few millimetres
off. `test_qhull_shim` covers the hull half of the same dylib.
"""

from std.math import abs
from std.testing import assert_true, assert_equal, TestSuite

from mojo_rl.physics3d.collision.qhull_native import (
    poly_order, qhull_shim_available,
)


def test_the_shim_is_reachable() raises:
    """If this fails everything below is testing the FALLBACK."""
    print("=== the shim is reachable ===")
    var ok = qhull_shim_available()
    print("  qhull_shim_available:", ok)
    assert_true(
        ok,
        "libmrl_qhull.dylib not found — the engine would silently fall back to"
        " REVERSE first-seen polygon order, which is a ~70% approximation of"
        " MuJoCo's. Build it with `pixi run build-qhull`.",
    )


def test_the_order_is_a_permutation() raises:
    """Every key comes back exactly once."""
    print("=== the returned order is a permutation ===")
    var n = 64
    var keys = List[Float64](capacity=2 * n)
    for i in range(n):
        # Keys are ROUNDED angle pairs, so integral doubles — the same shape
        # `MeshPolygonKey` produces, which is what makes their bit patterns
        # (and so their hashes) structured.
        keys.append(Float64(i - 30))
        keys.append(Float64((i * 7) % 40))
    var dst = List[Int32](length=n, fill=Int32(-1))
    var nw = poly_order(Pointer(to=keys[0]), n, Pointer(to=dst[0]))
    print("  keys", n, " returned", nw)
    assert_equal(nw, n, "the map dropped or invented entries")
    var seen = List[Bool](length=n, fill=False)
    for i in range(n):
        var v = Int(dst[i])
        assert_true(
            v >= 0 and v < n,
            "index " + String(v) + " out of range — the shim wrote garbage",
        )
        assert_true(not seen[v], "index " + String(v) + " returned twice")
        seen[v] = True


def test_it_is_neither_identity_nor_reversal() raises:
    """⚠ THE POINT OF THE WHOLE EXERCISE.

    A hash-table iteration order is not a rule. If this permutation ever
    becomes the identity or a plain reversal, the map call has been replaced by
    one — and this project measured, with an instrumented MuJoCo build, that no
    rule reproduces the reference (forward agrees 26.7-31.5%, reversed
    68.7-73.8%; only the map itself is exact).
    """
    print("=== it is a hash order, not a rule ===")
    var n = 64
    var keys = List[Float64](capacity=2 * n)
    for i in range(n):
        keys.append(Float64(i - 30))
        keys.append(Float64((i * 7) % 40))
    var dst = List[Int32](length=n, fill=Int32(-1))
    var nw = poly_order(Pointer(to=keys[0]), n, Pointer(to=dst[0]))
    assert_equal(nw, n)
    var ident = 0
    var revd = 0
    for i in range(n):
        if Int(dst[i]) == i:
            ident += 1
        if Int(dst[i]) == n - 1 - i:
            revd += 1
    print("  positions matching identity", ident, " reversal", revd, " of", n)
    assert_true(
        ident < n,
        "the order is the IDENTITY — the map call has been replaced by"
        " 'emit in insertion order', which is 26.7-31.5% right",
    )
    assert_true(
        revd < n,
        "the order is a plain REVERSAL — the map call has been replaced by the"
        " old ~70% approximation",
    )


def test_duplicates_collapse_like_mujocos_find_then_emplace() raises:
    """MuJoCo does `find` then `emplace`: only a FIRST sighting makes a node,
    and only nodes are iterated. A repeated key must not produce a second
    entry."""
    print("=== duplicate keys collapse ===")
    var keys = List[Float64]()
    for i in range(8):
        keys.append(Float64(i))
        keys.append(Float64(i * 2))
    # every key again, in a different order
    for i in range(7, -1, -1):
        keys.append(Float64(i))
        keys.append(Float64(i * 2))
    var n = 16
    var dst = List[Int32](length=n, fill=Int32(-1))
    var nw = poly_order(Pointer(to=keys[0]), n, Pointer(to=dst[0]))
    print("  16 insertions of 8 distinct keys -> entries", nw)
    assert_equal(
        nw, 8,
        "a repeated key created a second map entry; MuJoCo's `find`-then-"
        "`emplace` cannot do that, and an extra entry is an extra POLYGON",
    )
    for i in range(nw):
        assert_true(
            Int(dst[i]) < 8,
            "a duplicate's insertion index was returned; only the FIRST"
            " sighting exists as a node",
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
