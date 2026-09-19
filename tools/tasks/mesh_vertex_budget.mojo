"""The hull-vertex budget a scene needs — the number `NMESH_VERTS` restates.

    pixi run mojo run -I . tools/tasks/mesh_vertex_budget.mojo mojo_rl/tasks/scenes/so101_tower.xml

`fields_build` refuses a budget below what the COLLIDABLE hulls need, and its
error names the exact figure. This asks with a budget of one vertex and prints
that figure, so a config's `NMESH_VERTS` is a number read off the loader and
not a guess doubled until it fits (`so_arm101_xml.SO_ARM101_NMESH_VERTS` says
why a figure copied from `mjModel` is wrong: our hulls keep ~26% more).

⚠ THE FIRST RUN ON A NEW MESH IS SLOW — the hull is computed and cached
(`collision/hull_cache.mojo`); the second run reads the cache.
"""

from std.sys import argv

from mojo_rl.physics3d.fields import Model, DynDims
from mojo_rl.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)

comptime DT = DType.float64


def main() raises:
    var args = argv()
    if len(args) < 2:
        raise Error("usage: mesh_vertex_budget.mojo <scene.xml>")
    var path = String(args[1])
    var fmd = parse_model_runtime(path)
    var dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=1)
    var m = Model[DT, DynDims](dims)
    try:
        build_model_runtime[DT](fmd, dims, m)
        print(path, "needs no mesh vertices (no collidable mesh)")
    except e:
        var msg = String(e)
        var k = msg.find("at least ")
        if k < 0:
            raise e
        var rest = String(msg[byte = k + 9 :])
        var end = rest.find(" ")
        var n = Int(String(rest[byte = 0 : end]))
        # rounded up to a multiple of 512, as the SO-101 constant is
        var padded = ((n + 511) // 512) * 512
        print(path)
        print("  collidable hull vertices needed:", n)
        print("  NMESH_VERTS (rounded to 512):", padded)
