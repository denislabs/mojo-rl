"""Runtime parse + build cost per model — studio plan §11, item 2.

Sets the tier boundary in the studio's two-tier edit loop: a STRUCTURAL edit
recompiles the scene and rebuilds the model, so this number decides whether a
rebuild can sit on a click (yes if ~ms) or needs to be off the drag path.

min-of-N, and the four stages timed SEPARATELY because they have different
fixes: parse is text, dims is arithmetic, build is allocation + fill, spec is
the actuation bundle.

⚠ nmesh_verts CANNOT be derived before the build (the meshes load inside it),
so this doubles as a test of the retry-on-raise loop the plan proposes: start
at 0, double until the builder stops raising.
"""

from std.time import perf_counter_ns

from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.parser import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from mojo_rl.physics3d.parser.runtime_load import spec_fields_runtime

comptime DT = DType.float64
comptime REPS = 5


def bench_one(path: String, label: String) raises:
    # ── stage 1: parse (and find the mesh budget by retry) ───────────────
    var verts = 0
    var ok = False
    while not ok:
        try:
            var f0 = parse_model_runtime(path)
            var d0 = dims_from_flat(f0, max_contacts=64, nmesh_verts=verts)
            var m0 = Model[DT, DynDims](d0)
            build_model_runtime[DT](f0, d0, m0)
            ok = True
        except e:
            if verts == 0:
                verts = 4096
            elif verts < 1048576:
                verts = verts * 4
            else:
                print(label, "— GAVE UP at nmesh_verts =", verts, ":", e)
                return

    var t_parse = Int(1) << 62
    var t_dims = Int(1) << 62
    var t_build = Int(1) << 62
    var t_spec = Int(1) << 62
    var nq = 0
    var nv = 0
    var ngeom = 0
    var nbody = 0

    for _ in range(REPS):
        var a = perf_counter_ns()
        var fmd = parse_model_runtime(path)
        var b = perf_counter_ns()
        var dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=verts)
        var c = perf_counter_ns()
        var m = Model[DT, DynDims](dims)
        build_model_runtime[DT](fmd, dims, m)
        var d = perf_counter_ns()
        var sf = spec_fields_runtime[DT](fmd, dims)
        var e = perf_counter_ns()

        if b - a < t_parse:
            t_parse = b - a
        if c - b < t_dims:
            t_dims = c - b
        if d - c < t_build:
            t_build = d - c
        if e - d < t_spec:
            t_spec = e - d
        nq = dims.get_nq()
        nv = dims.get_nv()
        ngeom = dims.get_ngeom()
        nbody = dims.get_nbody()
        # keep them alive to the end of the iteration
        _ = sf^
        _ = m^

    var ms = Float64(1_000_000.0)
    var total = Float64(t_parse + t_dims + t_build + t_spec) / ms
    print(
        label, "| nq", nq, "nv", nv, "nbody", nbody, "ngeom", ngeom,
        "verts", verts,
    )
    print(
        "   parse", Float64(t_parse) / ms, "ms | dims", Float64(t_dims) / ms,
        "ms | build", Float64(t_build) / ms, "ms | spec",
        Float64(t_spec) / ms, "ms | TOTAL", total, "ms",
    )


def main() raises:
    print("=== runtime parse + build, min of", REPS, "===")
    bench_one("mojo_rl/envs/walker2d/assets/walker2d.xml", "walker2d ")
    bench_one("mojo_rl/envs/dm_control/assets/quadruped_run.xml", "quadruped")
    bench_one("mojo_rl/envs/humanoid/assets/humanoid.xml", "humanoid ")
    bench_one("mojo_rl/envs/robots/assets/so_arm100.xml", "so_arm100")
    bench_one("mojo_rl/envs/dm_control/assets/dog_stand_walk.xml", "dog      ")
