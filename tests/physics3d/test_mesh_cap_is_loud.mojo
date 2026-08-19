"""`MAX_GPU_MESHES` — enough headroom, and FATAL rather than silent past it.

WHY THIS EXISTS
===============
A collidable mesh past `MAX_GPU_MESHES` gets a hull built and an id assigned,
and then no `mesh_meta` row. Every consumer reads vertadr/vertnum 0 and
collides against an EMPTY mesh — wrong physics that RUNS. The cap used to be
16 and announced the overflow with a printed `ERROR:`, which in a viewer
scrolls away while the model keeps simulating.

This tree has paid for the same shape twice: the separate 16-asset `<mesh>`
cap in `full_parser` left SO-ARM100's moving jaw without its two contact
surfaces, and it took a per-geom `rbound` diff against MuJoCo to notice.

Two arms, and they test different things:

1. **Headroom.** The cap is 256 now; the mesh-heaviest models available here
   need single digits. If this arm ever gets close, raise the constant — it
   sizes one `[N, 4]` table and nothing else.
2. **The overflow is fatal.** Built from a SYNTHETIC model, because no real
   one comes near: 300 `<mesh>` assets, all pointing at the same small STL
   under different names, each on its own collidable geom. That is the only
   way to reach the branch, and a branch no test can reach is a branch that
   will be wrong when it fires.

Run: pixi run mojo run -I . tests/physics3d/test_mesh_cap_is_loud.mojo
"""

from mojo_rl.physics3d.fields import Model, DynDims
from mojo_rl.physics3d.gpu.constants import MAX_GPU_MESHES, MODEL_MESH_META_SIZE
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, read_model_source,
)
from mojo_rl.physics3d.parser.full_parser import parse_xml_full

comptime DT = DType.float64

comptime STL = String("mojo_rl/envs/robots/assets/so_arm100/Base.stl")


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


def loaded_meshes(path: String, verts: Int) raises -> Int:
    """How many `mesh_meta` rows a real model actually fills."""
    var src = read_model_source(path)
    var fmd = parse_xml_full(src[0], src[1])
    var dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=verts)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var n = 0
    for i in range(MAX_GPU_MESHES):
        if Int(Float64(m.mesh_meta.data[i * MODEL_MESH_META_SIZE + 1])) > 0:
            n += 1
    return n


def synth(n_meshes: Int) -> String:
    """A model with `n_meshes` DISTINCT collidable mesh assets, one file.

    ⚠ DISTINCT NAMES, ONE FILE. The cap counts mesh ASSETS, not geoms and not
    files, so reusing the path is what keeps this cheap — 300 assets of the
    same 5 KB hull rather than 300 files on disk.
    """
    var s = String('<mujoco model="synth"><asset>')
    for i in range(n_meshes):
        s += '<mesh name="m' + String(i) + '" file="' + STL + '"/>'
    s += "</asset><worldbody>"
    for i in range(n_meshes):
        s += (
            '<body name="b' + String(i) + '" pos="0 0 '
            + String(i) + '"><freejoint/>'
            + '<geom type="mesh" mesh="m' + String(i) + '"/></body>'
        )
    s += "</worldbody></mujoco>"
    return s^


def main() raises:
    var t = Tally()
    print("=== MAX_GPU_MESHES =", MAX_GPU_MESHES, "===")

    # ── 1. headroom on real models ────────────────────────────────────────
    print("--- what real models need ---")
    var so = loaded_meshes(
        String("mojo_rl/envs/robots/assets/so_arm100.xml"), 8192
    )
    var saw = loaded_meshes(
        String("mojo_rl/envs/metaworld/assets/sawyer_reach.xml"), 8192
    )
    print("    so_arm100", so, "  sawyer", saw)
    # ⚠ NON-VACUITY: a model that loads ZERO collidable meshes says nothing
    # about a cap on collidable meshes.
    t.truth(so > 0 and saw > 0,
            String("both fixtures load collidable meshes (", so, ", ", saw,
                   ") — the headroom arm is live"))
    var worst = so if so > saw else saw
    t.truth(worst * 4 <= MAX_GPU_MESHES,
            String("4x headroom over the heaviest model (", worst, " of ",
                   MAX_GPU_MESHES, ")"))

    # ── 2. the overflow is FATAL ──────────────────────────────────────────
    print("--- past the cap ---")
    # Just UNDER the cap must still build, or the arm below would pass on a
    # model that fails for some unrelated reason.
    var under = MAX_GPU_MESHES - 4
    var built = True
    try:
        var fmd = parse_xml_full(synth(under), String(""))
        var dims = dims_from_flat(fmd, max_contacts=8, nmesh_verts=1 << 21)
        var m = Model[DT, DynDims](dims)
        build_model_runtime[DT](fmd, dims, m)
    except e:
        built = False
        print("    (under-cap model failed:", e, ")")
    t.truth(built,
            String("a model with ", under, " collidable meshes still builds"))

    var raised = False
    var named = False
    try:
        var fmd2 = parse_xml_full(synth(MAX_GPU_MESHES + 20), String(""))
        var dims2 = dims_from_flat(fmd2, max_contacts=8, nmesh_verts=1 << 22)
        var m2 = Model[DT, DynDims](dims2)
        build_model_runtime[DT](fmd2, dims2, m2)
    except e:
        raised = True
        # The message must carry BOTH numbers, or the reader cannot tell how
        # far over the cap they are and what to set it to.
        var msg = String(e)
        named = (
            msg.find("MAX_GPU_MESHES") != -1
            and msg.find(String(MAX_GPU_MESHES + 20)) != -1
        )
    t.truth(raised, "exceeding the cap RAISES (it used to print and continue)")
    t.truth(named, "the message names the cap AND the number needed")

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_mesh_cap_is_loud: " + String(t.fails) + " failed")
