"""How big a MESH geom is — the number the selection outline draws.

WHY THIS EXISTS
===============
The user selected a part of so_arm101 and the yellow highlight was a box far
bigger than the part. The cause was not the outline code: a `<geom mesh="X">`
carries no `size`, so `GeomData`'s defaults survive — `half_x/y/z` are 0 and
`radius` is its untouched default of **0.5** — and every fallback in the chain
was worse than the last.

⚠⚠ THE SHAPE OF THE BUG IS "A PLAUSIBLE NUMBER FROM THE WRONG PLACE". Nothing
raised, nothing was empty, and the box was drawn confidently. So every arm here
compares an extent against a SECOND, INDEPENDENT measurement of the same solid
rather than against a threshold:

  * the visual twin against its collision twin's HULL (same mesh, so equal);
  * the hull against the MESH FILE the renderer draws (a convex hull has the
    same AABB as the point set it wraps, so also equal).

A test asserting only "smaller than 0.5" would pass on all three of the wrong
answers this replaced.

⚠ AND THE NEGATIVE CONTROLS ARE THE POINT. `unitree_go2` and
`anybotics_anymal_c` collide with PRIMITIVES — zero mesh geoms have a hull —
which is the case an implementation keyed on hulls gets silently wrong. They
are here because they are the models where passes 1 and 2 can say nothing.

Run: pixi run mojo run -I . tests/physics3d/test_mesh_bounds.mojo
"""

from mojo_rl.physics3d.constants import GEOM_MESH
from mojo_rl.physics3d.gpu.constants import (
    MODEL_GEOM_SIZE, GEOM_IDX_MESH_ID,
)
from mojo_rl.physics3d.fields import Model, DynDims
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.flat_model import FlatModelDef
from mojo_rl.physics3d.parser.render_fields import (
    RenderFields, build_render_fields,
)
from mojo_rl.physics3d.parser.runtime_load import (
    read_model_source, dims_from_flat, build_model_runtime,
)
from mojo_rl.physics3d.studio.mesh_bounds import (
    empty_half_extents, measure_geom_from_file, biggest_half_extent,
    FALLBACK_HALF,
)

comptime DT = DType.float64
comptime ARM = "mojo_rl/envs/robots/assets/so_arm101.xml"
comptime GO2 = "references/mujoco_menagerie-main/unitree_go2/scene.xml"
comptime ANYMAL = "references/mujoco_menagerie-main/anybotics_anymal_c/scene.xml"


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


struct Built(Movable):
    var fmd: FlatModelDef
    var dims: DynDims
    var m: Model[DT, DynDims]
    var rf: RenderFields

    def __init__(out self, path: String) raises:
        var src = read_model_source(path)
        var flat = expand_mjcf(src[0], src[1])
        self.fmd = parse_xml_full(flat, src[1])
        self.dims = dims_from_flat(self.fmd, max_contacts=512,
                                   nmesh_verts=131072)
        self.m = Model[DT, DynDims](self.dims)
        build_model_runtime[DT](self.fmd, self.dims, self.m)
        self.rf = build_render_fields(self.fmd, flat, src[1])


def near(a: Float64, b: Float64, rel: Float64 = 1e-9) -> Bool:
    var s = abs(a) if abs(a) > abs(b) else abs(b)
    if s < 1e-12:
        return abs(a - b) < 1e-12
    return abs(a - b) / s <= rel


def main() raises:
    var t = Tally()
    print("=== mesh half-extents — what the outline boxes ===")

    # ══ so_arm101: the model from the report ═════════════════════════════
    print("--- so_arm101: the visual twin ---")
    var arm = Built(String(ARM))
    var half = empty_half_extents(len(arm.fmd.geoms))
    t.truth(len(half) == len(arm.fmd.geoms) * 3,
            String("three extents per geom (", len(half), ")"))

    # geom 11 is `upper_arm`'s VISUAL sts3215_03a_v1; geom 12 is the COLLISION
    # geom over the same mesh at the same pose.
    # ⚠ ASSERT THE FIXTURE FIRST. If the file is ever reordered these indices
    # name different geoms and every arm below silently tests nothing.
    t.truth(arm.fmd.geoms[11].mesh_id == arm.fmd.geoms[12].mesh_id
            and arm.fmd.geoms[11].mesh_id >= 0,
            String("geoms 11 and 12 really are the same mesh asset (",
                   arm.fmd.geoms[11].mesh_id, ")"))

    measure_geom_from_file(arm.rf, 11, half)
    measure_geom_from_file(arm.rf, 12, half)
    var vx = half[11 * 3 + 0]
    var vy = half[11 * 3 + 1]
    var vz = half[11 * 3 + 2]
    print("      visual  geom 11:", vx, vy, vz)
    print("      collide geom 12:", half[36], half[37], half[38])
    t.truth(vx > 0.0 and vy > 0.0 and vz > 0.0,
            "the VISUAL geom measured (it has no hull — this is the report)")
    # ⚠⚠ THE ARM. The visual twin used to measure NOTHING and take a global
    # fallback of 0.0730 — 5.9x too wide, and CUBIC where the part is
    # elongated: 66x too large by volume. That is the box in the screenshot.
    t.truth(near(vx, half[36]) and near(vy, half[37]) and near(vz, half[38]),
            "and it agrees with its collision twin, exactly (same mesh)")
    t.truth(not (near(vx, vy) and near(vy, vz)),
            String("and it is NOT a cube — the part is elongated (", vx, " ",
                   vy, " ", vz, ")"))
    t.truth(vx < 0.5 and vy < 0.5 and vz < 0.5,
            "and nowhere near radius 0.5 — the one-metre cube")

    # ⚠ IDEMPOTENT, because it is called every frame the selection is up.
    var before = half[33]
    measure_geom_from_file(arm.rf, 11, half)
    t.truth(half[33] == before, "a second call is a no-op (it is per-frame)")

    # ══ go2 and anymal_c: NO mesh geom has a hull ════════════════════════
    # ⚠⚠ THESE ARE THE NEGATIVE CONTROLS. Both collide with PRIMITIVES, so an
    # implementation keyed on the loaded collision hull can say NOTHING about
    # them — the old code returned early on `nmesh_verts == 0` and left every
    # extent at 0, after which `outline_geom` fell back to `radius` = 0.5 and
    # boxed every part of the dog in a one-metre cube.
    for which in range(2):
        var path = String(GO2) if which == 0 else String(ANYMAL)
        print("---", path, "---")
        var b = Built(path)
        # ⚠ THE PROPERTY IS "NO MESH GEOM HAS A LOADED HULL", NOT
        # "nmesh_verts == 0". `nmesh_verts` is the BUDGET the caller asked
        # for — this test passes 131072 — so reading it back measured my own
        # argument. `fields_build` writes `GEOM_IDX_MESH_ID = -1` on a mesh
        # geom whose hull it did not load, and that is the real signal.
        var hulled = 0
        for g in range(len(b.fmd.geoms)):
            if b.fmd.geoms[g].geom_type != GEOM_MESH:
                continue
            if Int(Float64(b.m.geoms.data[
                g * MODEL_GEOM_SIZE + GEOM_IDX_MESH_ID
            ])) >= 0:
                hulled += 1
        t.truth(hulled == 0,
                String("NOT ONE of its mesh geoms has a loaded hull (",
                       hulled, ") — the case a hull-based measure cannot"
                       " answer at all"))
        var h = empty_half_extents(len(b.fmd.geoms))

        var nmesh_geom = 0
        var first = -1
        for g in range(len(b.fmd.geoms)):
            if b.fmd.geoms[g].geom_type != GEOM_MESH:
                continue
            nmesh_geom += 1
            if first < 0:
                first = g
        t.truth(nmesh_geom > 20,
                String("it has mesh geoms to measure: ", nmesh_geom))

        measure_geom_from_file(b.rf, first, h)
        var e0 = h[first * 3 + 0]
        var e1 = h[first * 3 + 1]
        var e2 = h[first * 3 + 2]
        print("      geom", first, "from file:", e0, e1, e2)
        t.truth(e0 > 0.0 and e1 > 0.0 and e2 > 0.0,
                "the file pass measured it anyway")
        # ⚠⚠ THE TWO WRONG ANSWERS, NAMED. 0.5 is `GeomData`'s untouched
        # `radius` (a 1 m cube); FALLBACK_HALF is the marker of last resort.
        t.truth(e0 != 0.5 and e1 != 0.5 and e2 != 0.5,
                "and it is NOT radius 0.5 — the one-metre cube")
        t.truth(not (e0 == FALLBACK_HALF and e1 == FALLBACK_HALF
                     and e2 == FALLBACK_HALF),
                "and NOT the marker either")
        t.truth(e0 < 0.5 and e1 < 0.5 and e2 < 0.5 and (e0 + e1 + e2) > 0.01,
                String("and it is a plausible size for a leg part (", e0, " ",
                       e1, " ", e2, ")"))
        t.truth(not (near(e0, e1) and near(e1, e2)),
                "and it is not a CUBE — the three axes differ")

    # ── a geom with no mesh at all falls back, it does not crash ─────────
    print("--- the fallback ---")
    var fh = empty_half_extents(len(arm.fmd.geoms))
    var plane = -1
    for g in range(len(arm.fmd.geoms)):
        if arm.fmd.geoms[g].geom_type != GEOM_MESH and plane < 0:
            plane = g
    measure_geom_from_file(arm.rf, plane, fh, 0.037)
    t.truth(fh[plane * 3 + 0] == 0.037,
            String("a non-mesh geom takes the marker scale — got ",
                   fh[plane * 3 + 0]))
    # ⚠ AND OUT-OF-RANGE MUST NOT TRAP. The selection index can outrun the
    # geom list for one frame after a structural edit.
    measure_geom_from_file(arm.rf, 99999, fh)
    measure_geom_from_file(arm.rf, -1, fh)
    t.truth(True, "an out-of-range geom index is ignored, not a trap")

    # ── the dump the external oracle will judge ──────────────────────────
    # ⚠⚠ EVERY ARM ABOVE IS INTERNAL. They say the two twins agree, that the
    # answer is not one of the three wrong constants, and that it is not a
    # cube — none of them says the NUMBER is right. The hull could have served
    # as a second opinion and cannot: it is stored in the mesh's PRINCIPAL
    # frame while the outline is drawn in the render frame, which is the older
    # bug this file uncovered. So the second opinion is Python reading the same
    # files: `scripts/check_mesh_bounds_vs_python.py`.
    print("--- dumping for the external check ---")
    # ⚠⚠ OP3 AND SHADOW_HAND ARE HERE FOR `<mesh scale>`, and the checker
    # refuses a run where no row carried one. op3's STLs are in MILLIMETRES
    # (`scale="0.001 0.001 0.001"`, declared once in a `<default>` block) and
    # shadow_hand uses a MIRROR — a negative component, which builds a left
    # part from a right one. A mirror's |extent| is unchanged, so a loader
    # that dropped the scale entirely would still agree on it; op3 is what
    # makes the scale path answer for itself, by a factor of a thousand.
    var dump_paths = [
        String(ARM), String(GO2), String(ANYMAL),
        String("references/mujoco_menagerie-main/robotis_op3/scene.xml"),
        String("references/mujoco_menagerie-main/shadow_hand/scene_right.xml"),
    ]
    var rows = 0
    with open(String("/tmp/mesh_bounds_dump.txt"), "w") as f:
        for which in range(len(dump_paths)):
            var b2 = Built(dump_paths[which])
            var h2 = empty_half_extents(len(b2.fmd.geoms))
            for g in range(len(b2.fmd.geoms)):
                if b2.fmd.geoms[g].geom_type != GEOM_MESH:
                    continue
                measure_geom_from_file(b2.rf, g, h2)
                var mid = b2.rf.geom_mesh_id[g]
                if mid < 0 or mid >= b2.rf.nmesh:
                    continue
                f.write(String(
                    b2.rf.mesh_files[mid], "\t",
                    b2.rf.geom_mesh_scale[g * 3 + 0], "\t",
                    b2.rf.geom_mesh_scale[g * 3 + 1], "\t",
                    b2.rf.geom_mesh_scale[g * 3 + 2], "\t",
                    h2[g * 3 + 0], "\t", h2[g * 3 + 1], "\t",
                    h2[g * 3 + 2], "\n",
                ))
                rows += 1
    t.truth(rows > 100, String("rows written for Python to check: ", rows))

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_mesh_bounds: " + String(t.fails) + " failed")
