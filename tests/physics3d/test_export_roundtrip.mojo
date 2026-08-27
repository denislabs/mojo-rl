"""Flattened export: writer -> parser -> the SAME model, and MuJoCo agrees — S5.

WHY EXPORT SERIALISES THE MODEL, NOT THE DOCUMENT
=================================================
The scene document is already MJCF, so saving it is one `open`. But S3's
fast-path edits live in the `FlatModelDef` — the document has nowhere to put
"geom 4's radius is now 0.077". Exporting the document would silently drop
every edit the user just made: the same drift the two-tier loop's
byte-identity gate exists to prevent, arriving at the last possible moment.

So export writes the RECORD, and this gate is the round trip the plan asks
for, in both directions:

* **writer -> our parser -> identical `FlatModelDef`.** A field the writer
  forgets comes back as its default, which is a number, not an error. Only a
  record-for-record comparison sees it.
* **writer -> MuJoCo.** Our parser reading our writer is a closed loop: any
  attribute we spell wrongly in a way we also READ wrongly cancels out
  perfectly. MuJoCo is the only thing that can catch that.

⚠ THE EDIT IS APPLIED BEFORE EXPORT ON PURPOSE. Round-tripping an unedited
model would pass even if export ignored the record entirely and re-read the
source file — which is exactly the bug this is guarding against.

⚠ THE MuJoCo HALF IS A SECOND STEP, because Mojo cannot call it. This gate
writes the export to `/tmp/physics3d_export_check.xml`; run

    pixi run python scripts/check_export_vs_mujoco.py

after it. Splitting the two is not ideal, and the alternative — embedding a
golden — would go stale silently, which is worse.

Run: pixi run mojo run -I . tests/physics3d/test_export_roundtrip.mojo
"""

from mojo_rl.physics3d.fields import Model, DynDims
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, read_model_source,
)
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.studio.writer import to_mjcf, unwritable
from mojo_rl.physics3d.studio.edit import (
    Edit, apply_edit, TARGET_GEOM, F_SIZE_0, F_POS_X, F_FRICTION,
)

comptime DT = DType.float64
comptime MODEL = String("mojo_rl/envs/walker2d/assets/walker2d.xml")


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

    def near(mut self, got: Float64, want: Float64, tol: Float64, msg: String):
        self.checks += 1
        if abs(got - want) <= tol:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg, "— want", want, "got", got)


def main() raises:
    var t = Tally()
    print("=== flattened export round trip ===")

    var src = read_model_source(MODEL)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var dims = dims_from_flat(fmd, max_contacts=64)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)

    # ⚠ EDIT FIRST. See the header: an unedited round trip passes even if the
    # writer ignored the record and re-read the file.
    apply_edit(fmd, m, Edit(TARGET_GEOM, 2, F_SIZE_0, 0.0777))
    apply_edit(fmd, m, Edit(TARGET_GEOM, 3, F_POS_X, -0.321))
    apply_edit(fmd, m, Edit(TARGET_GEOM, 4, F_FRICTION, 1.75))

    t.truth(unwritable(fmd).byte_length() == 0,
            "walker2d uses only sections the writer emits")

    var text = to_mjcf(fmd, String("exported"))
    var back = parse_xml_full(text, src[1])

    # ── counts ───────────────────────────────────────────────────────────
    print("--- counts survive ---")
    t.truth(len(back.bodies) == len(fmd.bodies),
            String("nbody ", len(back.bodies) + 1))
    t.truth(len(back.geoms) == len(fmd.geoms),
            String("ngeom ", len(back.geoms)))
    t.truth(len(back.joints) == len(fmd.joints),
            String("njoint ", len(back.joints)))
    t.truth(len(back.actuators) == len(fmd.actuators),
            String("nact ", len(back.actuators)))

    # ── the records, field by field ──────────────────────────────────────
    # ⚠ A FORGOTTEN FIELD COMES BACK AS ITS DEFAULT — a number, not an error.
    # Only comparing every field sees it, which is why this loops rather than
    # spot-checking the three that were edited.
    print("--- geoms, field for field ---")
    var bad = 0
    var first = String("")
    for i in range(len(fmd.geoms)):
        ref a = fmd.geoms[i]
        ref b = back.geoms[i]
        var d = 0
        if a.geom_type != b.geom_type:
            d += 1
        if abs(a.pos_x - b.pos_x) > 1e-12 or abs(a.pos_y - b.pos_y) > 1e-12 \
                or abs(a.pos_z - b.pos_z) > 1e-12:
            d += 1
        if abs(a.radius - b.radius) > 1e-12 \
                or abs(a.half_length - b.half_length) > 1e-12:
            d += 1
        if abs(a.half_x - b.half_x) > 1e-12 \
                or abs(a.half_y - b.half_y) > 1e-12 \
                or abs(a.half_z - b.half_z) > 1e-12:
            d += 1
        if abs(a.friction - b.friction) > 1e-12:
            d += 1
        if a.condim != b.condim or a.contype != b.contype \
                or a.conaffinity != b.conaffinity or a.group != b.group:
            d += 1
        if abs(a.rgba_r - b.rgba_r) > 1e-12 \
                or abs(a.rgba_a - b.rgba_a) > 1e-12:
            d += 1
        if d > 0:
            bad += 1
            if first.byte_length() == 0:
                first = String("geom ") + String(i) + " ("
                first += String(len(fmd.geom_names) > i)
                first += ") differs in " + String(d) + " group(s)"
    if bad > 0:
        print("      ", first)
    t.truth(bad == 0,
            String("all ", len(fmd.geoms), " geoms round-trip (", bad,
                   " differ)"))

    # ⚠⚠ THE EDITS THEMSELVES. If these three survive but the loop above had
    # passed vacuously (say the writer re-read the file), they would be the
    # ones to fail — they exist nowhere but the record.
    t.near(back.geoms[2].radius, 0.0777, 1e-12, "the EDITED radius exported")
    t.near(back.geoms[3].pos_x, -0.321, 1e-12,
           "the EDITED position exported (and it is NEGATIVE)")
    t.near(back.geoms[4].friction, 1.75, 1e-12, "the EDITED friction exported")

    print("--- joints and bodies ---")
    var jbad = 0
    for i in range(len(fmd.joints)):
        ref a = fmd.joints[i]
        ref b = back.joints[i]
        if a.jnt_type != b.jnt_type or a.body_id != b.body_id \
                or abs(a.axis_x - b.axis_x) > 1e-12 \
                or abs(a.axis_z - b.axis_z) > 1e-12 \
                or a.is_limited != b.is_limited \
                or abs(a.damping - b.damping) > 1e-12 \
                or abs(a.armature - b.armature) > 1e-12:
            jbad += 1
    t.truth(jbad == 0, String("all ", len(fmd.joints), " joints round-trip"))
    var bbad = 0
    for i in range(len(fmd.bodies)):
        ref a = fmd.bodies[i]
        ref b = back.bodies[i]
        if a.parent != b.parent or abs(a.pos_x - b.pos_x) > 1e-12 \
                or abs(a.pos_z - b.pos_z) > 1e-12:
            bbad += 1
    t.truth(bbad == 0,
            String("all ", len(fmd.bodies), " bodies keep parent and pose"))

    # ── names, which is what made a writer possible at all ────────────────
    print("--- names ---")
    var nbad = 0
    for i in range(len(fmd.body_names)):
        if i < len(back.body_names) and fmd.body_names[i] != back.body_names[i]:
            nbad += 1
    t.truth(nbad == 0, "body names survive the round trip")
    t.truth(text.find("torso") != -1,
            "the exported text names its bodies (a NAMELESS export is not"
            " acceptable — keyframes and sensors key on names)")

    # ── the writer refuses rather than losing a section ───────────────────
    print("--- refusal ---")
    var hum = parse_xml_full(
        expand_mjcf(read_model_source(
            String("mojo_rl/envs/humanoid/assets/humanoid.xml")
        )[0], String("mojo_rl/envs/humanoid/assets")),
        String("mojo_rl/envs/humanoid/assets"),
    )
    # humanoid has two fixed tendons.
    t.truth(unwritable(hum).byte_length() > 0,
            String("humanoid reports unwritable sections:", unwritable(hum)))
    var refused = False
    try:
        _ = to_mjcf(hum, String("x"))
    except e:
        refused = String(e).find("tendon") != -1
    t.truth(refused,
            "and the writer RAISES naming them, rather than emitting a file"
            " that loads and is a DIFFERENT model")

    # ── hand the file to the MuJoCo half ──────────────────────────────────
    var ef = open("/tmp/physics3d_export_check.xml", "w")
    ef.write(text)
    ef.close()
    var sf = open("/tmp/physics3d_export_source.txt", "w")
    sf.write(MODEL)
    sf.close()
    print("  wrote /tmp/physics3d_export_check.xml — now run")
    print("    pixi run python scripts/check_export_vs_mujoco.py")

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_export_roundtrip: " + String(t.fails) + " failed")
