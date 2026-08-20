"""Dropped-in PRIMITIVES, against MuJoCo — studio S4.

WHY THIS EXISTS
===============
A prop the user creates has no asset file, so it is written INLINE into the
scene's `<worldbody>` rather than through `<attach>` (which resolves against
`<asset><model file=>` and therefore needs one). That makes the writer the
only thing standing between "the user clicked box" and a model — no parser, no
expander, nothing else to catch a malformed tag.

⚠⚠ AND MASS/INERTIA ARE NOT WRITTEN AT ALL. MuJoCo derives both from the
geom's shape and density when a body has no `<inertial>`, and so does our
builder (`model/inertia_from_geom.mojo`). Emitting numbers we computed
ourselves would be a SECOND implementation of that derivation — checkable
against MuJoCo only by accident, and wrong in a way that looks like a physics
difference rather than an authoring bug. `density` is the knob; the compiler
does the rest.

So this gate compares OUR parse of the written scene against MuJoCo's, on the
numbers nobody wrote down: **the derived masses**.

⚠ THE SIZE ATTRIBUTE'S LENGTH IS PER TYPE and MuJoCo enforces it — a sphere
with three numbers is a load error, not a rounding. All four kinds are here
for that reason.

Run: pixi run mojo run -I . tests/physics3d/test_props_vs_mujoco.mojo
"""

from mojo_rl.physics3d.fields import Model, DynDims
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime,
)
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE, BODY_IDX_MASS,
)
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.studio.scene import (
    SceneDoc, scene_from_base,
    PROP_BOX, PROP_SPHERE, PROP_CAPSULE, PROP_CYLINDER,
)

comptime DT = DType.float64


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

    def near(mut self, got: Float64, want: Float64, tol: Float64,
             msg: String):
        self.checks += 1
        if abs(got - want) <= tol:
            print("  ok:", msg, "=", got)
        else:
            self.fails += 1
            print("  FAIL:", msg, "— MuJoCo", want, "we", got)


def build_scene() -> SceneDoc:
    var d = SceneDoc()
    d.base_xml = String('  <compiler angle="radian"/>')
    d.base_world = String(
        '    <geom name="floor" type="plane" size="5 5 0.1"/>'
    )
    _ = d.add_prop(PROP_BOX, 0.05, 0.06, 0.07, 0.0, 0.0, 0.5)
    _ = d.add_prop(PROP_SPHERE, 0.04, 0.0, 0.0, 0.2, 0.0, 0.5)
    _ = d.add_prop(PROP_CAPSULE, 0.03, 0.08, 0.0, 0.4, 0.0, 0.5)
    _ = d.add_prop(PROP_CYLINDER, 0.03, 0.05, 0.0, 0.6, 0.0, 0.5)
    _ = d.duplicate_prop(String("box1"))
    return d^


def main() raises:
    var t = Tally()
    print("=== dropped-in props vs MuJoCo 3.10.0 ===")
    var d = build_scene()
    var text = d.to_mjcf(String("props"))
    var fmd = parse_xml_full(expand_mjcf(text, String("")), String(""))
    var dims = dims_from_flat(fmd, max_contacts=64)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)

    var nq = 0
    var nv = 0
    for j in fmd.joints:
        nq += j.nq
        nv += j.nv

    print("--- counts ---")
    # ⚠ THE FLOOR COMES FROM `base_xml`'S OWN `<worldbody>` AND THE PROPS FROM
    # A SECOND ONE. MJCF allows repeated sections and MuJoCo merges them; this
    # arm is what proves OUR parser does too, because a writer that emits two
    # is only correct if both are read.
    t.truth(len(fmd.bodies) + 1 == 6,
            String("nbody 6 (world + 5 props) — got ", len(fmd.bodies) + 1))
    t.truth(len(fmd.geoms) == 6,
            String("ngeom 6 (floor + 5) — the SECOND <worldbody> was read"
                   " too; got ", len(fmd.geoms)))
    t.truth(nq == 35 and nv == 30,
            String("nq 35 / nv 30, five free joints — got ", nq, "/", nv))

    # ── the derived masses, which nobody wrote down ───────────────────────
    print("--- mass, DERIVED from shape x density ---")
    # ⚠⚠ READ FROM THE **MODEL**, NOT THE RECORD. `FlatModelDef.bodies[i].mass`
    # is what the XML SAID — and a prop writes no `mass=`, so it is
    # `BodyData`'s default of 1.0 for every one of them. The derivation from
    # shape x density runs in the BUILDER and lands in the packed body record.
    # The first draft of this file read the record and reported 1.0 five times
    # while the model was correct; the same confusion was live in the studio's
    # inspector, which showed every prop as 1 kg.
    var mass = List[Float64]()
    for b in range(1, dims.get_nbody()):
        mass.append(Float64(m.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_MASS]))
    # MuJoCo 3.10.0, density 1000:
    #   box   0.05x0.06x0.07 half-extents -> 1.68
    #   sphere r=0.04                     -> 0.26808257
    #   capsule r=0.03 halflen=0.08       -> 0.56548668
    #   cylinder r=0.03 halflen=0.05      -> 0.28274334
    t.near(mass[0], 1.68, 1e-6, "box mass")
    t.near(mass[1], 0.26808257, 1e-6, "sphere mass")
    t.near(mass[2], 0.56548668, 1e-6, "capsule mass")
    t.near(mass[3], 0.28274334, 1e-6, "cylinder mass")

    # ⚠ NON-VACUITY: if the writer emitted a `mass` attribute, these would
    # agree because we WROTE them, not because the derivation matched. The
    # text must not contain one.
    t.truth(text.find("mass=") == -1,
            "the writer emits NO mass= — the numbers above are derived, not"
            " transcribed")
    t.truth(text.find("density=") != -1, "density IS written (it is the knob)")

    # ── duplicate ─────────────────────────────────────────────────────────
    print("--- duplicate ---")
    t.near(mass[4], 1.68, 1e-6, "the copy has the box's mass")
    # ⚠ OFFSET, NOT COINCIDENT. Two free bodies at one pose interpenetrate and
    # the solver launches them apart on step 1 — which reads as a physics bug
    # rather than as a copy placed on top of its original.
    t.truth(fmd.bodies[4].pos_x != fmd.bodies[0].pos_x,
            String("the copy is OFFSET from the original (",
                   fmd.bodies[0].pos_x, " vs ", fmd.bodies[4].pos_x, ")"))
    t.truth(fmd.body_names[5] == "box2",
            String("the copy is uniquely named — got '", fmd.body_names[5],
                   "'"))

    # ── delete ────────────────────────────────────────────────────────────
    print("--- delete ---")
    var d2 = build_scene()
    d2.remove_prop(String("sphere1"))
    var fmd2 = parse_xml_full(
        expand_mjcf(d2.to_mjcf(String("props")), String("")), String("")
    )
    t.truth(len(fmd2.bodies) == 4,
            String("removing one prop leaves four — got ", len(fmd2.bodies)))
    var still = False
    for n in fmd2.body_names:
        if n == "capsule1":
            still = True
    t.truth(still, "the OTHER props keep their identity after a delete")

    # ── materialize on override, at ASSET granularity — V2.5 ──────────────
    # ⚠⚠ A SCENE REFERENCES ITS BASE BY PATH, and `<attach>` cannot express a
    # per-instance change (plan §11.1). So a scene written while the robot has
    # been edited reopens as the ORIGINAL robot — a composition pointing at
    # the wrong model, with nothing to say so. The studio writes the edited
    # copy and re-points the entry at it; this is that re-point.
    print("--- retarget_asset ---")
    var d3 = scene_from_base(String("mojo_rl/envs/ant/assets/ant.xml"))
    var before3 = d3.to_mjcf(String("s"))
    t.truth(before3.find(String("assets/ant.xml\"")) != -1,
            "the scene names its base model")
    t.truth(d3.retarget_asset(String("mojo_rl/envs/ant/assets/ant.xml"),
                              String("/tmp/edited_ant.xml")),
            "retarget_asset finds the entry")
    var after3 = d3.to_mjcf(String("s"))
    t.truth(after3.find(String("/tmp/edited_ant.xml")) != -1,
            "and the scene now names the EDITED file")
    # ⚠ NON-VACUITY: the original path must be GONE, not merely joined by the
    # new one — a scene naming both would load whichever MuJoCo saw first.
    t.truth(after3.find(String("mojo_rl/envs/ant/assets/ant.xml")) == -1,
            "and no longer names the original")
    # ⚠ AND THE CONTROL: a path the table does not hold must report False,
    # or the studio would print "pointed the scene at it" having done nothing.
    t.truth(not d3.retarget_asset(String("nope.xml"), String("x.xml")),
            "an unknown path reports False (control)")

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_props_vs_mujoco: " + String(t.fails) + " failed")
