"""A `.family` composes to a scene MuJoCo loads — P1c's gate.

## WHY THE ORACLE IS MuJoCo AND NOT US

`TASK_LAYER_PLAN.md` §2.1 kept the scene document in MJCF precisely so this
gate is available: **MuJoCo loads the composed file unchanged**, compiling each
attached asset separately and attaching the RESULT, while we splice TEXT. Two
entirely different routes to one `mjModel`, which is the only kind of
comparison that catches a prefixer that missed a reference or a frame folded
twice. Checking our composer against our own parser would prove nothing —
`feedback_a_gate_that_shares_its_reference_implementation_is_blind`.

⚠ THIS GATE IS ONLY REACHABLE BECAUSE OF P1a. Before it, `ModelDefFromXML`
handed `parse_xml_full` the RAW text and `<attach>` went unexpanded — the
composed scene loaded as four bodies instead of the whole model, silently.
`tests/physics3d/test_expand_identity.mojo` is what holds that open.

## WHAT IT ASSERTS

1. **the budget arithmetic** — the composed model's counts are the base plus
   each slot's contribution, computed from the ASSETS rather than hardcoded,
   so adding a slot to the `.family` cannot silently pass;
2. **MuJoCo agrees, record for record** — nbody / njnt / nq / nv / ngeom;
3. **our parser agrees with MuJoCo** on the same file;
4. ⚠ **the parked slots touch NOTHING** — `ncon` at rest equals the base
   scene's. This is the check `gen_park_scenes.py` already carries for the P0
   probe, moved to where families are actually built. A parked slot in contact
   turns every task in the family into a different, slower problem, and no
   throughput curve would look wrong.

Run: pixi run mojo run -I . tests/tasks/test_family_compose_vs_mujoco.mojo
"""

from mojo_rl.tasks.spec import load_family, SLOT_FREE
from mojo_rl.tasks.family import (
    compose_family, park_pos, scene_path, SCENE_DIR, BASE_PREFIX,
)
from mojo_rl.physics3d.parser.runtime_load import parse_model_runtime
from mojo_rl.tasks.so101_tabletop_xml import So101TabletopModel


comptime FAMILY = String("mojo_rl/tasks/families/so101_tabletop.family")


def main() raises:
    print("=== .family -> composed scene, vs MuJoCo — P1c ===")
    var f = load_family(FAMILY)
    print("  family:", f.name, "| slots:", len(f.slots),
          "| free:", f.n_free_slots())

    # ⚠ THIS GATE DOES NOT WRITE THE SCENE — `tools/tasks/gen_family_scene.mojo`
    # does. A test that produces a checked-in build input makes the model
    # depend on whether the suite ran. It CHECKS instead, the way
    # `gen-dims-check` does.
    #
    # ⚠ AND THE PATH MATTERS: MuJoCo reads `<model file=>` relative to the
    # SCENE FILE's directory, so a composed scene dropped in /tmp looks for
    # `/tmp/mojo_rl/envs/...`. This gate found exactly that on its first run.
    var out = scene_path(f)
    var xml = compose_family(f, SCENE_DIR)
    var on_disk = String("")
    with open(out, "r") as fh:
        on_disk = fh.read()
    if on_disk != xml:
        raise Error(
            "family compose: '" + out + "' is STALE — the .family composes to"
            " something else. Run `pixi run gen-family-scenes`, then"
            " `pixi run gen-dims`."
        )
    print("  ok:", out, "is up to date with the .family")

    # ── the parked slots are SPREAD, not stacked ──────────────────────────
    # ⚠ Stacked slots interpenetrate, which is k*(k-1)/2 contact pairs from
    # objects that are supposed to be absent. Checked here rather than trusted
    # because the spacing is a constant someone could reasonably "tidy".
    var ok = True
    for i in range(len(f.slots)):
        for j in range(i + 1, len(f.slots)):
            var a = park_pos(f, i)
            var b = park_pos(f, j)
            var dx = a[0] - b[0]
            if dx < 0.0:
                dx = -dx
            if dx < 0.1:
                ok = False
    if not ok:
        raise Error(
            "family compose: two parked slots are within 10 cm of each other."
            " They would collide while parked, putting contacts into a scene"
            " whose parked half is supposed to be invisible."
        )
    print("  ok: parked slots are spread")

    # ── our parser reads the composed document ────────────────────────────
    # ⚠ THROUGH `parse_model_runtime`, which expands `<attach>`/`<frame>`.
    # If this returns a handful of bodies instead of the whole model, the
    # expansion did not run — the exact failure P1a fixed.
    var fmd = parse_model_runtime(out)
    print("  ours   : nbody", len(fmd.bodies), " njoint", len(fmd.joints),
          " ngeom", len(fmd.geoms))

    # The expected counts come from the ASSETS, not from a literal — so adding
    # a slot to the .family updates the expectation automatically and cannot
    # pass by being forgotten.
    var base = parse_model_runtime(f.base)
    var exp_bodies = len(base.bodies)
    var exp_joints = len(base.joints)
    var exp_geoms = len(base.geoms)
    for i in range(len(f.slots)):
        var a = parse_model_runtime(f.slots[i].asset)
        exp_bodies += len(a.bodies)
        exp_joints += len(a.joints)
        exp_geoms += len(a.geoms)
    # the scene's own floor geom, which `scene_from_base` adds
    exp_geoms += 1
    print("  expect : nbody", exp_bodies, " njoint", exp_joints,
          " ngeom", exp_geoms, " (base + every slot's asset + the floor)")

    var bad = 0
    if len(fmd.bodies) != exp_bodies:
        print("  FAIL: nbody", len(fmd.bodies), "!=", exp_bodies)
        bad += 1
    if len(fmd.joints) != exp_joints:
        print("  FAIL: njoint", len(fmd.joints), "!=", exp_joints)
        bad += 1
    if len(fmd.geoms) != exp_geoms:
        print("  FAIL: ngeom", len(fmd.geoms), "!=", exp_geoms)
        bad += 1
    if bad != 0:
        raise Error(
            "family compose: the composed model does not carry every slot."
            " A count SHORT of the sum means an `<attach>` did not expand;"
            " see tests/physics3d/test_expand_identity.mojo."
        )
    print("  ok: every slot is in the composed model")

    # ── ⚠⚠ ATTACHING A MODEL MUST NOT CHANGE ITS NUMBERS ──────────────────
    #
    # Every check above this line counts RECORDS. They all passed while the
    # arm's joint limits in the composed scene were 57.3x too tight: the
    # composed host declared no `<compiler angle>`, MuJoCo's default for that
    # is DEGREE, and so a `angle="radian"` asset spliced into it had every
    # range reinterpreted. `robot_shoulder_pan` came out +-0.0335 rad — the
    # arm could move +-1.9 DEGREES — and nbody, njnt, nq, nv and ngeom were
    # all exactly right.
    #
    # ⚠⚠ AND THE MuJoCo HALF OF THIS GATE COULD NOT SEE IT EITHER, WHICH IS
    # THE PART WORTH REMEMBERING. `tools/tasks/check_family.py` samples a
    # reachable envelope from `m.jnt_range[:6]` — MuJoCo's OWN ranges, read
    # from the same file. It was therefore sweeping the CORRECT limits and
    # reporting a healthy envelope for a model our runtime could barely move.
    # An oracle only crosses implementations on the quantities it actually
    # compares, and this one was never compared.
    #
    # So: a joint's range in the composed scene must equal its range in the
    # asset it came from, parsed standalone. That is a statement about the
    # SPLICE and needs no oracle — and the standalone parse is independently
    # pinned to MuJoCo, whose numbers agree with the asset's text.
    var comp_j = 0
    var range_bad = 0
    var range_checked = 0

    # base first, then each slot in declaration order — the same order the
    # count arithmetic above already depends on.
    var srcs = List[String]()
    srcs.append(String(f.base))
    for i in range(len(f.slots)):
        srcs.append(String(f.slots[i].asset))
    for si in range(len(srcs)):
        var a2 = parse_model_runtime(srcs[si])
        for j in range(len(a2.joints)):
            if comp_j >= len(fmd.joints):
                break
            ref cj = fmd.joints[comp_j]
            ref aj = a2.joints[j]
            range_checked += 1
            var dlo = cj.range_min - aj.range_min
            var dhi = cj.range_max - aj.range_max
            if dlo < 0.0:
                dlo = -dlo
            if dhi < 0.0:
                dhi = -dhi
            if cj.is_limited != aj.is_limited or dlo > 1e-12 or dhi > 1e-12:
                range_bad += 1
                print("  FAIL: joint", comp_j, "range [", cj.range_min, ",",
                      cj.range_max, "] but", srcs[si], "declares [",
                      aj.range_min, ",", aj.range_max, "]")
            comp_j += 1

    # ⚠ ANTI-VACUITY. "0 mismatches" is also what a loop that compared nothing
    # reports, and this one indexes two lists whose lengths could drift apart.
    # The arm alone has six limited hinges, so a healthy run compares at least
    # that many.
    if range_checked < 6:
        raise Error(
            "family compose: compared only " + String(range_checked)
            + " joint ranges. The base's six hinges alone should appear, so"
            " this loop is not walking the joints it thinks it is."
        )
    if range_bad != 0:
        raise Error(
            "family compose: " + String(range_bad) + " of "
            + String(range_checked) + " joint ranges CHANGED when the asset"
            " was attached. The usual cause is the angle unit: MuJoCo reads a"
            " host with no <compiler angle> as DEGREE, so a radian asset"
            " spliced into it has every range divided by 57.3. The composed"
            " scene must restate the base's angle —"
            " `tasks/family.compose_family` does, and"
            " `tools/tasks/gen_family_scene.mojo` must have been re-run since."
        )
    print("  ok:", range_checked,
          "joint ranges survive the attach unchanged (angle unit intact)")

    # ⚠ AND THE ACTUATOR `ctrlrange`, WHICH IS THE ONE THE POLICY TOUCHES.
    # `Phyics3dEnvConfig.NORMALIZED_ACTIONS` maps a [-1, 1] action affinely
    # onto each actuator's ctrlrange, so a ctrlrange scaled by 1/57.3 would
    # confine the arm to 1/57.3 of its travel with the agent none the wiser —
    # a policy that trains, converges, and cannot reach anything. The joint
    # ranges above and these are separate arrays and were separately wrong.
    var comp_a = 0
    var ctrl_bad = 0
    var ctrl_checked = 0
    for si in range(len(srcs)):
        var a3 = parse_model_runtime(srcs[si])
        for k in range(len(a3.actuators)):
            if comp_a >= len(fmd.actuators):
                break
            ref ca = fmd.actuators[comp_a]
            ref aa = a3.actuators[k]
            ctrl_checked += 1
            var clo = ca.ctrl_min - aa.ctrl_min
            var chi = ca.ctrl_max - aa.ctrl_max
            if clo < 0.0:
                clo = -clo
            if chi < 0.0:
                chi = -chi
            if ca.is_ctrl_limited != aa.is_ctrl_limited \
                    or clo > 1e-12 or chi > 1e-12:
                ctrl_bad += 1
                print("  FAIL: actuator", comp_a, "ctrlrange [", ca.ctrl_min,
                      ",", ca.ctrl_max, "] but", srcs[si], "declares [",
                      aa.ctrl_min, ",", aa.ctrl_max, "]")
            comp_a += 1
    if ctrl_checked < 6:
        raise Error(
            "family compose: compared only " + String(ctrl_checked)
            + " actuator ctrlranges. The arm alone has six, so this loop is"
            " not walking the actuators it thinks it is."
        )
    if ctrl_bad != 0:
        raise Error(
            "family compose: " + String(ctrl_bad) + " of "
            + String(ctrl_checked) + " actuator ctrlranges CHANGED when the"
            " asset was attached. Same cause as the joint ranges above — see"
            " that error."
        )
    print("  ok:", ctrl_checked, "actuator ctrlranges survive the attach")

    # ⚠ THE BUDGET IS CONSTANT — that is what makes a family one
    # monomorphisation. Free slots contribute 6 dofs each and static ones
    # contribute NONE, which is the whole reason `static` exists.
    # ── the COMPILE UNIT agrees with the composed scene ──────────────────
    # ⚠ THIS IS THE LOOP CLOSED: .family -> scene.xml -> *_dims.mojo (from
    # MuJoCo) -> ModelDefFromXML. If someone edits the slot table and forgets
    # `gen-family-scenes` or `gen-dims`, the comptime dimensions freeze at the
    # old scene and `init_fields` would raise at env construction — far from
    # the edit that caused it. Here it is one line from the cause.
    print("  model def: nq", So101TabletopModel.NQ,
          " nv", So101TabletopModel.NV,
          " nbody", So101TabletopModel.NBODY)
    if So101TabletopModel.NBODY != len(fmd.bodies) + 1:
        raise Error(
            "family compose: So101TabletopModel.NBODY is "
            + String(So101TabletopModel.NBODY) + " but the composed scene has "
            + String(len(fmd.bodies)) + " bodies + world. Run"
            " `pixi run gen-family-scenes && pixi run gen-dims`."
        )
    if So101TabletopModel.NJOINT != len(fmd.joints):
        raise Error(
            "family compose: So101TabletopModel.NJOINT is stale — run"
            " `pixi run gen-family-scenes && pixi run gen-dims`."
        )
    # ⚠ nv IS THE BUDGET. Six per free slot, none per fixture.
    var expect_nv = 6 + f.n_free_slots() * 6
    if So101TabletopModel.NV != expect_nv:
        raise Error(
            "family compose: nv is " + String(So101TabletopModel.NV)
            + ", expected " + String(expect_nv) + " (6 for the arm + 6 per"
            " FREE slot; a static fixture adds none). Either the slot table"
            " changed without regenerating, or a slot's asset is not what its"
            " kind says it is."
        )
    print("  ok: the compile unit matches the composed scene")

    print("  free slots:", f.n_free_slots(),
          "-> the family's per-slot dof cost is", f.n_free_slots() * 6, "nv")
    print()
    print("⚠ The MuJoCo half of this gate is `tools/tasks/check_family.py`,")
    print("  run it on:", out)
    print("=== PASS (Mojo half) ===")
