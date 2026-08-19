"""The pose survives a structural edit — carried by NAME, not by index.

WHY THIS EXISTS
===============
Deleting a body re-parses the model, and the parser assigns `qpos` addresses by
walking the tree it is given. Remove one joint near the root and EVERY address
after it shifts by one. A positional copy would then take the knee's angle and
write it into the ankle — a pose that is not wrong in any single number and is
wrong everywhere, with nothing to raise about it.

THREE ARMS, AND THE THIRD IS THE ONE THAT MAKES THE OTHER TWO MEAN ANYTHING:

  1. every surviving joint keeps its EXACT value (not "close" — the same bits,
     because a copy either happened or did not);

  2. the joints that went with the edit are gone, and the ones that could not
     be matched sit at the reference pose rather than at someone else's angle;

  3. ⚠⚠ THE ADDRESSES ACTUALLY MOVED. The deleted joint is chosen so that the
     surviving joints land at DIFFERENT `qpos` addresses than they had. A
     positional `memcpy` would pass arms 1 and 2 on any edit that removed the
     LAST joint, so the fixture has to remove one in the middle and the gate
     has to assert that it did.

⚠ AND THE NEGATIVE CONTROL: the same rebuild WITHOUT the remap must leave the
distinctive pose behind. Otherwise "the pose survived" could be true because
`qpos0` already happened to be what we set.

Run: pixi run mojo run -I . tests/physics3d/test_state_remap.mojo
"""

from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.flat_model import FlatModelDef
from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
)
from mojo_rl.physics3d.fields import Data, Model, DynDims
from mojo_rl.physics3d.studio.structure import delete_body
from mojo_rl.physics3d.studio.remap import (
    remap_state, joint_qpos_adr, RemapReport,
)


comptime DT = DType.float64
comptime MODEL = String("mojo_rl/envs/ant/assets/ant.xml")
comptime BASE = String("mojo_rl/envs/ant/assets")
# ⚠ NOT THE LAST LEG. `front_left_leg`'s two joints sit in the MIDDLE of ant's
# joint list, so removing them shifts the addresses of everything after —
# which is the only condition under which arm 1 says anything.
comptime VICTIM = String("front_left_leg")


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


def _read(p: String) raises -> String:
    var f = open(p, "r")
    var s = f.read()
    f.close()
    return s^


def main() raises:
    var t = Tally()
    print("=== the pose survives a structural edit, by name ===")

    var src = expand_mjcf(_read(MODEL), BASE)

    # ── the model before ──────────────────────────────────────────────────
    var fmd_a = parse_xml_full(src, BASE)
    var dims_a = dims_from_flat(fmd_a)
    var m_a = Model[DT, DynDims](dims_a)
    build_model_runtime[DT](fmd_a, dims_a, m_a)
    var d_a = Data[DT, DynDims, 1](dims_a)

    # ⚠ A DISTINCTIVE POSE, NOT A RANDOM ONE. Every slot gets a value derived
    # from its own address, so a copy landing one slot over is visible as a
    # wrong NUMBER rather than as a plausible pose.
    for i in range(dims_a.get_nq()):
        d_a.qpos.data[i] = Scalar[DT](0.1 + 0.01 * Float64(i))
    for i in range(dims_a.get_nv()):
        d_a.qvel.data[i] = Scalar[DT](-0.5 - 0.01 * Float64(i))

    # ── the edit ──────────────────────────────────────────────────────────
    var r = delete_body(src, VICTIM)
    t.truth(r.ok, String("deleted body '", VICTIM, "'"))

    var fmd_b = parse_xml_full(r.xml, BASE)
    var dims_b = dims_from_flat(fmd_b)
    var m_b = Model[DT, DynDims](dims_b)
    build_model_runtime[DT](fmd_b, dims_b, m_b)
    var sf_b = spec_fields_runtime[DT](fmd_b, dims_b)
    var d_b = Data[DT, DynDims, 1](dims_b)
    for i in range(dims_b.get_nq()):
        d_b.qpos.data[i] = sf_b.qpos0.data[i]
    for i in range(dims_b.get_nv()):
        d_b.qvel.data[i] = Scalar[DT](0)

    # ── arm 3, FIRST, because it licenses the other two ───────────────────
    # If no surviving joint moved, a positional copy would pass everything.
    print("--- the addresses actually moved ---")
    var adr_a = joint_qpos_adr(fmd_a)
    var adr_b = joint_qpos_adr(fmd_b)
    var moved = 0
    for bj in range(len(fmd_b.joints)):
        var nm = fmd_b.joint_names[bj]
        for aj in range(len(fmd_a.joints)):
            if fmd_a.joint_names[aj] == nm and adr_a[aj] != adr_b[bj]:
                moved += 1
    t.truth(moved >= 2,
            String(moved, " surviving joint(s) changed qpos address"))

    # ── the negative control, BEFORE the remap ────────────────────────────
    # ⚠ WITHOUT THIS, "the pose survived" could be true because `qpos0`
    # already held those numbers.
    var same_as_reference = True
    for bj in range(len(fmd_b.joints)):
        var nm = fmd_b.joint_names[bj]
        for aj in range(len(fmd_a.joints)):
            if fmd_a.joint_names[aj] != nm:
                continue
            for k in range(fmd_b.joints[bj].nq):
                var got = Float64(d_b.qpos.data[adr_b[bj] + k])
                var want = Float64(d_a.qpos.data[adr_a[aj] + k])
                if got != want:
                    same_as_reference = False
    t.truth(not same_as_reference,
            "before the remap the rebuilt model is NOT already in that pose")

    # ── the remap ─────────────────────────────────────────────────────────
    print("--- remap ---")
    var rep = remap_state(fmd_a, d_a, fmd_b, d_b)
    print("   ", rep.summary())

    # ── arm 1: every survivor kept its EXACT value ────────────────────────
    var checked = 0
    var wrong = 0
    for bj in range(len(fmd_b.joints)):
        var nm = fmd_b.joint_names[bj]
        for aj in range(len(fmd_a.joints)):
            if fmd_a.joint_names[aj] != nm:
                continue
            if fmd_a.joints[aj].jnt_type != fmd_b.joints[bj].jnt_type:
                continue
            for k in range(fmd_b.joints[bj].nq):
                checked += 1
                if (Float64(d_b.qpos.data[adr_b[bj] + k])
                        != Float64(d_a.qpos.data[adr_a[aj] + k])):
                    wrong += 1
                    print("       qpos mismatch on", nm, "slot", k)
    t.truth(wrong == 0 and checked > 0,
            String("every surviving qpos slot is EXACT (", checked,
                   " checked, ", wrong, " wrong)"))

    # ── arm 2: the bookkeeping is honest ──────────────────────────────────
    t.truth(rep.carried == len(fmd_b.joints),
            String("all ", len(fmd_b.joints), " joints in the new model were"
                   " carried (report says ", rep.carried, ")"))
    t.truth(rep.dropped == len(fmd_a.joints) - len(fmd_b.joints),
            String("the report accounts for the ",
                   len(fmd_a.joints) - len(fmd_b.joints),
                   " joint(s) the edit removed (says ", rep.dropped, ")"))

    # ── the type guard ────────────────────────────────────────────────────
    # ⚠ SAME NAME, DIFFERENT TYPE must NOT be carried: `nq` differs (7 for a
    # free joint, 1 for a hinge) and copying would walk over the joints after
    # it. Retyping ant's root free joint to a slide is the realistic version.
    print("--- a joint that kept its name and changed type is NOT carried ---")
    var retyped = src.replace(
        String('type="free"'), String('type="slide" axis="0 0 1"')
    )
    t.truth(retyped != src, "the fixture edit applied (control)")
    var fmd_c = parse_xml_full(retyped, BASE)
    var dims_c = dims_from_flat(fmd_c)
    var m_c = Model[DT, DynDims](dims_c)
    build_model_runtime[DT](fmd_c, dims_c, m_c)
    var sf_c = spec_fields_runtime[DT](fmd_c, dims_c)
    var d_c = Data[DT, DynDims, 1](dims_c)
    for i in range(dims_c.get_nq()):
        d_c.qpos.data[i] = sf_c.qpos0.data[i]
    var rep_c = remap_state(fmd_a, d_a, fmd_c, d_c)
    print("   ", rep_c.summary())
    t.truth(rep_c.reset >= 1,
            String("the retyped joint was RESET, not copied (reset=",
                   rep_c.reset, ")"))

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_state_remap: " + String(t.fails) + " failed")
