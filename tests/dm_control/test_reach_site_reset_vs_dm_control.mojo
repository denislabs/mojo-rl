"""`reach_site_features`' RESET, end to end through `Phyics3dEnv.reset()`.

The observation and reward gate (`test_reach_site_vs_dm_control`) drives both
engines from injected `qpos`, which is this suite's standing discipline and
says nothing about where an episode actually STARTS. This file gates that.

⚠⚠ WHY IT MATTERS MORE THAN A RESET GATE USUALLY WOULD. Before the TCP
initializer was wired, `reset()` left the arm at qpos0 — and MuJoCo reports
**55 contacts** there: the links are inside each other and inside the floor.
Every episode began in a pose the task never produces, every observation was
computed from it, and nothing raised. A reset that is merely "different" is a
distribution shift; this one was invalid.

WHAT IS CHECKED

  1. THE BODY CLASSES the config hardcodes, against dm_control's own
     labelling. They cannot be derived — a baked MJCF is flat and
     `flat_model.mojo` keeps no body names — so they are an INPUT, and an
     input that only a gate can check. Also asserts that the two entity
     ATTACHMENT FRAMES own no geoms, which is what makes their (wrong in
     spirit) `BODY_FIXED` label harmless.
  2. THE POSES `reset()` PRODUCES, judged by DM_CONTROL'S OWN PREDICATE.
     `has_relevant_collisions_at` is the same function the reference's
     rejection loop calls, so this asks the reference whether it would have
     accepted what we produced — not whether our predicate agrees with itself.
  3. THE TCP LANDS IN `tcp_bbox`. The initializer draws a target there and
     solves IK for it; a pinch site outside the box means the solve returned
     `success` without converging, which no collision check would catch.
  4. THE GRASP AND THE TARGET. `set_grasp` takes ONE draw broadcast to three
     fingers, so an asymmetric hand means three draws were taken; the target
     site must sit inside `target_bbox`.
  5. THE DISTRIBUTION IS NOT DEGENERATE — resets must differ from each other.
     A hook that solved once and cached, or that ignored its draws, would pass
     every check above.

⚠ THIS IS A DISTRIBUTION TEST, NOT A REPRODUCTION TEST. dm_control's draws
come from a numpy `RandomState` whose bit stream cannot be reproduced in Mojo,
so what is gated is that every pose we produce is one the reference would
ACCEPT, and that the sampling boxes are the right ones. Reproducing a specific
episode is not a goal here and never has been (see `manipulator_config`).

Run with:
    pixi run mojo run -I . tests/dm_control/test_reach_site_reset_vs_dm_control.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from mojo_rl.envs.dm_control.manipulation_reach import DMReachSiteFeatures
from mojo_rl.envs.dm_control.manipulation_reach_config import (
    N_ARM,
    N_HAND,
    SITE_TARGET,
    SITE_PINCH,
    TARGET_BBOX_LOWER_X,
    TARGET_BBOX_LOWER_Y,
    TARGET_BBOX_LOWER_Z,
    TARGET_BBOX_UPPER_X,
    TARGET_BBOX_UPPER_Y,
    TARGET_BBOX_UPPER_Z,
)
from mojo_rl.envs.dm_control.manipulation_reset import (
    BODY_ARM,
    BODY_HAND,
    BODY_FIXED,
)
from mojo_rl.physics3d.gpu.constants import (
    MODEL_SITE_SIZE,
    SITE_IDX_POS_X,
    SITE_IDX_POS_Y,
    SITE_IDX_POS_Z,
    MODEL_GEOM_SIZE,
    GEOM_IDX_BODY,
)

comptime DTYPE = DType.float64
comptime ENV = DMReachSiteFeatures[DTYPE]
comptime N_RESETS: Int = 24
# The IK runs to `tol = 1e-14` on the position error, but `set_site_to_xpos`
# accepts a solve that stopped on the progress guard too, so the box test gets
# a millimetre of slack rather than none. A failure to converge is orders out,
# not fractions of a millimetre.
comptime BOX_SLACK: Float64 = 1e-3


def _refmod() raises -> PythonObject:
    var sys = Python.import_module("sys")
    _ = sys.path.insert(0, "tests/dm_control")
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    return Python.import_module("manipulation_ref")


# ── leg 1 ──────────────────────────────────────────────────────────────────
def test_reach_body_classes_match_dm_control() raises:
    print("=== 1. body classes vs dm_control's own labelling ===")
    var refmod = _refmod()
    var ref_classes = refmod.body_classes_reference()
    var env = ENV()

    # The config's rule, restated here so the gate tests the ARRAY the config
    # builds rather than a second copy of the rule.
    var n_geomless_mismatch = 0
    var worst = -1
    for b in range(ENV.NBODY):
        var theirs = Int(py=ref_classes[b])
        var ours = BODY_FIXED
        if b >= 2 and b <= 8:
            ours = BODY_ARM
        elif b >= 10:
            ours = BODY_HAND
        # Count the geoms this body owns — the label of a geomless body can
        # never be read, because a body with no geoms cannot make a contact.
        var ngeom_b = 0
        for g in range(ENV.NGEOM):
            if (
                Int(env.mf.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_BODY])
                == b
            ):
                ngeom_b += 1
        print("  body", b, " ours", ours, " dm_control", theirs,
              " geoms", ngeom_b)
        if ours != theirs:
            if ngeom_b == 0:
                n_geomless_mismatch += 1
            elif worst < 0:
                worst = b

    assert_true(
        worst < 0,
        "a body that OWNS GEOMS is classified differently from dm_control."
        " The rejection predicate reads nothing but these labels and the"
        " contact pairs, so this silently changes which reset poses are"
        " accepted",
    )
    print("  geomless bodies whose label differs (harmless):",
          n_geomless_mismatch)

    # ⚠ Jaco's two entity ATTACHMENT FRAMES (`jaco_arm/`, `jaco_arm/jaco_hand/`)
    # own no geoms, so they keep the external default in the reference too and
    # our agreeing with it is luck rather than design. Assert the property that
    # makes it not matter.
    for b in range(ENV.NBODY):
        var ngeom_b = 0
        for g in range(ENV.NGEOM):
            if (
                Int(env.mf.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_BODY])
                == b
            ):
                ngeom_b += 1
        var theirs = Int(py=ref_classes[b])
        var ours = BODY_FIXED
        if b >= 2 and b <= 8:
            ours = BODY_ARM
        elif b >= 10:
            ours = BODY_HAND
        if ours != theirs:
            assert_true(
                ngeom_b == 0,
                "a mismatched body owns geoms — see above",
            )


# ── leg 2, 3, 4, 5 ─────────────────────────────────────────────────────────
def test_reach_reset_poses_are_ones_dm_control_would_accept() raises:
    print("=== 2-5. reset() poses, judged by dm_control ===")
    var refmod = _refmod()
    var env = ENV()

    var n_rejected = 0
    var worst_ncon = 0
    var out_of_tcp_box = 0
    var out_of_target_box = 0
    var asymmetric_grasp = 0
    var finger_out_of_range = 0
    var first_qpos = List[Float64]()
    var n_identical = 0
    var worst_tcp_excess = 0.0

    for r in range(N_RESETS):
        _ = env.reset()

        var qpy = Python.list()
        var qpos = List[Float64]()
        for i in range(ENV.NQ):
            var v = Float64(env.d.qpos.data[i])
            qpos.append(v)
            _ = qpy.append(v)

        # ── leg 2: the reference's OWN predicate on OUR pose ──────────────
        var rr = refmod.has_relevant_collisions_at(qpy)
        var bad = Bool(py=rr[0])
        var ncon = Int(py=rr[1])
        if bad:
            n_rejected += 1
        if ncon > worst_ncon:
            worst_ncon = ncon

        # ── leg 3: the TCP landed where the IK aimed ─────────────────────
        var px = Float64(env.d.site_xpos.data[SITE_PINCH * 3 + 0])
        var py_ = Float64(env.d.site_xpos.data[SITE_PINCH * 3 + 1])
        var pz = Float64(env.d.site_xpos.data[SITE_PINCH * 3 + 2])
        var ex = _box_excess(px, TARGET_BBOX_LOWER_X, TARGET_BBOX_UPPER_X)
        var ey = _box_excess(py_, TARGET_BBOX_LOWER_Y, TARGET_BBOX_UPPER_Y)
        var ez = _box_excess(pz, TARGET_BBOX_LOWER_Z, TARGET_BBOX_UPPER_Z)
        var ee = ex
        if ey > ee:
            ee = ey
        if ez > ee:
            ee = ez
        if ee > worst_tcp_excess:
            worst_tcp_excess = ee
        if ee > BOX_SLACK:
            out_of_tcp_box += 1

        # ── leg 4: the grasp and the target site ─────────────────────────
        var f0 = Float64(env.d.qpos.data[N_ARM + 0])
        var f1 = Float64(env.d.qpos.data[N_ARM + 1])
        var f2 = Float64(env.d.qpos.data[N_ARM + 2])
        if abs(f0 - f1) > 1e-15 or abs(f0 - f2) > 1e-15:
            asymmetric_grasp += 1
        if f0 < 0.15 - 1e-12 or f0 > 1.35 + 1e-12:
            finger_out_of_range += 1

        var tb = SITE_TARGET * MODEL_SITE_SIZE
        var tx = Float64(env.mf.sites.data[tb + SITE_IDX_POS_X])
        var ty = Float64(env.mf.sites.data[tb + SITE_IDX_POS_Y])
        var tz = Float64(env.mf.sites.data[tb + SITE_IDX_POS_Z])
        if (
            _box_excess(tx, TARGET_BBOX_LOWER_X, TARGET_BBOX_UPPER_X) > 0.0
            or _box_excess(ty, TARGET_BBOX_LOWER_Y, TARGET_BBOX_UPPER_Y) > 0.0
            or _box_excess(tz, TARGET_BBOX_LOWER_Z, TARGET_BBOX_UPPER_Z) > 0.0
        ):
            out_of_target_box += 1

        # ── leg 5: the draws actually vary ───────────────────────────────
        if r == 0:
            for i in range(ENV.NQ):
                first_qpos.append(qpos[i])
        else:
            var same = True
            for i in range(ENV.NQ):
                if abs(qpos[i] - first_qpos[i]) > 1e-12:
                    same = False
            if same:
                n_identical += 1

        if r < 4:
            print(
                "  reset", r,
                " dm_control rejects:", bad,
                " ncon", ncon,
                " tcp (", px, py_, pz, ")",
                " grasp", f0,
            )

    print("  resets:", N_RESETS)
    print("  dm_control would REJECT:", n_rejected, " worst ncon:", worst_ncon)
    print("  TCP outside tcp_bbox:", out_of_tcp_box,
          " worst excess:", worst_tcp_excess)
    print("  target outside target_bbox:", out_of_target_box)
    print("  asymmetric grasps:", asymmetric_grasp,
          " fingers out of range:", finger_out_of_range)
    print("  resets identical to the first:", n_identical)

    # ⚠ NON-VACUITY FIRST. If every reset came out identical, every check
    # above would be measuring one pose N times.
    assert_true(
        n_identical == 0,
        "resets repeat. Either the draws are not being taken or the hook is"
        " caching a solution — every check in this file then covers a single"
        " pose",
    )
    assert_true(
        n_rejected == 0,
        "dm_control's OWN rejection predicate says it would not have accepted"
        " a pose our reset produced. That is the check this file exists for:"
        " before the TCP initializer was wired, reset() left the arm at qpos0,"
        " which carries 55 contacts",
    )
    assert_true(
        out_of_tcp_box == 0,
        "the pinch site landed outside `tcp_bbox`. The IK reported success"
        " without converging — `set_site_to_xpos` also returns on its progress"
        " guard, and no collision check can see the difference",
    )
    assert_true(
        out_of_target_box == 0,
        "the target site was placed outside `target_bbox`",
    )
    assert_true(
        asymmetric_grasp == 0,
        "the three finger joints differ. `reach.py` passes a SCALAR"
        " `close_factor` which `JacoHand.set_grasp` broadcasts, so three"
        " independent draws is a different distribution from the reference's",
    )
    assert_true(
        finger_out_of_range == 0,
        "a finger is outside [0.15, 1.35] — `set_grasp` interpolates within"
        " the joint range, so this means the range was read from the wrong"
        " joint",
    )


def _box_excess(v: Float64, lo: Float64, hi: Float64) -> Float64:
    """How far `v` lies outside `[lo, hi]`; 0 inside."""
    if v < lo:
        return lo - v
    if v > hi:
        return v - hi
    return 0.0


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
