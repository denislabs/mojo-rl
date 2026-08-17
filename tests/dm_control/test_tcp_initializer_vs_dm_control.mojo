"""`ToolCenterPointInitializer` — the collision rejection sampler — vs dm_control.

This is the last piece of the dm_control Phase 7 reset path. Two things are
gated, and they fail in different ways:

  1. `has_relevant_collisions` — the ACCEPT/REJECT decision itself, compared
     against dm_control's own predicate over a sweep of arm poses. Both sides
     are driven from the SAME `qpos`, so a disagreement is the predicate or
     the contact set, never the sampling.
  2. `tool_center_point_initializer` — the loop around it: that a rejected
     sample restores the arm, that an accepted one does not, that exhaustion
     leaves the entry pose, and that the two failure counters actually
     separate IK failure from collision rejection.

⚠ THE PREDICATE CAN ONLY BE AS GOOD AS THE CONTACT SET, which is why the
contact gate came first (`test_jaco_contacts_vs_mujoco.mojo`). It established
that penetrating BODY-PAIRS agree with MuJoCo exactly — ours-only 0 AND
mujoco-only 0 over 60 poses. That is precisely the input this predicate reads:
it looks at nothing but the pairs and their `dist` sign. Had that gate been
skipped, a green result here would have meant "our predicate agrees with
itself".

⚠ BODY CLASSES ARE INJECTED, and they have to be. dm_control classifies a GEOM
by which entity model owns it (`geom.root is arm_model`); a baked MJCF is flat
and `parser/flat_model.mojo` keeps no body names, so nothing in our model can
recover that. `manipulation_ref.body_classes_reference` derives the labelling
with dm_control's own objects and this test feeds it to both sides — so what
is gated is the RULE, not a second guess at the labelling. See the note in
`envs/dm_control/manipulation_reset.mojo`.

Run with:
    pixi run mojo run -I . tests/dm_control/test_tcp_initializer_vs_dm_control.mojo
"""

from std.math import abs
from std.python import Python
from std.testing import assert_true, TestSuite
from std.collections import InlineArray

from mojo_rl.physics3d.fields import Model, Data, Dims, DimsLike
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.fields_build import build_model_fields_from_flat
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.collision.contact_detection import detect_contacts
from mojo_rl.envs.dm_control.manipulation_reset import (
    has_relevant_collisions,
    tool_center_point_initializer,
    BODY_ARM,
    BODY_HAND,
    BODY_FIXED,
)

comptime DTYPE = DType.float64
comptime NBODY: Int = 17
comptime NQ: Int = 9
comptime NV: Int = 9
comptime NJOINT: Int = 9
comptime NGEOM: Int = 21
comptime NSITE: Int = 12
comptime NEXCLUDE: Int = 4
comptime NMESH_VERTS: Int = 60000
comptime MAXC: Int = 256
comptime MD = Dims[
    nq=NQ,
    nv=NV,
    nbody=NBODY,
    njoint=NJOINT,
    ngeom=NGEOM,
    nsite=NSITE,
    max_contacts=MAXC,
    nequality=0,
    ntendon=0,
    nexclude=NEXCLUDE,
    nmesh_verts=NMESH_VERTS,
    npair=0,
]
comptime NDOF: Int = 6

comptime N_POSES: Int = 60


def _read(path: String) raises -> String:
    var builtins = Python.import_module("builtins")
    var f = builtins.open(path, "r")
    var txt = String(f.read())
    _ = f.close()
    return txt


struct _Fixture:
    """Jaco model + data + the reference handles every test here needs."""

    var d: Data[DTYPE, MD, 1]
    var mf: Model[DTYPE, MD]
    var body_class: InlineArray[Int, NBODY]

    def __init__(out self) raises:
        var sys = Python.import_module("sys")
        _ = sys.path.insert(0, "tests/dm_control")
        var warnings = Python.import_module("warnings")
        _ = warnings.filterwarnings("ignore")
        var tempfile = Python.import_module("tempfile")
        var os = Python.import_module("os")
        var refmod = Python.import_module("manipulation_ref")

        var tmp = String(tempfile.mkdtemp(prefix="jaco_tcp_"))
        var xml_path = String(refmod.bake("reach_site_features", tmp))
        var cwd = String(os.getcwd())
        _ = os.chdir(tmp)
        var fmd = parse_xml_full(_read(xml_path))
        self.mf = Model[DTYPE, MD]()
        build_model_fields_from_flat[DTYPE](fmd, self.mf)
        _ = os.chdir(cwd)
        self.d = Data[DTYPE, MD, 1]()

        var cls = refmod.body_classes_reference()
        self.body_class = InlineArray[Int, NBODY](fill=BODY_FIXED)
        for b in range(NBODY):
            self.body_class[b] = Int(py=cls[b])


def test_has_relevant_collisions_matches_dm_control() raises:
    print("=== has_relevant_collisions vs dm_control ===")
    var np = Python.import_module("numpy")
    var refmod = Python.import_module("manipulation_ref")
    var fx = _Fixture()

    # ── the classification, before anything depends on it ────────────────
    var n_arm = 0
    var n_hand = 0
    var n_fixed = 0
    for b in range(NBODY):
        if fx.body_class[b] == BODY_ARM:
            n_arm += 1
        elif fx.body_class[b] == BODY_HAND:
            n_hand += 1
        elif fx.body_class[b] == BODY_FIXED:
            n_fixed += 1
    print("  body classes: arm", n_arm, " hand", n_hand, " fixed", n_fixed)
    assert_true(
        n_arm > 0 and n_hand > 0 and n_fixed > 0,
        "the labelling collapsed to one class — every branch of the predicate"
        " below would then be testing the same thing",
    )
    # ⚠ Jaco's two ENTITY ATTACHMENT FRAMES (`jaco_arm/`, `jaco_arm/jaco_hand/`)
    # own no geoms, so they keep the array's `BODY_FIXED` default even though
    # they are structurally part of the robot. That is only harmless because a
    # geomless body cannot appear in a contact. Assert it rather than trust it:
    # if one ever gained a geom, every contact on it would be silently counted
    # as robot-versus-ground.
    var geomless = refmod.bodies_without_geoms()
    for i in range(Int(py=Python.evaluate("len")(geomless))):
        var b = Int(py=geomless[i])
        assert_true(
            fx.body_class[b] == BODY_FIXED,
            "a geomless body is not at the default class — the assumption"
            " that its label is never read no longer holds",
        )

    var lo = refmod.arm_joint_bounds()[0]
    var hi = refmod.arm_joint_bounds()[1]
    var rng = np.random.default_rng(4)

    var n_agree = 0
    var n_true = 0
    var n_false = 0
    var worst_pose = -1

    for t in range(N_POSES):
        var qpy = np.zeros(NQ)
        for i in range(NQ):
            var v: Float64
            if i < NDOF:
                v = Float64(
                    py=rng.uniform(
                        Python.evaluate("float")(lo[i]),
                        Python.evaluate("float")(hi[i]),
                    )
                )
            else:
                v = Float64(py=rng.uniform(0.15, 1.35))
            qpy[i] = v
            fx.d.qpos.data[i] = Scalar[DTYPE](v)

        forward_kinematics["cpu"](fx.d, fx.mf)
        detect_contacts["cpu"](fx.d, fx.mf)
        var ours = has_relevant_collisions[DTYPE](fx.d, fx.body_class)

        var rr = refmod.has_relevant_collisions_at(qpy)
        var theirs = Bool(py=rr[0])

        if ours == theirs:
            n_agree += 1
        elif worst_pose < 0:
            worst_pose = t
        if theirs:
            n_true += 1
        else:
            n_false += 1

    print("  poses:", N_POSES, " agree:", n_agree,
          "  dm_control said TRUE on", n_true, " FALSE on", n_false)
    if worst_pose >= 0:
        print("  first disagreement at pose", worst_pose)

    # ⚠ NON-VACUITY FIRST. A sweep that never collides would agree 60/60 while
    # exercising only the `return False` path.
    assert_true(
        n_true >= 5 and n_false >= 5,
        "the sweep is one-sided — both the reject and the accept branch have"
        " to occur or agreement means nothing",
    )
    assert_true(
        n_agree == N_POSES,
        "our collision predicate disagreed with dm_control's. The contact"
        " gate established that penetrating body-pairs match exactly, so a"
        " disagreement here is the CLASSIFICATION or the rule, not the"
        " contact set",
    )


def test_tcp_initializer_rejection_loop() raises:
    print("=== tool_center_point_initializer: the rejection loop ===")
    var np = Python.import_module("numpy")
    var mujoco = Python.import_module("mujoco")
    var refmod = Python.import_module("manipulation_ref")
    var fx = _Fixture()

    var tempfile = Python.import_module("tempfile")
    var os = Python.import_module("os")
    var tmp = String(tempfile.mkdtemp(prefix="jaco_tcp2_"))
    var xml_path = String(refmod.bake("reach_site_features", tmp))
    var cwd = String(os.getcwd())
    _ = os.chdir(tmp)
    var mm = mujoco.MjModel.from_xml_path(xml_path)
    _ = os.chdir(cwd)
    var dat = mujoco.MjData(mm)

    # Our TCP site index, located by position exactly as the IK gate does.
    for i in range(NQ):
        var qv = 0.11 * Float64(i + 1) - 0.4
        dat.qpos[i] = qv
        fx.d.qpos.data[i] = Scalar[DTYPE](qv)
    mujoco.mj_forward(mm, dat)
    forward_kinematics["cpu"](fx.d, fx.mf)
    var mj_tcp = Int(
        py=mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_SITE, refmod.TCP_SITE)
    )
    var our_tcp = -1
    var best_e = 1e30
    for s in range(NSITE):
        var e = 0.0
        for k in range(3):
            var dd = abs(
                Float64(fx.d.site_xpos.data[s * 3 + k])
                - Float64(py=dat.site_xpos[mj_tcp][k])
            )
            if dd > e:
                e = dd
        if e < best_e:
            best_e = e
            our_tcp = s
    assert_true(best_e < 1e-9, "could not locate our TCP site by position")

    var arm_names = refmod.arm_joint_names()
    var bounds = refmod.arm_joint_bounds()
    var adr_py = refmod.arm_qpos_adr()
    var dof_idx = InlineArray[Int, NDOF](fill=0)
    var qpos_adr = InlineArray[Int, NDOF](fill=0)
    var lower = InlineArray[Float64, NDOF](fill=0.0)
    var upper = InlineArray[Float64, NDOF](fill=0.0)
    for a in range(NDOF):
        var jid = Int(
            py=mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_JOINT, arm_names[a])
        )
        dof_idx[a] = Int(py=mm.jnt_dofadr[jid])
        qpos_adr[a] = Int(py=adr_py[a])
        lower[a] = Float64(py=bounds[0][a])
        upper[a] = Float64(py=bounds[1][a])

    var down = InlineArray[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
    down[0] = Scalar[DTYPE](0.70710678118)
    down[1] = Scalar[DTYPE](0.70710678118)

    comptime MAX_ATT: Int = 10
    comptime MAX_SAMP: Int = 10

    var draws = refmod.retry_pose_draws(MAX_SAMP * (MAX_ATT - 1), 7)
    var retry = List[Scalar[DTYPE]]()
    for k in range(MAX_SAMP * (MAX_ATT - 1) * NDOF):
        retry.append(Scalar[DTYPE](Float64(py=draws[k])))

    # ── entry pose, which a rejected or exhausted run must restore ───────
    var entry = InlineArray[Float64, NQ](fill=0.0)
    for i in range(NQ):
        var v = 0.11 * Float64(i + 1) - 0.4
        entry[i] = v
        fx.d.qpos.data[i] = Scalar[DTYPE](v)

    # ── A: targets DRIVEN INTO THE FLOOR, so the loop must exhaust ───────
    # The arena plane is at z = 0 and counts as a relevant collision, so a
    # target below it is either unreachable (IK failure) or reachable only in
    # collision. Either way no sample can be accepted, which is what makes
    # this a test of the exhaustion path rather than of luck.
    var bad_targets = List[Scalar[DTYPE]]()
    for s in range(MAX_SAMP):
        bad_targets.append(Scalar[DTYPE](0.02 * Float64(s) - 0.05))
        bad_targets.append(Scalar[DTYPE](0.0))
        bad_targets.append(Scalar[DTYPE](-0.25))
    var rbad = tool_center_point_initializer[DTYPE, NDOF](
        fx.d, fx.mf, our_tcp, bad_targets, down, dof_idx, qpos_adr,
        lower, upper, retry, fx.body_class, False, MAX_ATT, MAX_SAMP,
    )
    print("  [A] unreachable/colliding targets -> success", rbad.success,
          " samples", rbad.samples,
          " ik_failures", rbad.ik_failures,
          " collision_rejections", rbad.collision_rejections)
    assert_true(
        not rbad.success,
        "the initializer accepted a pose driven below the arena floor",
    )
    assert_true(
        rbad.ik_failures + rbad.collision_rejections == rbad.samples,
        "every consumed sample must be accounted for by exactly one failure"
        " reason — otherwise the counters cannot be used to tell a bad"
        " workspace from a bad IK budget",
    )
    var worst_restore = 0.0
    for i in range(NQ):
        var e = abs(Float64(fx.d.qpos.data[i]) - entry[i])
        if e > worst_restore:
            worst_restore = e
    print("  [A] worst |qpos - entry| after exhaustion:", worst_restore)
    assert_true(
        worst_restore < 1e-12,
        "an exhausted run left the arm at the last REJECTED pose. The"
        " reference restores `initial_qpos` on every failed sample, so a"
        " caller that ignores the return value must still see its entry pose",
    )

    # ── B: a reachable target, which must be accepted ────────────────────
    for i in range(NQ):
        fx.d.qpos.data[i] = Scalar[DTYPE](entry[i])
    var good_targets = List[Scalar[DTYPE]]()
    for s in range(MAX_SAMP):
        good_targets.append(Scalar[DTYPE](0.0))
        good_targets.append(Scalar[DTYPE](0.0))
        good_targets.append(Scalar[DTYPE](0.36 + 0.01 * Float64(s)))
    var rgood = tool_center_point_initializer[DTYPE, NDOF](
        fx.d, fx.mf, our_tcp, good_targets, down, dof_idx, qpos_adr,
        lower, upper, retry, fx.body_class, False, MAX_ATT, MAX_SAMP,
    )
    print("  [B] reachable target -> success", rgood.success,
          " samples", rgood.samples,
          " ik_failures", rgood.ik_failures,
          " collision_rejections", rgood.collision_rejections)
    assert_true(
        rgood.success,
        "the initializer could not place the TCP at a clearly reachable,"
        " collision-free target — the accept path never runs",
    )
    # On acceptance the reference does NOT restore, and the contacts left in
    # `d` are the accepted pose's. Check both, since returning early past the
    # restore is the one thing that separates accept from reject.
    var moved = 0.0
    for a in range(NDOF):
        var e = abs(Float64(fx.d.qpos.data[qpos_adr[a]]) - entry[qpos_adr[a]])
        if e > moved:
            moved = e
    print("  [B] arm moved from entry by:", moved)
    assert_true(
        moved > 1e-6,
        "an ACCEPTED sample left the arm at its entry pose — the accept path"
        " is restoring when it must not, and the solved pose is being thrown"
        " away",
    )
    assert_true(
        not has_relevant_collisions[DTYPE](
            fx.d, fx.body_class
        ),
        "the pose left in `d` on acceptance is in relevant collision — the"
        " contacts were not recomputed for the pose actually returned",
    )

    # ── C: ignore_collisions accepts what B's predicate would reject ─────
    # Same floor-driven targets as A. With the predicate switched off the only
    # remaining failure mode is IK, so if A rejected anything for COLLISION
    # this must now get further.
    for i in range(NQ):
        fx.d.qpos.data[i] = Scalar[DTYPE](entry[i])
    var rign = tool_center_point_initializer[DTYPE, NDOF](
        fx.d, fx.mf, our_tcp, bad_targets, down, dof_idx, qpos_adr,
        lower, upper, retry, fx.body_class, True, MAX_ATT, MAX_SAMP,
    )
    print("  [C] ignore_collisions -> success", rign.success,
          " samples", rign.samples,
          " ik_failures", rign.ik_failures,
          " collision_rejections", rign.collision_rejections)
    assert_true(
        rign.collision_rejections == 0,
        "`ignore_collisions` still rejected a sample for collision — the flag"
        " is not reaching the predicate",
    )
    # ⚠ THE DISCRIMINATING PART IS THAT [A] AND [C] SHARE `bad_targets`.
    # [A] rejected all ten; with the predicate switched off the very first one
    # is accepted. Asserting only "success" here would also pass if the flag
    # did nothing and the targets were simply fine.
    assert_true(
        rign.success and rbad.collision_rejections > 0,
        "`ignore_collisions` must accept the same targets the predicate"
        " rejected — otherwise this case proves nothing about the flag",
    )

    # ── D: reject, THEN accept — the transition neither A nor B covers ───
    # A exhausts without accepting and B accepts on its first sample, so
    # "carry on correctly after a rejection" is untested by both. If the
    # restore left the arm somewhere the next IK cannot solve from, only this
    # case would notice.
    comptime N_BAD: Int = 3
    for i in range(NQ):
        fx.d.qpos.data[i] = Scalar[DTYPE](entry[i])
    var mixed = List[Scalar[DTYPE]]()
    for s in range(N_BAD):
        mixed.append(Scalar[DTYPE](0.02 * Float64(s) - 0.05))
        mixed.append(Scalar[DTYPE](0.0))
        mixed.append(Scalar[DTYPE](-0.25))
    for s in range(MAX_SAMP - N_BAD):
        mixed.append(Scalar[DTYPE](0.0))
        mixed.append(Scalar[DTYPE](0.0))
        mixed.append(Scalar[DTYPE](0.36 + 0.01 * Float64(s)))
    var rmix = tool_center_point_initializer[DTYPE, NDOF](
        fx.d, fx.mf, our_tcp, mixed, down, dof_idx, qpos_adr,
        lower, upper, retry, fx.body_class, False, MAX_ATT, MAX_SAMP,
    )
    print("  [D] ", N_BAD, "bad then good -> success", rmix.success,
          " samples", rmix.samples,
          " ik_failures", rmix.ik_failures,
          " collision_rejections", rmix.collision_rejections)
    assert_true(
        rmix.success and rmix.samples == N_BAD + 1,
        "the loop did not accept on the first GOOD target after rejecting the"
        " bad ones — either it stopped early or the restore left a pose the"
        " next solve could not recover from",
    )
    assert_true(
        rmix.collision_rejections == N_BAD,
        "the rejected samples were not all counted as collisions",
    )
    assert_true(
        not has_relevant_collisions[DTYPE](
            fx.d, fx.body_class
        ),
        "the pose accepted after a rejection is in relevant collision",
    )

    # ── E: IK failure, so the OTHER counter is exercised at least once ───
    # Everything above fails by collision; `ik_failures` stays 0 throughout,
    # which means the branch that separates the two failure modes has never
    # run. A target metres outside the workspace cannot be solved at all.
    for i in range(NQ):
        fx.d.qpos.data[i] = Scalar[DTYPE](entry[i])
    var far = List[Scalar[DTYPE]]()
    for s in range(MAX_SAMP):
        far.append(Scalar[DTYPE](5.0 + Float64(s)))
        far.append(Scalar[DTYPE](5.0))
        far.append(Scalar[DTYPE](5.0))
    var rfar = tool_center_point_initializer[DTYPE, NDOF](
        fx.d, fx.mf, our_tcp, far, down, dof_idx, qpos_adr,
        lower, upper, retry, fx.body_class, False, MAX_ATT, MAX_SAMP,
    )
    print("  [E] out-of-workspace -> success", rfar.success,
          " samples", rfar.samples,
          " ik_failures", rfar.ik_failures,
          " collision_rejections", rfar.collision_rejections)
    assert_true(
        not rfar.success and rfar.ik_failures > 0,
        "a target metres outside the workspace was not recorded as an IK"
        " failure — the two counters cannot be told apart, which is the whole"
        " reason they are separate",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
