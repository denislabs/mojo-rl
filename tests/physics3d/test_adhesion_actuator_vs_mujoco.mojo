"""`<adhesion body=...>` (`mjTRN_BODY`) — the transmission IS the contact set.

WHY THIS EXISTS
===============
`<adhesion>` was the last unmodelled actuator element with a model behind it.
flybody declares eight pads and reported `nact` **70** against MuJoCo's **78**
— eight controls consumed by nothing, and every index past the first one
misaligned for anyone driving the model.

⚠ ITS MOMENT IS NOT A `(qadr, dadr, coef)` TRIPLE AND CANNOT BE ONE.
`mj_transmission`'s body arm (engine_core_smooth.c:1623) sets `length = 0` and
builds the moment as **minus the average of the contact NORMAL Jacobians over
every contact involving the named body** — so it changes with the contact set
and belongs beside the site transmission in `pose_transmission.mojo`, which is
also the only place that re-detects contacts at the current pose. The force law
itself is ordinary: `mjs_setToAdhesion` sets gaintype FIXED, biastype NONE and
`ctrllimited = 1`, i.e. `force = gain * ctrl`.

⚠⚠ THE GAINS ARE IN A `<default>` CLASS AND NOWHERE ELSE. flybody writes

    <default class="adhesion_claw"><adhesion group="3" ctrlrange="0 1"
                                             gain="0.985"/></default>

and every `<adhesion ... class="adhesion_claw" body="..."/>` element carries no
`gain` and no `ctrlrange`. The defaults walker looked for four actuator tags;
`<adhesion>` is a fifth. Arm 2 below is the one that gates it, and it bites
because the two classes disagree — claws 0.985, labra 1.0 — so a build reading
the element alone answers 1.0 everywhere and fails on six of eight.

FIVE ARMS:

  1. `nact` — 78, MuJoCo's `nu`.
  2. the RECORD — bodies, gains and control ranges, all of which come from the
     class chain.
  3. ⚠⚠ the FORCE, and the SIGN. Adhesion PULLS: the moment carries a minus
     sign MuJoCo applies and this engine's own contact-row convention
     (`body_a - body_b`) could have cancelled. Built without it, flybody's
     free-joint z force came out `+4.534230e-01` against MuJoCo's
     `-4.534230e-01` — an exact negation. So the arm asserts the value, not
     just the magnitude.
  4. ⚠ the `<geom gap>` GAP, PINNED TO ITS EXACT SIZE. flybody's eight
     adhesion geoms are the only ones in this tree with `<geom gap>`, and
     3.10.0 DETECTS out to `margin + gap` while excluding from the solver at
     `dist >= margin`. This engine models no gap, so a contact in that band is
     not detected here at all — and at flybody's keyframe exactly one is:
     floor/claw_T1_left at dist 9.88e-04 against an includemargin of 5.0e-04.
     It is that pad's ONLY contact, so `adhere_claw_T1_left` reads zero here.
     The arm asserts the whole difference from MuJoCo is that ONE pad, worth
     exactly its gain — drop it from the command and the answer is exact
     again. When `<geom gap>` lands this arm goes RED and should be deleted,
     which is the point of writing it this way rather than as a tolerance.
  5. an unresolved `body=` must RAISE — `-1` and the worldbody share index 0,
     and either would be a silent zero-force pad.

Run: pixi run mojo run -I . tests/physics3d/test_adhesion_actuator_vs_mujoco.mojo
"""

from mojo_rl.physics3d.parser.runtime_load import (
    dims_from_flat, build_model_runtime, spec_fields_runtime,
    read_model_source,
)
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.expander import expand_mjcf
from mojo_rl.physics3d.parser.flat_model import ACT_KIND_ADHESION
from mojo_rl.physics3d.fields import Data, Model, DynDims, SpecFields
from mojo_rl.physics3d.fields.dynamics_scratch import DynamicsScratch
from mojo_rl.physics3d.dynamics.actuation import apply_actions_fields
from mojo_rl.physics3d.dynamics.pose_transmission import (
    apply_pose_transmission, model_has_adhesion,
)
from mojo_rl.physics3d.studio.stepping import STUDIO_DT
from mojo_rl.physics3d.gpu.constants import (
    MODEL_ACTUATOR_SIZE, KEY_IDX_NQPOS,
    ACT_IDX_KIND, ACT_IDX_KP, ACT_IDX_BODY_ID,
    ACT_IDX_CTRL_MIN, ACT_IDX_CTRL_MAX, ACT_IDX_CTRL_LIMITED,
)


comptime DT = STUDIO_DT
comptime FLYBODY = String(
    "references/mujoco_menagerie-main/flybody/scene.xml"
)
# MuJoCo's own, read off `mjModel`: actuators 70..77, bodies 5/6/27/35/43/51/
# 59/67, gains 1.0 1.0 then six 0.985, ctrlrange (0, 1) and ctrllimited 1 on
# all eight. ⚠ THE GAP-BAND PAD IS INDEX 2 OF THESE (`adhere_claw_T1_left`,
# body 27).
comptime ADH_FIRST = 70
comptime N_ADH = 8
comptime GAP_PAD = 2


def _mj_qfrc_all() -> List[Float64]:
    """MuJoCo `qfrc_actuator` at keyframe 0 with every adhesion pad at 1.0."""
    return [
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        -4.92499999999999982e+00, -8.84885398680677060e-02,
        -2.28117059521481813e-01, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        -1.89953796064622760e-02, -1.90073400173606313e-02,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +3.16636269437371304e-02, +2.00685794207135053e-02,
        -8.23406028098272458e-02, +5.62330612514919288e-03,
        +1.00965873992389168e-01, -5.43071286757114691e-02,
        -2.96050414352438847e-02, -1.75415935126915258e-02,
        -1.09557022555416342e-02, -6.70432003624760273e-03,
        -3.35216001812380137e-03, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +3.37085933108242156e-02, +1.34973295435040306e-02,
        -1.46091826912099126e-01, -2.91092034278940971e-03,
        +1.65859351543847844e-01, -9.82087467691516447e-02,
        -6.20822602324167944e-02, -3.56080741281473712e-02,
        -2.21434209644510993e-02, -1.45356555834518127e-02,
        -7.26782779172589245e-03, +3.36323508512381061e-02,
        +1.33135963301397493e-02, -1.46179162190761325e-01,
        -2.24975534407825762e-03, +1.65904901496267271e-01,
        -9.74017327087170898e-02, -6.15922503383927941e-02,
        -3.56048172563220022e-02, -2.20266445324073766e-02,
        -1.44572214477070842e-02, -7.22861072385352910e-03,
        -3.65654116915050259e-02, +6.05905932781312384e-02,
        -1.58540270884344914e-01, -3.09241110274291603e-02,
        +1.54163774080609856e-01, -1.00365591281800559e-01,
        -7.16627268204441392e-02, -4.33439557679154014e-02,
        -2.62590947561209369e-02, -1.71277378059872844e-02,
        -8.56386890299365607e-03, -3.61475909905508430e-02,
        +6.40248588095841537e-02, -1.56837842219516271e-01,
        -3.19059141559821305e-02, +1.52577865086217318e-01,
        -1.01150554577150414e-01, -7.13967356746257331e-02,
        -4.28970694929315735e-02, -2.58228273011142234e-02,
        -1.67725579955637812e-02, -8.38627899778187670e-03,
    ]

def _mj_qfrc_nogap() -> List[Float64]:
    """MuJoCo `qfrc_actuator` at keyframe 0 with every adhesion pad at 1.0, EXCEPT `adhere_claw_T1_left` (the gap-band one) at 0."""
    return [
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        -3.93999999999999995e+00, -6.04852443756998719e-04,
        -3.19809704747382129e-01, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        -1.89953796064622760e-02, -1.90073400173606313e-02,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +0.00000000000000000e+00, +0.00000000000000000e+00,
        +3.37085933108242156e-02, +1.34973295435040306e-02,
        -1.46091826912099126e-01, -2.91092034278940971e-03,
        +1.65859351543847844e-01, -9.82087467691516447e-02,
        -6.20822602324167944e-02, -3.56080741281473712e-02,
        -2.21434209644510993e-02, -1.45356555834518127e-02,
        -7.26782779172589245e-03, +3.36323508512381061e-02,
        +1.33135963301397493e-02, -1.46179162190761325e-01,
        -2.24975534407825762e-03, +1.65904901496267271e-01,
        -9.74017327087170898e-02, -6.15922503383927941e-02,
        -3.56048172563220022e-02, -2.20266445324073766e-02,
        -1.44572214477070842e-02, -7.22861072385352910e-03,
        -3.65654116915050259e-02, +6.05905932781312384e-02,
        -1.58540270884344914e-01, -3.09241110274291603e-02,
        +1.54163774080609856e-01, -1.00365591281800559e-01,
        -7.16627268204441392e-02, -4.33439557679154014e-02,
        -2.62590947561209369e-02, -1.71277378059872844e-02,
        -8.56386890299365607e-03, -3.61475909905508430e-02,
        +6.40248588095841537e-02, -1.56837842219516271e-01,
        -3.19059141559821305e-02, +1.52577865086217318e-01,
        -1.01150554577150414e-01, -7.13967356746257331e-02,
        -4.28970694929315735e-02, -2.58228273011142234e-02,
        -1.67725579955637812e-02, -8.38627899778187670e-03,
    ]


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def eq(mut self, got: Int, want: Int, msg: String):
        self.checks += 1
        if got == want:
            print("  ok:", msg, "=", got)
        else:
            self.fails += 1
            print("  FAIL:", msg, "got", got, "want", want)

    def close(mut self, got: Float64, want: Float64, tol: Float64, msg: String):
        self.checks += 1
        if abs(got - want) <= tol:
            print("  ok:", msg, "=", got)
        else:
            self.fails += 1
            print("  FAIL:", msg, "got", got, "want", want)

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


def _worst(got: List[Float64], want: List[Float64]) -> Float64:
    var w = 0.0
    for i in range(len(want)):
        var e = abs(got[i] - want[i])
        if e > w:
            w = e
    return w


def _qfrc(
    sf: SpecFields[DT, DynDims],
    mut m: Model[DT, DynDims],
    dims: DynDims,
    timestep: Float64,
    drop_gap: Bool,
) raises -> List[Float64]:
    """`d.qfrc` at flybody's keyframe with every adhesion pad commanded 1.0.

    ⚠ FROM THE KEYFRAME, NOT `qpos0`. The pads are only in contact in the
    keyframe pose; at `qpos0` the fly is off the ground and every adhesion
    moment is zero, which would make each arm below vacuously true.
    """
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var n_act = dims.get_nact()
    var d = Data[DT, DynDims, 1](dims)
    var nqp = Int(Float64(sf.key_meta.data[KEY_IDX_NQPOS]))
    for i in range(nq):
        d.qpos.data[i] = sf.qpos0.data[i]
    for i in range(min(nqp, nq)):
        d.qpos.data[i] = sf.key_qpos.data[i]
    for i in range(nv):
        d.qvel.data[i] = Scalar[DT](0)
    var ctrl = List[Float64]()
    for i in range(n_act):
        var on = i >= ADH_FIRST and i < ADH_FIRST + N_ADH
        if drop_gap and i == ADH_FIRST + GAP_PAD:
            on = False
        ctrl.append(1.0 if on else 0.0)
    var act = List[Scalar[DT]](length=n_act, fill=Scalar[DT](0))
    var sc = DynamicsScratch[DT, DynDims, 1](dims)
    apply_actions_fields[DT](sf, d, ctrl, act, timestep)
    apply_pose_transmission[DT](sf, m, d, sc, ctrl, act, timestep)
    var out = List[Float64]()
    for i in range(nv):
        out.append(Float64(d.qfrc.data[i]))
    return out^


def main() raises:
    var t = Tally()
    print("=== <adhesion body=> vs MuJoCo 3.10.0 (flybody) ===")

    var src = read_model_source(FLYBODY)
    var fmd = parse_xml_full(expand_mjcf(src[0], src[1]), src[1])
    var dims = dims_from_flat(fmd, max_contacts=128, nmesh_verts=65536)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var n_act = dims.get_nact()

    # ── 1. nact ──────────────────────────────────────────────────────────
    print("--- nact ---")
    t.eq(n_act, 78, "nact (was 70 — eight <adhesion> were skipped)")
    t.truth(model_has_adhesion(sf), "the model reports adhesion")

    # ── 2. the record, off the <default> class chain ─────────────────────
    print("--- gains and ranges, which live ONLY in a class ---")
    var want_body: List[Int] = [5, 6, 27, 35, 43, 51, 59, 67]
    var want_gain: List[Float64] = [
        1.0, 1.0, 0.985, 0.985, 0.985, 0.985, 0.985, 0.985,
    ]
    for k in range(N_ADH):
        var o = (ADH_FIRST + k) * MODEL_ACTUATOR_SIZE
        t.eq(Int(sf.actuators.data[o + ACT_IDX_KIND]), ACT_KIND_ADHESION,
             "actuator " + String(ADH_FIRST + k) + " kind")
        t.eq(Int(sf.actuators.data[o + ACT_IDX_BODY_ID]), want_body[k],
             "actuator " + String(ADH_FIRST + k) + " body")
        t.close(Float64(sf.actuators.data[o + ACT_IDX_KP]), want_gain[k],
                0.0, "actuator " + String(ADH_FIRST + k) + " gain")
        t.close(Float64(sf.actuators.data[o + ACT_IDX_CTRL_MIN]), 0.0,
                0.0, "actuator " + String(ADH_FIRST + k) + " ctrlrange lo")
        t.close(Float64(sf.actuators.data[o + ACT_IDX_CTRL_MAX]), 1.0,
                0.0, "actuator " + String(ADH_FIRST + k) + " ctrlrange hi")
        t.truth(sf.actuators.data[o + ACT_IDX_CTRL_LIMITED] != 0,
                "actuator " + String(ADH_FIRST + k)
                + " is ctrllimited (mjs_setToAdhesion sets it always)")
    # ⚠ NON-VACUITY: a build that read the ELEMENT and ignored the class would
    # answer MuJoCo's bare default of 1.0 for all eight and pass nothing here.
    t.truth(want_gain[0] != want_gain[2],
            "the two classes carry DIFFERENT gains (1.0 vs 0.985)")

    # ── 3/4. the force, the sign, and the gap ────────────────────────────
    print("--- every pad at 1.0 EXCEPT the gap-band one ---")
    var got_ng = _qfrc(sf, m, dims, fmd.timestep, True)
    var want_ng = _mj_qfrc_nogap()
    # ⚠⚠ TWO DOFS ARE EXCLUDED AND THEY ARE NOT AN ADHESION DEFECT. `labrum_left`
    # and `labrum_right` (dofs 12 and 13) are the only pair whose contact is
    # between two ELLIPSOIDS, and our narrowphase puts that contact somewhere
    # slightly different from MuJoCo's: dist 8.993e-06 against 5.106e-05, and a
    # normal 1.8 degrees away (ours (-0.03205, -0.99945, 0.00846), MuJoCo's
    # (-0.00147, -0.99999, 0.00232)). The adhesion moment READS that normal and
    # that point, so it inherits the difference — 1.684e-04 on dof 12, 8.0e-05
    # on dof 13. Every one of the other 106 dofs is exact.
    #
    # ⚠ SEPARATED RATHER THAN TOLERATED GLOBALLY: a single loose bound over all
    # 108 would hide an adhesion regression behind an ellipsoid one. Fixing the
    # ellipsoid pair should make these two arms mergeable.
    comptime LABRUM_A = 12
    comptime LABRUM_B = 13
    var e_ng = 0.0
    var e_lab = 0.0
    for i in range(nv):
        var e = abs(got_ng[i] - want_ng[i])
        if i == LABRUM_A or i == LABRUM_B:
            if e > e_lab:
                e_lab = e
        elif e > e_ng:
            e_ng = e
    print("    worst |d qfrc_actuator| over the other", nv - 2, "dofs =", e_ng)
    print("    the two labrum dofs (ellipsoid-ellipsoid contact)  =", e_lab)
    t.truth(e_ng < 1e-15,
            "seven pads reproduce MuJoCo's qfrc_actuator to float64 noise"
            " on every dof but the two the ellipsoid pair reaches")
    t.truth(e_lab < 2.0e-04,
            "and the ellipsoid pair's inherited error stays where it was"
            " measured (1.684e-04)")
    # ⚠ NON-VACUITY for the arm above: the labrum dofs must actually CARRY a
    # force, or "within 2e-04" would be true of two zeros.
    t.truth(abs(want_ng[LABRUM_A]) > 1e-3,
            "the labrum dofs carry a real adhesion force (not two zeros)")

    # ⚠⚠ THE SIGN. Adhesion PULLS the body onto its contacts, so the free
    # joint's z force must be NEGATIVE. Built without the minus sign this read
    # `+4.534230e-01` against MuJoCo's `-4.534230e-01`, and a magnitude-only
    # arm would have passed.
    t.truth(want_ng[2] < 0.0, "MuJoCo's free-joint z force is NEGATIVE")
    t.truth(got_ng[2] < 0.0, "and so is ours — adhesion pulls, it does not push")

    print("--- every pad at 1.0, including the one in the gap band ---")
    var got_all = _qfrc(sf, m, dims, fmd.timestep, False)
    var want_all = _mj_qfrc_all()
    # ⚠⚠ THIS ARM USED TO ASSERT A DIFFERENCE OF EXACTLY 0.985. `<geom gap>`
    # was unmodelled, so `adhere_claw_T1_left` — whose ONLY contact sits in
    # the band at dist 9.88e-04 against an includemargin of 5.0e-04 — had no
    # contact to pull on and read zero, and the whole discrepancy was that one
    # pad's gain. Gap is modelled now (`test_geom_gap_vs_mujoco`), so the arm
    # is the same shape as the one above: exact everywhere the ellipsoid pair
    # does not reach. It was written as an EQUALITY rather than a tolerance so
    # that closing gap would force this rewrite instead of passing quietly.
    var e_all = 0.0
    var e_all_lab = 0.0
    for i in range(nv):
        var e = abs(got_all[i] - want_all[i])
        if i == LABRUM_A or i == LABRUM_B:
            if e > e_all_lab:
                e_all_lab = e
        elif e > e_all:
            e_all = e
    print("    worst |d qfrc_actuator| over the other", nv - 2, "dofs =", e_all)
    print("    the two labrum dofs                                =", e_all_lab)
    t.truth(e_all < 1e-15,
            "ALL EIGHT pads reproduce MuJoCo — the gap-band one included")
    t.truth(e_all_lab < 2.0e-04,
            "and the ellipsoid pair's inherited error is unchanged by it")
    # ⚠ NON-VACUITY: the gap pad must actually be PULLING, or "exact" would be
    # true of a build that still ignored it. Its whole contribution lands on
    # the free joint's z dof, where its contact normal is the floor's.
    t.truth(abs(want_all[2] - want_ng[2]) > 0.9,
            "the gap-band pad is worth ~0.985 on the free joint's z dof —"
            " so this arm and the one above are looking at different physics")

    # ── 5. an unresolved `body=` must raise ──────────────────────────────
    print("--- an `body=` naming nothing ---")
    var bad = String(
        "<mujoco><worldbody><body name='b'><joint name='j' type='hinge'/>"
        "<geom type='sphere' size='0.1'/></body></worldbody>"
        "<actuator><adhesion name='a' body='typo' gain='1'"
        " ctrlrange='0 1'/></actuator></mujoco>"
    )
    var raised = False
    try:
        var _f = parse_xml_full(expand_mjcf(bad, String("")), String(""))
    except e:
        raised = True
        print("    raised:", e)
    t.truth(raised, "an unresolved `body=` raises (0 is the worldbody AND"
                    " not-found, so neither could be told from a pad in"
                    " the air)")

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error(
            "test_adhesion_actuator_vs_mujoco: " + String(t.fails) + " failed"
        )
