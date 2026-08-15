"""Phase 1a.1 — the differential gate that licenses the whole phase.

`docs/PHYSICS3D_RUNTIME_DIMS_ASSESSMENT.md` §10.9. Phase 1a moves actuator data
off the comptime `ComptimeActData` (`_acd`) and onto the runtime records. That
is ~20 arrays of MJCF semantics hand-ported into a second parser — a textbook
silent-bug generator — UNLESS it is diffed against the thing it replaces.

This is that diff, and it only works while `_acd` still exists. Same instrument
as `tests/physics3d/test_defaults_index_equivalence.mojo` (`585216bb`).

Ported field groups are ASSERTED; groups not yet ported are only printed, so
a new group's mismatches stay visible without disarming the ones already
landed. Verified by negative control (perturb one field -> the gate fails with
the field named), per `feedback_confirm_the_code_under_test_actually_runs`.

Round 1 covered the fields BOTH sides already carried, which was deliberately a
measurement and not an assumption: `full_parser._fill_actuators` resolves `gear`
from `defaults.motor_gear` — the ROOT `<default>` only — while `_acd` walks the
full nested class chain via `_class_attr_inherited`. dog has 24 distinct classes
over 38 actuators whose tags carry only `name`/`class`/`tendon`, so the two are
EXPECTED to disagree there. `test_dog_actuator_gain` proves `_acd` is the
correct side (nine distinct gains, `max |d(gainprm)| = 0.0` vs MuJoCo).

⇒ A failure here is not a regression. It is the two-parser split
(`feedback_physics3d_two_parser_paths`) finally being measured, and it sizes the
real work in 1a.1: the runtime parser needs class resolution on the actuator
path, not just extra fields.

RESULT, ROUND 1 (2026-08-14) — the prediction above was RIGHT ABOUT THE CAUSE
and WRONG ABOUT THE FIELD:

    cartpole    all fields agree (no <default> class to resolve)
    quadruped   ctrl_min 4/12, ctrl_max 8/12, ctrl_limited 12/12, kind 12/12
    dog         ctrl_limited 38/38, kind 38/38
    gear        0 mismatches EVERYWHERE  <- and it is VACUOUS, see below

`gear` agrees only because every dog and quadruped gear is 1.0 and cartpole's
is on the element — nothing in round 1 exercises class-resolved gear. The
guard below now says so instead of letting that read as a passing field.

Where they differ, the runtime side is the un-resolved one:

    quadruped[1]  runtime ctrl[-1.0, 1.0]   _acd ctrl[-1.0, 1.1]
    quadruped[2]  runtime ctrl[-1.0, 1.0]   _acd ctrl[-0.8, 0.8]

i.e. `_fill_actuators` falls back to the ROOT default `[-1,1]` wherever the
real value arrives through a `<default class=...>`, and reports
`ctrl_limited = 0` where `_acd` (the side the engine actually uses, and the
side `6705af47` fixed) reports 1. `kind` differs systematically because `_acd`
classifies a `<general>` by its compiled gain/bias behaviour — MOTOR for dog,
POSITION for quadruped — while `_fill_actuators` maps the TAG literally to
GENERAL. That one is an interface mismatch, not obviously a defect, but the
two encodings are not interchangeable.

⚠ `_acd` RE-MATERIALIZES ON EVERY READ — materialize ONCE into a local, then
copy element-by-element through a `comptime for` into runtime `List`s. A
comptime `InlineArray` also cannot be indexed by a runtime value. Both traps are
documented in `tests/dm_control/test_dog_actuator_gain.mojo`.

Run with:
    pixi run mojo run -I . tests/physics3d/test_actuator_record_equivalence.mojo
"""

from std.math import abs
from std.testing import assert_true, TestSuite

from mojo_rl.physics3d.parser import (
    parse_xml_full,
    parse_xml,
    ModelDefFromXML,
)
from mojo_rl.physics3d.parser.flat_model import FlatModelDef
from mojo_rl.physics3d.parser.fields_build import build_spec_fields
from mojo_rl.physics3d.fields import SpecFields
from mojo_rl.physics3d.gpu.constants import (
    TENDON_MAX_WRAPS,
    MODEL_ACTUATOR_SIZE,
    ACT_IDX_KIND,
    ACT_IDX_GEAR,
    ACT_IDX_CTRL_MIN,
    ACT_IDX_CTRL_MAX,
    ACT_IDX_CTRL_LIMITED,
    ACT_IDX_FORCE_MIN,
    ACT_IDX_FORCE_MAX,
    ACT_IDX_FORCE_LIMITED,
    ACT_IDX_KP,
    ACT_IDX_KV,
    ACT_IDX_DYN_TAU,
    ACT_IDX_ACT_ADR,
    ACT_IDX_TRN_N,
    ACT_IDX_DOF_ADR,
    ACT_IDX_TRN_QADR_0,
    ACT_IDX_TRN_DADR_0,
    ACT_IDX_TRN_COEF_0,
    MODEL_ACT_TENDON_SIZE,
    POSE_IDX_QPOS0_NQ,
    POSE_IDX_FREE_JOINT_QPOS_ADR,
    KEY_META_SIZE,
    KEY_IDX_TIME,
    KEY_IDX_NQPOS,
    KEY_IDX_NQVEL,
    KEY_IDX_NCTRL,
    ACTTEN_IDX_STIFFNESS,
    ACTTEN_IDX_SPRING_LO,
    ACTTEN_IDX_SPRING_HI,
    ACTTEN_IDX_TRN_N,
    ACTTEN_IDX_TRN_QADR_0,
    ACTTEN_IDX_TRN_DADR_0,
    ACTTEN_IDX_TRN_COEF_0,
    JLIM_SIZE,
    JLIM_IDX_LIMITED,
    JLIM_IDX_QPOS_ADR,
    JLIM_IDX_RANGE_MIN,
    JLIM_IDX_RANGE_MAX,
)

from mojo_rl.envs.dm_control.cartpole.cartpole_xml import DMCartpole1Model
from mojo_rl.envs.dm_control.quadruped.quadruped_xml import DMQuadrupedWalkModel
from mojo_rl.envs.dm_control.dog.dog_xml import DMDogStandWalkModel
from mojo_rl.envs.dm_control.manipulation_reach_def import (
    ReachSiteFeaturesModel,
)
from mojo_rl.envs.dm_control.fish.fish_xml import DMFishSwimModel
from mojo_rl.envs.dm_control.finger.finger_xml import DMFingerSpinModel
from mojo_rl.envs.ant.ant_xml import AntModel
from mojo_rl.envs.robots.so_arm100_xml import SoArm100Model


@fieldwise_init
struct _Report(Copyable, Movable):
    """Per-field mismatch counts, so the output SIZES the gap rather than just
    failing. A single boolean would say the split exists; these say how much of
    it each field owns."""

    var n_gear: Int
    var n_cmin: Int
    var n_cmax: Int
    var n_clim: Int
    var n_kind: Int
    var n_flim: Int
    var n_fmin: Int
    var n_fmax: Int
    var n_dyn: Int
    var n_aadr: Int
    var n_dofa: Int
    var n_trnn: Int
    var n_trn: Int
    var n_ten_act: Int
    var n_kp: Int
    var n_kv: Int
    var worst_gear: Float64
    var worst_gear_i: Int


def _compare[
    NACT_C: Int,
    NTEN_C: Int,
    NQ_C: Int,
    NV_C: Int,
    NKEY_C: Int,
    NJOINT_C: Int,
](
    name: String,
    nact: Int,
    xml: String,
    gears: List[Float64],
    cmin: List[Float64],
    cmax: List[Float64],
    clim: List[Int],
    kinds: List[Int],
    flim: List[Int],
    fmin: List[Float64],
    fmax: List[Float64],
    dyn: List[Float64],
    aadr: List[Int],
    na_acd: Int,
    dofa: List[Int],
    trnn: List[Int],
    tq: List[Int],
    td_: List[Int],
    tc: List[Float64],
    wraps: Int,
    kps: List[Float64],
    kvs: List[Float64],
    bad_acd: Int,
    ten_k: List[Float64],
    ten_lo: List[Float64],
    ten_hi: List[Float64],
    nten_acd: Int,
    ten_n: List[Int],
    ten_tq: List[Int],
    ten_td: List[Int],
    ten_tc: List[Float64],
) raises -> _Report:
    """Runtime `FlatModelDef.actuators[i]` against the comptime `_acd` arrays."""
    var fmd = parse_xml_full(xml)
    var n_rt = len(fmd.actuators)

    print("--- ", name, " ---")
    print("  nact: comptime `_acd` =", nact, "  runtime FlatModelDef =", n_rt)
    if n_rt != nact:
        print(
            "  ⚠ COUNT DIFFERS — every per-index comparison below compares"
            " different actuators; treat the field counts as meaningless."
        )

    var r = _Report(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, -1)
    var n = n_rt if n_rt < nact else nact
    for i in range(n):
        var a = fmd.actuators[i]
        var dg = abs(a.gear - gears[i])
        if dg > 0.0:
            r.n_gear += 1
            if dg > r.worst_gear:
                r.worst_gear = dg
                r.worst_gear_i = i
        if abs(a.ctrl_min - cmin[i]) > 0.0:
            r.n_cmin += 1
        if abs(a.ctrl_max - cmax[i]) > 0.0:
            r.n_cmax += 1
        if Int(a.is_ctrl_limited) != clim[i]:
            r.n_clim += 1
        if a.kind != kinds[i]:
            r.n_kind += 1
        if Int(a.force_limited) != flim[i]:
            r.n_flim += 1
        if abs(a.force_min - fmin[i]) > 0.0:
            r.n_fmin += 1
        if abs(a.force_max - fmax[i]) > 0.0:
            r.n_fmax += 1
        if abs(a.dyn_tau - dyn[i]) > 0.0:
            r.n_dyn += 1
        if a.act_adr != aadr[i]:
            r.n_aadr += 1
        if a.dof_adr != dofa[i]:
            r.n_dofa += 1
        if a.trn_n != trnn[i]:
            r.n_trnn += 1
        if abs(a.kp - kps[i]) > 0.0:
            r.n_kp += 1
        if abs(a.kv - kvs[i]) > 0.0:
            r.n_kv += 1
        if trnn[i] > 1:
            r.n_ten_act += 1  # multi-wrap == a TENDON transmission
        # Only the first trn_n entries are meaningful on either side.
        var nw = a.trn_n if a.trn_n < trnn[i] else trnn[i]
        for k in range(nw):
            var rb = i * TENDON_MAX_WRAPS + k
            if (
                fmd.motor_trn_qadr[rb] != tq[i * wraps + k]
                or fmd.motor_trn_dadr[rb] != td_[i * wraps + k]
                or abs(fmd.motor_trn_coef[rb] - tc[i * wraps + k]) > 0.0
            ):
                r.n_trn += 1

    # ⚠ NON-VACUITY. `gear` matched on every model in round 1 — but dog's and
    # quadruped's gears are ALL 1.0, so that agreement discriminates NOTHING
    # (feedback_degenerate_test_pose_gates_nothing). Say so, rather than
    # letting a vacuous 0 read as a passing field.
    if n > 1:
        var g_flat = True
        var fl_flat = True
        var fr_flat = True
        for i in range(1, n):
            if gears[i] != gears[0]:
                g_flat = False
            if flim[i] != flim[0]:
                fl_flat = False
            if fmin[i] != fmin[0] or fmax[i] != fmax[0]:
                fr_flat = False
        if g_flat:
            print("  ⚠ all", n, "gears are", gears[0],
                  "— `gear` is VACUOUS here")
        var dyn_flat = True
        for i in range(1, n):
            if dyn[i] != dyn[0]:
                dyn_flat = False
        if dyn_flat and dyn[0] == 0.0:
            print("  ⚠ no actuator has a dyntype — `dyn_tau`/`act_adr`/`na`"
                  " are VACUOUS here")
        if fl_flat and flim[0] == 0:
            print("  ⚠ no actuator is force-limited — `force_*` is VACUOUS"
                  " here")
        elif fr_flat:
            print("  ⚠ all force ranges identical — `force_*` is weak here")

    print("  mismatches over", n, "actuators:")
    print("    gear         =", r.n_gear, "  worst |d| =", r.worst_gear,
          " at i =", r.worst_gear_i)
    print("    ctrl_min     =", r.n_cmin)
    print("    ctrl_max     =", r.n_cmax)
    print("    ctrl_limited =", r.n_clim)
    print("    kind         =", r.n_kind)
    print("    force_limited=", r.n_flim)
    print("    force_min    =", r.n_fmin)
    print("    force_max    =", r.n_fmax)
    print("    dyn_tau      =", r.n_dyn)
    print("    act_adr      =", r.n_aadr)
    print("    dof_adr      =", r.n_dofa)
    print("    trn_n        =", r.n_trnn)
    print("    kp           =", r.n_kp)
    print("    kv           =", r.n_kv)
    print("    bad_actuator: runtime =", fmd.bad_actuator, " _acd =", bad_acd)
    print("    trn wraps    =", r.n_trn,
          "  (multi-wrap/tendon actuators here:", r.n_ten_act, ")")
    print("    na: runtime  =", fmd.na, "  _acd =", na_acd,
          "->", "OK" if fmd.na == na_acd else "DIFFERS")

    # ── ASSERT what is PORTED; only MEASURE what is not ──────────────────
    #
    # Until now `_compare` printed and never failed, so the groups already
    # landed (`224135af`, `3f8cb6df`) could regress silently — a measurement
    # probe, not a gate. These assertions close that. A field group graduates
    # from the printed block to here the moment its port lands, so the next
    # group's mismatches stay visible without disarming the previous ones.
    assert_true(
        n_rt == nact,
        String(name, ": actuator COUNT differs (", n_rt, " vs ", nact,
               ") — every per-index comparison is meaningless"),
    )
    assert_true(r.n_cmin == 0 and r.n_cmax == 0 and r.n_clim == 0,
        String(name, ": ctrlrange/ctrllimited disagree with `_acd` — ",
               r.n_cmin, "/", r.n_cmax, "/", r.n_clim))
    assert_true(r.n_gear == 0,
        String(name, ": gear disagrees with `_acd` in ", r.n_gear))
    assert_true(r.n_flim == 0 and r.n_fmin == 0 and r.n_fmax == 0,
        String(name, ": forcerange/forcelimited disagree with `_acd` — ",
               r.n_flim, "/", r.n_fmin, "/", r.n_fmax))
    assert_true(r.n_kind == 0,
        String(name, ": `kind` disagrees with `_acd` in ", r.n_kind,
               " — the force LAW, not a label"))
    assert_true(r.n_kp == 0 and r.n_kv == 0,
        String(name, ": kp/kv disagree with `_acd` — ", r.n_kp, "/", r.n_kv))
    assert_true(fmd.bad_actuator == bad_acd,
        String(name, ": bad_actuator runtime ", fmd.bad_actuator,
               " vs `_acd` ", bad_acd))
    # ── tendon springs (a different list, so compared separately) ────────
    var n_ten = len(fmd.tendons)
    print("    tendons: runtime =", n_ten, "  _acd ntendon =", nten_acd)
    var n_tk = 0
    var n_tsp = 0
    var spring_seen = False
    var nt = n_ten if n_ten < nten_acd else nten_acd
    for t in range(nt):
        var td = fmd.tendons[t]
        if abs(td.stiffness - ten_k[t]) > 0.0:
            n_tk += 1
        if ten_k[t] != 0.0:
            spring_seen = True
        if (
            abs(td.spring_lo - ten_lo[t]) > 0.0
            or abs(td.spring_hi - ten_hi[t]) > 0.0
        ):
            n_tsp += 1
    print("    tendon stiffness =", n_tk, "  springlength =", n_tsp,
          "  (a tendon with stiffness != 0 here:", spring_seen, ")")
    if nt > 0 and not spring_seen:
        print("  ⚠ no tendon has stiffness — the tendon-spring rows are"
              " VACUOUS here")
    assert_true(n_ten == nten_acd,
        String(name, ": tendon COUNT runtime ", n_ten, " vs `_acd` ",
               nten_acd))
    assert_true(n_tk == 0 and n_tsp == 0,
        String(name, ": tendon springs disagree with `_acd` — stiffness ",
               n_tk, " springlength ", n_tsp))

    assert_true(r.n_dyn == 0 and r.n_aadr == 0,
        String(name, ": dyn_tau/act_adr disagree with `_acd` — ",
               r.n_dyn, "/", r.n_aadr))
    assert_true(r.n_dofa == 0 and r.n_trnn == 0 and r.n_trn == 0,
        String(name, ": transmission disagrees with `_acd` — dof_adr ",
               r.n_dofa, " trn_n ", r.n_trnn, " wraps ", r.n_trn))
    assert_true(fmd.na == na_acd,
        String(name, ": na runtime ", fmd.na, " vs `_acd` ", na_acd,
               " — phase 1a.4 asserts a comptime NA against this"))
    # Every ported group is asserted. Still unported: qpos0, keyframes,
    # tendon stiffness/springs/limits.

    # ⚠ A COUNT SAYS "THEY DIFFER", NOT "ONE IS WRONG". An encoding mismatch
    # and a parse defect both read as N/N. Dump the first few so the DIRECTION
    # is visible.
    var nd = 3 if n > 3 else n
    for i in range(nd):
        var a = fmd.actuators[i]
        print("    [", i, "] runtime: gear", a.gear, " ctrl[", a.ctrl_min,
              ",", a.ctrl_max, "] lim", Int(a.is_ctrl_limited), " kind", a.kind,
              " force[", a.force_min, ",", a.force_max, "] flim",
              Int(a.force_limited))
        print("        _acd  : gear", gears[i], " ctrl[", cmin[i],
              ",", cmax[i], "] lim", clim[i], " kind", kinds[i],
              " force[", fmin[i], ",", fmax[i], "] flim", flim[i])

    # ═══ PHASE 1a.2 — the packed TENSORS, against the same `_acd` values ══
    #
    # ⚠ THIS IS A SECOND TRANSCRIPTION AND NEEDS ITS OWN GATE. Everything
    # above proves `FlatModelDef` agrees with `_acd`; it says nothing about
    # whether `build_spec_fields` copies those records into the right SLOTS.
    # An off-by-one in the column layout, a field written to the wrong index,
    # a wrap stride confusion — all invisible to the record diff and all
    # silently wrong in the engine once 1a.3 repoints the readers.
    #
    # ⚠⚠ THE WRAP STRIDES DIFFER AND THAT IS THE EASIEST THING TO GET WRONG
    # HERE. `_acd` strides by `M._WRAPS`, which COLLAPSES TO 1 on a model with
    # no tendons; the tensor strides by `TENDON_MAX_WRAPS` = 16 always. On
    # cartpole those are 1 and 16. Comparing them without converting reads
    # actuator i+1's data as actuator i's wrap 1.
    _fields_diff[NACT_C, NTEN_C, NQ_C, NV_C, NKEY_C, NJOINT_C](
        name, fmd, n_rt, gears, cmin, cmax, clim, kinds, flim, fmin, fmax,
        dyn, aadr, dofa, trnn, tq, td_, tc, wraps, kps, kvs,
        ten_k, ten_lo, ten_hi, nten_acd, ten_n, ten_tq, ten_td, ten_tc,
    )
    return r^


def _fields_diff[
    NACT_C: Int,
    NTEN_C: Int,
    NQ_C: Int,
    NV_C: Int,
    NKEY_C: Int,
    NJOINT_C: Int,
](
    name: String,
    fmd: FlatModelDef,
    n_rt: Int,
    gears: List[Float64],
    cmin: List[Float64],
    cmax: List[Float64],
    clim: List[Int],
    kinds: List[Int],
    flim: List[Int],
    fmin: List[Float64],
    fmax: List[Float64],
    dyn: List[Float64],
    aadr: List[Int],
    dofa: List[Int],
    trnn: List[Int],
    tq: List[Int],
    td_: List[Int],
    tc: List[Float64],
    wraps: Int,
    kps: List[Float64],
    kvs: List[Float64],
    ten_k: List[Float64],
    ten_lo: List[Float64],
    ten_hi: List[Float64],
    nten_acd: Int,
    ten_n: List[Int],
    ten_tq: List[Int],
    ten_td: List[Int],
    ten_tc: List[Float64],
) raises:
    """`SpecFields` tensor slots against the `_acd` values (phase 1a.2)."""
    if n_rt != NACT_C:
        # The count assert above already failed and named it; building the
        # tensors would just raise the parser/dimension error on top and hide
        # the real message.
        print("  (skipping tensor diff — actuator count differs)")
        return

    var sf = SpecFields[
        DType.float64, NACT_C, NTEN_C, NQ_C, NV_C, NKEY_C, NJOINT_C
    ]()
    build_spec_fields[
        DType.float64, NACT_C, NTEN_C, NQ_C, NV_C, NKEY_C, NJOINT_C
    ](fmd, sf)

    var n_scalar = 0
    var n_wrap = 0
    var first = String("")
    for i in range(NACT_C):
        var o = i * MODEL_ACTUATOR_SIZE

        @parameter
        def chk(slot: Int, want: Float64, field: String):
            if sf.actuators.data[o + slot] != Scalar[DType.float64](want):
                n_scalar += 1
                if first.byte_length() == 0:
                    first = String(
                        field, "[", i, "] tensor ",
                        Float64(sf.actuators.data[o + slot]),
                        " vs `_acd` ", want,
                    )

        chk(ACT_IDX_KIND, Float64(kinds[i]), "kind")
        chk(ACT_IDX_GEAR, gears[i], "gear")
        chk(ACT_IDX_CTRL_MIN, cmin[i], "ctrl_min")
        chk(ACT_IDX_CTRL_MAX, cmax[i], "ctrl_max")
        chk(ACT_IDX_CTRL_LIMITED, Float64(clim[i]), "ctrl_limited")
        chk(ACT_IDX_FORCE_MIN, fmin[i], "force_min")
        chk(ACT_IDX_FORCE_MAX, fmax[i], "force_max")
        chk(ACT_IDX_FORCE_LIMITED, Float64(flim[i]), "force_limited")
        chk(ACT_IDX_KP, kps[i], "kp")
        chk(ACT_IDX_KV, kvs[i], "kv")
        chk(ACT_IDX_DYN_TAU, dyn[i], "dyn_tau")
        chk(ACT_IDX_ACT_ADR, Float64(aadr[i]), "act_adr")
        chk(ACT_IDX_TRN_N, Float64(trnn[i]), "trn_n")
        chk(ACT_IDX_DOF_ADR, Float64(dofa[i]), "dof_adr")
        # Wraps: `_acd` strides by `wraps`, the tensor by TENDON_MAX_WRAPS.
        for k in range(trnn[i]):
            var a_i = i * wraps + k
            if (
                sf.actuators.data[o + ACT_IDX_TRN_QADR_0 + k]
                != Scalar[DType.float64](tq[a_i])
                or sf.actuators.data[o + ACT_IDX_TRN_DADR_0 + k]
                != Scalar[DType.float64](td_[a_i])
                or sf.actuators.data[o + ACT_IDX_TRN_COEF_0 + k]
                != Scalar[DType.float64](tc[a_i])
            ):
                n_wrap += 1

    var n_ten_s = 0
    var n_ten_w = 0
    var nt = len(fmd.tendons) if len(fmd.tendons) < nten_acd else nten_acd
    for t in range(nt):
        var o = t * MODEL_ACT_TENDON_SIZE
        if (
            sf.act_tendons.data[o + ACTTEN_IDX_STIFFNESS]
            != Scalar[DType.float64](ten_k[t])
            or sf.act_tendons.data[o + ACTTEN_IDX_SPRING_LO]
            != Scalar[DType.float64](ten_lo[t])
            or sf.act_tendons.data[o + ACTTEN_IDX_SPRING_HI]
            != Scalar[DType.float64](ten_hi[t])
            or sf.act_tendons.data[o + ACTTEN_IDX_TRN_N]
            != Scalar[DType.float64](ten_n[t])
        ):
            n_ten_s += 1
        for k in range(ten_n[t]):
            var a_i = t * wraps + k
            if (
                sf.act_tendons.data[o + ACTTEN_IDX_TRN_QADR_0 + k]
                != Scalar[DType.float64](ten_tq[a_i])
                or sf.act_tendons.data[o + ACTTEN_IDX_TRN_DADR_0 + k]
                != Scalar[DType.float64](ten_td[a_i])
                or sf.act_tendons.data[o + ACTTEN_IDX_TRN_COEF_0 + k]
                != Scalar[DType.float64](ten_tc[a_i])
            ):
                n_ten_w += 1

    # ⚠ NON-VACUITY, the trap that fired twice in 1a.1. A tensor diff over a
    # model whose actuators have no transmission wraps, or whose tendons have
    # no springs, reports 0 while testing nothing. Say which.
    var n_wrapped = 0
    for i in range(NACT_C):
        if trnn[i] > 0:
            n_wrapped += 1
    print("    TENSORS: scalars =", n_scalar, " wraps =", n_wrap,
          " tendon rows =", n_ten_s, " tendon wraps =", n_ten_w)
    print("             (actuators with a transmission:", n_wrapped, "of",
          NACT_C, "; tendon rows compared:", nt, ")")
    if n_wrapped == 0:
        print("  ⚠ no actuator has a transmission — the tensor WRAP slots are"
              " VACUOUS here")
    if nt == 0:
        print("  ⚠ no tendons — the act_tendons tensor is VACUOUS here")

    assert_true(n_scalar == 0,
        String(name, ": SpecFields scalar slots disagree with `_acd` in ",
               n_scalar, " — first: ", first))
    assert_true(n_wrap == 0,
        String(name, ": SpecFields transmission wraps disagree with `_acd`"
               " in ", n_wrap))
    assert_true(n_ten_s == 0 and n_ten_w == 0,
        String(name, ": SpecFields tendon rows disagree with `_acd` — ",
               n_ten_s, " scalars / ", n_ten_w, " wraps"))


# ═══ The `ctrlrange="0 0"` fixture — added 2026-08-15 AFTER it cost a bug ═══
#
# ⚠⚠ THE GATE WAS BLIND HERE AND A REAL DEFECT GOT THROUGH. MuJoCo reads
# `ctrlrange="0 0"` as the UNDEFINED marker, so such an actuator is UNLIMITED;
# `_fill_actuators` set `is_ctrl_limited = True` for any present `ctrlrange`,
# clamped the command to [0, 0], and delivered ZERO FORCE where MuJoCo delivers
# the full command. Every model in this gate (cartpole, quadruped, dog, jaco,
# fish) either omits `ctrlrange` or gives a real one — so the mismatch existed
# on every run and the gate reported 0. `test_ctrllimited_vs_mujoco` caught it
# instead, at `a5`: dof 5, ours 0.0, MuJoCo 5.0.
#
# Mirrors that file's `ctrllimited_matrix` fixture, trimmed to the four cases
# that take DIFFERENT branches of the resolver. Duplicated rather than imported
# because a test module is not an importable package here; if one changes, the
# other must.
comptime CTRLLIM_XML = String(
    """<mujoco model="ctrllimited_matrix_gate">
  <option timestep="0.001" gravity="0 0 0"/>
  <worldbody>
    <body name="b0" pos="0 0 0"><joint name="j0" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass="1"/></body>
    <body name="b1" pos="0 1 0"><joint name="j1" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass="1"/></body>
    <body name="b2" pos="0 2 0"><joint name="j2" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass="1"/></body>
    <body name="b3" pos="0 3 0"><joint name="j3" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass="1"/></body>
  </worldbody>
  <actuator>
    <motor name="a0" joint="j0"/>
    <motor name="a1" joint="j1" ctrlrange="-2 3"/>
    <motor name="a2" joint="j2" ctrlrange="-1 1" ctrllimited="false"/>
    <motor name="a3" joint="j3" ctrlrange="0 0"/>
  </actuator>
</mujoco>"""
)

comptime _clpm = parse_xml(CTRLLIM_XML)
comptime CtrlLimModel = ModelDefFromXML[
    xml=CTRLLIM_XML,
    nbody=_clpm.NBODY, njoint=_clpm.NJOINT, nq=_clpm.NQ, nv=_clpm.NV,
    ngeom=_clpm.NGEOM, nact=_clpm.NACT, ntex=_clpm.NTEX, nmat=_clpm.NMAT,
    nlight=_clpm.NLIGHT, ncam=_clpm.NCAM, nsite=_clpm.NSITE, neq=_clpm.NEQ,
    nexclude=_clpm.NEXCLUDE, npair=_clpm.NPAIR, max_tendon=_clpm.NTENDON,
    max_condim=_clpm.MAX_CONDIM, max_contacts=8,
    obs_dim_override=1, obs_qpos_skip=0,
    timestep=_clpm.TIMESTEP, noslip_iter=_clpm.NOSLIP_ITER,
]


def test_cartpole() raises:
    comptime M = DMCartpole1Model
    comptime acd = materialize[M._acd]()
    comptime NACT = M.nact
    var gears = List[Float64](capacity=NACT)
    var cmin = List[Float64](capacity=NACT)
    var cmax = List[Float64](capacity=NACT)
    var clim = List[Int](capacity=NACT)
    var kinds = List[Int](capacity=NACT)
    var flim = List[Int](capacity=NACT)
    var fmin = List[Float64](capacity=NACT)
    var fmax = List[Float64](capacity=NACT)
    var dyn = List[Float64](capacity=NACT)
    var aadr = List[Int](capacity=NACT)
    var dofa = List[Int](capacity=NACT)
    var trnn = List[Int](capacity=NACT)
    var tq = List[Int]()
    var td_ = List[Int]()
    var tc = List[Float64]()
    var kps = List[Float64](capacity=NACT)
    var kvs = List[Float64](capacity=NACT)
    var ten_k = List[Float64]()
    var ten_lo = List[Float64]()
    var ten_hi = List[Float64]()
    var ten_n = List[Int]()
    var ten_tq = List[Int]()
    var ten_td = List[Int]()
    var ten_tc = List[Float64]()
    comptime for ti in range(M._NTEN):
        ten_k.append(materialize[acd.tendon_stiffness[ti]]())
        ten_lo.append(materialize[acd.tendon_spring_lo[ti]]())
        ten_hi.append(materialize[acd.tendon_spring_hi[ti]]())
        ten_n.append(materialize[acd.tendon_trn_n[ti]]())
        comptime for wk in range(M._WRAPS):
            ten_tq.append(materialize[acd.tendon_trn_qadr[ti * M._WRAPS + wk]]())
            ten_td.append(materialize[acd.tendon_trn_dadr[ti * M._WRAPS + wk]]())
            ten_tc.append(materialize[acd.tendon_trn_coef[ti * M._WRAPS + wk]]())
    comptime for ai in range(NACT):
        gears.append(materialize[acd.motor_gears[ai]]())
        cmin.append(materialize[acd.motor_ctrl_min[ai]]())
        cmax.append(materialize[acd.motor_ctrl_max[ai]]())
        clim.append(materialize[acd.motor_ctrl_limited[ai]]())
        kinds.append(materialize[acd.motor_kind[ai]]())
        flim.append(materialize[acd.motor_force_limited[ai]]())
        fmin.append(materialize[acd.motor_force_min[ai]]())
        fmax.append(materialize[acd.motor_force_max[ai]]())
        dyn.append(materialize[acd.motor_dyn_tau[ai]]())
        aadr.append(materialize[acd.motor_act_adr[ai]]())
        dofa.append(materialize[acd.motor_dof_adr[ai]]())
        trnn.append(materialize[acd.motor_trn_n[ai]]())
        kps.append(materialize[acd.motor_kp[ai]]())
        kvs.append(materialize[acd.motor_kv[ai]]())
        comptime for wk in range(M._WRAPS):
            tq.append(materialize[acd.motor_trn_qadr[ai * M._WRAPS + wk]]())
            td_.append(materialize[acd.motor_trn_dadr[ai * M._WRAPS + wk]]())
            tc.append(materialize[acd.motor_trn_coef[ai * M._WRAPS + wk]]())
    _ = _compare[NACT, M._NTEN, M.NQ, M.NV, M.NKEY, M.NJOINT](
        "cartpole (1 actuator, no <default> class)",
        NACT, String(M.xml), gears, cmin, cmax, clim, kinds, flim, fmin, fmax,
        dyn, aadr, materialize[acd.na](),
        dofa, trnn, tq, td_, tc, M._WRAPS,
        kps, kvs, materialize[acd.bad_actuator](),
        ten_k, ten_lo, ten_hi, materialize[acd.ntendon](),
        ten_n, ten_tq, ten_td, ten_tc,
    )


def test_ctrllimited_matrix() raises:
    """The `ctrlrange="0 0"` case the gate was blind to — see the fixture.

    ⚠ VACUITY IS THE POINT OF THIS ONE. It is here for `ctrl_limited` alone:
    four actuators, three of them unlimited for three DIFFERENT reasons (no
    range, explicit `ctrllimited="false"`, and the degenerate `"0 0"`). The
    print below says how many differ so the row cannot go flat unnoticed.
    """
    comptime M = CtrlLimModel
    comptime acd = materialize[M._acd]()
    comptime NACT = M.nact
    var gears = List[Float64](capacity=NACT)
    var cmin = List[Float64](capacity=NACT)
    var cmax = List[Float64](capacity=NACT)
    var clim = List[Int](capacity=NACT)
    var kinds = List[Int](capacity=NACT)
    var flim = List[Int](capacity=NACT)
    var fmin = List[Float64](capacity=NACT)
    var fmax = List[Float64](capacity=NACT)
    var dyn = List[Float64](capacity=NACT)
    var aadr = List[Int](capacity=NACT)
    var dofa = List[Int](capacity=NACT)
    var trnn = List[Int](capacity=NACT)
    var tq = List[Int]()
    var td_ = List[Int]()
    var tc = List[Float64]()
    var kps = List[Float64](capacity=NACT)
    var kvs = List[Float64](capacity=NACT)
    var ten_k = List[Float64]()
    var ten_lo = List[Float64]()
    var ten_hi = List[Float64]()
    var ten_n = List[Int]()
    var ten_tq = List[Int]()
    var ten_td = List[Int]()
    var ten_tc = List[Float64]()
    comptime for ti in range(M._NTEN):
        ten_k.append(materialize[acd.tendon_stiffness[ti]]())
        ten_lo.append(materialize[acd.tendon_spring_lo[ti]]())
        ten_hi.append(materialize[acd.tendon_spring_hi[ti]]())
        ten_n.append(materialize[acd.tendon_trn_n[ti]]())
        comptime for wk in range(M._WRAPS):
            ten_tq.append(materialize[acd.tendon_trn_qadr[ti * M._WRAPS + wk]]())
            ten_td.append(materialize[acd.tendon_trn_dadr[ti * M._WRAPS + wk]]())
            ten_tc.append(materialize[acd.tendon_trn_coef[ti * M._WRAPS + wk]]())
    comptime for ai in range(NACT):
        gears.append(materialize[acd.motor_gears[ai]]())
        cmin.append(materialize[acd.motor_ctrl_min[ai]]())
        cmax.append(materialize[acd.motor_ctrl_max[ai]]())
        clim.append(materialize[acd.motor_ctrl_limited[ai]]())
        kinds.append(materialize[acd.motor_kind[ai]]())
        flim.append(materialize[acd.motor_force_limited[ai]]())
        fmin.append(materialize[acd.motor_force_min[ai]]())
        fmax.append(materialize[acd.motor_force_max[ai]]())
        dyn.append(materialize[acd.motor_dyn_tau[ai]]())
        aadr.append(materialize[acd.motor_act_adr[ai]]())
        dofa.append(materialize[acd.motor_dof_adr[ai]]())
        trnn.append(materialize[acd.motor_trn_n[ai]]())
        kps.append(materialize[acd.motor_kp[ai]]())
        kvs.append(materialize[acd.motor_kv[ai]]())
        comptime for wk in range(M._WRAPS):
            tq.append(materialize[acd.motor_trn_qadr[ai * M._WRAPS + wk]]())
            td_.append(materialize[acd.motor_trn_dadr[ai * M._WRAPS + wk]]())
            tc.append(materialize[acd.motor_trn_coef[ai * M._WRAPS + wk]]())
    _ = _compare[NACT, M._NTEN, M.NQ, M.NV, M.NKEY, M.NJOINT](
        "ctrllimited matrix (the `ctrlrange=\"0 0\"` case)",
        NACT, String(M.xml), gears, cmin, cmax, clim, kinds, flim, fmin, fmax,
        dyn, aadr, materialize[acd.na](),
        dofa, trnn, tq, td_, tc, M._WRAPS,
        kps, kvs, materialize[acd.bad_actuator](),
        ten_k, ten_lo, ten_hi, materialize[acd.ntendon](),
        ten_n, ten_tq, ten_td, ten_tc,
    )


def test_quadruped() raises:
    comptime M = DMQuadrupedWalkModel
    comptime acd = materialize[M._acd]()
    comptime NACT = M.nact
    var gears = List[Float64](capacity=NACT)
    var cmin = List[Float64](capacity=NACT)
    var cmax = List[Float64](capacity=NACT)
    var clim = List[Int](capacity=NACT)
    var kinds = List[Int](capacity=NACT)
    var flim = List[Int](capacity=NACT)
    var fmin = List[Float64](capacity=NACT)
    var fmax = List[Float64](capacity=NACT)
    var dyn = List[Float64](capacity=NACT)
    var aadr = List[Int](capacity=NACT)
    var dofa = List[Int](capacity=NACT)
    var trnn = List[Int](capacity=NACT)
    var tq = List[Int]()
    var td_ = List[Int]()
    var tc = List[Float64]()
    var kps = List[Float64](capacity=NACT)
    var kvs = List[Float64](capacity=NACT)
    var ten_k = List[Float64]()
    var ten_lo = List[Float64]()
    var ten_hi = List[Float64]()
    var ten_n = List[Int]()
    var ten_tq = List[Int]()
    var ten_td = List[Int]()
    var ten_tc = List[Float64]()
    comptime for ti in range(M._NTEN):
        ten_k.append(materialize[acd.tendon_stiffness[ti]]())
        ten_lo.append(materialize[acd.tendon_spring_lo[ti]]())
        ten_hi.append(materialize[acd.tendon_spring_hi[ti]]())
        ten_n.append(materialize[acd.tendon_trn_n[ti]]())
        comptime for wk in range(M._WRAPS):
            ten_tq.append(materialize[acd.tendon_trn_qadr[ti * M._WRAPS + wk]]())
            ten_td.append(materialize[acd.tendon_trn_dadr[ti * M._WRAPS + wk]]())
            ten_tc.append(materialize[acd.tendon_trn_coef[ti * M._WRAPS + wk]]())
    comptime for ai in range(NACT):
        gears.append(materialize[acd.motor_gears[ai]]())
        cmin.append(materialize[acd.motor_ctrl_min[ai]]())
        cmax.append(materialize[acd.motor_ctrl_max[ai]]())
        clim.append(materialize[acd.motor_ctrl_limited[ai]]())
        kinds.append(materialize[acd.motor_kind[ai]]())
        flim.append(materialize[acd.motor_force_limited[ai]]())
        fmin.append(materialize[acd.motor_force_min[ai]]())
        fmax.append(materialize[acd.motor_force_max[ai]]())
        dyn.append(materialize[acd.motor_dyn_tau[ai]]())
        aadr.append(materialize[acd.motor_act_adr[ai]]())
        dofa.append(materialize[acd.motor_dof_adr[ai]]())
        trnn.append(materialize[acd.motor_trn_n[ai]]())
        kps.append(materialize[acd.motor_kp[ai]]())
        kvs.append(materialize[acd.motor_kv[ai]]())
        comptime for wk in range(M._WRAPS):
            tq.append(materialize[acd.motor_trn_qadr[ai * M._WRAPS + wk]]())
            td_.append(materialize[acd.motor_trn_dadr[ai * M._WRAPS + wk]]())
            tc.append(materialize[acd.motor_trn_coef[ai * M._WRAPS + wk]]())
    _ = _compare[NACT, M._NTEN, M.NQ, M.NV, M.NKEY, M.NJOINT](
        "quadruped (3 classes / 12 actuators, tendons + dyntype)",
        NACT, String(M.xml), gears, cmin, cmax, clim, kinds, flim, fmin, fmax,
        dyn, aadr, materialize[acd.na](),
        dofa, trnn, tq, td_, tc, M._WRAPS,
        kps, kvs, materialize[acd.bad_actuator](),
        ten_k, ten_lo, ten_hi, materialize[acd.ntendon](),
        ten_n, ten_tq, ten_td, ten_tc,
    )


def test_dog() raises:
    comptime M = DMDogStandWalkModel
    comptime acd = materialize[M._acd]()
    comptime NACT = M.nact
    var gears = List[Float64](capacity=NACT)
    var cmin = List[Float64](capacity=NACT)
    var cmax = List[Float64](capacity=NACT)
    var clim = List[Int](capacity=NACT)
    var kinds = List[Int](capacity=NACT)
    var flim = List[Int](capacity=NACT)
    var fmin = List[Float64](capacity=NACT)
    var fmax = List[Float64](capacity=NACT)
    var dyn = List[Float64](capacity=NACT)
    var aadr = List[Int](capacity=NACT)
    var dofa = List[Int](capacity=NACT)
    var trnn = List[Int](capacity=NACT)
    var tq = List[Int]()
    var td_ = List[Int]()
    var tc = List[Float64]()
    var kps = List[Float64](capacity=NACT)
    var kvs = List[Float64](capacity=NACT)
    var ten_k = List[Float64]()
    var ten_lo = List[Float64]()
    var ten_hi = List[Float64]()
    var ten_n = List[Int]()
    var ten_tq = List[Int]()
    var ten_td = List[Int]()
    var ten_tc = List[Float64]()
    comptime for ti in range(M._NTEN):
        ten_k.append(materialize[acd.tendon_stiffness[ti]]())
        ten_lo.append(materialize[acd.tendon_spring_lo[ti]]())
        ten_hi.append(materialize[acd.tendon_spring_hi[ti]]())
        ten_n.append(materialize[acd.tendon_trn_n[ti]]())
        comptime for wk in range(M._WRAPS):
            ten_tq.append(materialize[acd.tendon_trn_qadr[ti * M._WRAPS + wk]]())
            ten_td.append(materialize[acd.tendon_trn_dadr[ti * M._WRAPS + wk]]())
            ten_tc.append(materialize[acd.tendon_trn_coef[ti * M._WRAPS + wk]]())
    comptime for ai in range(NACT):
        gears.append(materialize[acd.motor_gears[ai]]())
        cmin.append(materialize[acd.motor_ctrl_min[ai]]())
        cmax.append(materialize[acd.motor_ctrl_max[ai]]())
        clim.append(materialize[acd.motor_ctrl_limited[ai]]())
        kinds.append(materialize[acd.motor_kind[ai]]())
        flim.append(materialize[acd.motor_force_limited[ai]]())
        fmin.append(materialize[acd.motor_force_min[ai]]())
        fmax.append(materialize[acd.motor_force_max[ai]]())
        dyn.append(materialize[acd.motor_dyn_tau[ai]]())
        aadr.append(materialize[acd.motor_act_adr[ai]]())
        dofa.append(materialize[acd.motor_dof_adr[ai]]())
        trnn.append(materialize[acd.motor_trn_n[ai]]())
        kps.append(materialize[acd.motor_kp[ai]]())
        kvs.append(materialize[acd.motor_kv[ai]]())
        comptime for wk in range(M._WRAPS):
            tq.append(materialize[acd.motor_trn_qadr[ai * M._WRAPS + wk]]())
            td_.append(materialize[acd.motor_trn_dadr[ai * M._WRAPS + wk]]())
            tc.append(materialize[acd.motor_trn_coef[ai * M._WRAPS + wk]]())
    _ = _compare[NACT, M._NTEN, M.NQ, M.NV, M.NKEY, M.NJOINT](
        "dog stand/walk (24 classes / 38 actuators — the sharp case)",
        NACT, String(M.xml), gears, cmin, cmax, clim, kinds, flim, fmin, fmax,
        dyn, aadr, materialize[acd.na](),
        dofa, trnn, tq, td_, tc, M._WRAPS,
        kps, kvs, materialize[acd.bad_actuator](),
        ten_k, ten_lo, ten_hi, materialize[acd.ntendon](),
        ten_n, ten_tq, ten_td, ten_tc,
    )


def test_reach_forcerange() raises:
    """The model that makes `force_*` NON-VACUOUS.

    cartpole/quadruped/dog declare no `forcerange` at all, so their zeros
    agree with `_acd`'s zeros and prove nothing — the same trap `gear` fell
    into. jaco (manipulation reach) declares three distinct ranges
    (`-30.5 30.5`, `-6.8 6.8`, `-1 1`), so this is the model that actually
    exercises the resolution.
    """
    comptime M = ReachSiteFeaturesModel
    comptime acd = materialize[M._acd]()
    comptime NACT = M.nact
    var gears = List[Float64](capacity=NACT)
    var cmin = List[Float64](capacity=NACT)
    var cmax = List[Float64](capacity=NACT)
    var clim = List[Int](capacity=NACT)
    var kinds = List[Int](capacity=NACT)
    var flim = List[Int](capacity=NACT)
    var fmin = List[Float64](capacity=NACT)
    var fmax = List[Float64](capacity=NACT)
    var dyn = List[Float64](capacity=NACT)
    var aadr = List[Int](capacity=NACT)
    var dofa = List[Int](capacity=NACT)
    var trnn = List[Int](capacity=NACT)
    var tq = List[Int]()
    var td_ = List[Int]()
    var tc = List[Float64]()
    var kps = List[Float64](capacity=NACT)
    var kvs = List[Float64](capacity=NACT)
    var ten_k = List[Float64]()
    var ten_lo = List[Float64]()
    var ten_hi = List[Float64]()
    var ten_n = List[Int]()
    var ten_tq = List[Int]()
    var ten_td = List[Int]()
    var ten_tc = List[Float64]()
    comptime for ti in range(M._NTEN):
        ten_k.append(materialize[acd.tendon_stiffness[ti]]())
        ten_lo.append(materialize[acd.tendon_spring_lo[ti]]())
        ten_hi.append(materialize[acd.tendon_spring_hi[ti]]())
        ten_n.append(materialize[acd.tendon_trn_n[ti]]())
        comptime for wk in range(M._WRAPS):
            ten_tq.append(materialize[acd.tendon_trn_qadr[ti * M._WRAPS + wk]]())
            ten_td.append(materialize[acd.tendon_trn_dadr[ti * M._WRAPS + wk]]())
            ten_tc.append(materialize[acd.tendon_trn_coef[ti * M._WRAPS + wk]]())
    comptime for ai in range(NACT):
        gears.append(materialize[acd.motor_gears[ai]]())
        cmin.append(materialize[acd.motor_ctrl_min[ai]]())
        cmax.append(materialize[acd.motor_ctrl_max[ai]]())
        clim.append(materialize[acd.motor_ctrl_limited[ai]]())
        kinds.append(materialize[acd.motor_kind[ai]]())
        flim.append(materialize[acd.motor_force_limited[ai]]())
        fmin.append(materialize[acd.motor_force_min[ai]]())
        fmax.append(materialize[acd.motor_force_max[ai]]())
        dyn.append(materialize[acd.motor_dyn_tau[ai]]())
        aadr.append(materialize[acd.motor_act_adr[ai]]())
        dofa.append(materialize[acd.motor_dof_adr[ai]]())
        trnn.append(materialize[acd.motor_trn_n[ai]]())
        kps.append(materialize[acd.motor_kp[ai]]())
        kvs.append(materialize[acd.motor_kv[ai]]())
        comptime for wk in range(M._WRAPS):
            tq.append(materialize[acd.motor_trn_qadr[ai * M._WRAPS + wk]]())
            td_.append(materialize[acd.motor_trn_dadr[ai * M._WRAPS + wk]]())
            tc.append(materialize[acd.motor_trn_coef[ai * M._WRAPS + wk]]())
    _ = _compare[NACT, M._NTEN, M.NQ, M.NV, M.NKEY, M.NJOINT](
        "jaco reach (3 distinct forceranges — the non-vacuous force case)",
        NACT, String(M.xml), gears, cmin, cmax, clim, kinds, flim, fmin, fmax,
        dyn, aadr, materialize[acd.na](),
        dofa, trnn, tq, td_, tc, M._WRAPS,
        kps, kvs, materialize[acd.bad_actuator](),
        ten_k, ten_lo, ten_hi, materialize[acd.ntendon](),
        ten_n, ten_tq, ten_td, ten_tc,
    )


def test_fish() raises:
    """Added BEFORE the tendon-spring fields, not after.

    fish is the ONLY model in the tree with tendon springs — two `<fixed
    stiffness="1e-4">`. Every other model here declares none, so a
    `tendon_stiffness` row gated on cartpole/quadruped/dog/jaco alone would
    read 0 mismatches while testing nothing. That trap has already fired three
    times in this file (`gear`, `force_*`, `dyn_tau`); adding the model first
    is the cheap way not to make it four.

    fish also carries a tendon-TRANSMISSION actuator, so it independently
    re-covers the multi-wrap path that quadruped and dog exercise.
    """
    comptime M = DMFishSwimModel
    comptime acd = materialize[M._acd]()
    comptime NACT = M.nact
    var gears = List[Float64](capacity=NACT)
    var cmin = List[Float64](capacity=NACT)
    var cmax = List[Float64](capacity=NACT)
    var clim = List[Int](capacity=NACT)
    var kinds = List[Int](capacity=NACT)
    var flim = List[Int](capacity=NACT)
    var fmin = List[Float64](capacity=NACT)
    var fmax = List[Float64](capacity=NACT)
    var dyn = List[Float64](capacity=NACT)
    var aadr = List[Int](capacity=NACT)
    var dofa = List[Int](capacity=NACT)
    var trnn = List[Int](capacity=NACT)
    var tq = List[Int]()
    var td_ = List[Int]()
    var tc = List[Float64]()
    var kps = List[Float64](capacity=NACT)
    var kvs = List[Float64](capacity=NACT)
    var ten_k = List[Float64]()
    var ten_lo = List[Float64]()
    var ten_hi = List[Float64]()
    var ten_n = List[Int]()
    var ten_tq = List[Int]()
    var ten_td = List[Int]()
    var ten_tc = List[Float64]()
    comptime for ti in range(M._NTEN):
        ten_k.append(materialize[acd.tendon_stiffness[ti]]())
        ten_lo.append(materialize[acd.tendon_spring_lo[ti]]())
        ten_hi.append(materialize[acd.tendon_spring_hi[ti]]())
        ten_n.append(materialize[acd.tendon_trn_n[ti]]())
        comptime for wk in range(M._WRAPS):
            ten_tq.append(materialize[acd.tendon_trn_qadr[ti * M._WRAPS + wk]]())
            ten_td.append(materialize[acd.tendon_trn_dadr[ti * M._WRAPS + wk]]())
            ten_tc.append(materialize[acd.tendon_trn_coef[ti * M._WRAPS + wk]]())
    comptime for ai in range(NACT):
        gears.append(materialize[acd.motor_gears[ai]]())
        cmin.append(materialize[acd.motor_ctrl_min[ai]]())
        cmax.append(materialize[acd.motor_ctrl_max[ai]]())
        clim.append(materialize[acd.motor_ctrl_limited[ai]]())
        kinds.append(materialize[acd.motor_kind[ai]]())
        flim.append(materialize[acd.motor_force_limited[ai]]())
        fmin.append(materialize[acd.motor_force_min[ai]]())
        fmax.append(materialize[acd.motor_force_max[ai]]())
        dyn.append(materialize[acd.motor_dyn_tau[ai]]())
        aadr.append(materialize[acd.motor_act_adr[ai]]())
        dofa.append(materialize[acd.motor_dof_adr[ai]]())
        trnn.append(materialize[acd.motor_trn_n[ai]]())
        kps.append(materialize[acd.motor_kp[ai]]())
        kvs.append(materialize[acd.motor_kv[ai]]())
        comptime for wk in range(M._WRAPS):
            tq.append(materialize[acd.motor_trn_qadr[ai * M._WRAPS + wk]]())
            td_.append(materialize[acd.motor_trn_dadr[ai * M._WRAPS + wk]]())
            tc.append(materialize[acd.motor_trn_coef[ai * M._WRAPS + wk]]())
    _ = _compare[NACT, M._NTEN, M.NQ, M.NV, M.NKEY, M.NJOINT](
        "fish swim (the ONLY model with tendon springs)",
        NACT, String(M.xml), gears, cmin, cmax, clim, kinds, flim, fmin, fmax,
        dyn, aadr, materialize[acd.na](),
        dofa, trnn, tq, td_, tc, M._WRAPS,
        kps, kvs, materialize[acd.bad_actuator](),
        ten_k, ten_lo, ten_hi, materialize[acd.ntendon](),
        ten_n, ten_tq, ten_td, ten_tc,
    )


def _compare_pose[
    NACT_C: Int,
    NTEN_C: Int,
    NQ_C: Int,
    NV_C: Int,
    NKEY_C: Int,
    NJOINT_C: Int,
](
    name: String,
    xml: String,
    qpos0: List[Float64],
    nq_acd: Int,
    fj_adr: Int,
    ktime: List[Float64],
    knqpos: List[Int],
    knqvel: List[Int],
    knctrl: List[Int],
    kqpos: List[Float64],
    kqvel: List[Float64],
    kctrl: List[Float64],
    nkey_acd: Int,
    nq0: Int,
    nact0: Int,
    jlim: List[Int],
    jadr: List[Int],
    jlo: List[Float64],
    jhi: List[Float64],
) raises:
    """Model-level records: qpos0 and keyframes.

    Separate from `_compare` because these are per-MODEL, not per-actuator,
    and threading them through that signature would have made it unreadable
    for no gain.
    """
    var fmd = parse_xml_full(xml)
    print("---  ", name, "  ---")
    print("    qpos0_nq: runtime =", fmd.qpos0_nq, " _acd =", nq_acd)
    print("    free_joint_qpos_adr: runtime =", fmd.free_joint_qpos_adr,
          " _acd =", fj_adr)
    print("    nkey: runtime =", fmd.nkey, " _acd =", nkey_acd)

    var n_q = 0
    var nq_cmp = fmd.qpos0_nq if fmd.qpos0_nq < nq_acd else nq_acd
    var q_nonzero = False
    for i in range(nq_cmp):
        if i < len(fmd.qpos0) and i < len(qpos0):
            if abs(fmd.qpos0[i] - qpos0[i]) > 0.0:
                n_q += 1
            if qpos0[i] != 0.0:
                q_nonzero = True
    if n_q > 0:  # kept: a count alone does not show WHICH way it is wrong
        for i in range(nq_cmp if nq_cmp < 8 else 8):
            print("        qpos0[", i, "] runtime", fmd.qpos0[i],
                  " _acd", qpos0[i])
    print("    qpos0 mismatches =", n_q, " over", nq_cmp,
          " (any non-zero entry:", q_nonzero, ")")
    if not q_nonzero and nq_cmp > 0:
        print("  ⚠ qpos0 is ALL ZERO here — this row is VACUOUS")

    var n_k = 0
    var k_nonzero = False
    var nk = fmd.nkey if fmd.nkey < nkey_acd else nkey_acd
    for k in range(nk):
        if abs(fmd.key_time[k] - ktime[k]) > 0.0:
            n_k += 1
        if fmd.key_nqpos[k] != knqpos[k]:
            n_k += 1
        for i in range(nq0):
            var a = fmd.key_qpos[k * nq0 + i] if k * nq0 + i < len(
                fmd.key_qpos
            ) else 0.0
            var b = kqpos[k * nq0 + i]
            if abs(a - b) > 0.0:
                n_k += 1
            if b != 0.0:
                k_nonzero = True
        for i in range(nact0):
            var a2 = fmd.key_ctrl[k * nact0 + i] if k * nact0 + i < len(
                fmd.key_ctrl
            ) else 0.0
            if abs(a2 - kctrl[k * nact0 + i]) > 0.0:
                n_k += 1
    print("    keyframe mismatches =", n_k, " over", nk,
          "keys (any non-zero key qpos:", k_nonzero, ")")
    if nkey_acd == 0:
        print("  ⚠ no <keyframe> here — the keyframe rows are VACUOUS")

    assert_true(fmd.qpos0_nq == nq_acd,
        String(name, ": qpos0_nq runtime ", fmd.qpos0_nq, " vs `_acd` ",
               nq_acd))
    assert_true(fmd.free_joint_qpos_adr == fj_adr,
        String(name, ": free_joint_qpos_adr runtime ",
               fmd.free_joint_qpos_adr, " vs `_acd` ", fj_adr))
    assert_true(n_q == 0, String(name, ": qpos0 disagrees in ", n_q))
    assert_true(fmd.nkey == nkey_acd,
        String(name, ": nkey runtime ", fmd.nkey, " vs `_acd` ", nkey_acd))
    assert_true(n_k == 0, String(name, ": keyframes disagree in ", n_k))

    # ═══ PHASE 1a.4 — the pose/keyframe TENSORS, same `_acd` values ═══════
    #
    # ⚠ WITHOUT THIS THE NEW TENSORS ARE GREEN AND UNTESTED. Everything above
    # diffs `FlatModelDef` against `_acd`; `build_spec_fields` is a SECOND
    # transcription and 1a.2's `kp <- kv` negative control is the reason to
    # believe a column can be wrong while the record is right.
    var sf = SpecFields[
        DType.float64, NACT_C, NTEN_C, NQ_C, NV_C, NKEY_C, NJOINT_C
    ]()
    build_spec_fields[
        DType.float64, NACT_C, NTEN_C, NQ_C, NV_C, NKEY_C, NJOINT_C
    ](fmd, sf)

    var n_tq = 0
    for i in range(nq_cmp):
        if i < NQ_C and sf.qpos0.data[i] != Scalar[DType.float64](qpos0[i]):
            n_tq += 1
    var t_nq = Int(sf.pose_meta.data[POSE_IDX_QPOS0_NQ])
    var t_fj = Int(sf.pose_meta.data[POSE_IDX_FREE_JOINT_QPOS_ADR])

    var n_tk = 0
    for k in range(nk):
        var o = k * KEY_META_SIZE
        if (
            sf.key_meta.data[o + KEY_IDX_TIME]
            != Scalar[DType.float64](ktime[k])
            or Int(sf.key_meta.data[o + KEY_IDX_NQPOS]) != knqpos[k]
            or Int(sf.key_meta.data[o + KEY_IDX_NQVEL]) != knqvel[k]
            or Int(sf.key_meta.data[o + KEY_IDX_NCTRL]) != knctrl[k]
        ):
            n_tk += 1
        for i in range(NQ_C):
            if sf.key_qpos.data[k * NQ_C + i] != Scalar[DType.float64](
                kqpos[k * nq0 + i]
            ):
                n_tk += 1
        # ⚠ THE STRIDES DIFFER: `_acd` walks `k * nq0 + i`, the tensor
        # `k * NV_C + i`. Reading them the same way walks into the next key on
        # any model with nq != nv — which is every model with a free joint.
        for i in range(NV_C):
            if sf.key_qvel.data[k * NV_C + i] != Scalar[DType.float64](
                kqvel[k * nq0 + i]
            ):
                n_tk += 1
        for i in range(NACT_C):
            if sf.key_ctrl.data[k * NACT_C + i] != Scalar[DType.float64](
                kctrl[k * nact0 + i]
            ):
                n_tk += 1

    print("    TENSORS: qpos0 =", n_tq, " keyframes =", n_tk,
          "  pose_meta nq =", t_nq, " free_joint_adr =", t_fj)

    assert_true(n_tq == 0,
        String(name, ": SpecFields qpos0 disagrees with `_acd` in ", n_tq))
    assert_true(t_nq == nq_acd and t_fj == fj_adr,
        String(name, ": SpecFields pose_meta disagrees — nq ", t_nq, " vs ",
               nq_acd, ", free_joint_adr ", t_fj, " vs ", fj_adr))
    assert_true(n_tk == 0,
        String(name, ": SpecFields keyframe rows disagree with `_acd` in ",
               n_tk))

    # ── joint limits (the `enforce_limits` clamp) ────────────────────────
    #
    # ⚠⚠ THIS ROW DOES NOT ASSERT AGAINST `_acd`, AND THAT IS THE FINDING.
    # `_acd`'s joint scan reads `limited`/`range` off the ELEMENT TAG ONLY
    # (`xml_parser.mojo:4493`) — no `<default class=...>` resolution, the same
    # gap 1a.1 fixed on the actuator path (`224135af`). Measured here:
    #
    #     so_arm100  j0  tensor lim 1 [-1.92, 1.92]   `_acd` lim 0 [0, 0]
    #     quadruped  17 of 17 joints disagree
    #     ant         1 of  9,  finger 1 of 3
    #
    # so_arm100 keeps EVERY range in a `<default class="Rotation">`-style
    # block, so `_acd` saw none of them and reported the model unlimited.
    # `enforce_limits` — the ONLY consumer of those four arrays — therefore
    # clamped NOTHING on any model whose ranges come from a class. The
    # runtime side is the correct one and is pinned to MuJoCo elsewhere
    # (`test_quadruped_body_and_joint_constants_match_mujoco` reports
    # "limited joints = 16" against `mjModel`).
    #
    # ⇒ The `_acd` diff below is PRINTED as a measurement. What is ASSERTED is
    # the second leg — tensor against the runtime record it is built from —
    # which is what 1a.2's `kp <- kv` control showed can independently break.
    var n_jl_acd = 0
    var n_jl = 0
    var n_limited = 0
    for j in range(NJOINT_C):
        var o = j * JLIM_SIZE
        if jlim[j] != 0:
            n_limited += 1
        if (
            Int(sf.joint_limits.data[o + JLIM_IDX_LIMITED]) != jlim[j]
            or Int(sf.joint_limits.data[o + JLIM_IDX_QPOS_ADR]) != jadr[j]
            or sf.joint_limits.data[o + JLIM_IDX_RANGE_MIN]
            != Scalar[DType.float64](jlo[j])
            or sf.joint_limits.data[o + JLIM_IDX_RANGE_MAX]
            != Scalar[DType.float64](jhi[j])
        ):
            n_jl_acd += 1
        if j < len(fmd.joints):
            var jd = fmd.joints[j]
            if (
                (sf.joint_limits.data[o + JLIM_IDX_LIMITED] != 0)
                != jd.is_limited
                or sf.joint_limits.data[o + JLIM_IDX_RANGE_MIN]
                != Scalar[DType.float64](jd.range_min)
                or sf.joint_limits.data[o + JLIM_IDX_RANGE_MAX]
                != Scalar[DType.float64](jd.range_max)
            ):
                n_jl += 1
    var n_lim_rt = 0
    for j in range(NJOINT_C):
        if sf.joint_limits.data[j * JLIM_SIZE + JLIM_IDX_LIMITED] != 0:
            n_lim_rt += 1
    print("    joint limits: vs record =", n_jl, " of", NJOINT_C,
          "  (limited: runtime", n_lim_rt, " `_acd`", n_limited,
          " -> `_acd` disagrees on", n_jl_acd, ")")
    if n_lim_rt == 0:
        print("  ⚠ NO joint is limited — the joint_limits rows are VACUOUS")
    assert_true(n_jl == 0,
        String(name, ": SpecFields joint_limits disagree with the runtime",
               " record in ", n_jl))


# ═══ The `nq != nv` keyframe fixture — the STRIDE the tree cannot test ═══
#
# ⚠⚠ `_acd.key_qvel` STRIDES BY `NQ0`; the `SpecFields` tensor is honestly
# `[NKEY, NV]` and strides by NV. On every model in this repo that carries a
# `<keyframe>` — so_arm100, and it is the ONLY one — nq == nv == 6, so the two
# expressions are the same number and a stride bug is INVISIBLE. Verified:
# swapping the tensor's stride to NQ leaves all 10 tests green.
#
# ⚠ AND TWO KEYS ARE REQUIRED, not one. The strides differ by `k * (nq - nv)`,
# which is ZERO for k = 0 whatever the dims. so_arm100 has two keys and still
# could not see it because its dims tie; a free-jointed model with one key
# would fail for the mirror-image reason.
#
# So: a free joint (nq 7 / nv 6) plus a hinge => nq 8, nv 7, and two keys with
# deliberately distinct qvel values.
comptime KEYSTRIDE_XML = String(
    """<mujoco model="keyframe_stride">
  <option timestep="0.002"/>
  <worldbody>
    <body name="root" pos="0 0 .5">
      <freejoint/>
      <geom type="sphere" size=".05" mass="1"/>
      <body name="arm" pos=".1 0 0">
        <joint name="j" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 .1 0 0" size=".02" mass=".1"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="a" joint="j" gear="2" ctrlrange="-1 1"/></actuator>
  <keyframe>
    <key name="k0" time="0.25"
         qpos="0.1 0.2 0.3 0.7071068 0 0.7071068 0 0.4"
         qvel="1 2 3 4 5 6 7" ctrl="0.5"/>
    <key name="k1" time="0.75"
         qpos="-0.1 -0.2 -0.3 1 0 0 0 -0.4"
         qvel="11 12 13 14 15 16 17" ctrl="-0.5"/>
  </keyframe>
</mujoco>"""
)

comptime _kspm = parse_xml(KEYSTRIDE_XML)
comptime KeyStrideModel = ModelDefFromXML[
    xml=KEYSTRIDE_XML,
    nbody=_kspm.NBODY, njoint=_kspm.NJOINT, nq=_kspm.NQ, nv=_kspm.NV,
    ngeom=_kspm.NGEOM, nact=_kspm.NACT, ntex=_kspm.NTEX, nmat=_kspm.NMAT,
    nlight=_kspm.NLIGHT, ncam=_kspm.NCAM, nsite=_kspm.NSITE, neq=_kspm.NEQ,
    nexclude=_kspm.NEXCLUDE, npair=_kspm.NPAIR, max_tendon=_kspm.NTENDON,
    max_condim=_kspm.MAX_CONDIM, max_contacts=8,
    obs_dim_override=1, obs_qpos_skip=0,
    timestep=_kspm.TIMESTEP, noslip_iter=_kspm.NOSLIP_ITER,
    nkey = 2,
]


def test_pose_finger() raises:
    comptime M = DMFingerSpinModel
    comptime acd = materialize[M._acd]()
    var qp = List[Float64]()
    comptime for i in range(M._NQ0):
        qp.append(materialize[acd.qpos0[i]]())
    var jlim = List[Int]()
    var jadr = List[Int]()
    var jlo = List[Float64]()
    var jhi = List[Float64]()
    comptime for j in range(M._NJNT):
        jlim.append(1 if materialize[acd.joint_is_limited[j]]() else 0)
        jadr.append(materialize[acd.joint_qpos_adr[j]]())
        jlo.append(materialize[acd.joint_range_min[j]]())
        jhi.append(materialize[acd.joint_range_max[j]]())
    var kt = List[Float64]()
    var knq = List[Int]()
    var knv = List[Int]()
    var knc = List[Int]()
    comptime for k in range(acd.NKEYS):
        kt.append(materialize[acd.key_time[k]]())
        knq.append(materialize[acd.key_nqpos[k]]())
        knv.append(materialize[acd.key_nqvel[k]]())
        knc.append(materialize[acd.key_nctrl[k]]())
    var kq = List[Float64]()
    var kv = List[Float64]()
    comptime for i in range(acd.NKEYS * M._NQ0):
        kq.append(materialize[acd.key_qpos[i]]())
        # ⚠ `_acd.key_qvel` STRIDES BY `_NQ0`, NOT `_NV` — one allocation
        # shape for both arrays on the comptime side. The tensor is honestly
        # [NKEY, NV], so the diff below has to convert.
        kv.append(materialize[acd.key_qvel[i]]())
    var kc = List[Float64]()
    comptime for i in range(acd.NKEYS * M._NACT):
        kc.append(materialize[acd.key_ctrl[i]]())
    _compare_pose[M.NACT, M._NTEN, M.NQ, M.NV, M.NKEY, M.NJOINT](
        "finger spin (joint ref=-90 -> -pi/2, the deg-conversion case)",
        String(M.xml), qp, materialize[acd.nq](),
        materialize[acd.free_joint_qpos_adr](),
        kt, knq, knv, knc, kq, kv, kc, materialize[acd.nkey](),
        M._NQ0, M._NACT, jlim, jadr, jlo, jhi,
    )


def test_pose_key_stride() raises:
    """The `nq != nv` + two-keys case, for the key_qvel STRIDE. See the
    fixture: no model in the tree can distinguish it."""
    comptime M = KeyStrideModel
    comptime acd = materialize[M._acd]()
    var qp = List[Float64]()
    comptime for i in range(M._NQ0):
        qp.append(materialize[acd.qpos0[i]]())
    var jlim = List[Int]()
    var jadr = List[Int]()
    var jlo = List[Float64]()
    var jhi = List[Float64]()
    comptime for j in range(M._NJNT):
        jlim.append(1 if materialize[acd.joint_is_limited[j]]() else 0)
        jadr.append(materialize[acd.joint_qpos_adr[j]]())
        jlo.append(materialize[acd.joint_range_min[j]]())
        jhi.append(materialize[acd.joint_range_max[j]]())
    var kt = List[Float64]()
    var knq = List[Int]()
    var knv = List[Int]()
    var knc = List[Int]()
    comptime for k in range(acd.NKEYS):
        kt.append(materialize[acd.key_time[k]]())
        knq.append(materialize[acd.key_nqpos[k]]())
        knv.append(materialize[acd.key_nqvel[k]]())
        knc.append(materialize[acd.key_nctrl[k]]())
    var kq = List[Float64]()
    var kv = List[Float64]()
    comptime for i in range(acd.NKEYS * M._NQ0):
        kq.append(materialize[acd.key_qpos[i]]())
        # ⚠ `_acd.key_qvel` STRIDES BY `_NQ0`, NOT `_NV` — one allocation
        # shape for both arrays on the comptime side. The tensor is honestly
        # [NKEY, NV], so the diff below has to convert.
        kv.append(materialize[acd.key_qvel[i]]())
    var kc = List[Float64]()
    comptime for i in range(acd.NKEYS * M._NACT):
        kc.append(materialize[acd.key_ctrl[i]]())
    _compare_pose[M.NACT, M._NTEN, M.NQ, M.NV, M.NKEY, M.NJOINT](
        "keyframe stride (free joint: nq 8 != nv 7, two keys)",
        String(M.xml), qp, materialize[acd.nq](),
        materialize[acd.free_joint_qpos_adr](),
        kt, knq, knv, knc, kq, kv, kc, materialize[acd.nkey](),
        M._NQ0, M._NACT, jlim, jadr, jlo, jhi,
    )


def test_pose_ant() raises:
    comptime M = AntModel
    comptime acd = materialize[M._acd]()
    var qp = List[Float64]()
    comptime for i in range(M._NQ0):
        qp.append(materialize[acd.qpos0[i]]())
    var jlim = List[Int]()
    var jadr = List[Int]()
    var jlo = List[Float64]()
    var jhi = List[Float64]()
    comptime for j in range(M._NJNT):
        jlim.append(1 if materialize[acd.joint_is_limited[j]]() else 0)
        jadr.append(materialize[acd.joint_qpos_adr[j]]())
        jlo.append(materialize[acd.joint_range_min[j]]())
        jhi.append(materialize[acd.joint_range_max[j]]())
    var kt = List[Float64]()
    var knq = List[Int]()
    var knv = List[Int]()
    var knc = List[Int]()
    comptime for k in range(acd.NKEYS):
        kt.append(materialize[acd.key_time[k]]())
        knq.append(materialize[acd.key_nqpos[k]]())
        knv.append(materialize[acd.key_nqvel[k]]())
        knc.append(materialize[acd.key_nctrl[k]]())
    var kq = List[Float64]()
    var kv = List[Float64]()
    comptime for i in range(acd.NKEYS * M._NQ0):
        kq.append(materialize[acd.key_qpos[i]]())
        # ⚠ `_acd.key_qvel` STRIDES BY `_NQ0`, NOT `_NV` — one allocation
        # shape for both arrays on the comptime side. The tensor is honestly
        # [NKEY, NV], so the diff below has to convert.
        kv.append(materialize[acd.key_qvel[i]]())
    var kc = List[Float64]()
    comptime for i in range(acd.NKEYS * M._NACT):
        kc.append(materialize[acd.key_ctrl[i]]())
    _compare_pose[M.NACT, M._NTEN, M.NQ, M.NV, M.NKEY, M.NJOINT](
        "ant (<custom> init_qpos OVERRIDES the joint refs)",
        String(M.xml), qp, materialize[acd.nq](),
        materialize[acd.free_joint_qpos_adr](),
        kt, knq, knv, knc, kq, kv, kc, materialize[acd.nkey](),
        M._NQ0, M._NACT, jlim, jadr, jlo, jhi,
    )


def test_pose_so_arm100() raises:
    comptime M = SoArm100Model
    comptime acd = materialize[M._acd]()
    var qp = List[Float64]()
    comptime for i in range(M._NQ0):
        qp.append(materialize[acd.qpos0[i]]())
    var jlim = List[Int]()
    var jadr = List[Int]()
    var jlo = List[Float64]()
    var jhi = List[Float64]()
    comptime for j in range(M._NJNT):
        jlim.append(1 if materialize[acd.joint_is_limited[j]]() else 0)
        jadr.append(materialize[acd.joint_qpos_adr[j]]())
        jlo.append(materialize[acd.joint_range_min[j]]())
        jhi.append(materialize[acd.joint_range_max[j]]())
    var kt = List[Float64]()
    var knq = List[Int]()
    var knv = List[Int]()
    var knc = List[Int]()
    comptime for k in range(acd.NKEYS):
        kt.append(materialize[acd.key_time[k]]())
        knq.append(materialize[acd.key_nqpos[k]]())
        knv.append(materialize[acd.key_nqvel[k]]())
        knc.append(materialize[acd.key_nctrl[k]]())
    var kq = List[Float64]()
    var kv = List[Float64]()
    comptime for i in range(acd.NKEYS * M._NQ0):
        kq.append(materialize[acd.key_qpos[i]]())
        # ⚠ `_acd.key_qvel` STRIDES BY `_NQ0`, NOT `_NV` — one allocation
        # shape for both arrays on the comptime side. The tensor is honestly
        # [NKEY, NV], so the diff below has to convert.
        kv.append(materialize[acd.key_qvel[i]]())
    var kc = List[Float64]()
    comptime for i in range(acd.NKEYS * M._NACT):
        kc.append(materialize[acd.key_ctrl[i]]())
    _compare_pose[M.NACT, M._NTEN, M.NQ, M.NV, M.NKEY, M.NJOINT](
        "so_arm100 (the ONLY model with a <keyframe>)",
        String(M.xml), qp, materialize[acd.nq](),
        materialize[acd.free_joint_qpos_adr](),
        kt, knq, knv, knc, kq, kv, kc, materialize[acd.nkey](),
        M._NQ0, M._NACT, jlim, jadr, jlo, jhi,
    )


def test_pose_quadruped_pose() raises:
    comptime M = DMQuadrupedWalkModel
    comptime acd = materialize[M._acd]()
    var qp = List[Float64]()
    comptime for i in range(M._NQ0):
        qp.append(materialize[acd.qpos0[i]]())
    var jlim = List[Int]()
    var jadr = List[Int]()
    var jlo = List[Float64]()
    var jhi = List[Float64]()
    comptime for j in range(M._NJNT):
        jlim.append(1 if materialize[acd.joint_is_limited[j]]() else 0)
        jadr.append(materialize[acd.joint_qpos_adr[j]]())
        jlo.append(materialize[acd.joint_range_min[j]]())
        jhi.append(materialize[acd.joint_range_max[j]]())
    var kt = List[Float64]()
    var knq = List[Int]()
    var knv = List[Int]()
    var knc = List[Int]()
    comptime for k in range(acd.NKEYS):
        kt.append(materialize[acd.key_time[k]]())
        knq.append(materialize[acd.key_nqpos[k]]())
        knv.append(materialize[acd.key_nqvel[k]]())
        knc.append(materialize[acd.key_nctrl[k]]())
    var kq = List[Float64]()
    var kv = List[Float64]()
    comptime for i in range(acd.NKEYS * M._NQ0):
        kq.append(materialize[acd.key_qpos[i]]())
        # ⚠ `_acd.key_qvel` STRIDES BY `_NQ0`, NOT `_NV` — one allocation
        # shape for both arrays on the comptime side. The tensor is honestly
        # [NKEY, NV], so the diff below has to convert.
        kv.append(materialize[acd.key_qvel[i]]())
    var kc = List[Float64]()
    comptime for i in range(acd.NKEYS * M._NACT):
        kc.append(materialize[acd.key_ctrl[i]]())
    _compare_pose[M.NACT, M._NTEN, M.NQ, M.NV, M.NKEY, M.NJOINT](
        "quadruped (free joint -> body pos + qw=1)",
        String(M.xml), qp, materialize[acd.nq](),
        materialize[acd.free_joint_qpos_adr](),
        kt, knq, knv, knc, kq, kv, kc, materialize[acd.nkey](),
        M._NQ0, M._NACT, jlim, jadr, jlo, jhi,
    )


def main() raises:
    print("Phase 1a.1 differential gate — runtime records vs comptime `_acd`")
    print("see docs/PHYSICS3D_RUNTIME_DIMS_ASSESSMENT.md §10.9")
    print("Ported groups ASSERT; unported ones only print. `kind` is a")
    print("pending encoding decision (phase 1a.3), not a defect.")
    print("")
    TestSuite.discover_tests[__functions_in_module()]().run()
