"""Phase 1a.1 — the differential gate that licenses the whole phase.

`docs/PHYSICS3D_RUNTIME_DIMS_ASSESSMENT.md` §10.9. Phase 1a moves actuator data
off the comptime `ComptimeActData` (`_acd`) and onto the runtime records. That
is ~20 arrays of MJCF semantics hand-ported into a second parser — a textbook
silent-bug generator — UNLESS it is diffed against the thing it replaces.

This is that diff, and it only works while `_acd` still exists. Same instrument
as `tests/physics3d/test_defaults_index_equivalence.mojo` (`585216bb`).

Round 1 covers the fields BOTH sides already carry, which is deliberately a
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
from std.testing import TestSuite

from mojo_rl.physics3d.parser import parse_xml_full

from mojo_rl.envs.dm_control.cartpole.cartpole_xml import DMCartpole1Model
from mojo_rl.envs.dm_control.quadruped.quadruped_xml import DMQuadrupedWalkModel
from mojo_rl.envs.dm_control.dog.dog_xml import DMDogStandWalkModel
from mojo_rl.envs.dm_control.manipulation_reach_def import (
    ReachSiteFeaturesModel,
)


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
    var worst_gear: Float64
    var worst_gear_i: Int


def _compare(
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

    var r = _Report(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, -1)
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
    print("    na: runtime  =", fmd.na, "  _acd =", na_acd,
          "->", "OK" if fmd.na == na_acd else "DIFFERS")

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
    return r^


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
    _ = _compare(
        "cartpole (1 actuator, no <default> class)",
        NACT, String(M.xml), gears, cmin, cmax, clim, kinds, flim, fmin, fmax,
        dyn, aadr, materialize[acd.na](),
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
    _ = _compare(
        "quadruped (3 classes / 12 actuators, tendons + dyntype)",
        NACT, String(M.xml), gears, cmin, cmax, clim, kinds, flim, fmin, fmax,
        dyn, aadr, materialize[acd.na](),
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
    _ = _compare(
        "dog stand/walk (24 classes / 38 actuators — the sharp case)",
        NACT, String(M.xml), gears, cmin, cmax, clim, kinds, flim, fmin, fmax,
        dyn, aadr, materialize[acd.na](),
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
    _ = _compare(
        "jaco reach (3 distinct forceranges — the non-vacuous force case)",
        NACT, String(M.xml), gears, cmin, cmax, clim, kinds, flim, fmin, fmax,
        dyn, aadr, materialize[acd.na](),
    )


def main() raises:
    print("Phase 1a.1 differential gate — runtime records vs comptime `_acd`")
    print("see docs/PHYSICS3D_RUNTIME_DIMS_ASSESSMENT.md §10.9")
    print("⚠ ROUND 1 IS A MEASUREMENT, NOT A PASS/FAIL — it sizes the split.")
    print("")
    TestSuite.discover_tests[__functions_in_module()]().run()
