"""`parse_xml`'s dimensions against MuJoCo's, over every model in the tree.

⚠ THIS GATE EXISTS BECAUSE PHASE 1b IS ABOUT TO REPLACE `parse_xml`. The dims
are the last thing forcing the MJCF to be a comptime `String`: `parse_xml`
runs in the comptime interpreter, the interpreter cannot `open()` a file
(§10.2), so the XML has to be in the source. Phase 1b generates the dims from
MuJoCo instead and reads the XML at runtime. That swap is only safe if MuJoCo
and `parse_xml` already agree — so this measures it BEFORE anything moves,
and keeps measuring it afterwards as the standing "our parser agrees with
`mjModel`" check.

Nothing checked that systematically before this file. `parse_xml` is used at
comptime by 56 model instantiations and its counts were verified per-model, by
hand, whenever a model was ported.

WHAT IS COMPARED — 20 fields × 56 models, every composed MJCF in the tree
(gym MuJoCo, dm_control suite + manipulation, dog, SO-ARM, Sawyer):

  counts     NBODY NJOINT NQ NV NGEOM NACT NTEX NMAT NLIGHT NCAM NSITE
             NEQ NEXCLUDE NPAIR NTENDON
  <option>   TIMESTEP NOSLIP_ITER CCD_TOL CCD_ITER
  derived    MAX_CONDIM

⚠ TWO ROWS ARE NOT RAW MuJoCo FIELDS, and both would be wrong if they were:

  * NEQ is the weld/connect/joint EQUALITY SLAB. `<equality><tendon>`
    (mjEQ_TENDON) is deliberately excluded — it rides on the tendon record
    via `TENDON_IDX_IS_EQUALITY`, not in that slab (`xml_parser.mojo:2407`).
    Comparing raw `m.neq` reports 9 differences across quadruped, manipulator
    and stacker that are a difference of DEFINITION, not of counting. The row
    subtracts the tendon equalities so it measures the slab it sizes.
  * MAX_CONDIM is ours "largest `condim=` anywhere in the file"; MuJoCo keeps
    condim per-geom and per-pair, so the comparison is against the max over
    both of those arrays.

⚠ THREE FIELDS WERE VACUOUS OVER THE 56 REAL MODELS — `NPAIR`, `CCD_TOL` and
`CCD_ITER` take exactly ONE value each across all of them (0, 1e-6, 35: the
defaults). Rows like that compare a default against itself and would pass with
the parser deleted. `_FIXTURE_XML` exists solely to give them a second value,
and the run prints the distinct-value count per field so a row that goes
vacuous later says so out loud rather than going quietly green.
See `feedback_the_sweep_was_not_the_distribution`.

Run: pixi run mojo run -I . tests/physics3d/test_model_dims_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true

from mojo_rl.physics3d.parser import parse_xml

from mojo_rl.envs.ant.ant_xml import ant_xml
from mojo_rl.envs.half_cheetah.half_cheetah_xml import half_cheetah_xml
from mojo_rl.envs.hopper.hopper_xml import hopper_xml
from mojo_rl.envs.humanoid.humanoid_xml import humanoid_xml
from mojo_rl.envs.humanoid_standup.humanoid_standup_xml import humanoid_standup_xml
from mojo_rl.envs.inverted_double_pendulum.inverted_double_pendulum_xml import inverted_double_pendulum_xml
from mojo_rl.envs.inverted_pendulum.inverted_pendulum_xml import inverted_pendulum_xml
from mojo_rl.envs.pusher.pusher_xml import pusher_xml
from mojo_rl.envs.reacher.reacher_xml import reacher_xml
from mojo_rl.envs.swimmer.swimmer_xml import swimmer_xml
from mojo_rl.envs.walker2d.walker2d_xml import walker2d_xml
from mojo_rl.envs.metaworld.sawyer_reach_xml import sawyer_reach_xml
from mojo_rl.envs.robots.so_arm100_xml import SO_ARM100_XML
from mojo_rl.envs.robots.so_arm101_xml import SO_ARM101_XML
from mojo_rl.envs.dm_control.acrobot.acrobot_xml import dm_acrobot_xml
from mojo_rl.envs.dm_control.ball_in_cup.ball_in_cup_xml import dm_ball_in_cup_xml
from mojo_rl.envs.dm_control.cartpole.cartpole_xml import (
    dm_cartpole1_xml,
    dm_cartpole2_xml,
    dm_cartpole3_xml
)
from mojo_rl.envs.dm_control.cheetah.cheetah_xml import dm_cheetah_xml
from mojo_rl.envs.dm_control.finger.finger_xml import (
    dm_finger_xml,
    dm_finger_spin_xml
)
from mojo_rl.envs.dm_control.fish.fish_xml import dm_fish_xml
from mojo_rl.envs.dm_control.hopper.hopper_xml import dm_hopper_xml
from mojo_rl.envs.dm_control.humanoid.humanoid_xml import dm_humanoid_xml
from mojo_rl.envs.dm_control.humanoid_cmu.humanoid_cmu_xml import dm_humanoid_cmu_xml
from mojo_rl.envs.dm_control.manipulator.manipulator_xml import (
    dm_manipulator_bring_ball_xml,
    dm_manipulator_bring_peg_xml,
    dm_manipulator_insert_ball_xml,
    dm_manipulator_insert_peg_xml
)
from mojo_rl.envs.dm_control.pendulum.pendulum_xml import dm_pendulum_xml
from mojo_rl.envs.dm_control.point_mass.point_mass_xml import dm_point_mass_xml
from mojo_rl.envs.dm_control.quadruped.quadruped_xml import (
    dm_quadruped_walk_xml,
    dm_quadruped_run_xml,
    dm_quadruped_fetch_xml
)
from mojo_rl.envs.dm_control.reacher.reacher_xml import dm_reacher_xml
from mojo_rl.envs.dm_control.stacker.stacker_xml import (
    dm_stacker_2_xml,
    dm_stacker_4_xml
)
from mojo_rl.envs.dm_control.swimmer.swimmer_xml import (
    dm_swimmer6_xml,
    dm_swimmer15_xml
)
from mojo_rl.envs.dm_control.walker.walker_xml import dm_walker_xml
from mojo_rl.envs.dm_control.manipulation_lift_box_xml import lift_large_box_xml
from mojo_rl.envs.dm_control.manipulation_place_cradle_xml import place_cradle_xml
from mojo_rl.envs.dm_control.manipulation_place_brick_xml import place_brick_xml
from mojo_rl.envs.dm_control.manipulation_lift_brick_xml import lift_brick_xml
from mojo_rl.envs.dm_control.manipulation_reassemble5_xml import reassemble5_xml
from mojo_rl.envs.dm_control.manipulation_reach_xml import reach_site_features_xml
from mojo_rl.envs.dm_control.manipulation_reach_duplo_xml import reach_duplo_xml
from mojo_rl.envs.dm_control.manipulation_stack_3_bricks_xml import stack_3_bricks_xml
from mojo_rl.envs.dm_control.manipulation_stack3r_xml import stack_3_random_xml
from mojo_rl.envs.dm_control.manipulation_stack_2_bricks_moveable_base_xml import stack_2_bricks_moveable_base_xml
from mojo_rl.envs.dm_control.manipulation_stack2_xml import stack_2_bricks_xml
from mojo_rl.envs.dm_control.dog.dog_xml import (
    dm_dog_stand_walk_xml,
    dm_dog_run_xml,
    dm_dog_trot_xml
)
from mojo_rl.envs.dm_control.dog.dog_fetch_xml import dm_dog_fetch_xml


# ⚠ NON-VACUITY FIXTURE. Every real model in the tree leaves `<pair>`,
# `ccd_tolerance` and `ccd_iterations` at their defaults, so those three rows
# compare a constant against itself over all 56. This one model gives each a
# second value: two `<pair>`s (one with an explicit condim), a non-default
# ccd tolerance and iteration count, a non-default noslip count, and an
# `<equality>` carrying BOTH a `<joint>` and a `<connect>` so the slab count
# is exercised by more than the manipulation welds.
comptime _FIXTURE_XML = """<mujoco model="dims_fixture">
  <option timestep="0.004" noslip_iterations="3" ccd_tolerance="1e-08"
          ccd_iterations="17"/>
  <worldbody>
    <light name="l0" pos="0 0 3"/>
    <camera name="c0" pos="0 -3 1"/>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body name="a" pos="0 0 1">
      <joint name="ja" type="hinge" axis="0 1 0"/>
      <geom name="ga" type="capsule" size=".05 .2" condim="6"/>
      <site name="sa" pos="0 0 .2" size=".01"/>
      <body name="b" pos="0 0 .4">
        <joint name="jb" type="hinge" axis="0 1 0"/>
        <geom name="gb" type="sphere" size=".06"/>
      </body>
    </body>
    <body name="c" pos="1 0 1">
      <joint name="jc" type="free"/>
      <geom name="gc" type="box" size=".05 .05 .05"/>
    </body>
  </worldbody>
  <contact>
    <pair geom1="ga" geom2="gc" condim="4"/>
    <pair geom1="gb" geom2="gc"/>
    <exclude body1="a" body2="c"/>
  </contact>
  <equality>
    <joint name="eqj" joint1="ja" joint2="jb" polycoef="0 1 0 0 0"/>
    <connect name="eqc" body1="c" body2="world" anchor="1 0 1"/>
  </equality>
</mujoco>"""


comptime NFIELD: Int = 20


def field_names() -> List[String]:
    return [
        "NBODY", "NJOINT", "NQ", "NV", "NGEOM", "NACT", "NTEX", "NMAT",
        "NLIGHT", "NCAM", "NSITE", "NEQ", "NEXCLUDE", "NPAIR", "NTENDON",
        "TIMESTEP", "NOSLIP_ITER", "CCD_TOL", "CCD_ITER", "MAX_CONDIM",
    ]


def _i(o: PythonObject) raises -> Int:
    return Int(py=o)


def _f(o: PythonObject) raises -> Float64:
    return Float64(py=o)


struct Tally(Copyable, Movable):
    var compared: Int
    var differing: Int
    var models: Int
    # Per-field record of what was actually observed, so the run can report
    # how many DISTINCT values each row saw. See the vacuity note in the
    # module docstring.
    var vals: List[List[Float64]]

    def __init__(out self):
        self.compared = 0
        self.differing = 0
        self.models = 0
        self.vals = List[List[Float64]]()
        for _ in range(NFIELD):
            self.vals.append(List[Float64]())


def _cmp_i(
    mut t: Tally, fi: Int, model: String, field: String, ours: Int, theirs: Int
) raises:
    t.compared += 1
    t.vals[fi].append(Float64(ours))
    if ours != theirs:
        t.differing += 1
        print("  DIFF", model, "." + field, ": ours=", ours, " mujoco=", theirs)


def _cmp_f(
    mut t: Tally, fi: Int, model: String, field: String, ours: Float64,
    theirs: Float64
) raises:
    t.compared += 1
    t.vals[fi].append(ours)
    if abs(ours - theirs) > 1e-12:
        t.differing += 1
        print("  DIFF", model, "." + field, ": ours=", ours, " mujoco=", theirs)


def check(mut t: Tally, name: String, xml: String) raises:
    """Compare one model's `parse_xml` dims against `mjModel`'s."""
    var mj = Python.import_module("mujoco")
    # ⚠ NOT wrapped in try/except. A model MuJoCo refuses is a failure of this
    # gate, not a row to skip -- a silent skip is how a corpus quietly shrinks.
    var m = mj.MjModel.from_xml_string(xml)

    t.models += 1
    var pm = parse_xml(xml)

    _cmp_i(t, 0, name, "NBODY", pm.NBODY, _i(m.nbody))
    _cmp_i(t, 1, name, "NJOINT", pm.NJOINT, _i(m.njnt))
    _cmp_i(t, 2, name, "NQ", pm.NQ, _i(m.nq))
    _cmp_i(t, 3, name, "NV", pm.NV, _i(m.nv))
    _cmp_i(t, 4, name, "NGEOM", pm.NGEOM, _i(m.ngeom))
    _cmp_i(t, 5, name, "NACT", pm.NACT, _i(m.nu))
    _cmp_i(t, 6, name, "NTEX", pm.NTEX, _i(m.ntex))
    _cmp_i(t, 7, name, "NMAT", pm.NMAT, _i(m.nmat))
    _cmp_i(t, 8, name, "NLIGHT", pm.NLIGHT, _i(m.nlight))
    _cmp_i(t, 9, name, "NCAM", pm.NCAM, _i(m.ncam))
    _cmp_i(t, 10, name, "NSITE", pm.NSITE, _i(m.nsite))

    # The equality SLAB only -- mjEQ_TENDON (3) rides on the tendon record.
    var neq_slab = 0
    for i in range(_i(m.neq)):
        if _i(m.eq_type[i]) != 3:
            neq_slab += 1
    _cmp_i(t, 11, name, "NEQ", pm.NEQ, neq_slab)

    _cmp_i(t, 12, name, "NEXCLUDE", pm.NEXCLUDE, _i(m.nexclude))
    _cmp_i(t, 13, name, "NPAIR", pm.NPAIR, _i(m.npair))
    _cmp_i(t, 14, name, "NTENDON", pm.NTENDON, _i(m.ntendon))
    _cmp_f(t, 15, name, "TIMESTEP", pm.TIMESTEP, _f(m.opt.timestep))
    _cmp_i(
        t, 16, name, "NOSLIP_ITER", pm.NOSLIP_ITER,
        _i(m.opt.noslip_iterations),
    )
    _cmp_f(t, 17, name, "CCD_TOL", pm.CCD_TOL, _f(m.opt.ccd_tolerance))
    _cmp_i(t, 18, name, "CCD_ITER", pm.CCD_ITER, _i(m.opt.ccd_iterations))

    # Ours is file-wide; MuJoCo keeps it per-geom and per-pair.
    var mx = 3
    for i in range(_i(m.ngeom)):
        var c = _i(m.geom_condim[i])
        if c > mx:
            mx = c
    for i in range(_i(m.npair)):
        var c = _i(m.pair_dim[i])
        if c > mx:
            mx = c
    _cmp_i(t, 19, name, "MAX_CONDIM", pm.MAX_CONDIM, mx)


def main() raises:
    var t = Tally()
    print("=== parse_xml dims vs MuJoCo — every composed model in the tree ===")

    check(t, "_FIXTURE", String(_FIXTURE_XML))
    check(t, "ant_xml", String(ant_xml))
    check(t, "half_cheetah_xml", String(half_cheetah_xml))
    check(t, "hopper_xml", String(hopper_xml))
    check(t, "humanoid_xml", String(humanoid_xml))
    check(t, "humanoid_standup_xml", String(humanoid_standup_xml))
    check(t, "inverted_double_pendulum_xml", String(inverted_double_pendulum_xml))
    check(t, "inverted_pendulum_xml", String(inverted_pendulum_xml))
    check(t, "pusher_xml", String(pusher_xml))
    check(t, "reacher_xml", String(reacher_xml))
    check(t, "swimmer_xml", String(swimmer_xml))
    check(t, "walker2d_xml", String(walker2d_xml))
    check(t, "sawyer_reach_xml", String(sawyer_reach_xml))
    check(t, "SO_ARM100_XML", String(SO_ARM100_XML))
    check(t, "SO_ARM101_XML", String(SO_ARM101_XML))
    check(t, "dm_acrobot_xml", String(dm_acrobot_xml))
    check(t, "dm_ball_in_cup_xml", String(dm_ball_in_cup_xml))
    check(t, "dm_cartpole1_xml", String(dm_cartpole1_xml))
    check(t, "dm_cartpole2_xml", String(dm_cartpole2_xml))
    check(t, "dm_cartpole3_xml", String(dm_cartpole3_xml))
    check(t, "dm_cheetah_xml", String(dm_cheetah_xml))
    check(t, "dm_finger_xml", String(dm_finger_xml))
    check(t, "dm_finger_spin_xml", String(dm_finger_spin_xml))
    check(t, "dm_fish_xml", String(dm_fish_xml))
    check(t, "dm_hopper_xml", String(dm_hopper_xml))
    check(t, "dm_humanoid_xml", String(dm_humanoid_xml))
    check(t, "dm_humanoid_cmu_xml", String(dm_humanoid_cmu_xml))
    check(t, "dm_manipulator_bring_ball_xml", String(dm_manipulator_bring_ball_xml))
    check(t, "dm_manipulator_bring_peg_xml", String(dm_manipulator_bring_peg_xml))
    check(t, "dm_manipulator_insert_ball_xml", String(dm_manipulator_insert_ball_xml))
    check(t, "dm_manipulator_insert_peg_xml", String(dm_manipulator_insert_peg_xml))
    check(t, "dm_pendulum_xml", String(dm_pendulum_xml))
    check(t, "dm_point_mass_xml", String(dm_point_mass_xml))
    check(t, "dm_quadruped_walk_xml", String(dm_quadruped_walk_xml))
    check(t, "dm_quadruped_run_xml", String(dm_quadruped_run_xml))
    check(t, "dm_quadruped_fetch_xml", String(dm_quadruped_fetch_xml))
    check(t, "dm_reacher_xml", String(dm_reacher_xml))
    check(t, "dm_stacker_2_xml", String(dm_stacker_2_xml))
    check(t, "dm_stacker_4_xml", String(dm_stacker_4_xml))
    check(t, "dm_swimmer6_xml", String(dm_swimmer6_xml))
    check(t, "dm_swimmer15_xml", String(dm_swimmer15_xml))
    check(t, "dm_walker_xml", String(dm_walker_xml))
    check(t, "lift_large_box_xml", String(lift_large_box_xml))
    check(t, "place_cradle_xml", String(place_cradle_xml))
    check(t, "place_brick_xml", String(place_brick_xml))
    check(t, "lift_brick_xml", String(lift_brick_xml))
    check(t, "reassemble5_xml", String(reassemble5_xml))
    check(t, "reach_site_features_xml", String(reach_site_features_xml))
    check(t, "reach_duplo_xml", String(reach_duplo_xml))
    check(t, "stack_3_bricks_xml", String(stack_3_bricks_xml))
    check(t, "stack_3_random_xml", String(stack_3_random_xml))
    check(t, "stack_2_bricks_moveable_base_xml", String(stack_2_bricks_moveable_base_xml))
    check(t, "stack_2_bricks_xml", String(stack_2_bricks_xml))
    check(t, "dm_dog_stand_walk_xml", String(dm_dog_stand_walk_xml))
    check(t, "dm_dog_run_xml", String(dm_dog_run_xml))
    check(t, "dm_dog_trot_xml", String(dm_dog_trot_xml))
    check(t, "dm_dog_fetch_xml", String(dm_dog_fetch_xml))

    print()
    print("models compared:", t.models)
    print("rows compared  :", t.compared)
    print("rows differing :", t.differing)

    print()
    print("--- distinct values observed per field (vacuity guard) ---")
    var vacuous = 0
    var fnames = field_names()
    for fi in range(NFIELD):
        var seen = List[Float64]()
        for v in t.vals[fi]:
            var dup = False
            for w in seen:
                if w == v:
                    dup = True
                    break
            if not dup:
                seen.append(v)
        var mark = String("")
        if len(seen) < 2:
            mark = "   <-- VACUOUS: one value across every model"
            vacuous += 1
        print("  ", fnames[fi], ":", len(seen), "distinct", mark)

    print()
    print("fields carrying no information:", vacuous, "of", NFIELD)

    assert_true(t.models == 57, "expected 57 models, got " + String(t.models))
    assert_true(
        t.compared == 57 * NFIELD,
        "expected " + String(57 * NFIELD) + " rows, got " + String(t.compared),
    )
    assert_true(
        vacuous == 0,
        String(vacuous)
        + " field(s) took a single value across every model — those rows"
        " compare a default against itself and prove nothing. Add a fixture.",
    )
    assert_true(
        t.differing == 0,
        String(t.differing) + " dim(s) disagree with MuJoCo — see DIFF above",
    )
    print()
    print("PASS")
