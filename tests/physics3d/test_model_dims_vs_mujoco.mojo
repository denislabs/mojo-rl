"""Model dimensions: `parse_xml` AND the generated constants, both vs MuJoCo.

⚠ THREE-WAY ON PURPOSE. Phase 1b replaces the comptime dim scan with constants
generated from `mjModel` (`tools/gen_model_dims.py`), because `parse_xml` runs
in the comptime interpreter and the interpreter cannot `open()` a file — which
is the only reason the MJCF has to be embedded in Mojo source at all (§10.2).
Two things therefore need checking, and they are NOT the same check:

  ours  = parse_xml(the embedded string)   -- the scanner that still ships
  gen   = the generated ParsedModel        -- what models are BUILT from
  both compared against mjModel, the authority on what a model's counts ARE
  (`feedback_count_model_elements_with_mujoco_not_grep`: the dog's njnt is 75
  and no amount of tag-counting says so).

Comparing `gen` against `ours` instead would be the blind kind of gate — a
generator agreeing with the thing it replaces proves nothing about either, and
this tree has already shipped two parsers that agreed perfectly on a wrong
default for months (`feedback_a_gate_that_shares_its_reference_implementation
_is_blind`). MuJoCo is the third party. `tools/gen_model_dims.py --check`
covers the other failure mode — a generated file that is merely STALE — which
this gate cannot see, because it reads the committed constants.

WHAT IS COMPARED — 20 fields × 57 models × 2 sources. Every composed MJCF in
the tree: gym MuJoCo, dm_control suite + manipulation, dog, SO-ARM, Sawyer.

  counts     NBODY NJOINT NQ NV NGEOM NACT NTEX NMAT NLIGHT NCAM NSITE
             NEQ NEXCLUDE NPAIR NTENDON
  <option>   TIMESTEP NOSLIP_ITER CCD_TOL CCD_ITER
  derived    MAX_CONDIM

⚠ TWO ROWS ARE NOT RAW MuJoCo FIELDS, and both would be wrong if they were:

  * NEQ is the weld/connect/joint EQUALITY SLAB. `<equality><tendon>`
    (mjEQ_TENDON) is deliberately excluded — it rides on the tendon record via
    `TENDON_IDX_IS_EQUALITY`, not in that slab (`xml_parser.mojo:2407`).
    Comparing raw `m.neq` reports 9 differences across quadruped, manipulator
    and stacker that are a difference of DEFINITION, not of counting.
  * MAX_CONDIM is ours file-wide; MuJoCo keeps condim per-geom and per-pair,
    so the comparison is the max over both of those arrays.

⚠ THREE FIELDS WERE VACUOUS over the 56 real models — `NPAIR`, `CCD_TOL` and
`CCD_ITER` take exactly ONE value each across all of them (0, 1e-6, 35: the
defaults). Rows like that compare a default against itself and would pass with
the parser deleted. `_FIXTURE_XML` exists solely to give them a second value.
The run prints the distinct-value count per field and FAILS on a vacuous one.
See `feedback_the_sweep_was_not_the_distribution`.

⚠ The fixture has no generated twin — it is not a model in the tree — so its
`gen` column is skipped and the row counts below account for that.

Run: pixi run mojo run -I . tests/physics3d/test_model_dims_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true

from mojo_rl.physics3d.parser import parse_xml
from mojo_rl.physics3d.parser.xml_parser import ParsedModel

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
from mojo_rl.envs.dm_control.reacher.reacher_xml import (
    dm_reacher_xml,
    dm_reacher_hard_xml,
)
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
from mojo_rl.envs.ant.ant_dims import ANT_DIMS
from mojo_rl.envs.half_cheetah.half_cheetah_dims import HALF_CHEETAH_DIMS
from mojo_rl.envs.hopper.hopper_dims import HOPPER_DIMS
from mojo_rl.envs.humanoid.humanoid_dims import HUMANOID_DIMS
from mojo_rl.envs.humanoid_standup.humanoid_standup_dims import HUMANOID_STANDUP_DIMS
from mojo_rl.envs.inverted_double_pendulum.inverted_double_pendulum_dims import INVERTED_DOUBLE_PENDULUM_DIMS
from mojo_rl.envs.inverted_pendulum.inverted_pendulum_dims import INVERTED_PENDULUM_DIMS
from mojo_rl.envs.pusher.pusher_dims import PUSHER_DIMS
from mojo_rl.envs.reacher.reacher_dims import REACHER_DIMS
from mojo_rl.envs.swimmer.swimmer_dims import SWIMMER_DIMS
from mojo_rl.envs.walker2d.walker2d_dims import WALKER2D_DIMS
from mojo_rl.envs.metaworld.sawyer_reach_dims import SAWYER_REACH_DIMS
from mojo_rl.envs.robots.so_arm100_dims import SO_ARM100_DIMS
from mojo_rl.envs.robots.so_arm101_dims import SO_ARM101_DIMS
from mojo_rl.envs.dm_control.acrobot.acrobot_dims import DM_ACROBOT_DIMS
from mojo_rl.envs.dm_control.ball_in_cup.ball_in_cup_dims import DM_BALL_IN_CUP_DIMS
from mojo_rl.envs.dm_control.cartpole.cartpole_dims import (
    DM_CARTPOLE1_DIMS,
    DM_CARTPOLE2_DIMS,
    DM_CARTPOLE3_DIMS
)
from mojo_rl.envs.dm_control.cheetah.cheetah_dims import DM_CHEETAH_DIMS
from mojo_rl.envs.dm_control.finger.finger_dims import (
    DM_FINGER_DIMS,
    DM_FINGER_SPIN_DIMS
)
from mojo_rl.envs.dm_control.fish.fish_dims import DM_FISH_DIMS
from mojo_rl.envs.dm_control.hopper.hopper_dims import DM_HOPPER_DIMS
from mojo_rl.envs.dm_control.humanoid.humanoid_dims import DM_HUMANOID_DIMS
from mojo_rl.envs.dm_control.humanoid_cmu.humanoid_cmu_dims import DM_HUMANOID_CMU_DIMS
from mojo_rl.envs.dm_control.manipulator.manipulator_dims import (
    DM_MANIPULATOR_BRING_BALL_DIMS,
    DM_MANIPULATOR_BRING_PEG_DIMS,
    DM_MANIPULATOR_INSERT_BALL_DIMS,
    DM_MANIPULATOR_INSERT_PEG_DIMS
)
from mojo_rl.envs.dm_control.pendulum.pendulum_dims import DM_PENDULUM_DIMS
from mojo_rl.envs.dm_control.point_mass.point_mass_dims import DM_POINT_MASS_DIMS
from mojo_rl.envs.dm_control.quadruped.quadruped_dims import (
    DM_QUADRUPED_WALK_DIMS,
    DM_QUADRUPED_RUN_DIMS,
    DM_QUADRUPED_FETCH_DIMS
)
from mojo_rl.envs.dm_control.reacher.reacher_dims import (
    DM_REACHER_DIMS,
    DM_REACHER_HARD_DIMS,
)
from mojo_rl.envs.dm_control.stacker.stacker_dims import (
    DM_STACKER_2_DIMS,
    DM_STACKER_4_DIMS
)
from mojo_rl.envs.dm_control.swimmer.swimmer_dims import (
    DM_SWIMMER6_DIMS,
    DM_SWIMMER15_DIMS
)
from mojo_rl.envs.dm_control.walker.walker_dims import DM_WALKER_DIMS
from mojo_rl.envs.dm_control.manipulation_lift_box_dims import LIFT_LARGE_BOX_DIMS
from mojo_rl.envs.dm_control.manipulation_place_cradle_dims import PLACE_CRADLE_DIMS
from mojo_rl.envs.dm_control.manipulation_place_brick_dims import PLACE_BRICK_DIMS
from mojo_rl.envs.dm_control.manipulation_lift_brick_dims import LIFT_BRICK_DIMS
from mojo_rl.envs.dm_control.manipulation_reassemble5_dims import REASSEMBLE5_DIMS
from mojo_rl.envs.dm_control.manipulation_reach_dims import REACH_SITE_FEATURES_DIMS
from mojo_rl.envs.dm_control.manipulation_reach_duplo_dims import REACH_DUPLO_DIMS
from mojo_rl.envs.dm_control.manipulation_stack_3_bricks_dims import STACK_3_BRICKS_DIMS
from mojo_rl.envs.dm_control.manipulation_stack3r_dims import STACK_3_RANDOM_DIMS
from mojo_rl.envs.dm_control.manipulation_stack_2_bricks_moveable_base_dims import STACK_2_BRICKS_MOVEABLE_BASE_DIMS
from mojo_rl.envs.dm_control.manipulation_stack2_dims import STACK_2_BRICKS_DIMS
from mojo_rl.envs.dm_control.dog.dog_dims import (
    DM_DOG_STAND_WALK_DIMS,
    DM_DOG_RUN_DIMS,
    DM_DOG_TROT_DIMS
)
from mojo_rl.envs.dm_control.dog.dog_fetch_dims import DM_DOG_FETCH_DIMS


# ⚠ NON-VACUITY FIXTURE. Every real model in the tree leaves `<pair>`,
# `ccd_tolerance` and `ccd_iterations` at their defaults, so those three rows
# compare a constant against itself over all 56. This one model gives each a
# second value: two `<pair>`s (one with an explicit condim), a non-default ccd
# tolerance and iteration count, a non-default noslip count, and an
# `<equality>` carrying BOTH a `<joint>` and a `<connect>` so the slab count is
# exercised by more than the manipulation welds.
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
    var generated: Int
    # Per-field record of what was actually observed, so the run can report
    # how many DISTINCT values each row saw. See the vacuity note above.
    var vals: List[List[Float64]]

    def __init__(out self):
        self.compared = 0
        self.differing = 0
        self.models = 0
        self.generated = 0
        self.vals = List[List[Float64]]()
        for _ in range(NFIELD):
            self.vals.append(List[Float64]())


def _cmp_i(
    mut t: Tally, fi: Int, src: String, model: String, ours: Int, theirs: Int
) raises:
    t.compared += 1
    t.vals[fi].append(Float64(ours))
    if ours != theirs:
        t.differing += 1
        print(
            "  DIFF", model, "." + field_names()[fi], "[" + src + "]",
            ": ours=", ours, " mujoco=", theirs,
        )


def _cmp_f(
    mut t: Tally, fi: Int, src: String, model: String, ours: Float64,
    theirs: Float64
) raises:
    t.compared += 1
    t.vals[fi].append(ours)
    if abs(ours - theirs) > 1e-12:
        t.differing += 1
        print(
            "  DIFF", model, "." + field_names()[fi], "[" + src + "]",
            ": ours=", ours, " mujoco=", theirs,
        )


def _against_mujoco(
    mut t: Tally, src: String, name: String, pm: ParsedModel, m: PythonObject
) raises:
    """One ParsedModel — from either source — against `mjModel`."""
    _cmp_i(t, 0, src, name, pm.NBODY, _i(m.nbody))
    _cmp_i(t, 1, src, name, pm.NJOINT, _i(m.njnt))
    _cmp_i(t, 2, src, name, pm.NQ, _i(m.nq))
    _cmp_i(t, 3, src, name, pm.NV, _i(m.nv))
    _cmp_i(t, 4, src, name, pm.NGEOM, _i(m.ngeom))
    _cmp_i(t, 5, src, name, pm.NACT, _i(m.nu))
    _cmp_i(t, 6, src, name, pm.NTEX, _i(m.ntex))
    _cmp_i(t, 7, src, name, pm.NMAT, _i(m.nmat))
    _cmp_i(t, 8, src, name, pm.NLIGHT, _i(m.nlight))
    _cmp_i(t, 9, src, name, pm.NCAM, _i(m.ncam))
    _cmp_i(t, 10, src, name, pm.NSITE, _i(m.nsite))

    # The equality SLAB only — mjEQ_TENDON (3) rides on the tendon record.
    var neq_slab = 0
    for i in range(_i(m.neq)):
        if _i(m.eq_type[i]) != 3:
            neq_slab += 1
    _cmp_i(t, 11, src, name, pm.NEQ, neq_slab)

    _cmp_i(t, 12, src, name, pm.NEXCLUDE, _i(m.nexclude))
    _cmp_i(t, 13, src, name, pm.NPAIR, _i(m.npair))
    _cmp_i(t, 14, src, name, pm.NTENDON, _i(m.ntendon))
    _cmp_f(t, 15, src, name, pm.TIMESTEP, _f(m.opt.timestep))
    _cmp_i(
        t, 16, src, name, pm.NOSLIP_ITER, _i(m.opt.noslip_iterations)
    )
    _cmp_f(t, 17, src, name, pm.CCD_TOL, _f(m.opt.ccd_tolerance))
    _cmp_i(t, 18, src, name, pm.CCD_ITER, _i(m.opt.ccd_iterations))

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
    _cmp_i(t, 19, src, name, pm.MAX_CONDIM, mx)


def check(
    mut t: Tally, name: String, path: String, xml: String, gen: ParsedModel
) raises:
    """One model: both the scanned and the generated dims, vs `mjModel`.

    ⚠ MuJoCo loads BY PATH, our scanner from the STRING, and that asymmetry is
    deliberate. Asset paths inside a model are relative to the MODEL FILE
    (§10.5 decision 1), so `from_xml_string` cannot resolve sawyer's meshes —
    it would look beside the CWD. The two sources are byte-identical, which is
    what `test_xml_assets_match_source` exists to guarantee.
    """
    var mj = Python.import_module("mujoco")
    # ⚠ NOT wrapped in try/except. A model MuJoCo refuses is a failure of this
    # gate, not a row to skip — a silent skip is how a corpus quietly shrinks.
    var m = mj.MjModel.from_xml_path(path)

    t.models += 1
    t.generated += 1
    _against_mujoco(t, "scan", name, parse_xml(xml), m)
    _against_mujoco(t, "gen", name, gen, m)


def check_scan_only(mut t: Tally, name: String, xml: String) raises:
    """The fixture, which has no generated twin — it is not a tree model.

    Loaded from the STRING because it has no file and cites no assets.
    """
    var mj = Python.import_module("mujoco")
    var m = mj.MjModel.from_xml_string(xml)
    t.models += 1
    _against_mujoco(t, "scan", name, parse_xml(xml), m)


def main() raises:
    var t = Tally()
    print("=== model dims: scanned + generated, both vs MuJoCo ===")

    check_scan_only(t, "_FIXTURE", String(_FIXTURE_XML))
    check(t, "ant_xml", "mojo_rl/envs/ant/assets/ant.xml", String(ant_xml), materialize[ANT_DIMS]())
    check(t, "half_cheetah_xml", "mojo_rl/envs/half_cheetah/assets/half_cheetah.xml", String(half_cheetah_xml), materialize[HALF_CHEETAH_DIMS]())
    check(t, "hopper_xml", "mojo_rl/envs/hopper/assets/hopper.xml", String(hopper_xml), materialize[HOPPER_DIMS]())
    check(t, "humanoid_xml", "mojo_rl/envs/humanoid/assets/humanoid.xml", String(humanoid_xml), materialize[HUMANOID_DIMS]())
    check(t, "humanoid_standup_xml", "mojo_rl/envs/humanoid_standup/assets/humanoid_standup.xml", String(humanoid_standup_xml), materialize[HUMANOID_STANDUP_DIMS]())
    check(t, "inverted_double_pendulum_xml", "mojo_rl/envs/inverted_double_pendulum/assets/inverted_double_pendulum.xml", String(inverted_double_pendulum_xml), materialize[INVERTED_DOUBLE_PENDULUM_DIMS]())
    check(t, "inverted_pendulum_xml", "mojo_rl/envs/inverted_pendulum/assets/inverted_pendulum.xml", String(inverted_pendulum_xml), materialize[INVERTED_PENDULUM_DIMS]())
    check(t, "pusher_xml", "mojo_rl/envs/pusher/assets/pusher.xml", String(pusher_xml), materialize[PUSHER_DIMS]())
    check(t, "reacher_xml", "mojo_rl/envs/reacher/assets/reacher.xml", String(reacher_xml), materialize[REACHER_DIMS]())
    check(t, "swimmer_xml", "mojo_rl/envs/swimmer/assets/swimmer.xml", String(swimmer_xml), materialize[SWIMMER_DIMS]())
    check(t, "walker2d_xml", "mojo_rl/envs/walker2d/assets/walker2d.xml", String(walker2d_xml), materialize[WALKER2D_DIMS]())
    check(t, "sawyer_reach_xml", "mojo_rl/envs/metaworld/assets/sawyer_reach.xml", String(sawyer_reach_xml), materialize[SAWYER_REACH_DIMS]())
    check(t, "SO_ARM100_XML", "mojo_rl/envs/robots/assets/so_arm100.xml", String(SO_ARM100_XML), materialize[SO_ARM100_DIMS]())
    check(t, "SO_ARM101_XML", "mojo_rl/envs/robots/assets/so_arm101.xml", String(SO_ARM101_XML), materialize[SO_ARM101_DIMS]())
    check(t, "dm_acrobot_xml", "mojo_rl/envs/dm_control/assets/acrobot.xml", String(dm_acrobot_xml), materialize[DM_ACROBOT_DIMS]())
    check(t, "dm_ball_in_cup_xml", "mojo_rl/envs/dm_control/assets/ball_in_cup.xml", String(dm_ball_in_cup_xml), materialize[DM_BALL_IN_CUP_DIMS]())
    check(t, "dm_cartpole1_xml", "mojo_rl/envs/dm_control/assets/cartpole1.xml", String(dm_cartpole1_xml), materialize[DM_CARTPOLE1_DIMS]())
    check(t, "dm_cartpole2_xml", "mojo_rl/envs/dm_control/assets/cartpole2.xml", String(dm_cartpole2_xml), materialize[DM_CARTPOLE2_DIMS]())
    check(t, "dm_cartpole3_xml", "mojo_rl/envs/dm_control/assets/cartpole3.xml", String(dm_cartpole3_xml), materialize[DM_CARTPOLE3_DIMS]())
    check(t, "dm_cheetah_xml", "mojo_rl/envs/dm_control/assets/cheetah.xml", String(dm_cheetah_xml), materialize[DM_CHEETAH_DIMS]())
    check(t, "dm_finger_xml", "mojo_rl/envs/dm_control/assets/finger.xml", String(dm_finger_xml), materialize[DM_FINGER_DIMS]())
    check(t, "dm_finger_spin_xml", "mojo_rl/envs/dm_control/assets/finger_spin.xml", String(dm_finger_spin_xml), materialize[DM_FINGER_SPIN_DIMS]())
    check(t, "dm_fish_xml", "mojo_rl/envs/dm_control/assets/fish.xml", String(dm_fish_xml), materialize[DM_FISH_DIMS]())
    check(t, "dm_hopper_xml", "mojo_rl/envs/dm_control/assets/hopper.xml", String(dm_hopper_xml), materialize[DM_HOPPER_DIMS]())
    check(t, "dm_humanoid_xml", "mojo_rl/envs/dm_control/assets/humanoid.xml", String(dm_humanoid_xml), materialize[DM_HUMANOID_DIMS]())
    check(t, "dm_humanoid_cmu_xml", "mojo_rl/envs/dm_control/assets/humanoid_cmu.xml", String(dm_humanoid_cmu_xml), materialize[DM_HUMANOID_CMU_DIMS]())
    check(t, "dm_manipulator_bring_ball_xml", "mojo_rl/envs/dm_control/assets/manipulator_bring_ball.xml", String(dm_manipulator_bring_ball_xml), materialize[DM_MANIPULATOR_BRING_BALL_DIMS]())
    check(t, "dm_manipulator_bring_peg_xml", "mojo_rl/envs/dm_control/assets/manipulator_bring_peg.xml", String(dm_manipulator_bring_peg_xml), materialize[DM_MANIPULATOR_BRING_PEG_DIMS]())
    check(t, "dm_manipulator_insert_ball_xml", "mojo_rl/envs/dm_control/assets/manipulator_insert_ball.xml", String(dm_manipulator_insert_ball_xml), materialize[DM_MANIPULATOR_INSERT_BALL_DIMS]())
    check(t, "dm_manipulator_insert_peg_xml", "mojo_rl/envs/dm_control/assets/manipulator_insert_peg.xml", String(dm_manipulator_insert_peg_xml), materialize[DM_MANIPULATOR_INSERT_PEG_DIMS]())
    check(t, "dm_pendulum_xml", "mojo_rl/envs/dm_control/assets/pendulum.xml", String(dm_pendulum_xml), materialize[DM_PENDULUM_DIMS]())
    check(t, "dm_point_mass_xml", "mojo_rl/envs/dm_control/assets/point_mass.xml", String(dm_point_mass_xml), materialize[DM_POINT_MASS_DIMS]())
    check(t, "dm_quadruped_walk_xml", "mojo_rl/envs/dm_control/assets/quadruped_walk.xml", String(dm_quadruped_walk_xml), materialize[DM_QUADRUPED_WALK_DIMS]())
    check(t, "dm_quadruped_run_xml", "mojo_rl/envs/dm_control/assets/quadruped_run.xml", String(dm_quadruped_run_xml), materialize[DM_QUADRUPED_RUN_DIMS]())
    check(t, "dm_quadruped_fetch_xml", "mojo_rl/envs/dm_control/assets/quadruped_fetch.xml", String(dm_quadruped_fetch_xml), materialize[DM_QUADRUPED_FETCH_DIMS]())
    check(t, "dm_reacher_xml", "mojo_rl/envs/dm_control/assets/reacher.xml", String(dm_reacher_xml), materialize[DM_REACHER_DIMS]())
    check(t, "dm_reacher_hard_xml", "mojo_rl/envs/dm_control/assets/reacher_hard.xml", String(dm_reacher_hard_xml), materialize[DM_REACHER_HARD_DIMS]())
    check(t, "dm_stacker_2_xml", "mojo_rl/envs/dm_control/assets/stacker_2.xml", String(dm_stacker_2_xml), materialize[DM_STACKER_2_DIMS]())
    check(t, "dm_stacker_4_xml", "mojo_rl/envs/dm_control/assets/stacker_4.xml", String(dm_stacker_4_xml), materialize[DM_STACKER_4_DIMS]())
    check(t, "dm_swimmer6_xml", "mojo_rl/envs/dm_control/assets/swimmer6.xml", String(dm_swimmer6_xml), materialize[DM_SWIMMER6_DIMS]())
    check(t, "dm_swimmer15_xml", "mojo_rl/envs/dm_control/assets/swimmer15.xml", String(dm_swimmer15_xml), materialize[DM_SWIMMER15_DIMS]())
    check(t, "dm_walker_xml", "mojo_rl/envs/dm_control/assets/walker.xml", String(dm_walker_xml), materialize[DM_WALKER_DIMS]())
    check(t, "lift_large_box_xml", "mojo_rl/envs/dm_control/assets/manipulation/lift_large_box.xml", String(lift_large_box_xml), materialize[LIFT_LARGE_BOX_DIMS]())
    check(t, "place_cradle_xml", "mojo_rl/envs/dm_control/assets/manipulation/place_cradle.xml", String(place_cradle_xml), materialize[PLACE_CRADLE_DIMS]())
    check(t, "place_brick_xml", "mojo_rl/envs/dm_control/assets/manipulation/place_brick.xml", String(place_brick_xml), materialize[PLACE_BRICK_DIMS]())
    check(t, "lift_brick_xml", "mojo_rl/envs/dm_control/assets/manipulation/lift_brick.xml", String(lift_brick_xml), materialize[LIFT_BRICK_DIMS]())
    check(t, "reassemble5_xml", "mojo_rl/envs/dm_control/assets/manipulation/reassemble5.xml", String(reassemble5_xml), materialize[REASSEMBLE5_DIMS]())
    check(t, "reach_site_features_xml", "mojo_rl/envs/dm_control/assets/manipulation/reach_site_features.xml", String(reach_site_features_xml), materialize[REACH_SITE_FEATURES_DIMS]())
    check(t, "reach_duplo_xml", "mojo_rl/envs/dm_control/assets/manipulation/reach_duplo.xml", String(reach_duplo_xml), materialize[REACH_DUPLO_DIMS]())
    check(t, "stack_3_bricks_xml", "mojo_rl/envs/dm_control/assets/manipulation/stack_3_bricks.xml", String(stack_3_bricks_xml), materialize[STACK_3_BRICKS_DIMS]())
    check(t, "stack_3_random_xml", "mojo_rl/envs/dm_control/assets/manipulation/stack_3_random.xml", String(stack_3_random_xml), materialize[STACK_3_RANDOM_DIMS]())
    check(t, "stack_2_bricks_moveable_base_xml", "mojo_rl/envs/dm_control/assets/manipulation/stack_2_bricks_moveable_base.xml", String(stack_2_bricks_moveable_base_xml), materialize[STACK_2_BRICKS_MOVEABLE_BASE_DIMS]())
    check(t, "stack_2_bricks_xml", "mojo_rl/envs/dm_control/assets/manipulation/stack_2_bricks.xml", String(stack_2_bricks_xml), materialize[STACK_2_BRICKS_DIMS]())
    check(t, "dm_dog_stand_walk_xml", "mojo_rl/envs/dm_control/assets/dog_stand_walk.xml", String(dm_dog_stand_walk_xml), materialize[DM_DOG_STAND_WALK_DIMS]())
    check(t, "dm_dog_run_xml", "mojo_rl/envs/dm_control/assets/dog_run.xml", String(dm_dog_run_xml), materialize[DM_DOG_RUN_DIMS]())
    check(t, "dm_dog_trot_xml", "mojo_rl/envs/dm_control/assets/dog_trot.xml", String(dm_dog_trot_xml), materialize[DM_DOG_TROT_DIMS]())
    check(t, "dm_dog_fetch_xml", "mojo_rl/envs/dm_control/assets/dog_fetch.xml", String(dm_dog_fetch_xml), materialize[DM_DOG_FETCH_DIMS]())

    print()
    print("models compared :", t.models, "(of which generated:", t.generated, ")")
    print("rows compared   :", t.compared)
    print("rows differing  :", t.differing)

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

    # 57 models, 56 of which contribute a second (generated) source.
    assert_true(t.models == 58, "expected 58 models, got " + String(t.models))
    assert_true(
        t.generated == 57,
        "expected 57 generated dim sets, got " + String(t.generated)
        + " — a model lost its *_dims.mojo",
    )
    assert_true(
        t.compared == (58 + 57) * NFIELD,
        "expected " + String((58 + 57) * NFIELD) + " rows, got "
        + String(t.compared),
    )
    assert_true(
        vacuous == 0,
        String(vacuous)
        + " field(s) took a single value across every model — those rows"
        " compare a default against itself and prove nothing. Add a fixture.",
    )
    assert_true(
        t.differing == 0,
        String(t.differing) + " dim(s) disagree with MuJoCo — see DIFF above."
        " A [gen] row means tools/gen_model_dims.py is wrong or stale; a"
        " [scan] row means parse_xml is.",
    )
    print()
    print("PASS")
