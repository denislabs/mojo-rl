"""The env's scalar action bounds survived leaving comptime, unchanged.

⚠ THIS GATE EXISTS TO PROVE A NON-CHANGE. `ModelDefLike.CTRL_MIN/CTRL_MAX`
were comptime `Float64` members computed by `_xml_default_motor_ctrlrange`
scanning the raw MJCF in the comptime interpreter — the last such scan inside
`ModelDefFromXML`, and a comptime reader of the XML is exactly what pins a
model to a `String` in Mojo source (§10.2). Phase 1b moved them onto
`FlatModelDef` and into `Model.meta`.

⚠⚠ THE VALUES ARE KNOWN TO BE WRONG ON SOME MODELS AND MUST STAY WRONG. They
are a model-wide SUMMARY read from a ROOT `<default><motor ctrlrange>`, not
the clamp: `apply_actions` clamps each actuator to its own range, while this
pair only sizes the box a policy is told to sample from. Measured against
dm_control's `action_spec`, `reach_site_features` advertises (-1, 1) where the
real bounds are ±0.6283, ±0.8378 and ±5.0. Correcting that changes the action
scaling of every shipped env and is owed its own before/after measurement —
it must not ride along on a refactor. So this gate asserts SAMENESS, not
correctness. `test_per_actuator_action_bounds` is where the wrongness is
measured on purpose.

⚠ THE TWO READERS ARE NOT THE SAME ALGORITHM, which is why this is worth
running rather than assuming. `_xml_default_motor_ctrlrange` looks only for
`<motor>` in the root defaults; `full_parser._parse_one_default_block` also
accepts `<general>`, `<position>` and `<velocity>`. A model that puts its root
ctrlrange on a `<general>` tag WOULD get a different answer from the two — no
model in the tree does today, and this gate is what would say so if one
arrived.

⚠ NON-VACUITY: 52 of the 56 models simply take the (-1, 1) fallback, so most
rows compare a default against itself. The run counts models carrying a
NON-default pair and fails if that count reaches zero.

Run: pixi run mojo run -I . tests/physics3d/test_ctrl_range_source.mojo
"""

from std.math import abs
from std.testing import assert_true

from mojo_rl.physics3d.parser import parse_xml_full
from mojo_rl.physics3d.parser.xml_parser import _xml_default_motor_ctrlrange

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


struct Tally(Copyable, Movable):
    var models: Int
    var bad: Int
    var nondefault: Int

    def __init__(out self):
        self.models = 0
        self.bad = 0
        self.nondefault = 0


def check[xml: String](mut t: Tally, name: String) raises:
    """One model: the runtime record against the comptime scan it replaced."""
    var fmd = parse_xml_full(String(xml))
    comptime c = _xml_default_motor_ctrlrange[xml]()

    t.models += 1
    if abs(fmd.default_motor_ctrl_min - c[0]) > 1e-12 or abs(
        fmd.default_motor_ctrl_max - c[1]
    ) > 1e-12:
        t.bad += 1
        print(
            "  DIFF", name, ": runtime=(", fmd.default_motor_ctrl_min, ",",
            fmd.default_motor_ctrl_max, ")  comptime=(", c[0], ",", c[1], ")",
        )

    # A pair that is not the (-1, 1) fallback is a row that tested something.
    if abs(c[0] + 1.0) > 1e-12 or abs(c[1] - 1.0) > 1e-12:
        t.nondefault += 1


def main() raises:
    var t = Tally()
    print("=== root-default motor ctrlrange: runtime record vs comptime ===")
    check[ant_xml](t, "ant_xml")
    check[half_cheetah_xml](t, "half_cheetah_xml")
    check[hopper_xml](t, "hopper_xml")
    check[humanoid_xml](t, "humanoid_xml")
    check[humanoid_standup_xml](t, "humanoid_standup_xml")
    check[inverted_double_pendulum_xml](t, "inverted_double_pendulum_xml")
    check[inverted_pendulum_xml](t, "inverted_pendulum_xml")
    check[pusher_xml](t, "pusher_xml")
    check[reacher_xml](t, "reacher_xml")
    check[swimmer_xml](t, "swimmer_xml")
    check[walker2d_xml](t, "walker2d_xml")
    check[sawyer_reach_xml](t, "sawyer_reach_xml")
    check[SO_ARM100_XML](t, "SO_ARM100_XML")
    check[SO_ARM101_XML](t, "SO_ARM101_XML")
    check[dm_acrobot_xml](t, "dm_acrobot_xml")
    check[dm_ball_in_cup_xml](t, "dm_ball_in_cup_xml")
    check[dm_cartpole1_xml](t, "dm_cartpole1_xml")
    check[dm_cartpole2_xml](t, "dm_cartpole2_xml")
    check[dm_cartpole3_xml](t, "dm_cartpole3_xml")
    check[dm_cheetah_xml](t, "dm_cheetah_xml")
    check[dm_finger_xml](t, "dm_finger_xml")
    check[dm_finger_spin_xml](t, "dm_finger_spin_xml")
    check[dm_fish_xml](t, "dm_fish_xml")
    check[dm_hopper_xml](t, "dm_hopper_xml")
    check[dm_humanoid_xml](t, "dm_humanoid_xml")
    check[dm_humanoid_cmu_xml](t, "dm_humanoid_cmu_xml")
    check[dm_manipulator_bring_ball_xml](t, "dm_manipulator_bring_ball_xml")
    check[dm_manipulator_bring_peg_xml](t, "dm_manipulator_bring_peg_xml")
    check[dm_manipulator_insert_ball_xml](t, "dm_manipulator_insert_ball_xml")
    check[dm_manipulator_insert_peg_xml](t, "dm_manipulator_insert_peg_xml")
    check[dm_pendulum_xml](t, "dm_pendulum_xml")
    check[dm_point_mass_xml](t, "dm_point_mass_xml")
    check[dm_quadruped_walk_xml](t, "dm_quadruped_walk_xml")
    check[dm_quadruped_run_xml](t, "dm_quadruped_run_xml")
    check[dm_quadruped_fetch_xml](t, "dm_quadruped_fetch_xml")
    check[dm_reacher_xml](t, "dm_reacher_xml")
    check[dm_stacker_2_xml](t, "dm_stacker_2_xml")
    check[dm_stacker_4_xml](t, "dm_stacker_4_xml")
    check[dm_swimmer6_xml](t, "dm_swimmer6_xml")
    check[dm_swimmer15_xml](t, "dm_swimmer15_xml")
    check[dm_walker_xml](t, "dm_walker_xml")
    check[lift_large_box_xml](t, "lift_large_box_xml")
    check[place_cradle_xml](t, "place_cradle_xml")
    check[place_brick_xml](t, "place_brick_xml")
    check[lift_brick_xml](t, "lift_brick_xml")
    check[reassemble5_xml](t, "reassemble5_xml")
    check[reach_site_features_xml](t, "reach_site_features_xml")
    check[reach_duplo_xml](t, "reach_duplo_xml")
    check[stack_3_bricks_xml](t, "stack_3_bricks_xml")
    check[stack_3_random_xml](t, "stack_3_random_xml")
    check[stack_2_bricks_moveable_base_xml](t, "stack_2_bricks_moveable_base_xml")
    check[stack_2_bricks_xml](t, "stack_2_bricks_xml")
    check[dm_dog_stand_walk_xml](t, "dm_dog_stand_walk_xml")
    check[dm_dog_run_xml](t, "dm_dog_run_xml")
    check[dm_dog_trot_xml](t, "dm_dog_trot_xml")
    check[dm_dog_fetch_xml](t, "dm_dog_fetch_xml")

    print()
    print("models compared           :", t.models)
    print("values that MOVED         :", t.bad)
    print("models with a NON-default :", t.nondefault, "(the rest take (-1, 1))")

    assert_true(
        t.models == 56,
        "expected 56 models, compared " + String(t.models),
    )
    assert_true(
        t.bad == 0,
        String(t.bad) + " model(s) changed their advertised action bounds —"
        " that is a behaviour change, not a refactor. See DIFF above.",
    )
    assert_true(
        t.nondefault > 0,
        "every model took the (-1, 1) fallback — this gate compared a default"
        " against itself and proved nothing",
    )
    print()
    print("PASS")
