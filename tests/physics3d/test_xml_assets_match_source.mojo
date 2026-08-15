"""Every extracted `.xml` asset is byte-identical to the MJCF string in source.

⚠ THIS IS THE DIFFERENTIAL GATE FOR PHASE 1b's EXTRACTION, and it is only
meaningful while BOTH copies exist. Phase 1b moves ~1.1 MB of MJCF out of 50
`*_xml.mojo` files and onto disk so models can be edited without a rebuild.
Until the comptime strings are deleted there are two copies of every model,
and two copies drift — silently, because a model that renders and steps looks
fine whichever copy it read.

The 56 `.xml` files were WRITTEN BY MOJO from these same comptime strings, so
they started identical by construction rather than by a Python
re-implementation of `merge_mjcf` that could disagree. This gate is what keeps
them identical: edit one side only and it fails on the next run.

⚠ IT IS ALSO THE COMPOSITION GATE. 34 of these models are not literal strings
in source — they are `merge_mjcf(...)` of shared dm_control fragments, or raw
`+` concatenation (dog splices a floor geom between a head and a tail). The
files are FLAT: whatever the composition produced, verbatim. That is what
makes the extraction lossless and order-preserving; `<include>`-shaped files
would move the floor geom to the end of `<worldbody>` and renumber every geom
id after it.

⚠ NON-VACUITY: comparing "" to "" passes. The run prints the bytes actually
compared and fails if any file is empty or if the total falls short of the
1.1 MB that was extracted.

Run: pixi run mojo run -I . tests/physics3d/test_xml_assets_match_source.mojo
"""

from std.testing import assert_true

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


# The extraction wrote 1_100_474 bytes over 56 files. A run that compares
# materially less than that is comparing something other than the corpus.
comptime MIN_TOTAL_BYTES: Int = 1_090_000
comptime EXPECT_MODELS: Int = 57


struct Tally(Copyable, Movable):
    var models: Int
    var bytes: Int
    var bad: Int

    def __init__(out self):
        self.models = 0
        self.bytes = 0
        self.bad = 0


def check(mut t: Tally, name: String, path: String, embedded: String) raises:
    """One model: the file on disk vs the MJCF string still in source."""
    var on_disk: String
    try:
        with open(path, "r") as f:
            on_disk = f.read()
    except:
        t.bad += 1
        print("  MISSING", name, "->", path)
        return

    t.models += 1
    t.bytes += embedded.byte_length()

    if on_disk.byte_length() == 0:
        t.bad += 1
        print("  EMPTY", name, "->", path)
        return

    if on_disk != embedded:
        t.bad += 1
        print(
            "  DIFFERS", name, "->", path,
            ": disk=", on_disk.byte_length(),
            "bytes, source=", embedded.byte_length(), "bytes",
        )
        # First differing byte, so the report says WHERE rather than just THAT.
        var n = on_disk.byte_length()
        if embedded.byte_length() < n:
            n = embedded.byte_length()
        var db = on_disk.as_bytes()
        var eb = embedded.as_bytes()
        for i in range(n):
            if db[i] != eb[i]:
                print("    first difference at byte", i)
                return
        print("    identical up to byte", n, "— one side is longer")


def main() raises:
    var t = Tally()
    print("=== extracted .xml assets vs the MJCF strings in source ===")

    check(t, "ant_xml", "mojo_rl/envs/ant/assets/ant.xml", String(ant_xml))
    check(t, "half_cheetah_xml", "mojo_rl/envs/half_cheetah/assets/half_cheetah.xml", String(half_cheetah_xml))
    check(t, "hopper_xml", "mojo_rl/envs/hopper/assets/hopper.xml", String(hopper_xml))
    check(t, "humanoid_xml", "mojo_rl/envs/humanoid/assets/humanoid.xml", String(humanoid_xml))
    check(t, "humanoid_standup_xml", "mojo_rl/envs/humanoid_standup/assets/humanoid_standup.xml", String(humanoid_standup_xml))
    check(t, "inverted_double_pendulum_xml", "mojo_rl/envs/inverted_double_pendulum/assets/inverted_double_pendulum.xml", String(inverted_double_pendulum_xml))
    check(t, "inverted_pendulum_xml", "mojo_rl/envs/inverted_pendulum/assets/inverted_pendulum.xml", String(inverted_pendulum_xml))
    check(t, "pusher_xml", "mojo_rl/envs/pusher/assets/pusher.xml", String(pusher_xml))
    check(t, "reacher_xml", "mojo_rl/envs/reacher/assets/reacher.xml", String(reacher_xml))
    check(t, "swimmer_xml", "mojo_rl/envs/swimmer/assets/swimmer.xml", String(swimmer_xml))
    check(t, "walker2d_xml", "mojo_rl/envs/walker2d/assets/walker2d.xml", String(walker2d_xml))
    check(t, "sawyer_reach_xml", "mojo_rl/envs/metaworld/assets/sawyer_reach.xml", String(sawyer_reach_xml))
    check(t, "SO_ARM100_XML", "mojo_rl/envs/robots/assets/so_arm100.xml", String(SO_ARM100_XML))
    check(t, "SO_ARM101_XML", "mojo_rl/envs/robots/assets/so_arm101.xml", String(SO_ARM101_XML))
    check(t, "dm_acrobot_xml", "mojo_rl/envs/dm_control/assets/acrobot.xml", String(dm_acrobot_xml))
    check(t, "dm_ball_in_cup_xml", "mojo_rl/envs/dm_control/assets/ball_in_cup.xml", String(dm_ball_in_cup_xml))
    check(t, "dm_cartpole1_xml", "mojo_rl/envs/dm_control/assets/cartpole1.xml", String(dm_cartpole1_xml))
    check(t, "dm_cartpole2_xml", "mojo_rl/envs/dm_control/assets/cartpole2.xml", String(dm_cartpole2_xml))
    check(t, "dm_cartpole3_xml", "mojo_rl/envs/dm_control/assets/cartpole3.xml", String(dm_cartpole3_xml))
    check(t, "dm_cheetah_xml", "mojo_rl/envs/dm_control/assets/cheetah.xml", String(dm_cheetah_xml))
    check(t, "dm_finger_xml", "mojo_rl/envs/dm_control/assets/finger.xml", String(dm_finger_xml))
    check(t, "dm_finger_spin_xml", "mojo_rl/envs/dm_control/assets/finger_spin.xml", String(dm_finger_spin_xml))
    check(t, "dm_fish_xml", "mojo_rl/envs/dm_control/assets/fish.xml", String(dm_fish_xml))
    check(t, "dm_hopper_xml", "mojo_rl/envs/dm_control/assets/hopper.xml", String(dm_hopper_xml))
    check(t, "dm_humanoid_xml", "mojo_rl/envs/dm_control/assets/humanoid.xml", String(dm_humanoid_xml))
    check(t, "dm_humanoid_cmu_xml", "mojo_rl/envs/dm_control/assets/humanoid_cmu.xml", String(dm_humanoid_cmu_xml))
    check(t, "dm_manipulator_bring_ball_xml", "mojo_rl/envs/dm_control/assets/manipulator_bring_ball.xml", String(dm_manipulator_bring_ball_xml))
    check(t, "dm_manipulator_bring_peg_xml", "mojo_rl/envs/dm_control/assets/manipulator_bring_peg.xml", String(dm_manipulator_bring_peg_xml))
    check(t, "dm_manipulator_insert_ball_xml", "mojo_rl/envs/dm_control/assets/manipulator_insert_ball.xml", String(dm_manipulator_insert_ball_xml))
    check(t, "dm_manipulator_insert_peg_xml", "mojo_rl/envs/dm_control/assets/manipulator_insert_peg.xml", String(dm_manipulator_insert_peg_xml))
    check(t, "dm_pendulum_xml", "mojo_rl/envs/dm_control/assets/pendulum.xml", String(dm_pendulum_xml))
    check(t, "dm_point_mass_xml", "mojo_rl/envs/dm_control/assets/point_mass.xml", String(dm_point_mass_xml))
    check(t, "dm_quadruped_walk_xml", "mojo_rl/envs/dm_control/assets/quadruped_walk.xml", String(dm_quadruped_walk_xml))
    check(t, "dm_quadruped_run_xml", "mojo_rl/envs/dm_control/assets/quadruped_run.xml", String(dm_quadruped_run_xml))
    check(t, "dm_quadruped_fetch_xml", "mojo_rl/envs/dm_control/assets/quadruped_fetch.xml", String(dm_quadruped_fetch_xml))
    check(t, "dm_reacher_xml", "mojo_rl/envs/dm_control/assets/reacher.xml", String(dm_reacher_xml))
    check(t, "dm_reacher_hard_xml", "mojo_rl/envs/dm_control/assets/reacher_hard.xml", String(dm_reacher_hard_xml))
    check(t, "dm_stacker_2_xml", "mojo_rl/envs/dm_control/assets/stacker_2.xml", String(dm_stacker_2_xml))
    check(t, "dm_stacker_4_xml", "mojo_rl/envs/dm_control/assets/stacker_4.xml", String(dm_stacker_4_xml))
    check(t, "dm_swimmer6_xml", "mojo_rl/envs/dm_control/assets/swimmer6.xml", String(dm_swimmer6_xml))
    check(t, "dm_swimmer15_xml", "mojo_rl/envs/dm_control/assets/swimmer15.xml", String(dm_swimmer15_xml))
    check(t, "dm_walker_xml", "mojo_rl/envs/dm_control/assets/walker.xml", String(dm_walker_xml))
    check(t, "lift_large_box_xml", "mojo_rl/envs/dm_control/assets/manipulation/lift_large_box.xml", String(lift_large_box_xml))
    check(t, "place_cradle_xml", "mojo_rl/envs/dm_control/assets/manipulation/place_cradle.xml", String(place_cradle_xml))
    check(t, "place_brick_xml", "mojo_rl/envs/dm_control/assets/manipulation/place_brick.xml", String(place_brick_xml))
    check(t, "lift_brick_xml", "mojo_rl/envs/dm_control/assets/manipulation/lift_brick.xml", String(lift_brick_xml))
    check(t, "reassemble5_xml", "mojo_rl/envs/dm_control/assets/manipulation/reassemble5.xml", String(reassemble5_xml))
    check(t, "reach_site_features_xml", "mojo_rl/envs/dm_control/assets/manipulation/reach_site_features.xml", String(reach_site_features_xml))
    check(t, "reach_duplo_xml", "mojo_rl/envs/dm_control/assets/manipulation/reach_duplo.xml", String(reach_duplo_xml))
    check(t, "stack_3_bricks_xml", "mojo_rl/envs/dm_control/assets/manipulation/stack_3_bricks.xml", String(stack_3_bricks_xml))
    check(t, "stack_3_random_xml", "mojo_rl/envs/dm_control/assets/manipulation/stack_3_random.xml", String(stack_3_random_xml))
    check(t, "stack_2_bricks_moveable_base_xml", "mojo_rl/envs/dm_control/assets/manipulation/stack_2_bricks_moveable_base.xml", String(stack_2_bricks_moveable_base_xml))
    check(t, "stack_2_bricks_xml", "mojo_rl/envs/dm_control/assets/manipulation/stack_2_bricks.xml", String(stack_2_bricks_xml))
    check(t, "dm_dog_stand_walk_xml", "mojo_rl/envs/dm_control/assets/dog_stand_walk.xml", String(dm_dog_stand_walk_xml))
    check(t, "dm_dog_run_xml", "mojo_rl/envs/dm_control/assets/dog_run.xml", String(dm_dog_run_xml))
    check(t, "dm_dog_trot_xml", "mojo_rl/envs/dm_control/assets/dog_trot.xml", String(dm_dog_trot_xml))
    check(t, "dm_dog_fetch_xml", "mojo_rl/envs/dm_control/assets/dog_fetch.xml", String(dm_dog_fetch_xml))

    print()
    print("models compared:", t.models)
    print("bytes compared :", t.bytes)
    print("mismatches     :", t.bad)

    assert_true(
        t.models == EXPECT_MODELS,
        "expected " + String(EXPECT_MODELS) + " models, compared "
        + String(t.models) + " — a file went missing",
    )
    assert_true(
        t.bytes >= MIN_TOTAL_BYTES,
        "only " + String(t.bytes) + " bytes compared, expected at least "
        + String(MIN_TOTAL_BYTES) + " — the corpus shrank",
    )
    assert_true(
        t.bad == 0,
        String(t.bad) + " asset(s) differ from their source string — the two"
        " copies have drifted; see the report above",
    )
    print()
    print("PASS")
