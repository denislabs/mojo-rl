from physics3d.robot.half_cheetah_def import HalfCheetahRobot
from testing import assert_true, TestSuite


fn test_robot() raises:
    print("=== Robot Tests ===\n")

    print(HalfCheetahRobot.nbody)


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
