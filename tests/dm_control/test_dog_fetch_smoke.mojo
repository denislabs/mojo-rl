"""Does dog `fetch` BUILD, and does its model match the reference dimensions?

The cheapest possible first contact with the compiler: no stepping, no MuJoCo
comparison beyond the dimensions the generator already verified in Python.
`dog_fetch_xml.mojo` is 76 kB of comptime MJCF and `dog_fetch_config.mojo` has
never been through a compiler, so this exists to separate "it parses and the
model builds" from every later question.
"""

from std.testing import assert_true, TestSuite

from mojo_rl.envs.dm_control.dog import (
    DMDogFetchModel,
    DOG_FETCH_OBS_DIM,
)


def test_dog_fetch_model_dims() raises:
    print("--- dog fetch: model dimensions ---")
    comptime M = DMDogFetchModel
    print("  nbody ", M.NBODY, " njoint", M.NJOINT)
    print("  nq    ", M.NQ, " nv    ", M.NV)
    print("  ngeom ", M.NGEOM, " nsite ", M.NSITE, " nact", M.nact)
    print("  obs   ", DOG_FETCH_OBS_DIM)

    # Measured from the compiled reference model before the port was written.
    assert_true(M.NBODY == 63, "nbody should be 63 (dog 62 + ball)")
    assert_true(M.NJOINT == 75, "njoint should be 75 (dog 74 + ball_root)")
    assert_true(M.NQ == 87, "nq should be 87 (dog 80 + free joint 7)")
    assert_true(M.NV == 85, "nv should be 85 (dog 79 + free joint 6)")
    assert_true(M.NGEOM == 134, "ngeom should be 134 (dog 128 + ball/target/4 walls)")
    assert_true(M.NSITE == 12, "nsite should be 12 — fetch adds none")
    assert_true(M.nact == 38, "nact should be 38 — fetch adds no actuator")
    assert_true(DOG_FETCH_OBS_DIM == 232, "obs should be 223 + 6 + 3")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
