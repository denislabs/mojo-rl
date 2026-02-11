"""Test WorldBody with GeomSpec/PlaneGeom."""

from envs.half_cheetah import HalfCheetah
from envs.half_cheetah.half_cheetah_def import HalfCheetahWorldBody
from envs.hopper import Hopper
from envs.hopper.hopper_def import HopperWorldBody
from physics3d.model import GeomSpec, PlaneGeom, WorldBody


fn main():
    # Test HalfCheetah construction with WorldBody
    var hc = HalfCheetah()
    print("HalfCheetah model created successfully")
    print("  ground_z =", hc.model.ground_z)
    print("  friction =", hc.model.friction)
    print("  ground_contype =", hc.model.ground_contype)
    print("  ground_conaffinity =", hc.model.ground_conaffinity)

    # Verify HalfCheetah WorldBody values match MuJoCo half_cheetah.xml
    var hc_ok = True
    if hc.model.ground_z != 0.0:
        print("  FAIL: ground_z should be 0.0")
        hc_ok = False
    if hc.model.friction != 0.4:
        print("  FAIL: friction should be 0.4 (from PlaneGeom)")
        hc_ok = False
    if hc.model.ground_contype != 1:
        print("  FAIL: ground_contype should be 1")
        hc_ok = False
    if hc.model.ground_conaffinity != 1:
        print("  FAIL: ground_conaffinity should be 1")
        hc_ok = False
    if hc_ok:
        print("  HalfCheetah WorldBody: PASS")

    # Test Hopper construction with WorldBody
    var hp = Hopper()
    print("\nHopper model created successfully")
    print("  ground_z =", hp.model.ground_z)
    print("  friction =", hp.model.friction)
    print("  ground_contype =", hp.model.ground_contype)
    print("  ground_conaffinity =", hp.model.ground_conaffinity)

    # Verify Hopper WorldBody values match MuJoCo hopper.xml
    var hp_ok = True
    if hp.model.ground_z != 0.0:
        print("  FAIL: ground_z should be 0.0")
        hp_ok = False
    if hp.model.friction != 0.9:
        print("  FAIL: friction should be 0.9 (from PlaneGeom)")
        hp_ok = False
    if hp.model.ground_contype != 1:
        print("  FAIL: ground_contype should be 1")
        hp_ok = False
    if hp.model.ground_conaffinity != 1:
        print("  FAIL: ground_conaffinity should be 1")
        hp_ok = False
    if hp_ok:
        print("  Hopper WorldBody: PASS")

    if hc_ok and hp_ok:
        print("\nAll WorldBody tests PASSED!")
    else:
        print("\nSome tests FAILED!")
