"""Test inertiafromgeom against MuJoCo reference.

Compares our computed body mass/inertia/ipos/iquat (from inertiafromgeom)
against MuJoCo Python output for HalfCheetah and Hopper models.

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_inertiafromgeom_vs_mujoco.mojo
"""

from std.python import Python, PythonObject
from std.math import abs
from std.collections import InlineArray
from std.testing import assert_true, TestSuite

from std.gpu.host import DeviceContext
from mojo_rl.physics3d.fields import Model
from mojo_rl.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    BODY_IDX_MASS,
    BODY_IDX_IXX,
    BODY_IDX_IPOS_X,
)

from mojo_rl.envs.half_cheetah.half_cheetah_xml import HalfCheetahModel
from mojo_rl.envs.half_cheetah.half_cheetah_config import HalfCheetahConfig
from mojo_rl.envs.hopper.hopper_xml import HopperModel


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float64

# Tolerances
comptime MASS_TOL: Float64 = 1e-6  # Mass absolute tolerance
comptime INERTIA_TOL: Float64 = 1e-6  # Inertia absolute tolerance
comptime IPOS_TOL: Float64 = 1e-6  # ipos (CoM offset) tolerance


# =============================================================================
# HalfCheetah test
# =============================================================================


def test_half_cheetah() raises:
    """Compare inertiafromgeom output for HalfCheetah against MuJoCo."""
    print("--- Test: HalfCheetah inertiafromgeom ---")

    comptime NQ = HalfCheetahModel.NQ
    comptime NV = HalfCheetahModel.NV
    comptime NBODY = HalfCheetahModel.NBODY
    comptime NJOINT = HalfCheetahModel.NJOINT
    comptime NGEOM = HalfCheetahModel.NGEOM
    comptime MAX_CONTACTS = HalfCheetahConfig.MAX_CONTACTS

    # Spec-direct fields build — <compiler inertiafromgeom> + settotalmass
    # run inside init_fields (fields_build; G4).
    var ctx = DeviceContext()
    var mf = Model[
        DTYPE, NV, NBODY, NJOINT, NGEOM, HalfCheetahModel.MAX_EQUALITY,
        HalfCheetahModel.MAX_TENDON, HalfCheetahModel.NSITE,
        HalfCheetahModel.NEXCLUDE, 0,
    ]()
    HalfCheetahModel.init_fields[DTYPE, 0](ctx, mf)

    # Get MuJoCo reference
    var mujoco = Python.import_module("mujoco")
    var xml_path = "./references/Gymnasium-main/gymnasium/envs/mujoco/assets/half_cheetah.xml"
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)

    # Compare body mass, inertia, ipos
    var all_pass = True
    var max_mass_err: Float64 = 0.0
    var max_inertia_err: Float64 = 0.0
    var max_ipos_err: Float64 = 0.0

    for i in range(1, NBODY):
        # Mass
        var our_mass = Float64(mf.bodies.data[i * MODEL_BODY_SIZE + BODY_IDX_MASS])
        var mj_mass = Float64(py=mj_model.body_mass[i])
        var mass_err = abs(our_mass - mj_mass)
        if mass_err > max_mass_err:
            max_mass_err = mass_err
        if mass_err > MASS_TOL:
            print(
                "  FAIL body",
                i,
                "mass: ours=",
                our_mass,
                "mj=",
                mj_mass,
                "err=",
                mass_err,
            )
            all_pass = False

        # Inertia (3 principal moments)
        for k in range(3):
            var our_I = Float64(mf.bodies.data[i * MODEL_BODY_SIZE + BODY_IDX_IXX + k])
            var mj_I = Float64(py=mj_model.body_inertia[i][k])
            var I_err = abs(our_I - mj_I)
            if I_err > max_inertia_err:
                max_inertia_err = I_err
            if I_err > INERTIA_TOL:
                print(
                    "  FAIL body",
                    i,
                    "inertia[",
                    k,
                    "]: ours=",
                    our_I,
                    "mj=",
                    mj_I,
                    "err=",
                    I_err,
                )
                all_pass = False

        # ipos (CoM offset from body origin)
        for k in range(3):
            var our_ipos = Float64(mf.bodies.data[i * MODEL_BODY_SIZE + BODY_IDX_IPOS_X + k])
            var mj_ipos = Float64(py=mj_model.body_ipos[i][k])
            var ipos_err = abs(our_ipos - mj_ipos)
            if ipos_err > max_ipos_err:
                max_ipos_err = ipos_err
            if ipos_err > IPOS_TOL:
                print(
                    "  FAIL body",
                    i,
                    "ipos[",
                    k,
                    "]: ours=",
                    our_ipos,
                    "mj=",
                    mj_ipos,
                    "err=",
                    ipos_err,
                )
                all_pass = False

    if all_pass:
        print(
            "  ALL OK  max_mass_err=",
            max_mass_err,
            " max_inertia_err=",
            max_inertia_err,
            " max_ipos_err=",
            max_ipos_err,
        )
    assert_true(all_pass, "Inertia from geom mismatch for: HalfCheetah")


# =============================================================================
# Hopper test
# =============================================================================


def test_hopper() raises:
    """Compare inertiafromgeom output for Hopper against MuJoCo."""
    print("--- Test: Hopper inertiafromgeom ---")

    comptime NQ = HopperModel.NQ
    comptime NV = HopperModel.NV
    comptime NBODY = HopperModel.NBODY
    comptime NJOINT = HopperModel.NJOINT
    comptime NGEOM = HopperModel.NGEOM

    # Spec-direct fields build — <compiler inertiafromgeom> runs inside
    # init_fields (fields_build; G4). Hopper has no settotalmass.
    var ctx = DeviceContext()
    var mf = Model[
        DTYPE, NV, NBODY, NJOINT, NGEOM, HopperModel.MAX_EQUALITY,
        HopperModel.MAX_TENDON, HopperModel.NSITE, HopperModel.NEXCLUDE, 0,
    ]()
    HopperModel.init_fields[DTYPE, 0](ctx, mf)

    # Get MuJoCo reference
    var mujoco = Python.import_module("mujoco")
    var xml_path = "./references/Gymnasium-main/gymnasium/envs/mujoco/assets/hopper.xml"
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)

    # Compare body mass, inertia, ipos
    var all_pass = True
    var max_mass_err: Float64 = 0.0
    var max_inertia_err: Float64 = 0.0
    var max_ipos_err: Float64 = 0.0

    for i in range(1, NBODY):
        # Mass
        var our_mass = Float64(mf.bodies.data[i * MODEL_BODY_SIZE + BODY_IDX_MASS])
        var mj_mass = Float64(py=mj_model.body_mass[i])
        var mass_err = abs(our_mass - mj_mass)
        if mass_err > max_mass_err:
            max_mass_err = mass_err
        if mass_err > MASS_TOL:
            print(
                "  FAIL body",
                i,
                "mass: ours=",
                our_mass,
                "mj=",
                mj_mass,
                "err=",
                mass_err,
            )
            all_pass = False

        # Inertia (3 principal moments)
        for k in range(3):
            var our_I = Float64(mf.bodies.data[i * MODEL_BODY_SIZE + BODY_IDX_IXX + k])
            var mj_I = Float64(py=mj_model.body_inertia[i][k])
            var I_err = abs(our_I - mj_I)
            if I_err > max_inertia_err:
                max_inertia_err = I_err
            if I_err > INERTIA_TOL:
                print(
                    "  FAIL body",
                    i,
                    "inertia[",
                    k,
                    "]: ours=",
                    our_I,
                    "mj=",
                    mj_I,
                    "err=",
                    I_err,
                )
                all_pass = False

        # ipos (CoM offset from body origin)
        for k in range(3):
            var our_ipos = Float64(mf.bodies.data[i * MODEL_BODY_SIZE + BODY_IDX_IPOS_X + k])
            var mj_ipos = Float64(py=mj_model.body_ipos[i][k])
            var ipos_err = abs(our_ipos - mj_ipos)
            if ipos_err > max_ipos_err:
                max_ipos_err = ipos_err
            if ipos_err > IPOS_TOL:
                print(
                    "  FAIL body",
                    i,
                    "ipos[",
                    k,
                    "]: ours=",
                    our_ipos,
                    "mj=",
                    mj_ipos,
                    "err=",
                    ipos_err,
                )
                all_pass = False

    if all_pass:
        print(
            "  ALL OK  max_mass_err=",
            max_mass_err,
            " max_inertia_err=",
            max_inertia_err,
            " max_ipos_err=",
            max_ipos_err,
        )
    assert_true(all_pass, "Inertia from geom mismatch for: Hopper")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
