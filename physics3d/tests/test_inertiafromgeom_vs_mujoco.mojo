"""Test inertiafromgeom against MuJoCo reference.

Compares our computed body mass/inertia/ipos/iquat (from inertiafromgeom)
against MuJoCo Python output for HalfCheetah and Hopper models.

Run with:
    cd mojo-rl && pixi run mojo run physics3d/tests/test_inertiafromgeom_vs_mujoco.mojo
"""

from python import Python, PythonObject
from math import abs
from collections import InlineArray

from physics3d.types import Model, Data, _max_one
from physics3d.model.model_def import Bodies, Joints, Geoms, ModelDef, ModelDefaults
from physics3d.model.inertia_from_geom import compute_inertia_from_geoms

from envs.half_cheetah.half_cheetah_def import (
    HalfCheetahModel,
    HalfCheetahBodies,
    HalfCheetahJoints,
    HalfCheetahGeoms,
    HalfCheetahParams,
    HalfCheetahDefaults,
)
from envs.hopper.hopper_def import (
    HopperModel,
    HopperBodies,
    HopperJoints,
    HopperGeoms,
    HopperDefaults,
)


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

fn test_half_cheetah() raises -> Bool:
    """Compare inertiafromgeom output for HalfCheetah against MuJoCo."""
    print("--- Test: HalfCheetah inertiafromgeom ---")

    comptime NQ = HalfCheetahModel.NQ
    comptime NV = HalfCheetahModel.NV
    comptime NBODY = HalfCheetahModel.NBODY
    comptime NJOINT = HalfCheetahModel.NJOINT
    comptime NGEOM = HalfCheetahModel.NGEOM
    comptime MAX_CONTACTS = HalfCheetahParams[DTYPE].MAX_CONTACTS

    # Build model with inertiafromgeom enabled
    var model = Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, 0, 1]()
    HalfCheetahBodies.setup_model[DTYPE, NQ, NV, NJOINT, MAX_CONTACTS, NGEOM, 0, 1](model)
    HalfCheetahJoints.setup_model[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NGEOM, 0, 1, HalfCheetahDefaults](model)
    HalfCheetahGeoms.setup_model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, 0, 1, HalfCheetahDefaults](model)

    # Run inertiafromgeom (overwrite body mass/inertia/ipos from geoms)
    compute_inertia_from_geoms(model)

    # Apply settotalmass=14
    var total_mass = Scalar[DTYPE](0)
    for i in range(1, NBODY):
        total_mass += model.body_mass[i]
    if total_mass > 0:
        var scale = Scalar[DTYPE](14.0) / total_mass
        for i in range(1, NBODY):
            model.body_mass[i] *= scale
            for k in range(3):
                model.body_inertia[i * 3 + k] *= scale

    # Get MuJoCo reference
    var mujoco = Python.import_module("mujoco")
    var xml_path = "../Gymnasium-main/gymnasium/envs/mujoco/assets/half_cheetah.xml"
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)

    # Compare body mass, inertia, ipos
    var all_pass = True
    var max_mass_err: Float64 = 0.0
    var max_inertia_err: Float64 = 0.0
    var max_ipos_err: Float64 = 0.0

    for i in range(1, NBODY):
        # Mass
        var our_mass = Float64(model.body_mass[i])
        var mj_mass = Float64(py=mj_model.body_mass[i])
        var mass_err = abs(our_mass - mj_mass)
        if mass_err > max_mass_err:
            max_mass_err = mass_err
        if mass_err > MASS_TOL:
            print("  FAIL body", i, "mass: ours=", our_mass, "mj=", mj_mass, "err=", mass_err)
            all_pass = False

        # Inertia (3 principal moments)
        for k in range(3):
            var our_I = Float64(model.body_inertia[i * 3 + k])
            var mj_I = Float64(py=mj_model.body_inertia[i][k])
            var I_err = abs(our_I - mj_I)
            if I_err > max_inertia_err:
                max_inertia_err = I_err
            if I_err > INERTIA_TOL:
                print("  FAIL body", i, "inertia[", k, "]: ours=", our_I, "mj=", mj_I, "err=", I_err)
                all_pass = False

        # ipos (CoM offset from body origin)
        for k in range(3):
            var our_ipos = Float64(model.body_ipos[i * 3 + k])
            var mj_ipos = Float64(py=mj_model.body_ipos[i][k])
            var ipos_err = abs(our_ipos - mj_ipos)
            if ipos_err > max_ipos_err:
                max_ipos_err = ipos_err
            if ipos_err > IPOS_TOL:
                print("  FAIL body", i, "ipos[", k, "]: ours=", our_ipos, "mj=", mj_ipos, "err=", ipos_err)
                all_pass = False

    if all_pass:
        print("  ALL OK  max_mass_err=", max_mass_err, " max_inertia_err=", max_inertia_err, " max_ipos_err=", max_ipos_err)
    return all_pass


# =============================================================================
# Hopper test
# =============================================================================

fn test_hopper() raises -> Bool:
    """Compare inertiafromgeom output for Hopper against MuJoCo."""
    print("--- Test: Hopper inertiafromgeom ---")

    comptime NQ = HopperModel.NQ
    comptime NV = HopperModel.NV
    comptime NBODY = HopperModel.NBODY
    comptime NJOINT = HopperModel.NJOINT
    comptime NGEOM = HopperModel.NGEOM

    # Build model with inertiafromgeom enabled
    var model = Model[DTYPE, NQ, NV, NBODY, NJOINT, 20, NGEOM, 0, 1]()
    HopperBodies.setup_model[DTYPE, NQ, NV, NJOINT, 20, NGEOM, 0, 1](model)
    HopperJoints.setup_model[DTYPE, NQ, NV, NBODY, 20, NGEOM, 0, 1, HopperDefaults](model)
    HopperGeoms.setup_model[DTYPE, NQ, NV, NBODY, NJOINT, 20, NGEOM, 0, 1, HopperDefaults](model)

    # Run inertiafromgeom (overwrite body mass/inertia/ipos from geoms)
    compute_inertia_from_geoms(model)

    # Hopper has no settotalmass

    # Get MuJoCo reference
    var mujoco = Python.import_module("mujoco")
    var xml_path = "../Gymnasium-main/gymnasium/envs/mujoco/assets/hopper.xml"
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)

    # Compare body mass, inertia, ipos
    var all_pass = True
    var max_mass_err: Float64 = 0.0
    var max_inertia_err: Float64 = 0.0
    var max_ipos_err: Float64 = 0.0

    for i in range(1, NBODY):
        # Mass
        var our_mass = Float64(model.body_mass[i])
        var mj_mass = Float64(py=mj_model.body_mass[i])
        var mass_err = abs(our_mass - mj_mass)
        if mass_err > max_mass_err:
            max_mass_err = mass_err
        if mass_err > MASS_TOL:
            print("  FAIL body", i, "mass: ours=", our_mass, "mj=", mj_mass, "err=", mass_err)
            all_pass = False

        # Inertia (3 principal moments)
        for k in range(3):
            var our_I = Float64(model.body_inertia[i * 3 + k])
            var mj_I = Float64(py=mj_model.body_inertia[i][k])
            var I_err = abs(our_I - mj_I)
            if I_err > max_inertia_err:
                max_inertia_err = I_err
            if I_err > INERTIA_TOL:
                print("  FAIL body", i, "inertia[", k, "]: ours=", our_I, "mj=", mj_I, "err=", I_err)
                all_pass = False

        # ipos (CoM offset from body origin)
        for k in range(3):
            var our_ipos = Float64(model.body_ipos[i * 3 + k])
            var mj_ipos = Float64(py=mj_model.body_ipos[i][k])
            var ipos_err = abs(our_ipos - mj_ipos)
            if ipos_err > max_ipos_err:
                max_ipos_err = ipos_err
            if ipos_err > IPOS_TOL:
                print("  FAIL body", i, "ipos[", k, "]: ours=", our_ipos, "mj=", mj_ipos, "err=", ipos_err)
                all_pass = False

    if all_pass:
        print("  ALL OK  max_mass_err=", max_mass_err, " max_inertia_err=", max_inertia_err, " max_ipos_err=", max_ipos_err)
    return all_pass


# =============================================================================
# Main
# =============================================================================

fn main() raises:
    var passed = 0
    var failed = 0

    if test_half_cheetah():
        passed += 1
    else:
        failed += 1

    if test_hopper():
        passed += 1
    else:
        failed += 1

    print("\n============================================================")
    print("Results:", passed, "passed,", failed, "failed out of", passed + failed)
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("============================================================")
