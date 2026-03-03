"""Minimal compile test: import HalfCheetahModel from half_cheetah_xml.mojo.

If this compiles and runs, the XML model definition is working.

Run with:
    cd mojo-rl && pixi run mojo run test_xml_model_import.mojo
"""

from envs.half_cheetah.half_cheetah_xml import HalfCheetahModel


fn main() raises:
    print("HalfCheetahModel imported successfully")
    print("  nbody  =", HalfCheetahModel.nbody)
    print("  njoint =", HalfCheetahModel.njoint)
    print("  nq     =", HalfCheetahModel.nq)
    print("  nv     =", HalfCheetahModel.nv)
    print("  ngeom  =", HalfCheetahModel.ngeom)
    print("  nact   =", HalfCheetahModel.nact)
    print("  nsite  =", HalfCheetahModel.nsite)
