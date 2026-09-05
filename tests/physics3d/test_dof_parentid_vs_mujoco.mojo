"""`Model.dof_parentid` against MuJoCo's own `m->dof_parentid`, entry for entry.

The tree-ordered LDL (`dynamics/ldl.mojo`, `_ldl_factor_tree_env`) walks this
table to know which entries of `L` exist; a wrong parent is a wrong
factorisation with no error anywhere. MuJoCo exposes the table directly, so
the gate is a plain comparison over the same model list the tree-block gate
uses (`tree_block_goldens`).

⚠ VACUITY IS THE DEFAULT FAILURE: the tail prints models COMPARED beside
dofs DIFFERING and fails if fewer than 20 models were compared.

Run: pixi run mojo run -I . tests/physics3d/test_dof_parentid_vs_mujoco.mojo
"""

from std.python import Python
from std.testing import assert_true

from mojo_rl.physics3d.parser.runtime_load import (
    parse_model_runtime,
    dims_from_flat,
    build_model_runtime,
)
from mojo_rl.physics3d.fields import Model, DynDims
from tests.physics3d.tree_block_goldens import blk_case_count, blk_path

comptime DT = DType.float64


def main() raises:
    var mujoco = Python.import_module("mujoco")
    print("=== dof_parentid vs MuJoCo 3.10.0 ===")
    var models_compared = 0
    var dofs_compared = 0
    var dofs_differing = 0
    var models_differing = 0
    for c in range(blk_case_count()):
        var path = blk_path(c)
        var fmd = parse_model_runtime(path)
        var dims = dims_from_flat(fmd, max_contacts=16, nmesh_verts=0)
        var m = Model[DT, DynDims](dims)
        var built = True
        try:
            build_model_runtime[DT](fmd, dims, m)
        except:
            built = False
        if not built:
            dims = dims_from_flat(
                fmd, max_contacts=16, nmesh_verts=262144, nmesh_tri=0
            )
            m = Model[DT, DynDims](dims)
            build_model_runtime[DT](fmd, dims, m)
        var nv = dims.get_nv()
        var mm = mujoco.MjModel.from_xml_path(path)
        var mj_nv = Int(py=mm.nv)
        if mj_nv != nv:
            print("---", path, ": nv", nv, "vs MuJoCo", mj_nv, "-- skipped")
            continue
        models_compared += 1
        var bad = 0
        for d in range(nv):
            var want = Int(py=mm.dof_parentid[d])
            var got = Int(m.dof_parentid.data[d])
            dofs_compared += 1
            if want != got:
                bad += 1
                dofs_differing += 1
                if bad <= 4:
                    print("      ", path, "dof", d, ": ours", got, "MuJoCo", want)
        if bad > 0:
            models_differing += 1
        print("---", path, "nv", nv, ":", nv - bad, "/", nv, "parents agree")
    print(
        "models compared:", models_compared, " differing:", models_differing,
        " dofs compared:", dofs_compared, " differing:", dofs_differing,
    )
    assert_true(models_compared >= 20, "fewer than 20 models compared")
    assert_true(dofs_differing == 0, "dof_parentid disagrees with MuJoCo")
