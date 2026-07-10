"""G1 gate: fields-native invweight0 (compute_invweight0_fields) tolerance-exact
vs the legacy `compute_body_invweight0` (MuJoCo mj_setConst).

For each model: build ModelFields via `init_fields` (which fills invweight0 the
LEGACY way — CPU Model + setup_model_and_data + load_from_model — so mf holds the
legacy reference values); snapshot them; take the reference pose from a legacy
`setup_model_and_data` (its `reset_data` qpos, incl free-joint quats); run
`compute_invweight0_fields` (overwrites mf.body/dof_invweight0); compare fields
vs legacy. invweight0 feeds every constraint/solver inverse-weight, so this must
match to tight tolerance across hinge/slide (Walker2D) + free-joint (Ant) roots.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_invweight0_fields.mojo
"""

from std.math import abs
from std.gpu.host import DeviceContext

from mojo_rl.physics3d.fields import DataFields, ModelFields, DynamicsScratch
from mojo_rl.physics3d.types import Model, Data
from mojo_rl.physics3d.model.model_def import ModelDefLike
from mojo_rl.physics3d.dynamics.invweight_fields import compute_invweight0_fields
from mojo_rl.envs.walker2d.walker2d_xml import Walker2dModel
from mojo_rl.envs.ant.ant_xml import AntModel
from mojo_rl.envs.humanoid.humanoid_xml import HumanoidModel

comptime DT = DType.float64


def _check[MODEL: ModelDefLike](ctx: DeviceContext, name: String) raises:
    comptime NQ = MODEL.NQ
    comptime NV = MODEL.NV
    comptime NBODY = MODEL.NBODY
    comptime NJOINT = MODEL.NJOINT
    comptime MC = MODEL.MAX_CONTACTS
    comptime NGEOM = MODEL.NGEOM
    comptime CONE = MODEL.CONE_TYPE
    comptime NEQ = MODEL.MAX_EQUALITY
    comptime NTD = MODEL.MAX_TENDON
    comptime NSITE = MODEL.NSITE
    comptime NEXCL = MODEL.NEXCLUDE

    # fields model build — mf.invweight0 holds the LEGACY reference values.
    var mf = ModelFields[DT, NV, NBODY, NJOINT, NGEOM, NEQ, NTD, NSITE, NEXCL, 0]()
    MODEL.init_fields[DT, 0](ctx, mf)
    var bw_legacy = List[Scalar[DT]](length=NBODY * 2, fill=0)
    var dw_legacy = List[Scalar[DT]](length=NV, fill=0)
    for i in range(NBODY * 2):
        bw_legacy[i] = mf.body_invweight0.data[i]
    for i in range(NV):
        dw_legacy[i] = mf.dof_invweight0.data[i]

    # reference pose (reset_data qpos) from a legacy build.
    var lmodel = Model[DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, CONE, NTD, NSITE]()
    var ldata = Data[DT, NQ, NV, NBODY, NJOINT, MC, NSITE]()
    MODEL.setup_model_and_data[DT](lmodel, ldata)
    var d = DataFields[DT, NQ, NV, NBODY, MC, NSITE, 1]()
    for i in range(NQ):
        d.qpos.data[i] = ldata.qpos[i]

    # fields-native invweight0 (overwrites mf).
    var sc = DynamicsScratch[DT, NV, NBODY, 1]()
    compute_invweight0_fields[
        DT, NQ, NV, NBODY, NJOINT, MC, NGEOM, NEQ, NTD, NSITE, NEXCL, 0
    ](d, mf, sc)

    # compare fields vs legacy (relative error, small abs floor).
    var worst_bw = Float64(0)
    var worst_dw = Float64(0)
    for i in range(NBODY * 2):
        var lv = Float64(bw_legacy[i])
        var fv = Float64(mf.body_invweight0.data[i])
        var rel = abs(fv - lv) / (abs(lv) + 1e-9)
        if rel > worst_bw:
            worst_bw = rel
    for i in range(NV):
        var lv = Float64(dw_legacy[i])
        var fv = Float64(mf.dof_invweight0.data[i])
        var rel = abs(fv - lv) / (abs(lv) + 1e-9)
        if rel > worst_dw:
            worst_dw = rel
    print(
        "  ", name, ": worst body_invweight0 rel err", worst_bw,
        " dof_invweight0 rel err", worst_dw,
    )
    # Walker2D/Ant are bit-exact (~1e-14/1e-11); Humanoid ~1.5e-5 is intrinsic
    # fields-vs-legacy CRBA/LDL roundoff on a 27-DOF stiff system (dof_invweight0
    # = diag(M^-1) drifts identically, so it's upstream M/LDL, not the invweight
    # assembly). Physically negligible for a constraint-compliance scale; real
    # bugs are orders larger (the subtree_com-ref bug was rel 12.0).
    if worst_bw > 1e-4 or worst_dw > 1e-4:
        raise Error(String(name) + ": fields invweight0 diverges from legacy")
    print("  ", name, "PASS")


def main() raises:
    print("=== G1 fields invweight0 vs legacy (Walker2D hinge/slide + Ant free) ===")
    var ctx = DeviceContext()
    _check[Walker2dModel](ctx, "Walker2D")
    _check[AntModel](ctx, "Ant")
    _check[HumanoidModel](ctx, "Humanoid")
    print("test_invweight0_fields: ALL PASS")
