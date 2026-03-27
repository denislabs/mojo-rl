from mojo_rl.physics3d.types import Model, Data
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.dynamics.jacobian import compute_subtree_com, compute_cdof
from mojo_rl.envs.humanoid.humanoid_xml import HumanoidModel

comptime DTYPE = DType.float64
comptime NQ = HumanoidModel.NQ
comptime NV = HumanoidModel.NV
comptime NBODY = HumanoidModel.NBODY
comptime NJOINT = HumanoidModel.NJOINT
comptime NGEOM = HumanoidModel.NGEOM
comptime MAX_CONTACTS = HumanoidModel.MAX_CONTACTS

def main() raises:
    var model = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
        HumanoidModel.MAX_EQUALITY, HumanoidModel.CONE_TYPE,
        HumanoidModel.MAX_TENDON, HumanoidModel.NSITE,
    ]()
    var data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HumanoidModel.NSITE]()
    HumanoidModel.setup_model_and_data(model, data)
    for i in range(NQ):
        data.qpos[i] = Scalar[DTYPE](0)
    data.qpos[2] = Scalar[DTYPE](3.0)
    data.qpos[3] = Scalar[DTYPE](1.0)
    forward_kinematics(model, data)

    print("=== body_rootid ===")
    for b in range(NBODY):
        print("  body", b, "parent=", model.body_parent[b], "rootid=", model.body_rootid[b])

    var stcom = List[Scalar[DTYPE]](capacity=NBODY*3)
    for _ in range(NBODY*3):
        stcom.append(Scalar[DTYPE](0))
    compute_subtree_com[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS](model, data, stcom)

    print("\n=== subtree_com[root=1] vs xipos (offset for each body) ===")
    for b in range(NBODY):
        var root = model.body_rootid[b]
        var dx = Float64(data.xipos[b*3] - stcom[root*3])
        var dy = Float64(data.xipos[b*3+1] - stcom[root*3+1])
        var dz = Float64(data.xipos[b*3+2] - stcom[root*3+2])
        print("  body", b, "rootid=", root, "offset=(", dx, dy, dz, ")")

    # Compute cdof with and without subtree_com
    var cdof_old = List[Scalar[DTYPE]](capacity=NV*6)
    for _ in range(NV*6):
        cdof_old.append(Scalar[DTYPE](0))
    compute_cdof[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS](model, data, cdof_old)

    var cdof_new = List[Scalar[DTYPE]](capacity=NV*6)
    for _ in range(NV*6):
        cdof_new.append(Scalar[DTYPE](0))
    compute_cdof[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS](model, data, cdof_new, stcom)

    print("\n=== cdof comparison (DOFs 0-5 = free joint) ===")
    for d in range(6):
        var s_old = String("  DOF ") + String(d) + " old: ["
        var s_new = String("  DOF ") + String(d) + " new: ["
        for k in range(6):
            if k > 0:
                s_old += ", "
                s_new += ", "
            s_old += String(Float64(cdof_old[d*6+k]))
            s_new += String(Float64(cdof_new[d*6+k]))
        print(s_old + "]")
        print(s_new + "]")
