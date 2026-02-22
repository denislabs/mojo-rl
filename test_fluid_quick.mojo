"""Quick smoke test: fluid forces compile and produce non-zero output."""
from physics3d.types import Model, Data, _max_one
from physics3d.dynamics.fluid_forces import compute_fluid_forces
from physics3d.joint_types import JointDef

fn main():
    # 2-body system: worldbody (0) + one capsule body (1) with a slide joint
    # NQ=1, NV=1, NBODY=2, NJOINT=1, MAX_CONTACTS=4
    alias DTYPE = DType.float64
    alias NQ = 1; alias NV = 1; alias NBODY = 2; alias NJOINT = 1; alias MC = 4
    alias CDOF_SIZE = _max_one[NV * 6]()
    alias V_SIZE = _max_one[NV]()

    var model = Model[DTYPE, NQ, NV, NBODY, NJOINT, MC]()
    model.opt_density = 4000.0    # Swimmer density (kg/m³)
    model.opt_viscosity = 0.1     # Swimmer viscosity (Pa·s)
    model.body_mass[1] = 1.0
    # Inertia for body 1: Ixx=0.01, Iyy=0.05, Izz=0.05
    model.body_inertia[3] = 0.01  # body 1 Ixx
    model.body_inertia[4] = 0.05  # body 1 Iyy
    model.body_inertia[5] = 0.05  # body 1 Izz
    model.body_parent[1] = 0
    # Configure joint: body 1 has a slide joint along x, DOF 0
    model.num_joints = 1
    model.joints[0] = JointDef[DTYPE].create_slide(
        body_id=1,
        qpos_adr=0,
        dof_adr=0,
        pos=(Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](0)),
        axis=(Scalar[DTYPE](1), Scalar[DTYPE](0), Scalar[DTYPE](0)),
    )

    var data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MC]()
    # Body 1 moving at vx=1 m/s in world frame
    data.xvel[3] = 1.0   # vx for body 1
    # Identity quaternion for body 1: [x=0, y=0, z=0, w=1]
    data.xquat[4] = 0.0; data.xquat[5] = 0.0; data.xquat[6] = 0.0; data.xquat[7] = 1.0
    # CoM position for body 1 (world frame, at origin)
    data.xipos[3] = 0.0; data.xipos[4] = 0.0; data.xipos[5] = 0.0

    # cdof: DOF 0 is x-slide → ang=[0,0,0], lin=[1,0,0]
    var cdof = InlineArray[Scalar[DTYPE], CDOF_SIZE](uninitialized=True)
    cdof[0] = 0.0; cdof[1] = 0.0; cdof[2] = 0.0
    cdof[3] = 1.0; cdof[4] = 0.0; cdof[5] = 0.0

    var f_net = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    f_net[0] = 0.0

    compute_fluid_forces[DTYPE, NQ, NV, NBODY, NJOINT, MC](model, data, cdof, f_net)

    print("Fluid drag on DOF 0 (vx=1 m/s, rho=4000, mu=0.1):", f_net[0])
    if f_net[0] < -1.0:
        print("PASS: fluid drag opposes velocity (< -1.0)")
    else:
        print("FAIL: expected drag < -1.0, got", f_net[0])

    # Test early-out: no fluid when density=viscosity=0
    var model2 = Model[DTYPE, NQ, NV, NBODY, NJOINT, MC]()
    model2.body_mass[1] = 1.0; model2.body_inertia[3] = 0.01
    model2.body_inertia[4] = 0.05; model2.body_inertia[5] = 0.05
    model2.body_parent[1] = 0
    model2.num_joints = 1
    model2.joints[0] = model.joints[0]
    var data2 = Data[DTYPE, NQ, NV, NBODY, NJOINT, MC]()
    data2.xvel[3] = 100.0; data2.xquat[7] = 1.0
    var f_net2 = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    f_net2[0] = 0.0
    compute_fluid_forces[DTYPE, NQ, NV, NBODY, NJOINT, MC](model2, data2, cdof, f_net2)
    if f_net2[0] == 0.0:
        print("PASS: zero drag when density=viscosity=0 (early-out works)")
    else:
        print("FAIL: expected 0 force with no fluid, got", f_net2[0])
