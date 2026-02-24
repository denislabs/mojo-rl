"""Prototype: FlatModelDef — InlineArray-based alternative to heterogeneous
Bodies/Joints type parameters.

The current design stores all body/joint data as compile-time TYPE constants:
    Bodies[Torso, BThigh, ...] — each body is a separate struct with comptime fields

The proposed new design stores everything in InlineArray of plain data structs:
    FlatModelDef[8, 9, 9, 9, 9, 6] with InlineArray[BodyData, 8]

This prototype answers three key questions:
  Q1. Does InlineArray[BodyData, NBODY] work in Mojo when BodyData is a custom struct?
  Q2. Can pm.NBODY (field of a comptime ParsedModel) be used as a struct type param?
  Q3. Does setup_model work with a regular for loop over InlineArray values?

Expected output:
  [Q1] InlineArray[BodyData, 8] created, bodies[0].mass = 14.0
  [Q2] FlatModelDef[8, 9, 9, 9, 9, 6] instantiated at comptime
  [Q3] Model setup via InlineArray loop completed, FK ran successfully
"""

from collections import InlineArray
from physics3d.parser import ParsedModel, parse_xml
from physics3d.types import Model, Data, ConeType
from physics3d.kinematics.forward_kinematics import forward_kinematics
from physics3d.joint_types import JNT_HINGE, JNT_SLIDE

# =============================================================================
# Flat data structs — must be ImplicitlyCopyable for use in InlineArray
# =============================================================================


struct BodyData(Copyable, ImplicitlyCopyable, Movable):
    """Flat runtime body data — replaces a compile-time BodySpec type.

    Note: String names are omitted here to keep the type trivially copyable.
    In setup_model we pass generic "body_N" names.
    """

    var parent: Int
    var mass: Float64
    var pos_x: Float64
    var pos_y: Float64
    var pos_z: Float64
    var quat_x: Float64
    var quat_y: Float64
    var quat_z: Float64
    var quat_w: Float64
    var ipos_x: Float64
    var ipos_y: Float64
    var ipos_z: Float64
    var iquat_x: Float64
    var iquat_y: Float64
    var iquat_z: Float64
    var iquat_w: Float64
    var ixx: Float64
    var iyy: Float64
    var izz: Float64

    fn __init__(
        out self,
        parent: Int = 0,
        mass: Float64 = 1.0,
        pos_x: Float64 = 0.0,
        pos_y: Float64 = 0.0,
        pos_z: Float64 = 0.0,
        quat_x: Float64 = 0.0,
        quat_y: Float64 = 0.0,
        quat_z: Float64 = 0.0,
        quat_w: Float64 = 1.0,
        ipos_x: Float64 = 0.0,
        ipos_y: Float64 = 0.0,
        ipos_z: Float64 = 0.0,
        iquat_x: Float64 = 0.0,
        iquat_y: Float64 = 0.0,
        iquat_z: Float64 = 0.0,
        iquat_w: Float64 = 1.0,
        ixx: Float64 = 0.01,
        iyy: Float64 = 0.01,
        izz: Float64 = 0.01,
    ):
        self.parent = parent
        self.mass = mass
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.quat_x = quat_x
        self.quat_y = quat_y
        self.quat_z = quat_z
        self.quat_w = quat_w
        self.ipos_x = ipos_x
        self.ipos_y = ipos_y
        self.ipos_z = ipos_z
        self.iquat_x = iquat_x
        self.iquat_y = iquat_y
        self.iquat_z = iquat_z
        self.iquat_w = iquat_w
        self.ixx = ixx
        self.iyy = iyy
        self.izz = izz


struct JointData(Copyable, ImplicitlyCopyable, Movable):
    """Flat runtime joint data — replaces a compile-time JointSpec type."""

    var jnt_type: Int  # JNT_HINGE=3, JNT_SLIDE=2
    var body_id: Int
    var nq: Int  # 1 for hinge/slide
    var nv: Int
    var pos_x: Float64  # anchor in body frame
    var pos_y: Float64
    var pos_z: Float64
    var axis_x: Float64
    var axis_y: Float64
    var axis_z: Float64
    var range_min: Float64
    var range_max: Float64
    var is_limited: Bool
    var armature: Float64
    var damping: Float64
    var stiffness: Float64

    fn __init__(
        out self,
        jnt_type: Int = JNT_HINGE,
        body_id: Int = 1,
        nq: Int = 1,
        nv: Int = 1,
        pos_x: Float64 = 0.0,
        pos_y: Float64 = 0.0,
        pos_z: Float64 = 0.0,
        axis_x: Float64 = 0.0,
        axis_y: Float64 = 1.0,
        axis_z: Float64 = 0.0,
        range_min: Float64 = -1.0,
        range_max: Float64 = 1.0,
        is_limited: Bool = False,
        armature: Float64 = 0.1,
        damping: Float64 = 0.01,
        stiffness: Float64 = 8.0,
    ):
        self.jnt_type = jnt_type
        self.body_id = body_id
        self.nq = nq
        self.nv = nv
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.axis_x = axis_x
        self.axis_y = axis_y
        self.axis_z = axis_z
        self.range_min = range_min
        self.range_max = range_max
        self.is_limited = is_limited
        self.armature = armature
        self.damping = damping
        self.stiffness = stiffness


# =============================================================================
# FlatModelDef — integer-parameterized, InlineArray-based model definition
# =============================================================================


struct FlatModelDef[
    NBODY: Int,
    NJOINT: Int,
    NQ: Int,
    NV: Int,
    NGEOM: Int,
    NACT: Int,
]:
    """Model definition using flat InlineArrays instead of heterogeneous types.

    Dimensions come from XML parser output. Data stored in InlineArray[BodyData/JointData].
    setup_model() uses a regular for loop — no @parameter needed.
    """

    # KEY TEST: does InlineArray[custom_struct, struct_param] work?
    var bodies: InlineArray[BodyData, Self.NBODY]
    var joints: InlineArray[JointData, Self.NJOINT]
    var gravity_z: Float64
    var timestep: Float64

    fn __init__(out self):
        # KEY TEST: does InlineArray fill constructor work with custom struct?
        self.bodies = InlineArray[BodyData, Self.NBODY](fill=BodyData())
        self.joints = InlineArray[JointData, Self.NJOINT](fill=JointData())
        self.gravity_z = -9.81
        self.timestep = 0.01

    fn setup_model[
        DTYPE: DType,
        MAX_CONTACTS: Int,
        MAX_EQUALITY: Int = 0,
        CONE_TYPE: Int = ConeType.ELLIPTIC,
        MAX_TENDON: Int = 0,
        NSITE: Int = 0,
    ](
        self,
        mut model: Model[
            DTYPE,
            Self.NQ,
            Self.NV,
            Self.NBODY,
            Self.NJOINT,
            MAX_CONTACTS,
            Self.NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
            NSITE,
        ],
    ):
        """Write body and joint data from InlineArrays to the Model struct.

        Uses regular for loops — no @parameter needed since we write
        Float64/Int values, not instantiate per-type specializations.
        """
        model.gravity = SIMD[DTYPE, 4](
            Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](self.gravity_z), Scalar[DTYPE](0)
        )
        model.timestep = Scalar[DTYPE](self.timestep)

        # Bodies (index 1..NBODY-1; worldbody=0 is pre-initialized by Model)
        for i in range(Self.NBODY - 1):
            var b = self.bodies[i].copy()
            var body_idx = i + 1
            model.set_body(
                body_idx,
                name="body_" + String(i),
                mass=Scalar[DTYPE](b.mass),
                inertia=(
                    Scalar[DTYPE](b.ixx),
                    Scalar[DTYPE](b.iyy),
                    Scalar[DTYPE](b.izz),
                ),
            )
            model.set_body_parent(body_idx, b.parent)
            model.set_body_local_frame(
                body_idx,
                pos=(
                    Scalar[DTYPE](b.pos_x),
                    Scalar[DTYPE](b.pos_y),
                    Scalar[DTYPE](b.pos_z),
                ),
                quat=(
                    Scalar[DTYPE](b.quat_x),
                    Scalar[DTYPE](b.quat_y),
                    Scalar[DTYPE](b.quat_z),
                    Scalar[DTYPE](b.quat_w),
                ),
            )
            model.set_body_ipos_iquat(
                body_idx,
                ipos=(
                    Scalar[DTYPE](b.ipos_x),
                    Scalar[DTYPE](b.ipos_y),
                    Scalar[DTYPE](b.ipos_z),
                ),
                iquat=(
                    Scalar[DTYPE](b.iquat_x),
                    Scalar[DTYPE](b.iquat_y),
                    Scalar[DTYPE](b.iquat_z),
                    Scalar[DTYPE](b.iquat_w),
                ),
            )

        # Joints — use Model.add_hinge_joint / add_slide_joint API
        for j in range(Self.NJOINT):
            var jd = self.joints[j].copy()
            if jd.jnt_type == JNT_HINGE:
                _ = model.add_hinge_joint(
                    jd.body_id,
                    pos=(
                        Scalar[DTYPE](jd.pos_x),
                        Scalar[DTYPE](jd.pos_y),
                        Scalar[DTYPE](jd.pos_z),
                    ),
                    axis=(
                        Scalar[DTYPE](jd.axis_x),
                        Scalar[DTYPE](jd.axis_y),
                        Scalar[DTYPE](jd.axis_z),
                    ),
                    range_min=Scalar[DTYPE](jd.range_min),
                    range_max=Scalar[DTYPE](jd.range_max),
                    armature=Scalar[DTYPE](jd.armature),
                    damping=Scalar[DTYPE](jd.damping),
                    stiffness=Scalar[DTYPE](jd.stiffness),
                )
            else:  # JNT_SLIDE
                _ = model.add_slide_joint(
                    jd.body_id,
                    pos=(
                        Scalar[DTYPE](jd.pos_x),
                        Scalar[DTYPE](jd.pos_y),
                        Scalar[DTYPE](jd.pos_z),
                    ),
                    axis=(
                        Scalar[DTYPE](jd.axis_x),
                        Scalar[DTYPE](jd.axis_y),
                        Scalar[DTYPE](jd.axis_z),
                    ),
                    armature=Scalar[DTYPE](jd.armature),
                    damping=Scalar[DTYPE](jd.damping),
                    stiffness=Scalar[DTYPE](jd.stiffness),
                )


# =============================================================================
# Inline HalfCheetah XML
# =============================================================================


comptime half_cheetah_xml = """
<mujoco model="cheetah">
  <default>
    <joint armature=".1" damping=".01" limited="true" stiffness="8"/>
    <geom conaffinity="0" condim="3" contype="1" friction=".4 .1 .1"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
  </default>
  <worldbody>
    <geom name="floor" pos="0 0 0" type="plane" size="40 40 40"/>
    <body name="torso" pos="0 0 .7">
      <joint armature="0" axis="1 0 0" damping="0" limited="false" name="rootx" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 0 1" damping="0" limited="false" name="rootz" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="false" name="rooty" pos="0 0 0" stiffness="0" type="hinge"/>
      <geom fromto="-.5 0 0 .5 0 0" name="torso" size="0.046" type="capsule"/>
      <body name="bthigh" pos="-.5 0 0">
        <joint axis="0 1 0" damping="6" name="bthigh" pos="0 0 0" range="-.52 1.05" stiffness="240" type="hinge"/>
        <geom name="bthigh" size="0.046 .145" type="capsule"/>
        <body name="bshin" pos=".16 0 -.25">
          <joint axis="0 1 0" damping="4.5" name="bshin" pos="0 0 0" range="-.785 .785" stiffness="180" type="hinge"/>
          <geom name="bshin" size="0.046 .15" type="capsule"/>
          <body name="bfoot" pos="-.28 0 -.14">
            <joint axis="0 1 0" damping="3" name="bfoot" pos="0 0 0" range="-.4 .785" stiffness="120" type="hinge"/>
            <geom name="bfoot" size="0.046 .094" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="fthigh" pos=".5 0 0">
        <joint axis="0 1 0" damping="4.5" name="fthigh" pos="0 0 0" range="-1 .7" stiffness="180" type="hinge"/>
        <geom name="fthigh" size="0.046 .133" type="capsule"/>
        <body name="fshin" pos="-.14 0 -.24">
          <joint axis="0 1 0" damping="3" name="fshin" pos="0 0 0" range="-1.2 .87" stiffness="120" type="hinge"/>
          <geom name="fshin" size="0.046 .106" type="capsule"/>
          <body name="ffoot" pos=".13 0 -.18">
            <joint axis="0 1 0" damping="1.5" name="ffoot" pos="0 0 0" range="-.5 .5" stiffness="60" type="hinge"/>
            <geom name="ffoot" size="0.046 .07" type="capsule"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor gear="120" joint="bthigh" name="bthigh"/>
    <motor gear="90" joint="bshin" name="bshin"/>
    <motor gear="60" joint="bfoot" name="bfoot"/>
    <motor gear="120" joint="fthigh" name="fthigh"/>
    <motor gear="60" joint="fshin" name="fshin"/>
    <motor gear="30" joint="ffoot" name="ffoot"/>
  </actuator>
</mujoco>
"""


fn main():
    # =========================================================================
    # Q1: Does InlineArray[BodyData, N] work with a custom struct?
    # =========================================================================
    var flat_runtime = FlatModelDef[8, 9, 9, 9, 9, 6]()

    # Mutate one body to verify indexing works
    flat_runtime.bodies[0] = BodyData(parent=0, mass=14.0, pos_z=0.7)
    print("[Q1] InlineArray[BodyData, 8] created and indexed")
    print("     bodies[0].mass =", flat_runtime.bodies[0].mass, " (expected 14.0)")
    print("     bodies[0].pos_z =", flat_runtime.bodies[0].pos_z, " (expected 0.7)")
    print()

    # =========================================================================
    # Q2: Can pm.NBODY (field of comptime ParsedModel) be a struct type param?
    # =========================================================================
    comptime pm = parse_xml(half_cheetah_xml)

    # THE key experiment: use pm.NBODY directly as a struct type parameter
    comptime flat_ct = FlatModelDef[pm.NBODY, pm.NJOINT, pm.NQ, pm.NV, pm.NGEOM, pm.NACT]()

    print("[Q2] FlatModelDef[pm.NBODY, ...] instantiated at comptime ✓")
    print("     pm.NBODY =", pm.NBODY, "| pm.NJOINT =", pm.NJOINT)

    @parameter
    if pm.NBODY == 8 and pm.NJOINT == 9:
        print("     @parameter branch: NBODY==8, NJOINT==9 confirmed at comptime ✓")
    print()

    # =========================================================================
    # Q3: Does setup_model work via InlineArray for loop → Model → FK?
    # =========================================================================
    # Build minimal HalfCheetah-shaped FlatModelDef (just torso body + rootz joint)
    var flat = FlatModelDef[8, 9, 9, 9, 9, 6]()

    # Set torso: parent=worldbody(0), mass=14.0, pos_z=0.7
    flat.bodies[0] = BodyData(
        parent=0, mass=14.0,
        pos_x=0.0, pos_y=0.0, pos_z=0.7,
        quat_w=1.0, ixx=0.1, iyy=0.1, izz=0.05,
    )

    # rootz slide joint on torso (body_id=1, axis=Z)
    flat.joints[0] = JointData(
        jnt_type=JNT_SLIDE,
        body_id=1, nq=1, nv=1,
        axis_x=0.0, axis_y=0.0, axis_z=1.0,
        is_limited=False, armature=0.0, damping=0.0, stiffness=0.0,
    )

    # Create Model/Data with dimensions matching the FlatModelDef
    var model = Model[DType.float64, 9, 9, 8, 9, 10, 9, 0, ConeType.ELLIPTIC, 0, 0]()
    var data = Data[DType.float64, 9, 9, 8, 9, 10, 0]()

    # Call setup_model — uses a regular for loop over InlineArray
    flat.setup_model[DType.float64, 10](model)

    print("[Q3] setup_model via InlineArray for loop:")
    print("     body 1 mass =", Float64(model.body_mass[1]), " (expected 14.0)")
    print("     body 1 pos_z =", Float64(model.body_pos[1 * 3 + 2]), " (expected 0.7)")
    print()

    # Run FK to verify the model is usable with the existing physics engine
    forward_kinematics(model, data)
    print("[Q3] forward_kinematics ran on InlineArray-setup model ✓")
    print("     torso xpos_z =", Float64(data.xpos[1 * 3 + 2]), " (expected ~0.7)")
