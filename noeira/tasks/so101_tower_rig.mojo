"""The so101_tower rig as a vision policy sees it — pixels and units, written once.

    from noeira.tasks.so101_tower_rig import (
        TowerRenderer, make_tower_renderer, tower_cameras, pack_camera_u8,
        So101TowerUnits,
    )

Two tools produce or consume the student's observation and must agree to the
byte: `examples/so101/tower_demo_rerender.mojo` renders the TRAINING frames
from recorded states, and `examples/so101/tower_act_eval.mojo` renders the
frames the student acts on in the closed loop. A picture or a unit spelled
twice is `_a_rule_written_inline_twice_drifts`, and its failure here would be
silent: a student that scores well on the store and worse in the loop, with
nothing raising. So both import these.

## The pixels

The env's own model shape (`TOWER_MD`), 320x240 (`act/config`'s
`SO101_IMG_W` / `SO101_IMG_H`, the real import's working resolution), 4x
MSAA, groups 0+2 (props + the arm's and stand's visual meshes; group 3 is the
collision set), the preview's background, overhead in SLOT 0 and wrist in
SLOT 1 (the real dataset's sorted video-key order). A frame is packed uint8
CHW, top row first, `round(x * 255)` clamped — `pack_camera_u8`.

⚠ THE POSE IS HOST FK OF `qpos`, NOT THE ENV'S DEVICE `xpos`. The tower
config leaves `SYNC_FK_AFTER_STEP` off, so after a step the env's derived
poses describe the state one substep BEFORE the integrated `qpos`. The store
was rendered from `forward_kinematics` of the recorded `qpos`; the loop does
the same into its own `Data`, so frame t is the picture of state t in both.

## The units (`So101TowerUnits`)

The store's `qpos` / `action` are LeRobot's: the five body joints in DEGREES,
the gripper 0..100 by FRACTION of its range (`robot/so101/sim_map.mojo`, sign
+1, offset 0). The env's action is each joint TARGET normalised onto the
actuator's `ctrlrange` to [-1, 1]. Both directions of both maps live here.
"""

from std.math import pi

from max.gpu.host import DeviceContext
from noeira.math3d import Vec3 as Vec3Generic

from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.physics3d.fields import Model, actuator_column
from noeira.physics3d.gpu.constants import ACT_IDX_CTRL_MAX, ACT_IDX_CTRL_MIN
from noeira.physics3d.parser.flat_model import FlatModelDef
from noeira.physics3d.raytrace import BatchedCameraRenderer
from noeira.physics3d.raytrace.visual import build_visual_model
from noeira.tasks.family_config import So101TowerConfig
from noeira.tasks.so101_tower_xml import So101TowerModel


comptime RIG_DT = DType.float32
"""The device's type — the batched env's `Data` is float32."""
comptime TOWER_MD = Phyics3dEnv[So101TowerModel, So101TowerConfig, RIG_DT, False].MD
"""⚠ THE ENV'S OWN MODEL SHAPE, so the renderer is the kernel a
render-in-the-loop eval over the batched env instantiates."""
comptime RIG_CAM_W = 320
comptime RIG_CAM_H = 240
comptime RIG_N_CAMS = 2
comptime RIG_NPIX = RIG_CAM_W * RIG_CAM_H
comptime RIG_CAM_ELEMS = 3 * RIG_NPIX
comptime RIG_IMG_ELEMS = RIG_N_CAMS * RIG_CAM_ELEMS
comptime RIG_SAMPLES = 4
"""MuJoCo's `offsamples`: 4x MSAA. One ray per pixel aliases the jaw and the
brick's edges, the two things the student must localise."""
comptime RIG_VISUAL_GROUP_MASK: Int = (1 << 0) | (1 << 2)
comptime RIG_BACKGROUND_R: Float64 = 0.82
comptime RIG_BACKGROUND_G: Float64 = 0.86
comptime RIG_BACKGROUND_B: Float64 = 0.90
comptime RIG_ACT = 6
comptime RIG_GRIPPER = 5
comptime RIG_DR_TARGET = (0.32, 0.0, 0.0)
"""What `--dr`'s extra spot lights aim at: the desk mat's centre
(`scenes/so101_tower.xml`, the `desk_mat` frame)."""

comptime TowerRenderer[LANES: Int] = BatchedCameraRenderer[
    RIG_DT, TOWER_MD, LANES, RIG_CAM_W, RIG_CAM_H, False, True, RIG_SAMPLES
]


def _camera(names: List[String], suffix: String) raises -> Int:
    for i in range(len(names)):
        if String(names[i]).endswith(suffix):
            return i
    raise Error("so101 tower rig: no camera named *" + suffix + " in the scene")


def tower_cameras(fmd: FlatModelDef) raises -> List[Int]:
    """`[overhead, wrist]` — slot 0, slot 1."""
    var cams = List[Int]()
    cams.append(_camera(fmd.camera_names, String("overhead_cam")))
    cams.append(_camera(fmd.camera_names, String("wrist_cam")))
    return cams^


def rig_background() -> Vec3Generic[RIG_DT]:
    return Vec3Generic[RIG_DT](
        Scalar[RIG_DT](RIG_BACKGROUND_R), Scalar[RIG_DT](RIG_BACKGROUND_G),
        Scalar[RIG_DT](RIG_BACKGROUND_B),
    )


def make_tower_model(ctx: DeviceContext) raises -> Model[RIG_DT, TOWER_MD]:
    """The rig's model, on the device — what the renderer reads."""
    var m = Model[RIG_DT, TOWER_MD]()
    So101TowerModel.init_fields[RIG_DT](ctx, m)
    m.upload_all(ctx)
    return m^


def make_tower_renderer[
    LANES: Int
](
    ctx: DeviceContext, fmd: FlatModelDef, mut m: Model[RIG_DT, TOWER_MD]
) raises -> TowerRenderer[LANES]:
    """The renderer with the rig's visual set and background, camera slot 0."""
    var cams = tower_cameras(fmd)
    var r = TowerRenderer[LANES](ctx, m, cams[0])
    r.set_visual(
        ctx,
        build_visual_model[RIG_DT, TOWER_MD](
            fmd, m, group_mask=RIG_VISUAL_GROUP_MASK
        ),
    )
    r.background = rig_background()
    return r^


@always_inline
def rig_byte(x: Float64) -> UInt8:
    """A tracer float in [0, 1] to the byte the store holds."""
    var v = Int(x * 255.0 + 0.5)
    if v < 0:
        v = 0
    if v > 255:
        v = 255
    return UInt8(v)


def pack_camera_u8(
    src: Pointer[Scalar[RIG_DT], MutAnyOrigin],
    lane: Int,
    dst: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    dst_off: Int,
) -> Bool:
    """Lane `lane` of the renderer's interleaved `rgb` (copied to the host,
    `mptr(host_buffer.unsafe_ptr())`) ->
    uint8 CHW at `dst + dst_off`. Returns True when every byte is the same
    (a flat picture: a camera inside a mesh, a pose that never arrived)."""
    var base = lane * RIG_NPIX * 3
    var first = rig_byte(Float64(src[unsafe_offset=base]))
    var all_same = True
    for q in range(RIG_NPIX):
        for c in range(3):
            var b = rig_byte(Float64(src[unsafe_offset = base + q * 3 + c]))
            dst[unsafe_offset = dst_off + c * RIG_NPIX + q] = b
            if b != first:
                all_same = False
    return all_same


struct So101TowerUnits(Copyable, Movable):
    """LeRobot units <-> the model's joint values <-> the env's action word.

    `lo` / `hi` are the actuators' `ctrlrange`, read from the MODEL (for a
    `<position>` servo it is the joint's range) — never a copy of the numbers.
    """

    var lo: List[Float64]
    var hi: List[Float64]

    def __init__(out self) raises:
        var sf = So101TowerModel.make_spec_fields[DType.float64]()
        var lo_col = actuator_column(sf, ACT_IDX_CTRL_MIN, RIG_ACT)
        var hi_col = actuator_column(sf, ACT_IDX_CTRL_MAX, RIG_ACT)
        self.lo = List[Float64]()
        self.hi = List[Float64]()
        for k in range(RIG_ACT):
            self.lo.append(Float64(lo_col[k]))
            self.hi.append(Float64(hi_col[k]))

    def joint_to_lerobot(self, k: Int, q: Float64) -> Float64:
        """Radians (body) / the gripper hinge -> degrees / 0..100."""
        if k == RIG_GRIPPER:
            return 100.0 * (q - self.lo[k]) / (self.hi[k] - self.lo[k])
        return q * 180.0 / pi

    def lerobot_to_joint(self, k: Int, v: Float64) -> Float64:
        if k == RIG_GRIPPER:
            return self.lo[k] + v / 100.0 * (self.hi[k] - self.lo[k])
        return v * pi / 180.0

    def action_to_joint(self, k: Int, a: Float64) -> Float64:
        """The env's normalised action word -> the joint target it commands."""
        return self.lo[k] + (a + 1.0) * 0.5 * (self.hi[k] - self.lo[k])

    def joint_to_action(self, k: Int, q: Float64) -> Float64:
        """A joint target -> the env's action word, NOT clamped (the caller
        counts saturations before clamping)."""
        return 2.0 * (q - self.lo[k]) / (self.hi[k] - self.lo[k]) - 1.0
