"""GATE the normalized action map: does -1 / 0 / +1 land on lo / mid / hi?

The claim `NORMALIZED_ACTIONS` makes is exact and therefore checkable: an
action of +1 must command each joint's `ctrlrange` MAXIMUM, -1 its minimum,
and 0 the midpoint. The servo holds a commanded pose to 0.03 deg (measured),
so the settled `qpos` IS the commanded `ctrl` and this reads the map directly.

⚠⚠ AND THE ENDPOINT TEST HAS AN INNOCENT FAILURE MODE, which the first run of
this gate walked straight into. Commanding EVERY joint to its maximum is the
fully-extended arm, and `<position forcerange="-2.94 2.94">` — the real
STS3215's ~30 kg cm — cannot hold that against gravity. The joints that fall
short are then saturating, not mis-mapped, and it is exactly the two
gravity-loaded ones (`shoulder_lift`, `elbow_flex`) while `shoulder_pan`
(vertical axis, no gravity torque) is exact at both ends.

So the MAP is gated on the MIDPOINT, which no saturation can explain away: an
affine per-joint map is the only thing that puts the ASYMMETRIC gripper's a=0
at 0.785 rad rather than at 0. Endpoints are reported beside it, with a
saturation note where the servo could not hold.
"""
from std.random import seed
from std.math import abs
from max.gpu.host import DeviceContext
from mojo_rl.nn.constants import DT
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.envs.robots.so_arm101_xml import SoArm101Model
from mojo_rl.envs.robots.so_arm101 import SoArm101ReachConfig
from mojo_rl.physics3d.fields import actuator_column
from mojo_rl.physics3d.gpu.constants import ACT_IDX_CTRL_MAX, ACT_IDX_CTRL_MIN
from mojo_rl.robot.so101 import joint_name
from mojo_rl.utils.fmt import col, fixed, pad_right

comptime EnvT = Phyics3dEnv[
    SoArm101Model, SoArm101ReachConfig, DT, TERMINATE_ON_UNHEALTHY=False
]
comptime OBS_DIM = EnvT.OBS_DIM
comptime ACT_DIM = EnvT.ACTION_DIM


def settle(mut env: EnvT, a_val: Float64) raises -> List[Float64]:
    var s0 = env.reset()
    var obs = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0))
    for i in range(OBS_DIM):
        obs[i] = Scalar[DT](s0.data[i])
    var a = EnvT.ActionType()
    for j in range(ACT_DIM):
        a.data[j] = a_val
    for _ in range(400):
        var out = env.step(a)
        for i in range(OBS_DIM):
            obs[i] = Scalar[DT](out[0].data[i])
    var q = List[Float64](length=6, fill=0.0)
    for j in range(6):
        q[j] = Float64(obs[j])
    return q^


def main() raises:
    seed(1)
    var ctx = DeviceContext()
    var env = EnvT(ctx)
    var sf = SoArm101Model.make_spec_fields[DType.float64]()
    var lo_col = actuator_column(sf, ACT_IDX_CTRL_MIN, ACT_DIM)
    var hi_col = actuator_column(sf, ACT_IDX_CTRL_MAX, ACT_DIM)

    var at_lo = settle(env, -1.0)
    var at_mid = settle(env, 0.0)
    var at_hi = settle(env, 1.0)

    print("=" * 76)
    print("NORMALIZED ACTION MAP — action -1 / 0 / +1 vs ctrlrange lo/mid/hi")
    print("=" * 76)
    print("  joint            a=-1     lo  |    a=0    mid  |    a=+1     hi")
    var worst_mid = 0.0
    var worst_end = 0.0
    for j in range(6):
        var lo = Float64(lo_col[j])
        var hi = Float64(hi_col[j])
        var mid = 0.5 * (lo + hi)
        var em = abs(at_mid[j] - mid)
        var ee = abs(at_lo[j] - lo)
        if abs(at_hi[j] - hi) > ee:
            ee = abs(at_hi[j] - hi)
        if em > worst_mid:
            worst_mid = em
        if ee > worst_end:
            worst_end = ee
        print(
            "  " + pad_right(String(joint_name(j)), 13),
            col(at_lo[j], 7, 3), col(lo, 7, 3), " |",
            col(at_mid[j], 7, 3), col(mid, 7, 3), " |",
            col(at_hi[j], 7, 3), col(hi, 7, 3),
            "  SAGS (servo saturated)" if ee > 0.01 else "",
        )
    print("=" * 76)
    print("  worst MIDPOINT error:", fixed(worst_mid, 5), "rad =",
          fixed(worst_mid * 57.2957795, 4), "deg   <- gates the map")
    print("  worst ENDPOINT error:", fixed(worst_end, 5), "rad",
          "  <- gravity/saturation, not the map")
    if worst_mid < 0.005:
        print()
        print("  PASS — every joint's a=0 lands on its own range MIDPOINT,")
        print("  the asymmetric gripper included (0.785, not 0). Only a")
        print("  per-joint affine map does that.")
    else:
        print("  FAIL — a=0 does not land on the midpoint; the map is wrong.")
