"""Sawyer Reach-v3 demo with 3D rendering and random mocap control.

Renders the Sawyer arm with random XYZ mocap position actions.

Run with:
    pixi run mojo run -I . examples/metaworld/sawyer_reach_demo.mojo
"""

from std.collections import InlineArray
from std.random import seed, random_float64
from std.time import perf_counter_ns
from std.math import sqrt

from mojo_rl.physics3d.types import Model, Data, ConeType
from mojo_rl.physics3d.integrator import ImplicitFastIntegrator
from mojo_rl.physics3d.solver import NewtonSolver
from mojo_rl.physics3d.kinematics import forward_kinematics
from mojo_rl.physics3d.model.model_renderer import ModelRenderer
from mojo_rl.envs.metaworld.sawyer_reach_xml import SawyerReachModel, pm


comptime DTYPE = DType.float64
comptime MAX_CONTACTS = 30
comptime MAX_EQUALITY = 6
comptime FRAME_SKIP = 5
comptime ACTION_SCALE: Float64 = 0.01

comptime MOCAP_LOW_X: Float64 = -0.2
comptime MOCAP_LOW_Y: Float64 = 0.5
comptime MOCAP_LOW_Z: Float64 = 0.06
comptime MOCAP_HIGH_X: Float64 = 0.2
comptime MOCAP_HIGH_Y: Float64 = 0.7
comptime MOCAP_HIGH_Z: Float64 = 0.6

comptime HAND_INIT_X: Float64 = 0.0
comptime HAND_INIT_Y: Float64 = 0.6
comptime HAND_INIT_Z: Float64 = 0.2

comptime MOCAP_BODY_IDX = 23
comptime HAND_BODY_IDX = 19
comptime NUM_STEPS = 2000


def clamp64(val: Float64, lo: Float64, hi: Float64) -> Float64:
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val


def main() raises:
    seed(42)
    print("=" * 60)
    print("Sawyer Reach-v3 Demo — 3D Rendering + Random Mocap Control")
    print("=" * 60)
    print("Bodies:", pm.NBODY, " Joints:", pm.NJOINT,
          " NQ:", pm.NQ, " NV:", pm.NV, " Geoms:", pm.NGEOM)
    print()

    var model = Model[
        DTYPE, pm.NQ, pm.NV, pm.NBODY, pm.NJOINT,
        MAX_CONTACTS, pm.NGEOM, MAX_EQUALITY,
        ConeType.ELLIPTIC, 0, pm.NSITE,
    ]()
    var data = Data[
        DTYPE, pm.NQ, pm.NV, pm.NBODY, pm.NJOINT,
        MAX_CONTACTS, pm.NSITE,
    ]()
    SawyerReachModel.setup_model_and_data[DTYPE](model, data)

    var mocap_x = HAND_INIT_X
    var mocap_y = HAND_INIT_Y
    var mocap_z = HAND_INIT_Z
    data.set_mocap_pos(MOCAP_BODY_IDX,
        Scalar[DTYPE](mocap_x), Scalar[DTYPE](mocap_y), Scalar[DTYPE](mocap_z))
    data.set_mocap_quat(MOCAP_BODY_IDX,
        Scalar[DTYPE](0), Scalar[DTYPE](1), Scalar[DTYPE](0), Scalar[DTYPE](1))
    forward_kinematics(model, data)

    print("Model ready. Initializing renderer...")
    var renderer = ModelRenderer[SawyerReachModel](
        width=1280, height=720,
        visual_radius_scale=1.0, axes_offset=1.5,
        vel_arrow_height=0.0, vel_arrow_scale=0.0,
    )
    renderer.init()
    print("Renderer open. Close window to exit.\n")

    var step = 0
    var start_ns = perf_counter_ns()

    while step < NUM_STEPS and renderer.is_open():
        var dx = random_float64(-1.0, 1.0) * ACTION_SCALE
        var dy = random_float64(-1.0, 1.0) * ACTION_SCALE
        var dz = random_float64(-1.0, 1.0) * ACTION_SCALE

        mocap_x = clamp64(mocap_x + dx, MOCAP_LOW_X, MOCAP_HIGH_X)
        mocap_y = clamp64(mocap_y + dy, MOCAP_LOW_Y, MOCAP_HIGH_Y)
        mocap_z = clamp64(mocap_z + dz, MOCAP_LOW_Z, MOCAP_HIGH_Z)

        data.set_mocap_pos(MOCAP_BODY_IDX,
            Scalar[DTYPE](mocap_x), Scalar[DTYPE](mocap_y), Scalar[DTYPE](mocap_z))

        for _ in range(FRAME_SKIP):
            ImplicitFastIntegrator[SOLVER=NewtonSolver].step(model, data, verbose=False)

        forward_kinematics(model, data)

        var xpos = InlineArray[Scalar[DTYPE], pm.NBODY * 3](uninitialized=True)
        var xquat = InlineArray[Scalar[DTYPE], pm.NBODY * 4](uninitialized=True)
        for i in range(pm.NBODY * 3):
            xpos[i] = data.xpos[i]
        for i in range(pm.NBODY * 4):
            xquat[i] = data.xquat[i]
        renderer.render_from_body_state(xpos, xquat, pm.NBODY, vel_x=0.0)
        renderer.delay(16)  # ~60 FPS
        if renderer.check_quit():
            break

        if step % 200 == 0:
            var hand_x = Float64(data.xpos[HAND_BODY_IDX * 3])
            var hand_y = Float64(data.xpos[HAND_BODY_IDX * 3 + 1])
            var hand_z = Float64(data.xpos[HAND_BODY_IDX * 3 + 2])
            print("Step", step, " mocap=(",
                mocap_x.__round__(3), mocap_y.__round__(3), mocap_z.__round__(3),
                ") hand=(", hand_x.__round__(3), hand_y.__round__(3), hand_z.__round__(3), ")")

        step += 1

    renderer.close()
    var total_ms = (perf_counter_ns() - start_ns) / 1_000_000
    print("\nDone!", step, "steps in", Int(total_ms), "ms",
          " FPS:", Int(Float64(step) / (Float64(total_ms) / 1000.0)))
