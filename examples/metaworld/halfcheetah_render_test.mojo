"""HalfCheetah render test — same pattern as sawyer_reach_demo.mojo.

If this works but Sawyer doesn't, the issue is model-specific.
If this also fails, the issue is in the script pattern.

Run with:
    pixi run -e apple mojo run -I . examples/metaworld/halfcheetah_render_test.mojo
"""

from std.collections import InlineArray
from std.random import seed, random_float64
from std.time import perf_counter_ns

from mojo_rl.physics3d.types import Model, Data, ConeType
from mojo_rl.physics3d.integrator import ImplicitFastIntegrator
from mojo_rl.physics3d.solver import NewtonSolver
from mojo_rl.physics3d.kinematics import forward_kinematics
from mojo_rl.physics3d.model.model_renderer import ModelRenderer
from mojo_rl.envs.half_cheetah.half_cheetah_xml import HalfCheetahModel
from mojo_rl.physics3d.parser import parse_xml


comptime DTYPE = DType.float64
comptime hc_xml = HalfCheetahModel.xml
comptime pm = parse_xml(hc_xml)
comptime NUM_STEPS = 500


def main() raises:
    seed(42)
    print("=" * 60)
    print("HalfCheetah Render Test — Same pattern as Sawyer demo")
    print("=" * 60)
    print("Bodies:", pm.NBODY, " Joints:", pm.NJOINT,
          " NQ:", pm.NQ, " NV:", pm.NV, " Geoms:", pm.NGEOM)
    print()

    var model = Model[
        DTYPE, pm.NQ, pm.NV, pm.NBODY, pm.NJOINT,
        20, pm.NGEOM, 0,
        ConeType.PYRAMIDAL, 0, pm.NSITE,
    ]()
    var data = Data[
        DTYPE, pm.NQ, pm.NV, pm.NBODY, pm.NJOINT,
        20, pm.NSITE,
    ]()
    HalfCheetahModel.setup_model_and_data[DTYPE](model, data)
    forward_kinematics(model, data)
    print("Model ready.")

    var renderer = ModelRenderer[HalfCheetahModel](
        width=1280, height=720,
        visual_radius_scale=1.0, axes_offset=1.5,
        vel_arrow_height=0.0, vel_arrow_scale=0.0,
    )
    renderer.init()
    print("Renderer open. Close window to exit.\n")

    var step = 0
    while step < NUM_STEPS and renderer.is_open():
        for _ in range(5):
            ImplicitFastIntegrator[SOLVER=NewtonSolver].step(
                model, data, verbose=False)
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

        if step % 100 == 0:
            print("Step", step)
        step += 1

    renderer.close()
    print("\nDone!", step, "steps")
