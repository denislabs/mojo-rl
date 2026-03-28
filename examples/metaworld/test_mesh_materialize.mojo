"""Test: comptime String from InlineArray → draw_mesh."""

from std.collections import InlineArray
from mojo_rl.render import Renderer3D, Camera3D, Color
from mojo_rl.render.light import Light
from mojo_rl.math3d import Vec3 as V3, Quat as Q4

comptime Vec3 = V3[DType.float64]
comptime Quat = Q4[DType.float64]


def build_names() -> InlineArray[String, 2]:
    var a = InlineArray[String, 2](fill=String(""))
    a[0] = "tablebody"
    a[1] = "tabletop"
    return a^


def build_files() -> InlineArray[String, 2]:
    var a = InlineArray[String, 2](fill=String(""))
    a[0] = "mojo_rl/envs/metaworld/assets/meshes/table/tablebody.stl"
    a[1] = "mojo_rl/envs/metaworld/assets/meshes/table/tabletop.stl"
    return a^


comptime mesh_names = build_names()
comptime mesh_files = build_files()


def main() raises:
    # Test 1: access comptime InlineArray[String] at runtime
    print("Step 1: access comptime InlineArray[String]...")
    comptime for i in range(2):
        comptime n: String = mesh_names[i]
        comptime f: String = mesh_files[i]
        print("  mesh[", i, "]:", n, "→", f)
    print("OK!")

    # Test 2: render
    var cam = Camera3D(
        eye=Vec3(0.0, -1.5, 0.8),
        target=Vec3(0.0, 0.6, 0.0),
        up=Vec3(0.0, 0.0, 1.0),
        fov=45.0, aspect=16.0 / 9.0,
        screen_width=1280, screen_height=720,
    )
    var lights = List[Light]()
    lights.append(Light(
        mode=0, dir_x=0.5, dir_y=0.5, dir_z=-1.0,
        color_r=0.7, color_g=0.7, color_b=0.7,
        ambient=0.3, specular_intensity=0.3,
        specular_exponent=10.0, cast_shadow=True,
    ))
    var renderer = Renderer3D(
        width=1280, height=720, camera=cam,
        draw_grid=True, draw_axes=True, lights=lights,
    )
    var title = String("InlineArray Mesh Test")
    renderer.init(title)
    print("Renderer open.")

    var step = 0
    while not renderer.check_quit() and step < 300:
        renderer.begin_frame()

        # Use comptime for to unroll mesh draws
        comptime for mi in range(2):
            comptime n: String = mesh_names[mi]
            comptime f: String = mesh_files[mi]
            renderer.draw_mesh(
                name=n, file_path=f,
                center=Vec3(0.0, 0.6, -0.65),
                orientation=Quat(0.0, 0.0, 0.0, 1.0),
                color=Color(150, 100, 50, 255),
            )

        renderer.end_frame()
        renderer.delay_ms(16)
        step += 1

    renderer.close()
    print("Done!", step, "frames")
