"""Test: STL mesh with PNG texture."""

from mojo_rl.render import Renderer3D, Camera3D, Color
from mojo_rl.render.light import Light
from mojo_rl.math3d import Vec3 as V3, Quat as Q4

comptime Vec3 = V3[DType.float64]
comptime Quat = Q4[DType.float64]


def main() raises:
    var cam = Camera3D(
        eye=Vec3(0.0, -1.5, 0.8),
        target=Vec3(0.0, 0.6, 0.0),
        up=Vec3(0.0, 0.0, 1.0),
        fov=45.0,
        aspect=16.0 / 9.0,
        screen_width=1280,
        screen_height=720,
    )
    var lights = List[Light]()
    lights.append(
        Light(
            mode=0,
            dir_x=0.5,
            dir_y=0.5,
            dir_z=-1.0,
            color_r=0.7,
            color_g=0.7,
            color_b=0.7,
            ambient=0.3,
            specular_intensity=0.3,
            specular_exponent=10.0,
            cast_shadow=True,
        )
    )
    var renderer = Renderer3D(
        width=1280,
        height=720,
        camera=cam,
        draw_grid=True,
        draw_axes=True,
        lights=lights,
    )
    var title = String("Textured Mesh Test")
    renderer.init(title)
    print("Renderer open. Close window to exit.")

    # Debug: load STL and check UVs
    from mojo_rl.render.stl_loader import load_stl

    var debug_mesh = load_stl(
        "mojo_rl/envs/metaworld/assets/meshes/table/tablebody.stl"
    )
    print("Mesh vertices:", debug_mesh.vertices.byte_length())
    for i in range(min(6, debug_mesh.vertices.byte_length())):
        var v = debug_mesh.vertices[i]
        print("  v[", i, "] pos=(", v.px, v.py, v.pz, ") uv=(", v.u, v.v, ")")

    var step = 0
    while not renderer.check_quit() and step < 500:
        renderer.begin_frame()

        # Table body WITH wood texture
        renderer.draw_mesh(
            name="tablebody",
            file_path=(
                "mojo_rl/envs/metaworld/assets/meshes/table/tablebody.stl"
            ),
            center=Vec3(0.0, 0.6, -0.65),
            orientation=Quat(0.0, 0.0, 0.0, 1.0),
            color=Color(
                255, 255, 255, 255
            ),  # White base — texture provides color
            texture_name="wood2",
            texture_path="mojo_rl/envs/metaworld/assets/textures/wood2.png",
        )

        # Table top with wood texture
        renderer.draw_box(
            center=Vec3(0.0, 0.6, -0.027),
            orientation=Quat(0.0, 0.0, 0.0, 1.0),
            half_extents=Vec3(0.7, 0.4, 0.027),
            color=Color(
                150, 100, 50, 255
            ),  # Fallback color (no texture on box)
        )

        # Untextured sphere for comparison
        renderer.draw_sphere(
            center=Vec3(0.3, 0.6, 0.05),
            radius=0.05,
            color=Color(200, 50, 50, 255),
        )

        renderer.end_frame()
        renderer.delay_ms(16)
        step += 1

    renderer.close()
    print("Done!", step, "frames")
