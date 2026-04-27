"""Minimal test: load and render the exact MetaWorld table STL files directly."""

from mojo_rl.render import Renderer3D, Camera3D, Color
from mojo_rl.render.light import Light
from mojo_rl.render.stl_loader import load_stl
from mojo_rl.math3d import Vec3 as V3, Quat as Q4

comptime Vec3 = V3[DType.float64]
comptime Quat = Q4[DType.float64]


def main() raises:
    # Step 1: just load the STL files
    print("Loading tablebody.stl...")
    var mesh1 = load_stl(
        "mojo_rl/envs/metaworld/assets/meshes/table/tablebody.stl"
    )
    print(
        "  vertices:",
        mesh1.vertices.byte_length(),
        " indices:",
        mesh1.indices.byte_length(),
    )

    print("Loading tabletop.stl...")
    var mesh2 = load_stl(
        "mojo_rl/envs/metaworld/assets/meshes/table/tabletop.stl"
    )
    print(
        "  vertices:",
        mesh2.vertices.byte_length(),
        " indices:",
        mesh2.indices.byte_length(),
    )

    print("STL loading OK!")

    # Step 2: create renderer
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
    var title = String("MetaWorld Table Mesh Test")
    renderer.init(title)
    print("Renderer open. Close window to exit.")

    # Step 3: render loop
    var step = 0
    while not renderer.check_quit() and step < 300:
        renderer.begin_frame()

        # Draw tablebody mesh at the MetaWorld table position
        renderer.draw_mesh(
            name="tablebody",
            file_path=(
                "mojo_rl/envs/metaworld/assets/meshes/table/tablebody.stl"
            ),
            center=Vec3(0.0, 0.6, -0.65),
            orientation=Quat(0.0, 0.0, 0.0, 1.0),
            color=Color(150, 100, 50, 255),
        )

        # Draw a reference box at table surface
        renderer.draw_box(
            center=Vec3(0.0, 0.6, -0.027),
            orientation=Quat(0.0, 0.0, 0.0, 1.0),
            half_extents=Vec3(0.7, 0.4, 0.027),
            color=Color(150, 100, 50, 128),
        )

        renderer.end_frame()
        renderer.delay_ms(16)
        step += 1

    renderer.close()
    print("Done!", step, "frames")
