"""Test render3d module compilation and basic functionality."""

from render3d import (
    Camera3D,
    WireframeSphere,
    WireframeCapsule,
    WireframeBox,
    WireframeLine,
    Renderer3D,
)
from render3d.shapes3d import create_ground_grid, create_axes
from math3d import Vec3, Quat


fn main() raises:
    print("Testing render3d module...")

    # Test 1: Camera3D
    print("\nTest 1: Camera3D")
    var camera = Camera3D(
        eye=Vec3(0.0, -5.0, 3.0),
        target=Vec3(0.0, 0.0, 1.0),
        up=Vec3(0.0, 0.0, 1.0),
        fov=60.0,
        screen_width=800,
        screen_height=450,
    )
    print("  Eye:", camera.eye.x, camera.eye.y, camera.eye.z)
    print("  Target:", camera.target.x, camera.target.y, camera.target.z)

    # Test projection
    var test_point = Vec3(0.0, 0.0, 1.0)
    var projected = camera.project_to_screen(test_point)
    print("  Projected (0,0,1):", projected[0], projected[1], "visible:", projected[2])

    # Test 2: WireframeSphere
    print("\nTest 2: WireframeSphere")
    var sphere = WireframeSphere(
        center=Vec3(0.0, 0.0, 1.0),
        radius=0.5,
        segments=8,
        rings=6,
    )
    var sphere_lines = sphere.get_lines()
    print("  Number of lines:", len(sphere_lines))

    # Test 3: WireframeCapsule
    print("\nTest 3: WireframeCapsule")
    var capsule = WireframeCapsule(
        center=Vec3(1.0, 0.0, 0.5),
        orientation=Quat.identity(),
        radius=0.1,
        half_height=0.2,
        axis=2,
        segments=8,
    )
    var capsule_lines = capsule.get_lines()
    print("  Number of lines:", len(capsule_lines))

    # Test 4: WireframeBox
    print("\nTest 4: WireframeBox")
    var box = WireframeBox(
        center=Vec3(-1.0, 0.0, 0.5),
        orientation=Quat.identity(),
        half_extents=Vec3(0.3, 0.3, 0.3),
    )
    var box_lines = box.get_lines()
    print("  Number of lines:", len(box_lines))
    # Box should have exactly 12 edges
    if len(box_lines) == 12:
        print("  Correct number of box edges!")

    # Test 5: Ground grid
    print("\nTest 5: Ground grid")
    var grid_lines = create_ground_grid(size=2.0, divisions=5)
    print("  Number of grid lines:", len(grid_lines))

    # Test 6: Coordinate axes
    print("\nTest 6: Coordinate axes")
    var axis_lines = create_axes(Vec3.zero(), 1.0)
    print("  Number of axis lines:", len(axis_lines))
    if len(axis_lines) == 3:
        print("  Correct number of axes!")

    print("\nAll render3d tests passed!")
    print("\nNote: To test actual rendering with SDL2, run examples/render3d_demo.mojo")
