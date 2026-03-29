"""Test: Mesh support function correctness."""

from mojo_rl.physics3d.collision.gjk_support import support_mesh, support_box


def test_mesh_support() raises:
    """Test that mesh support returns correct extremal points."""
    # Simple cube mesh: 8 vertices at (±0.5, ±0.5, ±0.5)
    var verts = List[Float64]()
    for sx in range(2):
        for sy in range(2):
            for sz in range(2):
                verts.append(Float64(sx) - 0.5)
                verts.append(Float64(sy) - 0.5)
                verts.append(Float64(sz) - 0.5)

    # Mesh at origin, identity quat
    print("=== Mesh support (cube at origin) ===")

    # Support along +Z should return (*, *, 0.5)
    var s1 = support_mesh[DType.float64](
        0.0, 0.0, 1.0,  # dir = +Z
        0.0, 0.0, 0.0,  # pos = origin
        0.0, 0.0, 0.0, 1.0,  # identity quat
        verts, 0, 8)
    print("  dir=(0,0,1) → support=", Float64(s1[0]), Float64(s1[1]), Float64(s1[2]),
          "(expected z=0.5)")

    # Support along -Z should return (*, *, -0.5)
    var s2 = support_mesh[DType.float64](
        0.0, 0.0, -1.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        verts, 0, 8)
    print("  dir=(0,0,-1) → support=", Float64(s2[0]), Float64(s2[1]), Float64(s2[2]),
          "(expected z=-0.5)")

    # Support along +X
    var s3 = support_mesh[DType.float64](
        1.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        verts, 0, 8)
    print("  dir=(1,0,0) → support=", Float64(s3[0]), Float64(s3[1]), Float64(s3[2]),
          "(expected x=0.5)")

    # Now with mesh translated to (0, 0, 3)
    print("\n=== Mesh support (cube at z=3) ===")
    var s4 = support_mesh[DType.float64](
        0.0, 0.0, 1.0,
        0.0, 0.0, 3.0,  # pos = (0, 0, 3)
        0.0, 0.0, 0.0, 1.0,
        verts, 0, 8)
    print("  dir=(0,0,1) → support=", Float64(s4[0]), Float64(s4[1]), Float64(s4[2]),
          "(expected z=3.5)")

    var s5 = support_mesh[DType.float64](
        0.0, 0.0, -1.0,
        0.0, 0.0, 3.0,
        0.0, 0.0, 0.0, 1.0,
        verts, 0, 8)
    print("  dir=(0,0,-1) → support=", Float64(s5[0]), Float64(s5[1]), Float64(s5[2]),
          "(expected z=2.5)")

    # Compare with box support at same position
    print("\n=== Box support (same cube at z=3) ===")
    var b1 = support_box[DType.float64](
        0.0, 0.0, 1.0,
        0.0, 0.0, 3.0,
        0.0, 0.0, 0.0, 1.0,
        0.5, 0.5, 0.5)
    print("  dir=(0,0,1) → support=", Float64(b1[0]), Float64(b1[1]), Float64(b1[2]),
          "(expected z=3.5)")

    var b2 = support_box[DType.float64](
        0.0, 0.0, -1.0,
        0.0, 0.0, 3.0,
        0.0, 0.0, 0.0, 1.0,
        0.5, 0.5, 0.5)
    print("  dir=(0,0,-1) → support=", Float64(b2[0]), Float64(b2[1]), Float64(b2[2]),
          "(expected z=2.5)")


def main() raises:
    test_mesh_support()
