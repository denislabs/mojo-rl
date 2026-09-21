# math3d/ - 3D Mathematics Library

Core 3D math types used by the physics3d engine and 3D renderer.

## Files

| File | Types | Description |
|------|-------|-------------|
| `vec3.mojo` | `Vec3` | 3D vector with arithmetic, dot, cross, normalize, length |
| `quat.mojo` | `Quat` | Unit quaternion for 3D rotations (mul, rotate, slerp, to_mat3) |
| `mat3.mojo` | `Mat3` | 3x3 matrix (rotation, inverse, determinant, from_quat) |
| `mat4.mojo` | `Mat4` | 4x4 affine transform (perspective, look_at, translate, scale) |
| `math_gpu.mojo` | GPU variants | GPU-accelerated versions of the above operations |
