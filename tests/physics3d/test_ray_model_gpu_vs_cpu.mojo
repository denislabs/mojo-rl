"""`ray_model` on the GPU against the same call on the CPU.

    pixi run mojo run -I . tests/physics3d/test_ray_model_gpu_vs_cpu.mojo

⚠⚠ THIS IS THE GATE THE RAY LIBRARY WAS UNIFIED FOR, AND IT IS NOT A
FORMALITY. `physics3d/ray` is ONE implementation over `LayoutTensor` precisely
so a kernel and the host run the same code — but "the same code" is exactly
what Metal has silently miscompiled FOUR times in this engine, always the same
way: a per-thread array indexed by a RUNTIME value reads back the wrong value,
with no crash and no diagnostic. `87960e10` is the most recent and its message
says the fix is the storage class, not the algorithm.

Two such arrays were removed from `ray/` before this file existed —
`RayBoxHit.all` (six face distances, written at `2*axis + side`, read at
`all[i]`, on the path of EVERY heightfield ray) and `ray_model`'s six-boolean
`group_mask`. Neither could fail on the CPU. This gate is what says they are
actually gone, and what will catch the fifth one.

⚠ float32 ON BOTH LEGS. Metal rejects `double` outright, so the CPU leg is
built at float32 too — comparing a float32 GPU answer against a float64 CPU
answer would report the DTYPE as a GPU defect. The residual here is therefore
the two targets' arithmetic, not their precision.

⚠ THE SCENE CARRIES EVERY GEOM TYPE, including the heightfield and the mesh,
because the traversal's job is to DISPATCH and a dispatch is only tested by the
branch it gets wrong. A ray that only ever meets spheres proves nothing about
the `ray_hfield` call three branches down.

⚠ `geomid` IS COMPARED, NOT JUST THE DISTANCE. Three of the five defects
`test_ray_model_vs_mujoco` was falsified against left `max |dt|` untouched —
they showed only as a different geom at a different depth. A GPU miscompute has
the same shape: it returns a plausible number for the wrong surface.
"""

from std.math import abs, sqrt
from std.sys import has_accelerator
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite
from layout import Layout, LayoutTensor

from mojo_rl.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric
from mojo_rl.physics3d.fields import Data, Model, Dims, init_hfield_data
from mojo_rl.physics3d.parser.full_parser import parse_xml_full
from mojo_rl.physics3d.parser.fields_build import build_model_fields_from_flat
from mojo_rl.physics3d.kinematics.forward_kinematics import forward_kinematics
from mojo_rl.physics3d.gpu.constants import (
    MODEL_GEOM_SIZE,
    MODEL_BODY_SIZE,
    MODEL_MESH_META_SIZE,
    MAX_GPU_MESHES,
    MODEL_HFIELD_META_SIZE,
    MAX_GPU_HFIELDS,
)
from mojo_rl.physics3d.ray import ray_model
from mojo_rl.nn.core.tensor import TensorImpl

# ⚠ FLOAT32: Metal rejects `double`, so both legs run at float32 — see the
# module docstring on why a float64 CPU baseline would be the wrong control.
comptime GT = DType.float32
comptime NRAY = 512
comptime TPB = 64

comptime NQ = 7
comptime NV = 6
comptime NBODY = 2
comptime NJOINT = 1
comptime NGEOM = 11
comptime NHF = 64
comptime NTRI = 64

comptime GD = Dims[
    nq=NQ,
    nv=NV,
    nbody=NBODY,
    njoint=NJOINT,
    ngeom=NGEOM,
    nsite=0,
    max_contacts=16,
    nmesh_verts=256,
    nmesh_tri=NTRI,
    nhfield_data=NHF,
]

comptime L_GEOMS = Layout.row_major(NGEOM, MODEL_GEOM_SIZE)
comptime L_BODIES = Layout.row_major(NBODY, MODEL_BODY_SIZE)
comptime L_XPOS = Layout.row_major(1, NBODY * 3)
comptime L_XQUAT = Layout.row_major(1, NBODY * 4)
comptime L_MESH_META = Layout.row_major(MAX_GPU_MESHES * MODEL_MESH_META_SIZE)
comptime L_TRI = Layout.row_major(NTRI * 9)
comptime L_HF_META = Layout.row_major(MAX_GPU_HFIELDS * MODEL_HFIELD_META_SIZE)
comptime L_HF = Layout.row_major(NHF)
comptime L_RAYS = Layout.row_major(NRAY, 6)
comptime L_OUT = Layout.row_major(NRAY, 5)

comptime SCENE = String(
    """
<mujoco model="ray gpu gate">
  <asset>
    <hfield name="terrain" file="tests/physics3d/assets/hf_8x8.bin" size="0.4 0.4 0.15 0.05"/>
    <mesh name="notch" file="tests/physics3d/assets/notch.stl"/>
    <material name="ghost" rgba="0.9 0.2 0.2 0"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 -0.6"/>
    <geom name="terrain" type="hfield" hfield="terrain" pos="0.55 0.35 -0.35" euler="8 -6 20"/>
    <geom name="target_sphere" type="sphere" size="0.09" pos="0 0 0"/>
    <geom name="a_box" type="box" size="0.07 0.05 0.06" pos="-0.45 0.20 0.05" euler="20 -35 15"/>
    <geom name="a_capsule" type="capsule" size="0.04 0.09" pos="0.40 -0.30 0.10" euler="55 10 -20"/>
    <geom name="a_cylinder" type="cylinder" size="0.05 0.07" pos="-0.35 -0.35 0.12" euler="-25 40 5"/>
    <geom name="an_ellipsoid" type="ellipsoid" size="0.09 0.04 0.06" pos="0.30 0.42 -0.05" euler="10 25 -40"/>
    <geom name="a_mesh" type="mesh" mesh="notch" pos="-0.10 0.50 0.08" euler="15 -25 40"/>
    <geom name="invisible_front" type="sphere" size="0.05" pos="0 0 0.45" rgba="0.2 0.7 0.2 0"/>
    <geom name="matinvis_front" type="sphere" size="0.05" pos="0 0 0.30" material="ghost" rgba="0.2 0.2 0.9 1"/>
    <body name="excl" pos="0 0 0.60">
      <freejoint/>
      <geom name="exclude_me" type="sphere" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""
)


struct Lcg(Copyable, Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def u01(mut self) -> Float64:
        self.s = self.s * 1664525 + 1013904223
        return Float64((self.s >> 16) & 0xFFFFFFF) / Float64(0x10000000)

    def sym(mut self, a: Float64) -> Float64:
        return (self.u01() * 2.0 - 1.0) * a


def _ray_kernel[
    T: DType
](
    geoms: LayoutTensor[T, L_GEOMS, MutAnyOrigin],
    bodies: LayoutTensor[T, L_BODIES, MutAnyOrigin],
    xpos: LayoutTensor[T, L_XPOS, MutAnyOrigin],
    xquat: LayoutTensor[T, L_XQUAT, MutAnyOrigin],
    mesh_meta: LayoutTensor[T, L_MESH_META, MutAnyOrigin],
    mesh_tris: LayoutTensor[T, L_TRI, MutAnyOrigin],
    hf_meta: LayoutTensor[T, L_HF_META, MutAnyOrigin],
    hf_data: LayoutTensor[T, L_HF, MutAnyOrigin],
    rays: LayoutTensor[T, L_RAYS, MutAnyOrigin],
    out_t: LayoutTensor[T, L_OUT, MutAnyOrigin],
) where T.is_floating_point():
    """One thread per ray.

    ⚠ PARAMETERIZED ON `T` FOR A COMPILATION REASON, not a generality one. A
    kernel with NO parameters is instantiated for the HOST as well as the
    device, and `block_dim`/`thread_idx` then fail with "current compilation
    target does not support `_get_intrinsic_name`". Every kernel in this
    engine carries parameters for the same reason.

    ⚠ No shared memory and no per-thread arrays — that `ray/` holds neither is
    the property this file exists to check.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= NRAY:
        return
    var eye = Vec3Generic[T](
        rebind[Scalar[T]](rays[i, 0]),
        rebind[Scalar[T]](rays[i, 1]),
        rebind[Scalar[T]](rays[i, 2]),
    )
    var vec = Vec3Generic[T](
        rebind[Scalar[T]](rays[i, 3]),
        rebind[Scalar[T]](rays[i, 4]),
        rebind[Scalar[T]](rays[i, 5]),
    )
    var h = ray_model[T](
        geoms, NGEOM, bodies, xpos, xquat, 0,
        mesh_meta, mesh_tris, hf_meta, hf_data, NHF,
        eye, vec, -1,
    )
    out_t[i, 0] = h.t
    out_t[i, 1] = Scalar[T](h.geom)
    out_t[i, 2] = h.normal.x
    out_t[i, 3] = h.normal.y
    out_t[i, 4] = h.normal.z


def test_ray_model_gpu_matches_cpu() raises:
    comptime if not has_accelerator():
        print("  SKIP — no accelerator on this machine")
        return

    var ctx = DeviceContext()
    var fmd = parse_xml_full(SCENE, String("."))
    var m = Model[GT, GD]()
    build_model_fields_from_flat[GT](fmd, m)
    var d = Data[GT, GD, 1]()
    init_hfield_data(d, m)
    forward_kinematics["cpu", GT, GD, 1](d, m)

    # ── the rays, identical on both legs ──────────────────────────────────
    var rays = TensorImpl[GT].alloc(NRAY * 6)
    var rng = Lcg(0xA11CE)
    for i in range(NRAY):
        var eye: Vec3Generic[Float64.dtype]
        var aim: Vec3Generic[Float64.dtype]
        if i % 4 == 1:
            # Down the z-axis column: `exclude_me` 0.60, the two invisible
            # decoys 0.45 and 0.30, the target at 0. Exercises the filter.
            eye = Vec3Generic[DType.float64](
                rng.sym(0.02), rng.sym(0.02), 1.1 + rng.u01() * 0.4
            )
            aim = Vec3Generic[DType.float64](rng.sym(0.02), rng.sym(0.02), -0.2)
        elif i % 4 == 2:
            # Origin INSIDE a geom, pointing out — the family that caught the
            # capsule cap defect against MuJoCo.
            eye = Vec3Generic[DType.float64](
                rng.sym(0.5), rng.sym(0.5), rng.sym(0.3)
            )
            aim = Vec3Generic[DType.float64](
                eye.x + rng.sym(1.0), eye.y + rng.sym(1.0), eye.z + rng.sym(1.0)
            )
        else:
            eye = Vec3Generic[DType.float64](
                rng.sym(1.6), rng.sym(1.6), rng.sym(1.6)
            )
            aim = Vec3Generic[DType.float64](
                rng.sym(0.6), rng.sym(0.6), rng.sym(0.5)
            )
        rays.data[i * 6 + 0] = Scalar[GT](eye.x)
        rays.data[i * 6 + 1] = Scalar[GT](eye.y)
        rays.data[i * 6 + 2] = Scalar[GT](eye.z)
        rays.data[i * 6 + 3] = Scalar[GT](aim.x - eye.x)
        rays.data[i * 6 + 4] = Scalar[GT](aim.y - eye.y)
        rays.data[i * 6 + 5] = Scalar[GT](aim.z - eye.z)

    # ── the CPU leg ───────────────────────────────────────────────────────
    var cpu_t = List[Float64]()
    var cpu_g = List[Int]()
    var cpu_n = List[Float64]()
    var geoms_c = m.geoms.lt["cpu", L_GEOMS]()
    var bodies_c = m.bodies.lt["cpu", L_BODIES]()
    var xpos_c = d.xpos.lt["cpu", L_XPOS]()
    var xquat_c = d.xquat.lt["cpu", L_XQUAT]()
    var mm_c = m.mesh_meta.lt["cpu", L_MESH_META]()
    var mt_c = m.mesh_tris.lt["cpu", L_TRI]()
    var hm_c = m.hfield_meta.lt["cpu", L_HF_META]()
    var hd_c = d.hfield_data.lt["cpu", L_HF]()
    for i in range(NRAY):
        var eye = Vec3Generic[GT](
            rays.data[i * 6 + 0], rays.data[i * 6 + 1], rays.data[i * 6 + 2]
        )
        var vec = Vec3Generic[GT](
            rays.data[i * 6 + 3], rays.data[i * 6 + 4], rays.data[i * 6 + 5]
        )
        var h = ray_model[GT](
            geoms_c, NGEOM, bodies_c, xpos_c, xquat_c, 0,
            mm_c, mt_c, hm_c, hd_c, NHF, eye, vec, -1,
        )
        cpu_t.append(Float64(h.t))
        cpu_g.append(h.geom)
        cpu_n.append(Float64(h.normal.x))
        cpu_n.append(Float64(h.normal.y))
        cpu_n.append(Float64(h.normal.z))

    # ── the GPU leg ───────────────────────────────────────────────────────
    var out = TensorImpl[GT].alloc(NRAY * 5)
    m.upload_all(ctx)
    d.upload_all(ctx)
    rays.upload(ctx)
    out.upload(ctx)
    comptime BLOCKS = (NRAY + TPB - 1) // TPB
    ctx.enqueue_function[_ray_kernel[GT]](
        m.geoms.lt["gpu", L_GEOMS](),
        m.bodies.lt["gpu", L_BODIES](),
        d.xpos.lt["gpu", L_XPOS](),
        d.xquat.lt["gpu", L_XQUAT](),
        m.mesh_meta.lt["gpu", L_MESH_META](),
        m.mesh_tris.lt["gpu", L_TRI](),
        m.hfield_meta.lt["gpu", L_HF_META](),
        d.hfield_data.lt["gpu", L_HF](),
        rays.lt["gpu", L_RAYS](),
        out.lt["gpu", L_OUT](),
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )
    out.download(ctx)
    ctx.synchronize()

    # ── compare ───────────────────────────────────────────────────────────
    var hits = 0
    var split = 0
    var wrong_geom = 0
    var worst_t = 0.0
    var worst_n = 0.0
    for i in range(NRAY):
        var gt = Float64(out.data[i * 5 + 0])
        var gg = Int(Float64(out.data[i * 5 + 1]))
        var ct = cpu_t[i]
        if (ct >= 0.0) != (gt >= 0.0):
            split += 1
            continue
        if ct < 0.0:
            continue
        hits += 1
        if gg != cpu_g[i]:
            wrong_geom += 1
            continue
        worst_t = max(worst_t, abs(ct - gt))
        for k in range(3):
            worst_n = max(
                worst_n,
                abs(cpu_n[i * 3 + k] - Float64(out.data[i * 5 + 2 + k])),
            )

    print("  hits", hits, "/", NRAY)
    print("  splits        ", split)
    print("  wrong geomid  ", wrong_geom)
    print("  worst |dt|      ", worst_t)
    print("  worst |dnormal| ", worst_n)

    assert_true(
        hits > NRAY // 4,
        "only " + String(hits) + " of " + String(NRAY) + " rays hit anything"
        " — a sweep of misses agrees on both targets and proves nothing",
    )
    assert_true(
        split == 0,
        String(split) + " rays where one TARGET hit and the other missed —"
        " a per-thread miscompute reads back a wrong value, and this is the"
        " column it shows in first",
    )
    assert_true(
        wrong_geom == 0,
        String(wrong_geom) + " rays named a different geom on the GPU. ⚠ A"
        " DISTANCE COMPARISON WOULD NOT SEE THIS: the wrong surface returns a"
        " perfectly plausible number.",
    )
    # Both legs are float32 and run the same code, so this is the two targets'
    # instruction selection (FMA contraction, mostly) and nothing else.
    assert_true(worst_t < 1e-4, "worst |dt| " + String(worst_t))
    assert_true(worst_n < 1e-3, "worst |dnormal| " + String(worst_n))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
