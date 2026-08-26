"""The kernel and the host that owns its buffers.

One thread per `(env, pixel)`. No shared memory, no per-thread array, no
window, no swapchain, no draw command — the whole scene is the batched `Data`
the physics already wrote, read in place.

⚠⚠ THE KERNEL IS EXPENSIVE TO COMPILE, AND THE NUMBER IS MEASURED, NOT
GUESSED. `render_pixel` inlines `ray_model` TWICE — once for the primary ray
and once for the shadow ray — and `ray_model` is itself a dispatch over every
geom type. On Apple/Metal `test_camera_render_gpu_vs_cpu` spends ~550 s in that
one compile, against ~28 s for `test_ray_model_gpu_vs_cpu`, whose kernel
inlines `ray_model` once. The RUN is not the cost: the CPU control leg renders
the same 4 608 pixels in about a second.

⇒ if that becomes a problem, the lever is `use_shadows` as a COMPTIME
parameter rather than a runtime field, which lets a caller that does not want
shadows compile a kernel with one `ray_model` in it. It is left runtime here
because shadows are wanted by default and two instantiations is the price of
the alternative.

⚠ RGB IS FLOAT, NOT PACKED `uint32`. The reference packs ABGR into a `uint32`
and ships an unpack kernel (`render_util.unpack_rgb_kernel`) because Warp's
render context stores every camera in one buffer and memory is the binding
constraint. Here the consumer is an `nn` module that wants `[0, 1]` floats, so
packing would be a kernel to compress followed by a kernel to undo it. The
price is real and worth stating: at 1024 envs and 84x84 an RGB frame is 87 MB
at float32 against 22 MB packed. If a batch ever needs the packed form, it is
an output-format parameter on this struct and nothing else changes — the pixel
function returns a `Vec3` either way.
"""

from std.math import ceildiv
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.math3d import Vec3 as Vec3Generic

from ..fields import Data, Model
from ..fields.dims import DimsLike
from ..gpu.constants import (
    TPB,
    MODEL_GEOM_SIZE,
    MODEL_BODY_SIZE,
    MODEL_GEOM_RGBA_SIZE,
    MODEL_MESH_META_SIZE,
    MAX_GPU_MESHES,
    MODEL_HFIELD_META_SIZE,
    MAX_GPU_HFIELDS,
    MAX_GPU_CAMERAS,
    MODEL_CAM_SIZE,
    CAM_IDX_ACTIVE,
    CAM_IDX_MODE,
    CAM_IDX_FOVY,
    CAM_IDX_REF_SET,
)
from .camera import (
    camera_world_frame,
    RT_CAM_MODE_TRACK,
    RT_CAM_MODE_TRACKCOM,
)
from .render import render_pixel

comptime RGB_CHANNELS: Int = 3


@always_inline
def _pos(n: Int) -> Int:
    """`_at_least_one` — every tensor allocates one element even when unused."""
    return n if n > 0 else 1


struct BatchedCameraRenderer[
    DTYPE: DType,
    D: DimsLike,
    BATCH: Int,
    WIDTH: Int,
    HEIGHT: Int,
](Copyable, Movable):
    """RGB + depth + segmentation for one camera, over every lane.

    ⚠ ONE CAMERA PER RENDERER, BY CHOICE. The reference packs N cameras of
    different resolutions into one buffer and resolves `(camid, local pixel)`
    from a cumulative-size scan at the top of every thread. That scan is a loop
    over cameras in the innermost hot path, and it buys a case — cameras of
    DIFFERENT resolutions in one launch — that nothing here has. Two cameras
    are two renderers and two launches.
    """

    comptime NPIX: Int = Self.WIDTH * Self.HEIGHT
    comptime NGEOM_F: Int = _pos(Self.D.NGEOM)
    comptime NMESH_TRI_F: Int = _pos(Self.D.NMESH_TRI)
    comptime NHF_F: Int = _pos(Self.D.NHFIELD_DATA)

    comptime L_GEOMS = Layout.row_major(Self.NGEOM_F, MODEL_GEOM_SIZE)
    comptime L_RGBA = Layout.row_major(Self.NGEOM_F * MODEL_GEOM_RGBA_SIZE)
    comptime L_BODIES = Layout.row_major(Self.D.NBODY, MODEL_BODY_SIZE)
    comptime L_B3 = Layout.row_major(Self.BATCH, Self.D.NBODY * 3)
    comptime L_B4 = Layout.row_major(Self.BATCH, Self.D.NBODY * 4)
    comptime L_CAM = Layout.row_major(MAX_GPU_CAMERAS * MODEL_CAM_SIZE)
    comptime L_MESH_META = Layout.row_major(
        MAX_GPU_MESHES * MODEL_MESH_META_SIZE
    )
    comptime L_TRI = Layout.row_major(Self.NMESH_TRI_F * 9)
    comptime L_HF_META = Layout.row_major(
        MAX_GPU_HFIELDS * MODEL_HFIELD_META_SIZE
    )
    comptime L_HF = Layout.row_major(Self.BATCH * Self.NHF_F)
    comptime L_RGB = Layout.row_major(Self.BATCH, Self.NPIX * RGB_CHANNELS)
    comptime L_SCALARPIX = Layout.row_major(Self.BATCH, Self.NPIX)

    var rgb: DeviceBuffer[Self.DTYPE]
    """`[BATCH, HEIGHT*WIDTH*3]`, row-major, channels interleaved, in [0, 1].

    ⚠ ROW 0 IS THE TOP OF THE IMAGE — see `camera_pixel_ray`. A consumer
    writing this straight to a PNG needs no flip; one feeding a convolution
    needs to know which end is the sky."""

    var depth: DeviceBuffer[Self.DTYPE]
    """`[BATCH, HEIGHT*WIDTH]` PLANAR depth in metres; **0 means no hit**.
    See `PixelHit.depth` — the sentinel is deliberate and a normaliser must
    handle it."""

    var seg: DeviceBuffer[Self.DTYPE]
    """`[BATCH, HEIGHT*WIDTH]` geom id, or -1 for background.

    ⚠ STORED AS `DTYPE`, NOT AN INTEGER TYPE, so the whole renderer stays one
    dtype and one kernel. Geom counts here are in the hundreds, far inside
    float32's exact-integer range; a model with more than 2^24 geoms would
    have other problems first."""

    var cam: Int
    var light_dir: Vec3Generic[Self.DTYPE]
    """The direction the light TRAVELS, as `mjModel.light_dir` is. The default
    points down and slightly forward, matching the key light `Renderer3D`
    uses, so a scene looks the same way up in both pipelines."""
    var background: Vec3Generic[Self.DTYPE]
    var use_shadows: Bool

    def __init__(
        out self,
        ctx: DeviceContext,
        mut m: Model[Self.DTYPE, Self.D],
        cam: Int = 0,
    ) raises:
        """Allocate the buffers and REFUSE a camera that would render wrong.

        ⚠⚠ THE CHECKS ARE THE POINT OF HAVING A HOST ENTRY POINT AT ALL. Every
        failure this guards against renders a picture rather than an error:
        an inactive row has `fovy = 0` (a degenerate frustum, one point), a
        tracking camera with no reference sits at its body's origin looking
        along world -Z, and an out-of-range index reads whatever is in the
        next row. A kernel cannot raise, so this is the only place any of
        them can be named.
        """
        comptime assert Self.WIDTH > 0 and Self.HEIGHT > 0, (
            "BatchedCameraRenderer: WIDTH and HEIGHT must be positive."
        )
        if cam < 0 or cam >= MAX_GPU_CAMERAS:
            raise Error(
                String("BatchedCameraRenderer: camera index ") + String(cam)
                + " is outside [0, " + String(MAX_GPU_CAMERAS) + ")."
            )
        var cb = cam * MODEL_CAM_SIZE
        if m.cameras.data[cb + CAM_IDX_ACTIVE] == 0:
            raise Error(
                String("BatchedCameraRenderer: camera ") + String(cam)
                + " is not a camera this model declares — the row is padding."
                " Its `fovy` is 0, which renders a single point rather than"
                " failing. Does the MJCF have a <camera> at this index?"
            )
        var fovy = m.cameras.data[cb + CAM_IDX_FOVY]
        if not (fovy > 0):
            raise Error(
                String("BatchedCameraRenderer: camera ") + String(cam)
                + " has fovy = " + String(Float64(fovy))
                + ", a degenerate frustum."
            )
        var mode = Int(m.cameras.data[cb + CAM_IDX_MODE])
        if mode == RT_CAM_MODE_TRACK or mode == RT_CAM_MODE_TRACKCOM:
            if m.cameras.data[cb + CAM_IDX_REF_SET] == 0:
                raise Error(
                    String("BatchedCameraRenderer: camera ") + String(cam)
                    + ' is mode="track"/"trackcom", which reads the reference'
                    " pose MuJoCo's compiler bakes at qpos0, and that pose has"
                    " not been taken. Call"
                    " `raytrace.init_camera_reference(d, m)` after a reset's"
                    " forward kinematics and subtree pass. Without it the"
                    " camera renders from the body origin along world -Z,"
                    " which is a picture and not an error."
                )

        self.cam = cam
        self.rgb = ctx.enqueue_create_buffer[Self.DTYPE](
            Self.BATCH * Self.NPIX * RGB_CHANNELS
        )
        self.depth = ctx.enqueue_create_buffer[Self.DTYPE](
            Self.BATCH * Self.NPIX
        )
        self.seg = ctx.enqueue_create_buffer[Self.DTYPE](
            Self.BATCH * Self.NPIX
        )
        ctx.enqueue_memset(self.rgb, 0)
        ctx.enqueue_memset(self.depth, 0)
        ctx.enqueue_memset(self.seg, 0)
        # Down and slightly forward — `Renderer3D`'s key light direction, so a
        # scene is lit from the same side in the viewer and in the tracer.
        self.light_dir = Vec3Generic[Self.DTYPE](
            Scalar[Self.DTYPE](-0.35),
            Scalar[Self.DTYPE](-0.25),
            Scalar[Self.DTYPE](-0.90),
        )
        self.background = Vec3Generic[Self.DTYPE](
            Scalar[Self.DTYPE](0.60),
            Scalar[Self.DTYPE](0.72),
            Scalar[Self.DTYPE](0.90),
        )
        self.use_shadows = True

    def render(
        mut self,
        ctx: DeviceContext,
        mut d: Data[Self.DTYPE, Self.D, Self.BATCH],
        mut m: Model[Self.DTYPE, Self.D],
    ) raises:
        """One launch: every lane, every pixel.

        ⚠ THE CALLER OWNS FRESHNESS. This reads `xpos`/`xquat`/`subtree_com`
        as the device holds them; it does not run forward kinematics. Render
        after the step's FK, or the image is one frame stale — which for a
        camera OBSERVATION is an off-by-one in the MDP and not a visual
        artefact anyone would notice.
        """

        @parameter
        @always_inline
        def cam_kernel(
            geoms: LayoutTensor[Self.DTYPE, Self.L_GEOMS, MutAnyOrigin],
            geom_rgba: LayoutTensor[Self.DTYPE, Self.L_RGBA, MutAnyOrigin],
            bodies: LayoutTensor[Self.DTYPE, Self.L_BODIES, MutAnyOrigin],
            xpos: LayoutTensor[Self.DTYPE, Self.L_B3, MutAnyOrigin],
            xquat: LayoutTensor[Self.DTYPE, Self.L_B4, MutAnyOrigin],
            subtree_com: LayoutTensor[Self.DTYPE, Self.L_B3, MutAnyOrigin],
            cameras: LayoutTensor[Self.DTYPE, Self.L_CAM, MutAnyOrigin],
            mesh_meta: LayoutTensor[
                Self.DTYPE, Self.L_MESH_META, MutAnyOrigin
            ],
            mesh_tris: LayoutTensor[Self.DTYPE, Self.L_TRI, MutAnyOrigin],
            hfield_meta: LayoutTensor[
                Self.DTYPE, Self.L_HF_META, MutAnyOrigin
            ],
            hfield_data: LayoutTensor[Self.DTYPE, Self.L_HF, MutAnyOrigin],
            rgb_out: LayoutTensor[Self.DTYPE, Self.L_RGB, MutAnyOrigin],
            depth_out: LayoutTensor[
                Self.DTYPE, Self.L_SCALARPIX, MutAnyOrigin
            ],
            seg_out: LayoutTensor[
                Self.DTYPE, Self.L_SCALARPIX, MutAnyOrigin
            ],
            # ⚠ `Int32`, NOT `Int`. Neither `Int` nor `UInt` conforms to
            # `DevicePassable` — "use a fixed-width type" — so a plain `Int`
            # kernel operand does not compile. Same for `shadows` below.
            cam: Int32,
            lx: Scalar[Self.DTYPE],
            ly: Scalar[Self.DTYPE],
            lz: Scalar[Self.DTYPE],
            br: Scalar[Self.DTYPE],
            bg: Scalar[Self.DTYPE],
            bb: Scalar[Self.DTYPE],
            shadows: Int32,
        ):
            # ⚠ AN INTEGER, NOT A `Bool`, for the same reason.
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= Self.BATCH * Self.NPIX:
                return
            # ⚠⚠ THE POSITIVE BRANCH CARRIES THE EVIDENCE, AND AN EARLY
            # `comptime if not ...: return` DOES NOT. `camera_world_frame` and
            # `render_pixel` are both constrained to a floating-point DTYPE,
            # and the compiler accepts the calls only inside a branch that
            # PROVES the constraint — the guard-clause spelling reads
            # identically and fails with "lacking evidence to prove
            # correctness". Same shape, same lesson, as the ray hook in
            # `quadruped_escape_config`.
            comptime if Self.DTYPE.is_floating_point():
                var env = i // Self.NPIX
                var pix = i - env * Self.NPIX
                var py = pix // Self.WIDTH
                var pxx = pix - py * Self.WIDTH

                # ⚠ THE FRAME IS RECOMPUTED PER PIXEL, not hoisted. Same trade
                # `ray_model` makes for geom poses and for the same reason: a
                # thread owns one pixel and cannot hold a scene, and the shared
                # alternative is a per-thread array — the storage class Metal has
                # silently miscomputed four times here.
                var frame = camera_world_frame[Self.DTYPE](
                    cameras, xpos, xquat, subtree_com, env, Int(cam)
                )
                var hit = render_pixel[Self.DTYPE](
                    geoms,
                    Self.D.NGEOM,
                    geom_rgba,
                    bodies,
                    xpos,
                    xquat,
                    env,
                    mesh_meta,
                    mesh_tris,
                    hfield_meta,
                    hfield_data,
                    Self.D.NHFIELD_DATA,
                    frame,
                    Self.WIDTH,
                    Self.HEIGHT,
                    pxx,
                    py,
                    Vec3Generic[Self.DTYPE](lx, ly, lz),
                    Vec3Generic[Self.DTYPE](br, bg, bb),
                    shadows != 0,
                )
                rgb_out[env, pix * RGB_CHANNELS + 0] = hit.rgb.x
                rgb_out[env, pix * RGB_CHANNELS + 1] = hit.rgb.y
                rgb_out[env, pix * RGB_CHANNELS + 2] = hit.rgb.z
                depth_out[env, pix] = hit.depth
                seg_out[env, pix] = Scalar[Self.DTYPE](hit.geom)

        var total = Self.BATCH * Self.NPIX
        ctx.enqueue_function[cam_kernel](
            m.geoms.lt["gpu", Self.L_GEOMS](),
            m.geom_rgba.lt["gpu", Self.L_RGBA](),
            m.bodies.lt["gpu", Self.L_BODIES](),
            d.xpos.lt["gpu", Self.L_B3](),
            d.xquat.lt["gpu", Self.L_B4](),
            d.subtree_com.lt["gpu", Self.L_B3](),
            m.cameras.lt["gpu", Self.L_CAM](),
            m.mesh_meta.lt["gpu", Self.L_MESH_META](),
            m.mesh_tris.lt["gpu", Self.L_TRI](),
            m.hfield_meta.lt["gpu", Self.L_HF_META](),
            d.hfield_data.lt["gpu", Self.L_HF](),
            LayoutTensor[Self.DTYPE, Self.L_RGB](self.rgb),
            LayoutTensor[Self.DTYPE, Self.L_SCALARPIX](self.depth),
            LayoutTensor[Self.DTYPE, Self.L_SCALARPIX](self.seg),
            Int32(self.cam),
            self.light_dir.x,
            self.light_dir.y,
            self.light_dir.z,
            self.background.x,
            self.background.y,
            self.background.z,
            Int32(1) if self.use_shadows else Int32(0),
            grid_dim=(ceildiv(total, TPB),),
            block_dim=(TPB,),
        )

    def render_cpu(
        mut self,
        mut d: Data[Self.DTYPE, Self.D, Self.BATCH],
        mut m: Model[Self.DTYPE, Self.D],
        mut rgb: List[Scalar[Self.DTYPE]],
        mut depth: List[Scalar[Self.DTYPE]],
        mut seg: List[Scalar[Self.DTYPE]],
    ) raises:
        """The same pixels, on the host, into plain `List`s.

        ⚠⚠ THIS IS THE CONTROL LEG OF THE GATE, AND IT IS THE SAME CODE. It
        calls `camera_world_frame` and `render_pixel` exactly as the kernel
        does — the only difference is `lt["cpu", ...]` and a serial loop. That
        is the property `tests/physics3d/test_ray_model_gpu_vs_cpu.mojo`
        exists to defend one layer down, and the reason a GPU miscompute shows
        up here as a DIFFERENCE rather than as two matching wrong answers.

        The lists are resized, not appended to, so a caller may reuse them.
        """
        var n = Self.BATCH * Self.NPIX
        rgb = List[Scalar[Self.DTYPE]](
            length=n * RGB_CHANNELS, fill=Scalar[Self.DTYPE](0)
        )
        depth = List[Scalar[Self.DTYPE]](length=n, fill=Scalar[Self.DTYPE](0))
        seg = List[Scalar[Self.DTYPE]](length=n, fill=Scalar[Self.DTYPE](0))

        # ⚠ THE POSITIVE BRANCH, for the same reason as the kernel above.
        comptime if Self.DTYPE.is_floating_point():
            var geoms_c = m.geoms.lt["cpu", Self.L_GEOMS]()
            var rgba_c = m.geom_rgba.lt["cpu", Self.L_RGBA]()
            var bodies_c = m.bodies.lt["cpu", Self.L_BODIES]()
            var xpos_c = d.xpos.lt["cpu", Self.L_B3]()
            var xquat_c = d.xquat.lt["cpu", Self.L_B4]()
            var com_c = d.subtree_com.lt["cpu", Self.L_B3]()
            var cams_c = m.cameras.lt["cpu", Self.L_CAM]()
            var mm_c = m.mesh_meta.lt["cpu", Self.L_MESH_META]()
            var mt_c = m.mesh_tris.lt["cpu", Self.L_TRI]()
            var hm_c = m.hfield_meta.lt["cpu", Self.L_HF_META]()
            var hd_c = d.hfield_data.lt["cpu", Self.L_HF]()

            for env in range(Self.BATCH):
                var frame = camera_world_frame[Self.DTYPE](
                    cams_c, xpos_c, xquat_c, com_c, env, self.cam
                )
                for py in range(Self.HEIGHT):
                    for pxx in range(Self.WIDTH):
                        var pix = py * Self.WIDTH + pxx
                        var hit = render_pixel[Self.DTYPE](
                            geoms_c,
                            Self.D.NGEOM,
                            rgba_c,
                            bodies_c,
                            xpos_c,
                            xquat_c,
                            env,
                            mm_c,
                            mt_c,
                            hm_c,
                            hd_c,
                            Self.D.NHFIELD_DATA,
                            frame,
                            Self.WIDTH,
                            Self.HEIGHT,
                            pxx,
                            py,
                            self.light_dir,
                            self.background,
                            self.use_shadows,
                        )
                        var b = env * Self.NPIX + pix
                        rgb[b * RGB_CHANNELS + 0] = hit.rgb.x
                        rgb[b * RGB_CHANNELS + 1] = hit.rgb.y
                        rgb[b * RGB_CHANNELS + 2] = hit.rgb.z
                        depth[b] = hit.depth
                        seg[b] = Scalar[Self.DTYPE](hit.geom)
