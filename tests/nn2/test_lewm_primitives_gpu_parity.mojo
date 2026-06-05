"""CPU↔GPU parity for the LeWM nn2 primitives (Phase A).

Runs `LayerNormNoAffine`, `Modulate`, `Gate` forward + vjp on identical
deterministic inputs through both a CPU instance and a GPU instance, and
asserts the outputs and every input-gradient match within 1e-4. (Legacy
`nn` reported bitwise parity for Modulate/Gate on Metal; LayerNorm's
block reduction may differ in the last ULPs, hence the small tolerance.)

Run with the GPU env, e.g.:
    pixi run -e apple mojo run -I . tests/nn2/test_lewm_primitives_gpu_parity.mojo
"""

from std.memory import alloc
from std.gpu.host import DeviceContext, DeviceBuffer
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.primitives.layer_norm_no_affine import LayerNormNoAffine
from mojo_rl.nn2.primitives.modulate import Modulate
from mojo_rl.nn2.primitives.gate import Gate


comptime TOL: Scalar[DT] = 1e-4


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


def _p(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def _det(i: Int, scale: Float64) -> Scalar[DT]:
    var v = (Float64((i * 2654435761) % 1000) / 500.0) - 1.0
    return Scalar[DT](v * scale)


def _cmp(name: String, cpu: UnsafePointer[Scalar[DT], MutAnyOrigin],
         gpu: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int) raises:
    var maxd: Scalar[DT] = 0.0
    for k in range(n):
        var d = (cpu[k] - gpu[k]).__abs__()
        if d > maxd:
            maxd = d
    print("  ", name, " max|cpu-gpu| =", maxd)
    assert_true(maxd < TOL, name + " CPU/GPU parity")


# ──────────────────────────────────────────────────────────────────────
def test_layer_norm_no_affine_parity() raises:
    print("test_layer_norm_no_affine_parity ...")
    comptime BATCH = 4
    comptime DIM = 16
    comptime N = BATCH * DIM
    var ctx = DeviceContext()

    var x = _a(N); var w = _a(N)
    for k in range(N):
        x[k] = _det(k + 1, 2.0)
        w[k] = _det(k + 7, 1.0)

    # CPU
    var y_cpu = _a(N); var gx_cpu = _a(N)
    var mc = LayerNormNoAffine[DIM].make[target="cpu", INIT=Kaiming]()
    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var yc_t = TileTensor(y_cpu, row_major[BATCH, DIM]())
    mc.forward["cpu", BATCH](x_t, output=yc_t)
    var w_t = TileTensor(w, row_major[BATCH, DIM]())
    var gxc_t = TileTensor(gx_cpu, row_major[BATCH, DIM]())
    mc.vjp["cpu", BATCH](w_t, gxc_t)

    # GPU
    var x_d = ctx.enqueue_create_buffer[DT](N)
    var y_d = ctx.enqueue_create_buffer[DT](N)
    var w_d = ctx.enqueue_create_buffer[DT](N)
    var gx_d = ctx.enqueue_create_buffer[DT](N)
    var x_h = ctx.enqueue_create_host_buffer[DT](N)
    var w_h = ctx.enqueue_create_host_buffer[DT](N)
    ctx.synchronize()
    for k in range(N):
        x_h.unsafe_ptr()[k] = x[k]
        w_h.unsafe_ptr()[k] = w[k]
    ctx.enqueue_copy(x_d, x_h); ctx.enqueue_copy(w_d, w_h)
    ctx.synchronize()

    var mg = LayerNormNoAffine[DIM].make[target="gpu", INIT=Kaiming](ctx)
    var xd_t = TileTensor(_p(x_d), row_major[BATCH, DIM]())
    var yd_t = TileTensor(_p(y_d), row_major[BATCH, DIM]())
    mg.forward["gpu", BATCH](xd_t, output=yd_t)
    var wd_t = TileTensor(_p(w_d), row_major[BATCH, DIM]())
    var gxd_t = TileTensor(_p(gx_d), row_major[BATCH, DIM]())
    mg.vjp["gpu", BATCH](wd_t, gxd_t)

    var y_gpu = _a(N); var gx_gpu = _a(N)
    var y_oh = ctx.enqueue_create_host_buffer[DT](N)
    var gx_oh = ctx.enqueue_create_host_buffer[DT](N)
    ctx.enqueue_copy(y_oh, y_d); ctx.enqueue_copy(gx_oh, gx_d)
    ctx.synchronize()
    for k in range(N):
        y_gpu[k] = y_oh.unsafe_ptr()[k]
        gx_gpu[k] = gx_oh.unsafe_ptr()[k]

    _cmp("LNNA fwd", y_cpu, y_gpu, N)
    _cmp("LNNA grad_x", gx_cpu, gx_gpu, N)
    x.free(); w.free(); y_cpu.free(); gx_cpu.free()
    y_gpu.free(); gx_gpu.free()
    print("  ok")


# ──────────────────────────────────────────────────────────────────────
def test_modulate_parity() raises:
    print("test_modulate_parity ...")
    comptime BATCH = 4
    comptime DIM = 12
    comptime N = BATCH * DIM
    var ctx = DeviceContext()

    var x = _a(N); var sc = _a(N); var sh = _a(N); var w = _a(N)
    for k in range(N):
        x[k] = _det(k + 1, 1.5)
        sc[k] = _det(k + 5, 0.8)
        sh[k] = _det(k + 9, 0.5)
        w[k] = _det(k + 13, 1.0)

    # CPU
    var y_cpu = _a(N); var gx_c = _a(N); var gs_c = _a(N); var gsh_c = _a(N)
    var mc = Modulate[DIM].make[target="cpu", INIT=Kaiming]()
    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var sc_t = TileTensor(sc, row_major[BATCH, DIM]())
    var sh_t = TileTensor(sh, row_major[BATCH, DIM]())
    var yc_t = TileTensor(y_cpu, row_major[BATCH, DIM]())
    mc.forward["cpu", BATCH](x_t, sc_t, sh_t, output=yc_t)
    var w_t = TileTensor(w, row_major[BATCH, DIM]())
    var gxc_t = TileTensor(gx_c, row_major[BATCH, DIM]())
    var gsc_t = TileTensor(gs_c, row_major[BATCH, DIM]())
    var gshc_t = TileTensor(gsh_c, row_major[BATCH, DIM]())
    mc.vjp["cpu", BATCH](w_t, gxc_t, gsc_t, gshc_t)

    # GPU
    var x_d = ctx.enqueue_create_buffer[DT](N)
    var sc_d = ctx.enqueue_create_buffer[DT](N)
    var sh_d = ctx.enqueue_create_buffer[DT](N)
    var w_d = ctx.enqueue_create_buffer[DT](N)
    var y_d = ctx.enqueue_create_buffer[DT](N)
    var gx_d = ctx.enqueue_create_buffer[DT](N)
    var gs_d = ctx.enqueue_create_buffer[DT](N)
    var gsh_d = ctx.enqueue_create_buffer[DT](N)
    var h = ctx.enqueue_create_host_buffer[DT](N)
    ctx.synchronize()

    @parameter
    def up(dst: DeviceBuffer[DT], src: UnsafePointer[Scalar[DT], MutAnyOrigin]) raises:
        for k in range(N):
            h.unsafe_ptr()[k] = src[k]
        ctx.enqueue_copy(dst, h)
        ctx.synchronize()
    up(x_d, x); up(sc_d, sc); up(sh_d, sh); up(w_d, w)

    var mg = Modulate[DIM].make[target="gpu", INIT=Kaiming](ctx)
    var xd_t = TileTensor(_p(x_d), row_major[BATCH, DIM]())
    var scd_t = TileTensor(_p(sc_d), row_major[BATCH, DIM]())
    var shd_t = TileTensor(_p(sh_d), row_major[BATCH, DIM]())
    var yd_t = TileTensor(_p(y_d), row_major[BATCH, DIM]())
    mg.forward["gpu", BATCH](xd_t, scd_t, shd_t, output=yd_t)
    var wd_t = TileTensor(_p(w_d), row_major[BATCH, DIM]())
    var gxd_t = TileTensor(_p(gx_d), row_major[BATCH, DIM]())
    var gsd_t = TileTensor(_p(gs_d), row_major[BATCH, DIM]())
    var gshd_t = TileTensor(_p(gsh_d), row_major[BATCH, DIM]())
    mg.vjp["gpu", BATCH](wd_t, gxd_t, gsd_t, gshd_t)

    var y_g = _a(N); var gx_g = _a(N); var gs_g = _a(N); var gsh_g = _a(N)

    @parameter
    def down(src: DeviceBuffer[DT], dst: UnsafePointer[Scalar[DT], MutAnyOrigin]) raises:
        ctx.enqueue_copy(h, src)
        ctx.synchronize()
        for k in range(N):
            dst[k] = h.unsafe_ptr()[k]
    down(y_d, y_g); down(gx_d, gx_g); down(gs_d, gs_g); down(gsh_d, gsh_g)

    _cmp("Modulate fwd", y_cpu, y_g, N)
    _cmp("Modulate grad_x", gx_c, gx_g, N)
    _cmp("Modulate grad_scale", gs_c, gs_g, N)
    _cmp("Modulate grad_shift", gsh_c, gsh_g, N)
    print("  ok")


# ──────────────────────────────────────────────────────────────────────
def test_gate_parity() raises:
    print("test_gate_parity ...")
    comptime BATCH = 4
    comptime DIM = 12
    comptime N = BATCH * DIM
    var ctx = DeviceContext()

    var x = _a(N); var g = _a(N); var br = _a(N); var w = _a(N)
    for k in range(N):
        x[k] = _det(k + 1, 1.5)
        g[k] = _det(k + 5, 0.8)
        br[k] = _det(k + 9, 1.2)
        w[k] = _det(k + 13, 1.0)

    # CPU
    var y_cpu = _a(N); var gx_c = _a(N); var gg_c = _a(N); var gbr_c = _a(N)
    var mc = Gate[DIM].make[target="cpu", INIT=Kaiming]()
    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var g_t = TileTensor(g, row_major[BATCH, DIM]())
    var br_t = TileTensor(br, row_major[BATCH, DIM]())
    var yc_t = TileTensor(y_cpu, row_major[BATCH, DIM]())
    mc.forward["cpu", BATCH](x_t, g_t, br_t, output=yc_t)
    var w_t = TileTensor(w, row_major[BATCH, DIM]())
    var gxc_t = TileTensor(gx_c, row_major[BATCH, DIM]())
    var ggc_t = TileTensor(gg_c, row_major[BATCH, DIM]())
    var gbrc_t = TileTensor(gbr_c, row_major[BATCH, DIM]())
    mc.vjp["cpu", BATCH](w_t, gxc_t, ggc_t, gbrc_t)

    # GPU
    var x_d = ctx.enqueue_create_buffer[DT](N)
    var g_d = ctx.enqueue_create_buffer[DT](N)
    var br_d = ctx.enqueue_create_buffer[DT](N)
    var w_d = ctx.enqueue_create_buffer[DT](N)
    var y_d = ctx.enqueue_create_buffer[DT](N)
    var gx_d = ctx.enqueue_create_buffer[DT](N)
    var gg_d = ctx.enqueue_create_buffer[DT](N)
    var gbr_d = ctx.enqueue_create_buffer[DT](N)
    var h = ctx.enqueue_create_host_buffer[DT](N)
    ctx.synchronize()

    @parameter
    def up(dst: DeviceBuffer[DT], src: UnsafePointer[Scalar[DT], MutAnyOrigin]) raises:
        for k in range(N):
            h.unsafe_ptr()[k] = src[k]
        ctx.enqueue_copy(dst, h)
        ctx.synchronize()
    up(x_d, x); up(g_d, g); up(br_d, br); up(w_d, w)

    var mg = Gate[DIM].make[target="gpu", INIT=Kaiming](ctx)
    var xd_t = TileTensor(_p(x_d), row_major[BATCH, DIM]())
    var gd_t = TileTensor(_p(g_d), row_major[BATCH, DIM]())
    var brd_t = TileTensor(_p(br_d), row_major[BATCH, DIM]())
    var yd_t = TileTensor(_p(y_d), row_major[BATCH, DIM]())
    mg.forward["gpu", BATCH](xd_t, gd_t, brd_t, output=yd_t)
    var wd_t = TileTensor(_p(w_d), row_major[BATCH, DIM]())
    var gxd_t = TileTensor(_p(gx_d), row_major[BATCH, DIM]())
    var ggd_t = TileTensor(_p(gg_d), row_major[BATCH, DIM]())
    var gbrd_t = TileTensor(_p(gbr_d), row_major[BATCH, DIM]())
    mg.vjp["gpu", BATCH](wd_t, gxd_t, ggd_t, gbrd_t)

    var y_g = _a(N); var gx_g = _a(N); var gg_g = _a(N); var gbr_g = _a(N)

    @parameter
    def down(src: DeviceBuffer[DT], dst: UnsafePointer[Scalar[DT], MutAnyOrigin]) raises:
        ctx.enqueue_copy(h, src)
        ctx.synchronize()
        for k in range(N):
            dst[k] = h.unsafe_ptr()[k]
    down(y_d, y_g); down(gx_d, gx_g); down(gg_d, gg_g); down(gbr_d, gbr_g)

    _cmp("Gate fwd", y_cpu, y_g, N)
    _cmp("Gate grad_x", gx_c, gx_g, N)
    _cmp("Gate grad_gate", gg_c, gg_g, N)
    _cmp("Gate grad_branch", gbr_c, gbr_g, N)
    print("  ok")


def main() raises:
    print("=" * 70)
    print("LeWM nn2 primitives — CPU/GPU parity (Phase A)")
    print("=" * 70)
    test_layer_norm_no_affine_parity()
    test_modulate_parity()
    test_gate_parity()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
