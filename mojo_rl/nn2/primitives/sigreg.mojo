"""SIGReg[DIM, SEQ_LEN, NUM_PROJ, KNOTS] — Sketch Isotropic Gaussian Regularizer.

nn2 port of the legacy `nn/.../sigreg.mojo` DiffOp. Epps-Pulley Gaussianity
statistic via random projections (Maes, Le Lidec et al., LeWM 2026): enforces
projected embeddings to look like N(0, I) samples in characteristic-function
space — no EMAs, pretrained encoders, or auxiliary supervision. The
load-bearing anti-collapse term in the LeWM JEPA loss.

ARITY=1, PARAM-free. Input (BATCH, SEQ_LEN*DIM) interpreted as (B, T, D);
output (BATCH, 1) — the scalar `stat` replicated to every row so a 1/B grad
seed gives chain-rule effective seed 1 (matches the reference `.mean()`).

    A ~ N(0,I_D)^{D×P}, columns L2-normalized          (random projection)
    z[b,t,p]   = sum_d input[b,t*D+d] · A[d,p]
    cm[t,p,k]  = mean_b cos(z·t_k),  sm = mean_b sin(z·t_k)
    stat = (B/(T·P)) · sum_{t,p,k} w_k·[(cm−φ_k)² + sm²]
with t_k = k·3/(K−1), φ_k = exp(−t_k²/2), w_k = trap_k·φ_k.

Leaf-owned state (stable across forward→vjp within a step):
  * cache_z [BATCH, T·P]  — the projected z, reused by backward.
  * workspace             — A / cm / sm / partials / scalar / dLdz, allocated
    ONCE per batch-size and reused (no per-step enqueue_create_buffer — the
    NVIDIA disk-blowup / stream-capture footgun).
PRNG: A is regenerated each call from `Int(cache_z.ptr)` as seed (the cache
pointer is stable across fwd/bwd), so forward and backward see the same A.

Numeric: GPU accumulates in DT (fp32); CPU uses Float64 reductions. Expect
~1e-3 relative agreement, not bit-exact.
"""

from std.gpu import thread_idx, block_idx, block_dim, global_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from std.math import sin, cos, sqrt, log, exp, pi
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, TPB
from ..core import Initializer, AMPPolicy, NoAMP, Cache
from ..core.module import Module, typed_view, typed_view_mut
from ..core.tensor_pack import TensorPack
from ..core.target_storage import (
    require_ctx,
    TargetStorage,
    assert_tag_for,
)


struct SIGReg[DIM: Int, SEQ_LEN: Int, NUM_PROJ: Int, KNOTS: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.SEQ_LEN * Self.DIM)
    comptime OUT_DIM = 1

    @staticmethod
    def display_label() -> String:
        return String("SIGReg")

    # cache_z [BATCH, T*P] — leaf-owned, reused by backward.
    var cache_z: Cache["cache_z"]
    # workspace (GPU) — A/cm/sm/partials/scalar/dLdz, allocated once per batch.
    var ws_dev: Optional[DeviceBuffer[DT]]
    var ts: TargetStorage

    def __init__(out self):
        self.cache_z = Cache["cache_z"]()
        self.ws_dev = None
        self.ts = TargetStorage.make_uninit()

    # ── comptime knot grid / weights ──────────────────────────────────
    @always_inline
    @staticmethod
    def _t_step() -> Float64:
        return 3.0 / Float64(Self.KNOTS - 1)

    @always_inline
    @staticmethod
    def _t_k(k: Int) -> Float64:
        return Float64(k) * Self._t_step()

    @always_inline
    @staticmethod
    def _phi_k(k: Int) -> Float64:
        var tk = Self._t_k(k)
        return exp(-tk * tk * 0.5)

    @always_inline
    @staticmethod
    def _w_k(k: Int) -> Float64:
        var dt = Self._t_step()
        var trap = dt if (k == 0 or k == Self.KNOTS - 1) else 2.0 * dt
        return trap * Self._phi_k(k)

    # ── workspace layout (elements) ───────────────────────────────────
    @always_inline
    @staticmethod
    def _n_partials() -> Int:
        return (Self.SEQ_LEN * Self.NUM_PROJ * Self.KNOTS + TPB - 1) // TPB

    @always_inline
    @staticmethod
    def _ws_off_a() -> Int:
        return 0

    @always_inline
    @staticmethod
    def _ws_off_cm() -> Int:
        return Self.DIM * Self.NUM_PROJ

    @always_inline
    @staticmethod
    def _ws_off_sm() -> Int:
        return Self._ws_off_cm() + Self.SEQ_LEN * Self.NUM_PROJ * Self.KNOTS

    @always_inline
    @staticmethod
    def _ws_off_partials() -> Int:
        return Self._ws_off_sm() + Self.SEQ_LEN * Self.NUM_PROJ * Self.KNOTS

    @always_inline
    @staticmethod
    def _ws_off_scalar() -> Int:
        return Self._ws_off_partials() + Self._n_partials()

    @always_inline
    @staticmethod
    def _ws_off_dLdz() -> Int:
        return Self._ws_off_scalar() + 1

    @always_inline
    @staticmethod
    def workspace_size_for[BATCH: Int]() -> Int:
        return Self._ws_off_dLdz() + BATCH * Self.SEQ_LEN * Self.NUM_PROJ

    # ── random projection (deterministic from seed) ───────────────────
    @staticmethod
    def _generate_a_cpu(
        seed: UInt64, a_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    ):
        for d in range(Self.DIM):
            for p in range(Self.NUM_PROJ):
                var idx = d * Self.NUM_PROJ + p
                var rng1 = PhiloxRandom(seed=seed, offset=UInt64(2 * idx))
                var rng2 = PhiloxRandom(seed=seed, offset=UInt64(2 * idx + 1))
                var u1 = Float64(rng1.step_uniform()[0])
                var u2 = Float64(rng2.step_uniform()[0])
                if u1 < 1e-10:
                    u1 = 1e-10
                var g = sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)
                a_ptr[idx] = Scalar[DT](g)
        for p in range(Self.NUM_PROJ):
            var sum_sq = Float64(0.0)
            for d in range(Self.DIM):
                var v = Float64(a_ptr[d * Self.NUM_PROJ + p])
                sum_sq += v * v
            var norm = sqrt(sum_sq + 1e-12)
            for d in range(Self.DIM):
                var v = Float64(a_ptr[d * Self.NUM_PROJ + p])
                a_ptr[d * Self.NUM_PROJ + p] = Scalar[DT](v / norm)

    # ── factory ───────────────────────────────────────────────────────
    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "SIGReg: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.KNOTS >= 2, "SIGReg: KNOTS must be >= 2"
        var m = Self()
        comptime if target == "cpu":
            m.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["SIGReg.make[target='gpu']"](ctx)
            m.ws_dev = ctx_v.enqueue_create_buffer[DT](1)
            m.ts = TargetStorage.make_gpu(ctx_v)
        return m^

    def _ensure_gpu(mut self, batch: Int) raises:
        var ctx = self.ts.ctx.value()
        self.cache_z.ensure_gpu(ctx, batch * Self.SEQ_LEN * Self.NUM_PROJ)
        self.ws.ensure_gpu(ctx, ws_size)
    # ── forward ───────────────────────────────────────────────────────
    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorPack[Self.ARITY],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["SIGReg", target](self.ts.target_tag)
        comptime D = Self.DIM
        comptime T = Self.SEQ_LEN
        comptime P = Self.NUM_PROJ
        comptime K = Self.KNOTS
        var input = inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var output_v = typed_view_mut[BATCH, 1](output)

        comptime if target == "cpu":
            self.cache_z.ensure_cpu(BATCH * T * P)
            var cache = TileTensor(self.cache_z.cpu, row_major[BATCH, T * P]())
            var seed = UInt64(Int(self.cache_z.cpu_ptr()))
            var a = InlineArray[Scalar[DT], D * P](uninitialized=True)
            Self._generate_a_cpu(seed, a.unsafe_ptr())

            for b in range(BATCH):
                for t in range(T):
                    for p in range(P):
                        var z = Scalar[DT](0)
                        for d in range(D):
                            z += input[b, t * D + d] * a[d * P + p]
                        cache[b, t * P + p] = z

            comptime NTPK = T * P * K
            var cm = InlineArray[Scalar[DT], NTPK](uninitialized=True)
            var sm = InlineArray[Scalar[DT], NTPK](uninitialized=True)
            for i in range(NTPK):
                cm[i] = Scalar[DT](0)
                sm[i] = Scalar[DT](0)
            var inv_b = Scalar[DT](1.0 / Float64(BATCH))
            for b in range(BATCH):
                for t in range(T):
                    for p in range(P):
                        var z = cache[b, t * P + p]
                        for k in range(K):
                            var tk = Scalar[DT](Self._t_k(k))
                            var arg = z * tk
                            var idx = (t * P + p) * K + k
                            cm[idx] += cos(arg) * inv_b
                            sm[idx] += sin(arg) * inv_b

            var prefactor = Scalar[DT](Float64(BATCH) / Float64(T * P))
            var stat = Scalar[DT](0)
            for t in range(T):
                for p in range(P):
                    for k in range(K):
                        var idx = (t * P + p) * K + k
                        var phi = Scalar[DT](Self._phi_k(k))
                        var wk = Scalar[DT](Self._w_k(k))
                        var diff = cm[idx] - phi
                        stat += wk * (diff * diff + sm[idx] * sm[idx])
            stat *= prefactor
            for b in range(BATCH):
                output_v[b, 0] = stat
        else:
            self._ensure_gpu(BATCH)
            var ctx = self.ts.ctx.value()
            comptime N_PARTIALS = Self._n_partials()
            var ws = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.ws_dev.value().unsafe_ptr()
            )
            var cache_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.cache_z.dev.value().unsafe_ptr()
            )
            var a_t = LayoutTensor[DT, Layout.row_major(D, P), MutAnyOrigin](
                ws + Self._ws_off_a()
            )
            var cm_t = LayoutTensor[
                DT, Layout.row_major(T, P * K), MutAnyOrigin
            ](ws + Self._ws_off_cm())
            var sm_t = LayoutTensor[
                DT, Layout.row_major(T, P * K), MutAnyOrigin
            ](ws + Self._ws_off_sm())
            var partials_ptr = ws + Self._ws_off_partials()
            var stat_ptr = ws + Self._ws_off_scalar()
            var in_t = LayoutTensor[
                DT, Layout.row_major(BATCH, T * D), MutAnyOrigin
            ](rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr))
            var cache_t = LayoutTensor[
                DT, Layout.row_major(BATCH, T * P), MutAnyOrigin
            ](cache_p)
            var out_t = LayoutTensor[
                DT, Layout.row_major(BATCH, 1), MutAnyOrigin
            ](rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output_v.ptr))
            var seed = UInt64(Int(cache_p))

            ctx.enqueue_function[_sr_gen_a_unnorm[D, P]](
                a_t, seed, grid_dim=((D * P + TPB - 1) // TPB,), block_dim=(TPB,)
            )
            ctx.enqueue_function[_sr_norm_a[D, P]](
                a_t, grid_dim=((P + TPB - 1) // TPB,), block_dim=(TPB,)
            )
            ctx.enqueue_function[_sr_project[BATCH, T, D, P]](
                in_t, a_t, cache_t,
                grid_dim=((BATCH * T * P + TPB - 1) // TPB,), block_dim=(TPB,),
            )
            ctx.enqueue_function[_sr_cm_sm[BATCH, T, P, K, True]](
                cache_t, cm_t, sm_t, partials_ptr,
                grid_dim=(N_PARTIALS,), block_dim=(TPB,),
            )
            ctx.enqueue_function[_sr_final_reduce[N_PARTIALS]](
                partials_ptr, stat_ptr, grid_dim=(1,), block_dim=(TPB,),
            )
            ctx.enqueue_function[_sr_broadcast_stat[BATCH, T, P]](
                stat_ptr, out_t,
                grid_dim=((BATCH + TPB - 1) // TPB,), block_dim=(TPB,),
            )

    # ── backward ──────────────────────────────────────────────────────
    def vjp[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        grad_inputs: TensorPack[Self.ARITY],
    ) raises:
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["SIGReg", target](self.ts.target_tag)
        comptime D = Self.DIM
        comptime T = Self.SEQ_LEN
        comptime P = Self.NUM_PROJ
        comptime K = Self.KNOTS
        var go = typed_view[BATCH, 1](grad_output)
        var gi = grad_inputs.tile[0, BATCH, Self.IN_DIMS[0]]()

        comptime if target == "cpu":
            var cache = TileTensor(self.cache_z.cpu, row_major[BATCH, T * P]())
            var seed = UInt64(Int(self.cache_z.cpu_ptr()))
            var a = InlineArray[Scalar[DT], D * P](uninitialized=True)
            Self._generate_a_cpu(seed, a.unsafe_ptr())

            comptime NTPK = T * P * K
            var cm = InlineArray[Scalar[DT], NTPK](uninitialized=True)
            var sm = InlineArray[Scalar[DT], NTPK](uninitialized=True)
            for i in range(NTPK):
                cm[i] = Scalar[DT](0)
                sm[i] = Scalar[DT](0)
            var inv_b = Scalar[DT](1.0 / Float64(BATCH))
            for b in range(BATCH):
                for t in range(T):
                    for p in range(P):
                        var z = cache[b, t * P + p]
                        for k in range(K):
                            var tk = Scalar[DT](Self._t_k(k))
                            var arg = z * tk
                            var idx = (t * P + p) * K + k
                            cm[idx] += cos(arg) * inv_b
                            sm[idx] += sin(arg) * inv_b

            var G = Scalar[DT](0)
            for b in range(BATCH):
                G += go[b, 0]
            var coef = G * Scalar[DT](2.0 / Float64(T * P))

            for b in range(BATCH):
                for t in range(T):
                    var dLdz = InlineArray[Scalar[DT], P](uninitialized=True)
                    for p in range(P):
                        var z = cache[b, t * P + p]
                        var acc = Scalar[DT](0)
                        for k in range(K):
                            var tk = Scalar[DT](Self._t_k(k))
                            var wk = Scalar[DT](Self._w_k(k))
                            var phi = Scalar[DT](Self._phi_k(k))
                            var idx = (t * P + p) * K + k
                            var arg = z * tk
                            var bracket = (
                                -(cm[idx] - phi) * sin(arg) + sm[idx] * cos(arg)
                            )
                            acc += wk * tk * bracket
                        dLdz[p] = coef * acc
                    for d in range(D):
                        var g = Scalar[DT](0)
                        for p in range(P):
                            g += a[d * P + p] * dLdz[p]
                        gi[b, t * D + d] = g
        else:
            self._ensure_gpu(BATCH)
            var ctx = self.ts.ctx.value()
            comptime N_PARTIALS = Self._n_partials()
            var ws = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.ws_dev.value().unsafe_ptr()
            )
            var cache_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.cache_z.dev.value().unsafe_ptr()
            )
            var a_t = LayoutTensor[DT, Layout.row_major(D, P), MutAnyOrigin](
                ws + Self._ws_off_a()
            )
            var cm_t = LayoutTensor[
                DT, Layout.row_major(T, P * K), MutAnyOrigin
            ](ws + Self._ws_off_cm())
            var sm_t = LayoutTensor[
                DT, Layout.row_major(T, P * K), MutAnyOrigin
            ](ws + Self._ws_off_sm())
            var partials_ptr = ws + Self._ws_off_partials()
            var g_ptr = ws + Self._ws_off_scalar()
            var dLdz_t = LayoutTensor[
                DT, Layout.row_major(BATCH, T * P), MutAnyOrigin
            ](ws + Self._ws_off_dLdz())
            var cache_t = LayoutTensor[
                DT, Layout.row_major(BATCH, T * P), MutAnyOrigin
            ](cache_p)
            var go_t = LayoutTensor[
                DT, Layout.row_major(BATCH, 1), MutAnyOrigin
            ](rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go.ptr))
            var gi_t = LayoutTensor[
                DT, Layout.row_major(BATCH, T * D), MutAnyOrigin
            ](rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gi.ptr))
            var seed = UInt64(Int(cache_p))

            ctx.enqueue_function[_sr_gen_a_unnorm[D, P]](
                a_t, seed, grid_dim=((D * P + TPB - 1) // TPB,), block_dim=(TPB,)
            )
            ctx.enqueue_function[_sr_norm_a[D, P]](
                a_t, grid_dim=((P + TPB - 1) // TPB,), block_dim=(TPB,)
            )
            ctx.enqueue_function[_sr_cm_sm[BATCH, T, P, K, False]](
                cache_t, cm_t, sm_t, partials_ptr,
                grid_dim=(N_PARTIALS,), block_dim=(TPB,),
            )
            ctx.enqueue_function[_sr_reduce_g[BATCH]](
                go_t, g_ptr, grid_dim=(1,), block_dim=(TPB,),
            )
            ctx.enqueue_function[_sr_dLdz[BATCH, T, P, K]](
                cache_t, cm_t, sm_t, dLdz_t, g_ptr,
                grid_dim=((BATCH * T * P + TPB - 1) // TPB,), block_dim=(TPB,),
            )
            ctx.enqueue_function[_sr_matmul_a[BATCH, T, D, P]](
                dLdz_t, a_t, gi_t,
                grid_dim=((BATCH * T * D + TPB - 1) // TPB,), block_dim=(TPB,),
            )


# ============================================================================
# Module-level GPU kernels — ported verbatim from nn/.../sigreg.mojo (already
# NVIDIA-validated). block.sum reductions (no atomics). Generic over DT.
# ============================================================================


def _sr_gen_a_unnorm[D: Int, P: Int](
    a_t: LayoutTensor[DT, Layout.row_major(D, P), MutAnyOrigin],
    seed: UInt64,
):
    var idx = Int(global_idx.x)
    if idx >= D * P:
        return
    var d_idx = idx // P
    var p_idx = idx % P
    var philox = PhiloxRandom(seed=seed, offset=UInt64(idx))
    var rand_vals = philox.step_uniform()
    var u1 = Float32(rand_vals[0]) + Float32(1e-8)
    var u2 = Float32(rand_vals[1])
    var mag = sqrt(Float32(-2.0) * log(u1))
    var g = mag * cos(u2 * Float32(6.283185307179586))
    a_t[d_idx, p_idx] = Scalar[DT](g)


def _sr_norm_a[D: Int, P: Int](
    a_t: LayoutTensor[DT, Layout.row_major(D, P), MutAnyOrigin],
):
    var p_idx = Int(global_idx.x)
    if p_idx >= P:
        return
    var sum_sq = Scalar[DT](0)
    for d in range(D):
        var v = rebind[Scalar[DT]](a_t[d, p_idx])
        sum_sq += v * v
    var inv_norm = Scalar[DT](1) / (sqrt(sum_sq) + Scalar[DT](1e-12))
    for d in range(D):
        var v = rebind[Scalar[DT]](a_t[d, p_idx])
        a_t[d, p_idx] = v * inv_norm


def _sr_project[BATCH: Int, T: Int, D: Int, P: Int](
    input_t: LayoutTensor[DT, Layout.row_major(BATCH, T * D), MutAnyOrigin],
    a_t: LayoutTensor[DT, Layout.row_major(D, P), MutAnyOrigin],
    cache_t: LayoutTensor[DT, Layout.row_major(BATCH, T * P), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * T * P:
        return
    var p_idx = idx % P
    var t_idx = (idx // P) % T
    var b = idx // (T * P)
    var z = Scalar[DT](0)
    for d in range(D):
        var xi = rebind[Scalar[DT]](input_t[b, t_idx * D + d])
        var aval = rebind[Scalar[DT]](a_t[d, p_idx])
        z += xi * aval
    cache_t[b, t_idx * P + p_idx] = z


def _sr_cm_sm[
    BATCH: Int, T: Int, P: Int, K: Int, INCLUDE_STAT: Bool
](
    cache_t: LayoutTensor[DT, Layout.row_major(BATCH, T * P), MutAnyOrigin],
    cm_t: LayoutTensor[DT, Layout.row_major(T, P * K), MutAnyOrigin],
    sm_t: LayoutTensor[DT, Layout.row_major(T, P * K), MutAnyOrigin],
    partials_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var dt = Scalar[DT](3.0 / Float64(K - 1))
    var inv_b = Scalar[DT](1.0 / Float64(BATCH))
    var contrib = Scalar[DT](0)
    if idx < T * P * K:
        var k_idx = idx % K
        var p_idx = (idx // K) % P
        var t_idx = idx // (K * P)
        var tk = dt * Scalar[DT](k_idx)
        var phi = exp(Scalar[DT](-0.5) * tk * tk)
        var cm_sum = Scalar[DT](0)
        var sm_sum = Scalar[DT](0)
        for b in range(BATCH):
            var z = rebind[Scalar[DT]](cache_t[b, t_idx * P + p_idx])
            var arg = z * tk
            cm_sum += cos(arg)
            sm_sum += sin(arg)
        var cm = cm_sum * inv_b
        var sm = sm_sum * inv_b
        cm_t[t_idx, p_idx * K + k_idx] = cm
        sm_t[t_idx, p_idx * K + k_idx] = sm
        comptime if INCLUDE_STAT:
            var trap = dt if (k_idx == 0 or k_idx == K - 1) else dt * Scalar[
                DT
            ](2.0)
            var wk = trap * phi
            var diff = cm - phi
            contrib = wk * (diff * diff + sm * sm)

    comptime if INCLUDE_STAT:
        var partial = block.sum[block_size=TPB, broadcast=False](
            val=SIMD[DT, 1](contrib)
        )
        if thread_idx.x == 0:
            partials_ptr[Int(block_idx.x)] = partial[0]


def _sr_final_reduce[N_PARTIALS: Int](
    partials_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    out_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    var tid = Int(thread_idx.x)
    var v = Scalar[DT](0)
    var i = tid
    while i < N_PARTIALS:
        v += partials_ptr[i]
        i += TPB
    var total = block.sum[block_size=TPB, broadcast=False](val=SIMD[DT, 1](v))
    if tid == 0:
        out_ptr[0] = total[0]


def _sr_broadcast_stat[BATCH: Int, T: Int, P: Int](
    stat_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    output_t: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
):
    var b = Int(global_idx.x)
    if b >= BATCH:
        return
    var prefactor = Scalar[DT](Float64(BATCH) / Float64(T * P))
    output_t[b, 0] = stat_ptr[0] * prefactor


def _sr_reduce_g[BATCH: Int](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, 1), MutAnyOrigin],
    g_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    var b = Int(thread_idx.x)
    var v = Scalar[DT](0)
    if b < BATCH:
        v = rebind[Scalar[DT]](grad_output[b, 0])
    var total = block.sum[block_size=TPB, broadcast=False](val=SIMD[DT, 1](v))
    if b == 0:
        g_ptr[0] = total[0]


def _sr_dLdz[BATCH: Int, T: Int, P: Int, K: Int](
    cache_t: LayoutTensor[DT, Layout.row_major(BATCH, T * P), MutAnyOrigin],
    cm_t: LayoutTensor[DT, Layout.row_major(T, P * K), MutAnyOrigin],
    sm_t: LayoutTensor[DT, Layout.row_major(T, P * K), MutAnyOrigin],
    dLdz_t: LayoutTensor[DT, Layout.row_major(BATCH, T * P), MutAnyOrigin],
    g_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * T * P:
        return
    var p_idx = idx % P
    var t_idx = (idx // P) % T
    var b = idx // (T * P)
    var z = rebind[Scalar[DT]](cache_t[b, t_idx * P + p_idx])
    var dt = Scalar[DT](3.0 / Float64(K - 1))
    var acc = Scalar[DT](0)
    for k in range(K):
        var tk = dt * Scalar[DT](k)
        var phi = exp(Scalar[DT](-0.5) * tk * tk)
        var trap = dt if (k == 0 or k == K - 1) else dt * Scalar[DT](2.0)
        var wk = trap * phi
        var cm = rebind[Scalar[DT]](cm_t[t_idx, p_idx * K + k])
        var sm = rebind[Scalar[DT]](sm_t[t_idx, p_idx * K + k])
        var arg = z * tk
        var bracket = -(cm - phi) * sin(arg) + sm * cos(arg)
        acc += wk * tk * bracket
    var G = g_ptr[0]
    var coef = G * Scalar[DT](2.0 / Float64(T * P))
    dLdz_t[b, t_idx * P + p_idx] = coef * acc


def _sr_matmul_a[BATCH: Int, T: Int, D: Int, P: Int](
    dLdz_t: LayoutTensor[DT, Layout.row_major(BATCH, T * P), MutAnyOrigin],
    a_t: LayoutTensor[DT, Layout.row_major(D, P), MutAnyOrigin],
    grad_input_t: LayoutTensor[DT, Layout.row_major(BATCH, T * D), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * T * D:
        return
    var d_idx = idx % D
    var t_idx = (idx // D) % T
    var b = idx // (T * D)
    var acc = Scalar[DT](0)
    for p in range(P):
        var dL = rebind[Scalar[DT]](dLdz_t[b, t_idx * P + p])
        var aval = rebind[Scalar[DT]](a_t[d_idx, p])
        acc += aval * dL
    grad_input_t[b, t_idx * D + d_idx] = acc
