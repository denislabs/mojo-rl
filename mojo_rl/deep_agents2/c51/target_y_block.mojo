"""C51TargetYBlock — categorical-projected target distribution.

Bellemare, Dabney & Munos 2017 (Algorithm 1). Operates on the
distributional Q-net output `[B, NA · N_ATOMS]` (`Q_target.forward(sp)`)
and produces a per-batch target distribution `m [B, N_ATOMS]`.

Pipeline (standard branch):
  1. logits_t = Q_target.forward(sp)               [B, NA · N_ATOMS]
  2. Per (b, a): softmax over N_ATOMS              p_t[b, a, k]
  3. Per (b, a): expected Q = Σ_k p_t[b, a, k]·z_k  exp_q[b, a]
  4. a*[b] = argmax_a exp_q[b, a]
  5. p_target[b, k] = p_t[b, a*[b], k]              [B, N_ATOMS]
  6. Projection: for each atom k,
        Tz_k = clip(r + γ^n · z_k · (1 − d), V_min, V_max)
        bj   = (Tz_k − V_min) / Δz
        m[b, ⌊bj⌋] += p_target[b, k] · (⌈bj⌉ − bj)
        m[b, ⌈bj⌉] += p_target[b, k] · (bj − ⌊bj⌋)
        (special case ⌊bj⌋ == ⌈bj⌉ → single bin).

Double-C51 branch (`DOUBLE=True`):
  - logits_on = Q_online.forward(sp)
  - exp_q from logits_on (the same softmax + Σ p·z), argmax → a*
  - distribution sampled from Q_target's softmax at a*

CPU + GPU. GPU branch lowers the per-row argmax/softmax + projection
loops into two BATCH-parallel kernels — no atomics (each thread writes
only its own m[b, *] row).

Self-contained scratch ownership. Output is written into the trainer's
`_mb_m` scratch (NOT `state.mb_y`, which is BATCH-wide; the C51 target
is BATCH·N_ATOMS-wide).
"""

from std.math import exp as fexp, log as flog
from std.gpu import global_idx
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.core.scratch import Scratch
from mojo_rl.nn2.core.scratch_walkers import init_scratch_auto
from mojo_rl.nn2.core.target_storage import TargetStorage, assert_tag_for
from layout import Layout, LayoutTensor, TileTensor, row_major


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — module-level (one thread per BATCH row, no atomics).
# ──────────────────────────────────────────────────────────────────────


def _c51_argmax_action_kernel[
    BATCH: Int, NA: Int, N_ATOMS: Int,
](
    logits: LayoutTensor[
        DT, Layout.row_major(BATCH, NA * N_ATOMS), MutAnyOrigin,
    ],
    z: LayoutTensor[DT, Layout.row_major(N_ATOMS), MutAnyOrigin],
    best_a_out: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
):
    """Per row b: pick `argmax_a Σ_k softmax(logits[b, a, ·])_k · z_k`."""
    var b = Int(global_idx.x)
    if b < BATCH:
        var best_a: Int = 0
        var best_eq: Scalar[DT] = Scalar[DT](0.0)
        for a in range(NA):
            var base = a * N_ATOMS
            var mx = rebind[Scalar[DT]](logits[b, base])
            for i in range(1, N_ATOMS):
                var v = rebind[Scalar[DT]](logits[b, base + i])
                if v > mx:
                    mx = v
            var s_exp: Scalar[DT] = Scalar[DT](0.0)
            for i in range(N_ATOMS):
                s_exp = s_exp + fexp(
                    rebind[Scalar[DT]](logits[b, base + i]) - mx
                )
            var eq: Scalar[DT] = Scalar[DT](0.0)
            for i in range(N_ATOMS):
                var p = (
                    fexp(rebind[Scalar[DT]](logits[b, base + i]) - mx)
                    / s_exp
                )
                eq = eq + p * rebind[Scalar[DT]](z[i])
            if a == 0 or eq > best_eq:
                best_eq = eq
                best_a = a
        best_a_out[b] = Scalar[DT](best_a)


def _c51_project_kernel[
    BATCH: Int, NA: Int, N_ATOMS: Int,
](
    logits_t: LayoutTensor[
        DT, Layout.row_major(BATCH, NA * N_ATOMS), MutAnyOrigin,
    ],
    best_a: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    r: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    d: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    z: LayoutTensor[DT, Layout.row_major(N_ATOMS), MutAnyOrigin],
    m_out: LayoutTensor[DT, Layout.row_major(BATCH, N_ATOMS), MutAnyOrigin],
    gamma_n: Scalar[DT],
    v_min: Scalar[DT],
    v_max: Scalar[DT],
    dz: Scalar[DT],
):
    """Bellemare Algorithm 1 — one thread per b. Zeroes m[b, *] then
    scatter-adds (l_idx, u_idx) bin pairs from softmax(logits_t[b, a*]·z)
    for the chosen action a*. No atomics needed."""
    var b = Int(global_idx.x)
    if b < BATCH:
        for k in range(N_ATOMS):
            m_out[b, k] = Scalar[DT](0.0)

        var a_star = Int(rebind[Scalar[DT]](best_a[b]))
        var tbase = a_star * N_ATOMS

        var mxt = rebind[Scalar[DT]](logits_t[b, tbase])
        for i in range(1, N_ATOMS):
            var v = rebind[Scalar[DT]](logits_t[b, tbase + i])
            if v > mxt:
                mxt = v
        var st_exp: Scalar[DT] = Scalar[DT](0.0)
        for i in range(N_ATOMS):
            st_exp = st_exp + fexp(
                rebind[Scalar[DT]](logits_t[b, tbase + i]) - mxt
            )

        var rb = rebind[Scalar[DT]](r[b])
        var nonterm = Scalar[DT](1.0) - rebind[Scalar[DT]](d[b])
        for j in range(N_ATOMS):
            var p_tgt = (
                fexp(rebind[Scalar[DT]](logits_t[b, tbase + j]) - mxt)
                / st_exp
            )
            var tz = rb + gamma_n * rebind[Scalar[DT]](z[j]) * nonterm
            if tz < v_min:
                tz = v_min
            if tz > v_max:
                tz = v_max
            var bj = (tz - v_min) / dz
            var l_idx = Int(bj)
            var u_idx = l_idx + 1
            if u_idx >= N_ATOMS:
                u_idx = N_ATOMS - 1
            if l_idx >= N_ATOMS:
                l_idx = N_ATOMS - 1
            if l_idx == u_idx:
                m_out[b, l_idx] = (
                    rebind[Scalar[DT]](m_out[b, l_idx]) + p_tgt
                )
            else:
                var u_w = bj - Scalar[DT](l_idx)
                var l_w = Scalar[DT](1.0) - u_w
                m_out[b, l_idx] = (
                    rebind[Scalar[DT]](m_out[b, l_idx]) + p_tgt * l_w
                )
                m_out[b, u_idx] = (
                    rebind[Scalar[DT]](m_out[b, u_idx]) + p_tgt * u_w
                )


struct C51TargetYBlock[
    Q_NET: Module,
    BATCH: Int,
    OBS: Int,
    NA: Int,
    N_ATOMS: Int,
    DOUBLE: Bool = False,
](Defaultable & Movable & ImplicitlyDestructible):
    """Owns:
      - `_logits_t [B · NA · N_ATOMS]`: Q_target(sp) output.
      - `_logits_on [B · NA · N_ATOMS]`: Q_online(sp) output (Double only).
      - `_z [N_ATOMS]`: atom support (z_k = V_min + k · Δz).
      - `_gamma_n`: γ^nstep baked at make() (n-step ready).
      - `V_min`, `V_max`: support bounds.
    """

    var _logits_t: Scratch["logits_t", Self.BATCH * Self.NA * Self.N_ATOMS, True]
    var _logits_on: Scratch["logits_on", Self.BATCH * Self.NA * Self.N_ATOMS, True]
    var _z: Scratch["z", Self.N_ATOMS, True]
    var _best_a: Scratch["best_a", Self.BATCH, True]

    var gamma_n: Scalar[DT]
    var v_min: Scalar[DT]
    var v_max: Scalar[DT]
    var dz: Scalar[DT]
    var ts: TargetStorage

    def __init__(out self):
        self._logits_t = Scratch[
            "logits_t", Self.BATCH * Self.NA * Self.N_ATOMS, True,
        ]()
        self._logits_on = Scratch[
            "logits_on", Self.BATCH * Self.NA * Self.N_ATOMS, True,
        ]()
        self._z = Scratch["z", Self.N_ATOMS, True]()
        self._best_a = Scratch["best_a", Self.BATCH, True]()
        self.gamma_n = Scalar[DT](0.99)
        self.v_min = Scalar[DT](-10.0)
        self.v_max = Scalar[DT](10.0)
        self.dz = Scalar[DT](0.0)
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString,
    ](
        gamma: Scalar[DT] = Scalar[DT](0.99),
        nstep: Int = 1,
        v_min: Scalar[DT] = Scalar[DT](-10.0),
        v_max: Scalar[DT] = Scalar[DT](10.0),
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Bakes γ^nstep into the bootstrap discount + sets the atom
        support `z_k = V_min + k · (V_max − V_min)/(N_ATOMS−1)`."""
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "C51TargetYBlock: target must be 'cpu' or 'gpu'"
        comptime assert (
            Self.Q_NET.IN_DIMS[0] == Self.OBS
        ), "C51TargetYBlock: Q_NET.IN_DIM must equal OBS"
        comptime assert (
            Self.Q_NET.OUT_DIM == Self.NA * Self.N_ATOMS
        ), "C51TargetYBlock: Q_NET.OUT_DIM must equal NA · N_ATOMS"
        comptime assert (
            Self.N_ATOMS >= 2
        ), "C51TargetYBlock: N_ATOMS must be ≥ 2 (Δz needs N_ATOMS−1)"
        var b = Self()
        b.ts = TargetStorage.make[target](ctx=ctx)
        var g_n: Scalar[DT] = Scalar[DT](1.0)
        for _ in range(nstep):
            g_n = g_n * gamma
        b.gamma_n = g_n
        b.v_min = v_min
        b.v_max = v_max
        b.dz = (v_max - v_min) / Scalar[DT](Self.N_ATOMS - 1)
        init_scratch_auto[Self, target=target](b, ctx)

        # Bake the z support — CPU mirror always (used by trainer's
        # action selection at greedy-eval time). On GPU also upload to
        # the device buffer (kernels read it directly).
        var z_p = b._z.cpu_ptr()
        for k in range(Self.N_ATOMS):
            z_p[k] = v_min + Scalar[DT](k) * b.dz
        comptime if target == "gpu":
            var ctx_v = ctx.value()
            var z_host = ctx_v.enqueue_create_host_buffer[DT](Self.N_ATOMS)
            ctx_v.synchronize()
            for k in range(Self.N_ATOMS):
                z_host.unsafe_ptr()[k] = v_min + Scalar[DT](k) * b.dz
            ctx_v.enqueue_copy(b._z.dev.value(), z_host)
            ctx_v.synchronize()
        return b^

    def z_ptr(mut self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """Expose the z support — used by the trainer's action selection
        and by the Q-update block to compute expected Q for argmax."""
        return self._z.cpu_ptr()

    def step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        mut q_target: Self.Q_NET,
        mut q_online: Self.Q_NET,
        mb_sp_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_r_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_d_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        mb_m_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Writes `mb_m_ptr` [B, N_ATOMS] target distribution in-place.
        `q_online` is ignored on the standard path (DOUBLE=False)."""
        assert_tag_for["C51TargetYBlock", target](self.ts.target_tag)
        comptime ROW = Self.NA * Self.N_ATOMS

        # Step 1: Q_target(sp) → _logits_t.
        var logits_t_p = self._logits_t.target_ptr[target]()
        var sp_t = TileTensor(mb_sp_ptr, row_major[Self.BATCH, Self.OBS]())
        var lt_t = TileTensor(logits_t_p, row_major[Self.BATCH, ROW]())
        q_target.forward[target, Self.BATCH, POLICY](sp_t, output=lt_t)

        # Step 2 (DOUBLE only): Q_online(sp) → _logits_on.
        comptime if Self.DOUBLE:
            var logits_on_p = self._logits_on.target_ptr[target]()
            var lo_t = TileTensor(
                logits_on_p, row_major[Self.BATCH, ROW](),
            )
            q_online.forward[target, Self.BATCH, POLICY](sp_t, output=lo_t)

        comptime if target == "gpu":
            var ctx = self.ts.ctx.value()
            var z_lt = LayoutTensor[
                DT, Layout.row_major(Self.N_ATOMS), MutAnyOrigin,
            ](self._z.dev_ptr())
            var best_a_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH), MutAnyOrigin,
            ](self._best_a.dev_ptr())
            var logits_t_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH, ROW), MutAnyOrigin,
            ](logits_t_p)

            # Argmax: source is _logits_on if DOUBLE else _logits_t.
            comptime arg_kernel = _c51_argmax_action_kernel[
                Self.BATCH, Self.NA, Self.N_ATOMS,
            ]
            comptime n_blocks_b = (Self.BATCH + TPB - 1) // TPB
            comptime if Self.DOUBLE:
                var logits_on_lt = LayoutTensor[
                    DT, Layout.row_major(Self.BATCH, ROW), MutAnyOrigin,
                ](self._logits_on.dev_ptr())
                ctx.enqueue_function[arg_kernel](
                    logits_on_lt, z_lt, best_a_lt,
                    grid_dim=n_blocks_b, block_dim=TPB,
                )
            else:
                ctx.enqueue_function[arg_kernel](
                    logits_t_lt, z_lt, best_a_lt,
                    grid_dim=n_blocks_b, block_dim=TPB,
                )

            # Project Q_target softmax(a*) onto z support → m.
            var r_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH), MutAnyOrigin,
            ](mb_r_ptr)
            var d_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH), MutAnyOrigin,
            ](mb_d_ptr)
            var m_lt = LayoutTensor[
                DT, Layout.row_major(Self.BATCH, Self.N_ATOMS), MutAnyOrigin,
            ](mb_m_ptr)
            comptime proj_kernel = _c51_project_kernel[
                Self.BATCH, Self.NA, Self.N_ATOMS,
            ]
            comptime n_blocks_p = (Self.BATCH + TPB - 1) // TPB
            ctx.enqueue_function[proj_kernel](
                logits_t_lt, best_a_lt, r_lt, d_lt, z_lt, m_lt,
                self.gamma_n, self.v_min, self.v_max, self.dz,
                grid_dim=n_blocks_p, block_dim=TPB,
            )
            return

        # CPU path below — only reached when target == "cpu" (early
        # `return` on the GPU branch above). logits_t_p / logits_on_p
        # from earlier resolved to CPU pointers via target_ptr.
        var z_p = self._z.cpu_ptr()

        # Zero output m first (we'll scatter-add into it).
        for k in range(Self.BATCH * Self.N_ATOMS):
            mb_m_ptr[k] = Scalar[DT](0.0)

        # Step 3: compute argmax action per row. Source is Q_online
        # (Double) or Q_target (standard). Result is stored in a List
        # so the per-row projection loop below can read it back without
        # relying on `comptime if`-scoped vars (Mojo's `var` inside a
        # comptime-if branch doesn't escape the branch).
        var best_a_per_row = List[Int](length=Self.BATCH, fill=0)
        comptime if Self.DOUBLE:
            var src_p = self._logits_on.cpu_ptr()
            for b in range(Self.BATCH):
                var best_a: Int = 0
                var best_eq: Scalar[DT] = Scalar[DT](0.0)
                for a in range(Self.NA):
                    var base = b * ROW + a * Self.N_ATOMS
                    var mxa = src_p[base]
                    for i in range(1, Self.N_ATOMS):
                        if src_p[base + i] > mxa:
                            mxa = src_p[base + i]
                    var s_exp: Scalar[DT] = Scalar[DT](0.0)
                    for i in range(Self.N_ATOMS):
                        s_exp = s_exp + fexp(src_p[base + i] - mxa)
                    var eq: Scalar[DT] = Scalar[DT](0.0)
                    for i in range(Self.N_ATOMS):
                        var p = fexp(src_p[base + i] - mxa) / s_exp
                        eq = eq + p * z_p[i]
                    if a == 0 or eq > best_eq:
                        best_eq = eq
                        best_a = a
                best_a_per_row[b] = best_a
        else:
            for b in range(Self.BATCH):
                var best_a: Int = 0
                var best_eq: Scalar[DT] = Scalar[DT](0.0)
                for a in range(Self.NA):
                    var base = b * ROW + a * Self.N_ATOMS
                    var mxa = logits_t_p[base]
                    for i in range(1, Self.N_ATOMS):
                        if logits_t_p[base + i] > mxa:
                            mxa = logits_t_p[base + i]
                    var s_exp: Scalar[DT] = Scalar[DT](0.0)
                    for i in range(Self.N_ATOMS):
                        s_exp = s_exp + fexp(logits_t_p[base + i] - mxa)
                    var eq: Scalar[DT] = Scalar[DT](0.0)
                    for i in range(Self.N_ATOMS):
                        var p = fexp(logits_t_p[base + i] - mxa) / s_exp
                        eq = eq + p * z_p[i]
                    if a == 0 or eq > best_eq:
                        best_eq = eq
                        best_a = a
                best_a_per_row[b] = best_a

        # Per-row projection: target distribution sampled from Q_target's
        # softmax at the chosen action, then projected onto z support.
        for b in range(Self.BATCH):
            var best_a = best_a_per_row[b]

            # 4. Target distribution p_target = softmax(Q_target[a*])
            var tbase = b * ROW + best_a * Self.N_ATOMS
            var mxt = logits_t_p[tbase]
            for i in range(1, Self.N_ATOMS):
                if logits_t_p[tbase + i] > mxt:
                    mxt = logits_t_p[tbase + i]
            var st_exp: Scalar[DT] = Scalar[DT](0.0)
            for i in range(Self.N_ATOMS):
                st_exp = st_exp + fexp(logits_t_p[tbase + i] - mxt)

            # 5. Projection (Bellemare Algorithm 1).
            var r = mb_r_ptr[b]
            var nonterm = Scalar[DT](1.0) - mb_d_ptr[b]
            for j in range(Self.N_ATOMS):
                var p_tgt_j = fexp(logits_t_p[tbase + j] - mxt) / st_exp
                var tz = r + self.gamma_n * z_p[j] * nonterm
                if tz < self.v_min:
                    tz = self.v_min
                if tz > self.v_max:
                    tz = self.v_max
                var bj = (tz - self.v_min) / self.dz
                var l_idx = Int(bj)
                var u_idx = l_idx + 1
                if u_idx >= Self.N_ATOMS:
                    u_idx = Self.N_ATOMS - 1
                if l_idx >= Self.N_ATOMS:
                    l_idx = Self.N_ATOMS - 1
                if l_idx == u_idx:
                    mb_m_ptr[b * Self.N_ATOMS + l_idx] = (
                        mb_m_ptr[b * Self.N_ATOMS + l_idx] + p_tgt_j
                    )
                else:
                    var u_w = bj - Scalar[DT](l_idx)
                    var l_w = Scalar[DT](1.0) - u_w
                    mb_m_ptr[b * Self.N_ATOMS + l_idx] = (
                        mb_m_ptr[b * Self.N_ATOMS + l_idx] + p_tgt_j * l_w
                    )
                    mb_m_ptr[b * Self.N_ATOMS + u_idx] = (
                        mb_m_ptr[b * Self.N_ATOMS + u_idx] + p_tgt_j * u_w
                    )
