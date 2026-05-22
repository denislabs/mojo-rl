"""Concat[*BRANCHES] — variadic column-concat.

Generalization of `Parallel[A, B]` to N branches sharing IN_DIM.
Each branch's forward output goes into a per-branch slab, then the
slabs get concatenated into the packed output. Backward splits the
grad_output into per-branch slices, runs each branch's backward into
a shared `gi_temp` slab, and accumulates the per-branch grad_inputs.

Constraints (comptime):
  - N >= 1
  - All BRANCHES share IN_DIM
  - OUT_DIM = Σ BRANCHES[i].OUT_DIM

CPU only (matches v1 — no validating GPU user yet).

Differences vs v1 `concat.mojo`:
  * `Module` conformance.
  * `backward[mode]` collapses v1's `backward` + `backward_input` —
    the mode flows into every branch.
  * `ts: TargetStorage` replaces the per-leaf tag/inference/ctx triplet.
  * `for_each_param` / `zero_grad` recurse over branches.
"""

from std.memory import alloc
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from ..constants import DT, CPU_SIMD_W
from ..core import Initializer, AMPPolicy, NoAMP, ParamVisitor
from ..core.module import Module
from ..core.target_storage import TargetStorage, assert_tag_for


# ──────────────────────────────────────────────────────────────────────
# Comptime helpers for variadic sum and cumulative offsets.
# ──────────────────────────────────────────────────────────────────────


def _total_out_dim[*BRANCHES: Module]() -> Int:
    var s: Int = 0
    comptime for i in range(BRANCHES.size):
        s += BRANCHES[i].OUT_DIM
    return s


def _cumulative_offset[index: Int, *BRANCHES: Module]() -> Int:
    var s: Int = 0
    comptime for j in range(index):
        s += BRANCHES[j].OUT_DIM
    return s


# ──────────────────────────────────────────────────────────────────────
# Concat
# ──────────────────────────────────────────────────────────────────────


struct Concat[*BRANCHES: Module](Module):
    comptime N = Self.BRANCHES.size
    comptime IN_DIM = Self.BRANCHES[0].IN_DIM
    comptime OUT_DIM = _total_out_dim[*Self.BRANCHES]()

    var branches: Tuple[*Self.BRANCHES]

    var out_slabs_cpu: List[UnsafePointer[Scalar[DT], MutAnyOrigin]]
    var out_slab_caps: List[Int]
    var gi_temp_cpu: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var gi_temp_cap: Int

    var ts: TargetStorage

    def __init__(out self):
        comptime assert Self.N >= 1, "Concat requires at least one branch"
        comptime for i in range(Self.N):
            comptime assert (
                Self.BRANCHES[i].IN_DIM == Self.BRANCHES[0].IN_DIM
            ), "Concat: all BRANCHES must share IN_DIM"
        self.branches = Tuple[*Self.BRANCHES]()
        self.out_slabs_cpu = List[UnsafePointer[Scalar[DT], MutAnyOrigin]]()
        self.out_slab_caps = List[Int]()
        self.gi_temp_cpu = alloc[Scalar[DT]](1)
        self.gi_temp_cap = 0
        self.ts = TargetStorage.make_uninit()

    def __init__(out self, var *branches: *Self.BRANCHES):
        """CPU variadic constructor — accepts pre-built CPU branches."""
        comptime assert Self.N >= 1, "Concat requires at least one branch"
        comptime for i in range(Self.N):
            comptime assert (
                Self.BRANCHES[i].IN_DIM == Self.BRANCHES[0].IN_DIM
            ), "Concat: all BRANCHES must share IN_DIM"
        self.branches = Tuple(*branches^)
        self.out_slabs_cpu = List[UnsafePointer[Scalar[DT], MutAnyOrigin]]()
        self.out_slab_caps = List[Int]()
        for _ in range(Self.N):
            self.out_slabs_cpu.append(alloc[Scalar[DT]](1))
            self.out_slab_caps.append(0)
        self.gi_temp_cpu = alloc[Scalar[DT]](1)
        self.gi_temp_cap = 0
        self.ts = TargetStorage.make_cpu()

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "Concat.make[target='gpu', INIT] requires a DeviceContext"
        )
        var c = Self()
        comptime for i in range(Self.N):
            c.branches[i] = Self.BRANCHES[i].make[target, INIT]()
        for _ in range(Self.N):
            c.out_slabs_cpu.append(alloc[Scalar[DT]](1))
            c.out_slab_caps.append(0)
        c.ts = TargetStorage.make_cpu()
        return c^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "Concat.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var c = Self()
        comptime for i in range(Self.N):
            c.branches[i] = Self.BRANCHES[i].make[target, INIT](ctx)
        c.ts = TargetStorage.make_gpu(ctx)
        return c^

    def __del__(deinit self):
        for p in self.out_slabs_cpu:
            p.free()
        self.gi_temp_cpu.free()

    def _ensure_slab_cpu[i: Int](mut self, needed: Int):
        if self.out_slab_caps[i] < needed:
            self.out_slabs_cpu[i].free()
            self.out_slabs_cpu[i] = alloc[Scalar[DT]](needed)
            self.out_slab_caps[i] = needed

    def _ensure_gi_temp_cpu(mut self, needed: Int):
        if self.gi_temp_cap < needed:
            self.gi_temp_cpu.free()
            self.gi_temp_cpu = alloc[Scalar[DT]](needed)
            self.gi_temp_cap = needed

    # ----- Forward ---------------------------------------------------------

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        input: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        comptime assert input.flat_rank == 2, "input rank-2"
        comptime assert output.flat_rank == 2, "output rank-2"
        assert_tag_for["Concat", target](self.ts.target_tag)

        comptime if target == "cpu":
            _concat_forward_cpu[target, BATCH, POLICY=POLICY](self, input, output)
        else:
            raise Error("Concat: GPU forward not yet implemented")

    # ----- Backward --------------------------------------------------------

    def backward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut grad_input: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        comptime assert grad_output.flat_rank == 2, "grad_output rank-2"
        comptime assert grad_input.flat_rank == 2, "grad_input rank-2"
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["Concat", target](self.ts.target_tag)

        comptime if target == "cpu":
            _concat_backward_cpu[target, BATCH, POLICY=POLICY, mode=mode](
                self, grad_output, grad_input,
            )
        else:
            raise Error("Concat: GPU backward not yet implemented")

    # ----- Walkers ---------------------------------------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["Concat", target](self.ts.target_tag)
        var sep = "." if prefix.byte_length() > 0 else ""
        comptime for i in range(Self.N):
            self.branches[i].for_each_param[target, V](
                prefix + sep + String(i), visitor,
            )

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["Concat", target](self.ts.target_tag)
        comptime for i in range(Self.N):
            self.branches[i].zero_grad[target]()


# ──────────────────────────────────────────────────────────────────────
# Free-function bodies (same shape as v1; mode flows into each branch).
# ──────────────────────────────────────────────────────────────────────


def _concat_forward_cpu[
    target: StaticString,
    BATCH: Int,
    POLICY: AMPPolicy,
    *BRANCHES: Module,
](
    mut c: Concat[*BRANCHES],
    input: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
    ],
    mut output: TileTensor[
        mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
        element_size=1, ...,
    ],
) raises:
    comptime assert input.flat_rank == 2, "input rank-2"
    comptime assert output.flat_rank == 2, "output rank-2"
    comptime N = BRANCHES.size

    comptime for i in range(N):
        c._ensure_slab_cpu[i](BATCH * BRANCHES[i].OUT_DIM)

    comptime for i in range(N):
        var slab_ptr = c.out_slabs_cpu[i]
        var slab_tt = TileTensor(slab_ptr, row_major[BATCH, BRANCHES[i].OUT_DIM]())
        c.branches[i].forward[target, BATCH, POLICY=POLICY](input, slab_tt)

    comptime for i in range(N):
        comptime off = _cumulative_offset[i, *BRANCHES]()
        comptime out_i = BRANCHES[i].OUT_DIM
        var slab_ptr = c.out_slabs_cpu[i]
        for b in range(BATCH):
            for j in range(out_i):
                output[b, off + j] = slab_ptr[b * out_i + j]


def _concat_backward_cpu[
    target: StaticString,
    BATCH: Int,
    POLICY: AMPPolicy,
    mode: StaticString,
    *BRANCHES: Module,
](
    mut c: Concat[*BRANCHES],
    grad_output: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
    ],
    mut grad_input: TileTensor[
        mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
        element_size=1, ...,
    ],
) raises:
    comptime assert grad_output.flat_rank == 2, "grad_output rank-2"
    comptime assert grad_input.flat_rank == 2, "grad_input rank-2"
    comptime N = BRANCHES.size
    comptime IN_DIM = BRANCHES[0].IN_DIM

    comptime for i in range(N):
        c._ensure_slab_cpu[i](BATCH * BRANCHES[i].OUT_DIM)
    c._ensure_gi_temp_cpu(BATCH * IN_DIM)

    var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input.ptr)
    var zero_v = SIMD[DT, CPU_SIMD_W](0)
    comptime N_TOTAL = BATCH * IN_DIM
    var k0 = 0
    while k0 + CPU_SIMD_W <= N_TOTAL:
        gi_p.store(k0, zero_v)
        k0 += CPU_SIMD_W
    while k0 < N_TOTAL:
        gi_p[k0] = Scalar[DT](0)
        k0 += 1

    comptime for i in range(N):
        comptime off = _cumulative_offset[i, *BRANCHES]()
        comptime out_i = BRANCHES[i].OUT_DIM
        var slab_ptr = c.out_slabs_cpu[i]
        for b in range(BATCH):
            for j in range(out_i):
                slab_ptr[b * out_i + j] = grad_output[b, off + j]
        var go_tt = TileTensor(slab_ptr, row_major[BATCH, out_i]())
        var gi_temp = TileTensor(c.gi_temp_cpu, row_major[BATCH, IN_DIM]())
        c.branches[i].backward[
            target, BATCH, POLICY=POLICY, mode=mode,
        ](go_tt, gi_temp)
        var ap = c.gi_temp_cpu
        var k = 0
        while k + CPU_SIMD_W <= N_TOTAL:
            gi_p.store(
                k,
                gi_p.load[width=CPU_SIMD_W](k) + ap.load[width=CPU_SIMD_W](k),
            )
            k += CPU_SIMD_W
        while k < N_TOTAL:
            gi_p[k] = gi_p[k] + ap[k]
            k += 1
