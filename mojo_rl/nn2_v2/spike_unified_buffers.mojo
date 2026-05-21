"""Unified-buffer skeleton — orchestrator-owns-slabs buffer model.

CPU only, fp32 only. Proves out the buffer-ownership design from
docs/NN2_AUDIT.md before retrofitting into nn2/.

# Design contract

  - Modules own only params + a forward-input *pointer alias* (no copy).
  - Orchestrators (here, Sequential) own all inter-module slabs.
  - Slabs do triple duty: forward activation → backward cache target
    → grad_input destination. Layout chosen so every read happens
    before any clobbering write.

Slab[i] timeline in a Sequential of N children:
  forward:   written with A_i (output of layer i; input of layer i+1)
  backward:  layer i+1's backward reads it as cache (still A_i),
             then overwrites with dL/dA_i (its grad_input)
             layer i's backward then reads it as grad_output

# Backward-order invariant (MUST hold inside every leaf)

  Param-grads first (reads from cache), THEN write grad_input.
  Otherwise grad_input clobbers the cache slot when Sequential aliases.

# Output-caching layers (Tanh)

  Tanh caches `y = tanh(x)`. With slab aliasing, the slab holding `y`
  gets clobbered by the next layer's backward before Tanh reads. So
  Tanh owns its own cache buffer (option 2 from the audit decisions).
  Linear and ReLU are input-caching → alias the slab.
"""

from std.math import tanh as ftanh
from std.memory import alloc
from layout import TileTensor, row_major

comptime DT = DType.float32


# ──────────────────────────────────────────────────────────────────────
# ParamVisitor — invoked once per leaf parameter during a tree walk.
# Spike-minimal: name + raw pointers + count. Real impl will widen.
# ──────────────────────────────────────────────────────────────────────


trait ParamVisitor(ImplicitlyDestructible):
    def visit(
        mut self,
        name: String,
        param_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        grad_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        n_elems: Int,
    ) raises:
        ...


# ──────────────────────────────────────────────────────────────────────
# Module trait — no buffer-ownership methods. Just compute + param walk.
# ──────────────────────────────────────────────────────────────────────


trait Module(Defaultable & Movable & ImplicitlyDestructible):
    comptime IN_DIM: Int
    comptime OUT_DIM: Int

    def forward[BATCH: Int](
        mut self,
        input: TileTensor[dtype=DT, element_size=1, ...],
        mut output: TileTensor[mut=True, dtype=DT, element_size=1, ...],
    ) raises:
        ...

    def backward[BATCH: Int](
        mut self,
        grad_output: TileTensor[dtype=DT, element_size=1, ...],
        mut grad_input: TileTensor[mut=True, dtype=DT, element_size=1, ...],
    ) raises:
        ...

    def for_each_param[V: ParamVisitor](
        mut self, prefix: String, mut visitor: V,
    ) raises:
        ...


# ──────────────────────────────────────────────────────────────────────
# Linear[IN, OUT] — input-caching, aliases the slab.
# ──────────────────────────────────────────────────────────────────────


struct Linear[IN: Int, OUT: Int](Module):
    comptime IN_DIM = Self.IN
    comptime OUT_DIM = Self.OUT
    comptime W_SIZE = Self.IN * Self.OUT
    comptime B_SIZE = Self.OUT

    var weight: List[Scalar[DT]]
    var bias:   List[Scalar[DT]]
    var grad_w: List[Scalar[DT]]
    var grad_b: List[Scalar[DT]]

    # The only "cache" — a borrowed pointer at the forward input.
    # Valid for the lifetime of whatever slab `input.ptr` aliases.
    # Sequential keeps slabs live across forward→backward; standalone
    # callers must guarantee the same.
    var _cached_input_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]

    def __init__(out self):
        self.weight = List[Scalar[DT]]()
        self.bias   = List[Scalar[DT]]()
        self.grad_w = List[Scalar[DT]]()
        self.grad_b = List[Scalar[DT]]()
        self._cached_input_ptr = UnsafePointer[
            Scalar[DT], MutAnyOrigin
        ](unsafe_from_address=0)

    @staticmethod
    def make_xavier(seed_offset: Int = 0) raises -> Self:
        """Tiny deterministic init — Xavier-ish via LCG. Good enough for
        a 2-layer XOR gradcheck."""
        var lin = Self()
        lin.weight = List[Scalar[DT]](length=Self.W_SIZE, fill=0.0)
        lin.bias   = List[Scalar[DT]](length=Self.B_SIZE, fill=0.0)
        lin.grad_w = List[Scalar[DT]](length=Self.W_SIZE, fill=0.0)
        lin.grad_b = List[Scalar[DT]](length=Self.B_SIZE, fill=0.0)
        var scale: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](Self.IN)
        var state: UInt64 = UInt64(0x12345678) + UInt64(seed_offset)
        for k in range(Self.W_SIZE):
            state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            var r = Scalar[DT]((Int(state >> 32) & 0xFFFF)) / Scalar[DT](65535.0)
            lin.weight[k] = (r - Scalar[DT](0.5)) * scale * Scalar[DT](2.0)
        return lin^

    def forward[BATCH: Int](
        mut self,
        input: TileTensor[dtype=DT, element_size=1, ...],
        mut output: TileTensor[mut=True, dtype=DT, element_size=1, ...],
    ) raises:
        comptime assert input.flat_rank  == 2, "input rank-2 [BATCH, IN]"
        comptime assert output.flat_rank == 2, "output rank-2 [BATCH, OUT]"

        # ── Alias the input pointer for backward. NO COPY. ──
        self._cached_input_ptr = rebind[
            UnsafePointer[Scalar[DT], MutAnyOrigin]
        ](input.ptr)

        # Naive scalar matmul + bias. Production would route through
        # linalg.matmul + SIMD bias-add — irrelevant for the design spike.
        var w_p = self.weight.unsafe_ptr()
        var b_p = self.bias.unsafe_ptr()
        for b in range(BATCH):
            for j in range(Self.OUT):
                var acc: Scalar[DT] = b_p[j]
                for i in range(Self.IN):
                    acc += input[b, i] * w_p[i * Self.OUT + j]
                output[b, j] = acc

    def backward[BATCH: Int](
        mut self,
        grad_output: TileTensor[dtype=DT, element_size=1, ...],
        mut grad_input: TileTensor[mut=True, dtype=DT, element_size=1, ...],
    ) raises:
        comptime assert grad_output.flat_rank == 2, "grad_output rank-2"
        comptime assert grad_input.flat_rank  == 2, "grad_input rank-2"

        # CRITICAL ORDERING: read cache (for grad_w) BEFORE writing
        # grad_input. With Sequential's slab aliasing,
        # `self._cached_input_ptr` and `grad_input.ptr` are the SAME
        # address.
        var cache_view = TileTensor(
            self._cached_input_ptr, row_major[BATCH, Self.IN]()
        )

        # 1) grad_w[i,j] += sum_b cache[b,i] * grad_output[b,j]   (reads cache)
        var gw_p = self.grad_w.unsafe_ptr()
        for i in range(Self.IN):
            for j in range(Self.OUT):
                var s: Scalar[DT] = 0.0
                for b in range(BATCH):
                    s += cache_view[b, i] * grad_output[b, j]
                gw_p[i * Self.OUT + j] += s

        # 2) grad_b[j] += sum_b grad_output[b,j]   (no cache read)
        var gb_p = self.grad_b.unsafe_ptr()
        for j in range(Self.OUT):
            var s: Scalar[DT] = 0.0
            for b in range(BATCH):
                s += grad_output[b, j]
            gb_p[j] += s

        # 3) grad_input[b,i] = sum_j grad_output[b,j] * W[i,j]   (writes — cache now invalid)
        var w_p = self.weight.unsafe_ptr()
        for b in range(BATCH):
            for i in range(Self.IN):
                var s: Scalar[DT] = 0.0
                for j in range(Self.OUT):
                    s += grad_output[b, j] * w_p[i * Self.OUT + j]
                grad_input[b, i] = s

    def zero_grad(mut self):
        for k in range(Self.W_SIZE):
            self.grad_w[k] = 0.0
        for j in range(Self.B_SIZE):
            self.grad_b[j] = 0.0

    def for_each_param[V: ParamVisitor](
        mut self, prefix: String, mut visitor: V,
    ) raises:
        var sep = "." if prefix.byte_length() > 0 else ""
        visitor.visit(
            prefix + sep + "weight",
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](self.weight.unsafe_ptr()),
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](self.grad_w.unsafe_ptr()),
            Self.W_SIZE,
        )
        visitor.visit(
            prefix + sep + "bias",
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](self.bias.unsafe_ptr()),
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](self.grad_b.unsafe_ptr()),
            Self.B_SIZE,
        )


# ──────────────────────────────────────────────────────────────────────
# ReLU[DIM] — input-caching, aliases the slab.
# Element-wise → in-place safe even when grad_input aliases cache.
# ──────────────────────────────────────────────────────────────────────


struct ReLU[DIM: Int](Module):
    comptime IN_DIM = Self.DIM
    comptime OUT_DIM = Self.DIM

    var _cached_input_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]

    def __init__(out self):
        self._cached_input_ptr = UnsafePointer[
            Scalar[DT], MutAnyOrigin
        ](unsafe_from_address=0)

    def forward[BATCH: Int](
        mut self,
        input: TileTensor[dtype=DT, element_size=1, ...],
        mut output: TileTensor[mut=True, dtype=DT, element_size=1, ...],
    ) raises:
        comptime assert input.flat_rank  == 2, "input rank-2 [BATCH, DIM]"
        comptime assert output.flat_rank == 2, "output rank-2 [BATCH, DIM]"
        self._cached_input_ptr = rebind[
            UnsafePointer[Scalar[DT], MutAnyOrigin]
        ](input.ptr)
        for b in range(BATCH):
            for d in range(Self.DIM):
                var x = input[b, d]
                output[b, d] = x if x > Scalar[DT](0.0) else Scalar[DT](0.0)

    def backward[BATCH: Int](
        mut self,
        grad_output: TileTensor[dtype=DT, element_size=1, ...],
        mut grad_input: TileTensor[mut=True, dtype=DT, element_size=1, ...],
    ) raises:
        comptime assert grad_output.flat_rank == 2, "grad_output rank-2"
        comptime assert grad_input.flat_rank  == 2, "grad_input rank-2"
        var cache_view = TileTensor(
            self._cached_input_ptr, row_major[BATCH, Self.DIM]()
        )
        # Element-wise: safe to write grad_input while reading the same
        # element of cache — no cross-element dependency.
        for b in range(BATCH):
            for d in range(Self.DIM):
                grad_input[b, d] = (
                    grad_output[b, d]
                    if cache_view[b, d] > Scalar[DT](0.0)
                    else Scalar[DT](0.0)
                )

    def for_each_param[V: ParamVisitor](
        mut self, prefix: String, mut visitor: V,
    ) raises:
        pass  # ReLU has no parameters.


# ──────────────────────────────────────────────────────────────────────
# Tanh[DIM] — output-caching (option 2 from the audit).
# Owns its own cache buffer so slab aliasing doesn't clobber it.
# ──────────────────────────────────────────────────────────────────────


struct Tanh[DIM: Int](Module):
    comptime IN_DIM = Self.DIM
    comptime OUT_DIM = Self.DIM

    # Owned cache — holds y = tanh(x) from forward, survives the
    # backward chain because it's separate from Sequential's slab.
    var _cache: List[Scalar[DT]]
    var _cache_cap: Int

    def __init__(out self):
        self._cache = List[Scalar[DT]]()
        self._cache_cap = 0

    def _ensure_cache(mut self, needed: Int):
        if self._cache_cap < needed:
            self._cache.resize(needed, 0.0)
            self._cache_cap = needed

    def forward[BATCH: Int](
        mut self,
        input: TileTensor[dtype=DT, element_size=1, ...],
        mut output: TileTensor[mut=True, dtype=DT, element_size=1, ...],
    ) raises:
        comptime assert input.flat_rank  == 2, "input rank-2 [BATCH, DIM]"
        comptime assert output.flat_rank == 2, "output rank-2 [BATCH, DIM]"
        self._ensure_cache(BATCH * Self.DIM)
        var c_p = self._cache.unsafe_ptr()
        for b in range(BATCH):
            for d in range(Self.DIM):
                var y = ftanh(input[b, d])
                output[b, d] = y
                c_p[b * Self.DIM + d] = y

    def backward[BATCH: Int](
        mut self,
        grad_output: TileTensor[dtype=DT, element_size=1, ...],
        mut grad_input: TileTensor[mut=True, dtype=DT, element_size=1, ...],
    ) raises:
        comptime assert grad_output.flat_rank == 2, "grad_output rank-2"
        comptime assert grad_input.flat_rank  == 2, "grad_input rank-2"
        var c_p = self._cache.unsafe_ptr()
        for b in range(BATCH):
            for d in range(Self.DIM):
                var y = c_p[b * Self.DIM + d]
                grad_input[b, d] = grad_output[b, d] * (Scalar[DT](1.0) - y * y)

    def for_each_param[V: ParamVisitor](
        mut self, prefix: String, mut visitor: V,
    ) raises:
        pass  # Tanh has no parameters.


# ──────────────────────────────────────────────────────────────────────
# Sequential[*MODULES] — owns N-1 slabs; threads them through forward
# AND backward. Slabs do triple duty.
# ──────────────────────────────────────────────────────────────────────


struct Sequential[*MODULES: Module](Module):
    comptime N = Self.MODULES.size
    comptime IN_DIM  = Self.MODULES[0].IN_DIM
    comptime OUT_DIM = Self.MODULES[Self.N - 1].OUT_DIM

    var children: Tuple[*Self.MODULES]
    var mid: List[UnsafePointer[Scalar[DT], MutAnyOrigin]]
    var mid_caps: List[Int]

    def __init__(out self):
        comptime assert Self.N >= 1, "Sequential requires at least one child"
        comptime if Self.N >= 2:
            comptime for i in range(Self.N - 1):
                comptime assert (
                    Self.MODULES[i].OUT_DIM == Self.MODULES[i + 1].IN_DIM
                ), "Sequential: adjacent dims must match"
        self.children = Tuple[*Self.MODULES]()
        self.mid = List[UnsafePointer[Scalar[DT], MutAnyOrigin]]()
        self.mid_caps = List[Int]()
        comptime if Self.N >= 2:
            for _ in range(Self.N - 1):
                self.mid.append(alloc[Scalar[DT]](1))
                self.mid_caps.append(0)

    def __init__(out self, var *children: *Self.MODULES):
        comptime assert Self.N >= 1, "Sequential requires at least one child"
        comptime if Self.N >= 2:
            comptime for i in range(Self.N - 1):
                comptime assert (
                    Self.MODULES[i].OUT_DIM == Self.MODULES[i + 1].IN_DIM
                ), "Sequential: adjacent dims must match"
        self.children = Tuple(*children^)
        self.mid = List[UnsafePointer[Scalar[DT], MutAnyOrigin]]()
        self.mid_caps = List[Int]()
        comptime if Self.N >= 2:
            for _ in range(Self.N - 1):
                self.mid.append(alloc[Scalar[DT]](1))
                self.mid_caps.append(0)

    def __del__(deinit self):
        for p in self.mid:
            p.free()

    def _ensure_slab[i: Int](mut self, needed: Int):
        if self.mid_caps[i] < needed:
            self.mid[i].free()
            self.mid[i] = alloc[Scalar[DT]](needed)
            self.mid_caps[i] = needed

    def forward[BATCH: Int](
        mut self,
        input: TileTensor[dtype=DT, element_size=1, ...],
        mut output: TileTensor[mut=True, dtype=DT, element_size=1, ...],
    ) raises:
        comptime if Self.N == 1:
            self.children[0].forward[BATCH](input, output)
            return

        # Lazy-grow slabs.
        comptime for i in range(Self.N - 1):
            self._ensure_slab[i](BATCH * Self.MODULES[i].OUT_DIM)

        # Layer 0: external input → slab[0]
        var s0 = TileTensor(
            self.mid[0], row_major[BATCH, Self.MODULES[0].OUT_DIM]()
        )
        self.children[0].forward[BATCH](input, s0)

        # Layers 1..N-2: slab[i-1] → slab[i]
        comptime for i in range(1, Self.N - 1):
            var in_v = TileTensor(
                self.mid[i - 1], row_major[BATCH, Self.MODULES[i].IN_DIM]()
            )
            var out_v = TileTensor(
                self.mid[i], row_major[BATCH, Self.MODULES[i].OUT_DIM]()
            )
            self.children[i].forward[BATCH](in_v, out_v)

        # Layer N-1: slab[N-2] → external output
        comptime if Self.N >= 2:
            var sN = TileTensor(
                self.mid[Self.N - 2],
                row_major[BATCH, Self.MODULES[Self.N - 1].IN_DIM](),
            )
            self.children[Self.N - 1].forward[BATCH](sN, output)

    def backward[BATCH: Int](
        mut self,
        grad_output: TileTensor[dtype=DT, element_size=1, ...],
        mut grad_input: TileTensor[mut=True, dtype=DT, element_size=1, ...],
    ) raises:
        comptime if Self.N == 1:
            self.children[0].backward[BATCH](grad_output, grad_input)
            return

        # Last layer N-1: reads cache from slab[N-2] (= A_{N-2}),
        # writes grad_input INTO slab[N-2] (now = dL/dA_{N-2}).
        comptime if Self.N >= 2:
            var sN = TileTensor(
                self.mid[Self.N - 2],
                row_major[BATCH, Self.MODULES[Self.N - 1].IN_DIM](),
            )
            self.children[Self.N - 1].backward[BATCH](grad_output, sN)

        # Layers N-2..1: read grad_output from slab[i], write grad_input
        # into slab[i-1].
        comptime for j in range(1, Self.N - 1):
            comptime i = Self.N - 1 - j
            var go_v = TileTensor(
                self.mid[i], row_major[BATCH, Self.MODULES[i].OUT_DIM]()
            )
            var gi_v = TileTensor(
                self.mid[i - 1], row_major[BATCH, Self.MODULES[i].IN_DIM]()
            )
            self.children[i].backward[BATCH](go_v, gi_v)

        # Layer 0: reads grad_output from slab[0], writes external grad_input.
        var s0 = TileTensor(
            self.mid[0], row_major[BATCH, Self.MODULES[0].OUT_DIM]()
        )
        self.children[0].backward[BATCH](s0, grad_input)

    def for_each_param[V: ParamVisitor](
        mut self, prefix: String, mut visitor: V,
    ) raises:
        var sep = "." if prefix.byte_length() > 0 else ""
        comptime for i in range(Self.N):
            self.children[i].for_each_param(prefix + sep + String(i), visitor)
