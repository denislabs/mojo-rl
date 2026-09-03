# +--------------------------------------------------------------------------+ #
# | SmolVLA — the static prefix KV cache
# +--------------------------------------------------------------------------+ #
"""Per-layer K/V for the image+language+state prefix, written once and read by
every denoising step.

Without it each of the ten Euler steps would recompute the whole prefix through
sixteen layers; with it the prefix is paid for once. That is the difference
between a demonstrator and a slideshow, which is why `VLA_PLAN.md` calls the
cache structural rather than an optimisation.

## Why ours and not MAX's

P0.2 drove `nn.kv_cache.generic_flash_attention_kv_cache_padded` and found it
reachable but owned by MAX's serving runtime: its `ContinuousBatchingKVCache`
invariants are for ragged, multi-tenant LLM serving, and getting them wrong by
hand produced either all-zero output or a 2^64-byte allocation, neither naming
the violated rule. A robot loop is **batch 1 with a fixed-length prefix** — none
of that machinery buys anything here, and all of it has to be satisfied. So:
fixed comptime shapes, one slab, no paging.

## The crop, designed away

The reference appends the suffix K/V into the same cache during a denoising step
and then calls `past_key_values.crop(prefix_len)` to undo it before the next
step. Forget the crop and the cache grows every step while silently changing
what each one attends to.

Here the prefix slab is **immutable after prefill**. A self-attention layer that
needs `[prefix; suffix]` gets it materialised into a separate scratch buffer, so
there is nothing to undo and no state that ten steps can corrupt. Same
arithmetic, one fewer way to be wrong.

## Layout

    k, v      [LAYERS, B, PREFIX, N_KV * HEAD_DIM]   post-RoPE
    scratch   [B, PREFIX + SUFFIX, N_KV * HEAD_DIM]

⚠ **The cache holds POST-RoPE keys.** The reference rotates q and k before
`past_key_values.update`, so a cached key already carries its absolute position.
Storing pre-RoPE keys and rotating on read would be shape-identical, finite, and
wrong — and wrong in the direction that looks like a mildly worse policy rather
than a bug.

⚠ **Reading a layer that was never written raises.** A zero-filled cache makes
attention put near-uniform weight on nothing in particular: finite output,
plausible magnitudes, no NaN. `filled` turns that into a failure with a name.
"""

from max.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor


struct SmolVLAKVCache[
    LAYERS: Int,
    PREFIX: Int,
    SUFFIX: Int,
    N_KV: Int,
    HEAD_DIM: Int,
    B: Int = 1,
](Movable):
    comptime KVW: Int = Self.N_KV * Self.HEAD_DIM
    comptime LAYER_N: Int = Self.B * Self.PREFIX * Self.KVW
    comptime TOTAL: Int = Self.LAYERS * Self.LAYER_N
    comptime FULL: Int = Self.PREFIX + Self.SUFFIX
    comptime SCRATCH_N: Int = Self.B * Self.FULL * Self.KVW
    comptime SUFFIX_N: Int = Self.B * Self.SUFFIX * Self.KVW

    var k: Tensor
    var v: Tensor
    var sk: Tensor
    """Scratch `[prefix; suffix]` keys — rebuilt per layer, never persistent."""
    var sv: Tensor
    var filled: List[Bool]
    var on_gpu: Bool

    def __init__(out self):
        self.k = Tensor()
        self.v = Tensor()
        self.sk = Tensor()
        self.sv = Tensor()
        self.filled = List[Bool]()
        self.on_gpu = False

    def __init__(out self, *, deinit move: Self):
        self.k = move.k^
        self.v = move.v^
        self.sk = move.sk^
        self.sv = move.sv^
        self.filled = move.filled^
        self.on_gpu = move.on_gpu

    @staticmethod
    def make[
        target: StaticString
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "SmolVLAKVCache: target must be 'cpu' or 'gpu'"
        )
        var c = Self()
        comptime if target == "cpu":
            c.k = Tensor.alloc(Self.TOTAL)
            c.v = Tensor.alloc(Self.TOTAL)
            c.sk = Tensor.alloc(Self.SCRATCH_N)
            c.sv = Tensor.alloc(Self.SCRATCH_N)
            for i in range(Self.TOTAL):
                c.k.data[i] = Scalar[DT](0)
                c.v.data[i] = Scalar[DT](0)
        else:
            var d = ctx.value()
            c.k = Tensor.alloc(Self.TOTAL)
            c.v = Tensor.alloc(Self.TOTAL)
            c.sk = Tensor.alloc(Self.SCRATCH_N)
            c.sv = Tensor.alloc(Self.SCRATCH_N)
            for i in range(Self.TOTAL):
                c.k.data[i] = Scalar[DT](0)
                c.v.data[i] = Scalar[DT](0)
            c.k.upload(d)
            c.v.upload(d)
            c.sk.ensure_gpu(d, Self.SCRATCH_N)
            c.sv.ensure_gpu(d, Self.SCRATCH_N)
            c.on_gpu = True
        for _ in range(Self.LAYERS):
            c.filled.append(False)
        return c^

    def _check_layer(self, layer: Int) raises:
        if layer < 0 or layer >= Self.LAYERS:
            raise Error(
                "SmolVLAKVCache: layer " + String(layer) + " out of "
                + String(Self.LAYERS)
            )

    def write_prefix[
        target: StaticString
    ](
        mut self, layer: Int, mut k_src: Tensor, mut v_src: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Store one layer's POST-RoPE prefix K/V. Called once, at prefill."""
        self._check_layer(layer)
        var off = layer * Self.LAYER_N
        comptime if target == "cpu":
            for i in range(Self.LAYER_N):
                self.k.data[off + i] = k_src.data[i]
                self.v.data[off + i] = v_src.data[i]
        else:
            var d = ctx.value()
            var kd = self.k.dev.value().create_sub_buffer[DT](
                off, Self.LAYER_N
            )
            var vd = self.v.dev.value().create_sub_buffer[DT](
                off, Self.LAYER_N
            )
            var ks = k_src.dev.value().create_sub_buffer[DT](0, Self.LAYER_N)
            var vs = v_src.dev.value().create_sub_buffer[DT](0, Self.LAYER_N)
            d.enqueue_copy(kd, ks)
            d.enqueue_copy(vd, vs)
        self.filled[layer] = True

    def _require_filled(self, layer: Int) raises:
        self._check_layer(layer)
        if not self.filled[layer]:
            raise Error(
                "SmolVLAKVCache: layer " + String(layer) + " was never written"
                " — a denoising step is reading a prefix that was never"
                " prefilled. Attention over a zero cache is finite and"
                " plausible, so this would not have shown up as a NaN."
            )

    def n_filled(self) -> Int:
        var n = 0
        for i in range(Self.LAYERS):
            if self.filled[i]:
                n += 1
        return n

    def offset_of(self, layer: Int) -> Int:
        """Flat element offset of `layer`'s slab in `k`/`v`."""
        return layer * Self.LAYER_N

    def read_prefix_cpu(self, layer: Int, i: Int) raises -> Scalar[DT]:
        """One cached key element — for gates and debugging, not the hot path."""
        self._require_filled(layer)
        return self.k.data[layer * Self.LAYER_N + i]

    def build_scratch[
        target: StaticString
    ](
        mut self, layer: Int, mut k_suf: Tensor, mut v_suf: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Materialise `[prefix; suffix]` for one layer into `sk`/`sv`.

        This is what a denoising SELF-attention layer attends over. The prefix
        slab is untouched, so nothing has to be cropped afterwards and ten steps
        in a row see the identical prefix.
        """
        self._require_filled(layer)
        var off = layer * Self.LAYER_N
        comptime if target == "cpu":
            for i in range(Self.LAYER_N):
                self.sk.data[i] = self.k.data[off + i]
                self.sv.data[i] = self.v.data[off + i]
            for i in range(Self.SUFFIX_N):
                self.sk.data[Self.LAYER_N + i] = k_suf.data[i]
                self.sv.data[Self.LAYER_N + i] = v_suf.data[i]
        else:
            var d = ctx.value()
            var kp = self.k.dev.value().create_sub_buffer[DT](
                off, Self.LAYER_N
            )
            var vp = self.v.dev.value().create_sub_buffer[DT](
                off, Self.LAYER_N
            )
            var kd = self.sk.dev.value().create_sub_buffer[DT](
                0, Self.LAYER_N
            )
            var vd = self.sv.dev.value().create_sub_buffer[DT](
                0, Self.LAYER_N
            )
            d.enqueue_copy(kd, kp)
            d.enqueue_copy(vd, vp)
            var kt = self.sk.dev.value().create_sub_buffer[DT](
                Self.LAYER_N, Self.SUFFIX_N
            )
            var vt = self.sv.dev.value().create_sub_buffer[DT](
                Self.LAYER_N, Self.SUFFIX_N
            )
            var ks = k_suf.dev.value().create_sub_buffer[DT](0, Self.SUFFIX_N)
            var vs = v_suf.dev.value().create_sub_buffer[DT](0, Self.SUFFIX_N)
            d.enqueue_copy(kt, ks)
            d.enqueue_copy(vt, vs)

    def read_layer_into[
        target: StaticString
    ](
        mut self, layer: Int, mut k_dst: Tensor, mut v_dst: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Copy one layer's cached prefix K/V out, for a CROSS layer to project.

        A cross-attention layer does not attend to the cache directly — it
        pushes the VLM's 320-wide K/V through its own `[320, 320]` projections
        first. The cache stays read-only.
        """
        self._require_filled(layer)
        var off = layer * Self.LAYER_N
        comptime if target == "cpu":
            k_dst.ensure(Self.LAYER_N)
            v_dst.ensure(Self.LAYER_N)
            for i in range(Self.LAYER_N):
                k_dst.data[i] = self.k.data[off + i]
                v_dst.data[i] = self.v.data[off + i]
        else:
            var d = ctx.value()
            k_dst.ensure_gpu(d, Self.LAYER_N)
            v_dst.ensure_gpu(d, Self.LAYER_N)
            d.enqueue_copy(
                k_dst.dev.value().create_sub_buffer[DT](0, Self.LAYER_N),
                self.k.dev.value().create_sub_buffer[DT](off, Self.LAYER_N),
            )
            d.enqueue_copy(
                v_dst.dev.value().create_sub_buffer[DT](0, Self.LAYER_N),
                self.v.dev.value().create_sub_buffer[DT](off, Self.LAYER_N),
            )
