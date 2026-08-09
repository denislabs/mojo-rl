"""MCTSExampleReplay — host ring buffer of (obs, packed-target) examples.

AlphaZero trains on individual self-play examples ``(obs, [mcts_π | z])`` with
Monte-Carlo full-game value targets (no bootstrap), so a flat per-example ring
suffices — no trajectory windows (those are MuZero's, landing later as a
sibling). Generic over ``TGT`` (the packed-target width = ``ACT + 1`` for AZ),
so MuZero's per-step storage can reuse it.

Storage-clean: the ring buffers are owned `List`s (RAII — freed by their own
destructor, no manual `alloc`/`free`, no leak risk, no `Pointer`). Inputs
arrive as `List` + offset (safe slices of the caller's trajectory buffer); the
training batch is produced straight into storage `Tensor`s for the ComputeGraph.

Uniform sampling with replacement via an internal xorshift64 RNG (seeded for
reproducible tests). The self-play driver packs ``[π | z]`` before ``record``.
"""

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor


@always_inline
def _xorshift64(s: UInt64) -> UInt64:
    var x = s
    x ^= x << 13
    x ^= x >> 7
    x ^= x << 17
    return x


struct MCTSExampleReplay[OBS: Int, TGT: Int, CAP: Int](
    Movable, Deinitable, Sized
):
    var obs: List[Scalar[DT]]  # CAP * OBS (owned, RAII)
    var tgt: List[Scalar[DT]]  # CAP * TGT (owned, RAII)
    var size: Int
    var pos: Int
    var rng_state: UInt64

    def __init__(out self, seed: UInt64 = 0x9E3779B97F4A7C15):
        self.obs = List[Scalar[DT]](length=Self.CAP * Self.OBS, fill=0)
        self.tgt = List[Scalar[DT]](length=Self.CAP * Self.TGT, fill=0)
        self.size = 0
        self.pos = 0
        self.rng_state = seed

    def __len__(self) -> Int:
        return self.size

    def can_sample(self, batch: Int) -> Bool:
        return self.size >= batch

    def record(
        mut self,
        obs_src: List[Scalar[DT]],
        obs_off: Int,
        tgt_src: List[Scalar[DT]],
        tgt_off: Int,
    ):
        """Append one example, reading `obs_src[obs_off : obs_off+OBS]` and
        `tgt_src[tgt_off : tgt_off+TGT]` (safe `List` slices — the caller's
        trajectory / packed-target buffers). Overwrites oldest on wraparound."""
        var ob = self.pos * Self.OBS
        for j in range(Self.OBS):
            self.obs[ob + j] = obs_src[obs_off + j]
        var tb = self.pos * Self.TGT
        for j in range(Self.TGT):
            self.tgt[tb + j] = tgt_src[tgt_off + j]
        self.pos = (self.pos + 1) % Self.CAP
        if self.size < Self.CAP:
            self.size += 1

    def sample_batch_tensors[
        B: Int
    ](mut self, mut obs_out: Tensor, mut tgt_out: Tensor):
        """Uniform-with-replacement sample of ``B`` examples straight into the
        storage Tensors' host `.data` (`obs_out[B*OBS]` / `tgt_out[B*TGT]`) —
        the clean nn-surface bridge feeding the ComputeGraph. Lazily sizes both
        Tensors."""
        obs_out.ensure(B * Self.OBS)
        tgt_out.ensure(B * Self.TGT)
        for i in range(B):
            self.rng_state = _xorshift64(self.rng_state)
            var idx = Int(self.rng_state % UInt64(self.size))
            var src_o = idx * Self.OBS
            var dst_o = i * Self.OBS
            for j in range(Self.OBS):
                obs_out.data[dst_o + j] = self.obs[src_o + j]
            var src_t = idx * Self.TGT
            var dst_t = i * Self.TGT
            for j in range(Self.TGT):
                tgt_out.data[dst_t + j] = self.tgt[src_t + j]
