"""MCTSExampleReplay — host ring buffer of (obs, packed-target) examples.

AlphaZero trains on individual self-play examples ``(obs, [mcts_π | z])`` with
Monte-Carlo full-game value targets (no bootstrap), so a flat per-example ring
suffices — no trajectory windows (those are MuZero's, landing later as a
sibling). Generic over ``TGT`` (the packed-target width = ``ACT + 1`` for AZ),
so MuZero's per-step storage can reuse it.

Uniform sampling with replacement via an internal xorshift64 RNG (seeded for
reproducible tests). The self-play driver packs ``[π | z]`` before ``record``.
"""

from std.memory import alloc
from mojo_rl.nn.constants import DT


@always_inline
def _xorshift64(s: UInt64) -> UInt64:
    var x = s
    x ^= x << 13
    x ^= x >> 7
    x ^= x << 17
    return x


struct MCTSExampleReplay[OBS: Int, TGT: Int, CAP: Int](
    Movable, ImplicitlyDestructible, Sized
):
    var obs: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var tgt: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var size: Int
    var pos: Int
    var rng_state: UInt64

    def __init__(out self, seed: UInt64 = 0x9E3779B97F4A7C15):
        self.obs = alloc[Scalar[DT]](Self.CAP * Self.OBS)
        self.tgt = alloc[Scalar[DT]](Self.CAP * Self.TGT)
        self.size = 0
        self.pos = 0
        self.rng_state = seed

    def __init__(out self, *, deinit take: Self):
        self.obs = take.obs
        self.tgt = take.tgt
        self.size = take.size
        self.pos = take.pos
        self.rng_state = take.rng_state

    def __del__(deinit self):
        self.obs.free()
        self.tgt.free()

    def __len__(self) -> Int:
        return self.size

    def can_sample(self, batch: Int) -> Bool:
        return self.size >= batch

    def record(
        mut self,
        obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        tgt_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ):
        """Append one example. ``obs_ptr`` has ``OBS`` cells, ``tgt_ptr`` has
        ``TGT`` cells (e.g. ``[mcts_π(ACT) | z(1)]``). Overwrites oldest on
        wraparound."""
        var ob = self.pos * Self.OBS
        for j in range(Self.OBS):
            self.obs[ob + j] = obs_ptr[j]
        var tb = self.pos * Self.TGT
        for j in range(Self.TGT):
            self.tgt[tb + j] = tgt_ptr[j]
        self.pos = (self.pos + 1) % Self.CAP
        if self.size < Self.CAP:
            self.size += 1

    def sample_batch[
        B: Int
    ](
        mut self,
        obs_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        tgt_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ):
        """Uniform-with-replacement sample of ``B`` examples into the packed
        output buffers ``obs_out[B*OBS]`` / ``tgt_out[B*TGT]``."""
        for i in range(B):
            self.rng_state = _xorshift64(self.rng_state)
            var idx = Int(self.rng_state % UInt64(self.size))
            var src_o = idx * Self.OBS
            var dst_o = i * Self.OBS
            for j in range(Self.OBS):
                obs_out[dst_o + j] = self.obs[src_o + j]
            var src_t = idx * Self.TGT
            var dst_t = i * Self.TGT
            for j in range(Self.TGT):
                tgt_out[dst_t + j] = self.tgt[src_t + j]
