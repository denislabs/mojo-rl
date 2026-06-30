"""Dreamer4FrameBuffer — online ring buffer of reward-bearing pixel transitions.

Generalizes `Dreamer4PongRewardBuffer` (offline, env-specific, drops at capacity)
into a compile-time-shaped CIRCULAR buffer for ONLINE training: once full it
overwrites the oldest frame instead of dropping new ones, so a long online run
keeps a sliding window of the most recent `CAP` transitions.

    frames:  CAP * (C*H*W) uint8   (pixels quantized from [0,1])
    actions: CAP uint8
    dones:   CAP uint8
    rewards: CAP fp32

Frames are addressed LOGICALLY (oldest = index 0 … newest = size-1); the physical
slot is `(oldest + i) % CAP`, where `oldest = (pos - size + CAP) % CAP`. Logical
order is temporal order, so a contiguous-T window `[i, i+T)` is temporally
contiguous regardless of where the ring write head sits. Window validity rejects
any window whose frames `[start, start+T-1)` cross an episode boundary (a set
`done`); the last frame may be `done` (a valid termination/prediction target).

Single-producer / single-consumer (act → append → sample-and-train). Not
thread-safe. Pure CPU; internal xorshift RNG for window starts.
"""

from mojo_rl.nn.constants import DT


struct Dreamer4FrameBuffer[C: Int, H: Int, W: Int, ACT: Int, CAP: Int](Movable):
    comptime FRAME: Int = Self.C * Self.H * Self.W

    var pos: Int        # physical write head (next slot to write / oldest when full)
    var size: Int       # number of stored frames (≤ CAP)
    var frames: List[UInt8]
    var actions: List[UInt8]
    var dones: List[UInt8]
    var rewards: List[Scalar[DT]]
    var rng: UInt64

    def __init__(out self, seed: UInt64 = 0x9E3779B97F4A7C15):
        self.pos = 0
        self.size = 0
        self.frames = List[UInt8](length=Self.CAP * Self.FRAME, fill=0)
        self.actions = List[UInt8](length=Self.CAP, fill=0)
        self.dones = List[UInt8](length=Self.CAP, fill=0)
        self.rewards = List[Scalar[DT]](length=Self.CAP, fill=Scalar[DT](0.0))
        self.rng = seed | 1

    def __init__(out self, *, deinit take: Self):
        self.pos = take.pos
        self.size = take.size
        self.frames = take.frames^
        self.actions = take.actions^
        self.dones = take.dones^
        self.rewards = take.rewards^
        self.rng = take.rng

    @always_inline
    def _u64(mut self) -> UInt64:
        var x = self.rng
        x ^= x >> 12
        x ^= x << 25
        x ^= x >> 27
        self.rng = x
        return x * 0x2545F4914F6CDD1D

    @always_inline
    def _oldest(self) -> Int:
        return (self.pos - self.size + Self.CAP) % Self.CAP

    @always_inline
    def _phys(self, logical: Int) -> Int:
        return (self._oldest() + logical) % Self.CAP

    def count(self) -> Int:
        return self.size

    # ------------------------------------------------------------------
    # Append a step (ring overwrite)
    # ------------------------------------------------------------------

    def add_step_fp32_list(
        mut self,
        obs: List[Scalar[DT]],
        action: Int,
        done: Bool,
        reward: Scalar[DT],
    ):
        """Append one step at the write head, overwriting the oldest when full.
        `obs` is the C×H×W frame in [0,1], quantized to uint8 via
        clamp(round(x·255), 0, 255)."""
        var base = self.pos * Self.FRAME
        for i in range(Self.FRAME):
            var v = obs[i] * 255.0 + 0.5
            if v < 0.0:
                v = 0.0
            elif v > 255.0:
                v = 255.0
            self.frames[base + i] = UInt8(Int(v))
        self.actions[self.pos] = UInt8(action)
        self.dones[self.pos] = UInt8(1) if done else UInt8(0)
        self.rewards[self.pos] = reward
        self.pos = (self.pos + 1) % Self.CAP
        if self.size < Self.CAP:
            self.size += 1

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _window_is_valid(self, logical_start: Int, T: Int) -> Bool:
        # Reject if any of frames [start, start+T-2] (logical) is `done` — an
        # episode boundary mid-window. The last frame may be done.
        for i in range(logical_start, logical_start + T - 1):
            if self.dones[self._phys(i)] != 0:
                return False
        return True

    def sample_reward_window_batch[
        B: Int, T: Int,
        pix_o: Origin[mut=True],
        act_o: Origin[mut=True],
        rew_o: Origin[mut=True],
        done_o: Origin[mut=True],
    ](
        mut self,
        pix_fp32: UnsafePointer[Scalar[DT], pix_o],       # [B*T*FRAME]
        act_onehot: UnsafePointer[Scalar[DT], act_o],     # [B*T*ACT]
        rew: UnsafePointer[Scalar[DT], rew_o],            # [B*T]
        done: UnsafePointer[Scalar[DT], done_o],          # [B*T]
    ) raises:
        """Sample B contiguous-T windows (logical/temporal order). Fills, per
        (b, t): fp32 pixels in [0,1] (CHW), one-hot action, reward, done flag."""
        if self.size < T:
            raise Error("Dreamer4FrameBuffer.sample_reward_window_batch: too small")
        var max_start = self.size - T
        for b in range(B):
            var start: Int = -1
            for _ in range(64):
                var r = Float64(self._u64() >> 11) * (1.0 / 9007199254740992.0)
                var cand = Int(r * Float64(max_start + 1))
                if cand > max_start:
                    cand = max_start
                if self._window_is_valid(cand, T):
                    start = cand
                    break
            if start < 0:
                for s in range(max_start + 1):
                    if self._window_is_valid(s, T):
                        start = s
                        break
            if start < 0:
                raise Error("Dreamer4FrameBuffer: no valid windows")
            for t in range(T):
                var bt = b * T + t
                var src = self._phys(start + t) * Self.FRAME
                var dst = bt * Self.FRAME
                for i in range(Self.FRAME):
                    pix_fp32[dst + i] = (
                        Scalar[DT](Float64(self.frames[src + i]))
                        * Scalar[DT](1.0 / 255.0)
                    )
                var a = Int(self.actions[self._phys(start + t)])
                for k in range(Self.ACT):
                    act_onehot[bt * Self.ACT + k] = Scalar[DT](0.0)
                if a >= 0 and a < Self.ACT:
                    act_onehot[bt * Self.ACT + a] = Scalar[DT](1.0)
                rew[bt] = self.rewards[self._phys(start + t)]
                done[bt] = Scalar[DT](
                    1.0 if self.dones[self._phys(start + t)] != 0 else 0.0
                )
