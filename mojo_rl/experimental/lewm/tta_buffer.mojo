"""TTAWindowBuffer — online window buffer for AdaJEPA test-time adaptation.

Rolling per-env ring of the last `T` (frame, action-block) pairs collected
during closed-loop MPC (docs/ADAJEPA_LEWM_TTA_PLAN.md §3). One control cycle
is one WM-time step, so the last `T` cycles form a window with exactly the
offline training layout: frames `t = 0..T-1` oldest→newest, and action row
`t` = the planner-space block executed AFTER rendering frame `t` (producing
frame `t+1`) — matching the dataset's dense-span reshape (`act[t]` covers
raw steps `[t·frameskip, (t+1)·frameskip)`).

Push protocol per cycle, per env (skip both pushes for done envs — their
window freezes at the last real transition):

    push_frame(b, frame_ptr)     # at render time, BEFORE the goal render
                                 # overwrites the staging buffer
    push_action(b, block_ptr)    # after execution (skipped if the env
                                 # finished mid-block); completes the pair

`fill` writes the batch tiles for `LeWMTrainer.train_step_masked`. Envs that
finished before accumulating `T` pairs borrow a donor env's window (first
env with a full ring) — real data, just duplicated rows.

Host-side only; the caller uploads the filled tiles to the device.
"""

from mojo_rl.nn.constants import DT


struct TTAWindowBuffer[BATCH: Int, T: Int, IMG_DIM: Int, ACT: Int](Movable):
    var frames: List[Scalar[DT]]
    """`[BATCH, T, IMG_DIM]` ring; slot = pair count % T."""
    var actions: List[Scalar[DT]]
    """`[BATCH, T, ACT]` ring, slot-aligned with `frames`."""
    var count: List[Int]
    """Per-env completed (frame, action) pairs — bumped by `push_action`."""

    def __init__(out self, enabled: Bool = True):
        """`enabled=False` allocates 1-element stubs — for callers that keep
        the buffer unconditionally but only push behind a runtime flag."""
        var nf = Self.BATCH * Self.T * Self.IMG_DIM if enabled else 1
        var na = Self.BATCH * Self.T * Self.ACT if enabled else 1
        self.frames = List[Scalar[DT]](length=nf, fill=Scalar[DT](0))
        self.actions = List[Scalar[DT]](length=na, fill=Scalar[DT](0))
        self.count = List[Int](length=Self.BATCH, fill=0)

    def push_frame[
        so: MutOrigin = MutAnyOrigin
    ](mut self, b: Int, src: Pointer[Scalar[DT], so]):
        """Stage env `b`'s current frame (IMG_DIM values) in its next slot.
        Overwrite-safe: re-pushing before `push_action` replaces the frame."""
        var base = (b * Self.T + self.count[b] % Self.T) * Self.IMG_DIM
        for i in range(Self.IMG_DIM):
            self.frames[base + i] = src[unsafe_offset=i]

    def push_action[
        so: MutOrigin = MutAnyOrigin
    ](mut self, b: Int, src: Pointer[Scalar[DT], so]):
        """Record the block (ACT values) executed after the staged frame and
        complete the pair (advances the ring)."""
        var base = (b * Self.T + self.count[b] % Self.T) * Self.ACT
        for i in range(Self.ACT):
            self.actions[base + i] = src[unsafe_offset=i]
        self.count[b] += 1

    def ready(self) -> Bool:
        """True once at least one env has a full window (donor exists)."""
        for b in range(Self.BATCH):
            if self.count[b] >= Self.T:
                return True
        return False

    def fill[
        po: MutOrigin = MutAnyOrigin,
        ao: MutOrigin = MutAnyOrigin,
    ](
        self,
        pix_out: Pointer[Scalar[DT], po],
        act_out: Pointer[Scalar[DT], ao],
    ) -> Bool:
        """Write the training tiles — pix `(BATCH, T·IMG_DIM)`, act
        `(BATCH, T·ACT)` — oldest→newest per env. Envs with a partial ring
        (done before T pairs) copy the donor's window. False if no env has
        a full window yet (nothing written)."""
        var donor = -1
        for b in range(Self.BATCH):
            if self.count[b] >= Self.T:
                donor = b
                break
        if donor < 0:
            return False
        for b in range(Self.BATCH):
            var s = b if self.count[b] >= Self.T else donor
            for t in range(Self.T):
                var slot = (self.count[s] - Self.T + t) % Self.T
                var fsrc = (s * Self.T + slot) * Self.IMG_DIM
                var fdst = (b * Self.T + t) * Self.IMG_DIM
                for i in range(Self.IMG_DIM):
                    pix_out[unsafe_offset=fdst + i] = self.frames[fsrc + i]
                var asrc = (s * Self.T + slot) * Self.ACT
                var adst = (b * Self.T + t) * Self.ACT
                for i in range(Self.ACT):
                    act_out[unsafe_offset=adst + i] = self.actions[asrc + i]
        return True
