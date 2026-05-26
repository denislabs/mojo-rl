"""J.1.b spike — TrainerGraph[*BLOCKS] walker.

Proves the three substrate properties before any SAC code moves:

  P1. `Tuple[*BLOCKS: TrainerBlock]` constructs from variadic trait pack
      (mirrors ComputeGraph's `Tuple[*Self.NODES]`).
  P2. `comptime for k in range(N): blocks[k].step_via(mut state)`
      walks blocks and lets each one mutate shared state.
  P3. A block can hold `UnsafePointer[OwnerStruct, MutAnyOrigin]` to
      simulate model wiring (block borrows trainer's model field).
  P4. `state.did_step = False` from any block short-circuits the walker
      (matches the audit's "skip the rest of train_step on under-filled
      buffer" semantics).

If these all pass, the substrate is good and J.1.c (SAC migration) can
proceed by lifting each helper into a TrainerBlock-conforming struct.
"""

# UnsafePointer is a builtin in mojo-nightly (no import needed).


# ──────────────────────────────────────────────────────────────────────
# TrainerState — minimal flow struct. The real one will carry minibatch
# Scratch fields; here we use plain Ints to keep the spike standalone.
# ──────────────────────────────────────────────────────────────────────


struct TrainerState(Defaultable & Movable & ImplicitlyDestructible):
    var step_idx: Int
    var visit_count: Int          # incremented by every block
    var alpha: Float64            # written by AlphaBlock, read by TargetYBlock
    var log_prob_mean: Float64    # written by ActorBlock, read by AlphaBlock
    var critic_loss: Float64
    var actor_loss: Float64
    var did_step: Bool

    def __init__(out self):
        self.step_idx = 0
        self.visit_count = 0
        self.alpha = 0.0
        self.log_prob_mean = 0.0
        self.critic_loss = 0.0
        self.actor_loss = 0.0
        self.did_step = True


# ──────────────────────────────────────────────────────────────────────
# TrainerBlock trait — same shape as the J.1.a sketch.
# Single uniform `step_via(mut state)` interface; no model params on
# the trait itself (blocks parametrise themselves with model types).
# ──────────────────────────────────────────────────────────────────────


trait TrainerBlock(Defaultable & Movable & ImplicitlyDestructible):
    def step_via(mut self, mut state: TrainerState) raises:
        ...


# ──────────────────────────────────────────────────────────────────────
# Owner stand-in for "trainer's model field". A block holds an
# UnsafePointer to a ModelStub and dereferences it during step.
# Real blocks will hold `UnsafePointer[OnlineTargetPair[CRITIC], ...]`.
# ──────────────────────────────────────────────────────────────────────


struct ModelStub(Defaultable & Movable & ImplicitlyDestructible):
    var value: Float64

    def __init__(out self):
        self.value = 42.0


# ──────────────────────────────────────────────────────────────────────
# Concrete blocks
# ──────────────────────────────────────────────────────────────────────


struct SampleBlockSpike(TrainerBlock):
    """P4 candidate — flips did_step to False when step_idx < 2."""
    var min_step: Int

    def __init__(out self):
        self.min_step = 2

    def step_via(mut self, mut state: TrainerState) raises:
        state.visit_count += 1
        if state.step_idx < self.min_step:
            state.did_step = False
            return
        # else: leave did_step=True, walker continues


struct TargetYBlockSpike(TrainerBlock):
    """Reads state.alpha (set by AlphaBlock in prior step), writes nothing
    interesting. Simulates target_y compute reading alpha."""

    def __init__(out self):
        pass

    def step_via(mut self, mut state: TrainerState) raises:
        state.visit_count += 1
        # would compute target_y here using state.alpha + state.mb_sp
        _ = state.alpha  # read


struct CriticUpdateBlockSpike(TrainerBlock):
    """P3 — holds UnsafePointer to ModelStub (simulates `pair1_ptr`)."""

    var model_ptr: UnsafePointer[ModelStub, MutAnyOrigin]

    def __init__(out self):
        self.model_ptr = UnsafePointer[ModelStub, MutAnyOrigin](
            unsafe_from_address=0,
        )

    def bind(mut self, ptr: UnsafePointer[ModelStub, MutAnyOrigin]):
        self.model_ptr = ptr

    def step_via(mut self, mut state: TrainerState) raises:
        state.visit_count += 1
        # Deref the bound model — proves typed-pointer wiring works.
        var v = self.model_ptr[].value
        state.critic_loss = v * 0.001


struct ActorBlockSpike(TrainerBlock):
    """Writes log_prob_mean (consumed by AlphaBlock next)."""

    def __init__(out self):
        pass

    def step_via(mut self, mut state: TrainerState) raises:
        state.visit_count += 1
        state.actor_loss = 0.5
        state.log_prob_mean = -1.234


struct AlphaBlockSpike(TrainerBlock):
    """Reads log_prob_mean (written by ActorBlock above) → writes alpha."""

    def __init__(out self):
        pass

    def step_via(mut self, mut state: TrainerState) raises:
        state.visit_count += 1
        # alpha = -log_prob_mean (toy update rule)
        state.alpha = -state.log_prob_mean


# ──────────────────────────────────────────────────────────────────────
# TrainerGraph walker — mirrors ComputeGraph's structure.
# ──────────────────────────────────────────────────────────────────────


struct TrainerGraph[*BLOCKS: TrainerBlock](Movable & ImplicitlyDestructible):
    comptime N = Self.BLOCKS.size

    var blocks: Tuple[*Self.BLOCKS]
    var state: TrainerState

    def __init__(out self):
        comptime assert Self.N >= 1, "TrainerGraph requires at least one block"
        self.blocks = Tuple[*Self.BLOCKS]()
        self.state = TrainerState()

    def step(mut self, step_idx: Int) raises -> Bool:
        self.state.step_idx = step_idx
        self.state.did_step = True
        comptime for k in range(Self.N):
            self.blocks[k].step_via(self.state)
            if not self.state.did_step:
                return False
        return True


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────


def main() raises:
    print("[J.1.b] TrainerGraph spike")
    print("======================================")

    # P1 — construct TrainerGraph with a 5-block tuple.
    print("[P1] Tuple[*BLOCKS: TrainerBlock] construction ...")
    var g = TrainerGraph[
        SampleBlockSpike,
        TargetYBlockSpike,
        CriticUpdateBlockSpike,
        ActorBlockSpike,
        AlphaBlockSpike,
    ]()
    print("  OK — Tuple of 5 trait-typed blocks constructed.")

    # P3 — bind the model pointer into CriticUpdateBlockSpike.
    print("[P3] Block holds UnsafePointer to external owner ...")
    var model = ModelStub()
    g.blocks[2].bind(UnsafePointer(to=model))
    print("  OK — UnsafePointer[ModelStub, MutAnyOrigin] bound into block[2].")

    # P4 — first 2 steps short-circuit (sample block sets did_step=False).
    print("[P4] short-circuit on did_step=False ...")
    var ran_0 = g.step(0)
    if ran_0 or g.state.visit_count != 1:
        raise Error(
            "P4 FAIL: step 0 should short-circuit after SampleBlock; "
            + "ran_0=" + String(ran_0)
            + " visit_count=" + String(g.state.visit_count)
        )
    print("  OK — step 0 short-circuits (ran=False, visit_count=1).")

    var ran_1 = g.step(1)
    if ran_1 or g.state.visit_count != 2:
        raise Error("P4 FAIL: step 1 should short-circuit after SampleBlock")
    print("  OK — step 1 short-circuits (ran=False, visit_count=2).")

    # P2 — step 2 runs the full chain (5 blocks visit state).
    g.state.visit_count = 0   # reset for clean count
    print("[P2] comptime-for walker mutates state across all blocks ...")
    var ran_2 = g.step(2)
    if not ran_2:
        raise Error("P2 FAIL: step 2 should run full chain (ran=True)")
    if g.state.visit_count != 5:
        raise Error(
            "P2 FAIL: expected visit_count=5, got "
            + String(g.state.visit_count)
        )
    # Verify inter-block flow: ActorBlock wrote log_prob_mean,
    # AlphaBlock read it and wrote alpha. So alpha == -(-1.234) == 1.234.
    if g.state.alpha != 1.234:
        raise Error(
            "P2 FAIL: inter-block flow broken. alpha="
            + String(g.state.alpha) + " expected 1.234"
        )
    # Verify CriticBlock dereffed model_ptr: critic_loss = 42.0 * 0.001 = 0.042
    if g.state.critic_loss != 0.042:
        raise Error(
            "P3 FAIL: model pointer deref didn't produce expected value. "
            + "critic_loss=" + String(g.state.critic_loss)
            + " expected 0.042"
        )
    print("  OK — 5 blocks visited, inter-block flow + model deref verified.")

    # Lifetime extender for the owner pointer (mirrors set_external usage
    # in real graph nodes). Spike-only; in the real trainer the model is
    # a struct field whose lifetime is the trainer's.
    _ = model^

    print("======================================")
    print("[J.1.b] ALL PROPERTIES PASS — substrate is good")
