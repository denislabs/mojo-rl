"""Cycle-accurate TIA primitives (Stage 1 of the cycle-accurate TIA).

These model Stella's per-color-clock object timing so that per-clock collision is
correct (see docs/ATARI_CYCLE_ACCURATE_TIA.md). They are NEW, standalone building
blocks — the production renderer still uses the absolute-position model in
tables.mojo until the cycle-accurate path is wired up and validated.

References (references/stella-master/src/emucore/tia/):
  - DelayQueue.{hxx,cxx} + TIA.cxx Delay enum (write latencies)
  - Ball.{hxx,cxx} (counter/decode/movement) — the simplest object, modelled here
  - TIA.cxx resxCounter() (strobe → counter value)

Constants (TIAConstants.hxx): H_PIXEL=160 (counter wrap), H_CLOCKS=228,
H_BLANK_CLOCKS=68, resxLateHblankThreshold=73.
"""




# TIA register write latencies, in color clocks (TIA.cxx:29-53 `Delay`).
comptime DELAY_HMOVE: Int = 6
comptime DELAY_PF: Int = 2
comptime DELAY_GRP: Int = 1
comptime DELAY_HMP: Int = 2
comptime DELAY_HMM: Int = 2
comptime DELAY_HMBL: Int = 2
comptime DELAY_HMCLR: Int = 2
comptime DELAY_REFP: Int = 1
comptime DELAY_ENABL: Int = 1
comptime DELAY_ENAM: Int = 1
comptime DELAY_VBLANK: Int = 1

comptime H_PIXEL: Int = 160  # object counter wrap
comptime BALL_DECODE: Int = 156  # Ball.hxx: myCounter == 156 starts rendering
comptime RENDER_COUNTER_OFFSET: Int = -4  # Ball.hxx renderCounterOffset

# resxCounter() values (TIA.cxx:55) for a RESxx strobe.
comptime RESX_HBLANK: Int = 159
comptime RESX_LATE_HBLANK: Int = 158
comptime RESX_FRAME: Int = 157

# lit_horizon() return value for "this object can never be lit with its
# current config" (disabled / blank pattern / RESMP-locked).
comptime LIT_NEVER: Int = 1 << 30


@always_inline
def tia_write_clock(start_clock: Int, cycles: Int) -> Int:
    """Color clock at which a 6502 store to a TIA register takes effect.

    A store writes on its LAST cycle, so the write lands (cycles-1) CPU cycles
    after the instruction start; each CPU cycle is 3 color clocks. `start_clock`
    is the beam color-clock at instruction start (AtariState.clock). This is the
    exact generalization of the old `_resp_pos` `+6` constant (which only held
    for the 3-cycle `STA zp`)."""
    return start_clock + (cycles - 1) * 3


@always_inline
def playfield_bit(
    pf0: UInt8, pf1: UInt8, pf2: UInt8, reflect: Bool, pixel: Int
) -> Bool:
    """Playfield on/off at `pixel` from PF0/PF1/PF2 (same mapping as
    tables.playfield_mask, but on plain register values for the beam-accurate
    cycle shadow). PF0 bits 4-7, PF1 bits 7-0 (reversed), PF2 bits 0-7; the right
    half repeats or mirrors at pixel 80 by CTRLPF reflect."""
    var x = pixel >> 2  # 4 color clocks per PF bit → 40 PF cells
    var pf_bit: Int
    if x < 20:
        pf_bit = x
    else:
        pf_bit = (39 - x) if reflect else (x - 20)
    if pf_bit < 4:
        return ((pf0 >> UInt8(pf_bit + 4)) & 1) != 0
    elif pf_bit < 12:
        return ((pf1 >> UInt8(11 - pf_bit)) & 1) != 0
    else:
        return ((pf2 >> UInt8(pf_bit - 12)) & 1) != 0


@always_inline
def resx_counter(hctr: Int, in_hblank: Bool) -> Int:
    """Strobe → object counter value (Stella TIA::resxCounter, TIA.cxx:2149).

    `hctr` is the color clock within the 228-clock line (0-227); `in_hblank` is
    True during (extended) HBLANK. In normal HBLANK the strobe lands at 159, in
    late/extended HBLANK (hctr >= 73) at 158, and in the visible frame at 157.
    """
    if in_hblank:
        return RESX_LATE_HBLANK if hctr >= 73 else RESX_HBLANK
    return RESX_FRAME


# ---------------------------------------------------------------------------
# DelayQueue: schedules register writes to take effect N color clocks later.
# ---------------------------------------------------------------------------

comptime DQ_CAP: Int = 16  # plenty: at most a couple writes pending per clock


struct DelayQueue(Copyable, Movable):
    """Fixed-capacity queue of pending TIA register writes (TIA.cxx myDelayQueue).

    Each entry counts down its remaining color clocks. `cycle_collect` is called
    once per color clock: it appends the (reg, value) of any entry that is due
    *this* clock to `due`, frees that slot, then decrements the rest — matching
    Stella's execute-then-advance ordering so a delay of N means "applied N clocks
    after the push".
    """

    var valid: InlineArray[Bool, DQ_CAP]
    var remaining: InlineArray[Int, DQ_CAP]
    var reg: InlineArray[UInt8, DQ_CAP]
    var value: InlineArray[UInt8, DQ_CAP]
    var count: Int  # live entries — lets cycle_collect early-out (hot path)

    def __init__(out self):
        self.valid = InlineArray[Bool, DQ_CAP](fill=False)
        self.remaining = InlineArray[Int, DQ_CAP](fill=0)
        self.reg = InlineArray[UInt8, DQ_CAP](fill=0)
        self.value = InlineArray[UInt8, DQ_CAP](fill=0)
        self.count = 0

    def push(mut self, reg: UInt8, value: UInt8, delay: Int):
        """Schedule `reg=value` to take effect `delay` color clocks from now."""
        for i in range(DQ_CAP):
            if not self.valid[i]:
                self.valid[i] = True
                self.remaining[i] = delay
                self.reg[i] = reg
                self.value[i] = value
                self.count += 1
                return
        # Overflow: drop oldest-equivalent (should never happen in practice).

    @always_inline
    def cycle_collect(
        mut self, mut due_reg: List[UInt8], mut due_val: List[UInt8]
    ):
        """Advance one color clock; append writes that fire this clock.

        Called once per color clock (~60k/frame); the queue is empty on the
        vast majority of clocks, so the count==0 early-out is the difference
        between O(1) and 2×DQ_CAP scans per clock.
        """
        if self.count == 0:
            return
        for i in range(DQ_CAP):
            if self.valid[i] and self.remaining[i] == 0:
                due_reg.append(self.reg[i])
                due_val.append(self.value[i])
                self.valid[i] = False
                self.count -= 1
        for i in range(DQ_CAP):
            if self.valid[i]:
                self.remaining[i] -= 1

    def pending(self) -> Int:
        var n = 0
        for i in range(DQ_CAP):
            if self.valid[i]:
                n += 1
        return n


# ---------------------------------------------------------------------------
# BallCounter: faithful port of Stella Ball counter/decode/movement.
# ---------------------------------------------------------------------------


struct BallCounter(Copyable, Movable):
    """Per-color-clock ball position model (Stella Ball.{hxx,cxx}).

    The ball has a horizontal counter that ticks every color clock and wraps at
    H_PIXEL. At counter==156 it starts rendering (render_counter = -4); the ball
    is "lit" (its collision/signal output) while render_counter is in [0, width).
    `tick()` returns the lit state for the clock being processed, computed at
    entry exactly like Stella (signal uses the pre-increment render_counter).
    Starfield/inverted-phase-clock corner cases are omitted for v1.
    """

    var counter: Int
    var render_counter: Int
    var is_rendering: Bool
    var width: Int  # 1,2,4,8 from CTRLPF bits 4-5
    # VDELBL double-buffer (same model as the player): ENABL sets enabl_new; a
    # GRP1 write shuffles enabl_old=enabl_new (Stella shuffleBL); VDELBL selects.
    # Resolved at RENDER time in tick() — resolving at write time (the old single
    # `enabled`) mis-handled the Breakout ball's vertical-delay.
    var enabl_new: Bool
    var enabl_old: Bool
    var vdel: Bool
    # HMOVE movement
    var is_moving: Bool
    var hmm_clocks: Int  # target movement-clock count (HMBL decoded)

    def __init__(out self):
        self.counter = 0
        self.render_counter = 0
        self.is_rendering = False
        self.width = 1
        self.enabl_new = False
        self.enabl_old = False
        self.vdel = False
        self.is_moving = False
        self.hmm_clocks = 0

    @always_inline
    def set_enabl_new(mut self, on: Bool):
        self.enabl_new = on

    @always_inline
    def shuffle(mut self):
        self.enabl_old = self.enabl_new

    @always_inline
    def set_vdel(mut self, value: Bool):
        self.vdel = value

    @always_inline
    def set_width_from_ctrlpf(mut self, ctrlpf: UInt8):
        var bits = Int((ctrlpf >> 4) & 0x03)
        self.width = 1 << bits  # 1,2,4,8

    @always_inline
    def set_hmbl(mut self, value: UInt8):
        # Ball.cxx: myHmmClocks = (value >> 4) ^ 0x08
        self.hmm_clocks = Int((value >> 4) ^ 0x08)

    @always_inline
    def resbl(mut self, strobe_counter: Int):
        """RESBL strobe: set counter from resxCounter() value (Ball.cxx:74)."""
        self.counter = strobe_counter
        self.is_rendering = True
        self.render_counter = RENDER_COUNTER_OFFSET + (strobe_counter - 157)

    @always_inline
    def start_movement(mut self):
        self.is_moving = True

    @always_inline
    def tick(mut self) -> Bool:
        """Advance one color clock; return the ball's lit (collision) state."""
        var en = self.enabl_old if self.vdel else self.enabl_new
        var lit = self.is_rendering and self.render_counter >= 0 and en

        if self.counter == BALL_DECODE:
            self.is_rendering = True
            self.render_counter = RENDER_COUNTER_OFFSET
        elif self.is_rendering:
            self.render_counter += 1
            if self.render_counter >= self.width:
                self.is_rendering = False

        self.counter += 1
        if self.counter >= H_PIXEL:
            self.counter = 0

        return lit

    @always_inline
    def movement_tick(mut self, movement_counter: Int, in_hblank: Bool):
        """One HMOVE movement tick (Ball.cxx:358 movementTick, simplified).

        Called every 4 color clocks while a HMOVE is in progress. Stops once the
        movement counter reaches the decoded HMBL value; otherwise, during HBLANK,
        nudges the ball one pixel (an extra counter tick)."""
        if not self.is_moving:
            return
        if movement_counter == self.hmm_clocks:
            self.is_moving = False
        elif in_hblank:
            _ = self.tick()

    def advance_n(mut self, n: Int):
        """Advance exactly n visible ticks (== n tick() calls), discarding lit.

        O(1) per segment everywhere: outside render windows it skips straight
        to the next decode; INSIDE them it crosses the window in closed form
        (rc increments once per tick until it reaches `width`), bounded by the
        next decode tick — bit-exact vs per-tick by the property test.
        Valid only while the config (width/vdel/enables) is static, i.e. the
        bulk fast path's no-pending-writes precondition."""
        var rem = n
        while rem > 0:
            var d = (BALL_DECODE - self.counter + H_PIXEL) % H_PIXEL
            if self.is_rendering:
                if d == 0:
                    # Decode tick restarts the window — process exactly.
                    _ = self.tick()
                    rem -= 1
                    continue
                var k = min(rem, d)
                # max(1, ·): width may have SHRUNK below rc since the window
                # started (CTRLPF write between bulk spans) — the reference
                # tick still increments rc once before ending the window.
                var t_end = max(1, self.width - self.render_counter)
                if t_end <= k:
                    # Window ends inside the segment; the remaining ticks
                    # (no decode within k) only advance the counter.
                    self.render_counter += t_end
                    self.is_rendering = False
                else:
                    self.render_counter += k
                self.counter = (self.counter + k) % H_PIXEL
                rem -= k
                continue
            if d >= rem:
                self.counter = (self.counter + rem) % H_PIXEL
                return
            self.counter = (self.counter + d) % H_PIXEL
            rem -= d
            _ = self.tick()  # the decode tick
            rem -= 1

    @always_inline
    def lit_horizon(self) -> Int:
        """Number of upcoming visible ticks guaranteed to produce lit=False
        (assuming no register writes land in the window). Conservative."""
        var en = self.enabl_old if self.vdel else self.enabl_new
        if not en:
            return LIT_NEVER
        if self.is_rendering:
            return 0
        return (BALL_DECODE - self.counter + H_PIXEL) % H_PIXEL


# ---------------------------------------------------------------------------
# Shared decode table (Stella DrawCounterDecodes): copies decode at counter 156
# (main) plus NUSIZ-dependent offsets for the extra copies. Same table for
# players and missiles (DrawCounterDecodes.cxx).
# ---------------------------------------------------------------------------


@always_inline
def decode_offset(nusiz_copies: Int, counter: Int) -> Int:
    """Visible ticks from `counter` until the next decode tick — i.e. the
    smallest i >= 0 such that decode_copy(nusiz_copies, counter + i) != 0
    (mod the 160-clock counter cycle). The bulk fast path uses this to skip
    straight to the next render window in O(1)."""
    var best = (156 - counter + H_PIXEL) % H_PIXEL
    if nusiz_copies == 1 or nusiz_copies == 3:
        best = min(best, (12 - counter + H_PIXEL) % H_PIXEL)
    if nusiz_copies == 2 or nusiz_copies == 3 or nusiz_copies == 6:
        best = min(best, (28 - counter + H_PIXEL) % H_PIXEL)
    if nusiz_copies == 4 or nusiz_copies == 6:
        best = min(best, (60 - counter + H_PIXEL) % H_PIXEL)
    return best


@always_inline
def decode_copy(nusiz_copies: Int, counter: Int) -> Int:
    """Return the copy number (1..3) decoding at `counter`, else 0.

    nusiz_copies = NUSIZ bits 0-2. Offsets: +16->counter 12, +32->28, +64->60
    (i.e. (156+offset) mod 160). Matches DrawCounterDecodes::DrawCounterDecodes.
    """
    if counter == 156:
        return 1
    if nusiz_copies == 1:
        if counter == 12:
            return 2
    elif nusiz_copies == 2:
        if counter == 28:
            return 2
    elif nusiz_copies == 3:
        if counter == 12:
            return 2
        if counter == 28:
            return 3
    elif nusiz_copies == 4:
        if counter == 60:
            return 2
    elif nusiz_copies == 6:
        if counter == 28:
            return 2
        if counter == 60:
            return 3
    return 0


# ---------------------------------------------------------------------------
# PlayerCounter: faithful port of Stella Player counter/decode/divider.
# ---------------------------------------------------------------------------

comptime PLAYER_RENDER_OFFSET: Int = -5  # Player.hxx renderCounterOffset


struct PlayerCounter(Copyable, Movable):
    """Per-color-clock player position model (Stella Player.{hxx,cxx}).

    Counter ticks every clock, wraps at 160. A copy decodes at counter 156 (+
    NUSIZ offsets); rendering then walks an 8-bit pattern via `sample_counter`,
    stepped at a rate set by `divider` (1/2/4 = normal/double/quad width). The
    player is lit when rendering, render_counter >= trip_point, and the pattern
    bit at sample_counter is set. Starfield / mid-render divider changes omitted.
    """

    var counter: Int
    var render_counter: Int
    var sample_counter: Int
    var is_rendering: Bool
    var copy_num: Int
    var divider: Int
    var trip_point: Int
    # VDEL double-buffer: GRP writes set grp_new; the OTHER player's GRP write
    # shuffles grp_old=grp_new (Stella shuffleP0/P1). The displayed pattern is
    # grp_old when vdel (VDELPx) is set, else grp_new — resolved at RENDER time
    # (in tick), exactly like the eol player_mask. Resolving at write time (the
    # old `grp` field) left killed-invader GRP clears masked by a stale old
    # pattern → "ghost" invaders.
    var grp_new: UInt8
    var grp_old: UInt8
    var vdel: Bool
    var reflect: Bool
    var nusiz_copies: Int
    var is_moving: Bool
    var hmm_clocks: Int

    def __init__(out self):
        self.counter = 0
        self.render_counter = 0
        self.sample_counter = 0
        self.is_rendering = False
        self.copy_num = 0
        self.divider = 1
        self.trip_point = 0
        self.grp_new = 0
        self.grp_old = 0
        self.vdel = False
        self.reflect = False
        self.nusiz_copies = 0
        self.is_moving = False
        self.hmm_clocks = 0

    @always_inline
    def set_nusiz(mut self, value: UInt8):
        self.nusiz_copies = Int(value & 0x07)
        if self.nusiz_copies == 5:
            self.divider = 2
        elif self.nusiz_copies == 7:
            self.divider = 4
        else:
            self.divider = 1
        self.trip_point = 0 if self.divider == 1 else 1

    @always_inline
    def set_grp_new(mut self, value: UInt8):
        self.grp_new = value

    @always_inline
    def shuffle(mut self):
        # The other player's GRP write copies new -> old (Stella shufflePatterns).
        self.grp_old = self.grp_new

    @always_inline
    def set_vdel(mut self, value: Bool):
        self.vdel = value

    @always_inline
    def set_reflect(mut self, reflect: Bool):
        self.reflect = reflect

    @always_inline
    def set_hmp(mut self, value: UInt8):
        self.hmm_clocks = Int((value >> 4) ^ 0x08)

    @always_inline
    def resp(mut self, strobe_counter: Int):
        self.counter = strobe_counter

    @always_inline
    def start_movement(mut self):
        self.is_moving = True

    @always_inline
    def movement_tick(mut self, movement_counter: Int, in_hblank: Bool):
        if not self.is_moving:
            return
        if movement_counter == self.hmm_clocks:
            self.is_moving = False
        elif in_hblank:
            _ = self.tick()

    @always_inline
    def _pattern_bit(self, sample: Int) -> Bool:
        # VDEL resolved here (render time): old pattern when delaying, else new.
        var g = self.grp_old if self.vdel else self.grp_new
        # Non-reflected: leftmost pixel = GRP bit 7 emitted first (sample 0).
        if self.reflect:
            return ((g >> UInt8(sample)) & 1) != 0
        return ((g >> UInt8(7 - sample)) & 1) != 0

    @always_inline
    def tick(mut self) -> Bool:
        """Advance one color clock; return the player's lit (signal) state."""
        var lit = False
        if self.is_rendering and self.render_counter >= self.trip_point:
            if self.sample_counter >= 0 and self.sample_counter <= 7:
                lit = self._pattern_bit(self.sample_counter)

        var d = decode_copy(self.nusiz_copies, self.counter)
        if d != 0:
            self.is_rendering = True
            self.sample_counter = 0
            self.render_counter = PLAYER_RENDER_OFFSET
            self.copy_num = d
        elif self.is_rendering:
            self.render_counter += 1
            if self.divider == 1:
                if self.render_counter > 0:
                    self.sample_counter += 1
            else:
                if (
                    self.render_counter > 1
                    and ((self.render_counter - 1) & (self.divider - 1)) == 0
                ):
                    self.sample_counter += 1
            if self.sample_counter > 7:
                self.is_rendering = False

        self.counter += 1
        if self.counter >= H_PIXEL:
            self.counter = 0
        return lit

    def advance_n(mut self, n: Int):
        """Advance exactly n visible ticks, discarding lit (see BallCounter).

        Render windows are crossed in closed form: per tick() the window
        advances `render_counter` and bumps `sample_counter` when
        rc > 0 (divider 1) / rc > 1 and (rc-1) % divider == 0 (divider
        2/4), ending when sample_counter exceeds 7. Both the window-end
        tick index and the increment count over k ticks have exact
        arithmetic forms; segments are bounded by the next decode tick
        (which restarts the window and is processed via tick())."""
        var rem = n
        while rem > 0:
            var d = decode_offset(self.nusiz_copies, self.counter)
            if self.is_rendering:
                if d == 0:
                    _ = self.tick()  # decode tick restarts the window
                    rem -= 1
                    continue
                var k = min(rem, d)
                var rc0 = self.render_counter
                var need = 8 - self.sample_counter
                if self.divider == 1:
                    # Gate: rc0 + t > 0 → the first max(0, -rc0) ticks are
                    # silent, then one increment per tick.
                    var t_end = max(0, -rc0) + need
                    if t_end <= k:
                        self.render_counter = rc0 + t_end
                        self.sample_counter = 8
                        self.is_rendering = False
                    else:
                        self.render_counter = rc0 + k
                        self.sample_counter += k - min(k, max(0, -rc0))
                else:
                    # Gate: m = rc0 + t - 1 must be a positive multiple of
                    # `divider`. First valid m is the smallest multiple of
                    # divider >= max(rc0, 1); the need-th is first + (need-1)·divider.
                    var first_m = (
                        (max(rc0, 1) + self.divider - 1) // self.divider
                    ) * self.divider
                    var t_end = first_m + (need - 1) * self.divider - rc0 + 1
                    if t_end <= k:
                        self.render_counter = rc0 + t_end
                        self.sample_counter = 8
                        self.is_rendering = False
                    else:
                        self.render_counter = rc0 + k
                        var last_m = rc0 + k - 1
                        if last_m >= first_m:
                            self.sample_counter += (
                                last_m - first_m
                            ) // self.divider + 1
                self.counter = (self.counter + k) % H_PIXEL
                rem -= k
                continue
            if d >= rem:
                self.counter = (self.counter + rem) % H_PIXEL
                return
            self.counter = (self.counter + d) % H_PIXEL
            rem -= d
            _ = self.tick()  # the decode tick
            rem -= 1

    @always_inline
    def lit_horizon(self) -> Int:
        """Upcoming visible ticks guaranteed lit-free (see BallCounter)."""
        var g = self.grp_old if self.vdel else self.grp_new
        if g == 0:
            return LIT_NEVER
        if self.is_rendering:
            return 0
        return decode_offset(self.nusiz_copies, self.counter)


# ---------------------------------------------------------------------------
# MissileCounter: faithful port of Stella Missile counter/decode.
# ---------------------------------------------------------------------------

comptime MISSILE_RENDER_OFFSET: Int = -4  # Missile.hxx renderCounterOffset


struct MissileCounter(Copyable, Movable):
    """Per-color-clock missile position model (Stella Missile.{hxx,cxx}).

    Like the ball but with NUSIZ copies (shared decode table) and a RESMP lock
    that suppresses the decode (missile hidden / locked to player). Lit when
    rendering, render_counter >= 0, and enabled. Starfield/movement edge cases
    omitted for v1.
    """

    var counter: Int
    var render_counter: Int
    var is_rendering: Bool
    var copy_num: Int
    var width: Int
    var nusiz_copies: Int
    var enabled: Bool
    var resmp: Bool
    var is_moving: Bool
    var hmm_clocks: Int

    def __init__(out self):
        self.counter = 0
        self.render_counter = 0
        self.is_rendering = False
        self.copy_num = 0
        self.width = 1
        self.nusiz_copies = 0
        self.enabled = False
        self.resmp = False
        self.is_moving = False
        self.hmm_clocks = 0

    @always_inline
    def set_nusiz(mut self, value: UInt8):
        self.nusiz_copies = Int(value & 0x07)
        self.width = 1 << Int((value >> 4) & 0x03)

    @always_inline
    def set_enam(mut self, value: UInt8):
        self.enabled = (value & 0x02) != 0

    @always_inline
    def set_resmp(mut self, value: UInt8):
        self.resmp = (value & 0x02) != 0

    @always_inline
    def set_hmm(mut self, value: UInt8):
        self.hmm_clocks = Int((value >> 4) ^ 0x08)

    @always_inline
    def resm(mut self, strobe_counter: Int):
        self.counter = strobe_counter

    @always_inline
    def start_movement(mut self):
        self.is_moving = True

    @always_inline
    def movement_tick(mut self, movement_counter: Int, in_hblank: Bool):
        if not self.is_moving:
            return
        if movement_counter == self.hmm_clocks:
            self.is_moving = False
        elif in_hblank:
            _ = self.tick()

    @always_inline
    def tick(mut self) -> Bool:
        """Advance one color clock; return the missile's lit (signal) state."""
        var visible = self.is_rendering and self.render_counter >= 0
        var lit = visible and self.enabled

        var d = decode_copy(self.nusiz_copies, self.counter)
        if d != 0 and not self.resmp:
            self.is_rendering = True
            self.render_counter = MISSILE_RENDER_OFFSET
            self.copy_num = d
        elif self.is_rendering:
            self.render_counter += 1
            if self.render_counter >= self.width:
                self.is_rendering = False

        self.counter += 1
        if self.counter >= H_PIXEL:
            self.counter = 0
        return lit

    def advance_n(mut self, n: Int):
        """Advance exactly n visible ticks, discarding lit (see BallCounter).

        Render windows are crossed in closed form like the ball (rc counts
        to `width`); RESMP suppresses the decode, so a resmp'd missile
        advances unbounded by decode ticks."""
        var rem = n
        while rem > 0:
            if self.is_rendering:
                var d: Int
                if self.resmp:
                    d = rem  # decode suppressed — no restart possible
                else:
                    d = decode_offset(self.nusiz_copies, self.counter)
                    if d == 0:
                        _ = self.tick()  # decode tick restarts the window
                        rem -= 1
                        continue
                var k = min(rem, d)
                # max(1, ·): width may have shrunk below rc mid-window (NUSIZ
                # write between bulk spans) — see BallCounter.
                var t_end = max(1, self.width - self.render_counter)
                if t_end <= k:
                    self.render_counter += t_end
                    self.is_rendering = False
                else:
                    self.render_counter += k
                self.counter = (self.counter + k) % H_PIXEL
                rem -= k
                continue
            if self.resmp:
                # Decode suppressed and not rendering: pure counter advance.
                self.counter = (self.counter + rem) % H_PIXEL
                return
            var d = decode_offset(self.nusiz_copies, self.counter)
            if d >= rem:
                self.counter = (self.counter + rem) % H_PIXEL
                return
            self.counter = (self.counter + d) % H_PIXEL
            rem -= d
            _ = self.tick()  # the decode tick
            rem -= 1

    @always_inline
    def lit_horizon(self) -> Int:
        """Upcoming visible ticks guaranteed lit-free (see BallCounter)."""
        if not self.enabled:
            return LIT_NEVER
        if self.is_rendering:
            return 0
        if self.resmp:
            return LIT_NEVER
        return decode_offset(self.nusiz_copies, self.counter)


# ---------------------------------------------------------------------------
# CycleTIA: aggregate of all object counters + movement + delay queue.
# ---------------------------------------------------------------------------
# Pure (no AtariState dependency, to avoid a circular import). The integration
# glue that drives this from TIA register writes and renders/collides against
# the playfield lives in cpu6502.mojo (which imports both). Persisted in
# AtariState across frames so object positions carry over like real hardware.

# Lit-bit positions returned by CycleTIA.tick().
comptime LIT_P0: Int = 0
comptime LIT_P1: Int = 1
comptime LIT_M0: Int = 2
comptime LIT_M1: Int = 3
comptime LIT_BL: Int = 4


struct CycleTIA(Copyable, Movable):
    """All five TIA objects ticked together per color clock (Stella TIA::cycle).

    `tick(in_hblank)` performs HMOVE movement (every 4 clocks while in progress),
    then ticks each object's counter, returning a 5-bit mask of which objects are
    lit at this color clock (bits LIT_P0..LIT_BL). `hctr` is the color clock
    within the current 228-clock line; `nextLine()` resets it.
    """

    var p0: PlayerCounter
    var p1: PlayerCounter
    var m0: MissileCounter
    var m1: MissileCounter
    var bl: BallCounter
    var dq: DelayQueue
    var hctr: Int
    var movement_in_progress: Bool
    var movement_clock: Int
    # HMOVE extends HBLANK by 8 color clocks (Stella myExtendedHblank). During
    # those 8 clocks objects do NOT tick (they got `hmm_clocks` extra comb ticks
    # in HBLANK instead), so net motion = hmm_clocks - 8. Without this every
    # HMOVE'd object drifts 8px and rows positioned per-line stagger. Set on the
    # HMOVE strobe, cleared at line start. Also draws the 8px black "comb" bar.
    var extended_hblank: Bool
    # Beam-accurate playfield + color shadow. The playfield is also rewritten
    # mid-line (Breakout brick rows, Pong score), so it must be applied at the
    # exact clock — not read live from AtariState — or the rendered frame (and
    # thus collisions) flicker. Updated by the cycle runner from logged writes.
    var pf0: UInt8
    var pf1: UInt8
    var pf2: UInt8
    var ctrlpf: UInt8  # reflect(bit0)/score(bit1)/priority(bit2) + ball width
    var colup0: UInt8
    var colup1: UInt8
    var colupf: UInt8
    var colubk: UInt8
    # Deferred bulk advance: visible ticks accumulated by the runner's bulk
    # fast path but not yet applied to the five object counters. Flushed
    # (one advance_objects call) before anything reads or strobes object
    # state — per-clock entry, delayed-write applies, frame end. Cuts the
    # per-sub-span 5×advance_n + 5×lit_horizon bookkeeping (~8-clock spans,
    # instruction-bounded) down to two adds per sub-span.
    var pending_ticks: Int

    def __init__(out self):
        self.p0 = PlayerCounter()
        self.p1 = PlayerCounter()
        self.m0 = MissileCounter()
        self.m1 = MissileCounter()
        self.bl = BallCounter()
        self.dq = DelayQueue()
        self.hctr = 0
        self.movement_in_progress = False
        self.movement_clock = 0
        self.extended_hblank = False
        self.pf0 = 0
        self.pf1 = 0
        self.pf2 = 0
        self.ctrlpf = 0
        self.colup0 = 0
        self.colup1 = 0
        self.colupf = 0
        self.colubk = 0
        self.pending_ticks = 0

    @always_inline
    def start_hmove(mut self):
        """HMOVE strobe: begin movement on all objects (TIA delayedWrite HMOVE)."""
        self.movement_in_progress = True
        self.movement_clock = 0
        self.extended_hblank = True
        self.p0.start_movement()
        self.p1.start_movement()
        self.m0.start_movement()
        self.m1.start_movement()
        self.bl.start_movement()

    @always_inline
    def _movement(mut self, in_hblank: Bool):
        if not self.movement_in_progress:
            return
        if (self.hctr & 0x03) == 0:
            var mc = 0 if self.movement_clock > 15 else self.movement_clock
            self.m0.movement_tick(mc, in_hblank)
            self.m1.movement_tick(mc, in_hblank)
            self.p0.movement_tick(mc, in_hblank)
            self.p1.movement_tick(mc, in_hblank)
            self.bl.movement_tick(mc, in_hblank)
            self.movement_clock += 1
            if self.movement_clock > 15:
                self.movement_in_progress = False

    @always_inline
    def tick(mut self, in_hblank: Bool) -> UInt8:
        """Advance all objects one color clock; return the 5-bit lit mask.

        CRITICAL: object counters advance ONLY during the visible 160 clocks
        (Stella: tickHframe ticks the counter; tickHblank does not). During
        HBLANK we only run HMOVE movement (the comb adds extra ticks there).
        Ticking every clock would gain +68 counter steps per scanline → every
        object drifts diagonally and fragments."""
        if in_hblank:
            self._movement(in_hblank)
            return 0
        var bits: UInt8 = 0
        if self.p0.tick():
            bits |= UInt8(1 << LIT_P0)
        if self.p1.tick():
            bits |= UInt8(1 << LIT_P1)
        if self.m0.tick():
            bits |= UInt8(1 << LIT_M0)
        if self.m1.tick():
            bits |= UInt8(1 << LIT_M1)
        if self.bl.tick():
            bits |= UInt8(1 << LIT_BL)
        return bits

    @always_inline
    def lit_safe_horizon(self) -> Int:
        """Visible ticks from now during which NO object can be lit (so no
        collision pair can latch), assuming no pending writes/movement."""
        var safe = self.p0.lit_horizon()
        safe = min(safe, self.p1.lit_horizon())
        safe = min(safe, self.m0.lit_horizon())
        safe = min(safe, self.m1.lit_horizon())
        return min(safe, self.bl.lit_horizon())

    @always_inline
    def advance_objects(mut self, n: Int):
        """Advance all five objects exactly n visible ticks (bulk fast path)."""
        self.p0.advance_n(n)
        self.p1.advance_n(n)
        self.m0.advance_n(n)
        self.m1.advance_n(n)
        self.bl.advance_n(n)

    @always_inline
    def flush_pending(mut self):
        """Apply deferred visible ticks (see `pending_ticks`) to the five
        object counters. Free when nothing is pending."""
        if self.pending_ticks > 0:
            var n = self.pending_ticks
            self.pending_ticks = 0
            self.advance_objects(n)

    @always_inline
    def next_line(mut self):
        self.hctr = 0
