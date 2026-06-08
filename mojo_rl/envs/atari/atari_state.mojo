"""Atari 2600 per-instance state.

Flat struct holding all emulator state for one environment instance.
Designed to be GPU-friendly (no pointers, fixed size, ~350 bytes).

Ported from CuLE (BSD-3): cule/atari/state.hpp, frame_state.hpp
"""

from .flags import RAM_SIZE


struct AtariState(Copyable, Movable):
    """Complete state of one Atari 2600 instance.

    All fields are value types — no pointers. This struct can be stored
    in GPU global memory and accessed per-thread.
    """

    # ========================================================================
    # 6502 CPU Registers (7 bytes)
    # ========================================================================
    var pc: UInt16  # Program counter
    var a: UInt8  # Accumulator
    var x: UInt8  # X index register
    var y: UInt8  # Y index register
    var sp: UInt8  # Stack pointer
    var flags: UInt8  # Status register (NV-BDIZC)

    # ========================================================================
    # System flags (4 bytes)
    # ========================================================================
    var sys_flags: UInt32  # Controller state, console switches, ALE flags

    # ========================================================================
    # TIA State (~60 bytes)
    # ========================================================================
    var tia_flags: UInt32  # TIA boolean flags (VBLANK, VSYNC, HMOVE, etc.)
    var collision: UInt16  # 15 collision bits packed

    # Graphics registers
    var grp0: UInt8  # Player 0 graphics
    var grp1: UInt8  # Player 1 graphics
    var grp0_old: UInt8  # Player 0 delayed graphics
    var grp1_old: UInt8  # Player 1 delayed graphics
    var enam0: UInt8  # Missile 0 enable (bit 1)
    var enam1: UInt8  # Missile 1 enable (bit 1)
    var enabl: UInt8  # Ball enable (bit 1)
    var enabl_old: UInt8  # Ball delayed enable

    # Position registers
    var pos_p0: UInt8  # Player 0 position (0-159)
    var pos_p1: UInt8  # Player 1 position
    var pos_m0: UInt8  # Missile 0 position
    var pos_m1: UInt8  # Missile 1 position
    var pos_bl: UInt8  # Ball position

    # Horizontal motion registers (signed 4-bit, stored as full byte)
    var hm_p0: UInt8
    var hm_p1: UInt8
    var hm_m0: UInt8
    var hm_m1: UInt8
    var hm_bl: UInt8

    # Size/number registers
    var nusiz0: UInt8  # Number-size player/missile 0
    var nusiz1: UInt8  # Number-size player/missile 1
    var ctrlpf: UInt8  # Playfield control

    # Playfield registers
    var pf0: UInt8
    var pf1: UInt8
    var pf2: UInt8

    # Color registers
    var colup0: UInt8  # Player 0 / missile 0 color
    var colup1: UInt8  # Player 1 / missile 1 color
    var colupf: UInt8  # Playfield / ball color
    var colubk: UInt8  # Background color

    # ========================================================================
    # RIOT State (timer + I/O, ~8 bytes)
    # ========================================================================
    var timer_value: UInt32  # Current timer value (counts down)
    var timer_interval: UInt32  # Timer interval (1, 8, 64, or 1024)
    var timer_clocks: UInt32  # Clocks remaining in current tick

    # ========================================================================
    # Timing / Frame tracking (8 bytes)
    # ========================================================================
    var scanline: UInt16  # Current scanline (0-261)
    var clock: UInt16  # Clock within scanline (0-227)
    var cpu_cycles: UInt32  # Cycles executed this frame
    var wsync: Bool  # WSYNC requested — halt CPU until end of scanline

    # ========================================================================
    # RL State (12 bytes)
    # ========================================================================
    var reward: Int32  # Reward this step (score delta)
    var score: Int32  # Current cumulative score
    var lives: UInt8  # Current lives
    var terminal: Bool  # Episode terminated
    var started: Bool  # Game has started (for lives-based termination)
    var frame_number: UInt32  # Total frames elapsed

    # ========================================================================
    # RAM (128 bytes)
    # ========================================================================
    var ram: InlineArray[UInt8, RAM_SIZE]

    # ========================================================================
    # ROM bank state (for bank-switched cartridges)
    # ========================================================================
    var current_bank: UInt8  # Currently active ROM bank

    # ========================================================================
    # Mid-scanline PF snapshot (captured at cycle ~36 for left/right PF split)
    # ========================================================================
    # The 2600 draws its 20 PF bits twice per line; kernels like Pong's score
    # and Breakout's bricks rewrite PF at the pixel-80 repeat boundary so each
    # half shows a different pattern. pf*_mid holds the left-half PF, while the
    # live pf* registers hold the right-half PF (see tables.playfield_mask).
    var pf0_mid: UInt8  # PF0 at beam midpoint (used for left half)
    var pf1_mid: UInt8  # PF1 at beam midpoint
    var pf2_mid: UInt8  # PF2 at beam midpoint

    # ========================================================================
    # Paddle controller state
    # ========================================================================
    var paddle_pos: UInt8  # Paddle position (0=top, 255=bottom)
    var paddle_charge: UInt8  # Capacitor charge counter (reset by VBLANK bit 7)

    def __init__(out self):
        """Initialize to power-on defaults."""
        # CPU
        self.pc = 0
        self.a = 0
        self.x = 0
        self.y = 0
        self.sp = 0xFD  # Stack starts at 0x01FD
        self.flags = 0x20  # Bit 5 always set

        # System
        self.sys_flags = 0

        # TIA
        self.tia_flags = 0
        self.collision = 0
        self.grp0 = 0
        self.grp1 = 0
        self.grp0_old = 0
        self.grp1_old = 0
        self.enam0 = 0
        self.enam1 = 0
        self.enabl = 0
        self.enabl_old = 0
        self.pos_p0 = 0
        self.pos_p1 = 0
        self.pos_m0 = 0
        self.pos_m1 = 0
        self.pos_bl = 0
        self.hm_p0 = 0
        self.hm_p1 = 0
        self.hm_m0 = 0
        self.hm_m1 = 0
        self.hm_bl = 0
        self.nusiz0 = 0
        self.nusiz1 = 0
        self.ctrlpf = 0
        self.pf0 = 0
        self.pf1 = 0
        self.pf2 = 0
        self.colup0 = 0
        self.colup1 = 0
        self.colupf = 0
        self.colubk = 0

        # RIOT
        self.timer_value = 0
        self.timer_interval = 1024  # Default prescaler
        self.timer_clocks = 0

        # Frame
        self.scanline = 0
        self.clock = 0
        self.cpu_cycles = 0
        self.wsync = False

        # RL
        self.reward = 0
        self.score = 0
        self.lives = 0
        self.terminal = False
        self.started = False
        self.frame_number = 0

        # RAM
        self.ram = InlineArray[UInt8, RAM_SIZE](fill=0)

        # Bank
        self.current_bank = 0

        # PF midpoint snapshot
        self.pf0_mid = 0
        self.pf1_mid = 0
        self.pf2_mid = 0

        # Paddle
        self.paddle_pos = 128  # Center position
        self.paddle_charge = 0

    def reset(mut self):
        """Reset to power-on state (preserves nothing)."""
        self = AtariState()
