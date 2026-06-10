"""Runtime game registry — one env, any game.

`AtariGame` is a DType-style value enum: a trivially-copyable id wrapped in a
struct with comptime constants, so call sites write `AtariGame.PONG` instead of
a magic number. Everything game-specific is dispatched at RUNTIME from it:

  - `game.rom_file()`        → "roms/<name>.bin" (ale-py ROM naming)
  - `game.num_actions()`     → minimal action set size
  - `game.action(idx)`       → idx-th ALE action of the minimal set
  - `game_signals(game,...)` → score / reward / lives / terminal from RAM

Runtime (vs comptime) dispatch is deliberate: this logic runs once per STEP (a
handful of RAM reads) while the emulator runs ~30k instructions per frame, so
the cost is invisible — and one compiled binary can play every game.

Minimal action sets are encoded as an 18-bit mask over the standard ALE action
ids (flags.mojo ACTION_* == ALE enum values) and enumerated in ASCENDING id
order, exactly like ALE's getMinimalActionSet(). NOTE: for Space Invaders this
ordering differs from the legacy SpaceInvadersDef.map_action (which puts LEFT
before FIRE); agents trained on one mapping are not action-compatible with the
other.

Per-game RAM logic is a faithful port of ALE's games/supported/<Game>.cpp
step() bodies (GPL reference, re-implemented). RAM addresses ≥ 0x80 are
mirrors of the 128-byte RIOT RAM — `_rb` masks to 7 bits like ALE's readRam.
"""

from ..flags import (
    RAM_SIZE,
    ACTION_NOOP,
    ACTION_FIRE,
    ACTION_UP,
    ACTION_RIGHT,
    ACTION_LEFT,
    ACTION_DOWN,
    ACTION_UPRIGHT,
    ACTION_UPLEFT,
    ACTION_DOWNRIGHT,
    ACTION_DOWNLEFT,
    ACTION_UPFIRE,
    ACTION_RIGHTFIRE,
    ACTION_LEFTFIRE,
    ACTION_DOWNFIRE,
    ACTION_UPRIGHTFIRE,
    ACTION_UPLEFTFIRE,
    ACTION_DOWNRIGHTFIRE,
    ACTION_DOWNLEFTFIRE,
)
from .helpers import (
    get_decimal_score,
    get_decimal_score_2,
    get_decimal_score_3,
)


@always_inline
def _bit(a: UInt8) -> UInt32:
    return UInt32(1) << UInt32(a)


@always_inline
def _rb(ram: InlineArray[UInt8, RAM_SIZE], addr: Int) -> Int:
    """Read a RAM byte by ALE address (mirrors masked to the 128-byte RAM)."""
    return Int(ram[addr & 0x7F])


# Mask of all 18 standard actions.
comptime ALL_ACTIONS_MASK: UInt32 = 0x3FFFF


struct AtariGame(Copyable, ImplicitlyCopyable, Movable, TrivialRegisterPassable):
    """Value-enum of supported games (DType-style)."""

    var id: UInt8

    comptime PONG = AtariGame(0)
    comptime BREAKOUT = AtariGame(1)
    comptime SPACE_INVADERS = AtariGame(2)
    comptime MS_PACMAN = AtariGame(3)
    comptime SEAQUEST = AtariGame(4)
    comptime QBERT = AtariGame(5)
    comptime ASTEROIDS = AtariGame(6)
    comptime FROSTBITE = AtariGame(7)
    comptime FREEWAY = AtariGame(8)
    comptime BOXING = AtariGame(9)
    comptime ENDURO = AtariGame(10)
    comptime AMIDAR = AtariGame(11)
    comptime ATLANTIS = AtariGame(12)

    comptime NUM_GAMES: Int = 13

    @always_inline
    def __init__(out self, id: UInt8):
        self.id = id

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        return self.id == other.id

    @always_inline
    def __ne__(self, other: Self) -> Bool:
        return self.id != other.id

    @staticmethod
    def from_id(id: Int) -> AtariGame:
        """Construct from a plain index (e.g. CLI arg); must be < NUM_GAMES."""
        return AtariGame(UInt8(id))

    @staticmethod
    def from_name(name: String) raises -> AtariGame:
        """Look up a game by its registry name (e.g. "ms_pacman")."""
        for gid in range(AtariGame.NUM_GAMES):
            var g = AtariGame.from_id(gid)
            if g.name() == name:
                return g
        raise Error("unknown game: " + name)

    def name(self) -> String:
        if self == AtariGame.PONG:
            return "pong"
        elif self == AtariGame.BREAKOUT:
            return "breakout"
        elif self == AtariGame.SPACE_INVADERS:
            return "space_invaders"
        elif self == AtariGame.MS_PACMAN:
            return "ms_pacman"
        elif self == AtariGame.SEAQUEST:
            return "seaquest"
        elif self == AtariGame.QBERT:
            return "qbert"
        elif self == AtariGame.ASTEROIDS:
            return "asteroids"
        elif self == AtariGame.FROSTBITE:
            return "frostbite"
        elif self == AtariGame.FREEWAY:
            return "freeway"
        elif self == AtariGame.BOXING:
            return "boxing"
        elif self == AtariGame.ENDURO:
            return "enduro"
        elif self == AtariGame.AMIDAR:
            return "amidar"
        elif self == AtariGame.ATLANTIS:
            return "atlantis"
        return "unknown"

    def rom_file(self) -> String:
        """ROM path relative to the repo root (ale-py ROM naming)."""
        return "roms/" + self.name() + ".bin"

    def action_mask(self) -> UInt32:
        """18-bit mask of the game's minimal action set (ALE isMinimal)."""
        if self == AtariGame.PONG or self == AtariGame.SPACE_INVADERS:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_RIGHTFIRE)
                | _bit(ACTION_LEFTFIRE)
            )
        elif self == AtariGame.BREAKOUT:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
            )
        elif self == AtariGame.MS_PACMAN:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_UP)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_DOWN)
                | _bit(ACTION_UPRIGHT)
                | _bit(ACTION_UPLEFT)
                | _bit(ACTION_DOWNRIGHT)
                | _bit(ACTION_DOWNLEFT)
            )
        elif self == AtariGame.QBERT:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_UP)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_DOWN)
            )
        elif self == AtariGame.ASTEROIDS:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_UP)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_DOWN)
                | _bit(ACTION_UPRIGHT)
                | _bit(ACTION_UPLEFT)
                | _bit(ACTION_UPFIRE)
                | _bit(ACTION_RIGHTFIRE)
                | _bit(ACTION_LEFTFIRE)
                | _bit(ACTION_DOWNFIRE)
                | _bit(ACTION_UPRIGHTFIRE)
                | _bit(ACTION_UPLEFTFIRE)
            )
        elif self == AtariGame.FREEWAY:
            return _bit(ACTION_NOOP) | _bit(ACTION_UP) | _bit(ACTION_DOWN)
        elif self == AtariGame.ENDURO:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_DOWN)
                | _bit(ACTION_DOWNRIGHT)
                | _bit(ACTION_DOWNLEFT)
                | _bit(ACTION_RIGHTFIRE)
                | _bit(ACTION_LEFTFIRE)
            )
        elif self == AtariGame.AMIDAR:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_UP)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_DOWN)
                | _bit(ACTION_UPFIRE)
                | _bit(ACTION_RIGHTFIRE)
                | _bit(ACTION_LEFTFIRE)
                | _bit(ACTION_DOWNFIRE)
            )
        elif self == AtariGame.ATLANTIS:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_RIGHTFIRE)
                | _bit(ACTION_LEFTFIRE)
            )
        # Seaquest, Frostbite, Boxing: full action set.
        return ALL_ACTIONS_MASK

    def num_actions(self) -> Int:
        """Size of the minimal action set."""
        var mask = self.action_mask()
        var n = 0
        for i in range(18):
            if (mask & (UInt32(1) << UInt32(i))) != 0:
                n += 1
        return n

    def action(self, action_idx: Int) -> UInt8:
        """Map [0, num_actions) to the idx-th ALE action (ascending id order,
        like ALE's getMinimalActionSet)."""
        var mask = self.action_mask()
        var seen = 0
        for i in range(18):
            if (mask & (UInt32(1) << UInt32(i))) != 0:
                if seen == action_idx:
                    return UInt8(i)
                seen += 1
        return ACTION_NOOP


@fieldwise_init
struct GameSignals(Copyable, ImplicitlyCopyable, Movable):
    """Per-step RL signals extracted from RAM."""

    var score: Int
    var reward: Int
    var lives: Int
    var terminal: Bool


def game_signals(
    game: AtariGame,
    ram: InlineArray[UInt8, RAM_SIZE],
    prev_score: Int,
) -> GameSignals:
    """Extract (score, reward, lives, terminal) — port of ALE's per-game step().

    `prev_score` is the score after the previous step (0 at episode start);
    reward = score delta with each game's quirks (wrap correction, terminal
    garbage suppression, clamping) applied exactly as in ALE.
    """
    if game == AtariGame.PONG:
        # ALE Pong.cpp: raw bytes, not BCD.
        var cpu = _rb(ram, 13)
        var player = _rb(ram, 14)
        var score = player - cpu
        return GameSignals(
            score, score - prev_score, 0, cpu == 21 or player == 21
        )

    elif game == AtariGame.BREAKOUT:
        var x = _rb(ram, 77)
        var y = _rb(ram, 76)
        var score = (x & 0x0F) + 10 * ((x & 0xF0) >> 4) + 100 * (y & 0x0F)
        var lives = _rb(ram, 57)
        return GameSignals(score, score - prev_score, lives, lives == 0)

    elif game == AtariGame.SPACE_INVADERS:
        var score = get_decimal_score_2(ram, 0xE8, 0xE6)
        var reward = score - prev_score
        if reward < 0:
            # Score wrapped (10000 is the maximum).
            reward = (10000 - prev_score) + score
        var lives = _rb(ram, 0xC9)
        var terminal = (_rb(ram, 0x98) & 0x80) != 0 or lives == 0
        return GameSignals(score, reward, lives, terminal)

    elif game == AtariGame.MS_PACMAN:
        var score = get_decimal_score_3(ram, 0xF8, 0xF9, 0xFA)
        var lives_byte = _rb(ram, 0xFB) & 0xF
        var terminal = lives_byte == 0 and _rb(ram, 0xA7) == 0x53
        return GameSignals(
            score, score - prev_score, (lives_byte & 0x7) + 1, terminal
        )

    elif game == AtariGame.SEAQUEST:
        var score = get_decimal_score_3(ram, 0xBA, 0xB9, 0xB8)
        return GameSignals(
            score,
            score - prev_score,
            _rb(ram, 0xBB) + 1,
            _rb(ram, 0xA3) != 0,
        )

    elif game == AtariGame.QBERT:
        # Lives byte counts down 2,1,0,0xFF,0xFE (signed); 0xFE = death.
        var b = _rb(ram, 0x88)
        var sb = b - 256 if b >= 128 else b
        var lives = sb + 2
        if lives < 0:
            lives = 0
        var terminal = b == 0xFE
        if terminal:
            # ALE: suppress the garbage score on the reset frame.
            return GameSignals(prev_score, 0, lives, True)
        var score = get_decimal_score_3(ram, 0xDB, 0xDA, 0xD9)
        return GameSignals(score, score - prev_score, lives, False)

    elif game == AtariGame.ASTEROIDS:
        var score = get_decimal_score_2(ram, 0xBE, 0xBD) * 10
        var reward = score - prev_score
        if reward < 0:
            reward += 100000  # score wrap
        var lives = _rb(ram, 0xBC) >> 4
        return GameSignals(score, reward, lives, lives == 0)

    elif game == AtariGame.FROSTBITE:
        var score = get_decimal_score_3(ram, 0xCA, 0xC9, 0xC8)
        var lives_byte = _rb(ram, 0xCC) & 0xF
        var terminal = lives_byte == 0 and (_rb(ram, 0xF1) & 0x80) != 0
        return GameSignals(
            score, score - prev_score, lives_byte + 1, terminal
        )

    elif game == AtariGame.FREEWAY:
        var score = get_decimal_score(ram, 103)
        var reward = score - prev_score
        if reward < 0:
            reward = 0
        if reward > 1:
            reward = 1
        return GameSignals(score, reward, 0, _rb(ram, 22) == 1)

    elif game == AtariGame.BOXING:
        var my_score = get_decimal_score(ram, 0x92)
        var oppt_score = get_decimal_score(ram, 0x93)
        # 0xC0 = KO sentinel.
        if _rb(ram, 0x92) == 0xC0:
            my_score = 100
        if _rb(ram, 0x93) == 0xC0:
            oppt_score = 100
        var score = my_score - oppt_score
        var terminal: Bool
        if my_score == 100 or oppt_score == 100:
            terminal = True
        else:
            var minutes = _rb(ram, 0x90) >> 4
            var seconds = (_rb(ram, 0x91) & 0xF) + (_rb(ram, 0x91) >> 4) * 10
            terminal = minutes == 0 and seconds == 0
        return GameSignals(score, score - prev_score, 0, terminal)

    elif game == AtariGame.ENDURO:
        var score = 0
        var level = _rb(ram, 0xAD)
        if level != 0:
            var cars_passed = get_decimal_score_2(ram, 0xAB, 0xAC)
            if level == 1:
                cars_passed = 200 - cars_passed
            else:
                cars_passed = 300 - cars_passed
            if level >= 2:
                # First level has 200 cars; 300 for every level after.
                score = 200 + (level - 2) * 300
            score += cars_passed
        return GameSignals(
            score, score - prev_score, 0, _rb(ram, 0xAF) == 0xFF
        )

    elif game == AtariGame.AMIDAR:
        var score = get_decimal_score_3(ram, 0xD9, 0xDA, 0xDB)
        var lives_byte = _rb(ram, 0xD6)
        return GameSignals(
            score, score - prev_score, lives_byte & 0xF, lives_byte == 0x80
        )

    elif game == AtariGame.ATLANTIS:
        var score = get_decimal_score_3(ram, 0xA2, 0xA3, 0xA1) * 100
        var lives = _rb(ram, 0xF1)
        if lives == 0xFF:
            # ALE: garbage gets written to 0xA1 on the terminal frame —
            # freeze the score and zero the reward.
            return GameSignals(prev_score, 0, lives, True)
        return GameSignals(score, score - prev_score, lives, False)

    return GameSignals(0, 0, 0, False)
