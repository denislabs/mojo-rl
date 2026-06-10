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
    ROM_AUTO,
    ROM_E0,
    ROM_FE,
    ROM_F8SC,
    ROM_F6SC,
)
from ..atari_state import AtariState
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
    # Wave 1 (alphabetical from here on)
    comptime ALIEN = AtariGame(13)
    comptime ASSAULT = AtariGame(14)
    comptime ASTERIX = AtariGame(15)
    comptime BANK_HEIST = AtariGame(16)
    comptime BATTLE_ZONE = AtariGame(17)
    comptime BEAM_RIDER = AtariGame(18)
    comptime BERZERK = AtariGame(19)
    comptime BOWLING = AtariGame(20)
    comptime CENTIPEDE = AtariGame(21)
    comptime CHOPPER_COMMAND = AtariGame(22)
    comptime CRAZY_CLIMBER = AtariGame(23)
    comptime DARK_CHAMBERS = AtariGame(24)
    comptime DEMON_ATTACK = AtariGame(25)
    comptime ELEVATOR_ACTION = AtariGame(26)
    comptime FISHING_DERBY = AtariGame(27)
    comptime JAMESBOND = AtariGame(28)
    comptime KLAX = AtariGame(29)
    comptime MONTEZUMA_REVENGE = AtariGame(30)
    comptime ROBOTANK = AtariGame(31)
    comptime TUTANKHAM = AtariGame(32)
    # Wave 2: completes the Atari-57 benchmark set (+ pooyan)
    comptime DEFENDER = AtariGame(33)
    comptime DOUBLE_DUNK = AtariGame(34)
    comptime GOPHER = AtariGame(35)
    comptime GRAVITAR = AtariGame(36)
    comptime HERO = AtariGame(37)
    comptime ICE_HOCKEY = AtariGame(38)
    comptime KANGAROO = AtariGame(39)
    comptime KRULL = AtariGame(40)
    comptime KUNG_FU_MASTER = AtariGame(41)
    comptime NAME_THIS_GAME = AtariGame(42)
    comptime PHOENIX = AtariGame(43)
    comptime PITFALL = AtariGame(44)
    comptime POOYAN = AtariGame(45)
    comptime PRIVATE_EYE = AtariGame(46)
    comptime RIVERRAID = AtariGame(47)
    comptime ROAD_RUNNER = AtariGame(48)
    comptime SKIING = AtariGame(49)
    comptime SOLARIS = AtariGame(50)
    comptime STAR_GUNNER = AtariGame(51)
    comptime SURROUND = AtariGame(52)
    comptime TENNIS = AtariGame(53)
    comptime TIME_PILOT = AtariGame(54)
    comptime UP_N_DOWN = AtariGame(55)
    comptime VENTURE = AtariGame(56)
    comptime VIDEO_PINBALL = AtariGame(57)
    comptime WIZARD_OF_WOR = AtariGame(58)
    comptime YARS_REVENGE = AtariGame(59)
    comptime ZAXXON = AtariGame(60)

    comptime NUM_GAMES: Int = 61

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
        elif self == AtariGame.ALIEN:
            return "alien"
        elif self == AtariGame.ASSAULT:
            return "assault"
        elif self == AtariGame.ASTERIX:
            return "asterix"
        elif self == AtariGame.BANK_HEIST:
            return "bank_heist"
        elif self == AtariGame.BATTLE_ZONE:
            return "battle_zone"
        elif self == AtariGame.BEAM_RIDER:
            return "beam_rider"
        elif self == AtariGame.BERZERK:
            return "berzerk"
        elif self == AtariGame.BOWLING:
            return "bowling"
        elif self == AtariGame.CENTIPEDE:
            return "centipede"
        elif self == AtariGame.CHOPPER_COMMAND:
            return "chopper_command"
        elif self == AtariGame.CRAZY_CLIMBER:
            return "crazy_climber"
        elif self == AtariGame.DARK_CHAMBERS:
            return "darkchambers"  # ale-py ROM name has no underscore
        elif self == AtariGame.DEMON_ATTACK:
            return "demon_attack"
        elif self == AtariGame.ELEVATOR_ACTION:
            return "elevator_action"
        elif self == AtariGame.FISHING_DERBY:
            return "fishing_derby"
        elif self == AtariGame.JAMESBOND:
            return "jamesbond"
        elif self == AtariGame.KLAX:
            return "klax"
        elif self == AtariGame.MONTEZUMA_REVENGE:
            return "montezuma_revenge"
        elif self == AtariGame.ROBOTANK:
            return "robotank"
        elif self == AtariGame.TUTANKHAM:
            return "tutankham"
        elif self == AtariGame.DEFENDER:
            return "defender"
        elif self == AtariGame.DOUBLE_DUNK:
            return "double_dunk"
        elif self == AtariGame.GOPHER:
            return "gopher"
        elif self == AtariGame.GRAVITAR:
            return "gravitar"
        elif self == AtariGame.HERO:
            return "hero"
        elif self == AtariGame.ICE_HOCKEY:
            return "ice_hockey"
        elif self == AtariGame.KANGAROO:
            return "kangaroo"
        elif self == AtariGame.KRULL:
            return "krull"
        elif self == AtariGame.KUNG_FU_MASTER:
            return "kung_fu_master"
        elif self == AtariGame.NAME_THIS_GAME:
            return "name_this_game"
        elif self == AtariGame.PHOENIX:
            return "phoenix"
        elif self == AtariGame.PITFALL:
            return "pitfall"
        elif self == AtariGame.POOYAN:
            return "pooyan"
        elif self == AtariGame.PRIVATE_EYE:
            return "private_eye"
        elif self == AtariGame.RIVERRAID:
            return "riverraid"
        elif self == AtariGame.ROAD_RUNNER:
            return "road_runner"
        elif self == AtariGame.SKIING:
            return "skiing"
        elif self == AtariGame.SOLARIS:
            return "solaris"
        elif self == AtariGame.STAR_GUNNER:
            return "star_gunner"
        elif self == AtariGame.SURROUND:
            return "surround"
        elif self == AtariGame.TENNIS:
            return "tennis"
        elif self == AtariGame.TIME_PILOT:
            return "time_pilot"
        elif self == AtariGame.UP_N_DOWN:
            return "up_n_down"
        elif self == AtariGame.VENTURE:
            return "venture"
        elif self == AtariGame.VIDEO_PINBALL:
            return "video_pinball"
        elif self == AtariGame.WIZARD_OF_WOR:
            return "wizard_of_wor"
        elif self == AtariGame.YARS_REVENGE:
            return "yars_revenge"
        elif self == AtariGame.ZAXXON:
            return "zaxxon"
        return "unknown"

    def rom_file(self) -> String:
        """ROM path relative to the repo root (ale-py ROM naming)."""
        return "roms/" + self.name() + ".bin"

    def mapper(self) -> UInt8:
        """Cartridge mapper override (ROM_* id), or ROM_AUTO for size-based.

        Size detection cannot distinguish F8 / E0 / FE / F8SC at 8K (or
        F6 / F6SC at 16K). Values baked from running ALE's
        Cartridge::autodetectType content signatures over the ROM set; only
        games whose mapper differs from the size default appear here.
        """
        if (
            self == AtariGame.JAMESBOND
            or self == AtariGame.MONTEZUMA_REVENGE
            or self == AtariGame.TUTANKHAM
        ):
            return ROM_E0
        elif self == AtariGame.ROBOTANK:
            return ROM_FE
        elif self == AtariGame.ELEVATOR_ACTION:
            return ROM_F8SC
        elif self == AtariGame.DARK_CHAMBERS or self == AtariGame.KLAX:
            return ROM_F6SC
        return ROM_AUTO

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
        elif self == AtariGame.ASSAULT:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_UP)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_RIGHTFIRE)
                | _bit(ACTION_LEFTFIRE)
            )
        elif (
            self == AtariGame.ASTERIX or self == AtariGame.CRAZY_CLIMBER
        ):
            # 4 directions + 4 diagonals, no fire.
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
        elif self == AtariGame.BEAM_RIDER:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_UP)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_UPRIGHT)
                | _bit(ACTION_UPLEFT)
                | _bit(ACTION_RIGHTFIRE)
                | _bit(ACTION_LEFTFIRE)
            )
        elif (
            self == AtariGame.BOWLING
            or self == AtariGame.POOYAN
            or self == AtariGame.UP_N_DOWN
        ):
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_UP)
                | _bit(ACTION_DOWN)
                | _bit(ACTION_UPFIRE)
                | _bit(ACTION_DOWNFIRE)
            )
        elif (
            self == AtariGame.DEMON_ATTACK
            or self == AtariGame.NAME_THIS_GAME
        ):
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_RIGHTFIRE)
                | _bit(ACTION_LEFTFIRE)
            )
        elif self == AtariGame.GOPHER:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_UP)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_UPFIRE)
                | _bit(ACTION_RIGHTFIRE)
                | _bit(ACTION_LEFTFIRE)
            )
        elif self == AtariGame.KUNG_FU_MASTER:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_UP)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_DOWN)
                | _bit(ACTION_DOWNRIGHT)
                | _bit(ACTION_DOWNLEFT)
                | _bit(ACTION_RIGHTFIRE)
                | _bit(ACTION_LEFTFIRE)
                | _bit(ACTION_DOWNFIRE)
                | _bit(ACTION_UPRIGHTFIRE)
                | _bit(ACTION_UPLEFTFIRE)
                | _bit(ACTION_DOWNRIGHTFIRE)
                | _bit(ACTION_DOWNLEFTFIRE)
            )
        elif self == AtariGame.PHOENIX:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_FIRE)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_DOWN)
                | _bit(ACTION_RIGHTFIRE)
                | _bit(ACTION_LEFTFIRE)
                | _bit(ACTION_DOWNFIRE)
            )
        elif self == AtariGame.SKIING:
            return _bit(ACTION_NOOP) | _bit(ACTION_RIGHT) | _bit(ACTION_LEFT)
        elif self == AtariGame.SURROUND:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_UP)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_DOWN)
            )
        elif (
            self == AtariGame.TIME_PILOT or self == AtariGame.WIZARD_OF_WOR
        ):
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
        elif self == AtariGame.VIDEO_PINBALL:
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
            )
        elif self == AtariGame.TUTANKHAM:
            return (
                _bit(ACTION_NOOP)
                | _bit(ACTION_UP)
                | _bit(ACTION_RIGHT)
                | _bit(ACTION_LEFT)
                | _bit(ACTION_DOWN)
                | _bit(ACTION_UPFIRE)
                | _bit(ACTION_RIGHTFIRE)
                | _bit(ACTION_LEFTFIRE)
            )
        # Full action set: Seaquest, Frostbite, Boxing, Alien, BankHeist,
        # BattleZone, Berzerk, Centipede, ChopperCommand, DarkChambers,
        # ElevatorAction, FishingDerby, Jamesbond, Klax, MontezumaRevenge,
        # Robotank.
        return ALL_ACTIONS_MASK

    def starting_actions(self) -> Tuple[UInt8, Int]:
        """(action, frames) to inject right after reset, before the agent
        acts — port of ALE's per-game getStartingActions(). Some games need
        an input to leave the title screen (FIRE), and DarkChambers ignores
        all input during its ~8 s boot animation."""
        if (
            self == AtariGame.ASTERIX
            or self == AtariGame.ENDURO
            or self == AtariGame.GOPHER
            or self == AtariGame.UP_N_DOWN
            or self == AtariGame.YARS_REVENGE
        ):
            return (ACTION_FIRE, 1)
        elif self == AtariGame.BEAM_RIDER:
            return (ACTION_RIGHT, 1)
        elif self == AtariGame.DARK_CHAMBERS:
            return (ACTION_NOOP, 486)
        elif (
            self == AtariGame.ELEVATOR_ACTION or self == AtariGame.GRAVITAR
        ):
            return (ACTION_FIRE, 16)
        elif self == AtariGame.DOUBLE_DUNK:
            return (ACTION_UPFIRE, 1)
        elif self == AtariGame.PITFALL or self == AtariGame.PRIVATE_EYE:
            return (ACTION_UP, 1)
        elif self == AtariGame.SKIING:
            return (ACTION_DOWN, 16)
        return (ACTION_NOOP, 0)

    def swap_ports(self) -> Bool:
        """Player 1 uses the RIGHT joystick port (Stella Console.SwapPorts
        property — Wizard of Wor is the only such game in the ALE set)."""
        return self == AtariGame.WIZARD_OF_WOR

    def select_until(self) -> Tuple[Int, Int]:
        """(ALE RAM address, desired value) for console-SELECT game-mode
        selection at reset, or (-1, 0) for none. Port of the DEFAULT-mode
        path of ALE setMode: Surround boots in 2-player mode (RAM $F9 == 0);
        ALE presses SELECT until $F9 == 1 (single player vs computer), then
        soft-resets."""
        if self == AtariGame.SURROUND:
            return (0xF9, 1)
        return (-1, 0)

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
    mut state: AtariState,
    prev_score: Int,
) -> GameSignals:
    """Extract (score, reward, lives, terminal) — port of ALE's per-game step().

    `prev_score` is the score after the previous step (0 at episode start);
    reward = score delta with each game's quirks (wrap correction, terminal
    garbage suppression, clamping) applied exactly as in ALE.

    Takes the full AtariState because some games need more than the 128-byte
    RIOT RAM: Klax reads its score from Superchip RAM (state.sc_ram), and
    games with cross-step latches in ALE (ChopperCommand's started flag,
    DarkChambers' last level) persist them in state.game_aux (mut). BeamRider
    reads state.lives (the previous step's lives, set by the caller after
    each call) to filter the blinking lives counter.
    """
    if game == AtariGame.PONG:
        # ALE Pong.cpp: raw bytes, not BCD.
        var cpu = _rb(state.ram, 13)
        var player = _rb(state.ram, 14)
        var score = player - cpu
        return GameSignals(
            score, score - prev_score, 0, cpu == 21 or player == 21
        )

    elif game == AtariGame.BREAKOUT:
        var x = _rb(state.ram, 77)
        var y = _rb(state.ram, 76)
        var score = (x & 0x0F) + 10 * ((x & 0xF0) >> 4) + 100 * (y & 0x0F)
        var lives = _rb(state.ram, 57)
        return GameSignals(score, score - prev_score, lives, lives == 0)

    elif game == AtariGame.SPACE_INVADERS:
        var score = get_decimal_score_2(state.ram, 0xE8, 0xE6)
        var reward = score - prev_score
        if reward < 0:
            # Score wrapped (10000 is the maximum).
            reward = (10000 - prev_score) + score
        var lives = _rb(state.ram, 0xC9)
        var terminal = (_rb(state.ram, 0x98) & 0x80) != 0 or lives == 0
        return GameSignals(score, reward, lives, terminal)

    elif game == AtariGame.MS_PACMAN:
        var score = get_decimal_score_3(state.ram, 0xF8, 0xF9, 0xFA)
        var lives_byte = _rb(state.ram, 0xFB) & 0xF
        var terminal = lives_byte == 0 and _rb(state.ram, 0xA7) == 0x53
        return GameSignals(
            score, score - prev_score, (lives_byte & 0x7) + 1, terminal
        )

    elif game == AtariGame.SEAQUEST:
        var score = get_decimal_score_3(state.ram, 0xBA, 0xB9, 0xB8)
        return GameSignals(
            score,
            score - prev_score,
            _rb(state.ram, 0xBB) + 1,
            _rb(state.ram, 0xA3) != 0,
        )

    elif game == AtariGame.QBERT:
        # Lives byte counts down 2,1,0,0xFF,0xFE (signed); 0xFE = death.
        var b = _rb(state.ram, 0x88)
        var sb = b - 256 if b >= 128 else b
        var lives = sb + 2
        if lives < 0:
            lives = 0
        var terminal = b == 0xFE
        if terminal:
            # ALE: suppress the garbage score on the reset frame.
            return GameSignals(prev_score, 0, lives, True)
        var score = get_decimal_score_3(state.ram, 0xDB, 0xDA, 0xD9)
        return GameSignals(score, score - prev_score, lives, False)

    elif game == AtariGame.ASTEROIDS:
        var score = get_decimal_score_2(state.ram, 0xBE, 0xBD) * 10
        var reward = score - prev_score
        if reward < 0:
            reward += 100000  # score wrap
        var lives = _rb(state.ram, 0xBC) >> 4
        return GameSignals(score, reward, lives, lives == 0)

    elif game == AtariGame.FROSTBITE:
        var score = get_decimal_score_3(state.ram, 0xCA, 0xC9, 0xC8)
        var lives_byte = _rb(state.ram, 0xCC) & 0xF
        var terminal = lives_byte == 0 and (_rb(state.ram, 0xF1) & 0x80) != 0
        return GameSignals(
            score, score - prev_score, lives_byte + 1, terminal
        )

    elif game == AtariGame.FREEWAY:
        var score = get_decimal_score(state.ram, 103)
        var reward = score - prev_score
        if reward < 0:
            reward = 0
        if reward > 1:
            reward = 1
        return GameSignals(score, reward, 0, _rb(state.ram, 22) == 1)

    elif game == AtariGame.BOXING:
        var my_score = get_decimal_score(state.ram, 0x92)
        var oppt_score = get_decimal_score(state.ram, 0x93)
        # 0xC0 = KO sentinel.
        if _rb(state.ram, 0x92) == 0xC0:
            my_score = 100
        if _rb(state.ram, 0x93) == 0xC0:
            oppt_score = 100
        var score = my_score - oppt_score
        var terminal: Bool
        if my_score == 100 or oppt_score == 100:
            terminal = True
        else:
            var minutes = _rb(state.ram, 0x90) >> 4
            var seconds = (_rb(state.ram, 0x91) & 0xF) + (_rb(state.ram, 0x91) >> 4) * 10
            terminal = minutes == 0 and seconds == 0
        return GameSignals(score, score - prev_score, 0, terminal)

    elif game == AtariGame.ENDURO:
        var score = 0
        var level = _rb(state.ram, 0xAD)
        if level != 0:
            var cars_passed = get_decimal_score_2(state.ram, 0xAB, 0xAC)
            if level == 1:
                cars_passed = 200 - cars_passed
            else:
                cars_passed = 300 - cars_passed
            if level >= 2:
                # First level has 200 cars; 300 for every level after.
                score = 200 + (level - 2) * 300
            score += cars_passed
        return GameSignals(
            score, score - prev_score, 0, _rb(state.ram, 0xAF) == 0xFF
        )

    elif game == AtariGame.AMIDAR:
        var score = get_decimal_score_3(state.ram, 0xD9, 0xDA, 0xDB)
        var lives_byte = _rb(state.ram, 0xD6)
        return GameSignals(
            score, score - prev_score, lives_byte & 0xF, lives_byte == 0x80
        )

    elif game == AtariGame.ATLANTIS:
        var score = get_decimal_score_3(state.ram, 0xA2, 0xA3, 0xA1) * 100
        var lives = _rb(state.ram, 0xF1)
        if lives == 0xFF:
            # ALE: garbage gets written to 0xA1 on the terminal frame —
            # freeze the score and zero the reward.
            return GameSignals(prev_score, 0, lives, True)
        return GameSignals(score, score - prev_score, lives, False)

    elif game == AtariGame.ALIEN:
        # Digits stored one per byte: 0x80 means blank (0), else byte >> 3.
        var score = 0
        var mult = 1
        for addr in [0x8B, 0x89, 0x87, 0x85, 0x83]:
            var b = _rb(state.ram, addr)
            score += (0 if b == 0x80 else b >> 3) * mult
            mult *= 10
        score *= 10
        var lives = _rb(state.ram, 0xC0) & 0xF
        return GameSignals(score, score - prev_score, lives, lives == 0)

    elif game == AtariGame.ASSAULT:
        var score = get_decimal_score_3(state.ram, 0x82, 0x81, 0x80)
        var lives = _rb(state.ram, 0xE5)
        return GameSignals(score, score - prev_score, lives, lives == 0)

    elif game == AtariGame.ASTERIX:
        var score = get_decimal_score_3(state.ram, 0xE0, 0xDF, 0xDE)
        var lives = _rb(state.ram, 0xD3) & 0xF
        # Cannot wait for lives==0: the player can restart on the very last
        # frame (lives==1, death_counter==1) by holding fire.
        var death_counter = _rb(state.ram, 0xC7)
        var terminal = death_counter == 0x01 and lives == 1
        return GameSignals(score, score - prev_score, lives, terminal)

    elif game == AtariGame.BANK_HEIST:
        var score = get_decimal_score_3(state.ram, 0xDA, 0xD9, 0xD8)
        var death_timer = _rb(state.ram, 0xCE)
        var lives = _rb(state.ram, 0xD5)
        var terminal = death_timer == 0x01 and lives == 0
        return GameSignals(score, score - prev_score, lives, terminal)

    elif game == AtariGame.BATTLE_ZONE:
        # Score digits use 10 as a blank sentinel.
        var first_val = _rb(state.ram, 0x9D)
        var first_right = first_val & 15
        var first_left = (first_val - first_right) >> 4
        if first_left == 10:
            first_left = 0
        var second_val = _rb(state.ram, 0x9E)
        var second_right = second_val & 15
        var second_left = (second_val - second_right) >> 4
        if second_right == 10:
            second_right = 0
        if second_left == 10:
            second_left = 0
        var score = (
            first_left + 10 * second_right + 100 * second_left
        ) * 1000
        var lives = _rb(state.ram, 0xBA) & 0xF
        return GameSignals(score, score - prev_score, lives, lives == 0)

    elif game == AtariGame.BEAM_RIDER:
        var score = get_decimal_score_3(state.ram, 9, 10, 11)
        # The lives counter blinks during the death animation; only commit
        # a one-life decrease once the animation flag (0x8C) is set.
        var prev_lives = Int(state.lives)
        var new_lives = _rb(state.ram, 0x85) + 1
        var lives = new_lives
        if new_lives == prev_lives - 1 and _rb(state.ram, 0x8C) != 0x01:
            lives = prev_lives
        var terminal = _rb(state.ram, 5) == 255
        return GameSignals(score, score - prev_score, lives, terminal)

    elif game == AtariGame.BERZERK:
        var score = get_decimal_score_3(state.ram, 95, 94, 93)
        var lives_byte = _rb(state.ram, 0xDA)
        if lives_byte == 0xFF:
            return GameSignals(score, score - prev_score, 0, True)
        return GameSignals(score, score - prev_score, lives_byte + 1, False)

    elif game == AtariGame.BOWLING:
        var score = get_decimal_score_2(state.ram, 0xA1, 0xA6)
        return GameSignals(
            score, score - prev_score, 0, _rb(state.ram, 0xA4) > 0x10
        )

    elif game == AtariGame.CENTIPEDE:
        var score = get_decimal_score_3(state.ram, 118, 117, 116)
        var reward = score - prev_score
        # ALE HACK: the score sometimes resets before termination.
        if reward < 0:
            reward = 0
        var lives = ((_rb(state.ram, 0xED) >> 4) & 0x7) + 1
        var terminal = (_rb(state.ram, 0xA6) & 0x40) != 0
        return GameSignals(score, reward, lives, terminal)

    elif game == AtariGame.CHOPPER_COMMAND:
        var score = get_decimal_score_2(state.ram, 0xEE, 0xEC) * 100
        var lives = _rb(state.ram, 0xE4) & 0xF
        # 0xC2 bit 0 is 1 once gameplay has started (chopper faces right);
        # latch it so mode-select screens (always facing left) don't read
        # as terminal. ALE keeps m_is_started for the same reason.
        state.game_aux |= Int32(_rb(state.ram, 0xC2) & 0x1)
        var terminal = state.game_aux != 0 and lives == 0
        return GameSignals(score, score - prev_score, lives, terminal)

    elif game == AtariGame.CRAZY_CLIMBER:
        # Digits stored one per byte (not BCD).
        var score = (
            _rb(state.ram, 0x82)
            + 10 * _rb(state.ram, 0x83)
            + 100 * _rb(state.ram, 0x84)
            + 1000 * _rb(state.ram, 0x85)
        ) * 100
        var reward = score - prev_score
        if reward < 0:
            reward = 0
        var lives = _rb(state.ram, 0xAA)
        return GameSignals(score, reward, lives, lives == 0)

    elif game == AtariGame.DARK_CHAMBERS:
        # game_aux holds the last seen level: levels only go up; a drop
        # means the game restarted (terminal). Score wrap is also terminal.
        var new_level = _rb(state.ram, 0xD5)
        if new_level < Int(state.game_aux):
            return GameSignals(prev_score, 0, 0, True)
        state.game_aux = Int32(new_level)
        var score = get_decimal_score_2(state.ram, 0xCC, 0xCF) * 10
        if score < prev_score:
            # Exceeded the maximum score.
            return GameSignals(prev_score, 0, 0, True)
        # Low 5 bits are health; the top 3 are item flags.
        var health = _rb(state.ram, 0xCA) & 0x1F
        return GameSignals(score, score - prev_score, health, health == 0)

    elif game == AtariGame.DEMON_ATTACK:
        var score = get_decimal_score_3(state.ram, 0x85, 0x83, 0x81)
        # ALE MGB: the score RAM is not initialized to 0 on boot.
        if (
            _rb(state.ram, 0x81) == 0xAB
            and _rb(state.ram, 0x83) == 0xCD
            and _rb(state.ram, 0x85) == 0xEA
        ):
            score = 0
        var lives_displayed = _rb(state.ram, 0xF2)
        var display_flag = _rb(state.ram, 0xF1)
        var terminal = lives_displayed == 0 and display_flag == 0xBD
        return GameSignals(
            score, score - prev_score, lives_displayed + 1, terminal
        )

    elif game == AtariGame.ELEVATOR_ACTION:
        var score = get_decimal_score_3(state.ram, 0x89, 0x88, 0x87)
        var lives = _rb(state.ram, 0x83)
        # 0x81 == 0 only on the start screen, where lives reads 0 too.
        var terminal = lives == 0 and _rb(state.ram, 0x81) != 0x00
        return GameSignals(score, score - prev_score, lives, terminal)

    elif game == AtariGame.FISHING_DERBY:
        var my_score = get_decimal_score(state.ram, 0xBD)
        var oppt_score = get_decimal_score(state.ram, 0xBE)
        var score = my_score - oppt_score
        # Either side reaching 99 (0x99 BCD) ends the game.
        var terminal = (
            _rb(state.ram, 0xBD) == 0x99 or _rb(state.ram, 0xBE) == 0x99
        )
        return GameSignals(score, score - prev_score, 0, terminal)

    elif game == AtariGame.JAMESBOND:
        var score = get_decimal_score_3(state.ram, 0xDC, 0xDD, 0xDE)
        var lives_byte = _rb(state.ram, 0x86) & 0xF
        # 0x8C reads 0x68 on death; the system loops back to the start
        # state after a while (where fire starts a new game).
        var terminal = lives_byte == 0 and _rb(state.ram, 0x8C) == 0x68
        return GameSignals(
            score, score - prev_score, lives_byte + 1, terminal
        )

    elif game == AtariGame.KLAX:
        # Score and level live in Superchip RAM (read port $F080-$F0FF →
        # sc_ram[addr - 0x80]); ALE reads them via the full memory map.
        var b_lo = Int(state.sc_ram[0x34])  # $F0B4
        var b_mid = Int(state.sc_ram[0x35])  # $F0B5
        var b_hi = Int(state.sc_ram[0x36])  # $F0B6
        var score = (
            10 * ((b_lo >> 4) & 0xF)
            + (b_lo & 0xF)
            + 1000 * ((b_mid >> 4) & 0xF)
            + 100 * (b_mid & 0xF)
            + 100000 * ((b_hi >> 4) & 0xF)
            + 10000 * (b_hi & 0xF)
        )
        var misses = Int(state.sc_ram[0x6E])  # $F0EE
        var max_misses = Int(state.sc_ram[0x69])  # $F0E9
        var level = Int(state.sc_ram[0x1D])  # $F09D
        var game_active = _rb(state.ram, 0xA8) == 4
        # The 25 bottom blocks live at 0xB3..0xCB; types 0/2/6/10/14 are
        # empty or level-end bonus fillers, not real blocks.
        var num_blocks = 0
        for i in range(25):
            var t = _rb(state.ram, 0xB3 + i)
            if t != 0 and t != 2 and t != 6 and t != 10 and t != 14:
                num_blocks += 1
        var terminal = (
            (max_misses > 0 and misses == max_misses)
            or (game_active and num_blocks == 25)
            or level == 0x99
        )
        return GameSignals(score, score - prev_score, 0, terminal)

    elif game == AtariGame.MONTEZUMA_REVENGE:
        var score = get_decimal_score_3(state.ram, 0x95, 0x94, 0x93)
        var new_lives = _rb(state.ram, 0xBA)
        var terminal = new_lives == 0 and _rb(state.ram, 0xFE) == 0x60
        return GameSignals(
            score, score - prev_score, (new_lives & 0x7) + 1, terminal
        )

    elif game == AtariGame.ROBOTANK:
        # Raw counters, not BCD: a squadron is 12 tanks.
        var score = _rb(state.ram, 0xB6) * 12 + _rb(state.ram, 0xB5)
        var lives = _rb(state.ram, 0xA8)
        var terminal = lives == 0 and _rb(state.ram, 0xB4) == 0xFF
        return GameSignals(
            score, score - prev_score, (lives & 0xF) + 1, terminal
        )

    elif game == AtariGame.TUTANKHAM:
        var score = get_decimal_score_2(state.ram, 0x9C, 0x9A)
        var lives_byte = _rb(state.ram, 0x9E)
        # 0x81 is 0x84 when the game is freshly loaded but not yet reset.
        var terminal = lives_byte == 0 and _rb(state.ram, 0x81) != 0x84
        return GameSignals(
            score, score - prev_score, lives_byte & 0x3, terminal
        )

    elif game == AtariGame.DEFENDER:
        # Six digits one per byte at 0x9C-0xA1; 0xA = blank zero.
        var score = 0
        var mult = 1
        for digit in range(6):
            var v = _rb(state.ram, 0x9C + digit) & 0xF
            if v == 0xA:
                v = 0
            score += v * mult
            mult *= 10
        var lives = _rb(state.ram, 0xC2)
        return GameSignals(score, score - prev_score, lives, lives == 0)

    elif game == AtariGame.DOUBLE_DUNK:
        var my_score = get_decimal_score(state.ram, 0xF6)
        var oppt_score = get_decimal_score(state.ram, 0xF7)
        var score = my_score - oppt_score
        var terminal = (my_score >= 24 or oppt_score >= 24) and _rb(
            state.ram, 0xFE
        ) == 0xE7
        return GameSignals(score, score - prev_score, 0, terminal)

    elif game == AtariGame.GOPHER:
        var score = get_decimal_score_3(state.ram, 0xB2, 0xB1, 0xB0)
        # Lives = popcount of the 3 carrot bits.
        var carrots = _rb(state.ram, 0xB4) & 0x7
        var lives = (carrots & 1) + ((carrots >> 1) & 1) + (
            (carrots >> 2) & 1
        )
        return GameSignals(score, score - prev_score, lives, carrots == 0)

    elif game == AtariGame.GRAVITAR:
        var score = get_decimal_score_3(state.ram, 9, 8, 7)
        var screen_byte = _rb(state.ram, 0x81)
        # 6 lives on the starting screen, else read from RAM.
        var lives = 6 if screen_byte == 0x0 else _rb(state.ram, 0x84) + 1
        return GameSignals(
            score, score - prev_score, lives, screen_byte == 0x01
        )

    elif game == AtariGame.HERO:
        var score = get_decimal_score_3(state.ram, 0xB9, 0xB8, 0xB7)
        var lives = _rb(state.ram, 0xB3)
        return GameSignals(score, score - prev_score, lives, lives == 0)

    elif game == AtariGame.ICE_HOCKEY:
        var my_score = get_decimal_score(state.ram, 0x8A)
        var oppt_score = get_decimal_score(state.ram, 0x8B)
        var score = my_score - oppt_score
        var reward = score - prev_score
        if reward > 1:
            reward = 1  # ALE clamps to +1
        # Game ends when the clock runs out.
        var terminal = (
            _rb(state.ram, 0x87) == 0 and _rb(state.ram, 0x86) == 0
        )
        return GameSignals(score, reward, 0, terminal)

    elif game == AtariGame.KANGAROO:
        var score = get_decimal_score_2(state.ram, 0xA8, 0xA7) * 100
        var lives_byte = _rb(state.ram, 0xAD)
        return GameSignals(
            score,
            score - prev_score,
            (lives_byte & 0x7) + 1,
            lives_byte == 0xFF,
        )

    elif game == AtariGame.KRULL:
        var score = get_decimal_score_3(state.ram, 0x9E, 0x9D, 0x9C)
        var lives = _rb(state.ram, 0x9F)
        var terminal = (
            lives == 0
            and _rb(state.ram, 0xA2) == 0x03
            and _rb(state.ram, 0x80) == 0x80
        )
        return GameSignals(
            score, score - prev_score, (lives & 0x7) + 1, terminal
        )

    elif game == AtariGame.KUNG_FU_MASTER:
        var score = get_decimal_score_3(state.ram, 0x9A, 0x99, 0x98)
        var lives_byte = _rb(state.ram, 0x9D)
        return GameSignals(
            score,
            score - prev_score,
            (lives_byte & 0x7) + 1,
            lives_byte == 0xFF,
        )

    elif game == AtariGame.NAME_THIS_GAME:
        var score = get_decimal_score_3(state.ram, 0xC6, 0xC5, 0xC4)
        var lives = _rb(state.ram, 0xC7) & 0x7
        return GameSignals(score, score - prev_score, lives, lives == 0)

    elif game == AtariGame.PHOENIX:
        var score = get_decimal_score_2(state.ram, 0xC8, 0xC9) * 10
        score += _rb(state.ram, 0xC7) >> 4
        score *= 10
        var terminal = _rb(state.ram, 0xCC) == 0x80
        var lives = _rb(state.ram, 0xCB) & 0x7
        return GameSignals(score, score - prev_score, lives, terminal)

    elif game == AtariGame.PITFALL:
        # Score starts at 2000 and can DECREASE (falls/time penalties);
        # reset_game syncs state.score so the first step doesn't see +2000.
        var score = get_decimal_score_3(state.ram, 0xD7, 0xD6, 0xD5)
        var lives_byte = _rb(state.ram, 0x80) >> 4
        # 0x9E nonzero while the player is uncontrollable (logo screen).
        var terminal = lives_byte == 0 and _rb(state.ram, 0x9E) != 0
        var lives: Int
        if lives_byte == 0xA:
            lives = 3
        elif lives_byte == 0x8:
            lives = 2
        else:
            lives = 1
        return GameSignals(score, score - prev_score, lives, terminal)

    elif game == AtariGame.POOYAN:
        var score = get_decimal_score_3(state.ram, 0x8A, 0x89, 0x88)
        var lives_byte = _rb(state.ram, 0x96)
        var terminal = lives_byte == 0x0 and _rb(state.ram, 0x98) == 0x05
        return GameSignals(
            score, score - prev_score, (lives_byte & 0x7) + 1, terminal
        )

    elif game == AtariGame.PRIVATE_EYE:
        var score = get_decimal_score_3(state.ram, 0xCA, 0xC9, 0xC8)
        # Copyright timer: 0x00 while running, 0x01 at game start.
        var t = _rb(state.ram, 0xC2)
        var terminal = t != 0x00 and t != 0x01
        return GameSignals(score, score - prev_score, 0, terminal)

    elif game == AtariGame.RIVERRAID:
        # Digits stored as value*8 (0,8,...,72); anything else reads as 0.
        var score = 0
        var mult = 1
        for addr in [87, 85, 83, 81, 79, 77]:
            var v = _rb(state.ram, addr)
            var d = v >> 3
            if (v & 7) != 0 or d > 9:
                d = 0
            score += d * mult
            mult *= 10
        # Terminal = lives byte going 0x59 (last life) -> 0x58 (empty).
        # game_aux carries the previous step's lives byte (ALE m_lives_byte).
        var lives_byte = _rb(state.ram, 0xC0)
        var terminal = lives_byte == 0x58 and Int(state.game_aux) == 0x59
        state.game_aux = Int32(lives_byte)
        var lives: Int
        if lives_byte == 0x58:
            lives = 4  # beginning of episode
        elif lives_byte == 0x59:
            lives = 1
        else:
            lives = lives_byte // 8 + 1
        return GameSignals(score, score - prev_score, lives, terminal)

    elif game == AtariGame.ROAD_RUNNER:
        # Four digits one per byte at 0xC9-0xCC; 0xA = blank zero.
        var score = 0
        var mult = 1
        for digit in range(4):
            var v = _rb(state.ram, 0xC9 + digit) & 0xF
            if v == 0xA:
                v = 0
            score += v * mult
            mult *= 10
        score *= 100
        var lives_byte = _rb(state.ram, 0xC4) & 0x7
        var terminal = lives_byte == 0 and (
            _rb(state.ram, 0xB9) != 0 or _rb(state.ram, 0xBD) != 0
        )
        return GameSignals(
            score, score - prev_score, lives_byte + 1, terminal
        )

    elif game == AtariGame.SKIING:
        # Score = elapsed time (counts UP); reward = NEGATIVE time delta.
        var centiseconds = get_decimal_score_2(state.ram, 0xEA, 0xE9)
        var score = _rb(state.ram, 0xE8) * 6000 + centiseconds
        var terminal = _rb(state.ram, 0x91) == 0xFF
        return GameSignals(score, prev_score - score, 0, terminal)

    elif game == AtariGame.SOLARIS:
        # Five digits displayed but six tracked.
        var score = get_decimal_score_3(state.ram, 0xDC, 0xDD, 0xDE) * 10
        var lives_byte = _rb(state.ram, 0xD9)
        return GameSignals(
            score, score - prev_score, lives_byte & 0xF, lives_byte == 0
        )

    elif game == AtariGame.STAR_GUNNER:
        # Four digits one per byte; 10 = blank zero.
        var score = 0
        var mult = 1
        for addr in [0x83, 0x84, 0x85, 0x86]:
            var v = _rb(state.ram, addr) & 0x0F
            if v == 10:
                v = 0
            score += v * mult
            mult *= 10
        score *= 100
        var lives_byte = _rb(state.ram, 0x87)
        # game_aux latches "game started" (lives byte reaches 5) so the
        # title screen's garbage lives byte reads as the initial 5.
        if lives_byte == 0x05:
            state.game_aux = 1
        var lives = (lives_byte & 0xF) if state.game_aux != 0 else 5
        return GameSignals(
            score, score - prev_score, lives, lives_byte == 0
        )

    elif game == AtariGame.SURROUND:
        var their_score = get_decimal_score(state.ram, 0xF6)
        var my_score = get_decimal_score(state.ram, 0xF7)
        var score = my_score - their_score
        var terminal = their_score == 10 or my_score == 10
        return GameSignals(score, score - prev_score, 0, terminal)

    elif game == AtariGame.TENNIS:
        # Reward: set-point delta if it changed this step, else game-score
        # delta. game_aux packs the previous deltas as two signed bytes
        # (low = score, high = points); 0 decodes to (0, 0) = reset state.
        var my_score = _rb(state.ram, 0xC5)
        var oppt_score = _rb(state.ram, 0xC6)
        var my_points = _rb(state.ram, 0xC7)
        var oppt_points = _rb(state.ram, 0xC8)
        var delta_score = my_score - oppt_score
        var delta_points = my_points - oppt_points
        var aux = Int(state.game_aux)
        var prev_ds = aux & 0xFF
        if prev_ds >= 128:
            prev_ds -= 256
        var prev_dp = (aux >> 8) & 0xFF
        if prev_dp >= 128:
            prev_dp -= 256
        var reward = 0
        if prev_dp != delta_points:
            reward = delta_points - prev_dp
        elif prev_ds != delta_score:
            reward = delta_score - prev_ds
        state.game_aux = Int32(
            ((delta_points & 0xFF) << 8) | (delta_score & 0xFF)
        )
        var terminal = (
            (my_points >= 6 and delta_points >= 2)
            or (oppt_points >= 6 and -delta_points >= 2)
            or my_points == 7
            or oppt_points == 7
        )
        return GameSignals(
            prev_score + reward, reward, 0, terminal
        )

    elif game == AtariGame.TIME_PILOT:
        var score = get_decimal_score_2(state.ram, 0x8D, 0x8F) * 100
        var terminal = _rb(state.ram, 0xA0) != 0
        # Only trust the lives byte while actually flying (screen 2);
        # game_aux carries the last good value across other screens.
        if (_rb(state.ram, 0x80) & 0xF) == 2:
            state.game_aux = Int32((_rb(state.ram, 0x8B) & 0x7) + 1)
        return GameSignals(
            score, score - prev_score, Int(state.game_aux), terminal
        )

    elif game == AtariGame.UP_N_DOWN:
        var score = get_decimal_score_3(state.ram, 0x82, 0x81, 0x80)
        var lives_byte = _rb(state.ram, 0x86) & 0xF
        var terminal = _rb(state.ram, 0x94) > 0x40 and lives_byte == 0
        return GameSignals(
            score, score - prev_score, lives_byte + 1, terminal
        )

    elif game == AtariGame.VENTURE:
        var score = get_decimal_score_2(state.ram, 0xC8, 0xC7) * 100
        var lives_byte = _rb(state.ram, 0xC6)
        var terminal = (
            lives_byte == 0
            and _rb(state.ram, 0xCD) == 0xFF
            and (_rb(state.ram, 0xBF) & 0x80) != 0
        )
        return GameSignals(
            score, score - prev_score, (lives_byte & 0x7) + 1, terminal
        )

    elif game == AtariGame.VIDEO_PINBALL:
        var score = get_decimal_score_3(state.ram, 0xB0, 0xB2, 0xB4)
        var terminal = (_rb(state.ram, 0xAF) & 0x1) != 0
        # Lives display as ball number; extra-ball flag adds one.
        var lives = (
            4 + (_rb(state.ram, 0xA8) & 0x1) - (_rb(state.ram, 0x99) & 0x7)
        )
        return GameSignals(score, score - prev_score, lives, terminal)

    elif game == AtariGame.WIZARD_OF_WOR:
        var score = get_decimal_score_2(state.ram, 0x86, 0x88)
        if score >= 8000:
            score -= 8000  # ALE MGB: score does not go beyond 999
        score *= 100
        var new_lives = _rb(state.ram, 0x8D) & 15
        var terminal = new_lives == 0 and _rb(state.ram, 0xF4) == 0xF8
        # Lives drop when entering the play field; only commit the value
        # while waiting (0xD7 bit 0 clear). game_aux carries it across.
        if (_rb(state.ram, 0xD7) & 0x1) == 0:
            state.game_aux = Int32(new_lives)
        return GameSignals(
            score, score - prev_score, Int(state.game_aux), terminal
        )

    elif game == AtariGame.YARS_REVENGE:
        var score = get_decimal_score_3(state.ram, 0xE2, 0xE1, 0xE0)
        var lives = _rb(state.ram, 0x9E) >> 4
        return GameSignals(score, score - prev_score, lives, lives == 0)

    elif game == AtariGame.ZAXXON:
        var score = get_decimal_score_2(state.ram, 0xE9, 0xE8) * 100
        # Lives read 0 before console RESET is pushed; our reset holds it.
        var lives = _rb(state.ram, 0xEA) & 0x7
        return GameSignals(score, score - prev_score, lives, lives == 0)

    return GameSignals(0, 0, 0, False)
