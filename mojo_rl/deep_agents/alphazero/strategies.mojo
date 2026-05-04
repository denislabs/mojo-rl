"""AlphaZero strategy traits — composable building blocks.

Mirrors the muzero/strategies.mojo pattern: comptime traits + named
concrete impls that select training behavior. Currently AZ-specific;
MuZero self-play can import these once it grows board augmentation.

Traits:
  - BoardAugmenter: data augmentation via board symmetries.
"""
from std.memory import UnsafePointer
from mojo_rl.nn import dtype


# ═══════════════════════════════════════════════════════════════════════════
# BoardAugmenter — symmetry augmentation on (obs, policy) replay samples
# ═══════════════════════════════════════════════════════════════════════════


trait BoardAugmenter:
    """Symmetry-based augmentation of (obs, policy) replay samples.

    Each emitted replay sample is replicated NUM_SYMMETRIES times under
    symmetry transformations of the board. Both obs and policy are
    permuted consistently so the supervised target stays valid.

    sym_idx == 0 must be the identity transform.
    """

    comptime NUM_SYMMETRIES: Int
    """Total number of symmetries including identity."""

    @staticmethod
    def augment_obs[
        OBS: Int,
    ](
        obs: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        sym_idx: Int,
        mut out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ):
        """Apply symmetry sym_idx to the observation vector.

        Args:
            obs: Input observation [OBS].
            sym_idx: Symmetry index in [0, NUM_SYMMETRIES).
            out: Output buffer [OBS] to write permuted observation.
        """
        ...

    @staticmethod
    def augment_policy[
        ACT: Int,
    ](
        policy: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        sym_idx: Int,
        mut out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ):
        """Apply symmetry sym_idx to the policy vector.

        Action permutation must be consistent with augment_obs: rotating
        the board 90° also rotates per-cell action indices.

        Args:
            policy: Input policy [ACT].
            sym_idx: Symmetry index in [0, NUM_SYMMETRIES).
            out: Output buffer [ACT] to write permuted policy.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# IdentityAugmenter — no augmentation
# ═══════════════════════════════════════════════════════════════════════════


struct IdentityAugmenter(BoardAugmenter):
    """No augmentation: a single identity copy per real sample.

    Use for games without exploitable symmetry (chess, shogi),
    single-player envs (CartPole, Atari), or evaluation-only configs.
    """

    comptime NUM_SYMMETRIES: Int = 1

    @staticmethod
    def augment_obs[
        OBS: Int,
    ](
        obs: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        sym_idx: Int,
        mut out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ):
        for i in range(OBS):
            out[i] = obs[i]

    @staticmethod
    def augment_policy[
        ACT: Int,
    ](
        policy: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        sym_idx: Int,
        mut out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ):
        for i in range(ACT):
            out[i] = policy[i]


# ═══════════════════════════════════════════════════════════════════════════
# D4SquareAugmenter — full dihedral group on a square board
# ═══════════════════════════════════════════════════════════════════════════


struct D4SquareAugmenter[
    SIDE: Int,
    PLANES: Int,
](BoardAugmenter):
    """Full D4 dihedral group (8 symmetries) for square boards.

    Used by TicTacToe (3×3), Go (square boards), Othello.

    Layouts:
      - obs: PLANES × SIDE × SIDE (row-major), so OBS == PLANES * SIDE * SIDE.
      - policy: SIDE × SIDE (one entry per cell), so ACT == SIDE * SIDE.

    Symmetry indices (pull-style: dst[r,c] = src[sr,sc]):
      0 = identity
      1 = horizontal flip (column reversal)
      2 = vertical flip (row reversal)
      3 = 180° rotation
      4 = 90° rotation CW
      5 = 270° rotation CW (= 90° CCW)
      6 = main-diagonal reflection (transpose)
      7 = anti-diagonal reflection (anti-transpose)
    """

    comptime NUM_SYMMETRIES: Int = 8

    @staticmethod
    def augment_obs[
        OBS: Int,
    ](
        obs: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        sym_idx: Int,
        mut out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ):
        if sym_idx == 0:
            for i in range(OBS):
                out[i] = obs[i]
            return
        comptime CHAN = Self.SIDE * Self.SIDE
        for plane in range(Self.PLANES):
            var p_off = plane * CHAN
            for r in range(Self.SIDE):
                for c in range(Self.SIDE):
                    var sr: Int
                    var sc: Int
                    if sym_idx == 1:
                        sr = r
                        sc = Self.SIDE - 1 - c
                    elif sym_idx == 2:
                        sr = Self.SIDE - 1 - r
                        sc = c
                    elif sym_idx == 3:
                        sr = Self.SIDE - 1 - r
                        sc = Self.SIDE - 1 - c
                    elif sym_idx == 4:
                        sr = Self.SIDE - 1 - c
                        sc = r
                    elif sym_idx == 5:
                        sr = c
                        sc = Self.SIDE - 1 - r
                    elif sym_idx == 6:
                        sr = c
                        sc = r
                    else:
                        sr = Self.SIDE - 1 - c
                        sc = Self.SIDE - 1 - r
                    out[p_off + r * Self.SIDE + c] = obs[
                        p_off + sr * Self.SIDE + sc
                    ]

    @staticmethod
    def augment_policy[
        ACT: Int,
    ](
        policy: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        sym_idx: Int,
        mut out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ):
        if sym_idx == 0:
            for i in range(ACT):
                out[i] = policy[i]
            return
        for r in range(Self.SIDE):
            for c in range(Self.SIDE):
                var sr: Int
                var sc: Int
                if sym_idx == 1:
                    sr = r
                    sc = Self.SIDE - 1 - c
                elif sym_idx == 2:
                    sr = Self.SIDE - 1 - r
                    sc = c
                elif sym_idx == 3:
                    sr = Self.SIDE - 1 - r
                    sc = Self.SIDE - 1 - c
                elif sym_idx == 4:
                    sr = Self.SIDE - 1 - c
                    sc = r
                elif sym_idx == 5:
                    sr = c
                    sc = Self.SIDE - 1 - r
                elif sym_idx == 6:
                    sr = c
                    sc = r
                else:
                    sr = Self.SIDE - 1 - c
                    sc = Self.SIDE - 1 - r
                out[r * Self.SIDE + c] = policy[sr * Self.SIDE + sc]


# ═══════════════════════════════════════════════════════════════════════════
# HFlipColumnAugmenter — horizontal flip on column-action games
# ═══════════════════════════════════════════════════════════════════════════


struct HFlipColumnAugmenter[
    ROWS: Int,
    COLS: Int,
    PLANES: Int,
](BoardAugmenter):
    """Horizontal flip for column-action games (e.g., Connect Four).

    Layouts:
      - obs: PLANES × ROWS × COLS (h-flip mirrors the column index).
      - policy: COLS (one entry per column, ACT == COLS); h-flip
        reverses the column ordering: action c ↔ action COLS-1-c.

    NUM_SYMMETRIES = 2: identity, horizontal flip.
    """

    comptime NUM_SYMMETRIES: Int = 2

    @staticmethod
    def augment_obs[
        OBS: Int,
    ](
        obs: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        sym_idx: Int,
        mut out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ):
        if sym_idx == 0:
            for i in range(OBS):
                out[i] = obs[i]
            return
        comptime CHAN = Self.ROWS * Self.COLS
        for plane in range(Self.PLANES):
            var p_off = plane * CHAN
            for row in range(Self.ROWS):
                for col in range(Self.COLS):
                    out[p_off + row * Self.COLS + col] = obs[
                        p_off + row * Self.COLS + (Self.COLS - 1 - col)
                    ]

    @staticmethod
    def augment_policy[
        ACT: Int,
    ](
        policy: UnsafePointer[Scalar[dtype], MutAnyOrigin],
        sym_idx: Int,
        mut out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ):
        if sym_idx == 0:
            for i in range(ACT):
                out[i] = policy[i]
            return
        for c in range(ACT):
            out[c] = policy[ACT - 1 - c]
