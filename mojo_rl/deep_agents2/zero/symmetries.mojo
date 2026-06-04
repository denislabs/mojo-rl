"""Board-symmetry augmentation for zero-series self-play targets.

Each self-play sample `(obs, π)` can be replicated under the board's symmetry
group: the same geometric permutation is applied to both the observation planes
and the visit-count policy, so the supervised target stays valid (the value
target `z` is a scalar — symmetry-invariant — and is copied unchanged). This
multiplies effective data and bakes the game's invariances into the net.

`BoardAugmenter` is a compile-time strategy (matches the legacy AlphaZero
surface). `sym_idx == 0` is always the identity. Layout conventions match the
board envs:
  * obs: `PLANES × ROWS × COLS` row-major (plane-major flat) — TicTacToe is
    `3 × 3 × 3` (mine/opp/empty), Connect4 `3 × 6 × 7`.
  * policy: one entry per cell (`D4` square games, `ACT == SIDE*SIDE`) or per
    column (`HFlip` column games, `ACT == COLS`).

Ported from `deep_agents/alphazero/strategies.mojo` onto `nn2` (`DT`).
"""

from mojo_rl.nn2.constants import DT


trait BoardAugmenter:
    """Symmetry augmentation of `(obs, policy)` samples. `sym_idx == 0` is the
    identity; total count is `NUM_SYMMETRIES`."""

    comptime NUM_SYMMETRIES: Int

    @staticmethod
    def augment_obs[OBS: Int](
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        sym_idx: Int,
        mut out: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ):
        """Write symmetry `sym_idx` of `obs[OBS]` into `out[OBS]`."""
        ...

    @staticmethod
    def augment_policy[ACT: Int](
        policy: UnsafePointer[Scalar[DT], MutAnyOrigin],
        sym_idx: Int,
        mut out: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ):
        """Write symmetry `sym_idx` of `policy[ACT]` into `out[ACT]`, with the
        same per-cell permutation as `augment_obs`."""
        ...


# ──────────────────────────────────────────────────────────────────────
# Identity (no augmentation) — for asymmetric games / single-player envs.
# ──────────────────────────────────────────────────────────────────────


struct IdentityAugmenter(BoardAugmenter):
    comptime NUM_SYMMETRIES: Int = 1

    @staticmethod
    def augment_obs[OBS: Int](
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        sym_idx: Int,
        mut out: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ):
        for i in range(OBS):
            out[i] = obs[i]

    @staticmethod
    def augment_policy[ACT: Int](
        policy: UnsafePointer[Scalar[DT], MutAnyOrigin],
        sym_idx: Int,
        mut out: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ):
        for i in range(ACT):
            out[i] = policy[i]


# ──────────────────────────────────────────────────────────────────────
# D4 dihedral group (8 symmetries) — square boards (TicTacToe, Go, Othello).
# ──────────────────────────────────────────────────────────────────────


@always_inline
def _d4_src(sym_idx: Int, side: Int, r: Int, c: Int) -> Tuple[Int, Int]:
    """Pull-style source cell for D4 symmetry `sym_idx`: `dst[r,c]=src[sr,sc]`.
      0 id · 1 h-flip · 2 v-flip · 3 rot180 · 4 rot90CW · 5 rot270CW
      6 transpose (main diag) · 7 anti-transpose (anti diag)."""
    if sym_idx == 1:
        return (r, side - 1 - c)
    elif sym_idx == 2:
        return (side - 1 - r, c)
    elif sym_idx == 3:
        return (side - 1 - r, side - 1 - c)
    elif sym_idx == 4:
        return (side - 1 - c, r)
    elif sym_idx == 5:
        return (c, side - 1 - r)
    elif sym_idx == 6:
        return (c, r)
    else:
        return (side - 1 - c, side - 1 - r)


struct D4SquareAugmenter[SIDE: Int, PLANES: Int](BoardAugmenter):
    """Full D4 (8 symmetries) for square boards. obs `PLANES×SIDE×SIDE`,
    policy `SIDE×SIDE` (`ACT == SIDE*SIDE`)."""

    comptime NUM_SYMMETRIES: Int = 8

    @staticmethod
    def augment_obs[OBS: Int](
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        sym_idx: Int,
        mut out: UnsafePointer[Scalar[DT], MutAnyOrigin],
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
                    var src = _d4_src(sym_idx, Self.SIDE, r, c)
                    out[p_off + r * Self.SIDE + c] = obs[
                        p_off + src[0] * Self.SIDE + src[1]
                    ]

    @staticmethod
    def augment_policy[ACT: Int](
        policy: UnsafePointer[Scalar[DT], MutAnyOrigin],
        sym_idx: Int,
        mut out: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ):
        if sym_idx == 0:
            for i in range(ACT):
                out[i] = policy[i]
            return
        for r in range(Self.SIDE):
            for c in range(Self.SIDE):
                var src = _d4_src(sym_idx, Self.SIDE, r, c)
                out[r * Self.SIDE + c] = policy[src[0] * Self.SIDE + src[1]]


# ──────────────────────────────────────────────────────────────────────
# Horizontal flip (2 symmetries) — column-action games (Connect Four).
# ──────────────────────────────────────────────────────────────────────


struct HFlipColumnAugmenter[ROWS: Int, COLS: Int, PLANES: Int](BoardAugmenter):
    """Identity + horizontal flip for column-action games. obs
    `PLANES×ROWS×COLS`, policy `COLS` (action `c ↔ COLS-1-c`)."""

    comptime NUM_SYMMETRIES: Int = 2

    @staticmethod
    def augment_obs[OBS: Int](
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        sym_idx: Int,
        mut out: UnsafePointer[Scalar[DT], MutAnyOrigin],
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
    def augment_policy[ACT: Int](
        policy: UnsafePointer[Scalar[DT], MutAnyOrigin],
        sym_idx: Int,
        mut out: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ):
        if sym_idx == 0:
            for i in range(ACT):
                out[i] = policy[i]
            return
        for c in range(ACT):
            out[c] = policy[ACT - 1 - c]
