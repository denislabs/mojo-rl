"""EpisodeTracker — rolling window over recent episode returns.

Encapsulates the "accumulate current episode return + roll past N
returns into a fixed-size window" pattern that both PPO and SAC
hand-rolled in Phase 6/7. `mean_return()` reports the windowed mean for
logging/exit-criterion checks.

The window is pre-filled with `initial_fill` so early-training prints
have a sensible value (the Phase 6 PPO example used -1600 for Pendulum;
Phase 7 SAC used -1250 — these are roughly the random-policy baseline).
"""

from ..constants import DT


@fieldwise_init
struct EpisodeTracker(Movable & ImplicitlyDestructible):
    var window: List[Scalar[DT]]
    var window_size: Int
    var idx: Int
    var current_return: Scalar[DT]
    var ep_count: Int

    @staticmethod
    def new(window_size: Int, initial_fill: Scalar[DT]) -> Self:
        return Self(
            window=List[Scalar[DT]](
                length=window_size, fill=initial_fill
            ),
            window_size=window_size,
            idx=0,
            current_return=Scalar[DT](0.0),
            ep_count=0,
        )

    def add_reward(mut self, reward: Scalar[DT]):
        """Accumulate one env-step reward into the current episode."""
        self.current_return = self.current_return + reward

    def end_episode(mut self):
        """Push current_return into the window and reset."""
        self.window[self.idx] = self.current_return
        self.idx = (self.idx + 1) % self.window_size
        self.ep_count += 1
        self.current_return = Scalar[DT](0.0)

    def mean_return(self) -> Scalar[DT]:
        """Mean over the rolling window (over `window_size` past returns)."""
        var s: Scalar[DT] = 0.0
        for k in range(self.window_size):
            s = s + self.window[k]
        return s / Scalar[DT](self.window_size)
