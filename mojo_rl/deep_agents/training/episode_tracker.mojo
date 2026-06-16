"""EpisodeTracker — rolling window over recent episode returns.

Accumulates the current episode return and rolls past N returns into a
fixed-size window. `mean_return()` reports the windowed mean for
logging / exit-criterion checks.

The window is pre-filled with `initial_fill` so early-training prints
have a sensible value (typical: the random-policy baseline for the env,
e.g. ~-1600 for Pendulum-PPO, ~-1250 for Pendulum-SAC).
"""

from mojo_rl.nn.constants import DT


@fieldwise_init
struct EpisodeTracker(Movable & ImplicitlyDeletable):
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

    def add_complete_return(mut self, ret: Scalar[DT]):
        """Push an externally-tracked complete-episode return into the
        window without touching `current_return`.

        For N_ENVS GPU drivers (Phase B.5b): the driver maintains one
        cumulative-reward accumulator per env on the host, then calls
        this method for whichever env's `done` flag just fired. Single-
        env drivers keep using `add_reward` + `end_episode`.
        """
        self.window[self.idx] = ret
        self.idx = (self.idx + 1) % self.window_size
        self.ep_count += 1

    def mean_return(self) -> Scalar[DT]:
        """Mean over the rolling window (over `window_size` past returns)."""
        var s: Scalar[DT] = 0.0
        for k in range(self.window_size):
            s = s + self.window[k]
        return s / Scalar[DT](self.window_size)
