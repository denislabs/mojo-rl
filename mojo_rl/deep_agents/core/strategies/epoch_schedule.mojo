"""Epoch schedule strategies for on-policy agents.

Stateless strategy types controlling the number of epochs, minibatch size,
shuffling, and advantage normalization for on-policy training.

Implementations:
  - SinglePass: 1 epoch, full rollout, no shuffle, no KL early stop (A2C)
  - MultiEpochMinibatch: configurable epochs/mb_size, shuffle, KL early stop (PPO)
"""


trait EpochSchedule:
    """Trait for epoch schedule strategies."""

    comptime USES_SHUFFLE: Bool
    comptime USES_KL_EARLY_STOP: Bool
    comptime SUPPORTS_MINIBATCH_NORM: Bool

    @staticmethod
    def get_num_epochs(num_epochs: Int) -> Int:
        ...

    @staticmethod
    def get_minibatch_size(minibatch_size: Int, buf_len: Int) -> Int:
        ...


# =============================================================================
# SinglePass — 1 epoch, full rollout (A2C)
# =============================================================================


struct SinglePass(EpochSchedule):
    """Single pass over the entire rollout buffer.

    Used by A2C: 1 epoch, minibatch = full rollout, no shuffle, no KL early stop.
    """

    comptime USES_SHUFFLE: Bool = False
    comptime USES_KL_EARLY_STOP: Bool = False
    comptime SUPPORTS_MINIBATCH_NORM: Bool = False

    @staticmethod
    def get_num_epochs(num_epochs: Int) -> Int:
        """Always returns 1."""
        return 1

    @staticmethod
    def get_minibatch_size(minibatch_size: Int, buf_len: Int) -> Int:
        """Returns the full buffer length."""
        return buf_len


# =============================================================================
# MultiEpochMinibatch — configurable multi-epoch training (PPO)
# =============================================================================


struct MultiEpochMinibatch(EpochSchedule):
    """Multi-epoch minibatch training with shuffle and KL early stopping.

    Used by PPO: configurable epochs and minibatch size, Fisher-Yates shuffle,
    per-minibatch advantage normalization, KL early stopping.
    """

    comptime USES_SHUFFLE: Bool = True
    comptime USES_KL_EARLY_STOP: Bool = True
    comptime SUPPORTS_MINIBATCH_NORM: Bool = True

    @staticmethod
    def get_num_epochs(num_epochs: Int) -> Int:
        """Returns the configured number of epochs."""
        return num_epochs

    @staticmethod
    def get_minibatch_size(minibatch_size: Int, buf_len: Int) -> Int:
        """Returns the configured minibatch size."""
        return minibatch_size
