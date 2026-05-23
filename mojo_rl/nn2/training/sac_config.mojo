"""SACConfig — runtime hyperparameters for the SAC trainer.

Phase A.4. Bundles every runtime knob the SAC trainer's `make` accepts
into one Saveable struct. Reflection-walked via `dump_state` /
`load_state` (`Saveable` conformance). `pretty_print` walks the same
fields for stdout-friendly diff vs defaults.

Compile-time architecture choice (network shapes, batch size, replay
capacity, OBS/ACT dims) stays on the trainer struct's type parameters
— those are NOT in this config since they encode the type identity
of the trainer.

Defaults match the keyword-argument values on `SACTrainer.make()` as of
Phase 4.6e. New SAC trainer instances built from `SACConfig.default()`
must produce bit-identical training to the keyword-arg path (verified
by the Pendulum 30k regression gate).
"""

from ..constants import DT
from ..core.saveable import Saveable
from ..core.save_scalar import SaveScalar, SaveI
from ..core.state_walker import dump_state, load_state


@fieldwise_init
struct SACConfig(Saveable):
    var actor_lr:             SaveScalar[DT]
    var critic_lr:            SaveScalar[DT]
    var alpha_lr:             SaveScalar[DT]
    var gamma:                SaveScalar[DT]
    var tau:                  SaveScalar[DT]
    var action_scale:         SaveScalar[DT]
    var init_alpha:           SaveScalar[DT]
    var target_entropy:       SaveScalar[DT]
    var initial_episode_fill: SaveScalar[DT]
    var learning_starts:      SaveI
    var window_size:          SaveI

    @staticmethod
    def default() -> Self:
        """Defaults match `SACTrainer.make()`'s keyword args. Changes
        here would shift the bit-identity baseline (`-167.572` on
        Pendulum 30k seed=42) — keep in lock-step with the trainer's
        defaults."""
        return Self(
            actor_lr=SaveScalar[DT](Scalar[DT](3e-4)),
            critic_lr=SaveScalar[DT](Scalar[DT](1e-3)),
            alpha_lr=SaveScalar[DT](Scalar[DT](3e-4)),
            gamma=SaveScalar[DT](Scalar[DT](0.99)),
            tau=SaveScalar[DT](Scalar[DT](0.005)),
            action_scale=SaveScalar[DT](Scalar[DT](1.0)),
            init_alpha=SaveScalar[DT](Scalar[DT](0.2)),
            target_entropy=SaveScalar[DT](Scalar[DT](-1.0)),
            initial_episode_fill=SaveScalar[DT](Scalar[DT](-1250.0)),
            learning_starts=SaveI(1_000),
            window_size=SaveI(10),
        )

    def save(self, mut out: String, prefix: String) raises:
        dump_state(self, out, prefix)

    def load(
        mut self, lines: List[String], mut idx: Int, prefix: String,
    ) raises:
        load_state(self, lines, idx, prefix)

    def pretty_print(self) raises:
        """Walk every Saveable field via reflection, print
        `<name> = <value>` lines to stdout. Same backend as save()."""
        var dump = String("")
        self.save(dump, String(""))
        print("--- SACConfig ---")
        print(dump, end="")
        print("-----------------")
