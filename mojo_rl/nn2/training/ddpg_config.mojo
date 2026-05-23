"""DDPGConfig — runtime hyperparameters for the DDPG trainer.

Phase A.4. Defaults match `DDPGTrainer.make()`'s keyword args.
"""

from ..constants import DT
from ..core.saveable import Saveable
from ..core.save_scalar import SaveScalar, SaveI
from ..core.state_walker import dump_state, load_state


@fieldwise_init
struct DDPGConfig(Saveable):
    var actor_lr:             SaveScalar[DT]
    var critic_lr:            SaveScalar[DT]
    var gamma:                SaveScalar[DT]
    var tau:                  SaveScalar[DT]
    var action_scale:         SaveScalar[DT]
    var noise_scale:          SaveScalar[DT]
    var initial_episode_fill: SaveScalar[DT]
    var learning_starts:      SaveI
    var window_size:          SaveI

    @staticmethod
    def default() -> Self:
        return Self(
            actor_lr=SaveScalar[DT](Scalar[DT](1e-4)),
            critic_lr=SaveScalar[DT](Scalar[DT](1e-3)),
            gamma=SaveScalar[DT](Scalar[DT](0.99)),
            tau=SaveScalar[DT](Scalar[DT](0.005)),
            action_scale=SaveScalar[DT](Scalar[DT](1.0)),
            noise_scale=SaveScalar[DT](Scalar[DT](0.1)),
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
        var dump = String("")
        self.save(dump, String(""))
        print("--- DDPGConfig ---")
        print(dump, end="")
        print("------------------")
