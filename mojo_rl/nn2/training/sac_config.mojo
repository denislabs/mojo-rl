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
from ..core.save_scalar import SaveScalar, SaveI, SaveBool
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
    # Phase B.3 — global L2 grad-norm clip applied uniformly to all 3
    # Adam optimizers (actor, critic1, critic2). `0.0` (default) is the
    # disabled sentinel — preserves bit-identity. Production runs would
    # typically set this to a finite value (e.g. 40.0 for CleanRL-style
    # SAC) once tuned.
    var max_grad_norm:        SaveScalar[DT]
    var learning_starts:      SaveI
    var window_size:          SaveI
    # Phase C.5 — mixed-precision hint. `True` asks the trainer to run
    # forward/backward kernels in bf16 compute (params + Adam moments
    # stay fp32). The Trainer struct picks `POLICY = Bf16Compute` vs
    # `NoAMP` at the comptime type level; this Saveable field just
    # records the user's choice so it survives checkpoint round-trips.
    # Default `False` → POLICY=NoAMP → bit-identical to pre-C.5.
    var use_bf16:             SaveBool
    # Phase C.4b — ERE (Emphasizing Recent Experience, Wang & Ross
    # 2019) auto-routing. `use_ere=True` makes the GPU factory call
    # `buf_gpu.enable_ere(eta, c_min, k_max)`. `eta` is the decay
    # factor (smaller = more recency bias; 1.0 = uniform), `c_min`
    # the lower clamp on the recent window (should be ≥ BATCH), and
    # `k_max` the cycle length after which `η^k` resets. Defaults
    # match the GPUReplay defaults. `use_ere=False` (default) → no
    # call → uniform sampling → bit-identical to pre-C.4.
    var use_ere:              SaveBool
    var ere_eta:              SaveScalar[DT]
    var ere_c_min:            SaveI
    var ere_k_max:            SaveI

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
            max_grad_norm=SaveScalar[DT](Scalar[DT](0.0)),
            learning_starts=SaveI(1_000),
            window_size=SaveI(10),
            use_bf16=SaveBool(False),
            use_ere=SaveBool(False),
            ere_eta=SaveScalar[DT](Scalar[DT](0.996)),
            ere_c_min=SaveI(256),
            ere_k_max=SaveI(1_000),
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
