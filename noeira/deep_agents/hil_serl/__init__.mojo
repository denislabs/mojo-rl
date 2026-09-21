"""HIL-SERL — human-in-the-loop sample-efficient RL (Luo et al., 2024),
the RLPD half: SAC whose replay keeps a pinned prefix of demonstrations and
draws half of every minibatch from it, optionally with a behaviour-cloning
term on that half.

    from noeira.deep_agents.hil_serl import HilSerlConfig, apply_hil_serl

    var hil = HilSerlConfig()
    # in the argument loop:
    elif hil.try_parse(flag, value, has_value, "sac task"): pass
    hil.validate("sac task")
    ...                                    # agent built, trainer set up
    apply_hil_serl(agent.trainer, hil, logger, "sac task")

The demos come from `deep_agents/demos` (the `.demo` file, the recorder, the
DAgger handover). The BC term itself is SAC's (`SACActorLoss.set_bc`).
Reference: `references/hil-serl-main/examples/train_rlpd.py`.
"""

from .config import HilSerlConfig
from .setup import apply_hil_serl
