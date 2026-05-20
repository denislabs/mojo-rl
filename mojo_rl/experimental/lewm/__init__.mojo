"""LeWM (LeWorldModel) — experimental JEPA-based world model.

See `docs/LEWM_PORT_PLAN.md` for the design + phased delivery plan.

APIs in this subdirectory may break without notice. Phase 1 (autodiff
primitives) is stable and lives in `mojo_rl/nn/autodiff/primitives/`.
"""

from .action_embedder import ActionEmbedder
from .encoder import LeWMEncoder
from .adaln_models import Modulate, Gate, LayerNormNoAffine
