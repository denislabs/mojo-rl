"""LeWM (LeWorldModel JEPA) — nn port (experimental).

Agent assembly for the LeWM world model on the nn framework. The
general-purpose building blocks (Modulate/Gate/LayerNormNoAffine/SIGReg
primitives, ConditionalTransformerBlock, RepeatConditional) live in
`mojo_rl/nn/` proper; this package wires them into the JEPA encoder,
action embedder, AR predictor, loss graph, and offline trainer.

See docs/LEWM_NN2_PORT_PLAN.md.
"""

from .encoder import LeWMEncoder, ActionEmbedder, PredProj, ARPredictor
