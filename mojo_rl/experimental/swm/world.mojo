"""What a world has to provide for the SWM-H training loop to run on it.

Phases 3-7 trained on the Mobius ring only, with the environment type welded
into the trainer. Phase 8 needs the same loop on a 2D place graph, and the
rule this codebase keeps tripping over is that a loop written twice drifts —
so the loop is written once, over this trait, and each world supplies the
five things the loop actually reads.

The two ORACLE methods at the bottom exist for the validity diagnostics only
(landmark R^2, nuisance R^2). No encoder under test ever sees them.
"""


trait SwmWorld(Copyable, Movable):
    comptime ELEM: DType
    """Element type of observations. A struct parameter cannot satisfy an
    associated alias directly, so each world binds `comptime ELEM = Self.dtype`."""

    def reset(mut self, seed: UInt64) raises:
        ...

    def step(mut self, action: Int) raises:
        ...

    def observation(mut self) -> List[Scalar[Self.ELEM]]:
        ...

    def place_id(self) -> Int:
        """Oracle place identity (design doc v2 §4.1)."""
        ...

    def place_label(self) -> Int:
        """The place index the TRAINER uses for the transport table and the
        per-place hinge. The oracle cell by default; under texture aliasing a
        world may return the texture label instead — what a content
        recogniser can deliver (G18 leg C, Phase 8)."""
        ...

    def explore_action(mut self) -> Int:
        """The data-collection policy: forward on a ring, a random walk on a
        grid. Part of the world because the transports are indexed by
        (action, place) and the world knows which actions exist."""
        ...

    def true_landmark(self) -> InlineArray[Scalar[Self.ELEM], 2]:
        """ORACLE: the transported part, unmixed and noiseless."""
        ...

    def nuisance_at(self, cell: Int) -> List[Scalar[Self.ELEM]]:
        """ORACLE: the non-transported texture of a cell."""
        ...
