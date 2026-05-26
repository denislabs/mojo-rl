"""J.1.b — TrainerGraph[*BLOCKS] walker.

Mirrors ComputeGraph: holds a variadic pack of TrainerBlock-conforming
structs, walks them in declared order during each train_step. Any block
that sets `state.did_step = False` short-circuits the rest of the walk.

All blocks must share the same (OBS, ACT, BATCH) — enforced at make.
"""

from ..training.trainer_block import TrainerBlock, TrainerState


struct TrainerGraph[
    OBS: Int,
    ACT: Int,
    BATCH: Int,
    *BLOCKS: TrainerBlock,
](Movable & ImplicitlyDestructible):
    comptime N = Self.BLOCKS.size

    var blocks: Tuple[*Self.BLOCKS]
    var state:  TrainerState[Self.OBS, Self.ACT, Self.BATCH]

    def __init__(out self):
        comptime assert Self.N >= 1, "TrainerGraph requires at least one block"
        comptime for i in range(Self.N):
            comptime assert Self.BLOCKS[i].OBS == Self.OBS, (
                "TrainerGraph: block.OBS must equal graph OBS"
            )
            comptime assert Self.BLOCKS[i].ACT == Self.ACT, (
                "TrainerGraph: block.ACT must equal graph ACT"
            )
            comptime assert Self.BLOCKS[i].BATCH == Self.BATCH, (
                "TrainerGraph: block.BATCH must equal graph BATCH"
            )
        self.blocks = Tuple[*Self.BLOCKS]()
        self.state  = TrainerState[Self.OBS, Self.ACT, Self.BATCH]()

    def step[target: StaticString = "cpu"](
        mut self, step_idx: Int
    ) raises -> Bool:
        self.state.step_idx = step_idx
        self.state.did_step = True
        # Comptime asserts at __init__ already proved that each block's
        # (OBS, ACT, BATCH) equals the graph's. Mojo's type checker
        # doesn't propagate that equality through the trait method's
        # typed state param, so rebind via UnsafePointer (no copy of
        # TrainerState, which holds Scratch fields).
        var state_p = UnsafePointer(to=self.state)
        comptime for k in range(Self.N):
            var block_state_p = rebind[
                UnsafePointer[
                    TrainerState[
                        Self.BLOCKS[k].OBS,
                        Self.BLOCKS[k].ACT,
                        Self.BLOCKS[k].BATCH,
                    ],
                    MutAnyOrigin,
                ]
            ](state_p)
            self.blocks[k].step_via[target](block_state_p[])
            if not self.state.did_step:
                return False
        return True
