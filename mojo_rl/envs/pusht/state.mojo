"""PushT observation state.

Observation: 18-D vector = [k0x, k0y, ..., k7x, k7y, agent_x, agent_y]
(8 T-block keypoints in world coordinates + agent position).

This matches the `environment_state_agent_pos` flavor of the pymunk reference,
flattened into a single vector for use with feed-forward RL agents.
"""

from mojo_rl.core import State
from .constants import PConstants


struct PushTState[DTYPE: DType](
    Copyable, ImplicitlyCopyable, Movable, State
):
    """Flattened keypoints + agent_pos observation."""

    var keypoints: InlineArray[Scalar[Self.DTYPE], PConstants.KEYPOINTS_DIM]
    var agent_pos: InlineArray[Scalar[Self.DTYPE], PConstants.AGENT_POS_DIM]

    def __init__(out self):
        self.keypoints = InlineArray[
            Scalar[Self.DTYPE], PConstants.KEYPOINTS_DIM
        ](fill=Scalar[Self.DTYPE](0.0))
        self.agent_pos = InlineArray[
            Scalar[Self.DTYPE], PConstants.AGENT_POS_DIM
        ](fill=Scalar[Self.DTYPE](0.0))

    def __init__(out self, *, copy: Self):
        self.keypoints = copy.keypoints
        self.agent_pos = copy.agent_pos

    def __init__(out self, *, deinit take: Self):
        self.keypoints = take.keypoints
        self.agent_pos = take.agent_pos

    def __eq__(self, other: Self) -> Bool:
        for i in range(PConstants.KEYPOINTS_DIM):
            if self.keypoints[i] != other.keypoints[i]:
                return False
        for i in range(PConstants.AGENT_POS_DIM):
            if self.agent_pos[i] != other.agent_pos[i]:
                return False
        return True

    def to_list(self) -> List[Scalar[Self.DTYPE]]:
        var out = List[Scalar[Self.DTYPE]](capacity=PConstants.OBS_DIM)
        for i in range(PConstants.KEYPOINTS_DIM):
            out.append(self.keypoints[i])
        for i in range(PConstants.AGENT_POS_DIM):
            out.append(self.agent_pos[i])
        return out^
