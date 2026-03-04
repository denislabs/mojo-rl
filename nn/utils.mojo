"""Shared utility functions for nn and deep_agents.

Provides free functions used across multiple agents to reduce code duplication.
"""

from .constants import dtype


fn fill_inline[
    N: Int, T: DType
](src: List[Scalar[T]], mut dst: InlineArray[Scalar[dtype], N]):
    """Fill an InlineArray in-place from a List[Scalar[T]].

    Converts each element via Scalar[dtype] cast.  Avoids ImplicitlyCopyable
    requirement on InlineArray — callers pass dst as mut instead of returning.

    Parameters:
        N: Size of the destination InlineArray (compile-time).
        T: Source element DType.

    Args:
        src: Source list (must have at least N elements).
        dst: Destination InlineArray (written in-place).
    """
    for i in range(N):
        dst[i] = Scalar[dtype](src[i])


fn obs_to_inline[
    N: Int, T: DType
](src: List[Scalar[T]]) -> InlineArray[Scalar[dtype], N]:
    """Convert a List observation to a fixed-size InlineArray with dtype cast.

    One-liner replacement for the two-step fill_inline pattern:

        var arr = InlineArray[Scalar[dtype], N](uninitialized=True)
        fill_inline(obs, arr)
        # use arr...

    becomes:

        var arr = obs_to_inline[N](obs)

    Parameters:
        N: Destination array size (compile-time).
        T: Source element DType.

    Args:
        src: Source list (must have at least N elements).

    Returns:
        InlineArray[Scalar[dtype], N] filled from src.
    """
    var arr = InlineArray[Scalar[dtype], N](uninitialized=True)
    for i in range(N):
        arr[i] = Scalar[dtype](src[i])
    return arr^


fn concat_obs_action[
    OBS: Int, ACT: Int
](
    obs: InlineArray[Scalar[dtype], OBS],
    act: InlineArray[Scalar[dtype], ACT],
    mut dst: InlineArray[Scalar[dtype], OBS + ACT],
):
    """Concatenate obs + action into a pre-allocated critic input buffer.

    Used by DDPG/TD3/SAC to build (obs ‖ action) critic inputs. Eliminates
    6-10 lines of index arithmetic appearing 3-4× per train_step.

    Parameters:
        OBS: Observation dimension (compile-time).
        ACT: Action dimension (compile-time).

    Args:
        obs: Observation InlineArray of length OBS (consumed).
        act: Action InlineArray of length ACT (consumed).
        dst: Destination InlineArray of length OBS+ACT (written in-place).
    """
    for i in range(OBS):
        dst[i] = obs[i]
    for i in range(ACT):
        dst[OBS + i] = act[i]
