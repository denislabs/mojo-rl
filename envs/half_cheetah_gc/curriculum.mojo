"""Curriculum learning scheduler for HalfCheetahGC environment.

Provides progressive difficulty adjustment during training:
- Starts with very lenient pitch bounds (basically no constraint)
- Gradually tightens bounds to prevent somersaulting

Usage:
    var scheduler = HalfCheetahCurriculum()

    # In training loop:
    var progress = Float64(iteration) / Float64(total_iterations)
    var params = scheduler.get_params(progress)
    HalfCheetahGC.init_model_gpu_with_curriculum(ctx, model_buf, params[1])
"""

from .constants_gc import HalfCheetahGCConstants
from core.env_traits import CurriculumScheduler


@fieldwise_init
struct HalfCheetahCurriculum(CurriculumScheduler):
    """Curriculum scheduler for HalfCheetah environment.

    Linearly interpolates max_pitch from lenient initial value
    to strict final value based on training progress.

    Index 0 (min_height) is unused for HalfCheetah — always 0.0.
    Index 1 (max_pitch) interpolates from ~172 deg to ~57 deg.
    """

    comptime initial_max_pitch: Scalar[DType.float64] = Scalar[DType.float64](
        3.0
    )
    comptime final_max_pitch: Scalar[DType.float64] = Scalar[DType.float64](
        1.0
    )

    @staticmethod
    fn get_params[DTYPE: DType](progress: Scalar[DTYPE]) -> List[Scalar[DTYPE]]:
        """Get curriculum parameters for given training progress.

        Uses linear interpolation from initial to final values.

        Args:
            progress: Training progress from 0.0 (start) to 1.0 (end).
                     Values outside [0, 1] are clamped.

        Returns:
            List with [min_height (unused, 0.0), max_pitch].
        """
        # Clamp progress to [0, 1]
        var p = progress
        if p < Scalar[DTYPE](0.0):
            p = Scalar[DTYPE](0.0)
        elif p > Scalar[DTYPE](1.0):
            p = Scalar[DTYPE](1.0)

        var params = List[Scalar[DTYPE]](capacity=2)

        # Index 0: min_height (unused for HalfCheetah)
        params[0] = Scalar[DTYPE](0.0)

        # Index 1: max_pitch - linear interpolation
        params[1] = Scalar[DTYPE](Self.initial_max_pitch) + p * (
            Scalar[DTYPE](Self.final_max_pitch)
            - Scalar[DTYPE](Self.initial_max_pitch)
        )

        return params^

    @staticmethod
    fn get_stage_name[DTYPE: DType](progress: Scalar[DTYPE]) -> String:
        """Get human-readable curriculum stage name.

        Args:
            progress: Training progress from 0.0 to 1.0.

        Returns:
            Stage name string.
        """
        if progress < Scalar[DTYPE](0.25):
            return "Stage 1/4: Very Lenient"
        elif progress < Scalar[DTYPE](0.5):
            return "Stage 2/4: Lenient"
        elif progress < Scalar[DTYPE](0.75):
            return "Stage 3/4: Moderate"
        else:
            return "Stage 4/4: Strict"
