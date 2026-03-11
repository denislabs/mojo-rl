"""Curriculum learning scheduler for Ant environment.

Provides progressive difficulty adjustment during training:
- Starts with lenient health bounds (easy to stay alive)
- Gradually tightens bounds toward MuJoCo defaults

Usage:
    var scheduler = AntCurriculum()

    # In training loop:
    var progress = Float64(iteration) / Float64(total_iterations)
    var params = scheduler.get_params(progress)
    Ant.update_curriculum_gpu(ctx, model_buf, params)
"""

from mojo_rl.core.env_traits import CurriculumScheduler


@fieldwise_init
struct AntCurriculum(CurriculumScheduler):
    """Curriculum scheduler for Ant environment.

    Linearly interpolates health bounds from lenient initial values
    to strict final values (MuJoCo defaults) based on training progress.

    Parameters interpolated:
    - min_height: 0.1 -> 0.2 (minimum healthy z height)
    - max_height: 1.5 -> 1.0 (maximum healthy z height)
    """

    comptime initial_min_height: Scalar[DType.float64] = Scalar[DType.float64](
        0.1
    )
    comptime final_min_height: Scalar[DType.float64] = Scalar[DType.float64](
        0.2
    )
    comptime initial_max_height: Scalar[DType.float64] = Scalar[DType.float64](
        1.5
    )
    comptime final_max_height: Scalar[DType.float64] = Scalar[DType.float64](
        1.0
    )

    @staticmethod
    fn get_params[DTYPE: DType](progress: Scalar[DTYPE]) -> List[Scalar[DTYPE]]:
        """Get curriculum parameters for given training progress.

        Args:
            progress: Training progress from 0.0 (start) to 1.0 (end).

        Returns:
            List with [min_height, max_height].
        """
        # Clamp progress to [0, 1]
        var p = progress
        if p < Scalar[DTYPE](0.0):
            p = Scalar[DTYPE](0.0)
        elif p > Scalar[DTYPE](1.0):
            p = Scalar[DTYPE](1.0)

        var params = List[Scalar[DTYPE]]()

        # Linear interpolation: initial + progress * (final - initial)
        params.append(
            Scalar[DTYPE](Self.initial_min_height)
            + p
            * (
                Scalar[DTYPE](Self.final_min_height)
                - Scalar[DTYPE](Self.initial_min_height)
            )
        )
        params.append(
            Scalar[DTYPE](Self.initial_max_height)
            + p
            * (
                Scalar[DTYPE](Self.final_max_height)
                - Scalar[DTYPE](Self.initial_max_height)
            )
        )

        return params^

    @staticmethod
    fn get_stage_name[DTYPE: DType](progress: Scalar[DTYPE]) -> String:
        """Get human-readable curriculum stage name."""
        if progress < Scalar[DTYPE](0.25):
            return "Stage 1/4: Very Lenient"
        elif progress < Scalar[DTYPE](0.5):
            return "Stage 2/4: Lenient"
        elif progress < Scalar[DTYPE](0.75):
            return "Stage 3/4: Moderate"
        else:
            return "Stage 4/4: Strict (MuJoCo)"
