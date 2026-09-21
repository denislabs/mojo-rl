"""SAC on `so101_tabletop` — the tabletop entry of the one family driver.

    pixi run -e nvidia mojo run -I . examples/tasks/sac_task_gpu.mojo so101_reach_clear
    ... --steps 200000        # lanes are comptime: N_ENVS = 32 in sac_family_driver.mojo

⚠⚠ NVIDIA ONLY: `nv = 24` exceeds Metal's per-thread stack (the driver's
header says where that was measured).

The driver — every flag, every baseline, every warning about the reward
scale, the target-tracking rate and the success criterion — is
`noeira/tasks/sac_family_driver.run_sac`, generic over the family's model
and config. This file binds it to `So101TabletopModel` +
`So101TabletopConfig` and files runs under project `so101`, as it always
did. `examples/tasks/sac_tower_gpu.mojo` is the same driver on the
`so101_tower` family. ONE instantiation per binary, on purpose: a family's
GPU kernels are minutes of compile each.
"""

from std.sys import argv

from noeira.tasks.family_config import So101TabletopConfig
from noeira.tasks.so101_tabletop_xml import So101TabletopModel
from noeira.tasks.sac_family_driver import run_sac


def main() raises:
    var args = List[String]()
    for a in argv():
        args.append(String(a))
    run_sac[So101TabletopModel, So101TabletopConfig](
        args,
        family_path=String("noeira/tasks/families/so101_tabletop.family"),
        project=String("so101"),
        driver=String("examples/tasks/sac_task_gpu.mojo"),
        default_task=String("so101_lift_brick"),
        shape_w_goal=So101TabletopConfig.SHAPE_W_GOAL,
        shape_w_reach=So101TabletopConfig.SHAPE_W_REACH,
        goal_margin_default=So101TabletopConfig.GOAL_MARGIN,
        reach_margin_default=So101TabletopConfig.REACH_MARGIN,
    )
