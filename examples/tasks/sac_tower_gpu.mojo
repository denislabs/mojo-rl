"""SAC on `so101_tower` — the rig's family, the sim half of sim-to-real.

    pixi run -e nvidia mojo run -I . examples/tasks/sac_tower_gpu.mojo so101_tower_reach_clear
    pixi run -e nvidia mojo run -I . examples/tasks/sac_tower_gpu.mojo so101_tower_cube_in_bowl \\
        --steps 1000000 --envs 64
    pixi run -e apple  mojo run -I . examples/tasks/sac_tower_gpu.mojo so101_tower_reach_clear \\
        --steps 20000 --warmup 20000      # does nv = 18 launch on Metal? (nv = 24 did not)

The driver is `mojo_rl/tasks/sac_family_driver.run_sac` — read its header
before reading a curve; every measured baseline there is the TABLETOP's, and
this family has none yet. This file binds it to `So101TowerModel` +
`So101TowerConfig` (`docs/tutorials/so101_tower.md` Stage 8) and files runs
under project `so101-tower`, next to the rig's recordings and policies.

⚠ BEFORE `so101_tower_lift_brick` OR `so101_tower_cube_in_bowl`: run
`examples/tasks/task_grasp_feasibility.mojo so101_tower_lift_brick`. The
printed 25 mm cube is inside the band the sim jaw was measured to hold on the
tabletop family, but that was a different arm asset (the wrist camera mount
carries the fixed jaw here) and nothing has measured it on THIS scene.

⚠ `so101_tower_reach_clear` is the smoke task: no free slot active, the arm's
reset noise as the whole start distribution. Its rate says the stack runs,
not that anything is solved — its tabletop twin's header says why.
"""

from std.sys import argv

from mojo_rl.tasks.family_config import So101TowerConfig
from mojo_rl.tasks.so101_tower_xml import So101TowerModel
from mojo_rl.tasks.sac_family_driver import run_sac


def main() raises:
    var args = List[String]()
    for a in argv():
        args.append(String(a))
    run_sac[So101TowerModel, So101TowerConfig](
        args,
        family_path=String("mojo_rl/tasks/families/so101_tower.family"),
        project=String("so101-tower"),
        driver=String("examples/tasks/sac_tower_gpu.mojo"),
        default_task=String("so101_tower_reach_clear"),
        shape_w_goal=So101TowerConfig.SHAPE_W_GOAL,
        shape_w_reach=So101TowerConfig.SHAPE_W_REACH,
        goal_margin_default=So101TowerConfig.GOAL_MARGIN,
        reach_margin_default=So101TowerConfig.REACH_MARGIN,
    )
