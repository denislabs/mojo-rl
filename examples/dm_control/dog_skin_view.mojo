"""dog, on its own, so the SKIN can be looked at without a 47-arm build.

    pixi run build-imgui                                      # ONCE
    pixi run mojo run -I . examples/dm_control/dog_skin_view.mojo
    pixi run mojo run -I . examples/dm_control/dog_skin_view.mojo zero

Same `viewer_core` as the real viewer — same sidebar, same run loop, same
window handoff — over a one-entry task table, so it costs one dog model def
instead of the whole suite.

WHAT TO LOOK AT. dog should be a DOG: a textured envelope, not the teal
capsule skeleton. Those capsules are `group="3"` and MuJoCo hides them; if you
can still see them, `geom_group` is not reaching `render_body_geoms`. If the
dog is there but grey, the skin loaded and its texture did not. If a region of
the mesh is collapsed toward the origin, a bone failed to bind — but
`tests/dm_control/test_dog_skin.mojo` gates that, so check it first.

⚠ `zero` IS THE MODE THAT SHOWS THE SKIN BEST. Under `sweep` (the default) all
38 actuators drive at once and the dog thrashes; a still, settling dog is
easier to judge a mesh by.

⚠ RUN THIS ON THE LAPTOP — it opens an SDL3 window and blocks on it.
"""

from std.random import seed
from std.sys import argv

from mojo_rl.envs.dm_control.viewer_core import (
    ViewerState, run_view, parse_drive, DRIVE_SWEEP,
)
from mojo_rl.render.imgui import imgui_shim_available
from mojo_rl.render.renderer3d import Renderer3D

from mojo_rl.envs.dm_control.dog.dog_xml import DMDogStandWalkModel
from mojo_rl.envs.dm_control.dog.dog_config import DMDogStandConfig

comptime SEED: Int = 0


def main() raises:
    seed(SEED)
    if not imgui_shim_available():
        print("Dear ImGui shim not built.  Run:  pixi run build-imgui")
        return

    var args = argv()
    var drive = parse_drive(String(args[1])) if len(args) > 1 else DRIVE_SWEEP

    var tasks = List[String]()
    tasks.append(String("dog_stand"))
    var domains = List[String]()
    domains.append(String("dog"))
    var domain_of = List[Int]()
    domain_of.append(0)

    var st = ViewerState(0, drive, 1.0, tasks^, domains^, domain_of^)
    # One task, so the loop runs once: there is nothing to switch to.
    while not st.quit:
        run_view[DMDogStandWalkModel, DMDogStandConfig](
            String("dog_stand"), st
        )

    if st.handoff:
        Renderer3D.close_handoff(st.handoff.value().copy())
        st.handoff = None
