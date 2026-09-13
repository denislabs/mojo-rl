"""Batched camera observations — a ray tracer over the batched `Data`.

`docs/DM_CONTROL_AND_CAMERA_ASSESSMENT_2026_08_24.md` §6/§9 item 3. The short
version of why this package is a tracer and not a renderer: MuJoCo Warp's
batched camera is `render.py` over `ray.py`, one kernel per (world, pixel),
and every problem the assessment listed for the rasteriser path — offscreen
target, no window, pixels-to-tensor, batching — is not a problem on this path
because there is no swapchain, no draw command and no SDL. It writes into a
`DeviceBuffer` shaped like the batched `Data` it reads.

⚠ IT DOES NOT REPLACE `mojo_rl/render/`. The SDL pipeline stays the VIEWER;
this is the OBSERVATION path. MuJoCo keeps the same split, and per single frame
the rasteriser is the cheaper of the two.

    camera.mojo     `mj_camlight`'s camera half, as tensor reads
    reference.mojo  `cam_pos0`/`cam_poscom0`/`cam_mat0`, which need FK
    visual_records  the appearance tables' record layouts
    visual.mojo     `VisualModel` — the scene AS DRAWN, beside the solver's
    appearance.mojo MuJoCo's lighting equation + texture sampling and texgen
    render.mojo     one pixel: primary ray, material, texel, lights, depth
    batch.mojo      the kernel and the host that owns its buffers
    host_render     one lane on the host over a RUNTIME-dimensioned model
"""

from .camera import (
    CameraFrame,
    camera_world_frame,
    camera_pixel_ray,
    RT_CAM_MODE_FIXED,
    RT_CAM_MODE_TRACK,
    RT_CAM_MODE_TRACKCOM,
    RT_CAM_MODE_TARGETBODY,
    RT_CAM_MODE_TARGETBODYCOM,
)
from .reference import init_camera_reference
from .appearance import sample_texture, geom_uv, shade_lights, Texel, UV
from .visual import (
    VisualModel,
    build_visual_model,
    visual_model_from_model,
)
from .visual_records import *
from .render import PixelHit, render_pixel
from .batch import BatchedCameraRenderer, RGB_CHANNELS
from .host_render import render_lane_cpu
