"""SPIR-V Shader Loading for Cross-Platform GPU 3D Renderer.

Loads pre-compiled SPIR-V bytecode from .spv files in the shaders/ directory.
These are GLSL 450 equivalents of the MSL shaders in gpu_shaders.mojo,
compiled with glslc (from shaderc). Run `pixi run compile-shaders` to rebuild.

SDL3 GPU SPIR-V binding convention:
  - Vertex:   set 0 = textures/samplers, set 1 = uniform buffers
  - Fragment: set 2 = textures/samplers, set 3 = uniform buffers
"""


def _load_spv(path: String) raises -> List[UInt8]:
    """Load a SPIR-V binary file into a byte list."""
    with open(path, "rb") as f:
        return f.read_bytes()


comptime _SHADER_DIR = "mojo_rl/render/shaders/"


def load_spirv_shaders() raises -> SPIRVShaders:
    """Load all pre-compiled SPIR-V shaders from disk."""
    return SPIRVShaders(
        solid_vert=_load_spv(_SHADER_DIR + "solid.vert.spv"),
        solid_frag=_load_spv(_SHADER_DIR + "solid.frag.spv"),
        ground_vert=_load_spv(_SHADER_DIR + "ground.vert.spv"),
        ground_frag=_load_spv(_SHADER_DIR + "ground.frag.spv"),
        line_vert=_load_spv(_SHADER_DIR + "line.vert.spv"),
        line_frag=_load_spv(_SHADER_DIR + "line.frag.spv"),
        shadow_vert=_load_spv(_SHADER_DIR + "shadow.vert.spv"),
        shadow_frag=_load_spv(_SHADER_DIR + "shadow.frag.spv"),
        reflection_frag=_load_spv(_SHADER_DIR + "reflection.frag.spv"),
        skybox_vert=_load_spv(_SHADER_DIR + "skybox.vert.spv"),
        skybox_frag=_load_spv(_SHADER_DIR + "skybox.frag.spv"),
        text_vert=_load_spv(_SHADER_DIR + "text.vert.spv"),
        text_frag=_load_spv(_SHADER_DIR + "text.frag.spv"),
    )


struct SPIRVShaders(Movable):
    """Container for all pre-compiled SPIR-V shader bytecode."""

    var solid_vert: List[UInt8]
    var solid_frag: List[UInt8]
    var ground_vert: List[UInt8]
    var ground_frag: List[UInt8]
    var line_vert: List[UInt8]
    var line_frag: List[UInt8]
    var shadow_vert: List[UInt8]
    var shadow_frag: List[UInt8]
    var reflection_frag: List[UInt8]
    var skybox_vert: List[UInt8]
    var skybox_frag: List[UInt8]
    var text_vert: List[UInt8]
    var text_frag: List[UInt8]

    def __init__(
        out self,
        var solid_vert: List[UInt8],
        var solid_frag: List[UInt8],
        var ground_vert: List[UInt8],
        var ground_frag: List[UInt8],
        var line_vert: List[UInt8],
        var line_frag: List[UInt8],
        var shadow_vert: List[UInt8],
        var shadow_frag: List[UInt8],
        var reflection_frag: List[UInt8],
        var skybox_vert: List[UInt8],
        var skybox_frag: List[UInt8],
        var text_vert: List[UInt8],
        var text_frag: List[UInt8],
    ):
        self.solid_vert = solid_vert^
        self.solid_frag = solid_frag^
        self.ground_vert = ground_vert^
        self.ground_frag = ground_frag^
        self.line_vert = line_vert^
        self.line_frag = line_frag^
        self.shadow_vert = shadow_vert^
        self.shadow_frag = shadow_frag^
        self.reflection_frag = reflection_frag^
        self.skybox_vert = skybox_vert^
        self.skybox_frag = skybox_frag^
        self.text_vert = text_vert^
        self.text_frag = text_frag^

    def __init__(out self, *, deinit take: Self):
        self.solid_vert = take.solid_vert^
        self.solid_frag = take.solid_frag^
        self.ground_vert = take.ground_vert^
        self.ground_frag = take.ground_frag^
        self.line_vert = take.line_vert^
        self.line_frag = take.line_frag^
        self.shadow_vert = take.shadow_vert^
        self.shadow_frag = take.shadow_frag^
        self.reflection_frag = take.reflection_frag^
        self.skybox_vert = take.skybox_vert^
        self.skybox_frag = take.skybox_frag^
        self.text_vert = take.text_vert^
        self.text_frag = take.text_frag^
