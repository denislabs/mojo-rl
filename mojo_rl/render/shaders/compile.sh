#!/bin/bash
# Compile GLSL 450 shaders to SPIR-V bytecode using glslc (from shaderc).
# Usage: pixi run compile-shaders
# Requires: shaderc package (pixi add shaderc)
#
# Output: .spv files alongside the .glsl sources.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

GLSLC="${GLSLC:-glslc}"

# Check glslc is available
if ! command -v "$GLSLC" &> /dev/null; then
    echo "Error: glslc not found. Install with: pixi add shaderc"
    exit 1
fi

echo "Compiling GLSL shaders to SPIR-V..."

compile() {
    local src="$1"
    local stage="$2"
    local out="${src%.glsl}.spv"
    echo "  $src -> $out"
    "$GLSLC" -O --target-env=vulkan1.0 -fshader-stage="$stage" "$src" -o "$out"
}

# Vertex shaders
compile solid.vert.glsl vertex
compile ground.vert.glsl vertex
compile line.vert.glsl vertex
compile shadow.vert.glsl vertex
compile skybox.vert.glsl vertex
compile text.vert.glsl vertex

# Fragment shaders
compile solid.frag.glsl fragment
compile ground.frag.glsl fragment
compile line.frag.glsl fragment
compile shadow.frag.glsl fragment
compile reflection.frag.glsl fragment
compile skybox.frag.glsl fragment
compile text.frag.glsl fragment

echo "Done! $(ls -1 *.spv 2>/dev/null | wc -l | tr -d ' ') SPIR-V files generated."
