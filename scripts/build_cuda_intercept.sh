#!/bin/bash
# Build the CUDA dlsym interceptor if not already built or if source is newer
SO="$PIXI_PROJECT_ROOT/mojo_rl/cuda/libcuda_intercept.so"
SRC="$PIXI_PROJECT_ROOT/mojo_rl/cuda/cuda_intercept.c"

if [ ! -f "$SRC" ]; then
    exit 0  # source doesn't exist, skip silently
fi

if [ ! -f "$SO" ] || [ "$SRC" -nt "$SO" ]; then
    gcc -shared -fPIC -o "$SO" "$SRC" -ldl 2>/dev/null && \
        echo "[cuda-intercept] Built $SO" || true
fi
