# Custom image for mojo-rl — compatible RunPod (CUDA 13) and Vast.ai
# SSH is pre-configured by the base image (no manual setup needed)
#
# Build for linux/amd64 from Mac M1:
#   docker buildx build --platform linux/amd64 -t <dockerhub-user>/mojo-rl-runpod:latest --push .
#
# RunPod template settings:
#   Container Image : <dockerhub-user>/mojo-rl-runpod:latest
#   Container Disk  : 20 GB
#   Volume Disk     : 50 GB  →  /workspace
#   Expose TCP Ports: 22
#   Start Command   : (leave empty — base image handles it)
#
# Vast.ai: use this image directly as custom template, no start command needed.

FROM runpod/base:1.0.3-cuda1300-ubuntu2404

# Dev tools (git is essential, the rest is convenience)
RUN apt-get update -q && \
    apt-get install -y -q --no-install-recommends \
        git \
        vim \
        htop \
        rsync \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install pixi — stored in /opt/pixi so it survives regardless of /root state
ENV PIXI_HOME=/opt/pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/opt/pixi/bin:$PATH"

# Persist pixi on PATH for interactive SSH sessions
RUN echo 'export PATH="/opt/pixi/bin:$PATH"' >> /root/.bashrc

# ---------------------------------------------------------------------------
# Pre-warm the pixi/conda package cache so that `pixi install -e nvidia`
# takes ~1 min at runtime instead of ~10 min (no downloading, only unpacking).
#
# How it works:
#   - pixi install downloads packages into the global rattler cache
#     (~/.cache/rattler/) which lives in the image layer
#   - the unpacked .pixi/envs/ is then deleted to save ~4 GB in the image
#   - at runtime `pixi install` finds all packages already cached and only
#     needs to unpack + link them → ~1 min
# ---------------------------------------------------------------------------
COPY pixi.toml pixi.lock /tmp/pixi-warmup/
WORKDIR /tmp/pixi-warmup
RUN pixi install -e nvidia && rm -rf /tmp/pixi-warmup
WORKDIR /root

# Use system ptxas (CUDA 13.x from base image) so Mojo bypasses its internal
# libnvptxcompiler check and compiles correctly for Blackwell (sm_120 / RTX 5090).
# pixi.toml [feature.nvidia.activation] also sets this for pixi-managed runs.
ENV MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas
