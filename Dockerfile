# RunPod custom template for mojo-rl
# Base: RunPod official image with CUDA 13.0 + Ubuntu 24.04
# SSH is pre-configured by the base image (no manual setup needed)
#
# Build for RunPod (linux/amd64 from Mac M1):
#   docker buildx build --platform linux/amd64 -t <dockerhub-user>/mojo-rl-runpod:latest --push .
#
# RunPod template settings:
#   Container Image : <dockerhub-user>/mojo-rl-runpod:latest
#   Container Disk  : 20 GB
#   Volume Disk     : 50 GB  →  /workspace
#   Expose TCP Ports: 22
#   Start Command   : (leave empty — base image handles it)

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

# Use system ptxas (CUDA 13.x from base image) so Mojo bypasses its internal
# libnvptxcompiler check and compiles correctly for Blackwell (sm_120 / RTX 5090).
# pixi.toml [feature.nvidia.activation] also sets this for pixi-managed runs.
ENV MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas
