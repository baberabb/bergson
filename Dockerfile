# ────────────────────────────────────────────────────────────────────────
#  Dockerfile — PyTorch 2.7.1-CUDA11.8-cuDNN9 + FAISS-GPU + TMUX/NVIM/HTOP
#               + ipykernel + ipywidgets + (dev-only) SSH
# ────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel

# == OS packages =========================================================
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        htop neovim openssh-server tmux wget \
    && rm -rf /var/lib/apt/lists/*

# Now install notebook tooling with dependencies
RUN conda install -y ipykernel ipywidgets && conda clean -afy

# Install FAISS
RUN conda install -c pytorch -c nvidia faiss-gpu=1.11.0
