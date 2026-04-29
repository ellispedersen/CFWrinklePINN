# CFWrinklePINN training image — PyTorch 2.11.0 + CUDA 13.0, Blackwell sm_122.
#
# Base: pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime
#   PyTorch 2.11.0 stable, CUDA 13.0, cuDNN 9, conda Python 3.11
#   sm_122 support fully landed in PyTorch 2.11.0 stable (March 2026).
#   ~8 GB compressed — well within Verda 100 GB free registry tier.
#   No NGC account required.
#
# The pytorch/pytorch image uses a conda-managed Python at /opt/conda.
# python3 -m venv --system-site-packages inherits conda's site-packages
# (including torch) so run_cross_scale_level4_cuda.sh activates the venv
# without reinstalling the wheel.
#
# Build:  docker build -t vccr.io/<PROJECT>/cfwrinkle-train:pt2110 .
# Push:   docker push vccr.io/<PROJECT>/cfwrinkle-train:pt2110

FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Venv inherits torch from conda's site-packages via --system-site-packages.
# run_cross_scale_level4_cuda.sh sources VENV_PATH/bin/activate (default /workspace/venv).
RUN python3 -m venv /workspace/venv --system-site-packages && \
    /workspace/venv/bin/pip install --no-cache-dir \
        "h5py>=3.8" \
        "numpy>=1.24" \
        "scipy>=1.10" \
        "tqdm>=4.65"

# Copy repo — code is pinned to the image tag for reproducibility.
# Rebuild + push when training code changes.
COPY . /workspace/repo

ENV PYTHONPATH=/workspace/repo
ENV VENV_PATH=/workspace/venv
WORKDIR /workspace/repo
