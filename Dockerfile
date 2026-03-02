# Base: official PyTorch 2.7.1 image with CUDA 11.8 + cuDNN 8
# Matches the environment from README (torch==2.7.1 --index-url .../cu118)
# Works on GPU hosts (CUDA) and CPU-only hosts (falls back automatically)
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

# Avoid interactive tzdata prompts
# Use non-interactive matplotlib backend (no display needed inside container)
ENV DEBIAN_FRONTEND=noninteractive \
    MPLBACKEND=Agg

# System utilities needed for Ray and GUI-less matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates \
      libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "simple_classification_Ray.py"]
