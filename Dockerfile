# Use NVIDIA Triton Server as the base image
FROM nvcr.io/nvidia/tritonserver:25.01-py3

# Set the working directory inside the container
WORKDIR /workspace

# Install system dependencies (including python3-venv for virtual environments)
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    bzip2 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment inside Triton’s Python backend
RUN python3 -m venv /opt/tritonserver/envs/iiithnmt-env && \
    /opt/tritonserver/envs/iiithnmt-env/bin/pip install --upgrade pip && \
    /opt/tritonserver/envs/iiithnmt-env/bin/pip install ctranslate2 numpy torch sentencepiece

# Set environment variables to use the Python virtual environment
ENV PATH="/opt/tritonserver/envs/iiithnmt-env/bin:$PATH"
ENV EXECUTION_ENV_PATH="/opt/tritonserver/envs/iiithnmt-env"

# Set Triton model repository path
ENV MODEL_REPOSITORY_PATH=/models

# Expose Triton ports
EXPOSE 8000 8001 8002

# Run Triton Inference Server with the model repository
CMD ["tritonserver", "--model-repository=/models"]

