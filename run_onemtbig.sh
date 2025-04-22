#!/bin/bash

set -e  # Exit on error


MODEL_URL="https://vandanresearch.sgp1.digitaloceanspaces.com/bhashaverse-models/machine-translation/onemtbig/iiith-onemtbig.zip"
MODEL_DIR="iiith-onemtbig"
DOCKER_IMAGE="iiith-onemtbig"
REPLACEMENT_MODEL="model_base.py"
TARGET_MODEL_PATH="$MODEL_DIR/iiith-onemtbig/model_onemtbig/1/model.py"

echo "Starting the setup process..."

# Step 1: Download the model if it does not exist
if [ ! -d "$MODEL_DIR" ]; then
    echo "Downloading model from $MODEL_URL..."
    wget -O model.zip "$MODEL_URL" --progress=bar:force
    echo "Extracting model..."
    unzip -q model.zip -d "$MODEL_DIR"
    rm model.zip
    echo "Model downloaded and extracted successfully."
else
    echo "Model directory already exists, skipping download."
fi

if [ -f "$REPLACEMENT_MODEL" ]; then
    echo "Replacing $TARGET_MODEL_PATH with $REPLACEMENT_MODEL"
    cp "$REPLACEMENT_MODEL" "$TARGET_MODEL_PATH"
else
    echo "Error: $REPLACEMENT_MODEL not found in current directory!"
    exit 1
fi

# Step 2: Build the Docker image
echo "Building the Docker image: $DOCKER_IMAGE..."
sudo docker build -t $DOCKER_IMAGE .

# Step 3: Run the Triton Server Container
echo "Running Triton Server container..."
sudo docker run --gpus=all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/$MODEL_DIR/$MODEL_DIR:/models $DOCKER_IMAGE

echo "Triton Server is now running!"

