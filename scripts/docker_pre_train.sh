#!/bin/bash

# Default Dockerfile
DOCKERFILE="docker/Dockerfile.pretrain"

# Set the image name
IMAGE_NAME="insight_pretrain"

# Define directories to be bind-mounted
MOUNT_DIRS=(
    "cleanrl"
)

# Ensure directories exist on the host
for DIR in "${MOUNT_DIRS[@]}"; do
    if [ ! -d "$DIR" ]; then
        echo "⚠️ Warning: Directory $DIR does not exist locally! Creating it..."
        mkdir -p "$DIR"
    fi
done

# Check if GPUs are available
if command -v nvidia-smi &> /dev/null; then
    echo "GPUs detected. Enabling GPU support in container."
    GPU_FLAG="--device nvidia.com/gpu=all"
else
    echo "No GPUs detected."
    GPU_FLAG=""
fi

# Increase shared memory allocation
SHM_SIZE="--shm-size=8G"

# Build the image using the specified Dockerfile
echo "🚀 Building the image using $DOCKERFILE..."
echo "Current directory: $(pwd)"
podman build -t "$IMAGE_NAME" -f "$DOCKERFILE" .

# Run the container with bind mounts
echo "🔄 Running the container with mounted directories..."
podman run --rm $GPU_FLAG $SHM_SIZE \
    $(for DIR in "${MOUNT_DIRS[@]}"; do echo "-v $(pwd)/$DIR:/$DIR "; done) \
    "$IMAGE_NAME"

echo "✅ Deployment finished! Check your local directories for generated files."
