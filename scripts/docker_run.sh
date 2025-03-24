#!/bin/bash

# Default Dockerfile
DOCKERFILE="Dockerfile"

# Process command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --all)
            DOCKERFILE="Dockerfile.all"
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift
done

# Set the image name
IMAGE_NAME="insight"

# Define directories to be bind-mounted
MOUNT_DIRS=(
    "cleanrl/runs"
    "cleanrl/equations"
    "cleanrl/models"
    "cleanrl/ppoeql_ocatari_videos"
)

# Ensure directories exist on the host (they will replace any created in the container)
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

# Build the image using the specified Dockerfile
echo "🚀 Building the image using $DOCKERFILE..."
echo "Current directory: $(pwd)"
podman build -t "$IMAGE_NAME" -f "$DOCKERFILE" .

# Run the container with bind mounts in detached mode, including GPU flags if available
echo "🔄 Running the container with mounted directories in detached mode..."
podman run --rm -d $GPU_FLAG \
    $(for DIR in "${MOUNT_DIRS[@]}"; do echo "-v $(pwd)/$DIR:/$DIR "; done) \
    "$IMAGE_NAME"

echo "✅ Deployment finished! Check your local directories for generated files."
