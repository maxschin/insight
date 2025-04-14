#!/bin/bash

# Default Dockerfile
DOCKERFILE="docker/Dockerfile"

# Flags
ALL_SET=false
ORIGINAL_SET=false
DETACHED=false

# Process command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --all)
            ALL_SET=true
            ;;
        --original)
            ORIGINAL_SET=true
            ;;
        --d)
            DETACHED=true
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift
done

# Determine Dockerfile based on flags
if $ALL_SET && $ORIGINAL_SET; then
    DOCKERFILE="docker/Dockerfile.all.original"
elif $ALL_SET; then
    DOCKERFILE="docker/Dockerfile.all"
elif $ORIGINAL_SET; then
    DOCKERFILE="docker/Dockerfile.true.original"
fi

# Set the image name
IMAGE_NAME="insight"

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

# Detached flag
if $DETACHED; then
    DETACHED_FLAG="-d"
    echo "Running container in detached mode."
else
    DETACHED_FLAG=""
    echo "Running container in foreground mode."
fi

# Increase shared memory allocation
SHM_SIZE="--shm-size=8G"

# Build the image using the specified Dockerfile
echo "🚀 Building the image using $DOCKERFILE..."
echo "Current directory: $(pwd)"
podman build -t "$IMAGE_NAME" -f "$DOCKERFILE" .

# Check shared memory available inside the container using a one-off run
echo "🔍 Checking shared memory inside the container..."
podman run --rm $GPU_FLAG $SHM_SIZE "$IMAGE_NAME" df -h /dev/shm

# Run the container with bind mounts
echo "🔄 Running the container with mounted directories..."
podman run --rm $DETACHED_FLAG $GPU_FLAG $SHM_SIZE \
    $(for DIR in "${MOUNT_DIRS[@]}"; do echo "-v $(pwd)/$DIR:/$DIR "; done) \
    "$IMAGE_NAME"

echo "✅ Deployment finished! Check your local directories for generated files."
