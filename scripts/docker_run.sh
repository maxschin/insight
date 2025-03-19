#!/bin/bash

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

# Build the image (directories inside the container will be overridden at runtime)
echo "🚀 Building the image..."
echo "Current directory: $(pwd)"
podman build -t "$IMAGE_NAME" .

# Run the container with bind mounts
echo "🔄 Running the container with mounted directories..."
podman run --rm \
    $(for DIR in "${MOUNT_DIRS[@]}"; do echo "-v $(pwd)/$DIR:/$DIR "; done) \
    "$IMAGE_NAME"

echo "✅ Deployment finished! Check your local directories for generated files."
