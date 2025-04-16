#!/bin/bash

# change to cleanrl directory
cd cleanrl

# define list of games for which pretraining should occur
games=("PongNoFrameskip-v4" "SeaquestNoFrameskip-v4" "MsPacmanNoFrameskip-v4" "SpaceInvadersNoFrameskip-v4" "FreewayNoFrameskip-v4")

echo "=== Starting batch processing for all games ==="

# generating datasets
echo "--- Generating datasets ---"
for game in "${games[@]}"; do
    echo "Generating dataset for $game..."
    python cnn/generate_dataset.py --game="$game"
done

# segmenting videos
echo "--- Segmenting videos ---"
for game in "${games[@]}"; do
    echo "Segmenting video for $game..."
    python cnn/segment_video.py --game="$game"
done

# transforming csv to json labels
echo "--- Transforming CSV to JSON labels ---"
for game in "${games[@]}"; do
    echo "Transforming data for $game..."
    python cnn/transform_data.py --game="$game"
done

# training cnn
echo "--- Training CNN ---"
for game in "${games[@]}"; do
    echo "Training CNN for $game..."
    python train/train_cnn_reorder.py --game="$game"
done

echo "=== All tasks completed successfully ==="
