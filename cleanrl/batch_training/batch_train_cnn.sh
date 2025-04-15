#!/bin/bash

# generating datasets
python3 batch_training/generate_dataset.py --game=Pong
python3 batch_training/generate_dataset.py --game=Seaquest
python3 batch_training/generate_dataset.py --game=MsPacman
python3 batch_training/generate_dataset.py --game=SpaceInvaders
python3 batch_training/generate_dataset.py --game=Freeway

# segmenting videos
python3 benchmark_object_detection/segment_video.py --game=Pong
python3 benchmark_object_detection/segment_video.py --game=Seaquest
python3 benchmark_object_detection/segment_video.py --game=MsPacman
python3 benchmark_object_detection/segment_video.py --game=SpaceInvaders
python3 benchmark_object_detection/segment_video.py --game=Freeway

# transforming csv to json labels
python3 benchmark_object_detection/transform_data.py --game=Pong
python3 benchmark_object_detection/transform_data.py --game=Seaquest
python3 benchmark_object_detection/transform_data.py --game=MsPacman
python3 benchmark_object_detection/transform_data.py --game=SpaceInvaders
python3 benchmark_object_detection/transform_data.py --game=Freeway

# training cnn
python3 train_cnn_reorder.py --game=Pong
python3 train_cnn_reorder.py --game=Seaquest
python3 train_cnn_reorder.py --game=MsPacman
python3 train_cnn_reorder.py --game=SpaceInvaders
python3 train_cnn_reorder.py --game=Freeway
