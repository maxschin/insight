#!/bin/bash

python3 benchmark_object_detection/segment_video.py --game=Pong
python3 benchmark_object_detection/transform_data.py --game=Pong
python3 train_cnn_reorder.py --game=Pong