import os
import sys
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

import argparse
import matplotlib.image
import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_process", type=bool, default=True)
    parser.add_argument("--game", type=str, default="Freeway")
    parser.add_argument("--test_frames_folder", type=str, default="test_frames/")
    parser.add_argument("--train_frames_folder", type=str, default="train_frames/")
    parser.add_argument("--resolution", type=tuple, default=(512, 512))
    path_to_cleanrl = os.path.join(os.path.dirname(__file__), '..')
    parser.add_argument("--game_folder", type=str, default=path_to_cleanrl + "/batch_training/Freeway/")
    parser.add_argument("--video_name", type=str, default="Freeway.mp4")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.batch_process:
        args.game_folder = os.path.join(SRC, "batch_training/" + args.game + '/')
        args.video_name = args.game + ".mp4"

    path = args.game_folder
    test_frames = args.test_frames_folder
    train_frames = args.train_frames_folder

    video_name = args.video_name
    resolution = args.resolution

    train_folder = os.path.join(path, train_frames)
    test_folder = os.path.join(path, test_frames)
    if not os.path.isdir(train_folder):
        os.mkdir(train_folder)

    if not os.path.isdir(test_folder):
        os.mkdir(test_folder)

    cap = cv2.VideoCapture(path + video_name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    idx = 0

    pbar = tqdm(total=frame_count)
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, args.resolution)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if idx < int(frame_count*0.8):
            cv2.imwrite(path + train_frames + "frame" + str(idx) + ".png", cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
        
        else:
            cv2.imwrite(path + test_frames + "frame" + str(idx-int(frame_count*0.8)) + ".png", cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))

        pbar.set_description(f"Processed frame {idx}")
        pbar.update(1)

        idx += 1
    
    cap.release()
