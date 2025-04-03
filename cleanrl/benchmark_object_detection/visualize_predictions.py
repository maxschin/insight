import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import json
import argparse
import random

from time import sleep

from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_process", type=bool, default=True)

    parser.add_argument("--game", type=str, default="Freeway")

    parser.add_argument("--frame", type=int, default=-1,
        help="the frame that will be visualized and safed, for -1 a random one will be chosen")

    path_to_cleanrl = os.path.join(os.path.dirname(__file__), '..')

    parser.add_argument("--path", type=str, default=path_to_cleanrl + "/batch_training/Freeway/test_frames",
        help="file path to the directory holding the json files with the bounding box data")   

    parser.add_argument("--file1", type=str, default="labels_ocatari.json",
        help="file path to the first json file with the results to be visualized")

    parser.add_argument("--file2", type=str, default=None,
        help="file path to the second json file with the results to be visualized, if it is none it will be ignored")

    parser.add_argument("--video", type=bool, default=False,
        help="if it is true the output will be a video")

    parser.add_argument("--all_images", type=bool, default=False,
        help="if it is ture all frames will be safed")
    
    parser.add_argument("--resolution", type=int, default=84,
        help="the resolution of the image")
    
    parser.add_argument("--save", type=bool, default=False)

    parser.add_argument("--output_path", type=str, default="images",
        help="the folder in which all the output is stores")

    parser.add_argument("--video_name", type=str, default="video",
        help="The name of the video file, the file will always be mp4 and the ending must not be added in the parameter")

    parser.add_argument("--fps", type=int, default=30,
        help="the frames per second shown in the video")

    parser.add_argument("--dpi", type=int, default=100,
        help="adjusts the resolution of the saved images")

    args = parser.parse_args()
    return args


def load_frame(path, frame_number, resolution):
    img = cv2.imread(path + "/frame" + str(frame_number) + ".png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)
    img = np.sum(np.multiply(img, np.array([0.2125, 0.7154, 0.0721])), axis=-1)

    return img/255

def visualize_frame(img, data1, data2, resolution, file1, file2):

    figure = plt.figure()

    plt.imshow(img, cmap="gray")

    plt.text(0, -5, file1.split('.')[0], color='r')
    # plt.text(0, -5, "labels_fastsam", color='r')


    if data2 is not None:
        plt.text(42, -5, file2.split('.')[0], color='b')

    # in pong there are only 6 objects, for general visualization the loop would have to run till 256 for performance reasons it is now only till 20
    # 20 so it can still capture some errors of fastsam which sometimes labels objects with higher number
    for i in range(1,256):

        # Object of number i might not exist
        try:
            obj1 = data1[str(i)]
        except(KeyError):
            continue

        plt.plot(obj1["coordinates"][1]*resolution, obj1["coordinates"][0]*resolution, marker='o', markersize=2, color='r')
        plt.text(obj1["coordinates"][1]*resolution + 2, obj1["coordinates"][0]*resolution - 2, str(i), fontsize=10, color='r')

        plt.plot([obj1["bounding_box"][0]*resolution, obj1["bounding_box"][2]*resolution], [obj1["bounding_box"][1]*resolution, obj1["bounding_box"][1]*resolution],
                color='r')
        plt.plot([obj1["bounding_box"][0]*resolution, obj1["bounding_box"][2]*resolution], [obj1["bounding_box"][3]*resolution, obj1["bounding_box"][3]*resolution],
                color='r')
        plt.plot([obj1["bounding_box"][0]*resolution, obj1["bounding_box"][0]*resolution], [obj1["bounding_box"][1]*resolution, obj1["bounding_box"][3]*resolution],
                color='r')
        plt.plot([obj1["bounding_box"][2]*resolution, obj1["bounding_box"][2]*resolution], [obj1["bounding_box"][1]*resolution, obj1["bounding_box"][3]*resolution],
                color='r')
        
        if data2 is None:
            continue

        # Object of number i might not exist
        try:
            obj2 = data2[str(i)]
        except(KeyError):
            continue

        plt.plot(obj2["coordinates"][1]*resolution, obj2["coordinates"][0]*resolution, marker='o', markersize=2, color='b')
        plt.text(obj2["coordinates"][1]*resolution + 2, obj2["coordinates"][0]*resolution - 2, str(i), fontsize=10, color='b')

        plt.plot([obj2["bounding_box"][0]*resolution, obj2["bounding_box"][2]*resolution], [obj2["bounding_box"][1]*resolution, obj2["bounding_box"][1]*resolution],
                color='b')
        plt.plot([obj2["bounding_box"][0]*resolution, obj2["bounding_box"][2]*resolution], [obj2["bounding_box"][3]*resolution, obj2["bounding_box"][3]*resolution],
                color='b')
        plt.plot([obj2["bounding_box"][0]*resolution, obj2["bounding_box"][0]*resolution], [obj2["bounding_box"][1]*resolution, obj2["bounding_box"][3]*resolution],
                color='b')
        plt.plot([obj2["bounding_box"][2]*resolution, obj2["bounding_box"][2]*resolution], [obj2["bounding_box"][1]*resolution, obj2["bounding_box"][3]*resolution],
                color='b')
    
    figure.canvas.draw()
    return figure

def figToNp(fig):
    return np.array(fig.canvas.renderer.buffer_rgba())


if __name__ == "__main__":
    args = parse_args()

    if args.batch_process:
        path_to_cleanrl = os.path.join(os.path.dirname(__file__), '..')
        args.path = path_to_cleanrl + "/batch_training/" + args.game + "/test_frames"

    plt.rcParams['figure.dpi'] = args.dpi

    data1 = json.load(open(args.path + '/' + args.file1))

    if args.file2 is not None:
        data2 = json.load(open(args.path + '/' + args.file2))
        if len(data1) != len(data2):
            print("the length of the two label files are not identical, this might lead to unexpected behaviour")


    if args.all_images:
        for i in range(0, len(data1)):

            print("state: {0:7.3f}%".format((i/len(data1)) * 100), end='\r')

            img = load_frame(args.path, i, args.resolution)

            if args.file2 is not None:
                fig = visualize_frame(img, data1[i], data2[i], args.resolution, args.file1, args.file2)
            else:
                fig = visualize_frame(img, data1[i], None, args.resolution, args.file1, None)

            fig.savefig(args.output_path + "/frame" + str(i) + ".png")

            plt.close(fig)

    else:
        if args.frame < 0:
            frame = random.randint(0, len(data1))
        elif args.frame >= len(data1):
            print("the frame index is out of bounds")
            exit()
        else:
            frame = args.frame

        img = load_frame(args.path, frame, args.resolution)

        if args.file2 is not None:
            fig = visualize_frame(img, data1[frame], data2[frame], args.resolution, args.file1, args.file2)
        else:
            fig = visualize_frame(img, data1[frame], None, args.resolution, args.file1, None)
            
        plt.show()
        if args.save:
            fig.savefig(args.output_path + "/frame" + str(frame) + ".png")



    if args.video:

        videodims = (int(640*(args.dpi/100)), int(480*(args.dpi/100)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')    
        video = cv2.VideoWriter(args.output_path + '/' + args.video_name + ".mp4", fourcc, args.fps, videodims)

        for i in range(0, len(data1)):

            print("state: {0:7.3f}%".format((i/len(data1)) * 100), end='\r')

            img = load_frame(args.path, i, args.resolution)
        
            if args.file2 is not None:
                fig = visualize_frame(img, data1[i], data2[i], args.resolution, args.file1, args.file2)
            else:
                fig = visualize_frame(img, data1[i], None, args.resolution, args.file1, None)

            np_img = figToNp(fig)

            plt.close(fig)

            video.write(cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR))

        video.release()