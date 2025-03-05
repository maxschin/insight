from transform_data import cnn_to_fastsam
import numpy as np
import argparse
import torch
import cv2
import os

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="models/PongNoFrameskip-v4842_grayTrue_objs256_seed1_od_ocatari_600epochen_reordered.pkl",
        help="file path to the model used to predict labels")

    parser.add_argument("--path", type=str, default="sam_track/assets/PongNoFrameskip-v4/PongNoFrameskip-v4_masks_test",
        help="file path to the directory holding the json files with the bounding box data")   
    
    parser.add_argument("--output", type=str, default="labels_cnn.json",
        help="the name of the output file")

    parser.add_argument("--resolution", type=int, default=84,
        help="the resolution of the images")

    args = parser.parse_args()
    return args


def load_images(path, resolution):
    images = []

    i = 0
    while os.path.exists(path + "/frame" + str(i) + ".png"):

        img = cv2.imread(path + "/frame" + str(i) + ".png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)
        img = np.sum(np.multiply(img, np.array([0.2125, 0.7154, 0.0721])), axis=-1)

        images.append(img/255)

        i+=1

    images = np.array(images)

    frames = []

    for i in range(0, len(images)):
        frame_stack = []

        frame_stack.append(images[i])
        frame_stack.append(images[(i+1) % len(images)])
        frame_stack.append(images[(i+2) % len(images)])
        frame_stack.append(images[(i+3) % len(images)])

        # double wrapped because for training the cnn uses batches of 4 frames each for inference the batch size is 1
        frame_stack = np.array(np.array([frame_stack]))
        frames.append(frame_stack)

    frames = np.array(frames)

    return torch.Tensor(frames)


if __name__ == "__main__":

    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)
    np.set_printoptions(precision=6)
    np.set_printoptions(suppress=True)

    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = torch.load('models/PongNoFrameskip-v4842_grayTrue_objs256_seed1_od_ocatari_600epochen_reordered.pkl', map_location=device)
    model.eval()

    images = load_images(args.path, args.resolution)

    all_coords = []
    all_shapes = []
    all_exists = []

    for img in images:
        img = img.to(device)

        pred_coord, pred_exist, pred_shape = model(img.float(), return_existence_logits=True, return_shape=True, clip_coordinates=False)

        all_coords.append(pred_coord.cpu().detach().numpy()[0])
        all_exists.append(pred_exist.cpu().detach().numpy()[0])
        all_shapes.append(pred_shape.cpu().detach().numpy()[0])
    
    cnn_to_fastsam(all_coords, all_shapes, all_exists, args.path + '/' + args.output)

