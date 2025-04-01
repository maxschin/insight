import numpy as np
import argparse
import torch
import cv2
import os

from torch.utils.data import Dataset, DataLoader
from transform_data import cnn_to_fastsam

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from agents.Normal_Cnn import OD_frames_gray2

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="../models/PongNoFrameskip-v4842_grayTrue_objs256_seed1_od_ocatari_600epochen_reordered.pkl",
        help="file path to the model used to predict labels")

    parser.add_argument("--path", type=str, default="../sam_track/assets/PongNoFrameskip-v4/PongNoFrameskip-v4_masks_test",
        help="file path to the directory holding the json files with the bounding box data")   
    
    parser.add_argument("--output", type=str, default="labels_cnn.json",
        help="the name of the output file")

    parser.add_argument("--resolution", type=int, default=84,
        help="the resolution of the images")
    
    parser.add_argument("--obj_vec_length", type=int, default=2,
        help="obj vector length")
    
    parser.add_argument("--n_objects", type=int, default=256,
        help="n_objects")

    args = parser.parse_args()
    return args


class AtariImageDataset(Dataset):
    def __init__(self, path, resolution, resize=True):
        self.images = []

        i=0
        while os.path.exists(path + "/frame" + str(i) + ".png"):

            print("state: {0:7d} entries".format(i), end='\r')

            img = cv2.imread(path + "/frame" + str(i) + ".png")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)
            img = np.sum(np.multiply(img, np.array([0.2125, 0.7154, 0.0721])), axis=-1)

            self.images.append(img/255)
            i+=1

        self.images = np.array(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        frame_stack = []

        frame_stack.append(self.images[idx])
        frame_stack.append(self.images[(idx+1) % len(self.images)])
        frame_stack.append(self.images[(idx+2) % len(self.images)])
        frame_stack.append(self.images[(idx+3) % len(self.images)])

        # double wrapped because for training the cnn uses batches of 4 frames each for inference the batch size is 1
        return torch.tensor(np.array(frame_stack))

if __name__ == "__main__":
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = torch.load(args.model_path, map_location=device)

    model.eval()

    img_data = AtariImageDataset(args.path, args.resolution)
    img_dataloader = DataLoader(img_data, batch_size=1, shuffle=False)

    all_coords = []
    all_shapes = []
    all_exists = []

    for i, img in enumerate(img_dataloader):

        print("state: {0:7.3f}%".format((i/len(img_dataloader)) * 100), "{:<10}".format(""), end='\r')
        
        img = img.to(device)

        pred_coord, pred_exist, pred_shape = model(img.float(), return_existence_logits=True, return_shape=True, clip_coordinates=False)

        all_coords.append(pred_coord.cpu().detach().numpy()[0])
        all_exists.append(pred_exist.cpu().detach().numpy()[0])
        all_shapes.append(pred_shape.cpu().detach().numpy()[0])
    
    cnn_to_fastsam(all_coords, all_shapes, all_exists, args.path + '/' + args.output)

