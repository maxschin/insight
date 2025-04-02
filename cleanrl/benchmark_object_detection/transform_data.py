import argparse
import csv
import re
import os
import json
import numpy as np
import torch

OBJECT_ORDER = []

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, default="../sam_track/assets/Pong_input",
        help="the path to the folder that contains both test and train datasets created with ocatari")
    
    parser.add_argument("--ocatari_labels_path", type=str, default="ocatari_labels_Pong_dqn.csv",
        help="the path to the file that contains the ocatari labels for test and train data")
    
    parser.add_argument("--train_labels_path", type=str, default="Pong_input_masks_train/labels_ocatari.json",
        help="the path where the file with the transformed ocatari train labels should be stored")

    parser.add_argument("--test_labels_path", type=str, default="Pong_input_masks_test/labels_ocatari.json",
        help="the path where the file with the transformed ocatari test labels should be stored")

    parser.add_argument("--path_obj_order", type=str, default="object_order.json",
        help="the path where the file containig the object order should be stored")

    parser.add_argument("--ocatari_frame_width", type=int, default=160, 
        help="the width of an image produced by ocatari")
    
    parser.add_argument("--ocatari_frame_height", type=int, default=210,
        help="the height of an image produced by ocatari")

    args = parser.parse_args()
    return args

def __normalize_frame_width(pos):
    return pos/args.ocatari_frame_width

def __normalize_frame_hight(pos):
    return pos/args.ocatari_frame_height

def __array_to_int(arr):
    out = []
    for i in range(0, len(arr)):
        out.append(int(arr[i]))
    return out

def __find_object_names(line):
    names = re.findall(r"[a-zA-Z]*", line)
    
    for name in names:
        if len(name) != 0 and name != "at" and name not in OBJECT_ORDER:
            OBJECT_ORDER.append(name)

def __ocatari_regex(line):
    objects = []
    for object_name in OBJECT_ORDER:
        regex = re.escape(object_name) + r" at \(.+?\), \(.+?\)"
        objects.append(re.findall(regex, line))
    
    return objects

# the regex has to be adjusted for each game individually as the names of the enteties change
# old version
def __ocatari_regex_pong(line):
    # use regex to extract the data from the VIS column

    enemy_score = re.findall(r"EnemyScore at \(.+?\), \(.+?\)", line['VIS'])
    player_score = re.findall(r"PlayerScore at \(.+?\), \(.+?\)", line['VIS'])
    enemy = re.findall(r"Enemy at \(.+?\), \(.+?\)", line['VIS'])
    player = re.findall(r"Player at \(.+?\), \(.+?\)", line['VIS'])
    ball = re.findall(r"Ball at \(.+?\), \(.+?\)", line['VIS'])

    pong_objs = {"enemy_score": enemy_score, "player_score": player_score, "enemy": enemy, "player": player, "ball": ball}

    return [pong_objs[PONG_OBJECT_ORDER[0]], pong_objs[PONG_OBJECT_ORDER[1]], pong_objs[PONG_OBJECT_ORDER[2]], 
            pong_objs[PONG_OBJECT_ORDER[3]], pong_objs[PONG_OBJECT_ORDER[4]]]
    
# Loads the data from a file and extracts the position, size and existance from it
def __load_ocatari_data(src):
    table = csv.DictReader(open(src))
    rows = list(table)

    frames = []
    
    # go through all lines in the csv file
    for j, line in enumerate(rows):

        print("state: {0:7.3f}%".format((j/len(rows)) * 100), "{:<10}".format(""), end='\r')

        __find_object_names(line['VIS'])
        objects = __ocatari_regex(line['VIS'])
        # objects = __ocatari_regex_pong(line)

        # turn data to int and fill missing objects with placeholder data -1 and mark as non existend 0.6
        for i in range(0, len(objects)):

            if len(objects[i]) >= 1:

                all_numbers = []
                for j in range(0, len(objects[i])):
                    all_numbers.append(__array_to_int(re.findall(r"\d+", objects[i][j])))

                all_numbers = np.transpose(np.array(all_numbers))

                numbers = [0,0,0,0]
                numbers[0] = min(all_numbers[0])
                numbers[1] = min(all_numbers[1])

                width = max(all_numbers[0] + all_numbers[2])
                hight = max(all_numbers[1] + all_numbers[3])

                numbers[2] = width - numbers[0]
                numbers[3] = hight - numbers[1]

                objects[i] = [(numbers[0], numbers[1]), (numbers[2], numbers[3]), (1)]
            else:
                objects[i] = [(-1,-1), (-1,-1), (0.6)] # Object does not exist

        frames.append(objects)

    return frames

# Arranges the data in the way the CNN uses, and normalizes it
def __arrange_data_cnn(frames):
    all_label = []
    all_label_weight = []
    all_shape = []

    # since the lookahead is 4 the last 4 frames are not looked at.
    for i in range(0, len(frames)-4):

        print("state: {0:7.3f}%".format((i/(len(frames)-4)) * 100), "{:<10}".format(""), end='\r')

        coord = np.zeros(2048)
        shape = np.zeros(2048)
        exist = np.zeros(1024)
        exist[:] = 0.6 # I don't know why but in the given data, 0.6 stands for non existing objects

        lookahead = 0
        while lookahead < 4:
            j = 0
            while j < len(frames[i + lookahead])*2: # only go through the objects, that were actually detected
                coord[j+1 + lookahead * 512] = __normalize_frame_width(frames[i + lookahead][int(j/2)][0][0] + frames[i + lookahead][int(j/2)][1][0]/2)
                coord[j + lookahead*512] = __normalize_frame_hight(frames[i + lookahead][int(j/2)][0][1] + frames[i + lookahead][int(j/2)][1][1]/2)

                shape[j+1 + lookahead * 512] = __normalize_frame_width(frames[i + lookahead][int(j/2)][1][0])
                shape[j + lookahead*512] = __normalize_frame_hight(frames[i + lookahead][int(j/2)][1][1])

                exist[int(j/2) + lookahead * 256] = frames[i + lookahead][int(j/2)][2]

                j += 2
            
            lookahead += 1

        all_label.append(torch.Tensor(coord))
        all_shape.append(torch.Tensor(shape))
        all_label_weight.append(torch.Tensor(exist))

    # all_label are the coordinates, all_shape are the shapes, all_label_weights are the existenc values.
    # "all" means that it is a list of tensors for all the frames except the last 4
    return (all_label, all_shape, all_label_weight)

# transform the data for all frames to fastsam
def __load_cnn_data(all_coord, all_shape, all_exist):
    json_train = []
    json_test = []

    for j in range(0, int(len(all_coord) * 0.8)+4): # the first 80% of frames are used for training and the rest for testing (+4 for because every datapoint looks at 4 frames)

        # detach data, so it's not a tensor anymore
        coord = all_coord[j].detach().numpy()
        shape = all_shape[j].detach().numpy()
        exist = all_exist[j].detach().numpy()

        json_train.append(__arrange_data_fastsam(coord, shape, exist))

    for j in range(int(len(all_coord) * 0.8)+4, len(all_coord)): # the first 80% of frames are used for training and the rest for testing

        # detach data, so it's not a tensor anymore
        coord = all_coord[j].detach().numpy()
        shape = all_shape[j].detach().numpy()
        exist = all_exist[j].detach().numpy()

        json_test.append(__arrange_data_fastsam(coord, shape, exist))

    return json.dumps(json_train), json.dumps(json_test)

# transfrom data for one fram from CNN format to fastsam format
def __arrange_data_fastsam(coord, shape, exist):
    frame = {}
    i = 0
    while int(i/2) < 256:
        if exist[int(i/2)] > 0.61:
            y1 = coord[i] - 0.5*shape[i]
            y2 = coord[i] + 0.5*shape[i]
            x1 = coord[i+1] - 0.5*shape[i+1]
            x2 = coord[i+1] + 0.5*shape[i+1]
            obj = {"coordinates":[float(coord[(i)]), float(coord[i+1])],
                    "bounding_box":[float(x1), float(y1), float(x2), float(y2)],
                    "rgb_value":[0, 0, 0]}
            frame[str(int((i+2)/2))] = obj
        i+=2
    return frame

# store the json data in a file
def __store_json(json, src):
    f = open(src, 'w')
    f.write(json)
    f.close()

def load_object_order(src):
    if os.path.isfile(src):
        file = open(src, 'r')
        return json.loads(file.read())

# loads ocatari labels from cvs file (src) and transforms it to the format the cnn uses
def ocatari_to_cnn(src):
    return __arrange_data_cnn(__load_ocatari_data(src))


def oca_obj_to_cnn_coords(oca_obj):
    # Same as oca_obj_to_cnn_coords_old, just optimized by DeepSeek R1, which leads to dramatic performance improvements
    # oca_obj shape: (256, 4, 5, 2, 2)
    B = oca_obj.size(0)  # Batch size (256)
    F_total, O_total = 4, 5  # Frames and objects
    
    # Extract positions and sizes, compute normalized centers
    pos_x = oca_obj[..., 0, 0]  # (B, 4, 5)
    size_x = oca_obj[..., 1, 0]
    center_x = (pos_x + size_x / 2) / args.ocatari_frame_width  # Normalized x coordinates
    
    pos_y = oca_obj[..., 0, 1]
    size_y = oca_obj[..., 1, 1]
    center_y = (pos_y + size_y / 2) / args.ocatari_frame_height  # Normalized y coordinates
    
    # Precompute target indices for scattering
    frames, objs = torch.arange(F_total, device=oca_obj.device), torch.arange(O_total, device=oca_obj.device)
    F, O = torch.meshgrid(frames, objs, indexing='ij')
    indices = (F * 512 + O * 2).unsqueeze(-1)  # Base indices for x
    indices = torch.cat([indices, indices + 1], dim=-1).flatten()  # (40,) indices for x and y
    
    # Interleave x and y values and scatter into output tensor
    values = torch.stack([center_x, center_y], dim=-1).view(B, -1)  # (256, 40)
    
    output = torch.zeros(B, 2048, device=oca_obj.device)
    output.scatter_(dim=1, index=indices.expand(B, -1), src=values)
    
    return output

# loads ocatari labels from csv file (src) and transforms it to the format fastsam uses (splits the labels in test and training labels)
def ocatari_to_fastsam(src, dst_train, dst_test, path_obj_order):
    all_coord, all_shape, all_exist = ocatari_to_cnn(src)

    json_train, json_test = __load_cnn_data(all_coord, all_shape, all_exist)
    __store_json(json_train, dst_train)
    __store_json(json_test, dst_test)
    __store_json(json.dumps(OBJECT_ORDER), path_obj_order)


# takes the output of the cnn and transforms it into the format fastsam uses
def cnn_to_fastsam(coord, shape, exist, path):
    json_labels = []

    for i in range(0, len(coord)):
        json_labels.append(__arrange_data_fastsam(coord[i], shape[i], exist[i]))
    
    print(path)
    __store_json(json.dumps(json_labels), path)

if __name__ == "__main__":
    args = parse_args()

    OBJECT_ORDER = load_object_order(args.path + '/' + args.path_obj_order)

    print(OBJECT_ORDER)

    ocatari_to_fastsam(args.path + '/' + args.ocatari_labels_path,
                       args.path + '/' + args.train_labels_path,
                       args.path + '/' + args.test_labels_path,
                       args.path + '/' + args.path_obj_order)