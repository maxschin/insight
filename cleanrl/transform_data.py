import csv
import re
import json
import numpy as np
import torch

# I am not sure that these are the correct dimensions of ocatari
frame_width = 160
frame_higth = 210

def __normalize_frame_width(pos):
    return pos/frame_width

def __normalize_frame_hight(pos):
    return pos/frame_higth

def __array_to_int(arr):
    out = []
    for i in range(0, len(arr)):
        out.append(int(arr[i]))
    return out
    
# Loads the data from a file and extracts the position, size and existance from it
# Must be adjusted for every new game
def __load_ocatari_data(src):
    table = csv.DictReader(open(src))

    frames = []
    
    # go through all lines in the csv file
    for line in table:

        # use regex to extract the data from the VIS column
        enemy_score = re.findall(r"EnemyScore at \(.+?\), \(.+?\)", line['VIS'])
        player_score = re.findall(r"PlayerScore at \(.+?\), \(.+?\)", line['VIS'])
        enemy = re.findall(r"Enemy at \(.+?\), \(.+?\)", line['VIS'])
        player = re.findall(r"Player at \(.+?\), \(.+?\)", line['VIS'])
        ball = re.findall(r"Ball at \(.+?\), \(.+?\)", line['VIS'])

        # objects = [enemy_score[:1], player_score[:1], enemy[:1], player[:1], ball[:1]] # for some reason there are sometimes two score objects in the ocatari data
        objects = [enemy_score, player_score, enemy, player, ball]

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
# Has to be adjusted for every new game
def __arrange_data_cnn(frames):
    all_label = []
    all_label_weight = []
    all_shape = []

    # since the lookahead is 4 the last 4 frames are not looked at.
    for i in range(0, len(frames)-4):
        coord = np.zeros(2048)
        shape = np.zeros(2048)
        exist = np.zeros(1024)
        exist[:] = 0.6 # I don't know why but in the given data, 0.6 stands for non existing objects

        lookahead = 0
        while lookahead < 4:
            j = 0
            while j < 10: # only the first 10 values (for Pong) have any values other then 0
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

# transfrom data for one fram from CNN to fastsam
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

# loads ocatari labels from cvs file (src) and transforms it to the format the cnn uses
def ocatari_to_cnn(src):
    return __arrange_data_cnn(__load_ocatari_data(src))

def ocatari_to_fastsam(src, dst_train, dst_test):

    print("transforming ocatari labels to fastsam lables")

    all_coord, all_shape, all_exist = ocatari_to_cnn(src)

    json_train, json_test = __load_cnn_data(all_coord, all_shape, all_exist)
    __store_json(json_train, dst_train)
    __store_json(json_test, dst_test)

# programm here
ocatari_to_fastsam('./sam_track/assets/Pong_input_masks/ocatari_labels_Pong_dqn.csv',
               './sam_track/assets/Pong_input_masks/Pong_input_masks_train/ocatari_labels.json',
               './sam_track/assets/Pong_input_masks/Pong_input_masks_test/ocatari_labels.json')