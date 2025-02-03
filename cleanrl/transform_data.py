import csv
import re
import numpy as np
import torch
import pandas as pd

frame_width = 160
frame_height = 210

def __normalize_frame_width(pos):
    return pos/frame_width

def __normalize_frame_height(pos):
    return pos/frame_height

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

        objects = [enemy_score, player_score, enemy, player, ball]

        # turn data to int and fill missing objects with placeholder data -1 and mark as non existend 0.6
        for i in range(0, len(objects)):
            if len(objects[i]) == 1:
                numbers = re.findall(r"\d+", objects[i][0])
                objects[i] = [(int(numbers[0]), int(numbers[1])), (int(numbers[2]), int(numbers[3])), (1)]
            else:
                objects[i] = [(-1,-1), (-1,-1), (0.6)] # Object does not exist

        print(objects)
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
                coord[j + lookahead * 512] = __normalize_frame_width(frames[i + lookahead][int(j/2)][0][1] + frames[i + lookahead][int(j/2)][1][1]/2)
                coord[j+1 + lookahead*512] = __normalize_frame_height(frames[i + lookahead][int(j/2)][0][0] + frames[i + lookahead][int(j/2)][1][0]/2)

                shape[j + lookahead * 512] = __normalize_frame_width(frames[i + lookahead][int(j/2)][1][1])
                shape[j+1 + lookahead*512] = __normalize_frame_height(frames[i + lookahead][int(j/2)][1][0])

                exist[int(j/2) + lookahead * 256] = frames[i + lookahead][int(j/2)][2]

                j += 2
            
            lookahead += 1

        all_label.append(torch.Tensor(coord))
        all_shape.append(torch.Tensor(shape))
        all_label_weight.append(torch.Tensor(exist))

    # all_label are the coordinates, all_shape are the shapes, all_label_weights are the existenc values.
    # "all" means that it is a list of tensors for all the frames except the last 4
    return (all_label, all_shape, all_label_weight)

def oca_obj_to_cnn_coords(oca_obj):
    # oca_obj has shape (256, 4, 5, 2, 2)
    # output needs to have shape (256, 2048)
    coords = []
    for step in oca_obj:
        row = np.zeros(2048)
        
        for frame_index in range(len(step)):
            for obj_index in range(len(step[frame_index])):
                # Saving x and y coodinate of center of object into row
                row[(obj_index * 2) + (512 * frame_index)] = __normalize_frame_width(step[frame_index][obj_index][0][0] + step[frame_index][obj_index][1][0]/2)
                row[(obj_index * 2 + 1) + (512 * frame_index)] = __normalize_frame_height(step[frame_index][obj_index][0][1] + step[frame_index][obj_index][1][1]/2)

        coords.append(row)

    coords = torch.Tensor(coords).to(device=oca_obj.device)
    return coords

# loads ocatari labels from cvs file (src) and transforms it to the format the cnn uses
def ocatari_to_cnn(src):
    return __arrange_data_cnn(__load_ocatari_data(src))


#(all_label, _, _) = ocatari_to_cnn("Pong_dqn_ocatari_object_data1.csv")
#all_label = pd.DataFrame(all_label)
#all_label.to_csv('all_label.csv')

