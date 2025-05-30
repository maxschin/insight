import argparse
from sklearn.metrics import mean_squared_error 
from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd
import json
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="../sam_track/assets/PongNoFrameskip-v4/PongNoFrameskip-v4_masks_test",
        help="file path to the directory holding the json files with the bounding box data")   
    
    parser.add_argument("--pred_labels", type=str, default='labels.json',
        help="the file containing the first lables")
    
    parser.add_argument("--true_labels", type=str, default='labels.json',
        help="the file containing the second labels")
    
    parser.add_argument("--obj", type=int, default=-1,
        help="The object for witch the loss should be calculated -1 means avaerage over all objects")
    args = parser.parse_args()
    return args

def parse_json_files(path1, path2):
    try:
        with open(path1, 'r') as file:
            data1 = json.load(file)
    except FileNotFoundError:
        print(f"Error: File {args.path}/labels.json not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: Invalid JSON format.")
        sys.exit(1)
    try:
        with open(path2, 'r') as file:
            data2 = json.load(file)
    except FileNotFoundError:
        print(f"Error: File {args.path}/labels.json not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: Invalid JSON format.")
        sys.exit(1)

    if len(data1) != len(data2):
        print(f"WARNING: json files do not match in number of frames! ({len(data1)} != {len(data2)})")

    
    def iou(bb1, bb2):
        if bb1 is None or bb2 is None:
            return 0  # IoU with a dummy object is zero

        inter_x1 = max(bb1[0][0], bb2[0][0])
        inter_y1 = max(bb1[0][1], bb2[0][1])
        inter_x2 = min(bb1[1][0], bb2[1][0])
        inter_y2 = min(bb1[1][1], bb2[1][1])

        # Compute intersection
        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)
        inter_area = inter_width * inter_height

        # Compute bounding box areas
        bb1_area = (bb1[1][0] - bb1[0][0]) * (bb1[1][1] - bb1[0][1])
        bb2_area = (bb2[1][0] - bb2[0][0]) * (bb2[1][1] - bb2[0][1])

        # Compute union area
        union_area = bb1_area + bb2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    rows = []

    for frame_index in range(min(len(data1), len(data2))):
        frame_data1 = data1[frame_index]
        frame_data2 = data2[frame_index]

        keys1 = list(frame_data1.keys())
        keys2 = list(frame_data2.keys())

        n1, n2 = len(keys1), len(keys2)
        max_size = max(n1, n2)

        # Pad with dummy objects to make square
        keys1 += [None] * (max_size - n1)
        keys2 += [None] * (max_size - n2)

        bounding_boxes1 = []
        bounding_boxes2 = []

        for key1 in keys1:
            if key1 is None:
                bounding_boxes1.append(None)  # Dummy object (no real bbox)
            else:
                bb1 = tuple((frame_data1[key1]['bounding_box'][i], frame_data1[key1]['bounding_box'][i+1])
                            for i in range(0, len(frame_data1[key1]['bounding_box']), 2))
                bounding_boxes1.append(bb1)

        for key2 in keys2:
            if key2 is None:
                bounding_boxes2.append(None)  # Dummy object (no real bbox)
            else:
                bb2 = tuple((frame_data2[key2]['bounding_box'][i], frame_data2[key2]['bounding_box'][i+1])
                            for i in range(0, len(frame_data2[key2]['bounding_box']), 2))
                bounding_boxes2.append(bb2)

        # Cost matrix (negative IoU for minimization)
        cost_matrix = np.zeros((max_size, max_size), dtype=np.float64)

        for i, bb1 in enumerate(bounding_boxes1):
            for j, bb2 in enumerate(bounding_boxes2):
                cost_matrix[i, j] = -iou(bb1, bb2)  # Negative because we're minimizing

        # Solve assignment problem
        matched_rows, matched_cols = linear_sum_assignment(cost_matrix)

        # Record results (ignore dummy-dummy pairs if you want)
        for i, j in zip(matched_rows, matched_cols):
            if keys1[i] is None and keys2[j] is None:
                continue  # Skip dummy-dummy matches (they don't mean anything)

            rows.append({
                'frame': frame_index,
                'object_id1': keys1[i] if keys1[i] is not None else 'DUMMY',
                'object_id2': keys2[j] if keys2[j] is not None else 'DUMMY',
                'bounding_box1': bounding_boxes1[i],
                'bounding_box2': bounding_boxes2[j],
                'iou': -cost_matrix[i, j]  # Convert back to positive IoU
            })

    # Convert list of rows to a DataFrame in one operation
    data = pd.DataFrame(rows)
    #print(data)
    return data
    
def compute_recall_precision_f1(df, iou_threshold=0.5):
    """
    Compute Precision, Recall, and F1 Score using the DataFrame.

    Parameters:
    - df: DataFrame containing IoU values.
    - iou_threshold: IoU value above which a detection is considered a True Positive.

    Returns:
    - precision, recall, f1_score
    """

    # Count True Positives (TP) - IoU >= threshold
    TP = (df['iou'] >= iou_threshold).sum()
    #print("TP", TP)

    # Count False Positives (FP) - Detected objects that are not correct
    FP = len(df[(df['iou'] < iou_threshold) & (df['object_id1'] != 'DUMMY')])
    #df[(df['iou'] < iou_threshold) & (df['object_id1'] != 'DUMMY')].to_csv('test_fp.csv')
    #print("FP", FP)

    # Count False Negatives (FN) - Ground truth objects that were missed
    FN = len(df[(df['iou'] < iou_threshold) & (df['object_id2'] != 'DUMMY')])
    #df[(df['iou'] < iou_threshold) & (df['object_id2'] != 'DUMMY')].to_csv('test_fn.csv')
    #print("FN", FN)

    # Compute Precision (Avoid division by zero)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # Compute Recall (Avoid division by zero)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Compute F1 Score (Avoid division by zero)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

def coordinate_error(path_pred_labels, path_true_labels, loss_fn=mean_squared_error, obj=0):
    file_pred = open(args.path + '/' + path_pred_labels)
    file_ture = open(args.path + '/' + path_true_labels)

    data_pred = json.load(file_pred)
    data_true = json.load(file_ture)

    center_pred = []
    center_true = []

    frame_map = []

    for i in range(0, min(len(data_pred), len(data_true))):
        for j in range(1,6):
            if obj==-1 or j==obj:
                if str(j) in data_pred[i] and str(j) in data_true[i]:
                    center_pred.append(data_pred[i][str(j)]["coordinates"])
                    center_true.append(data_true[i][str(j)]["coordinates"])

                    frame_map.append(i)

    center_pred = np.array(center_pred)
    center_true = np.array(center_true)

    diff = center_pred - center_true
    diff = np.square(diff)
    xy_avg = [(x[0] + x[1])/2 for x in diff]

    print("coordinate max mse: ", np.max(xy_avg), " in frame: ", frame_map[np.argmax(xy_avg)])

    return loss_fn(center_pred, center_true)

def shape_error(path_pred_labels, path_true_labels, loss_fn=mean_squared_error, obj=0):
    file_pred = open(args.path + '/' + path_pred_labels)
    file_ture = open(args.path + '/' + path_true_labels)

    data_pred = json.load(file_pred)
    data_true = json.load(file_ture)

    shape_pred = []
    shape_true = []

    frame_map = []

    for i in range(0, min(len(data_pred), len(data_true))):
        for j in range(1,6):
            if obj==-1 or j==obj:
                if str(j) in data_pred[i] and str(j) in data_true[i]:
                    shape_pred.append([data_pred[i][str(j)]["bounding_box"][2] - data_pred[i][str(j)]["bounding_box"][0], 
                                       data_pred[i][str(j)]["bounding_box"][3] - data_pred[i][str(j)]["bounding_box"][1]])
                    
                    shape_true.append([data_true[i][str(j)]["bounding_box"][2] - data_true[i][str(j)]["bounding_box"][0], 
                                       data_true[i][str(j)]["bounding_box"][3] - data_true[i][str(j)]["bounding_box"][1]])

                    frame_map.append(i)

    shape_pred = np.array(shape_pred)
    shape_true = np.array(shape_true)

    diff = shape_pred - shape_true
    diff = np.square(diff)
    xy_avg = [(x[0] + x[1])/2 for x in diff]

    # print(np.average(xy_avg))
    print("shape max mse: ", np.max(xy_avg), " in frame: ", frame_map[np.argmax(xy_avg)])

    return loss_fn(shape_pred, shape_true)

if __name__ == "__main__":
    args = parse_args()
    data = parse_json_files(args.path + '/' + args.pred_labels, args.path + '/' + args.true_labels)
    
    print("iou mean:", np.mean(data['iou']))
    #data.to_csv('benchmark_test.csv')
    precision, recall, f1_score = compute_recall_precision_f1(data)
    print("precision:", precision) 
    print("recall:", recall)
    print("f1-score:", f1_score)

    coordinate_loss = coordinate_error(args.pred_labels, args.true_labels, obj=args.obj)
    shape_loss = shape_error(args.pred_labels, args.true_labels, obj=args.obj)

    print("coordinate-loss:", coordinate_loss) 
    print("shape-loss:", shape_loss)

    
