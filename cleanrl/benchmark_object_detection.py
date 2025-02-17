import argparse
import numpy as np
import pandas as pd
import json
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="sam_track/assets/PongNoFrameskip-v4/PongNoFrameskip-v4_masks_test",
        help="file path to the directory holding the json files with the bounding box data")   
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

    rows = []

    for frame_index in range(min(len(data1), len(data2))):
        for (key1, value1), (key2, value2) in zip(data1[frame_index].items(), data2[frame_index].items()):
            # Convert bounding box lists to tuples
            bounding_box1 = tuple((value1['bounding_box'][i], value1['bounding_box'][i+1]) for i in range(0, len(value1['bounding_box']), 2))
            bounding_box2 = tuple((value2['bounding_box'][i], value2['bounding_box'][i+1]) for i in range(0, len(value2['bounding_box']), 2))

            # Append row data as a dictionary
            rows.append({
                'frame': frame_index,
                'object_id1': key1,
                'object_id2': key2,
                'bounding_box1': bounding_box1,
                'bounding_box2': bounding_box2,
                'iou': None
            })

    # Convert list of rows to a DataFrame in one operation
    data = pd.DataFrame(rows)
    print(data)
    return data
    
def compute_iou(df):
    """
    Compute IoU for all rows in a DataFrame.
    
    Assumes `bounding_box1` and `bounding_box2` are tuples of (top-left, bottom-right) coordinates.
    """
    def iou(row):
        bb1, bb2 = row['bounding_box1'], row['bounding_box2']

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

    # Apply IoU function to each row and store in 'iou' column
    df['iou'] = df.apply(iou, axis=1)
    return df
    
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

    # Count False Positives (FP) - Detected objects that are not correct
    FP = (df['iou'] < iou_threshold).sum()

    # Count False Negatives (FN) - Ground truth objects that were missed
    FN = len(df) - TP  # Assuming each row represents a detection attempt

    # Compute Precision (Avoid division by zero)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # Compute Recall (Avoid division by zero)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Compute F1 Score (Avoid division by zero)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


if __name__ == "__main__":
    args = parse_args()
    data = parse_json_files(args.path + "/labels.json", args.path + "/labels.json")
    data = compute_iou(data)
    #print(data)
    precision, recall, f1_score = compute_recall_precision_f1(data)
    print(precision, recall, f1_score)

    