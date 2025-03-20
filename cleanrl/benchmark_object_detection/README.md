# Benchmarking Object Detection
## Usage
### Create labels.json files
Fastsam:    To generate labels.json with fastsam as object detection use `scripts/dataset_generate.sh`

OcAtari:    To convert the labels created by OcAtari in a csv file to the required format in labels.json use `cleanrl/benchmark_object_detection/transform_data.py`
    flags:  `--path`: filepath to the folder containg both test and train images and the ocatari labels.csv file
            `--ocatari_labels_path`: the name of the ocatari file
            `--train_labels_path`: the path to the labels.json file where the converted training labels will be stored
            `--test_labels_path`: the path to the labels.json file where the converted test lables will be stored

CNN:        To predict labels with the CNN and save them in labels.json use `cleanrl/benchmark_object_detection/cnn_predict_labels.py`
    flags:  `--path`: the path to the folder containing all the images that for which the labels should be predicted
            `--model_path`: the path to were the object detection model is stored
            `--output`: the name of the file in which the labels will be stored

### Visualize labels.json files
To visualize the labels in a labels.json file use `cleanrl/benchmark_object_detection/visualize_predictions.py`
    flags:  `--path`: filepath to the directory holding the image files and the labels.json file containing the labels
            `--file1`: the first labels.json file containing labels to be visualized
            `--file2`: the second labels.json file containing labels to be visualized can be None to only visualize the labels from file1
            `--output_path`: the directory in which all the visualized images will be stored
            `--frame`: specifi the frame that should be visualized can be set to -1 to visualize a random frame
            `--all_images`: can be set True to visualize all frames in the directory
            `--video`: can be set to True to output a video of all frames in the path directory
            `--video_name`: choose the name under which to save the video


### Calculate performance metrics
Run benchmark_object_detection.py with the following flags:
- `--path`: filepath to the folder containing the two labels json-files
- `--pred_labels`: filename of the json file with the predicted labels (should have same number of entries as the ground truth labels file)
- `--true_labels`: filename of the json file with the ground truth labels

Returns: intersection-over-union values, precision, recall, f1-score, coordinate-loss and shape-loss between the two sets of labels
