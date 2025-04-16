# Benchmarking Object Detection
## Usage
### Create labels.json files
Fastsam:    To generate labels.json with fastsam as object detection use `scripts/dataset_generate.sh`

OcAtari:    To convert the labels created by OcAtari in a csv file to the required format in labels.json use `cleanrl/benchmark_object_detection/transform_data.py`
    flags:  `--path`: filepath to the folder containg both test and train images and the ocatari labels.csv file
            `--ocatari_labels_path`: the name of the ocatari file
            `--train_labels_path`: the path to the labels.json file where the converted training labels will be stored
            `--test_labels_path`: the path to the labels.json file where the converted test lables will be stored

CNN:        To predict labels with the CNN and save them in labels.json use `cleanrl/cnn/cnn_predict_labels.py`
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

### Models
unter https://hessenbox.tu-darmstadt.de/getlink/fiYDKePrsyGZn77vbzVZozFL/models diesem Link können bereits trainierte CNNs für die object detection gefunden werden die Modelle müssen in den models folder eingefügt werden.

### Examples

- Transform the ocatari labels into the json format for training
`python3 cnn/transform_data.py --batch_process=False --path=sam_track/assets/Pong_input/ --ocatari_labels_path=ocatari_labels_Pong_dqn.csv --train_labels_path=Pong_input_masks_train/labels_ocatari.json --test_labels_path=Pong_input_masks_test/labels_ocatari.json`

- Train the cnn on ocatari data
`python3 train/train_cnn_reorder.py --batch_process=False --train_path=sam_track/assets/Pong_input/Pong_input_masks_train --test_path=sam_track/assets/Pong_input/Pong_input_masks_test --train_labels=labels_ocatari.json --test_labels=labels_ocatari.json`

- Predict Labels with the CNN trained on ocatari
`python3 cnn/cnn_predict_labels.py --model_path=models/PongNoFrameskip-v4842_grayTrue_objs256_seed1_od_ocatari_600epochen_reordered.pkl --path=sam_track/assets/Pong_input/Pong_input_masks_test --output=labels_cnn_ocatari.json`

- Predict Labels with the CNN trained on ocatari
`python3 cnn/cnn_predict_labels.py --model_path=models/PongNoFrameskip-v4842_grayTrue_objs256_seed1_od_fastsam_600epochen_reordered.pkl --path=sam_track/assets/Pong_input/Pong_input_masks_test --output=labels_cnn_fastsam.json`

- Benchmark the cnn trained on fastsam against the ocatari ground truth
`python3 benchmark_object_detection/benchmark_object_detection.py --path=sam_track/assets/Pong_input/Pong_input_masks_test --true_labels=labels_ocatari.json --pred_labels=labels_cnn_fastsam.json`

- Benchmark the cnn trained on ocatari against the ocatari ground truth
`python3 benchmark_object_detection/benchmark_object_detection.py --path=sam_track/assets/Pong_input/Pong_input_masks_test --true_labels=labels_ocatari.json --pred_labels=labels_cnn_ocatari.json`

- Benchmark the cnn trained on fastsam aginast the fastsam ground truth
`python3 benchmark_object_detection/benchmark_object_detection.py --path=sam_track/assets/PongNoFrameskip-v4/PongNoFrameskip-v4_masks_test --true_labels=labels.json --pred_labels=labels_cnn_fastsam.json`

