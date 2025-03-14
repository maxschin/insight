# Benchmarking Object Detection
## Usage
### Create labels.json files
TODO: @Alex maybe you can write something here
### Calculate performance metrics
Run benchmark_object_detection.py with the following flags:
- `--path`: filepath to the folder containing the two labels json-files
- `--pred_labels`: filename of the json file with the predicted labels (should have same number of entries as the ground truth labels file)
- `--true_labels`: filename of the json file with the ground truth labels

Returns: intersection-over-union values, precision, recall, f1-score, coordinate-loss and shape-loss between the two sets of labels
