# multi-modal-3d-object-detection

## Docker Container Installation

Building

```bash
bash docker.sh build
```

Running

```bash
bash docker.sh run-local # on local machine (modify the docker.sh accordingly)
bash docker.sh run-tum # on TUM server
bash docker.sh run-setlabs # on Setlabs server
```

Accessing the terminal

```bash
bash docker.sh exec
```

Installing (inside the docker container)

```bash
cd mmdet3d && make
```

## TUMTraf-Intersection Dataset

---

### Preparing the temporal dataset

To run the converter scripts, pypcd package must be installed first. To install pypcd, run the following commands:

```bash
git clone https://github.com/DanielPollithy/pypcd
cd pypcd && pip install .
```

> [!WARNING]
> **If you have dataset fully ready, you can skip to the 5th step.**

1 - Merge all the files into one folder, then tokenize them by running the following command (if not tokenized already):

```bash
python tools/preprocessing/a9_tokenize.py --root-path ./data/tumtraf-i-no-split --out-path ./data/tumtraf-i-no-split --loglevel INFO
```

2 - Add difficulty labels to the dataset by running the following command:

```bash
python tools/preprocessing/tumtraf_add_difficulty_labels.py --root-path ./data/tumtraf-i-no-split --out-path ./data/tumtraf-i-no-split --loglevel INFO
```

3 - You can then run the following command to find the optimal balanced split and split the dataset into training, validation and test sets (reduce the perm-limit if it is taking too long to run):

```bash
python tools/preprocessing/create_a9_temporal_split.py --root-path ./data/tumtraf-i-no-split --out-path ./data/tumtraf-i --seed 42 --segment-size 30 --perm-limit 60000 --loglevel INFO -p 6 --include-all-classes --include-all-sequences  --include-same-classes-in-difficulty --difficulty-th 1.0 --include-same-classes-in-distance --distance-th 1.0 --include-same-classes-in-num-points --num-points-th 1.0 --include-same-classes-in-occlusion --occlusion-th 0.75 --point-cloud-range -25.0 -64.0 -10.0 64.0 64.0 0.0 --splits train val test --split-ratios 0.8 0.1 0.1 --exclude-classes OTHER
```

4 - In order to make new seperate sequence segments into to their own pseudo sequences, run the following command to tokenize the dataset again:

```bash
python tools/preprocessing/tumtraf_tokenize.py --root-path ./data/tumtraf-i --out-path ./data/tumtraf-i --loglevel INFO
```

5 - Finally, you can then run the following command to create the ready-to-go version of the dataset:

```bash
python tools/create_data.py tumtraf-i --root-path ./data/tumtraf-i --out-dir ./data/tumtraf-i-bevfusion --loglevel INFO
```

## Training

---

Lidar-only

```bash
torchpack dist-run -np <number_of_gpus> python tools/train.py <config_path>
```

Camera-only

```bash
torchpack dist-run -np <number_of_gpus> python tools/train.py <config_path> --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
```

Multi-modal

```bash
torchpack dist-run -np <number_of_gpus> python tools/train.py <config_path> --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth --load_from <lidar_checkpoint_path>
```

## Testing

---

```bash
torchpack dist-run -np 1 python tools/test.py <config_path> <checkpoint_path> --eval bbox
```

You can also use the following optional arguments by putting first:
- **--eval-options**  and then putting the following arguments
  - **extensive_report=True** if you would like to have an extensive report of the evaluation
  - **save_summary_path=<save_summary_path>** if you would like to save the evaluation summary

## Visualization

---

```bash
torchpack dist-run -np 1 python tools/visualize.py <config_path> --checkpoint <checkpoint_path> --mode pred --split test --out-dir <save_path>
```

You can also use the following optional arguments:
- **--include-combined** if you would like to save the visuals containing both predictions and ground truths
- **--save-bboxes** if you would like to save the bounding boxes as npy files
- **--save-labels** if you would like to save the labels as npy files
- **--max-samples N** if you would like to visualize only a subset of the dataset, example 100
- **--bbox-score N** if you would like to visualize only the bounding boxes with a score higher than N, example: 0.1

## Benchmarking

---

```bash
python tools/benchmark.py <config_path> <checkpoint_path>
```
