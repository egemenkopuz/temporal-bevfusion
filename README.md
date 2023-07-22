# multi-modal-3d-object-detection

# Docker Container Installation

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

Installing

```bash
cd mmdet3d && pip install -r requirements.txt
python setup.py develop
```

# A9 Dataset

---

## Preparing the temporal dataset

To run the converter scripts, pypcd package must be installed first. To install pypcd, run the following commands:

```bash
git clone https://github.com/DanielPollithy/pypcd
cd pypcd && pip install .
```

*If you have dataset fully ready, you can skip to the 5th step.*

1 - Merge all the files into one folder, then tokenize them by running the following command (if not tokenized already):

```bash
python tools/preprocessing/a9_tokenize.py --root-path ./data/a9_temporal_no_split --out-path ./data/a9_temporal_no_split --loglevel INFO
```

2 - Add difficulty labels to the dataset by running the following command:

```bash
python tools/preprocessing/a9_add_difficulty_labels.py --root-path ./data/a9_temporal_no_split --out-path ./data/a9_temporal_no_split --loglevel INFO
```

3 - You can then run the following command to find the optimal balanced split and split the dataset into training, validation and test sets (reduce the perm-limit if it is taking too long to run):

```bash
python tools/preprocessing/create_a9_temporal_split.py --root-path ./data/a9_temporal_no_split --out-path ./data/a9_temporal --seed 316 --segment-size 30 --perm-limit 100000 --loglevel INFO -p 6 --include-all-classes --include-all-sequences  --include-same_classes-in-difficulty --difficulty-th 0.8 --include-same_classes-in-distance --distance-th 0.8 --include-same_classes-in-num-points --num-points-th 0.5 --include-same_classes-in-occlusion --occlusion-th 0.3
```

4 - In order to make new seperate sequence segments into to their own pseudo sequences, run the following command to tokenize the dataset again:

```bash
python tools/preprocessing/a9_tokenize.py --root-path ./data/a9_temporal --out-path ./data/a9_temporal --loglevel INFO
```

5 - Finally, you can then run the following command to create the ready-to-go version of the dataset:

```bash
python tools/create_data.py a9 --root-path ./data/a9_temporal --out-dir ./data/a9_temporal_bevfusion --loglevel INFO
```

## Training

```bash
# BEV Fusion
torchpack dist-run -np 2 python tools/train.py configs/a9/det/transfusion/secfpn/camera+lidar/swint/convfuser.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth --load_from runs/lidar-only/latest.pth

# Lidar-only
torchpack dist-run -np 2 python tools/train.py configs/a9/det/transfusion/secfpn/lidar/voxelnet.yaml

# Camera-only
torchpack dist-run -np 2 python tools/train.py configs/a9/det/centerhead/lssfpn/camera/256x704/swint/default.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth

```

## Testing

BEV Fusion

```bash
# BEV Fusion
torchpack dist-run -np 2 python tools/test.py runs/bevfusion/configs.yaml runs/bevfusion/latest.pth --eval bbox --eval-options extensive_report=True

# Lidar-only
torchpack dist-run -np 2 python tools/test.py runs/lidar-only/configs.yaml runs/lidar-only/latest.pth --eval bbox --eval-options extensive_report=True

# Camera-only
torchpack dist-run -np 2 python tools/test.py runs/camera-only/configs.yaml runs/camera-only/latest.pth --eval bbox --eval-options extensive_report=True

```

## Visualization

Run the following commands for headless rendering (for open3d related scripts, do not install these inside the docker, use another form of env):

```bash
apt-get update && apt-get install libosmesa6-dev
pip install -r requirements-visual.txt
```

For built-in visualization, run the following commands:

```bash

# --max-samples 100 if you would like to visualize only a subset of the dataset
# --bbox-score 0.1 if you would like to visualize only the bounding boxes with a score higher than 0.1

# BEV Fusion
torchpack dist-run -np 1 python tools/visualize.py runs/bevfusion/configs.yaml --mode pred --split test --checkpoint runs/bevfusion/latest.pth --out-dir vis-lc --save-bboxes --save-labels

# Lidar-only
torchpack dist-run -np 1 python tools/visualize.py runs/lidar-only/configs.yaml --mode pred --split test --checkpoint runs/lidar-only/latest.pth --out-dir vis-l --save-bboxes --save-labels

# Camera-only
torchpack dist-run -np 1 python tools/visualize.py runs/camera-only/configs.yaml --mode pred --split test --checkpoint runs/camera-only/latest.pth --out-dir vis-c --save-bboxes --save-labels
```

## Benchmarking

```bash
python tools/benchmark.py runs/bevfusion/configs.yaml runs/bevfusion/latest.pth --samples 200
```
