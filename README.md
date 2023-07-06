# multi-modal-3d-object-detection


# Docker Container Installation

Building

```bash
bash docker.sh build
```

Running

```bash
bash docker.sh run-local # on local machine (modify the docker.sh accoridngly)
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

If you have dataset fully ready, you can skip to the 4th step.

1 - Merge all the files into one folder, then tokenize them by running the following command (if not tokenized already):

```bash
python tools/preprocessing/a9_tokenize.py --root-path ./data/a9_temporal_no_split --out-path ./data/a9_temporal_no_split --loglevel INFO
```

2 - You can then run the following command to find the optimal balanced split and split the dataset into training, validation and test sets:

```bash
python tools/preprocessing/create_a9_temporal_split.py --root-path ./data/a9_temporal_no_split --out-path ./data/a9_temporal --segment-size 30 --perm-limit 10000 --include-all-classes --include-all-sequences --loglevel INFO
```

3 - In order to make new seperate sequence segments into to their own pseudo sequences, run the following command to tokenize the dataset again:

```bash
python tools/preprocessing/a9_tokenize.py --root-path ./data/a9_temporal --out-path ./data/a9_temporal --loglevel INFO
```

4 - Finally, you can then run the following command to create the ready-to-go version of the dataset:

```bash
python tools/create_data.py a9 --root-path ./data/a9_temporal --out-dir ./data/a9_temporal_bevfusion --loglevel INFO
```

## Training


```bash
# BEV Fusion
torchpack dist-run -np 2 python tools/train.py configs/a9/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth --load_from runs/lidar-only-20/latest.pth

# Camera-only
torchpack dist-run -np 2 python tools/train.py configs/a9/det/centerhead/lssfpn/camera/256x704/swint/default.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth

# Lidar-only
torchpack dist-run -np 2 python tools/train.py configs/a9/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml
```

## Testing

BEV Fusion

```bash
# BEV Fusion
torchpack dist-run -np 2 python tools/test.py runs/bevfusion/configs.yaml runs/bevfusion/latest.pth --eval bbox

# Lidar-only
torchpack dist-run -np 2 python tools/test.py runs/lidar-only/configs.yaml runs/lidar-only-20/latest.pth --eval bbox

# Camera-only
torchpack dist-run -np 2 python tools/test.py runs/camera-only/configs.yaml runs/camera-only/latest.pth --eval bbox

```

## Visualization

Inside the docker container, run the following commands for headless rendering (for open3d related scripts, do not install these inside the docker, use another env):

```bash
apt-get update && apt-get install libosmesa6-dev
pip install -r requirements-visual.txt
```

For built-in visualization, run the following commands:

```bash

# BEV Fusion
torchpack dist-run -np 1 python tools/visualize.py runs/bevfusion/configs.yaml --mode pred --bbox-score 0.20 --checkpoint runs/bevfusion/latest.pth --out-dir vis-bevfusion-pred

# Lidar-only
torchpack dist-run -np 1 python tools/visualize.py runs/lidar-only/configs.yaml --mode pred --bbox-score 0.20 --checkpoint runs/lidar-only/latest.pth --out-dir vis-lidar-only-pred

# Camera-only
torchpack dist-run -np 1 python tools/visualize.py runs/camera-only/configs.yaml --mode pred --bbox-score 0.20 --checkpoint runs/camera-only/latest.pth --out-dir vis-camera-only-pred
```


## Benchmarking

```bash
python tools/benchmark.py runs/bevfusion/configs.yaml runs/bevfusion/latest.pth
```
