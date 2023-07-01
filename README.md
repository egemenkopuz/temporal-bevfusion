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

## Preparing the dataset

```bash
git clone https://github.com/DanielPollithy/pypcd
cd pypcd && pip install .
```

```bash
python tools/preprocessing/a9_tokenize.py --root-path ./data/a9 --out-dir ./data/a9_preprocessed --loglevel INFO # if not tokenized already
python tools/create_data.py a9 --root-path ./data/a9 --out-dir ./data/a9_bevfusion --loglevel INFO
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
