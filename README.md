# multi-modal-3d-object-detection


## Installation

Building

```bash
bash docker.sh build
```

Running

```bash
bash docker.sh run-tum # on TUM server
bash docker.sh run-setlabs # on Setlabs server
```

Accessing

```bash
bash docker.sh exec
```

Installing

```bash
cd mmdet3d && python setup.py develop
pip install -r requirements.txt
```

## A9 Create Data Script

```bash
git clone https://github.com/DanielPollithy/pypcd
cd pypcd && pip install .
```

```bash
python tools/create_data.py a9 --root-path ./data/a9 --out-dir ./data/a9_bevfusion --loglevel INFO
```

## Training

BEV Fusion

```bash
torchpack dist-run -np 2 python tools/train.py configs/a9/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth --load_from runs/lidar-only-20/epoch_20.pth
```

Camera-only

```bash
torchpack dist-run -np 2 python tools/train.py configs/a9/det/centerhead/lssfpn/camera/256x704/swint/default.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
```

Lidar-only

```bash
torchpack dist-run -np 2 python tools/train.py configs/a9/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml
```

## Visualization

Inside the docker container, run the following commands for headless rendering (for open3d related scripts):

```bash
apt-get update && apt-get install libosmesa6-dev
pip install -r requirements-visual.txt
```

```bash
torchpack dist-run -np 1 python tools/visualize.py pruns/lidar-only-20/configs.yaml --mode pred --bbox-score 0.20 --checkpoint runs/lidar-only-20/epoch_20.pth --out-dir vis-lidar-only-pred
```
