<h1 align="center">Temporal BEVFusion</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8-blue.svg" alt="Python 3.8"></a>
  <img src="https://img.shields.io/badge/pytorch-1.10.1-blue.svg" alt="PyTorch 1.10.1"></a>
  <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Black"></a>
</p>

Built on the repository of [BEVFusion: Multi-Task Multi-Sensor Fusion with
Unified Bird's-Eye View Representation](https://arxiv.org/abs/2205.13542).

## Table of Contents

- [Installation](#installation)
  - [Docker Container Installation](#docker-container-installation)
- [Dataset Preparation](#dataset-preparation)
  - [TUMTraf-Intersection Dataset](#tumtraf-intersection-dataset)
  - [OSDAR23 Dataset](#osdar23-dataset)
- [Training](#training)
  - [Lidar-only](#lidar-only)
  - [Camera-only](#camera-only)
  - [Multi-modal](#multi-modal)
- [Testing](#testing)
- [Visualization](#visualization)
- [Benchmarking](#benchmarking)
- [Compilation](#compilation)

## Installation


> [!WARNING]
> **You may need to add/remove/change the arguments in docker.sh for your use-case, For example, to create a custom volume.**


If you would like to use the docker container, you can build it by running the following command:

> [!INFO]
> **dev is for development and prod is for production.**

```bash
bash docker.sh build <dev/prod>
```

You can then run the container by running the following command:

```bash
bash docker.sh run <dev/prod>
```

You can access the container by running the following command:

```bash
bash docker.sh access <dev/prod>
```

If you build the dev image, you can use the following command to install the dependencies, otherwise you can skip this step:

```bash
make
```

<details>
  <summary>Click to see additional built-in commands</summary>

```bash
bash docker.sh stop <dev/prod>
```

```bash
bash docker.sh remove-container <dev/prod>
```

```bash
bash docker.sh remove-image <dev/prod>
```

```bash
bash docker.sh remove-all <dev/prod>
```

</details>


## Dataset Preparation

### TUMTraf-Intersection Dataset

<details>
  <summary>Click to expand</summary>

#### Preparing the temporal dataset

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

3 - You can then run the following command to find the optimally balanced split and split the dataset into training, validation and test sets (reduce the 'perm-limit' or increase the 'p' if it is taking too long to finish):

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

</details>

### OSDAR23 Dataset

<details>
  <summary>Click to expand</summary>

#### Preparing the temporal dataset

> [!WARNING]
> **If you have dataset fully ready, you can skip to the 3rd step.**

1 - Put all the sequences into one folder, then create seperate lidar labels folder with additional fields by running the following command:

```bash
python tools/preprocessing/osdar23_prepare.py --root-path ./data/osdar23_original --add-num-points --add-distance --loglevel INFO
```

2 - You can then run the following command to find the optimally balanced split and split the dataset into training, validation and test sets (reduce the 'perm-limit' or increase the 'p' if it is taking too long to finish):

```bash
python tools/preprocessing/osdar23_create_temporal_split.py --root-path ./data/osdar23_original --out-path ./data/osdar23 --seed 1337 --segment-size 20 --perm-limit 60000 --loglevel INFO -p 6 --include-all-classes --include-same-classes-in-distance --distance-th 0.9 --include-same-classes-in-num-points --num-points-th 0.9 --include-same-classes-in-occlusion --occlusion-th 0.85 --point-cloud-range -6.0 -156.0 -3.0 306.0 156.0 13.0 --splits train val --split-ratios 0.8 0.2 --exclude-classes lidar__cuboid__crowd lidar__cuboid__wagons lidar__cuboid__signal_bridge
```

3 - Finally, you can then run the following command to create the ready-to-go version of the dataset:

```bash
python tools/create_data.py osdar23 --root-path ./data/osdar23 --out-dir ./data/osdar23-bevfusion --use-highres --loglevel INFO
```

</details>


## Training

### Lidar-only

```bash
torchpack dist-run -np <number_of_gpus> python tools/train.py <config_path>
```

<details>
  <summary>Click to see examples</summary>

TUMTraf-Intersection

```bash
torchpack dist-run -np 1 python tools/train.py configs/tumtraf-i-baseline/det/transfusion/secfpn/lidar/voxelnet.yaml
```

OSDAR23

```bash
torchpack dist-run -np 1 python tools/train.py configs/osdar23-baseline/det/transfusion/secfpn/lidar/voxelnet.yaml
```

</details>

### Camera-only

```bash
torchpack dist-run -np <number_of_gpus> python tools/train.py <config_path> --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
```

<details>
  <summary>Click to see examples</summary>

TUMTraf-Intersection

```bash
torchpack dist-run -np 1 python tools/train.py configs/tumtraf-i-baseline/det/centerhead/lssfpn/camera/256x704/swint/default.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth

```

OSDAR23

```bash
torchpack dist-run -np 1 python tools/train.py configs/osdar23-baseline/det/centerhead/lssfpn/camera/256x704/swint/default.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth
```

</details>

### Multi-modal

```bash
torchpack dist-run -np <number_of_gpus> python tools/train.py <config_path> --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth --load_from <lidar_checkpoint_path>
```

<details>
  <summary>Click to see examples</summary>

TUMTraf-Intersection

```bash
torchpack dist-run -np 2 python tools/train.py configs/tumtraf-i-baseline/det/transfusion/secfpn/camera+lidar/256x704/swint/convfuser.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth --load_from checkpoints/lidar-run/latest.pth
```

OSDAR23

```bash
torchpack dist-run -np 2 python tools/train.py configs/osdar23-baseline/det/transfusion/secfpn/camera+lidar/256x704/swint/convfuser.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth --load_from checkpoints/lidar-run/latest.pth
```

</details>

## Testing

Following command will evaluate the model on the test set and save the results in the designated folder. In addition, if specific arguments provided, it will also save the evaluation summary and/or an extensive report of the evaluation.

```bash
torchpack dist-run -np 1 python tools/test.py <config_path> <checkpoint_path> --eval bbox
```

You can also use the following optional arguments by putting first:
- **--eval-options**  and then putting the following arguments
  - **extensive_report=True** if you would like to have an extensive report of the evaluation
  - **save_summary_path=<save_summary_path>** if you would like to save the evaluation summary

<details>
  <summary>Click to see an example</summary>

```bash
torchpack dist-run -np 1 python tools/test.py checkpoints/run/configs.yaml checkpoints/run/latest.pth --eval bbox --eval-options extensive_report=True save_summary_path=results/run/summary.json
```

</details>

## Visualization

Following command will visualize the predictions of the model on the test set and save the results in the designated folder. In addition, if specific arguments provided, it will also save the bounding boxes and/or the labels as npy files, as well as the visuals containing both predictions and ground truths.

```bash
torchpack dist-run -np 1 python tools/visualize.py <config_path> --checkpoint <checkpoint_path> --mode pred --split test --out-dir <save_path>
```

You can also use the following optional arguments:
- **--include-combined** if you would like to save the visuals containing both predictions and ground truths
- **--save-bboxes** if you would like to save the bounding boxes as npy files
- **--save-labels** if you would like to save the labels as npy files
- **--max-samples N** if you would like to visualize only a subset of the dataset, example 100
- **--bbox-score N** if you would like to visualize only the bounding boxes with a score higher than N, example: 0.1

<details>
  <summary>Click to see an example</summary>

```bash
torchpack dist-run -np 1 python tools/visualize.py checkpoints/run/configs.yaml --checkpoint checkpoints/run/latest.pth --mode pred --split test --out-dir results/run/visuals --include-combined --save-bboxes --save-labels --max-samples 100 --bbox-score 0.1
```

</details>

## Benchmarking

Following command will benchmark the model on the test set. In addition, if specific arguments provided, it will also save the benchmark results in a file.

```bash
python tools/benchmark.py <config_path> <checkpoint_path>
```

You can also use the following optional arguments:
- **--out** if you would like to save the benchmark results in a file

<details>
  <summary>Click to see an example</summary>

```bash
python tools/benchmark.py checkpoints/run/configs.yaml checkpoints/run/latest.pth --out results/run/benchmark.json
```

</details>

## Compilation

Following command will compile every other scripts such as evaluation, visualization and benchmarking scripts into one script. In addition, if specific arguments provided, it will also include the bounding boxes and/or the labels in the compilation.

```bash
python tools/compile.py <dataset> -c <checkpoints_folder_path> -i <compilation_id> -t <target_path> --include-bboxes --include-labels --images-include-combined --images-cam-bbox-score 0.15 --loglevel INFO
```

You can also use the following optional arguments:
- **--summary-foldername** if you would like to change the name of the folder containing the evaluation summary, Default: summary
- **--images-foldername** if you would like to change the name of the folder containing the images, Default: images
- **--videos-foldername** if you would like to change the name of the folder containing the videos, Default: videos
- **--override-testing** if you would like to override the testing results, Default: False
- **--override-images** if you would like to override the images, Default: False
- **--override-videos** if you would like to override the videos, Default: False
- **--override-benchmark** if you would like to override the benchmark results, Default: False
- **--images-include-combined** if you would like to include the visuals containing both predictions and ground truths, Default: False
- **--images-cam-bbox-score N** if you would like to visualize only the bounding boxes with a score higher than N, example: 0.1, Default: 0.0
- **--images-max-samples N** if you would like to visualize only a subset of the dataset, example 100, Default: None
- **--include-bboxes** if you would like to save the bounding boxes as npy files, Default: False
- **--include-labels** if you would like to save the labels as npy files, Default: False
- **--skip-test** if you would like to skip the testing, Default: False
- **--skip-images** if you would like to skip the images, Default: False
- **--skip-videos** if you would like to skip the videos, Default: False
- **--skip-benchmark** if you would like to skip the benchmarking, Default: False

<details>
  <summary>Click to see examples</summary>

TUMTraf-Intersection

```bash
python tools/compile.py tumtraf-i -c checkpoints/run -i run -t results --include-bboxes --include-labels --images-include-combined --images-cam-bbox-score 0.15 --loglevel INFO
```

OSDAR23

```bash
python tools/compile.py osdar23 -c checkpoints/run -i run -t results --include-bboxes --include-labels --images-include-combined --images-cam-bbox-score 0.15 --loglevel INFO
```

</details>