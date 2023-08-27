import argparse
import copy
import os

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from torchpack import distributed as dist
from torchpack.utils.config import configs
from tqdm import tqdm

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.core.utils.visualize import (
    visualize_camera_combined,
    visualize_lidar_combined,
)
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


DEFAULT_BEV_PRED_FOLDERNAME = "pred-bev"
DEFAULT_BBOXES_PRED_FOLDERNAME = "pred-bboxes"
DEFAULT_SCORES_PRED_FOLDERNAME = "pred-scores"
DEFAULT_LABELS_PRED_FOLDERNAME = "pred-labels"
DEFAULT_BBOXES_GT_FOLDERNAME = "gt-bboxes"
DEFAULT_LABELS_GT_FOLDERNAME = "gt-labels"


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj


def main() -> None:
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="viz")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-bboxes", action="store_true")
    parser.add_argument("--save-scores", action="store_true")
    parser.add_argument("--save-labels", action="store_true")
    parser.add_argument("--save-bboxes-dir", type=str, default=None)
    parser.add_argument("--save-scores-dir", type=str, default=None)
    parser.add_argument("--save-labels-dir", type=str, default=None)
    parser.add_argument("--include-combined", action="store_true")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    cfg.model.save_bev_features = {
        "out_dir": args.out_dir,
        "xlim": [cfg.point_cloud_range[d] for d in [0, 3]],
        "ylim": [cfg.point_cloud_range[d] for d in [1, 4]],
        "dataset": cfg.data.test.type,
    }

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    dataset = build_dataset(cfg.data[args.split])
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False,
    )

    # build the model and load checkpoint
    if args.mode == "pred":
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")

        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        model.eval()

    max_samples = args.max_samples if args.max_samples is not None else len(dataflow)
    for idx, data in enumerate(tqdm(dataflow, total=max_samples)):
        if idx >= max_samples:
            break

        metas = data["metas"].data[0][0]
        if isinstance(metas, list):
            metas = metas[-1]

        if cfg.data.test.type in ["TUMTrafIntersectionDataset", "OSDAR23Dataset"]:
            name = metas["lidar_path"].split("/")[-1][:-4]
        elif "token" in metas:
            name = "{}-{}".format(metas["timestamp"], metas["token"])
        else:
            name = metas["timestamp"]

        if args.mode == "pred":
            with torch.inference_mode():
                outputs = model(**data)

        if args.mode == "gt" and "gt_bboxes_3d" in data:
            bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
            labels = data["gt_labels_3d"].data[0][0].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                labels = labels[indices]
            if cfg.data.test.type not in ["TUMTrafIntersectionDataset", "OSDAR23Dataset"]:
                bboxes[..., 2] -= bboxes[..., 5] / 2
                bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
            else:
                bboxes = LiDARInstance3DBoxes(bboxes, box_dim=7)
        elif args.mode == "pred" and "boxes_3d" in outputs[0]:
            bboxes = outputs[0]["boxes_3d"].tensor.numpy()
            scores = outputs[0]["scores_3d"].numpy()
            labels = outputs[0]["labels_3d"].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            if args.bbox_score is not None:
                indices = scores >= args.bbox_score
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]
            if cfg.data.test.type not in ["TUMTrafIntersectionDataset", "OSDAR23Dataset"]:
                bboxes[..., 2] -= bboxes[..., 5] / 2
                bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
            else:
                bboxes = LiDARInstance3DBoxes(bboxes, box_dim=7)

            if args.include_combined:
                gt_bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
                gt_labels = data["gt_labels_3d"].data[0][0].numpy()

                gt_indices = gt_labels != -1

                gt_bboxes = gt_bboxes[gt_indices]
                gt_labels = gt_labels[gt_indices]

                if cfg.data.test.type not in [
                    "TUMTrafIntersectionDataset",
                    "OSDAR23Dataset",
                ]:
                    gt_bboxes[..., 2] -= gt_bboxes[..., 5] / 2
                    gt_bboxes = LiDARInstance3DBoxes(gt_bboxes, box_dim=9)
                else:
                    gt_bboxes = LiDARInstance3DBoxes(gt_bboxes, box_dim=7)
        else:
            bboxes = None
            scores = None
            labels = None

        if args.mode == "gt" and "gt_masks_bev" in data:
            masks = data["gt_masks_bev"].data[0].numpy()
            masks = masks.astype(np.bool)
        elif args.mode == "pred" and "masks_bev" in outputs[0]:
            masks = outputs[0]["masks_bev"].numpy()
            masks = masks >= args.map_score
        else:
            masks = None

        lidar = data["points"].data[0][0]
        if isinstance(lidar, list):
            lidar = lidar[-1].numpy()
        else:
            lidar = lidar.numpy()

        if "img" in data:
            for k, image_path in enumerate(metas["filename"]):
                image = mmcv.imread(image_path)

                filenames = metas["filename"]
                img_type = filenames[k].split("/")[-2]

                visualize_camera(
                    os.path.join(args.out_dir, f"pred-camera-{img_type}", f"{name}.jpg"),
                    image,
                    bboxes=bboxes,
                    labels=labels,
                    transform=metas["lidar2image"][k],
                    classes=cfg.object_classes,
                    dataset=cfg.data.train.dataset.type,
                )
                if args.include_combined:
                    visualize_camera_combined(
                        os.path.join(
                            args.out_dir, f"pred-camera-{img_type}-combined", f"{name}.jpg"
                        ),
                        image,
                        pred_bboxes=bboxes,
                        gt_bboxes=gt_bboxes,
                        transform=metas["lidar2image"][k],
                        dataset=cfg.data.train.dataset.type,
                    )

        if "points" in data:
            visualize_lidar(
                os.path.join(args.out_dir, "pred-bev", f"{name}.jpg"),
                lidar,
                bboxes=bboxes,
                labels=labels,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                classes=cfg.object_classes,
                dataset=cfg.data.train.dataset.type,
            )
            if args.include_combined:
                visualize_lidar_combined(
                    os.path.join(args.out_dir, "pred-bev-combined", f"{name}.jpg"),
                    lidar,
                    pred_bboxes=bboxes,
                    gt_bboxes=gt_bboxes,
                    xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                    ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                )

        # save scores
        if args.save_scores and scores is not None:
            if args.save_scores_dir:
                os.makedirs(args.save_scores_dir, exist_ok=True)
                np.save(os.path.join(args.save_scores_dir, f"{name}.npy"), scores)
            else:
                os.makedirs(
                    os.path.join(args.out_dir, DEFAULT_SCORES_PRED_FOLDERNAME), exist_ok=True
                )
                np.save(
                    os.path.join(args.out_dir, DEFAULT_SCORES_PRED_FOLDERNAME, f"{name}.npy"),
                    scores,
                )

        # save pred bboxes
        if args.save_bboxes and bboxes is not None:
            bboxes.tensor = bboxes.tensor.cpu()
            bboxes = bboxes.tensor.numpy()
            if args.save_bboxes_dir:
                os.makedirs(args.save_bboxes_dir, exist_ok=True)
                np.save(os.path.join(args.save_bboxes_dir, f"{name}.npy"), bboxes)
            else:
                os.makedirs(
                    os.path.join(args.out_dir, DEFAULT_BBOXES_PRED_FOLDERNAME), exist_ok=True
                )
                np.save(
                    os.path.join(args.out_dir, DEFAULT_BBOXES_PRED_FOLDERNAME, f"{name}.npy"),
                    bboxes,
                )

        # save pred labels
        if args.save_labels and labels is not None:
            if args.save_labels_dir:
                os.makedirs(args.save_labels_dir, exist_ok=True)
                np.save(os.path.join(args.save_labels_dir, f"{name}.npy"), labels)
            else:
                os.makedirs(
                    os.path.join(args.out_dir, DEFAULT_LABELS_PRED_FOLDERNAME), exist_ok=True
                )
                np.save(
                    os.path.join(args.out_dir, DEFAULT_LABELS_PRED_FOLDERNAME, f"{name}.npy"),
                    labels,
                )

        # save gt bboxes
        if args.save_bboxes:
            if args.mode == "gt" or not args.include_combined:
                gt_bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
            else:  # already filtered
                gt_bboxes.tensor = gt_bboxes.tensor.cpu()
                gt_bboxes = gt_bboxes.tensor.numpy()

            if args.save_bboxes_dir:
                os.makedirs(args.save_bboxes_dir, exist_ok=True)
                np.save(os.path.join(args.save_bboxes_dir, f"{name}.npy"), gt_bboxes)
            else:
                os.makedirs(os.path.join(args.out_dir, DEFAULT_BBOXES_GT_FOLDERNAME), exist_ok=True)
                np.save(
                    os.path.join(args.out_dir, DEFAULT_BBOXES_GT_FOLDERNAME, f"{name}.npy"),
                    gt_bboxes,
                )

        # save gt labels
        if args.save_labels:
            if args.mode == "gt" or not args.include_combined:
                gt_labels = data["gt_labels_3d"].data[0][0].numpy()
            else:  # already filtered
                gt_labels = gt_labels

            if args.save_labels_dir:
                os.makedirs(args.save_labels_dir, exist_ok=True)
                np.save(os.path.join(args.save_labels_dir, f"{name}.npy"), gt_labels)
            else:
                os.makedirs(os.path.join(args.out_dir, DEFAULT_LABELS_GT_FOLDERNAME), exist_ok=True)
                np.save(
                    os.path.join(args.out_dir, DEFAULT_LABELS_GT_FOLDERNAME, f"{name}.npy"),
                    gt_labels,
                )

        if masks is not None:
            visualize_map(
                os.path.join(args.out_dir, "map", f"{name}.png"),
                masks,
                classes=cfg.map_classes,
            )


if __name__ == "__main__":
    main()
