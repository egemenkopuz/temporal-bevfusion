import argparse
import os
import random
import time

import numpy as np
import torch
from mmcv import Config
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import convert_sync_batchnorm, get_root_logger, recursive_eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config file")
    parser.add_argument("--run-dir", required=False, help="run directory")
    parser.add_argument("--auto-run-dir", required=False, help="auto-run directory")
    args, opts = parser.parse_known_args()
    return args, opts


def train(args, opts):
    dist.init()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    if args.auto_run_dir is not None:
        args.run_dir = os.path.join(args.auto_run_dir, create_auto_dir_name(cfg))

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)

    cfg.run_dir = args.run_dir

    # dump config
    cfg.dump(os.path.join(cfg.run_dir, "configs.yaml"))

    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)

    # log some basic info
    logger.info(f"Config:\n{cfg.pretty_text}")

    # set random seeds
    if cfg.seed is not None:
        logger.info(f"Set random seed to {cfg.seed}, " f"deterministic mode: {cfg.deterministic}")
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    datasets = [build_dataset(cfg.data.train)]

    model = build_model(cfg.model)
    model.init_weights()
    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    logger.info(f"Model:\n{model}")
    train_model(
        model,
        datasets,
        cfg,
        distributed=True,
        validate=True,
        timestamp=timestamp,
    )


def create_auto_dir_name(cfg) -> str:
    """
    Create a directory name for auto-run mode.

    Args:
        cfg (Config): Config instance.

    Returns:
        str: Directory name.
    """
    out = []
    models = []
    info = []

    if cfg["model"]["encoders"]["lidar"] is None:
        modality = "c"  # means it is camera only
    elif cfg["model"]["encoders"]["camera"] is None:
        modality = "l"  # means it is lidar only
    else:
        modality = "lc"  # means it is fusion

    # fusion info
    if modality == "lc":
        # models
        fuser = cfg["model"].get("fuser", {}).get("type", None)
        if fuser is not None:
            models.append(fuser.lower())

    temporal_fuser = cfg["model"].get("temporal_fuser", None)
    if temporal_fuser is not None:
        models.append(temporal_fuser["type"].lower())

    # lidar info
    if modality.count("l") > 0:
        # models
        if cfg["model"]["encoders"]["lidar"]["backbone"]["type"] == "SparseEncoder":
            models.append("voxelnet")
        else:
            raise NotImplementedError

    # camera info
    if modality.count("c") > 0:
        # models
        if cfg["model"]["encoders"]["camera"]["backbone"]["type"] == "SwinTransformer":
            models.append("swint")
        else:
            raise NotImplementedError
        if cfg["model"]["encoders"]["camera"]["vtransform"]["type"] == "DepthLSSTransform":
            models.append("depthlss")
        elif cfg["model"]["encoders"]["camera"]["vtransform"]["type"] == "LSSTransform":
            models.append("lss")
        else:
            raise NotImplementedError
        # image size
        info.append(f"{cfg['image_size'][0]}x{cfg['image_size'][1]}")

    # grid size
    info.append(f"{cfg['grid_size'][0]}z{cfg['grid_size'][2]}")
    # voxel size
    info.append(
        f"{str(cfg['voxel_size'][0]).replace('.', 'xy')}-{str(cfg['voxel_size'][2]).replace('.', 'z')}"
    )

    # score threshold
    info.append(str(cfg["score_threshold"]).replace(".", "st"))

    # temporal queue length
    if cfg.get("queue_length", None) not in [0, None]:
        info.append(f"ql{cfg['queue_length']}")
        if cfg.get("apply_same_aug_to_seq", False):
            info.append("sameaugall")

    if cfg.get("queue_range_threshold", None) not in [0, None]:
        info.append(f"qrt{cfg['queue_range_threshold']}")

    # gt mode stop epoch
    if modality.count("l") > 0:
        if cfg["augment_gt_paste"]["max_epoch"] not in [None, -1]:
            info.append(f"gtp{cfg['augment_gt_paste']['max_epoch']}")

            if cfg["augment_gt_paste"]["apply_same_aug_to_seq"]:
                info.append("sameaug")

            rpd = cfg["augment_gt_paste"]["sampler"]["reduce_points_by_distance"]
            if rpd["prob"] > 0:
                info.append(f"rpd{str(rpd['prob']).replace('.', 'p')}")
                info.append(f"dt{rpd['distance_threshold']}")
                info.append(f"mr{str(rpd['max_ratio']).replace('.', 'p')}")

            if cfg["augment_gt_paste"]["sampler"]["cls_trans_lim"] is not None:
                info.append("trans")
            if cfg["augment_gt_paste"]["sampler"]["cls_rot_lim"] is not None:
                info.append("rot")

    # grid mask
    if modality.count("c") > 0:
        if cfg["augment2d"]["gridmask"]["max_epochs"] not in [None, -1]:
            info.append(f"gm{cfg['augment2d']['gridmask']['max_epochs']}")
            info.append(f"{str(cfg['augment2d']['gridmask']['prob']).replace('.', 'p')}")

    # augment3d
    if modality.count("l") > 0:
        info.append("aug3d")
        aug3d_scale_x = str(cfg["augment3d"]["scale"][0]).replace(".", "sx")
        info.append(aug3d_scale_x)
        aug3d_scale_y = str(cfg["augment3d"]["scale"][1]).replace(".", "sy")
        info.append(aug3d_scale_y)
        aug3d_rot_x = str(cfg["augment3d"]["rotate"][0]).replace(".", "rx")
        info.append(aug3d_rot_x)
        aug3d_rot_y = str(cfg["augment3d"]["rotate"][1]).replace(".", "ry")
        info.append(aug3d_rot_y)
        aug3d_trans = str(cfg["augment3d"]["translate"]).replace(".", "t")
        info.append(aug3d_trans)

    if cfg["deterministic"]:
        info.append("dtrmnstc")

    out.append(modality)
    out.extend(models)
    out.extend(info)

    return "-".join(out)


if __name__ == "__main__":
    args, opts = parse_args()
    train(args, opts)
