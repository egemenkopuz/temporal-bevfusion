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


def main():
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config file")
    parser.add_argument("--run-dir", required=False, help="run directory")
    parser.add_argument("--auto-run-dir", required=False, help="auto-run directory")
    args, opts = parser.parse_known_args()

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

    temporal_fuser = cfg["model"].get("temporal_fuser", {}).get("type", None)
    if temporal_fuser is not None:
        models.append(temporal_fuser.lower())

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

    # gt mode stop epoch
    if cfg["gt_paste_stop_epoch"] not in [None, -1]:
        info.append(f"gtp{cfg['gt_paste_stop_epoch']}")

    # grid mask
    if cfg["augment2d"]["gridmask"]["max_epochs"] not in [None, -1]:
        info.append(f"gm{cfg['augment2d']['grid_mask']['max_epochs']}")

    out.append(modality)
    out.extend(models)
    out.extend(info)

    return "-".join(out)


if __name__ == "__main__":
    main()
