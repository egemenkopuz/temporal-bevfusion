import argparse
import logging
import os
from glob import glob
from typing import Optional

from data_converter.create_gt_database import create_groundtruth_database


def tumtraf_intersection_data_prep(
    root_path: str,
    info_prefix: str,
    out_dir: str,
    labels_path: Optional[str] = None,
    workers: int = 4,
) -> None:
    """Prepare data related to TUMTraf-I dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the groundtruth database info.
        labels_path (str): Path of labels.
        workers (int): Number of workers to use. Default: 4
    """

    from data_converter import tumtraf_intersection_converter as tumtraf

    # get the basenames in root_path
    root_folders = [os.path.basename(x) for x in glob(os.path.join(root_path, "*"))]

    splits = []
    if "train" in root_folders:
        splits.append("training")
    else:
        raise ValueError("No training split found in root_path")

    if "val" in root_folders:
        splits.append("validation")
    if "test" in root_folders:
        splits.append("testing")

    assert len(splits) > 0, "No splits found in root_path"

    load_dir = os.path.join(root_path)
    save_dir = os.path.join(out_dir)

    os.makedirs(save_dir, exist_ok=True, mode=0o777)

    tumtraf.TUMTrafIntersectionConverter(
        splits, load_dir, save_dir, labels_path=labels_path, num_workers=workers
    ).convert(info_prefix)

    create_groundtruth_database(
        "TUMTrafIntersectionDataset",
        save_dir,
        info_prefix,
        f"{save_dir}/{info_prefix}_infos_train.pkl",
    )


def osdar23_data_prep(
    root_path: str,
    info_prefix: str,
    out_dir: str,
    labels_path: Optional[str] = None,
    use_highres: bool = False,
    workers: int = 4,
) -> None:
    """Prepare data related to OSDAR23 dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the groundtruth database info.
        labels_path (str): Path of labels.
        use_highres (bool): Whether to use high resolution images. Default: False
        workers (int): Number of workers to use. Default: 4
    """

    from data_converter import osdar23_converter as osdar23

    # get the basenames in root_path
    root_folders = [os.path.basename(x) for x in glob(os.path.join(root_path, "*"))]

    splits = []
    if "train" in root_folders:
        splits.append("training")
    else:
        raise ValueError("No training split found in root_path")

    if "val" in root_folders:
        splits.append("validation")
    if "test" in root_folders:
        splits.append("testing")

    assert len(splits) > 0, "No splits found in root_path"

    load_dir = os.path.join(root_path)
    save_dir = os.path.join(out_dir)

    os.makedirs(save_dir, exist_ok=True, mode=0o777)

    osdar23.OSDAR23Converter(
        splits, load_dir, save_dir, labels_path=labels_path, num_workers=workers
    ).convert(info_prefix, use_highres)

    create_groundtruth_database(
        "OSDAR23Dataset",
        save_dir,
        info_prefix,
        f"{save_dir}/{info_prefix}_infos_train.pkl",
    )


def nuscenes_data_prep(
    root_path,
    info_prefix,
    version,
    dataset_name,
    out_dir,
    max_sweeps=10,
    load_augmented=None,
) -> None:
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    from data_converter import nuscenes_converter as nuscenes_converter

    if load_augmented is None:
        # otherwise, infos must have been created, we just skip.
        nuscenes_converter.create_nuscenes_infos(
            root_path, info_prefix, version=version, max_sweeps=max_sweeps
        )

        # if version == "v1.0-test":
        #     info_test_path = osp.join(root_path, f"{info_prefix}_infos_test.pkl")
        #     nuscenes_converter.export_2d_annotation(root_path, info_test_path, version=version)
        #     return

        # info_train_path = osp.join(root_path, f"{info_prefix}_infos_train.pkl")
        # info_val_path = osp.join(root_path, f"{info_prefix}_infos_val.pkl")
        # nuscenes_converter.export_2d_annotation(root_path, info_train_path, version=version)
        # nuscenes_converter.export_2d_annotation(root_path, info_val_path, version=version)

    create_groundtruth_database(
        dataset_name,
        root_path,
        info_prefix,
        f"{out_dir}/{info_prefix}_infos_train.pkl",
        load_augmented=load_augmented,
    )


parser = argparse.ArgumentParser(description="Data converter arg parser")
parser.add_argument("dataset", metavar="kitti", help="name of the dataset")
parser.add_argument(
    "--root-path",
    type=str,
    required=True,
    help="specify the root path of dataset",
)
parser.add_argument(
    "--labels-path",
    type=str,
    default=None,
    required=False,
    help="specify the root path of labels",
)
parser.add_argument(
    "--version",
    type=str,
    default="v1.0",
    required=False,
    help="specify the dataset version, no need for kitti",
)
parser.add_argument(
    "--max-sweeps",
    type=int,
    default=10,
    required=False,
    help="specify sweeps of lidar per example",
)
parser.add_argument(
    "--out-dir",
    type=str,
    default="./data/kitti",
    required=False,
    help="name of info pkl",
)

parser.add_argument(
    "--use-highres", default=False, action="store_true", help="use highres images in osdar23"
)
parser.add_argument("--extra-tag", type=str, default="kitti")
parser.add_argument("--painted", default=False, action="store_true")
parser.add_argument("--virtual", default=False, action="store_true")
parser.add_argument("--workers", type=int, default=4, help="number of threads to be used")
parser.add_argument(
    "-log",
    "--loglevel",
    default="warning",
    help="Provide logging level. Example --loglevel debug, default=warning",
)

if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel.upper())
    load_augmented = None
    if args.virtual:
        if args.painted:
            load_augmented = "mvp"
        else:
            load_augmented = "pointpainting"

    if args.dataset == "nuscenes" and args.version != "v1.0-mini":
        train_version = f"{args.version}-trainval"
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name="NuScenesDataset",
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            load_augmented=load_augmented,
        )
        test_version = f"{args.version}-test"
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name="NuScenesDataset",
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            load_augmented=load_augmented,
        )
    elif args.dataset == "nuscenes" and args.version == "v1.0-mini":
        train_version = f"{args.version}"
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name="NuScenesDataset",
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            load_augmented=load_augmented,
        )
    elif str(args.dataset).lower() == "tumtraf-i":
        tumtraf_intersection_data_prep(
            root_path=args.root_path,
            info_prefix="tumtraf",
            out_dir=args.out_dir,
            labels_path=args.labels_path,
            workers=args.workers,
        )
    elif str(args.dataset).lower() == "osdar23":
        osdar23_data_prep(
            root_path=args.root_path,
            info_prefix="osdar23",
            out_dir=args.out_dir,
            labels_path=args.labels_path,
            use_highres=args.use_highres,
            workers=args.workers,
        )
