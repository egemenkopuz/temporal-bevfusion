import copy
import csv
import json
import logging
import os
import re
import subprocess
from argparse import ArgumentParser, Namespace
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import yaml

ERR_NAME_MAPPING = {
    "trans_err": "mATE",
    "scale_err": "mASE",
    "orient_err": "mAOE",
}

TUMTRAF_CLASSES = (
    "CAR",
    "TRAILER",
    "TRUCK",
    "VAN",
    "PEDESTRIAN",
    "BUS",
    "MOTORCYCLE",
    "BICYCLE",
    "EMERGENCY_VEHICLE",
    # "OTHER",
)

OSDAR23_CLASSES = (
    "lidar__cuboid__person",
    # "lidar__cuboid__signal",
    "lidar__cuboid__catenary_pole",
    "lidar__cuboid__signal_pole",
    # "lidar__cuboid__train",
    "lidar__cuboid__road_vehicle",
    "lidar__cuboid__buffer_stop",
    "lidar__cuboid__animal",
    # "lidar__cuboid__switch",
    # "lidar__cuboid__bicycle",
    # "lidar__cuboid__crowd",
    # "lidar__cuboid__wagons",
    # "lidar__cuboid__signal_bridge",
)

HEADERS = [
    "id",
    "eval_type",
    "sensors",
    "test_fps",
    "test_mem",
    "pcd_dim",
    "voxel_size",
    "ql",
    "qrt",
    "image_size",
    "grid_size",
    "gt_paste",
    "gridmask_p",
    "decoder_backbone",
    "decoder_neck",
    "encoder_cam_backbone",
    "encoder_cam_neck",
    "encoder_lid_backbone",
    "fuser",
    "temporal_fuser",
    "mAP",
    "mAOE",
    "mATE",
    "mASE",
]


def get_args() -> Namespace:
    """
    Parse given arguments for compile_results function.

    Returns:
        Namespace: parsed arguments
    """
    parser = ArgumentParser()

    # fmt: off
    parser.add_argument("dataset", help="dataset name", type=str)
    parser.add_argument("-c", "--checkpoints", type=str, required=True)
    parser.add_argument("-i", "--id", type=str, required=False)
    parser.add_argument("-t", "--target-path", type=str, default="results", required=False)
    parser.add_argument("--summary-foldername", type=str, default="summary", required=False)
    parser.add_argument("--images-foldername", type=str, default="images", required=False)
    parser.add_argument("--videos-foldername", type=str, default="videos", required=False)
    parser.add_argument("--configs-filename", type=str, default="configs.yaml", required=False)
    parser.add_argument("--test-results-filename", type=str, default="test_results.json", required=False)
    parser.add_argument("--benchmark-filename", type=str, default="benchmark.json", required=False)
    parser.add_argument("--test-results-csv-filename", type=str, default="test_results.csv", required=False)
    parser.add_argument("--override-testing", action="store_true", required=False)
    parser.add_argument("--override-images", action="store_true", required=False)
    parser.add_argument("--override-videos", action="store_true", required=False)
    parser.add_argument("--override-benchmark", action="store_true", required=False)
    parser.add_argument("--images-include-combined", action="store_true", required=False)
    parser.add_argument("--include-bboxes", action="store_true", required=False)
    parser.add_argument("--include-labels", action="store_true", required=False)
    parser.add_argument("--images-max-samples", type=int, default=None, required=False)
    parser.add_argument("--images-cam-bbox-score", type=float, default=None, required=False)
    parser.add_argument("--skip-test", action="store_true", required=False)
    parser.add_argument("--skip-images", action="store_true", required=False)
    parser.add_argument("--skip-videos", action="store_true", required=False)
    parser.add_argument("--skip-benchmark", action="store_true", required=False)
    parser.add_argument("-log", "--loglevel", default="info", help="provide logging level. Example --loglevel debug, default=warning",)
    # fmt: on

    return parser.parse_args()


def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_) if s]


def compile_results(
    dataset: str,
    checkpoints: str,
    id: str,
    target_path: str = "results",
    summary_foldername: str = "summary",
    images_foldername: str = "images",
    videos_foldername: str = "videos",
    configs_filename: str = "configs.yaml",
    test_results_filename: str = "test_results.json",
    benchmark_filename: str = "benchmark.json",
    test_results_csv_filename: str = "test_results.csv",
    override_testing: bool = False,
    override_images: bool = False,
    override_videos: bool = False,
    override_benchmark: bool = False,
    images_include_combined: bool = False,
    include_bboxes: bool = False,
    include_labels: bool = False,
    images_max_samples: int = None,
    images_cam_bbox_score: float = None,
    skip_test: bool = False,
    skip_images: bool = False,
    skip_videos: bool = False,
    skip_benchmark: bool = False,
    skip_compile: bool = False,
    loglevel: str = "info",
) -> None:
    """
    Compile results from given checkpoints.

    Args:
        checkpoints (str): path to checkpoints folder
        id (str): id of the experiment
        target_path (str, optional): path to store results. Defaults to "results".
        images_foldername (str, optional): filename of images folder. Defaults to "images".
        videos_foldername (str, optional): filename of videos folder. Defaults to "videos".
        configs_filename (str, optional): filename of configs yaml. Defaults to "configs.yaml".
        test_results_filename (str, optional): filename of test results json. Defaults to "test_results.json".
        benchmark_filename (str, optional): filename of benchmark json. Defaults to "benchmark.json".
        test_results_csv_filename (str, optional): filename of test results csv. Defaults to "test_results.csv".
        override_testing (bool, optional): override testing results. Defaults to False.
        override_images (bool, optional): override images. Defaults to False.
        override_videos (bool, optional): override videos. Defaults to False.
        override_benchmark (bool, optional): override benchmark. Defaults to False.
        images_include_combined (bool, optional): include combined images (gt and preds). Defaults to False.
        include_bboxes (bool, optional): save bounding boxes. Defaults to False.
        include_labels (bool, optional): save labels. Defaults to False.
        images_max_samples (int, optional): maximum number of samples to include in images. Defaults to None.
        images_cam_bbox_score (float, optional): minimum score of bounding boxes to include in images. Defaults to None.
        skip_test (bool, optional): skip testing. Defaults to False.
        skip_images (bool, optional): skip images. Defaults to False.
        skip_videos (bool, optional): skip videos. Defaults to False.
        skip_benchmark (bool, optional): skip benchmark. Defaults to False.
        skip_compile (bool, optional): skip compiling results. Defaults to False.
        loglevel (str, optional): logging level. Defaults to "info".
    """

    loglevel = loglevel.lower()

    chkpts = glob(os.path.join(checkpoints, "*"))
    results_w_category_dir = os.path.join(target_path, id)

    test_results_compiled_filename = f"{id}.csv"
    test_results_compiled_path = os.path.join(
        target_path,
        summary_foldername,
        test_results_compiled_filename,
    )

    logging.info(f"Results will be stored in {results_w_category_dir}")

    if not skip_test:
        logging.info("Testing")
        test(chkpts, results_w_category_dir, test_results_filename, override_testing, loglevel)

    if not skip_benchmark:
        logging.info("Benchmarking")
        benchmark(chkpts, results_w_category_dir, benchmark_filename, override_benchmark, loglevel)

    if not skip_compile:
        logging.info("Compiling results")
        compile(
            {os.path.basename(x): os.path.join(x, configs_filename) for x in chkpts},
            test_results_compiled_path,
            results_w_category_dir,
            test_results_filename,
            test_results_csv_filename,
            benchmark_filename,
            dataset,
        )

    if not skip_images:
        logging.info("Creating images")
        create_images(
            chkpts,
            results_w_category_dir,
            images_foldername,
            override_images,
            include_bboxes,
            include_labels,
            images_include_combined,
            images_max_samples,
            images_cam_bbox_score,
            loglevel,
        )

    if not skip_videos:
        logging.info("Creating videos")
        create_videos(
            results_w_category_dir,
            images_foldername,
            videos_foldername,
            override_videos,
            loglevel,
        )


def test(
    chkpts: List[str],
    results_w_category_dir: str,
    test_results_filename: str,
    override_testing: bool,
    loglevel: str = "info",
) -> None:
    """
    Test given checkpoints.

    Args:
        chkpts (List[str]): list of checkpoints
        results_w_category_dir (str): path to store results
        test_results_filename (str): filename of test results json
        override_testing (bool): override testing results
        loglevel (str): logging level
    """
    for x in chkpts:
        summary_path = os.path.join(
            results_w_category_dir,
            os.path.basename(x),
            test_results_filename,
        )

        if not override_testing and os.path.exists(summary_path):
            logging.info(f"Skipping, test results exist for {x}")
            continue

        cfg_path = os.path.join(x, "configs.yaml")
        pth_path = os.path.join(x, "latest.pth")

        command = f"torchpack dist-run -np 1 python tools/test.py \
            {cfg_path} {pth_path} \
            --eval bbox --eval-options extensive_report=True \
            save_summary_path={summary_path}"

        if not loglevel == "debug":
            command += " > /dev/null 2>&1"

        logging.info(f"Testing for {x}")
        os.system(command)


def create_images(
    chkpts: List[str],
    results_w_category_dir: str,
    images_foldername: str,
    override_visuals: bool,
    include_bboxes: bool,
    include_labels: bool,
    images_include_combined: bool,
    images_max_samples: Optional[int] = None,
    images_cam_bbox_score: Optional[float] = None,
    loglevel: str = "info",
) -> None:
    """
    Create images for given checkpoints.
        chkpts (List[str]): list of checkpoints
        results_w_category_dir (str): path to store results
        images_foldername (str): filename of images folder
        override_visuals (bool): override images
        include_bboxes (bool): save bounding boxes
        include_labels (bool): save labels
        images_include_combined (bool): include combined images (gt and preds)
        images_max_samples (Optional[int], optional): maximum number of samples to include in images. Defaults to None.
        images_cam_bbox_score (Optional[float], optional): minimum score of bounding boxes to include in images. Defaults to None.
        loglevel (str, optional): logging level. Defaults to "info".
    """
    for x in chkpts:
        cfg_path = os.path.join(x, "configs.yaml")
        pth_path = os.path.join(x, "latest.pth")
        id = x.split("/")[-1]

        out_dir = os.path.join(results_w_category_dir, id, images_foldername)
        bboxes_dir = os.path.join(results_w_category_dir, id, "bboxes-pred")
        labels_dir = os.path.join(results_w_category_dir, id, "labels-pred")

        if not override_visuals and os.path.exists(out_dir):
            size = subprocess.check_output(["du", "-sh", out_dir]).split()[0].decode("utf-8")
            logging.info(f"Skipping: visuals exist ({size}) for {x}")
            continue

        command = f"torchpack dist-run -np 1 python tools/visualize.py \
            {cfg_path} --checkpoint {pth_path} \
            --out-dir {out_dir} \
            --mode pred --split test"

        if include_bboxes:
            command += f" --save-bboxes  --save-bboxes-dir {bboxes_dir}"

        if include_labels:
            command += f" --save-labels --save-labels-dir {labels_dir}"

        if images_cam_bbox_score is not None and x.split("/")[-1][0] in ["C", "c"]:
            command += f" --bbox-score {images_cam_bbox_score}"

        if images_max_samples is not None:
            command += f" --max-samples {images_max_samples}"

        if images_include_combined:
            command += " --include-combined"

        if not loglevel == "debug":
            command += " > /dev/null 2>&1"

        logging.info(f"Creating images for {x}")
        os.system(command)

        size = subprocess.check_output(["du", "-sh", out_dir]).split()[0].decode("utf-8")
        logging.info(f"Directory size for {x}: {size}")


def benchmark(
    chkpts: List[str],
    results_w_category_dir: str,
    benchmark_filename: str,
    override_benchmark: bool,
    loglevel: str = "info",
):
    """
    Benchmark given checkpoints.

    Args:
        chkpts (List[str]): list of checkpoints
        results_w_category_dir (str): path to store results
        benchmark_filename (str): filename of benchmark json
        override_benchmark (bool): override benchmark
        loglevel (str): logging level
    """
    for x in chkpts:
        target_path = os.path.join(
            results_w_category_dir,
            os.path.basename(x),
            benchmark_filename,
        )

        cfg_path = os.path.join(x, "configs.yaml")
        pth_path = os.path.join(x, "latest.pth")

        if not override_benchmark and os.path.exists(target_path):
            logging.info(f"Skipping: benchmark exists for {x}")
            continue

        command = f"python tools/benchmark.py {cfg_path} {pth_path} --out {target_path}"
        if not loglevel == "debug":
            command += " > /dev/null 2>&1"

        logging.info(f"Benchmarking for {x}")
        os.system(command)


def create_videos(
    results_w_category_dir: str,
    images_foldername: str,
    videos_foldername: str,
    override_videos: bool,
    loglevel: str = "info",
) -> None:
    """
    Create videos for given checkpoints.

    Args:
        results_w_category_dir (str): path to store results
        images_foldername: str,
    videos_folder_name (str): filename of images folder
        videos_foldername (str): filename of videos folder
        override_videos (bool): override videos
        loglevel (str): logging level
    """
    for x in glob(os.path.join(results_w_category_dir, "*")):
        out_path = os.path.join(x, videos_foldername)
        if not override_videos and os.path.exists(out_path):
            size = subprocess.check_output(["du", "-sh", out_path]).split()[0].decode("utf-8")
            logging.info(f"Skipping: videos exist ({size}) for {x}")
            continue

        logging.info(f"Creating videos for {x}")
        for source_folder_dir in sorted(
            glob(os.path.join(x, images_foldername, "*")), key=natural_key
        ):
            folder_name = os.path.basename(source_folder_dir)
            if not os.path.isdir(source_folder_dir):
                continue

            target_path = os.path.join(out_path, folder_name + ".mp4")
            command = f"python tools/visualization/create_video.py -s {source_folder_dir} -t {target_path}"

            if not loglevel == "debug":
                command += " > /dev/null 2>&1"

            logging.info(f"Creating {folder_name} video for {x}")
            os.system(command)


def compile(
    config_paths: Dict[str, str],
    test_results_compiled_path: str,
    results_w_category_dir: str,
    test_results_filename: str,
    test_results_csv_filename: str,
    benchmark_filename: str,
    dataset: str,
) -> None:
    headers, classes = create_meta(dataset)
    os.makedirs(os.path.dirname(test_results_compiled_path), exist_ok=True)

    with open(test_results_compiled_path, "w") as fs:
        summary_writer = csv.DictWriter(fs, fieldnames=headers)
        summary_writer.writeheader()

        rows = []
        for _, x in enumerate(glob(os.path.join(results_w_category_dir, "*"))):
            if not os.path.isdir(x):
                continue

            logging.info(f"Compiling results for {x}")

            id = os.path.basename(x)
            path = os.path.join(x, test_results_filename)
            with open(path, "rb") as json_file:
                data = json.load(json_file)

            config_path = config_paths[id]
            model_meta = get_model_meta(config_path)

            out_csv = os.path.join(x, test_results_csv_filename)
            with open(out_csv, "w") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()

                sensors = ""
                if "lid" in id.lower():
                    sensors += "L"
                elif "cam" in id.lower():
                    sensors += "C"
                else:
                    sensors += "LC"

                # read benchmark data
                benchmark_path = os.path.join(x, benchmark_filename)
                benchmark_data = None
                with open(benchmark_path, "rb") as json_file:
                    benchmark_data = json.load(json_file)

                for eval_type, eval_data in data.items():
                    row = {
                        "id": id,
                        "sensors": sensors,
                        "test_fps": round(benchmark_data["fps"], 2),
                        "test_mem": round(benchmark_data["memory_allocated"], 2),
                        "eval_type": eval_type,
                        "mAP": eval_data["mean_ap"],
                    }

                    row.update(model_meta)

                    for x in classes:
                        if x in eval_data["class_counts"]:
                            row.update({f"num_{x}": eval_data["class_counts"][x]})
                        else:
                            row.update({f"num_{x}": ""})

                    for tp_name, tp_val in eval_data["tp_errors"].items():
                        if tp_name in ERR_NAME_MAPPING:
                            row.update({ERR_NAME_MAPPING[tp_name]: tp_val})

                    class_aps = eval_data["mean_dist_aps"]
                    class_tps = eval_data["label_tp_errors"]

                    for class_name in class_aps.keys():
                        row.update({f"{class_name}_mAP": class_aps[class_name]})
                        for tp_name, tp_val in class_tps[class_name].items():
                            if tp_name in ERR_NAME_MAPPING:
                                row.update(
                                    {
                                        f"{class_name}_{ERR_NAME_MAPPING[tp_name]}": class_tps[
                                            class_name
                                        ][tp_name]
                                    }
                                )

                    row.update({"num_gt_bboxes": eval_data["total_gt_bboxes"]})
                    rows.append(row)

                    writer.writerows([row])
        for row in rows:
            summary_writer.writerows([row])


def create_meta(dataset: str) -> Tuple[List[str], List[str]]:
    """
    Create metadata for csv file.

    Args:
        dataset (str): dataset name

    Returns:
        Tuple[List[str], List[str]]: headers and classes
    """
    headers = copy.deepcopy(HEADERS)
    if dataset.lower() == "tumtraf-i":
        classes = TUMTRAF_CLASSES
    elif dataset.lower() == "osdar23":
        classes = OSDAR23_CLASSES
    else:
        raise NotImplementedError

    for m in ["mAP", "mAOE", "mATE", "mASE"]:
        for x in classes:
            headers.append(f"{x}_{m}")

    headers.append("num_gt_bboxes")
    for x in classes:
        headers.append(f"num_{x}")

    return headers, classes


def get_model_meta(config_path: str) -> Dict[str, Any]:
    """
    Get model metadata from config file.

    Args:
        config_path (str): path to config file

    Returns:
        Dict[str, Any]: model metadata
    """
    with open(config_path, "rb") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    meta = {
        "pcd_dim": data["use_dim"],
        "voxel_size": data["voxel_size"][0],
        "ql": data["queue_length"],
        "qrt": data["queue_range_threshold"],
    }

    decoder = data["model"]["decoder"]

    meta.update(
        {
            "decoder_backbone": decoder.get("backbone", {}).get("type", None),
            "decoder_neck": decoder.get("neck", {}).get("type", None),
        }
    )

    encoder_cam = data["model"]["encoders"]["camera"]
    if encoder_cam is not None:
        meta.update(
            {
                "encoder_cam_backbone": encoder_cam.get("backbone", {}).get("type", None),
                "encoder_cam_neck": encoder_cam.get("neck", {}).get("type", None),
            }
        )
    encoder_lid = data["model"]["encoders"]["lidar"]
    if encoder_lid is not None:
        meta.update(
            {
                "encoder_lid_backbone": encoder_lid.get("backbone", {}).get("type", None),
            }
        )

    fuser = data["model"]["fuser"]
    if fuser is not None:
        meta.update({"fuser": fuser.get("type", None)})

    temporal_fuser = data["model"].get("temporal_fuser", None)
    if temporal_fuser is not None:
        meta.update({"temporal_fuser": temporal_fuser.get("type", None)})

    image_size_x, image_size_y = data["image_size"]
    image_size = f"{image_size_x}x{image_size_y}"
    gt_paste = False
    for x in data["train_pipeline"]:
        if x["type"] == "ObjectPaste":
            gt_paste = True

    gridmask_p = 0
    if "augment2d" in data and "gridmask" in data["augment2d"]:
        gridmask_p = data["augment2d"]["gridmask"]["prob"]
        meta.update({"gridmask_p": gridmask_p})

    grid_size = (
        data.get("model", {})
        .get("heads", {})
        .get("object", {})
        .get("train_cfg", {})
        .get("grid_size", None)
    )
    if grid_size is not None:
        grid_size = grid_size[0]

    meta.update(
        {
            "image_size": image_size,
            "grid_size": grid_size,
            "gt_paste": gt_paste,
        }
    )

    return meta


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=args.loglevel.upper())
    compile_results(**vars(args))
