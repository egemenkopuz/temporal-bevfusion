import copy
import json
import logging
import os
import re
import uuid
from argparse import ArgumentParser, Namespace
from glob import glob

import numpy as np
from pypcd import pypcd
from scipy.spatial.transform import Rotation as R

from mmdet3d.core.bbox import LiDARInstance3DBoxes, box_np_ops
from mmdet3d.core.bbox.structures.box_3d_mode import Box3DMode
from mmdet3d.core.points.lidar_points import LiDARPoints

LABELS_POINT_CLOUDS_FOLDERNAME = "labels_point_clouds"

DISTANCE_LEVELS = {
    "0-49": {"min": 0, "max": 50},
    "50-99": {"min": 50, "max": 100},
    "100-149": {"min": 100, "max": 150},
    "150-199": {"min": 150, "max": 200},
    "200-inf": {"min": 200, "max": 100000},
}

NUM_POINTS_LEVELS = {
    "0-199": {"min": 0, "max": 200},
    "200-499": {"min": 200, "max": 500},
    "500-999": {"min": 500, "max": 1000},
    "1000-1999": {"min": 1000, "max": 2000},
    "2000-2999": {"min": 2000, "max": 3000},
    "3000-inf": {"min": 3000, "max": 100000},
}


def get_args() -> Namespace:
    """
    Parse given arguments for add_osdar23_num_points function.

    Returns:
        Namespace: parsed arguments
    """
    parser = ArgumentParser()

    parser.add_argument("--root-path", type=str, required=True)
    parser.add_argument("--out-path", type=str, required=False)
    parser.add_argument("--add-num-points", action="store_true", required=False)
    parser.add_argument("--add-distance", action="store_true", required=False)
    parser.add_argument(
        "-log",
        "--loglevel",
        default="warning",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )

    return parser.parse_args()


def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_) if s]


def prepare_osdar23_labels(
    root_path: str,
    out_path: str = None,
    add_num_points: bool = False,
    add_distance: bool = False,
):
    sequence_paths = sorted(glob(os.path.join(root_path, "*")), key=natural_key)
    total_count = 0

    for sequence_path in sequence_paths:
        if not os.path.isdir(sequence_path):
            logging.warning(f"{sequence_path} is not a directory. Skipping...")
            continue

        if out_path is None:
            save_path = os.path.join(sequence_path, LABELS_POINT_CLOUDS_FOLDERNAME)
        else:
            save_path = os.path.join(
                out_path, os.path.basename(sequence_path), LABELS_POINT_CLOUDS_FOLDERNAME
            )

        logging.info(f"Processing {sequence_path}...")
        os.makedirs(save_path, exist_ok=True, mode=0o775)

        count = prepare(sequence_path, save_path, add_num_points, add_distance)
        logging.info(f"Number of prepared labels: {count} for {sequence_path}")
        total_count += count

    logging.info(f"Total number of prepared labels: {total_count}")


def prepare(
    sequence_path: str,
    target_path: str,
    add_num_points: bool,
    add_distance: bool,
) -> int:
    count = 0
    sequence_name = os.path.basename(sequence_path)
    main_labels_path = os.path.join(sequence_path, f"{sequence_name}_labels.json")

    if not os.path.exists(main_labels_path):
        logging.warning(f"{main_labels_path} does not exist. Skipping...")
        return

    with open(main_labels_path, "r") as f:
        main_labels = json.load(f)

    scene_token = uuid.uuid4().hex
    frame_tokens = {fi: uuid.uuid4().hex for fi in main_labels["openlabel"]["frames"].keys()}

    frame_idx_iter = 0
    for fi, fd in main_labels["openlabel"]["frames"].items():
        tmp_labels = copy.deepcopy(main_labels)
        tmp_fd = copy.deepcopy(fd)
        lidar_stream = tmp_fd["frame_properties"]["streams"]["lidar"]
        ts = lidar_stream["stream_properties"]["sync"]["timestamp"]
        label_name = f"{fi.zfill(3)}_{ts}"

        # fmt: off
        frame_token = frame_tokens[fi]
        prev_token = frame_tokens[str(int(fi) - 1)] if str(int(fi) - 1) in frame_tokens.keys() else None
        next_token = frame_tokens[str(int(fi) + 1)] if str(int(fi) + 1) in frame_tokens.keys() else None
        frame_idx = frame_idx_iter
        frame_idx_iter += 1
        # fmt: on

        tmp_fd["frame_properties"]["scene_token"] = scene_token
        tmp_fd["frame_properties"]["token"] = frame_token
        tmp_fd["frame_properties"]["prev"] = prev_token
        tmp_fd["frame_properties"]["next"] = next_token
        tmp_fd["frame_properties"]["frame_idx"] = frame_idx

        if add_num_points:
            pcd_path = os.path.join(sequence_path, "lidar", f"{label_name}.pcd")
            points = pypcd.PointCloud.from_path(pcd_path)
            np_x = np.array(points.pc_data["x"], dtype=np.float32)
            np_y = np.array(points.pc_data["y"], dtype=np.float32)
            np_z = np.array(points.pc_data["z"], dtype=np.float32)
            points = np.column_stack((np_x, np_y, np_z))
            points = LiDARPoints(points, points_dim=points.shape[-1], attribute_dims=None)
        else:
            points = None

        # fmt: off
        tmp_labels["openlabel"]["frames"] = {fi: tmp_fd}
        tmp_labels["openlabel"]["frames"][fi]["frame_properties"]["streams"] = {"lidar": lidar_stream}
        # fmt: on

        found = False
        for obj_id, obj in tmp_labels["openlabel"]["frames"][fi]["objects"].items():
            obj_data = obj["object_data"]
            other_label_types = []

            for x in obj_data.keys():
                if x != "cuboid":
                    other_label_types.append(x)
                else:
                    cuboid_data = copy.deepcopy(obj_data["cuboid"][0])
                    found = True

                    # fmt: off
                    num_points = calculate_num_points(cuboid_data, points) if add_num_points else None
                    num_points_level = determine_num_points_level(num_points) if add_num_points else None
                    distance = calculcate_distance(cuboid_data) if add_distance else None
                    distance_level = determine_distance_level(distance) if add_distance else None

                    if add_num_points:
                        if "num" not in cuboid_data["attributes"].keys():
                            cuboid_data["attributes"]["num"] = []
                        cuboid_data["attributes"]["num"].append({"name": "num_points", "val": num_points})
                        if "text" not in cuboid_data["attributes"].keys():
                            cuboid_data["attributes"]["text"] = []
                        cuboid_data["attributes"]["text"].append({"name": "num_points_level", "val": num_points_level})
                    if add_distance:
                        if "num" not in cuboid_data["attributes"].keys():
                            cuboid_data["attributes"]["num"] = []
                        cuboid_data["attributes"]["num"].append({"name": "distance", "val": distance})
                        if "text" not in cuboid_data["attributes"].keys():
                            cuboid_data["attributes"]["text"] = []
                        cuboid_data["attributes"]["text"].append({"name": "distance_level", "val": distance_level})
                    # fmt: on

                    tmp_labels["openlabel"]["frames"][fi]["objects"][obj_id]["object_data"][
                        "cuboid"
                    ] = [cuboid_data]

            for lt in other_label_types:
                del tmp_labels["openlabel"]["frames"][fi]["objects"][obj_id]["object_data"][lt]

        for obj_id, obj in copy.deepcopy(tmp_labels["openlabel"]["frames"][fi]["objects"]).items():
            # if obj is empty, remove it
            if not obj["object_data"]:
                del tmp_labels["openlabel"]["frames"][fi]["objects"][obj_id]

        if found:
            label_filename = label_name + ".json"
            with open(os.path.join(target_path, label_filename), "w") as f:
                logging.info(f"Saving {label_filename}...")
                json.dump(tmp_labels, f)
                count += 1

    return count


def get_occlusion(data: dict) -> str:
    for x in data["attributes"]["text"]:
        if x["name"] == "occlusion":
            return x["val"]


def calculate_num_points(data: dict, points) -> int:
    assert points is not None, "Points must be provided to calculate number of points inside bboxes"

    loc = np.asarray(data["val"][:3], dtype=np.float32)
    dim = np.asarray(data["val"][7:], dtype=np.float32)
    rot = np.asarray(data["val"][3:7], dtype=np.float32)
    rot_temp = R.from_quat(rot)
    rot_temp = rot_temp.as_euler("xyz", degrees=False)
    yaw = np.asarray(rot_temp[2], dtype=np.float32)
    gt_box = np.concatenate([loc, dim, -yaw], axis=None)
    gt_box = np.expand_dims(np.asarray(gt_box, dtype=np.float32), 0)

    bbox = (
        LiDARInstance3DBoxes(gt_box, box_dim=gt_box.shape[-1], origin=(0.5, 0.5, 0.5))
        .convert_to(Box3DMode.LIDAR)
        .tensor.numpy()
    )

    masks = box_np_ops.points_in_rbbox(points.coord.numpy(), bbox)
    points = points[masks.any(-1)]

    return points.shape[0]


def calculcate_distance(data: dict) -> float:
    return np.sqrt(np.sum(np.array(data["val"][:2]) ** 2))


def determine_distance_level(distance: float) -> str:
    assert distance is not None, "Distance must be provided to determine the level"

    for level, level_data in DISTANCE_LEVELS.items():
        if level_data["min"] <= distance < level_data["max"]:
            return level

    raise ValueError(f"Distance {distance} is not in any level")


def determine_num_points_level(num_points: int) -> str:
    assert num_points is not None, "Number of points must be provided to determine the level"

    for level, level_data in NUM_POINTS_LEVELS.items():
        if level_data["min"] <= num_points < level_data["max"]:
            return level

    raise ValueError(f"Number of points {num_points} is not in any level")


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=args.loglevel.upper())
    prepare_osdar23_labels(
        root_path=args.root_path,
        out_path=args.out_path,
        add_num_points=args.add_num_points,
        add_distance=args.add_distance,
    )
