import json
import logging
import os
import uuid
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from dataclasses import dataclass
from glob import glob
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm


def get_args() -> Namespace:
    """
    Parse given arguments for add_tumtraf_dataset_difficulty_labels function.

    Returns:
        Namespace: parsed arguments
    """
    parser = ArgumentParser()

    parser.add_argument("--root-path", type=str, required=True)
    parser.add_argument("--out-path", type=str, required=False)
    parser.add_argument("--threshold-train", type=float, default=0.5, required=False)
    parser.add_argument("--threshold-val", type=float, default=0.5, required=False)
    parser.add_argument("--threshold-test", type=float, default=0.5, required=False)
    parser.add_argument(
        "-log",
        "--loglevel",
        default="warning",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )

    return parser.parse_args()


@dataclass()
class FrameDetails:
    img_label_s1_name: str
    img_label_s2_name: str
    pcd_label_name: str
    ts: float
    token: str
    scene_token: str
    frame_idx: int
    prev: str
    next: str


def add_tumtraf_dataset_difficulty_labels(
    root_path: str,
    out_path: str = None,
):
    splits = [os.path.basename(x) for x in glob(os.path.join(root_path, "*"))]
    for split in splits:
        if split.lower() in ["images", "labels_images", "labels_point_clouds", "point_clouds"]:
            splits = [""]
            break

    for split in splits:
        if split.endswith(".json"):
            continue

        # fmt: off
        pcd_label_folder = os.path.join(root_path, split, "labels_point_clouds", "s110_lidar_ouster_south")
        # fmt: on

        pcd_label_paths = sorted(glob(os.path.join(pcd_label_folder, "*")))

        if not out_path:
            pcd_label_folder_out = pcd_label_folder
        else:
            # fmt: off
            pcd_label_folder_out = os.path.join(out_path, split, "labels_point_clouds", "s110_lidar_ouster_south")
            # fmt: on

            os.makedirs(pcd_label_folder_out, mode=0o777, exist_ok=True)

        assert len(pcd_label_paths) != 0, "No point cloud label files found"
        total_counts = defaultdict(int)
        for idx, path in enumerate(pcd_label_paths):
            pcd_label_name = os.path.basename(path)
            json_out_path = os.path.join(pcd_label_folder_out, pcd_label_name)
            json_data = None
            with open(path, "r") as f:
                json_data = json.load(f)
                frame_idx = list(json_data["openlabel"]["frames"].keys())[0]
                objects = json_data["openlabel"]["frames"][frame_idx]["objects"]
                for obj_id, obj in objects.items():
                    cuboid = obj["object_data"]["cuboid"]
                    attributes_text = cuboid["attributes"]["text"]
                    attributes_num = cuboid["attributes"]["text"]

                    loc = np.asarray(obj["object_data"]["cuboid"]["val"][:3], dtype=np.float32)
                    distance = np.sqrt(np.sum(np.array(loc[:2]) ** 2))

                    num_points = 0
                    for x in attributes_num:
                        if x["name"] == "num_points":
                            num_points = x["val"]
                    occlusion_level = "UNKNOWN"
                    for x in attributes_text:
                        if x["name"] == "occlusion":
                            occlusion_level = x["val"]

                    # calculate difficulty
                    if occlusion_level == "MOSTLY_OCCLUDED":
                        difficulty = "hard"
                    elif occlusion_level == "PARTIALLY_OCCLUDED":
                        difficulty = "moderate"
                    elif distance <= 40 or num_points > 50:
                        difficulty = "easy"
                    elif (distance > 40 and distance <= 50) or (
                        num_points > 20 and num_points <= 50
                    ):
                        difficulty = "moderate"
                    elif distance > 50 or num_points <= 20:
                        difficulty = "hard"
                    else:
                        difficulty = "non"

                    total_counts[difficulty] += 1

                    json_data["openlabel"]["frames"][frame_idx]["objects"][obj_id]["object_data"][
                        "cuboid"
                    ]["attributes"]["text"].append({"name": "difficulty", "val": difficulty})
            if json_data is not None:
                with open(json_out_path, "w") as f:
                    json.dump(json_data, f)

        print(f"Total counts for {split}: {total_counts}")


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=args.loglevel.upper())
    add_tumtraf_dataset_difficulty_labels(
        root_path=args.root_path,
        out_path=args.out_path,
    )
