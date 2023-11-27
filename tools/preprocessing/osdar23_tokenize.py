import copy
import json
import logging
import os
import re
import uuid
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from glob import glob

import numpy as np
from pypcd import pypcd
from scipy.spatial.transform import Rotation as R

from mmdet3d.core.bbox import LiDARInstance3DBoxes, box_np_ops
from mmdet3d.core.bbox.structures.box_3d_mode import Box3DMode
from mmdet3d.core.points.lidar_points import LiDARPoints

LABELS_POINT_CLOUDS_FOLDERNAME = "labels_point_clouds"


def get_args() -> Namespace:
    """
    Parse given arguments for tokenize_osdar23 function.

    Returns:
        Namespace: parsed arguments
    """
    parser = ArgumentParser()

    parser.add_argument("--root-path", type=str, required=True)
    parser.add_argument(
        "-log",
        "--loglevel",
        default="warning",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )

    return parser.parse_args()


def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_) if s]


def tokenize_osdar23_labels(root_path: str) -> None:
    splits = sorted(glob(os.path.join(root_path, "*")), key=natural_key)
    total_count = 0

    for split in splits:
        if not os.path.isdir(split):
            logging.warning(f"{split} is not a directory. Skipping...")
            continue

        logging.info(f"Tokenizing {split}...")
        count = tokenize(split)
        logging.info(f"Number of tokenized labels: {count} for {split}")
        total_count += count

    logging.info(f"Total number of tokenized labels: {total_count}")


def tokenize(
    split_path: str,
) -> int:
    count = 0

    label_paths = sorted(
        glob(os.path.join(split_path, LABELS_POINT_CLOUDS_FOLDERNAME, "*.json")), key=natural_key
    )

    labels_details = defaultdict(list)
    seq_ids = []
    seq_idx_max = {}

    for label_path in label_paths:
        basename = os.path.basename(label_path)
        seq_idx = basename.split("_")[0]
        seq_ids.append(seq_idx)
        seq_frame_idx = int(basename.split("_")[1])
        seq_idx_max[seq_idx] = max(seq_idx_max.get(seq_idx, 0), int(seq_frame_idx) + 1)

    labels_details = {k: [None] * seq_idx_max[k] for k in seq_ids}

    for label_path in label_paths:
        basename = os.path.basename(label_path)
        seq_idx = basename.split("_")[0]
        seq_frame_idx = int(basename.split("_")[1])

        token = uuid.uuid4().hex
        labels_details[seq_idx][seq_frame_idx] = {
            "label_path": label_path,
            "scene_token": None,
            "token": token,
            "prev": None,
            "next": None,
            "frame_idx": None,
        }

    for seq_idx, seq_labels in labels_details.items():
        for seq_frame_idx, seq_label in enumerate(seq_labels):
            if seq_label is None:
                continue

            # handle scene tokens
            if seq_frame_idx == 0:
                seq_label["scene_token"] = uuid.uuid4().hex
                seq_label["frame_idx"] = 0
            else:
                if seq_labels[seq_frame_idx - 1] is None:
                    seq_label["scene_token"] = uuid.uuid4().hex
                    seq_label["frame_idx"] = 0
                else:
                    seq_label["scene_token"] = seq_labels[seq_frame_idx - 1]["scene_token"]
                    seq_label["frame_idx"] = seq_labels[seq_frame_idx - 1]["frame_idx"] + 1

            # handle prev and next tokens
            if seq_frame_idx == 0:
                seq_label["prev"] = None
            else:
                if seq_labels[seq_frame_idx - 1] is None:
                    seq_label["prev"] = None
                else:
                    seq_label["prev"] = seq_labels[seq_frame_idx - 1]["token"]
            if seq_frame_idx == len(seq_labels) - 1:
                seq_label["next"] = None
            else:
                if seq_labels[seq_frame_idx + 1] is None:
                    seq_label["next"] = None
                else:
                    seq_label["next"] = seq_labels[seq_frame_idx + 1]["token"]

    # del none elements
    for seq_idx, seq_labels in labels_details.items():
        labels_details[seq_idx] = [x for x in seq_labels if x is not None]

    for seq_idx, seq_labels in labels_details.items():
        labels = None
        for seq_label in seq_labels:
            with open(seq_label["label_path"], "r") as f:
                labels = json.load(f)

            for _, fd in labels["openlabel"]["frames"].items():
                fd["frame_properties"]["original_scene_token"] = fd["frame_properties"][
                    "scene_token"
                ]
                fd["frame_properties"]["scene_token"] = seq_label["scene_token"]
                fd["frame_properties"]["token"] = seq_label["token"]
                fd["frame_properties"]["prev"] = seq_label["prev"]
                fd["frame_properties"]["next"] = seq_label["next"]
                fd["frame_properties"]["frame_idx"] = seq_label["frame_idx"]

            basename = os.path.basename(seq_label["label_path"])
            with open(seq_label["label_path"], "w") as f:
                logging.info(
                    f"Saving {basename:<20} frame_idx:{seq_label['frame_idx']} scene_token:{seq_label['scene_token']} token:{seq_label['token']} prev:{seq_label['prev']} next:{seq_label['next']}"
                )
                json.dump(labels, f)
                count += 1

    return count


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=args.loglevel.upper())
    tokenize_osdar23_labels(root_path=args.root_path)
