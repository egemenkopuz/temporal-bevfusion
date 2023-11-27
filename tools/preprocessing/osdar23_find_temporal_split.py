import json
import logging
import math
import multiprocessing
import os
import re
import shutil
import sys
import time
import warnings
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from dataclasses import dataclass, field
from glob import glob
from itertools import permutations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from tools.preprocessing.utils.table import create_header_line, log_table

warnings.filterwarnings("ignore")

CLASSES = [
    "lidar__cuboid__person",
    "lidar__cuboid__signal",
    "lidar__cuboid__catenary_pole",
    "lidar__cuboid__signal_pole",
    "lidar__cuboid__train",
    "lidar__cuboid__road_vehicle",
    "lidar__cuboid__buffer_stop",
    "lidar__cuboid__animal",
    "lidar__cuboid__switch",
    "lidar__cuboid__bicycle",
    "lidar__cuboid__crowd",
    "lidar__cuboid__wagons",
    "lidar__cuboid__signal_bridge",
]

CLASS_WEIGHTS = np.asarray([10, 5, 5, 5, 2, 5, 5, 10, 5, 2, 0, 0, 0]) * 50
NUM_POINTS_WEIGHTS = np.asarray([10, 5, 5, 5, 2, 5, 5, 10, 5, 2, 0, 0, 0]) * 1
OCCLUSION_WEIGHTS = np.asarray([10, 5, 5, 5, 2, 5, 5, 10, 5, 2, 0, 0, 0]) * 5
DISTANCE_WEIGHTS = np.asarray([10, 5, 5, 5, 2, 5, 5, 10, 5, 2, 0, 0, 0]) * 3

OCCLUSION_LEVELS = ["0-25 %", "25-50 %", "50-75 %", "75-99 %", "100 %"]
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
SPLIT_RATIO = [0.8, 0.1, 0.1]

BLACKLISTED_SEQUENCE_NAMES = ["4_station_pedestrian_bridge_4.4", "19_vegetation_curve_19.1"]

parent_img_folder_name = "images"
parent_img_labels_folder_name = "labels_images"
parent_pcd_folder_name = "lidar"
parent_pcd_labels_folder_name = "labels_point_clouds"

img_types = [
    "rgb_center",
    "rgb_left",
    "rgb_right",
    "rgb_highres_center",
    "rgb_highres_left",
    "rgb_highres_right",
]


def get_args() -> Namespace:
    """
    Parse given arguments for find_and_create_osdar23_temporal_split function.

    Returns:
        Namespace: parsed arguments
    """
    parser = ArgumentParser()

    parser.add_argument("--root-path", type=str, required=True)
    parser.add_argument("--out-path", type=str, required=True)
    parser.add_argument("--segment-size", type=int, default=10, required=False)
    parser.add_argument("--perm-limit", type=int, default=2e4, required=False)
    parser.add_argument("-p", type=int, default=4, required=False)
    parser.add_argument("--create", default=False, action="store_true")
    parser.add_argument("--include-all-classes", default=False, action="store_true")
    parser.add_argument("--include-all-sequences", default=False, action="store_true")
    parser.add_argument("--include-same-classes-in-distance", default=False, action="store_true")
    parser.add_argument("--distance-th", type=float, default=0.5, required=False)
    parser.add_argument("--include-same-classes-in-num-points", default=False, action="store_true")
    parser.add_argument("--num-points-th", type=float, default=0.5, required=False)
    parser.add_argument("--include-same-classes-in-occlusion", default=False, action="store_true")
    parser.add_argument("--occlusion-th", type=float, default=0.5, required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--split-ratios", nargs="+", default=["0.8", "0.1", "0.1"])
    parser.add_argument("--point-cloud-range", nargs="+", default=[])
    parser.add_argument("--exclude-classes", nargs="+", default=[])
    parser.add_argument(
        "-log",
        "--loglevel",
        default="info",
        help="Provide logging level. Example --loglevel debug, default=info",
    )

    return parser.parse_args()


@dataclass()
class TemporalSequenceDetails:
    # fmt: off
    scene_token: str
    sequence_name: str
    no_total_frames: int
    total_bboxes: int
    total_bboxes_filtered: int
    total_class_stats: Dict[str, int] = field(repr=True, compare=False)
    total_distance_stats: Dict[str, Dict[str, int]] = field(repr=False, compare=False)
    total_num_points_stats: Dict[str, Dict[str, int]] = field(repr=False, compare=False)
    total_occlusion_stats: Dict[str, Dict[str, int]] = field(repr=False, compare=False)
    frame_class_stats: List[Dict[str, int]] = field(default_factory=list, repr=False, compare=False)
    frame_distance_stats: List[Dict[str, Dict[str, int]]] = field(default_factory=list, repr=False, compare=False)
    frame_num_points_stats: List[Dict[str, Dict[str, int]]] = field(default_factory=list, repr=False, compare=False)
    frame_occlusion_stats: List[Dict[str, Dict[str, int]]] = field(default_factory=list, repr=False, compare=False)
    frame_img_rgb_center_paths: List[str] = field(default_factory=list, repr=False, compare=False)
    frame_img_rgb_left_paths: List[str] = field(default_factory=list, repr=False, compare=False)
    frame_img_rgb_right_paths: List[str] = field(default_factory=list, repr=False, compare=False)
    frame_img_rgb_highres_center_paths: List[str] = field(default_factory=list, repr=False, compare=False)
    frame_img_rgb_highres_left_paths: List[str] = field(default_factory=list, repr=False, compare=False)
    frame_img_rgb_highres_right_paths: List[str] = field(default_factory=list, repr=False, compare=False)
    frame_pcd_paths: List[str] = field(default_factory=list, repr=False, compare=False)
    frame_pcd_labels_paths: List[str] = field(default_factory=list, repr=False, compare=False)
    # fmt: on

    def __len__(self):
        return self.total_bboxes

    def get_class_stats_in_range(self, start: int, end: int) -> Dict[str, int]:
        return {cls: sum([x[cls] for x in self.frame_class_stats[start:end]]) for cls in CLASSES}

    def get_distance_stats_in_range(self, start: int, end: int) -> Dict[str, int]:
        return {
            cls: {
                distance: sum([x[cls][distance] for x in self.frame_distance_stats[start:end]])
                for distance in DISTANCE_LEVELS.keys()
            }
            for cls in CLASSES
        }

    def get_num_points_stats_in_range(self, start: int, end: int) -> Dict[str, int]:
        return {
            cls: {
                num_points: sum(
                    [x[cls][num_points] for x in self.frame_num_points_stats[start:end]]
                )
                for num_points in NUM_POINTS_LEVELS.keys()
            }
            for cls in CLASSES
        }

    def get_occlusion_stats_in_range(self, start: int, end: int) -> Dict[str, int]:
        return {
            cls: {
                occlusion: sum([x[cls][occlusion] for x in self.frame_occlusion_stats[start:end]])
                for occlusion in OCCLUSION_LEVELS
            }
            for cls in CLASSES
        }

    def get_path_list_in_range(self, start: int, end: int) -> Dict[str, List[str]]:
        return {
            # fmt: off
            "frame_img_rgb_center_paths": self.frame_img_rgb_center_paths[start:end],
            "frame_img_rgb_left_paths": self.frame_img_rgb_left_paths[start:end],
            "frame_img_rgb_right_paths": self.frame_img_rgb_right_paths[start:end],
            "frame_img_rgb_highres_center_paths": self.frame_img_rgb_highres_center_paths[start:end],
            "frame_img_rgb_highres_left_paths": self.frame_img_rgb_highres_left_paths[start:end],
            "frame_img_rgb_highres_right_paths": self.frame_img_rgb_highres_right_paths[start:end],
            "frame_pcd_paths": self.frame_pcd_paths[start:end],
            "frame_pcd_labels_paths": self.frame_pcd_labels_paths[start:end],
            # fmt: on
        }


def create_sequence_details(
    root_path: str, point_cloud_range: List[float] = [], sequences_seperated: bool = True
) -> Dict[str, TemporalSequenceDetails]:
    """
    Create a dictionary of sequence details.

    Args:
        root_path: root path of the dataset
        point_cloud_range: range of point cloud to include

    Returns:
        Dict: dictionary of sequence details
    """
    data: Dict[str, TemporalSequenceDetails] = {}

    pcds = []
    pcds_labels = []
    imgs = defaultdict(list)

    if sequences_seperated:
        seq_paths = sorted(glob(os.path.join(root_path, "*")), key=natural_key)
        # check if there is any blacklisted sequence
        remove_idx = []
        for x in BLACKLISTED_SEQUENCE_NAMES:
            for i, y in enumerate(seq_paths):
                if x == os.path.basename(y):
                    remove_idx.append(i)
        # remove blacklisted sequences
        for i in sorted(remove_idx, reverse=True):
            del seq_paths[i]

        # fmt: off
        for seq_root_path in seq_paths:
            pcds.extend(sorted(glob(os.path.join(seq_root_path, parent_pcd_folder_name, "*.pcd")), key=natural_key))
            pcds_labels.extend(sorted(glob(os.path.join(seq_root_path, parent_pcd_labels_folder_name, "*.json")), key=natural_key))
            for img_type in img_types:
                imgs[img_type].extend(sorted(glob(os.path.join(seq_root_path, img_type, "*")), key=natural_key))
        # fmt: on

        # check if there is any missing file
        for i in range(len(pcds)):
            assert os.path.basename(pcds[i])[:-4] == os.path.basename(pcds_labels[i])[:-5]
        for img_type in img_types:
            assert len(imgs[img_type]) == len(pcds) == len(pcds_labels)
    else:
        pcds.extend(
            sorted(glob(os.path.join(root_path, parent_pcd_folder_name, "*.pcd")), key=natural_key)
        )
        pcds_labels = sorted(glob(os.path.join(root_path, parent_pcd_labels_folder_name, "*.json")))
        found_img_types = [
            os.path.basename(x)
            for x in sorted(
                glob(os.path.join(root_path, parent_img_folder_name, "*")), key=natural_key
            )
        ]
        for img_type in found_img_types:
            imgs[img_type].extend(
                sorted(
                    glob(os.path.join(root_path, parent_img_folder_name, img_type, "*")),
                    key=natural_key,
                )
            )

    for i, lp in enumerate(tqdm(pcds_labels, desc="reading")):
        with open(lp, "r") as f:
            json_data = json.load(f)
            frame_idx = list(json_data["openlabel"]["frames"].keys())[0]
            frame_properties = json_data["openlabel"]["frames"][frame_idx]["frame_properties"]
            frame_objects = json_data["openlabel"]["frames"][frame_idx]["objects"]

            scene_token = frame_properties["scene_token"]
            # scene name is parent of parent directory name
            sequence_name = os.path.basename(os.path.dirname(os.path.dirname(lp)))
            if scene_token not in data:
                data[scene_token] = TemporalSequenceDetails(
                    scene_token=scene_token,
                    sequence_name=sequence_name,
                    no_total_frames=0,
                    total_bboxes=0,
                    total_bboxes_filtered=0,
                    total_class_stats={cls: 0 for cls in CLASSES},
                    total_distance_stats={
                        cls: {x: 0 for x in DISTANCE_LEVELS.keys()} for cls in CLASSES
                    },
                    total_num_points_stats={
                        cls: {x: 0 for x in NUM_POINTS_LEVELS.keys()} for cls in CLASSES
                    },
                    total_occlusion_stats={
                        cls: {x: 0 for x in OCCLUSION_LEVELS} for cls in CLASSES
                    },
                )

            class_stats = {cls: 0 for cls in CLASSES}
            distance_stats = {cls: {x: 0 for x in DISTANCE_LEVELS.keys()} for cls in CLASSES}
            num_points_stats = {cls: {x: 0 for x in NUM_POINTS_LEVELS.keys()} for cls in CLASSES}
            occlusion_stats = {cls: {x: 0 for x in OCCLUSION_LEVELS} for cls in CLASSES}

            for obj in frame_objects.values():
                if "cuboid" in obj["object_data"]:
                    obj_type = obj["object_data"]["cuboid"][0]["name"]
                    loc = np.asarray(obj["object_data"]["cuboid"][0]["val"][:3], dtype=np.float32)

                    # filter bbox outside of given point cloud range
                    if len(point_cloud_range) != 0 and not (
                        loc[0] > point_cloud_range[0]
                        and loc[0] < point_cloud_range[3]
                        and loc[1] > point_cloud_range[1]
                        and loc[1] < point_cloud_range[4]
                        and loc[2] > point_cloud_range[2]
                        and loc[2] < point_cloud_range[5]
                    ):
                        data[scene_token].total_bboxes_filtered += 1
                        continue
                    else:
                        data[scene_token].total_bboxes += 1

                    attributes = obj["object_data"]["cuboid"][0]["attributes"]

                    class_stats[obj_type] += 1
                    for x in attributes["text"]:
                        if x["name"] == "num_points_level":
                            num_points_level = x["val"]
                            data[scene_token].total_num_points_stats[obj_type][
                                num_points_level
                            ] += 1
                            num_points_stats[obj_type][num_points_level] += 1
                        elif x["name"] == "distance_level":
                            distance_level = x["val"]
                            data[scene_token].total_distance_stats[obj_type][distance_level] += 1
                            distance_stats[obj_type][distance_level] += 1
                        elif x["name"] == "occlusion":
                            occlusion = x["val"]
                            data[scene_token].total_occlusion_stats[obj_type][occlusion] += 1
                            occlusion_stats[obj_type][occlusion] += 1

            data[scene_token].frame_distance_stats.append(distance_stats)
            data[scene_token].frame_num_points_stats.append(num_points_stats)
            data[scene_token].frame_occlusion_stats.append(occlusion_stats)

            data[scene_token].no_total_frames += 1
            data[scene_token].frame_class_stats.append(class_stats)
            for x in class_stats:
                data[scene_token].total_class_stats[x] += class_stats[x]

            for img in imgs:
                getattr(data[scene_token], f"frame_img_{img}_paths").append(imgs[img][i])

            data[scene_token].frame_pcd_paths.append(pcds[i])
            data[scene_token].frame_pcd_labels_paths.append(pcds_labels[i])

    title = f"{'Token':<35} {'Sequence':<40} {'Frames':<10} {'Total Objects':<20} {'Filtered Objects':<20} {'Remaining Objects':<20}"
    logging.info(title)
    logging.info(create_header_line(len(title), "="))
    for x, y in data.items():
        tot = y.total_bboxes + y.total_bboxes_filtered
        filtered = f"{str(y.total_bboxes_filtered):<5}  ({y.total_bboxes_filtered / tot:.3f})"
        remaining = f"{str(y.total_bboxes):<5} ({y.total_bboxes / tot:.3f})"
        logging.info(
            f"{x:<35} {y.sequence_name:<40} {y.no_total_frames:<10} {tot:<20} {filtered:<20} {remaining:<20}"
        )

    return data


def calculate_segment_loss(
    perm: List[Tuple[str, Tuple[int, int]]],
    data: Dict[str, TemporalSequenceDetails],
    split_ratio: List[float] = [0.8, 0.1, 0.1],
    classes: List[str] = CLASSES,
    class_weights: Optional[List[float]] = None,
    distance_weights: Optional[List[float]] = None,
    num_points_weights: Optional[List[float]] = None,
    occlusion_weights: Optional[List[float]] = None,
    include_all_classes: bool = False,
    include_all_sequences: bool = False,
    include_same_classes_in_distance: bool = False,
    distance_threshold_ratio: float = 1.0,
    include_same_classes_in_num_points: bool = False,
    num_points_threshold_ratio: float = 1.0,
    include_same_classes_in_occlusion: bool = False,
    occlusion_threshold_ratio: float = 1.0,
) -> Dict[str, Any]:
    """
    Calculate the loss of a given permutation of segments.

    The loss is calculated by comparing the class distributions between splits.

    Args:
        segment_size: number of frames in a segment
        perm: permutation of segments
        data: dictionary of sequence details
        split_ratio: ratio of splits
        classes: list of classes
        class_weights: weights of classes
        difficulty_weights: weights of difficulty level
        distance_weights: weights of distance
        num_points_weights: weights of number of points
        occlusion_weights: weights of occlusion levels
        include_all_classes: whether to force to have each class presence in all splits
        include_all_sequencess: whether to force to have each sequence presence in all splits
        include_same_classes_in_distance: whether to force to have same classes in all splits with given distance threshold
        distance_threshold_ratio: distance threshold ratio
        include_same_classes_in_num_points: whether to force to have same classes in all splits with given number of points threshold
        num_points_threshold_ratio: number of points threshold ratio
        include_same_classes_in_occlusion: whether to force to have same classes in all splits with given occlusion threshold
        occlusion_threshold_ratio: occlusion threshold ratio

    Returns:
        Dict: dictionary of details
    """
    loss = 0
    split_count = len(split_ratio)

    if split_count == 3:  # train, val, test
        split_idx_ranges = [
            0,
            math.ceil(split_ratio[0] * len(perm)),
            math.ceil((split_ratio[0] + split_ratio[1]) * len(perm)),
            len(perm),
        ]
    elif split_count == 2:
        split_idx_ranges = [
            0,
            int(split_ratio[0] * len(perm)),
            len(perm),
        ]
    else:
        raise Exception("Number of splits must be either 2 or 3")

    assert all(
        [float(x).is_integer() for x in split_idx_ranges]
    ), "segment_size and split_ratio results in non-integer split_idx_ranges"

    seq_cls_assigned_split = [0] * split_idx_ranges[1]
    seq_cls_assigned_split.extend([1] * (split_idx_ranges[2] - split_idx_ranges[1]))
    if split_count == 3:  # train, val, test
        seq_cls_assigned_split.extend([2] * (split_idx_ranges[3] - split_idx_ranges[2]))

    split_cls_counts = [{cls: 0 for cls in classes} for _ in range(len(split_ratio))]
    split_occlusion_counts = [
        {cls: {x: 0 for x in OCCLUSION_LEVELS} for cls in classes} for _ in range(len(split_ratio))
    ]

    split_num_points_counts = [
        {cls: {x: 0 for x in NUM_POINTS_LEVELS.keys()} for cls in classes}
        for _ in range(len(split_ratio))
    ]
    split_distance_counts = [
        {cls: {x: 0 for x in DISTANCE_LEVELS.keys()} for cls in classes}
        for _ in range(len(split_ratio))
    ]

    class_counts = []
    occlusion_counts = []
    num_points_counts = []
    distance_counts = []

    for x in perm:
        token = x[0]
        start_idx = x[1][0]
        end_idx = x[1][1]

        class_counts.append(data[token].get_class_stats_in_range(start_idx, end_idx))
        occlusion_counts.append(data[token].get_occlusion_stats_in_range(start_idx, end_idx))
        num_points_counts.append(data[token].get_num_points_stats_in_range(start_idx, end_idx))
        distance_counts.append(data[token].get_distance_stats_in_range(start_idx, end_idx))

    for idx, (cls_c, occ_c, np_c, d_c) in enumerate(
        zip(class_counts, occlusion_counts, num_points_counts, distance_counts)
    ):
        if seq_cls_assigned_split[idx] == 0:  # in train split
            for cls in classes:
                split_cls_counts[0][cls] += cls_c[cls]
                for occl in OCCLUSION_LEVELS:
                    split_occlusion_counts[0][cls][occl] += occ_c[cls][occl]
                for nump in NUM_POINTS_LEVELS.keys():
                    split_num_points_counts[0][cls][nump] += np_c[cls][nump]
                for d in DISTANCE_LEVELS.keys():
                    split_distance_counts[0][cls][d] += d_c[cls][d]
        elif seq_cls_assigned_split[idx] == 1:  # in val split
            for cls in classes:
                split_cls_counts[1][cls] += cls_c[cls]
                for occl in OCCLUSION_LEVELS:
                    split_occlusion_counts[1][cls][occl] += occ_c[cls][occl]
                for nump in NUM_POINTS_LEVELS.keys():
                    split_num_points_counts[1][cls][nump] += np_c[cls][nump]
                for d in DISTANCE_LEVELS.keys():
                    split_distance_counts[1][cls][d] += d_c[cls][d]
        else:  # in test split
            x = 0
            for cls in classes:
                split_cls_counts[2][cls] += cls_c[cls]
                for occl in OCCLUSION_LEVELS:
                    split_occlusion_counts[2][cls][occl] += occ_c[cls][occl]
                for nump in NUM_POINTS_LEVELS.keys():
                    split_num_points_counts[2][cls][nump] += np_c[cls][nump]
                for d in DISTANCE_LEVELS.keys():
                    split_distance_counts[2][cls][d] += d_c[cls][d]

    # calculate the distributions in splits
    split_cls_ratios = []
    split_occlusion_ratios = []
    split_num_points_ratios = []
    split_distance_ratios = []

    for i in range(len(split_ratio)):
        total_split_cls_counts = sum(split_cls_counts[i].values())
        split_cls_ratios.append(
            {cls: x / total_split_cls_counts for cls, x in split_cls_counts[i].items()}
        )

    # force to have each class in all of the splits
    if include_all_classes:
        for x in split_cls_counts:
            for cls, count in x.items():
                if count == 0:
                    return {"loss": None, "reason": "include_all_classes"}

    # force to have each sequence in all of the splits
    if include_all_sequences:
        for i in range(len(split_ratio)):
            for j, seq in enumerate(perm):
                a = (
                    np.asarray(perm)[int(split_idx_ranges[i]) : int(split_idx_ranges[i + 1])] == seq
                ).sum()
                if a == 0:
                    return {"loss": None, "reason": "include_all_sequences"}

    for i in range(len(split_ratio)):
        total_split_occlusion_counts = [
            sum([split_occlusion_counts[i][cls][occl] for cls in classes])
            for occl in OCCLUSION_LEVELS
        ]
        split_occlusion_ratios.append(
            {
                cls: {
                    occl: split_occlusion_counts[i][cls][occl] / total_split_occlusion_counts[j]
                    if total_split_occlusion_counts[j] != 0
                    else 0
                    for j, occl in enumerate(OCCLUSION_LEVELS)
                }
                for cls in classes
            }
        )

    if include_same_classes_in_occlusion:
        class_presences = [
            {o: {cls: False for cls in classes} for o in OCCLUSION_LEVELS}
            for _ in range(split_count)
        ]
        for split_idx, x in enumerate(split_occlusion_counts):
            for cls, occs in x.items():
                for occ, v in occs.items():
                    if v > 0:
                        class_presences[split_idx][occ][cls] = True
        tot = 0
        for x in class_presences[1:]:
            for id, clss in x.items():
                for cls, cls_b in clss.items():
                    if cls_b == class_presences[0][id][cls]:
                        tot += 1
        th = len(classes) * (split_count - 1) * len(OCCLUSION_LEVELS)
        if th * occlusion_threshold_ratio > tot:
            return {"loss": None, "reason": "occlusion-threshold"}

    for i in range(len(split_ratio)):
        total_split_num_points_counts = [
            sum([split_num_points_counts[i][cls][nump] for cls in classes])
            for nump in NUM_POINTS_LEVELS.keys()
        ]
        split_num_points_ratios.append(
            {
                cls: {
                    nump: split_num_points_counts[i][cls][nump] / total_split_num_points_counts[j]
                    if total_split_num_points_counts[j] != 0
                    else 0
                    for j, nump in enumerate(NUM_POINTS_LEVELS.keys())
                }
                for cls in classes
            }
        )

    if include_same_classes_in_num_points:
        class_presences = [
            {p: {cls: False for cls in classes} for p in NUM_POINTS_LEVELS.keys()}
            for _ in range(split_count)
        ]
        for split_idx, x in enumerate(split_num_points_counts):
            for cls, ps in x.items():
                for p, v in ps.items():
                    if v > 0:
                        class_presences[split_idx][p][cls] = True
        tot = 0
        for x in class_presences[1:]:
            for id, clss in x.items():
                for cls, cls_b in clss.items():
                    if cls_b == class_presences[0][id][cls]:
                        tot += 1
        th = len(classes) * (split_count - 1) * len(NUM_POINTS_LEVELS.keys())
        if th * num_points_threshold_ratio > tot:
            return {"loss": None, "reason": "num_points-threshold"}

    for i in range(len(split_ratio)):
        total_split_distance_counts = [
            sum([split_distance_counts[i][cls][d] for cls in classes])
            for d in DISTANCE_LEVELS.keys()
        ]
        split_distance_ratios.append(
            {
                cls: {
                    d: split_distance_counts[i][cls][d] / total_split_distance_counts[j]
                    if total_split_distance_counts[j] != 0
                    else 0
                    for j, d in enumerate(DISTANCE_LEVELS.keys())
                }
                for cls in classes
            }
        )

    if include_same_classes_in_distance:
        class_presences = [
            {d: {cls: False for cls in classes} for d in DISTANCE_LEVELS.keys()}
            for _ in range(split_count)
        ]
        for split_idx, x in enumerate(split_distance_counts):
            for cls, dists in x.items():
                for dist, v in dists.items():
                    if v > 0:
                        class_presences[split_idx][dist][cls] = True
        tot = 0
        for x in class_presences[1:]:
            for id, clss in x.items():
                for cls, cls_b in clss.items():
                    if cls_b == class_presences[0][id][cls]:
                        tot += 1
        th = len(classes) * (split_count - 1) * len(DISTANCE_LEVELS.keys())
        if th * distance_threshold_ratio > tot:
            return {"loss": None, "reason": "distance-threshold"}

    # compare the class distributions between splits
    total_cls_loss = 0
    for i in range(1, len(split_cls_ratios)):
        for j, cls in enumerate(classes):
            if class_weights is None:
                total_cls_loss += abs(split_cls_ratios[0][cls] - split_cls_ratios[i][cls])
            else:
                total_cls_loss += (
                    abs(split_cls_ratios[0][cls] - split_cls_ratios[i][cls]) * class_weights[j]
                )
    loss += total_cls_loss

    # compare the occlusion distributions
    total_occlusion_loss = 0
    for i in range(1, len(split_cls_ratios)):
        for j in OCCLUSION_LEVELS:
            for k, cls in enumerate(classes):
                if occlusion_weights is None:
                    total_occlusion_loss += abs(
                        split_occlusion_ratios[0][cls][j] - split_occlusion_ratios[i][cls][j]
                    )
                else:
                    total_occlusion_loss += (
                        abs(split_occlusion_ratios[0][cls][j] - split_occlusion_ratios[i][cls][j])
                        * occlusion_weights[k]
                    )

    loss += total_occlusion_loss

    # compare the number of points distributions
    total_num_points_loss = 0
    for i in range(1, len(split_cls_ratios)):
        for j in NUM_POINTS_LEVELS.keys():
            for k, cls in enumerate(classes):
                if num_points_weights is None:
                    total_num_points_loss += abs(
                        split_num_points_ratios[0][cls][j] - split_num_points_ratios[i][cls][j]
                    )
                else:
                    total_num_points_loss += (
                        abs(split_num_points_ratios[0][cls][j] - split_num_points_ratios[i][cls][j])
                        * num_points_weights[k]
                    )

    loss += total_num_points_loss

    # compare the distance distributions
    total_distance_loss = 0
    for i in range(1, len(split_cls_ratios)):
        for j in DISTANCE_LEVELS.keys():
            for k, cls in enumerate(classes):
                if distance_weights is None:
                    total_distance_loss += abs(
                        split_distance_ratios[0][cls][j] - split_distance_ratios[i][cls][j]
                    )
                else:
                    total_distance_loss += (
                        abs(split_distance_ratios[0][cls][j] - split_distance_ratios[i][cls][j])
                        * distance_weights[k]
                    )

    loss += total_distance_loss

    return {
        "loss": loss,
        "split_idx_ranges": split_idx_ranges,
        "class": {
            "loss": total_cls_loss,
            "split_counts": split_cls_counts,
            "split_ratios": split_cls_ratios,
        },
        "occlusion": {
            "loss": total_occlusion_loss,
            "split_counts": split_occlusion_counts,
            "split_ratios": split_occlusion_ratios,
        },
        "num_points": {
            "loss": total_num_points_loss,
            "split_counts": split_num_points_counts,
            "split_ratios": split_num_points_ratios,
        },
        "distance": {
            "loss": total_distance_loss,
            "split_counts": split_distance_counts,
            "split_ratios": split_distance_ratios,
        },
        "perm": perm,
    }


# multi-processing task function
def search_task(position_, lock_, queue_, perms_, *args):
    total_iterations = len(perms_)
    n_found = 0
    best_loss = None
    best_segment_loss_details = {"loss": None}
    filter_stats = defaultdict(int)

    with lock_:
        bar = tqdm(
            desc=f"Searching {position_}",
            total=total_iterations,
            position=position_,
            leave=False,
            mininterval=1,
        )

    for p in perms_:
        segment_loss_details = calculate_segment_loss(p, *args)
        loss = segment_loss_details["loss"]
        if loss is not None and (best_loss is None or loss < best_loss):
            best_loss = loss
            best_segment_loss_details = segment_loss_details
            n_found += 1
        elif loss is None:
            reason = segment_loss_details["reason"]
            filter_stats[reason] += 1
        else:
            n_found += 1
        with lock_:
            bar.update(1)
    with lock_:
        bar.close()

    queue_.put((n_found, filter_stats, best_segment_loss_details))


# multi-processing copy task function
def copy_task(
    position_, lock_, split_source_paths_: List[Tuple], target_paths_, splits_, split_idx_ranges_
):
    total_iterations = (
        len(split_source_paths_) * len(split_source_paths_[0][2].values()) * (len(img_types) + 2)
    )

    with lock_:
        bar = tqdm(
            desc=f"Copying {position_}",
            total=total_iterations,
            position=position_,
            leave=False,
            mininterval=1,
        )

    for idx, seq_name, source_paths in split_source_paths_:
        seq_id = seq_name.split("_")[-1]
        if split_idx_ranges_[0] <= idx < split_idx_ranges_[1]:
            split = splits_[0]
        elif split_idx_ranges_[1] <= idx < split_idx_ranges_[2]:
            split = splits_[1]
        else:
            split = splits_[2]
        for source_name, paths in source_paths.items():
            target_root_path = target_paths_[split][source_name]
            for path in paths:
                new_name = seq_id + "_" + os.path.basename(path)
                target_path = os.path.join(target_root_path, new_name)
                shutil.copy2(path, target_path)
                with lock_:
                    bar.update(1)
    with lock_:
        bar.close()


def create_and_copy_split(
    target_path: str,
    data: Dict[str, TemporalSequenceDetails],
    perm: List[Tuple[str, Tuple[int]]],
    split_idx_ranges: List[float],
    splits: List[str] = ["train", "val", "test"],
    no_process: int = 1,
) -> None:
    """
    Create and copy the split to the target path.

    Args:
        target_path: target path to copy the split
        data: dictionary of sequence details
        perm: selected permutation of segments
        split_idx_ranges: split index ranges
        splits: list of splits
        no_process: number of processes for copying
    """
    split_source_paths = []

    for i, x in enumerate(perm):
        # get class counts for current segment
        token = x[0]
        start_idx = x[1][0]
        end_idx = x[1][1]
        split_source_paths.append(
            (
                i,
                data[token].sequence_name,
                data[token].get_path_list_in_range(start_idx, end_idx),
            )
        )

    target_paths = defaultdict(dict)
    for split in splits:
        # fmt: off
        for img_type in img_types:
            target_paths[split][f"frame_img_{img_type}_paths"] = os.path.join(target_path, split, parent_img_folder_name, img_type)
        target_paths[split]["frame_pcd_paths"] = os.path.join(target_path, split, parent_pcd_folder_name)
        target_paths[split]["frame_pcd_labels_paths"] = os.path.join(target_path, split, parent_pcd_labels_folder_name)

        for x, y in target_paths[split].items():
            os.makedirs(y, exist_ok=True)
        # fmt: on

    manager = multiprocessing.Manager()
    lock = manager.Lock()

    def custom_callback(e):
        logging.error(f"{type(e).__name__} {__file__} {e}")

    with multiprocessing.Pool(no_process) as pool:
        n = len(split_source_paths) // no_process
        ssp = [split_source_paths[i : i + n] for i in range(0, len(split_source_paths), n)]
        # print total in
        total_frame = 0
        for x in split_source_paths:
            total_frame += len(x[2]["frame_img_rgb_center_paths"])

        if len(ssp) > no_process:
            ssp[-2].extend(ssp[-1])
            ssp.pop(-1)
        for position in range(no_process):
            pool.apply_async(
                copy_task,
                [
                    position + 1,
                    lock,
                    ssp[position],
                    target_paths,
                    splits,
                    split_idx_ranges,
                ],
                error_callback=custom_callback,
            )
            time.sleep(2)

        pool.close()
        pool.join()


def save_split_details_and_config(
    target_path: str,
    bucket_details: Dict[str, int],
    data: Dict[str, TemporalSequenceDetails],
    perm: List[Tuple[str, Tuple[int]]],
    best_segment_loss_details,
    splits: List[str] = ["train", "val", "test"],
    split_ratio: List[int] = [0.8, 0.1, 0.1],
) -> None:
    """
    Save the split details and config to the target path.

    Args:
        target_path: target path to copy the split
        bucket_details: bucket details
        data: dictionary of sequence details
        perm: selected permutation of segments
        best_segment_loss_details: best segment loss details
        splits: list of splits
        split_ratio: ratio of splits
    """
    split_details = {}
    split_idx_ranges = best_segment_loss_details["split_idx_ranges"]

    for i, split in enumerate(splits):
        split_details[split] = {}
        split_details[split]["segment_size"] = segment_size
        split_details[split]["no_frames"] = segment_size * len(
            perm[int(split_idx_ranges[i]) : int(split_idx_ranges[i + 1])]
        )
        split_details[split]["original_sequences"] = {}
        for j, seq in enumerate(data.keys()):
            split_details[split]["original_sequences"][seq] = (
                np.asarray(perm)[int(split_idx_ranges[i]) : int(split_idx_ranges[i + 1])] == seq
            ).sum()

        for x in ["class", "distance", "num_points", "occlusion"]:
            split_details[split][x] = {
                "split_counts": best_segment_loss_details[x]["split_counts"][i],
                "split_ratios": best_segment_loss_details[x]["split_ratios"][i],
            }

    split_count = len(split_ratio)

    if split_count == 3:  # train, val, test
        split_idx_ranges = [
            0,
            math.ceil(split_ratio[0] * len(perm)),
            math.ceil((split_ratio[0] + split_ratio[1]) * len(perm)),
            len(perm),
        ]
    elif split_count == 2:
        split_idx_ranges = [
            0,
            int(split_ratio[0] * len(perm)),
            len(perm),
        ]

    seq_assigned_split = [0] * split_idx_ranges[1]
    seq_assigned_split.extend([1] * (split_idx_ranges[2] - split_idx_ranges[1]))
    if len(splits) == 3:  # train, val, test
        seq_assigned_split.extend([2] * (split_idx_ranges[3] - split_idx_ranges[2]))
    split_details["perm_details"] = [
        {x[0]: {"start": x[1][0], "end": x[1][1], "split": seq_assigned_split[i]}}
        for i, x in enumerate(perm)
    ]

    split_details["losses"] = {"total": best_segment_loss_details["loss"]}
    for x in ["class", "distance", "num_points", "occlusion"]:
        split_details["losses"][x] = best_segment_loss_details[x]["loss"]

    split_details["bucket_details"] = bucket_details
    split_details["perm"] = best_segment_loss_details["perm"]

    split_config = {
        "perm": best_segment_loss_details["perm"],
        "split_idx_ranges": split_idx_ranges,
        "splits": splits,
        "split_ratio": split_ratio,
        "data": data,
    }

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, TemporalSequenceDetails):
                return obj.__dict__
            return super(NpEncoder, self).default(obj)

    with open(os.path.join(target_path, "split_details.json"), "w") as f:
        json.dump(split_details, f, indent=4, cls=NpEncoder)

    with open(os.path.join(target_path, "split_config.json"), "w") as f:
        json.dump(split_config, f, indent=4, cls=NpEncoder)


def log_summary(
    best_segment_loss_details: Dict[str, Any],
    splits: List[str] = ["train", "val", "test"],
    classes: List[str] = CLASSES,
):
    """
    Log and print the summary of the best segment loss details.

    Args:
        best_segment_loss_details: best segment loss details
        splits: list of splits
        classes: list of classes
    """
    out = []
    split_idx_ranges = best_segment_loss_details["split_idx_ranges"]

    # logging.info split statistics
    out.append("")
    out.append(create_header_line(172, "="))
    out.append(f"Best Split Summary w/ loss : {best_segment_loss_details['loss']}")
    out.append(f"- Class loss: {best_segment_loss_details['class']['loss']}")
    out.append(f"- Distance loss: {best_segment_loss_details['distance']['loss']}")
    out.append(f"- No. Points loss: {best_segment_loss_details['num_points']['loss']}")
    out.append(f"- Occlusion loss: {best_segment_loss_details['occlusion']['loss']}")
    out.append(create_header_line(172, "="))
    out.append("Number of segments and frames in corresponding sequences:")
    out.append(create_header_line(172, "="))
    title = f"{'sequence':<40} {'token':<40}" + "".join(
        [f" {splits[i]:<15}" for i in range(len(splits))]
    )
    out.append(title)
    out.append(create_header_line(len(title), "-"))
    total_seq_counts = np.asarray([0] * len(splits), dtype=np.int32)
    for j, token in enumerate(data.keys()):
        seq_counts = np.asarray(
            [
                (
                    np.asarray(best_perm)[int(split_idx_ranges[x]) : int(split_idx_ranges[x + 1])]
                    == token
                ).sum()
                for x in range(len(splits))
            ],
            dtype=np.int32,
        )
        sequence_name = data[token].sequence_name
        info = f"{sequence_name:<40} {token:<40}" + "".join(
            [f" {seq_counts[i]:<15}" for i in range(len(splits))]
        )
        out.append(info)
        total_seq_counts += seq_counts
    out.append(create_header_line(len(title), "-"))
    info = f"{'':<40} {'All':<40}" + "".join(
        [f" {total_seq_counts[i]:<5}" for i in range(len(splits))]
    )
    out.append(info)

    # fmt: off
    header = "Number and ratios of classes in splits:"
    split_cls_counts = best_segment_loss_details["class"]["split_counts"]
    split_cls_ratios = best_segment_loss_details["class"]["split_ratios"]
    out.append(log_table(header, classes, splits, split_cls_counts, split_cls_ratios, None, 15, True, "", "Class"))

    header = "Number and ratios of objects with corresponding distance levels in splits:"
    split_distance_counts = best_segment_loss_details["distance"]["split_counts"]
    split_distance_ratios = best_segment_loss_details["distance"]["split_ratios"]
    distance_levels = list(DISTANCE_LEVELS.keys())
    out.append(log_table(header, classes, splits, split_distance_counts, split_distance_ratios, distance_levels, 15, True, "", "Class"))

    header = "Number and ratios of objects with corresponding number of points in splits:"
    split_num_points_counts = best_segment_loss_details["num_points"]["split_counts"]
    split_num_points_ratios = best_segment_loss_details["num_points"]["split_ratios"]
    num_points_levels = list(NUM_POINTS_LEVELS.keys())
    out.append(log_table(header, classes, splits, split_num_points_counts, split_num_points_ratios, num_points_levels, 15, True, "", "Class"))

    header = "Number and ratios of objects with occlusion levels in splits:"
    split_occlusion_counts = best_segment_loss_details["occlusion"]["split_counts"]
    split_occlusion_ratios = best_segment_loss_details["occlusion"]["split_ratios"]
    occlusion_levels = OCCLUSION_LEVELS
    out.append(log_table(header, classes, splits, split_occlusion_counts, split_occlusion_ratios, occlusion_levels, 15, True, "", "Class"))
    # fmt: on

    logging.info("\n".join(out))


def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_) if s]


if __name__ == "__main__":
    args = get_args()

    os.makedirs(args.out_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(args.out_path, "script.log"),
        filemode="w",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        level=args.loglevel.upper(),
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(vars(args))

    np.random.seed(args.seed)

    segment_size = args.segment_size
    perm_limit = args.perm_limit
    root_path = args.root_path
    target_path = args.out_path
    point_cloud_range = [float(x) for x in args.point_cloud_range]
    splits = args.splits
    split_ratios = [float(x) for x in args.split_ratios]
    classes = (
        CLASSES
        if len(args.exclude_classes) == 0
        else [x for x in CLASSES if x not in args.exclude_classes]
    )

    assert len(splits) == len(split_ratios)

    assert os.path.exists(root_path)

    logging.info("Creating sequence details...")
    data = create_sequence_details(root_path, point_cloud_range)

    assert len(data) != 0
    for x in data.values():
        assert len(x) != 0, "No frames found in sequence"

    logging.info("Creating bucket details...")
    title = f"{'Token':<35} {'Sequence':<40} {'Frames':<10} {'No. Segment':<15} {'Segments'}"
    logging.info(title)
    logging.info(create_header_line(len(title), "="))

    bucket_details = {}
    for x, y in data.items():
        bucket_no_frames = data[x].no_total_frames
        if bucket_no_frames < segment_size:
            bucket_details[x] = {"count": 1, "segment_size": bucket_no_frames}
        elif bucket_no_frames % segment_size == 0:
            bucket_details[x] = {
                "count": bucket_no_frames // segment_size,
                "segment_size": segment_size,
            }
        else:
            rem = bucket_no_frames % segment_size
            if rem >= segment_size // 3:
                count = (bucket_no_frames // segment_size) + 1
                bucket_details[x] = {
                    "count": count,
                    "segment_size": segment_size,
                    "last_segment_size": rem,
                }
            else:
                # alternative first increasing then decreasing search
                found = False
                for change_offset in [-1, 1]:
                    tmp_segment_size = segment_size + change_offset
                    for _ in range(20):
                        if bucket_no_frames % tmp_segment_size == 0:
                            found = True
                            break
                        tmp_segment_size += change_offset

                if not found:
                    raise ValueError(
                        f"Could not find a segment size for {x} with {bucket_no_frames} frames"
                    )

                bucket_details[x] = {
                    "count": bucket_no_frames // tmp_segment_size,
                    "segment_size": tmp_segment_size,
                }

        # fmt: off
        if "last_segment_size" in bucket_details[x]:
            segment_list = [bucket_details[x]["segment_size"]] * (bucket_details[x]["count"] - 1) + [bucket_details[x]["last_segment_size"]]
        else:
            segment_list = [bucket_details[x]["segment_size"]] * bucket_details[x]["count"]

        logging.info(f"{x:<35} {data[x].sequence_name:<40} {data[x].no_total_frames:<10} {bucket_details[x]['count']:<15} {segment_list}")
        # fmt: on

    segment_list = []
    for x, y in bucket_details.items():
        if "last_segment_size" in y:
            for count in range(y["count"] - 1):
                start_idx = count * y["segment_size"]
                end_idx = count * y["segment_size"] + y["segment_size"]
                segment_list.append((x, (start_idx, end_idx)))
            start_idx = (y["count"] - 1) * y["segment_size"]
            end_idx = (y["count"] - 1) * y["segment_size"] + y["last_segment_size"]
            segment_list.append((x, (start_idx, end_idx)))
        else:
            for count in range(y["count"]):
                start_idx = count * y["segment_size"]
                end_idx = count * y["segment_size"] + y["segment_size"]
                segment_list.append((x, (start_idx, end_idx)))
    segment_list = np.asarray(segment_list)

    perm = permutations(segment_list, len(segment_list))

    perms = []
    while len(perms) < perm_limit:
        perms.append(list(segment_list[np.random.permutation(len(segment_list))]))

    logging.info(f"Total number of iterations: {len(perms)}")

    best_perm: List[Tuple[str, Tuple[int]]] = None
    best_loss: float = None
    best_segment_loss_details = {}

    manager = multiprocessing.Manager()
    lock = manager.Lock()
    queue = manager.Queue()

    def custom_callback(e):
        logging.error(f"{type(e).__name__} {__file__} {e}")

    with multiprocessing.Pool(args.p) as pool:
        n = len(perms) // args.p
        perms = [perms[i : i + n] for i in range(0, len(perms), n)]
        start_time = time.perf_counter()
        for position in range(args.p):  # remained perms are not processed
            pool.apply_async(
                search_task,
                [
                    position + 1,
                    lock,
                    queue,
                    perms[position],
                    data,
                    split_ratios,
                    classes,
                    CLASS_WEIGHTS,
                    DISTANCE_WEIGHTS,
                    NUM_POINTS_WEIGHTS,
                    OCCLUSION_WEIGHTS,
                    args.include_all_classes,
                    args.include_all_sequences,
                    args.include_same_classes_in_distance,
                    args.distance_th,
                    args.include_same_classes_in_num_points,
                    args.num_points_th,
                    args.include_same_classes_in_occlusion,
                    args.occlusion_th,
                ],
                error_callback=custom_callback,
            )
            time.sleep(2)

        pool.close()
        pool.join()

    end_time = time.perf_counter()

    total_found = 0
    total_filter_stats = defaultdict(int)
    while not queue.empty():
        n_found, filter_stats, segment_loss_details = queue.get()
        total_found += n_found
        loss = segment_loss_details["loss"]

        for x, y in filter_stats.items():
            total_filter_stats[x] += y

        if loss is not None and (best_loss is None or loss < best_loss):
            best_loss = loss
            best_segment_loss_details = segment_loss_details
            best_perm = segment_loss_details["perm"]

    time.sleep(1)
    logging.info(f"Elapsed time: {end_time - start_time}")
    if len(total_filter_stats) != 0:
        logging.info("Filter reason counts: ")
        for x, y in total_filter_stats.items():
            logging.info(f"\t {x:<30} : {y:<15}")

    if len(best_segment_loss_details) == 0:
        logging.error("Could not find a split that satisfies the constraints...")
        exit()

    logging.info(f"Found total: {total_found} out of {perm_limit}")

    log_summary(best_segment_loss_details, splits, classes)

    logging.info("Saving original split details...")
    save_split_details_and_config(
        target_path,
        bucket_details,
        data,
        best_perm,
        best_segment_loss_details,
        splits,
        split_ratios,
    )

    if args.create:
        logging.info("Creating split folders and copying split files...")
        create_and_copy_split(
            target_path,
            data,
            best_perm,
            best_segment_loss_details["split_idx_ranges"],
            splits,
            args.p,
        )
