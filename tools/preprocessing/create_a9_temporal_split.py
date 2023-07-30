import json
import logging
import multiprocessing
import os
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

warnings.filterwarnings("ignore")

CLASSES = [
    "CAR",
    "TRUCK",
    "TRAILER",
    "BUS",
    "VAN",
    "BICYCLE",
    "MOTORCYCLE",
    "PEDESTRIAN",
    "EMERGENCY_VEHICLE",
    "OTHER",
]

CLASS_WEIGHTS = np.asarray([5, 12, 12, 5, 5, 15, 15, 20, 5, 5]) * 10
DIFFICULTY_WEIGHTS = np.asarray([5, 12, 12, 5, 5, 15, 15, 20, 5, 5]) * 4
NUM_POINTS_WEIGHTS = np.asarray([5, 12, 12, 5, 5, 15, 15, 20, 5, 5]) * 2
OCCLUSION_WEIGHTS = np.asarray([5, 12, 12, 5, 5, 15, 15, 20, 5, 5]) * 1
DISTANCE_WEIGHTS = np.asarray([5, 12, 12, 5, 5, 15, 15, 20, 5, 5]) * 4

DIFFICULTY_MAP = {
    "easy": {"distance": "d<40", "num_points": "n>50", "occlusion": "no_occluded"},
    "moderate": {
        "distance": "d40-50",
        "num_points": "n20-50",
        "occlusion": "partially_occluded",
    },
    "hard": {"distance": "d>50", "num_points": "n<20", "occlusion": "mostly_occluded"},
}
DISTANCE_MAP = {"d<40": [0, 40], "d40-50": [40, 50], "d>50": [50, 64]}
NUM_POINTS_MAP = {"n<20": [5, 20], "n20-50": [20, 50], "n>50": [50, 9999]}
OCCLUSION_MAP = {
    "no_occluded": "NOT_OCCLUDED",
    "partially_occluded": "PARTIALLY_OCCLUDED",
    "mostly_occluded": "MOSTLY_OCCLUDED",
    "unknown": "UNKNOWN",
}

SPLIT_RATIO = [0.8, 0.1, 0.1]

parent_img_folder_name = "images"
parent_img_labels_folder_name = "labels_images"
parent_pcd_folder_name = "point_clouds"
parent_pcd_labels_folder_name = "labels_point_clouds"

img_s1_folder_name = "s110_camera_basler_south1_8mm"
img_s2_folder_name = "s110_camera_basler_south2_8mm"
pcd_s_folder_name = "s110_lidar_ouster_south"
pcd_n_folder_name = "s110_lidar_ouster_north"


def get_args() -> Namespace:
    """
    Parse given arguments for find_and_create_a9_temporal_split function.

    Returns:
        Namespace: parsed arguments
    """
    parser = ArgumentParser()

    parser.add_argument("--root-path", type=str, required=True)
    parser.add_argument("--out-path", type=str, required=True)
    parser.add_argument("--segment-size", type=int, default=30, required=False)
    parser.add_argument("--perm-limit", type=int, default=2e4, required=False)
    parser.add_argument("-p", type=int, default=4, required=False)
    parser.add_argument("--include-all-classes", default=False, action="store_true")
    parser.add_argument("--include-all-sequences", default=False, action="store_true")
    parser.add_argument("--include-same-classes-in-difficulty", default=False, action="store_true")
    parser.add_argument("--difficulty-th", type=float, default=0.5, required=False)
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
    no_total_frames: int
    total_bboxes: int
    total_bboxes_filtered: int
    total_class_stats: Dict[str, int] = field(repr=True, compare=False)
    total_difficulty_stats: Dict[str, Dict[str, int]] = field(repr=False, compare=False)
    total_distance_stats: Dict[str, Dict[str, int]] = field(repr=False, compare=False)
    total_num_points_stats: Dict[str, Dict[str, int]] = field(repr=False, compare=False)
    total_occlusion_stats: Dict[str, Dict[str, int]] = field(repr=False, compare=False)
    frame_class_stats: List[Dict[str, int]] = field(default_factory=list, repr=False, compare=False)
    frame_difficulty_stats: List[Dict[str, Dict[str, int]]] = field(default_factory=list, repr=False, compare=False)
    frame_distance_stats: List[Dict[str, Dict[str, int]]] = field(default_factory=list, repr=False, compare=False)
    frame_num_points_stats: List[Dict[str, Dict[str, int]]] = field(default_factory=list, repr=False, compare=False)
    frame_occlusion_stats: List[Dict[str, Dict[str, int]]] = field(default_factory=list, repr=False, compare=False)
    frame_img_s1_paths: List[str] = field(default_factory=list, repr=False, compare=False)
    frame_img_s2_paths: List[str] = field(default_factory=list, repr=False, compare=False)
    frame_img_s1_label_paths: List[str] = field(default_factory=list, repr=False, compare=False)
    frame_img_s2_label_paths: List[str] = field(default_factory=list, repr=False, compare=False)
    frame_pcd_paths: List[str] = field(default_factory=list, repr=False, compare=False)
    frame_pcd_labels_paths: List[str] = field(default_factory=list, repr=False, compare=False)
    # fmt: on

    def __len__(self):
        return self.total_bboxes

    def get_class_stats_in_range(self, start: int, end: int) -> Dict[str, int]:
        return {cls: sum([x[cls] for x in self.frame_class_stats[start:end]]) for cls in CLASSES}

    def get_difficulty_stats_in_range(self, start: int, end: int) -> Dict[str, int]:
        return {
            cls: {
                difficulty: sum(
                    [x[cls][difficulty] for x in self.frame_difficulty_stats[start:end]]
                )
                for difficulty in DIFFICULTY_MAP.keys()
            }
            for cls in CLASSES
        }

    def get_distance_stats_in_range(self, start: int, end: int) -> Dict[str, int]:
        return {
            cls: {
                distance: sum([x[cls][distance] for x in self.frame_distance_stats[start:end]])
                for distance in DISTANCE_MAP.keys()
            }
            for cls in CLASSES
        }

    def get_num_points_stats_in_range(self, start: int, end: int) -> Dict[str, int]:
        return {
            cls: {
                num_points: sum(
                    [x[cls][num_points] for x in self.frame_num_points_stats[start:end]]
                )
                for num_points in NUM_POINTS_MAP.keys()
            }
            for cls in CLASSES
        }

    def get_occlusion_stats_in_range(self, start: int, end: int) -> Dict[str, int]:
        return {
            cls: {
                occlusion: sum([x[cls][occlusion] for x in self.frame_occlusion_stats[start:end]])
                for occlusion in OCCLUSION_MAP.values()
            }
            for cls in CLASSES
        }

    def get_path_list_in_range(self, start: int, end: int) -> Dict[str, List[str]]:
        return {
            "frame_img_s1_paths": self.frame_img_s1_paths[start:end],
            "frame_img_s2_paths": self.frame_img_s2_paths[start:end],
            "frame_img_s1_label_paths": self.frame_img_s1_label_paths[start:end],
            "frame_img_s2_label_paths": self.frame_img_s2_label_paths[start:end],
            "frame_pcd_paths": self.frame_pcd_paths[start:end],
            "frame_pcd_labels_paths": self.frame_pcd_labels_paths[start:end],
        }


def create_sequence_details(
    root_path: str, point_cloud_range: List[float] = []
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

    # fmt: off
    imgs_s1 = sorted(glob(os.path.join(root_path, parent_img_folder_name, img_s1_folder_name, "*")))
    imgs_s2 = sorted(glob(os.path.join(root_path, parent_img_folder_name, img_s2_folder_name, "*")))
    imgs_s1_labels = sorted(glob(os.path.join(root_path, parent_img_labels_folder_name, img_s1_folder_name, "*.json")))
    imgs_s2_labels = sorted(glob(os.path.join(root_path, parent_img_labels_folder_name, img_s2_folder_name, "*.json")))
    pcds = sorted(glob(os.path.join(root_path, parent_pcd_folder_name, pcd_s_folder_name, "*.pcd")))
    pcds_labels = sorted(glob(os.path.join(root_path, parent_pcd_labels_folder_name, pcd_s_folder_name, "*.json")))
    assert len(imgs_s1) == len(imgs_s2) == len(imgs_s1_labels) == len(imgs_s2_labels) == len(pcds) == len(pcds_labels)
    # fmt: on

    for i, lp in enumerate(tqdm(pcds_labels, desc="reading")):
        with open(lp, "r") as f:
            json_data = json.load(f)
            frame_idx = list(json_data["openlabel"]["frames"].keys())[0]
            frame_properties = json_data["openlabel"]["frames"][frame_idx]["frame_properties"]
            frame_objects = json_data["openlabel"]["frames"][frame_idx]["objects"]

            scene_token = frame_properties["scene_token"]
            if scene_token not in data:
                data[scene_token] = TemporalSequenceDetails(
                    scene_token=scene_token,
                    no_total_frames=0,
                    total_bboxes=0,
                    total_bboxes_filtered=0,
                    total_class_stats={cls: 0 for cls in CLASSES},
                    total_difficulty_stats={
                        cls: {x: 0 for x in DIFFICULTY_MAP.keys()} for cls in CLASSES
                    },
                    total_distance_stats={
                        cls: {x: 0 for x in DISTANCE_MAP.keys()} for cls in CLASSES
                    },
                    total_num_points_stats={
                        cls: {x: 0 for x in NUM_POINTS_MAP.keys()} for cls in CLASSES
                    },
                    total_occlusion_stats={
                        cls: {x: 0 for x in OCCLUSION_MAP.values()} for cls in CLASSES
                    },
                )

            class_stats = {cls: 0 for cls in CLASSES}
            difficulty_stats = {cls: {x: 0 for x in DIFFICULTY_MAP.keys()} for cls in CLASSES}
            distance_stats = {cls: {x: 0 for x in DISTANCE_MAP.keys()} for cls in CLASSES}
            num_points_stats = {cls: {x: 0 for x in NUM_POINTS_MAP.keys()} for cls in CLASSES}
            occlusion_stats = {cls: {x: 0 for x in OCCLUSION_MAP.values()} for cls in CLASSES}
            for obj in frame_objects.values():
                obj_type = obj["object_data"]["type"]
                if "cuboid" in obj["object_data"]:
                    loc = np.asarray(obj["object_data"]["cuboid"]["val"][:3], dtype=np.float32)

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

                    # distance from sensor
                    distance = np.sqrt(np.sum(np.array(loc[:2]) ** 2))
                    for k, v in DISTANCE_MAP.items():
                        if v[0] < distance <= v[1]:
                            data[scene_token].total_distance_stats[obj_type][k] += 1
                            distance_stats[obj_type][k] += 1

                    attributes = obj["object_data"]["cuboid"]["attributes"]

                    # difficulty and occlusion
                    difficulty = None
                    occlusion_level = None
                    for x in attributes["text"]:
                        if x["name"] == "difficulty":
                            difficulty = x["val"]
                            data[scene_token].total_difficulty_stats[obj_type][difficulty] += 1
                            difficulty_stats[obj_type][difficulty] += 1
                        elif x["name"] == "occlusion_level":
                            occlusion_level = x["val"]
                            if occlusion_level != "":
                                data[scene_token].total_occlusion_stats[obj_type][
                                    occlusion_level
                                ] += 1
                                occlusion_stats[obj_type][occlusion_level] += 1
                            else:
                                data[scene_token].total_occlusion_stats[obj_type]["UNKNOWN"] += 1
                                occlusion_stats[obj_type]["UNKNOWN"] += 1

                    # number of points
                    num_points = 0
                    for x in attributes["num"]:
                        if x["name"] == "num_points":
                            num_points = x["val"]
                            for k, v in NUM_POINTS_MAP.items():
                                if v[0] < num_points <= v[1]:
                                    data[scene_token].total_num_points_stats[obj_type][k] += 1
                                    num_points_stats[obj_type][k] += 1

                    class_stats[obj_type] += 1

            data[scene_token].frame_difficulty_stats.append(difficulty_stats)
            data[scene_token].frame_distance_stats.append(distance_stats)
            data[scene_token].frame_num_points_stats.append(num_points_stats)
            data[scene_token].frame_occlusion_stats.append(occlusion_stats)

            data[scene_token].no_total_frames += 1
            data[scene_token].frame_class_stats.append(class_stats)
            for x in class_stats:
                data[scene_token].total_class_stats[x] += class_stats[x]

            data[scene_token].frame_img_s1_paths.append(imgs_s1[i])
            data[scene_token].frame_img_s2_paths.append(imgs_s2[i])
            data[scene_token].frame_img_s1_label_paths.append(imgs_s1_labels[i])
            data[scene_token].frame_img_s2_label_paths.append(imgs_s2_labels[i])
            data[scene_token].frame_pcd_paths.append(pcds[i])
            data[scene_token].frame_pcd_labels_paths.append(pcds_labels[i])

    for x, y in data.items():
        tot = y.total_bboxes + y.total_bboxes_filtered
        logging.info(
            f"Sequence {x} : filtered {y.total_bboxes_filtered:<5} ({y.total_bboxes_filtered/tot:.3f}) - remained {y.total_bboxes:<5} ({y.total_bboxes/tot:.3f}) - total {tot}"
        )

    return data


def calculate_segment_loss(
    perm: List[Tuple[str, Tuple[int, int]]],
    data: Dict[str, TemporalSequenceDetails],
    split_ratio: List[float] = [0.8, 0.1, 0.1],
    classes: List[str] = CLASSES,
    class_weights: Optional[List[float]] = None,
    difficulty_weights: Optional[List[float]] = None,
    distance_weights: Optional[List[float]] = None,
    num_points_weights: Optional[List[float]] = None,
    occlusion_weights: Optional[List[float]] = None,
    include_all_classes: bool = False,
    include_all_sequences: bool = False,
    include_same_classes_in_difficulty: bool = False,
    difficulty_threshold_ratio: float = 1.0,
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
        include_same_classes_in_difficulty: whether to force to have same classes in all splits with given difficulty threshold
        difficulty_threshold_ratio: difficulty threshold ratio
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
            (split_ratio[0] * len(perm)),
            ((split_ratio[0] + split_ratio[1]) * len(perm)),
            len(perm),
        ]
    elif split_count == 2:
        split_idx_ranges = [
            0,
            (split_ratio[0] * len(perm)),
            len(perm),
        ]
    else:
        raise Exception("Number of splits must be either 2 or 3")

    assert all(
        [float(x).is_integer() for x in split_idx_ranges]
    ), "segment_size and split_ratio results in non-integer split_idx_ranges"

    seq_cls_assigned_split = [0] * int(len(perm) * split_ratio[0])
    seq_cls_assigned_split.extend([1] * int(len(perm) * split_ratio[1]))
    if split_count == 3:  # train, val, test
        seq_cls_assigned_split.extend([2] * int(len(perm) * split_ratio[2]))

    split_cls_counts = [{cls: 0 for cls in classes} for _ in range(len(split_ratio))]
    split_occlusion_counts = [
        {cls: {x: 0 for x in OCCLUSION_MAP.values()} for cls in classes}
        for _ in range(len(split_ratio))
    ]
    split_difficulty_counts = [
        {cls: {x: 0 for x in DIFFICULTY_MAP.keys()} for cls in classes}
        for _ in range(len(split_ratio))
    ]
    split_num_points_counts = [
        {cls: {x: 0 for x in NUM_POINTS_MAP.keys()} for cls in classes}
        for _ in range(len(split_ratio))
    ]
    split_distance_counts = [
        {cls: {x: 0 for x in DISTANCE_MAP.keys()} for cls in classes}
        for _ in range(len(split_ratio))
    ]

    class_counts = []
    difficulty_counts = []
    occlusion_counts = []
    num_points_counts = []
    distance_counts = []

    for x in perm:
        token = x[0]
        start_idx = x[1][0]
        end_idx = x[1][1]

        class_counts.append(data[token].get_class_stats_in_range(start_idx, end_idx))
        difficulty_counts.append(data[token].get_difficulty_stats_in_range(start_idx, end_idx))
        occlusion_counts.append(data[token].get_occlusion_stats_in_range(start_idx, end_idx))
        num_points_counts.append(data[token].get_num_points_stats_in_range(start_idx, end_idx))
        distance_counts.append(data[token].get_distance_stats_in_range(start_idx, end_idx))

    for idx, (cls_c, diff_c, occ_c, np_c, d_c) in enumerate(
        zip(class_counts, difficulty_counts, occlusion_counts, num_points_counts, distance_counts)
    ):
        if seq_cls_assigned_split[idx] == 0:  # in train split
            for cls in classes:
                split_cls_counts[0][cls] += cls_c[cls]
                for occl in OCCLUSION_MAP.values():
                    split_occlusion_counts[0][cls][occl] += occ_c[cls][occl]
                for diff in DIFFICULTY_MAP.keys():
                    split_difficulty_counts[0][cls][diff] += diff_c[cls][diff]
                for nump in NUM_POINTS_MAP.keys():
                    split_num_points_counts[0][cls][nump] += np_c[cls][nump]
                for d in DISTANCE_MAP.keys():
                    split_distance_counts[0][cls][d] += d_c[cls][d]
        elif seq_cls_assigned_split[idx] == 1:  # in val split
            for cls in classes:
                split_cls_counts[1][cls] += cls_c[cls]
                for occl in OCCLUSION_MAP.values():
                    split_occlusion_counts[1][cls][occl] += occ_c[cls][occl]
                for diff in DIFFICULTY_MAP.keys():
                    split_difficulty_counts[1][cls][diff] += diff_c[cls][diff]
                for nump in NUM_POINTS_MAP.keys():
                    split_num_points_counts[1][cls][nump] += np_c[cls][nump]
                for d in DISTANCE_MAP.keys():
                    split_distance_counts[1][cls][d] += d_c[cls][d]
        else:  # in test split
            for cls in classes:
                split_cls_counts[2][cls] += cls_c[cls]
                for occl in OCCLUSION_MAP.values():
                    split_occlusion_counts[2][cls][occl] += occ_c[cls][occl]
                for diff in DIFFICULTY_MAP.keys():
                    split_difficulty_counts[2][cls][diff] += diff_c[cls][diff]
                for nump in NUM_POINTS_MAP.keys():
                    split_num_points_counts[2][cls][nump] += np_c[cls][nump]
                for d in DISTANCE_MAP.keys():
                    split_distance_counts[2][cls][d] += d_c[cls][d]

    # calculate the distributions in splits
    split_cls_ratios = []
    split_difficulty_ratios = []
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
        total_split_difficulty_counts = [
            sum([split_difficulty_counts[i][cls][diff] for cls in classes])
            for diff in DIFFICULTY_MAP.keys()
        ]
        split_difficulty_ratios.append(
            {
                cls: {
                    diff: split_difficulty_counts[i][cls][diff] / total_split_difficulty_counts[j]
                    if total_split_difficulty_counts[j] != 0
                    else 0
                    for j, diff in enumerate(DIFFICULTY_MAP.keys())
                }
                for cls in classes
            }
        )

        # force each split to have same classes present with given thresholds
    if include_same_classes_in_difficulty:
        class_presences = [
            {diff: {cls: False for cls in classes} for diff in DIFFICULTY_MAP.keys()}
            for _ in range(split_count)
        ]
        for split_idx, x in enumerate(split_difficulty_counts):
            for cls, diffs in x.items():
                for diff, v in diffs.items():
                    if v > 0:
                        class_presences[split_idx][diff][cls] = True
        tot = 0
        for x in class_presences[1:]:
            for id, clss in x.items():
                for cls, cls_b in clss.items():
                    if cls_b == class_presences[0][id][cls]:
                        tot += 1
        th = len(classes) * (split_count - 1) * len(DIFFICULTY_MAP.keys())
        if th * difficulty_threshold_ratio > tot:
            return {"loss": None, "reason": "difficulty-threshold"}

    for i in range(len(split_ratio)):
        total_split_occlusion_counts = [
            sum([split_occlusion_counts[i][cls][occl] for cls in classes])
            for occl in OCCLUSION_MAP.values()
        ]
        split_occlusion_ratios.append(
            {
                cls: {
                    occl: split_occlusion_counts[i][cls][occl] / total_split_occlusion_counts[j]
                    if total_split_occlusion_counts[j] != 0
                    else 0
                    for j, occl in enumerate(OCCLUSION_MAP.values())
                }
                for cls in classes
            }
        )

    if include_same_classes_in_occlusion:
        class_presences = [
            {o: {cls: False for cls in classes} for o in OCCLUSION_MAP.values()}
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
        th = len(classes) * (split_count - 1) * len(OCCLUSION_MAP.values())
        if th * occlusion_threshold_ratio > tot:
            return {"loss": None, "reason": "occlusion-threshold"}

    for i in range(len(split_ratio)):
        total_split_num_points_counts = [
            sum([split_num_points_counts[i][cls][nump] for cls in classes])
            for nump in NUM_POINTS_MAP.keys()
        ]
        split_num_points_ratios.append(
            {
                cls: {
                    nump: split_num_points_counts[i][cls][nump] / total_split_num_points_counts[j]
                    if total_split_num_points_counts[j] != 0
                    else 0
                    for j, nump in enumerate(NUM_POINTS_MAP.keys())
                }
                for cls in classes
            }
        )

    if include_same_classes_in_num_points:
        class_presences = [
            {p: {cls: False for cls in classes} for p in NUM_POINTS_MAP.keys()}
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
        th = len(classes) * (split_count - 1) * len(NUM_POINTS_MAP.keys())
        if th * num_points_threshold_ratio > tot:
            return {"loss": None, "reason": "num_points-threshold"}

    for i in range(len(split_ratio)):
        total_split_distance_counts = [
            sum([split_distance_counts[i][cls][d] for cls in classes]) for d in DISTANCE_MAP.keys()
        ]
        split_distance_ratios.append(
            {
                cls: {
                    d: split_distance_counts[i][cls][d] / total_split_distance_counts[j]
                    if total_split_distance_counts[j] != 0
                    else 0
                    for j, d in enumerate(DISTANCE_MAP.keys())
                }
                for cls in classes
            }
        )

    if include_same_classes_in_distance:
        class_presences = [
            {d: {cls: False for cls in classes} for d in DISTANCE_MAP.keys()}
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
        th = len(classes) * (split_count - 1) * len(DISTANCE_MAP.keys())
        if th * distance_threshold_ratio > tot:
            return {"loss": None, "reason": "distance-threshold"}

    # compare the class distributions between splits
    total_cls_loss = 0
    for i in range(len(split_cls_ratios) - 1):
        for j, cls in enumerate(classes):
            if class_weights is None:
                total_cls_loss += abs(split_cls_ratios[i][cls] - split_cls_ratios[i + 1][cls])
            else:
                total_cls_loss += (
                    abs(split_cls_ratios[i][cls] - split_cls_ratios[i + 1][cls]) * class_weights[j]
                )
    loss += total_cls_loss

    # compare the difficulty distributions
    total_diff_loss = 0
    for i in range(len(split_cls_ratios) - 1):
        for j in DIFFICULTY_MAP.keys():
            for k, cls in enumerate(classes):
                if difficulty_weights is None:
                    total_diff_loss += abs(
                        split_difficulty_ratios[i][cls][j] - split_difficulty_ratios[i + 1][cls][j]
                    )
                else:
                    total_diff_loss += (
                        abs(
                            split_difficulty_ratios[i][cls][j]
                            - split_difficulty_ratios[i + 1][cls][j]
                        )
                        * difficulty_weights[k]
                    )

    loss += total_diff_loss

    # compare the occlusion distributions
    total_occlusion_loss = 0
    for i in range(len(split_cls_ratios) - 1):
        for j in OCCLUSION_MAP.values():
            for k, cls in enumerate(classes):
                if occlusion_weights is None:
                    total_occlusion_loss += abs(
                        split_occlusion_ratios[i][cls][j] - split_occlusion_ratios[i + 1][cls][j]
                    )
                else:
                    total_occlusion_loss += (
                        abs(
                            split_occlusion_ratios[i][cls][j]
                            - split_occlusion_ratios[i + 1][cls][j]
                        )
                        * occlusion_weights[k]
                    )

    loss += total_occlusion_loss

    # compare the number of points distributions
    total_num_points_loss = 0
    for i in range(len(split_cls_ratios) - 1):
        for j in NUM_POINTS_MAP.keys():
            for k, cls in enumerate(classes):
                if num_points_weights is None:
                    total_num_points_loss += abs(
                        split_num_points_ratios[i][cls][j] - split_num_points_ratios[i + 1][cls][j]
                    )
                else:
                    total_num_points_loss += (
                        abs(
                            split_num_points_ratios[i][cls][j]
                            - split_num_points_ratios[i + 1][cls][j]
                        )
                        * num_points_weights[k]
                    )

    loss += total_num_points_loss

    # compare the distance distributions
    total_distance_loss = 0
    for i in range(len(split_cls_ratios) - 1):
        for j in DISTANCE_MAP.keys():
            for k, cls in enumerate(classes):
                if distance_weights is None:
                    total_distance_loss += abs(
                        split_distance_ratios[i][cls][j] - split_distance_ratios[i + 1][cls][j]
                    )
                else:
                    total_distance_loss += (
                        abs(split_distance_ratios[i][cls][j] - split_distance_ratios[i + 1][cls][j])
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
        "difficulty": {
            "loss": total_diff_loss,
            "split_counts": split_difficulty_counts,
            "split_ratios": split_difficulty_ratios,
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


def create_and_copy_split(
    target_path: str,
    data: Dict[str, TemporalSequenceDetails],
    perm: List[Tuple[str, Tuple[int]]],
    split_idx_ranges: List[float],
    splits: List[str] = ["train", "val", "test"],
) -> None:
    """
    Create and copy the split to the target path.

    Args:
        target_path: target path to copy the split
        data: dictionary of sequence details
        perm: selected permutation of segments
        split_idx_ranges: split index ranges
        splits: list of splits
    """
    split_source_paths = []

    for x in perm:
        # get class counts for current segment
        token = x[0]
        start_idx = x[1][0]
        end_idx = x[1][1]
        split_source_paths.append(data[token].get_path_list_in_range(start_idx, end_idx))

    target_paths = {}
    for split in splits:
        # fmt: off
        os.makedirs(os.path.join(target_path, split, parent_img_folder_name, img_s1_folder_name), exist_ok=True)
        os.makedirs(os.path.join(target_path, split, parent_img_folder_name, img_s2_folder_name), exist_ok=True)
        os.makedirs(os.path.join(target_path, split, parent_img_labels_folder_name, img_s1_folder_name), exist_ok=True)
        os.makedirs(os.path.join(target_path, split, parent_img_labels_folder_name, img_s2_folder_name), exist_ok=True)
        os.makedirs(os.path.join(target_path, split, parent_pcd_folder_name, pcd_s_folder_name), exist_ok=True)
        os.makedirs(os.path.join(target_path, split, parent_pcd_labels_folder_name, pcd_s_folder_name), exist_ok=True)
        target_paths[split] = {
            "target_img_s1" : os.path.join(target_path, split, parent_img_folder_name, img_s1_folder_name),
            "target_img_s2" : os.path.join(target_path, split, parent_img_folder_name, img_s2_folder_name),
            "target_img_s1_label" : os.path.join(target_path, split, parent_img_labels_folder_name, img_s1_folder_name),
            "target_img_s2_label" : os.path.join(target_path, split, parent_img_labels_folder_name, img_s2_folder_name),
            "target_pcd" : os.path.join(target_path, split, parent_pcd_folder_name, pcd_s_folder_name),
            "target_pcd_label" : os.path.join(target_path, split, parent_pcd_labels_folder_name, pcd_s_folder_name),
        }
        # fmt: on

    for idx, source_paths in enumerate(split_source_paths):
        if split_idx_ranges[0] <= idx < split_idx_ranges[1]:
            split = splits[0]
        elif split_idx_ranges[1] <= idx < split_idx_ranges[2]:
            split = splits[1]
        else:
            split = splits[2]

        for source_name, paths in source_paths.items():
            target_root_path = None
            if source_name == "frame_img_s1_paths":
                target_root_path = target_paths[split]["target_img_s1"]
            elif source_name == "frame_img_s2_paths":
                target_root_path = target_paths[split]["target_img_s2"]
            elif source_name == "frame_img_s1_label_paths":
                target_root_path = target_paths[split]["target_img_s1_label"]
            elif source_name == "frame_img_s2_label_paths":
                target_root_path = target_paths[split]["target_img_s2_label"]
            elif source_name == "frame_pcd_paths":
                target_root_path = target_paths[split]["target_pcd"]
            elif source_name == "frame_pcd_labels_paths":
                target_root_path = target_paths[split]["target_pcd_label"]

            for path in paths:
                shutil.copy2(path, target_root_path)


def save_original_split_details(
    target_path: str,
    segment_size: int,
    perm: List[Tuple[str, Tuple[int]]],
    best_segment_loss_details,
    splits: List[str] = ["train", "val", "test"],
    split_ratio: List[int] = [0.8, 0.1, 0.1],
) -> None:
    """
    Save the original split details.

    Args:
        target_path: target path to copy the split
        segment_size: number of frames in a segment
        perm: selected permutation of segments
        best_segment_loss_details: best segment loss details
        splits: list of splits
        split_ratio: ratio of splits
    """
    orig_split_details = {}
    split_idx_ranges = best_segment_loss_details["split_idx_ranges"]

    for i, split in enumerate(splits):
        orig_split_details[split] = {}
        orig_split_details[split]["segment_size"] = segment_size
        orig_split_details[split]["no_frames"] = segment_size * len(
            perm[int(split_idx_ranges[i]) : int(split_idx_ranges[i + 1])]
        )
        orig_split_details[split]["original_sequences"] = {}
        for j, seq in enumerate(data.keys()):
            orig_split_details[split]["original_sequences"][seq] = (
                np.asarray(perm)[int(split_idx_ranges[i]) : int(split_idx_ranges[i + 1])] == seq
            ).sum()

        for x in ["class", "difficulty", "distance", "num_points", "occlusion"]:
            orig_split_details[split][x] = {
                "split_counts": best_segment_loss_details[x]["split_counts"][i],
                "split_ratios": best_segment_loss_details[x]["split_ratios"][i],
            }
    seq_assigned_split = [splits[0]] * int(len(perm) * split_ratio[0])
    seq_assigned_split.extend([splits[1]] * int(len(perm) * split_ratio[1]))
    if len(splits) == 3:
        seq_assigned_split.extend([splits[2]] * int(len(perm) * split_ratio[2]))
    orig_split_details["perm_details"] = [
        {x[0]: {"start": x[1][0], "end": x[1][1], "split": seq_assigned_split[i]}}
        for i, x in enumerate(perm)
    ]

    orig_split_details["losses"] = {"total": best_segment_loss_details["loss"]}
    for x in ["class", "difficulty", "distance", "num_points", "occlusion"]:
        orig_split_details["losses"][x] = best_segment_loss_details[x]["loss"]

    orig_split_details["perm"] = best_segment_loss_details["perm"]

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    with open(os.path.join(target_path, "orig_split_details.json"), "w") as f:
        json.dump(orig_split_details, f, indent=4, cls=NpEncoder)


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

    split_idx_ranges = best_segment_loss_details["split_idx_ranges"]

    # logging.info split statistics
    logging.info("=" * 172)
    logging.info(f"Best Split Summary w/ loss : {best_segment_loss_details['loss']}")
    logging.info(f"- Class loss: {best_segment_loss_details['class']['loss']}")
    logging.info(f"- Difficulty loss: {best_segment_loss_details['difficulty']['loss']}")
    logging.info(f"- Distance loss: {best_segment_loss_details['distance']['loss']}")
    logging.info(f"- No. Points loss: {best_segment_loss_details['num_points']['loss']}")
    logging.info(f"- Occlusion loss: {best_segment_loss_details['occlusion']['loss']}")
    logging.info("=" * 172)

    logging.info("")
    logging.info("=" * 172)
    logging.info("Number of segments and frames in corresponding sequences: ")
    title = f"{'Sequence':<40}" + "".join([f" {splits[i]:<15}" for i in range(len(splits))])
    logging.info(title)
    logging.info("-" * (40 + 15 * len(splits)))
    total_seq_counts = np.asarray([0] * len(splits), dtype=np.int32)
    for j, seq in enumerate(data.keys()):
        seq_counts = np.asarray(
            [
                (
                    np.asarray(best_perm)[int(split_idx_ranges[x]) : int(split_idx_ranges[x + 1])]
                    == seq
                ).sum()
                for x in range(len(splits))
            ],
            dtype=np.int32,
        )
        info = f"{seq:<40}" + "".join([f" {seq_counts[i]:<15}" for i in range(len(splits))])
        logging.info(info)
        total_seq_counts += seq_counts

    info = f"{'Total':<40}" + "".join(
        [
            f" {total_seq_counts[i]:<5} ({total_seq_counts[i] * segment_size:<5})"
            for i in range(len(splits))
        ]
    )
    logging.info(info)

    split_cls_counts = best_segment_loss_details["class"]["split_counts"]
    split_cls_ratios = best_segment_loss_details["class"]["split_ratios"]

    logging.info("")
    logging.info("=" * 172)
    logging.info("Number and ratios of classes in splits: ")
    logging.info("=" * 172)

    title = f"{'Sequence':<20}" + "".join([f" {splits[i]:<15}" for i in range(len(splits))])
    logging.info(title)
    logging.info("-" * (20 + 15 * len(splits)))
    for j, cls in enumerate(classes):
        info = f"{cls:<20}" + "".join(
            [
                f" {split_cls_counts[i][cls]:<5} ({split_cls_ratios[i][cls]:.3f})"
                for i in range(len(splits))
            ]
        )
        logging.info(info)

    tot = [sum([sum([split_cls_counts[i][c]]) for c in classes]) for i in range(len(splits))]
    logging.info("-" * (20 + 15 * len(splits)))
    info = f"{'Total':<20}" + "".join(  # type: ignore
        [
            f" {sum([x for x in split_cls_counts[i].values()]):<5} ({tot[i]/sum(tot):.3f})"
            for i in range(len(splits))
        ]
    )
    logging.info(info)

    split_diff_counts = best_segment_loss_details["difficulty"]["split_counts"]
    split_diff_ratios = best_segment_loss_details["difficulty"]["split_ratios"]

    logging.info("")
    logging.info("=" * 172)
    logging.info("Number and ratios of objects with corresponding difficulty levels in splits: ")
    logging.info("=" * 172)

    if len(splits) == 3:
        logging.info(
            "{:<20} | {:<15} {:<15} {:<15} | {:<15} {:<15} {:<15} | {:<15} {:<15} {:<15} |".format(
                "", "", "Easy", "", "", "Moderate", "", "", "Hard", ""
            )
        )
    else:
        logging.info(
            "{:<20} | {:<15} {:<15} | {:<15} {:<15} | {:<15} {:<15} |".format(
                "", "Easy", "", "Moderate", "", "Hard", ""
            )
        )

    title = f"{'Class':<20} |" + "".join(
        [
            f" {splits[0]:<15} {splits[1]:<15} {splits[2]:<15} |"
            if len(splits) == 3
            else f" {splits[0]:<15} {splits[1]:<15} |"
            for _ in range(len(DIFFICULTY_MAP.keys()))
        ]
    )
    logging.info(title)
    logging.info("-" * (20 + (15 * len(DIFFICULTY_MAP.keys()) * len(splits))))  # 172
    for j, cls in enumerate(classes):
        if len(splits) == 3:
            logging.info(
                "{:<20} | {:<15} {:<15} {:<15} | {:<15} {:<15} {:<15} | {:<15} {:<15} {:<15} |".format(
                    cls,
                    f"{split_diff_counts[0][cls]['easy']:<5} ({split_diff_ratios[0][cls]['easy']:.3f})",
                    f"{split_diff_counts[1][cls]['easy']:<5} ({split_diff_ratios[1][cls]['easy']:.3f})",
                    f"{split_diff_counts[2][cls]['easy']:<5} ({split_diff_ratios[2][cls]['easy']:.3f})",
                    f"{split_diff_counts[0][cls]['moderate']:<5} ({split_diff_ratios[0][cls]['moderate']:.3f})",
                    f"{split_diff_counts[1][cls]['moderate']:<5} ({split_diff_ratios[1][cls]['moderate']:.3f})",
                    f"{split_diff_counts[2][cls]['moderate']:<5} ({split_diff_ratios[2][cls]['moderate']:.3f})",
                    f"{split_diff_counts[0][cls]['hard']:<5} ({split_diff_ratios[0][cls]['hard']:.3f})",
                    f"{split_diff_counts[1][cls]['hard']:<5} ({split_diff_ratios[1][cls]['hard']:.3f})",
                    f"{split_diff_counts[2][cls]['hard']:<5} ({split_diff_ratios[2][cls]['hard']:.3f})",
                )
            )
        else:
            logging.info(
                "{:<20} | {:<15} {:<15} | {:<15} {:<15} | {:<15} {:<15} |".format(
                    cls,
                    f"{split_diff_counts[0][cls]['easy']:<5} ({split_diff_ratios[0][cls]['easy']:.3f})",
                    f"{split_diff_counts[1][cls]['easy']:<5} ({split_diff_ratios[1][cls]['easy']:.3f})",
                    f"{split_diff_counts[0][cls]['moderate']:<5} ({split_diff_ratios[0][cls]['moderate']:.3f})",
                    f"{split_diff_counts[1][cls]['moderate']:<5} ({split_diff_ratios[1][cls]['moderate']:.3f})",
                    f"{split_diff_counts[0][cls]['hard']:<5} ({split_diff_ratios[0][cls]['hard']:.3f})",
                    f"{split_diff_counts[1][cls]['hard']:<5} ({split_diff_ratios[1][cls]['hard']:.3f})",
                )
            )

    tot = [
        [sum([x[d] for x in split_diff_counts[i].values()]) for d in DIFFICULTY_MAP.keys()]
        for i in range(len(splits))
    ]
    logging.info("-" * (20 + (15 * len(DIFFICULTY_MAP.keys()) * len(splits))))  # 172
    if len(splits) == 3:
        logging.info(
            "{:<20} | {:<15} {:<15} {:<15} | {:<15} {:<15} {:<15} | {:<15} {:<15} {:<15} |".format(
                "Total",
                f"{sum([x['easy'] for x in split_diff_counts[0].values()]):<5} ({tot[0][0]/sum(tot[0]):.3f})",
                f"{sum([x['easy'] for x in split_diff_counts[1].values()]):<5} ({tot[1][0]/sum(tot[1]):.3f})",
                f"{sum([x['easy'] for x in split_diff_counts[2].values()]):<5} ({tot[2][0]/sum(tot[2]):.3f})",
                f"{sum([x['moderate'] for x in split_diff_counts[0].values()]):<5} ({tot[0][1]/sum(tot[0]):.3f})",
                f"{sum([x['moderate'] for x in split_diff_counts[1].values()]):<5} ({tot[1][1]/sum(tot[1]):.3f})",
                f"{sum([x['moderate'] for x in split_diff_counts[2].values()]):<5} ({tot[2][1]/sum(tot[2]):.3f})",
                f"{sum([x['hard'] for x in split_diff_counts[0].values()]):<5} ({tot[0][2]/sum(tot[0]):.3f})",
                f"{sum([x['hard'] for x in split_diff_counts[1].values()]):<5} ({tot[1][2]/sum(tot[1]):.3f})",
                f"{sum([x['hard'] for x in split_diff_counts[2].values()]):<5} ({tot[2][2]/sum(tot[2]):.3f})",
            )
        )
    else:
        logging.info(
            "{:<20} | {:<15} {:<15} | {:<15} {:<15} | {:<15} {:<15} |".format(
                "Total",
                f"{sum([x['easy'] for x in split_diff_counts[0].values()]):<5} ({tot[0][0]/sum(tot[0]):.3f})",
                f"{sum([x['easy'] for x in split_diff_counts[1].values()]):<5} ({tot[1][0]/sum(tot[1]):.3f})",
                f"{sum([x['moderate'] for x in split_diff_counts[0].values()]):<5} ({tot[0][1]/sum(tot[0]):.3f})",
                f"{sum([x['moderate'] for x in split_diff_counts[1].values()]):<5} ({tot[1][1]/sum(tot[1]):.3f})",
                f"{sum([x['hard'] for x in split_diff_counts[0].values()]):<5} ({tot[0][2]/sum(tot[0]):.3f})",
                f"{sum([x['hard'] for x in split_diff_counts[1].values()]):<5} ({tot[1][2]/sum(tot[1]):.3f})",
            )
        )

    split_distance_counts = best_segment_loss_details["distance"]["split_counts"]
    split_distance_ratios = best_segment_loss_details["distance"]["split_ratios"]

    logging.info("")
    logging.info("=" * 172)
    logging.info("Number and ratios of objects with corresponding distance levels in splits: ")
    logging.info("=" * 172)

    if len(splits) == 3:
        logging.info(
            "{:<20} | {:<15} {:<15} {:<15} | {:<15} {:<15} {:<15} | {:<15} {:<15} {:<15} |".format(
                "", "", "<40", "", "", "40-50", "", "", ">50", ""
            )
        )
    else:
        logging.info(
            "{:<20} | {:<15} {:<15} | {:<15} {:<15} | {:<15} {:<15} |".format(
                "", "<40", "", "40-50", "", ">50", ""
            )
        )

    title = f"{'Class':<20} |" + "".join(
        [
            f" {splits[0]:<15} {splits[1]:<15} {splits[2]:<15} |"
            if len(splits) == 3
            else f" {splits[0]:<15} {splits[1]:<15} |"
            for _ in range(len(DISTANCE_MAP.keys()))
        ]
    )
    logging.info(title)
    logging.info("-" * (20 + (15 * len(DISTANCE_MAP.keys()) * len(splits))))
    for j, cls in enumerate(classes):
        if len(splits) == 3:
            logging.info(
                "{:<20} | {:<15} {:<15} {:<15} | {:<15} {:<15} {:<15} | {:<15} {:<15} {:<15} |".format(
                    cls,
                    f"{split_distance_counts[0][cls]['d<40']:<5} ({split_distance_ratios[0][cls]['d<40']:.3f})",
                    f"{split_distance_counts[1][cls]['d<40']:<5} ({split_distance_ratios[1][cls]['d<40']:.3f})",
                    f"{split_distance_counts[2][cls]['d<40']:<5} ({split_distance_ratios[2][cls]['d<40']:.3f})",
                    f"{split_distance_counts[0][cls]['d40-50']:<5} ({split_distance_ratios[0][cls]['d40-50']:.3f})",
                    f"{split_distance_counts[1][cls]['d40-50']:<5} ({split_distance_ratios[1][cls]['d40-50']:.3f})",
                    f"{split_distance_counts[2][cls]['d40-50']:<5} ({split_distance_ratios[2][cls]['d40-50']:.3f})",
                    f"{split_distance_counts[0][cls]['d>50']:<5} ({split_distance_ratios[0][cls]['d>50']:.3f})",
                    f"{split_distance_counts[1][cls]['d>50']:<5} ({split_distance_ratios[1][cls]['d>50']:.3f})",
                    f"{split_distance_counts[2][cls]['d>50']:<5} ({split_distance_ratios[2][cls]['d>50']:.3f})",
                )
            )
        else:
            logging.info(
                "{:<20} | {:<15} {:<15} | {:<15} {:<15} | {:<15} {:<15} |".format(
                    cls,
                    f"{split_distance_counts[0][cls]['d<40']:<5} ({split_distance_ratios[0][cls]['d<40']:.3f})",
                    f"{split_distance_counts[1][cls]['d<40']:<5} ({split_distance_ratios[1][cls]['d<40']:.3f})",
                    f"{split_distance_counts[0][cls]['d40-50']:<5} ({split_distance_ratios[0][cls]['d40-50']:.3f})",
                    f"{split_distance_counts[1][cls]['d40-50']:<5} ({split_distance_ratios[1][cls]['d40-50']:.3f})",
                    f"{split_distance_counts[0][cls]['d>50']:<5} ({split_distance_ratios[0][cls]['d>50']:.3f})",
                    f"{split_distance_counts[1][cls]['d>50']:<5} ({split_distance_ratios[1][cls]['d>50']:.3f})",
                )
            )

    tot = [
        [sum([x[d] for x in split_distance_counts[i].values()]) for d in DISTANCE_MAP.keys()]
        for i in range(len(splits))
    ]
    logging.info("-" * (20 + (15 * len(DISTANCE_MAP.keys()) * len(splits))))
    if len(splits) == 3:
        logging.info(
            "{:<20} | {:<15} {:<15} {:<15} | {:<15} {:<15} {:<15} | {:<15} {:<15} {:<15} |".format(
                "Total",
                # fmt: off
                f"{sum([x['d<40'] for x in split_distance_counts[0].values()]):<5} ({tot[0][0]/sum(tot[0]):.3f})",
                f"{sum([x['d<40'] for x in split_distance_counts[1].values()]):<5} ({tot[1][0]/sum(tot[1]):.3f})",
                f"{sum([x['d<40'] for x in split_distance_counts[2].values()]):<5} ({tot[2][0]/sum(tot[2]):.3f})",
                f"{sum([x['d40-50'] for x in split_distance_counts[0].values()]):<5} ({tot[0][1]/sum(tot[0]):.3f})",
                f"{sum([x['d40-50'] for x in split_distance_counts[1].values()]):<5} ({tot[1][1]/sum(tot[1]):.3f})",
                f"{sum([x['d40-50'] for x in split_distance_counts[2].values()]):<5} ({tot[2][1]/sum(tot[2]):.3f})",
                f"{sum([x['d>50'] for x in split_distance_counts[0].values()]):<5} ({tot[0][2]/sum(tot[0]):.3f})",
                f"{sum([x['d>50'] for x in split_distance_counts[1].values()]):<5} ({tot[1][2]/sum(tot[1]):.3f})",
                f"{sum([x['d>50'] for x in split_distance_counts[2].values()]):<5} ({tot[2][2]/sum(tot[2]):.3f})",
                # fmt: on
            )
        )
    else:
        logging.info(
            "{:<20} | {:<15} {:<15} | {:<15} {:<15} | {:<15} {:<15} |".format(
                "Total",
                # fmt: off
                f"{sum([x['d<40'] for x in split_distance_counts[0].values()]):<5} ({tot[0][0]/sum(tot[0]):.3f})",
                f"{sum([x['d<40'] for x in split_distance_counts[1].values()]):<5} ({tot[1][0]/sum(tot[1]):.3f})",
                f"{sum([x['d40-50'] for x in split_distance_counts[0].values()]):<5} ({tot[0][1]/sum(tot[0]):.3f})",
                f"{sum([x['d40-50'] for x in split_distance_counts[1].values()]):<5} ({tot[1][1]/sum(tot[1]):.3f})",
                f"{sum([x['d>50'] for x in split_distance_counts[0].values()]):<5} ({tot[0][2]/sum(tot[0]):.3f})",
                f"{sum([x['d>50'] for x in split_distance_counts[1].values()]):<5} ({tot[1][2]/sum(tot[1]):.3f})",
                # fmt: on
            )
        )

    split_num_points_counts = best_segment_loss_details["num_points"]["split_counts"]
    split_num_points_ratios = best_segment_loss_details["num_points"]["split_ratios"]

    logging.info("")
    logging.info("=" * 172)
    logging.info("Number and ratios of objects with corresponding number of points in splits: ")
    logging.info("=" * 172)

    if len(splits) == 3:
        logging.info(
            "{:<20} | {:<15} {:<15} {:<15} | {:<15} {:<15} {:<15} | {:<15} {:<15} {:<15} |".format(
                "", "", "<20", "", "", "20-50", "", "", ">50", ""
            )
        )
    else:
        logging.info(
            "{:<20} | {:<15} {:<15} | {:<15} {:<15} | {:<15} {:<15} |".format(
                "", "<20", "", "20-50", "", ">50", ""
            )
        )

    title = f"{'Class':<20} |" + "".join(
        [
            f" {splits[0]:<15} {splits[1]:<15} {splits[2]:<15} |"
            if len(splits) == 3
            else f" {splits[0]:<15} {splits[1]:<15} |"
            for _ in range(len(NUM_POINTS_MAP.keys()))
        ]
    )
    logging.info(title)
    logging.info("-" * (20 + (15 * len(NUM_POINTS_MAP.keys()) * len(splits))))
    for j, cls in enumerate(classes):
        if len(splits) == 3:
            logging.info(
                "{:<20} | {:<15} {:<15} {:<15} | {:<15} {:<15} {:<15} | {:<15} {:<15} {:<15} |".format(
                    cls,
                    f"{split_num_points_counts[0][cls]['n<20']:<5} ({split_num_points_ratios[0][cls]['n<20']:.3f})",
                    f"{split_num_points_counts[1][cls]['n<20']:<5} ({split_num_points_ratios[1][cls]['n<20']:.3f})",
                    f"{split_num_points_counts[2][cls]['n<20']:<5} ({split_num_points_ratios[2][cls]['n<20']:.3f})",
                    f"{split_num_points_counts[0][cls]['n20-50']:<5} ({split_num_points_ratios[0][cls]['n20-50']:.3f})",
                    f"{split_num_points_counts[1][cls]['n20-50']:<5} ({split_num_points_ratios[1][cls]['n20-50']:.3f})",
                    f"{split_num_points_counts[2][cls]['n20-50']:<5} ({split_num_points_ratios[2][cls]['n20-50']:.3f})",
                    f"{split_num_points_counts[0][cls]['n>50']:<5} ({split_num_points_ratios[0][cls]['n>50']:.3f})",
                    f"{split_num_points_counts[1][cls]['n>50']:<5} ({split_num_points_ratios[1][cls]['n>50']:.3f})",
                    f"{split_num_points_counts[2][cls]['n>50']:<5} ({split_num_points_ratios[2][cls]['n>50']:.3f})",
                )
            )
        else:
            logging.info(
                "{:<20} | {:<15} {:<15} | {:<15} {:<15} | {:<15} {:<15} |".format(
                    cls,
                    f"{split_num_points_counts[0][cls]['n<20']:<5} ({split_num_points_ratios[0][cls]['n<20']:.3f})",
                    f"{split_num_points_counts[1][cls]['n<20']:<5} ({split_num_points_ratios[1][cls]['n<20']:.3f})",
                    f"{split_num_points_counts[0][cls]['n20-50']:<5} ({split_num_points_ratios[0][cls]['n20-50']:.3f})",
                    f"{split_num_points_counts[1][cls]['n20-50']:<5} ({split_num_points_ratios[1][cls]['n20-50']:.3f})",
                    f"{split_num_points_counts[0][cls]['n>50']:<5} ({split_num_points_ratios[0][cls]['n>50']:.3f})",
                    f"{split_num_points_counts[1][cls]['n>50']:<5} ({split_num_points_ratios[1][cls]['n>50']:.3f})",
                )
            )

    tot = [
        [sum([x[d] for x in split_num_points_counts[i].values()]) for d in NUM_POINTS_MAP.keys()]
        for i in range(len(splits))
    ]
    logging.info("-" * (20 + (15 * len(NUM_POINTS_MAP.keys()) * len(splits))))
    if len(splits) == 3:
        logging.info(
            "{:<20} | {:<15} {:<15} {:<15} | {:<15} {:<15} {:<15} | {:<15} {:<15} {:<15} |".format(
                "Total",
                # fmt: off
                f"{sum([x['n<20'] for x in split_num_points_counts[0].values()]):<5} ({tot[0][0]/sum(tot[0]):.3f})",
                f"{sum([x['n<20'] for x in split_num_points_counts[1].values()]):<5} ({tot[1][0]/sum(tot[1]):.3f})",
                f"{sum([x['n<20'] for x in split_num_points_counts[2].values()]):<5} ({tot[2][0]/sum(tot[2]):.3f})",
                f"{sum([x['n20-50'] for x in split_num_points_counts[0].values()]):<5} ({tot[0][1]/sum(tot[0]):.3f})",
                f"{sum([x['n20-50'] for x in split_num_points_counts[1].values()]):<5} ({tot[1][1]/sum(tot[1]):.3f})",
                f"{sum([x['n20-50'] for x in split_num_points_counts[2].values()]):<5} ({tot[2][1]/sum(tot[2]):.3f})",
                f"{sum([x['n>50'] for x in split_num_points_counts[0].values()]):<5} ({tot[0][2]/sum(tot[0]):.3f})",
                f"{sum([x['n>50'] for x in split_num_points_counts[1].values()]):<5} ({tot[1][2]/sum(tot[1]):.3f})",
                f"{sum([x['n>50'] for x in split_num_points_counts[2].values()]):<5} ({tot[2][2]/sum(tot[2]):.3f})",
                # fmt: on
            )
        )
    else:
        logging.info(
            "{:<20} | {:<15} {:<15} | {:<15} {:<15} | {:<15} {:<15} |".format(
                "Total",
                # fmt: off
                f"{sum([x['n<20'] for x in split_num_points_counts[0].values()]):<5} ({tot[0][0]/sum(tot[0]):.3f})",
                f"{sum([x['n<20'] for x in split_num_points_counts[1].values()]):<5} ({tot[1][0]/sum(tot[1]):.3f})",
                f"{sum([x['n20-50'] for x in split_num_points_counts[0].values()]):<5} ({tot[0][1]/sum(tot[0]):.3f})",
                f"{sum([x['n20-50'] for x in split_num_points_counts[1].values()]):<5} ({tot[1][1]/sum(tot[1]):.3f})",
                f"{sum([x['n>50'] for x in split_num_points_counts[0].values()]):<5} ({tot[0][2]/sum(tot[0]):.3f})",
                f"{sum([x['n>50'] for x in split_num_points_counts[1].values()]):<5} ({tot[1][2]/sum(tot[1]):.3f})",
                # fmt: on
            )
        )

    split_occlusion_counts = best_segment_loss_details["occlusion"]["split_counts"]
    split_occlusion_ratios = best_segment_loss_details["occlusion"]["split_ratios"]

    logging.info("")
    logging.info("=" * 172)
    logging.info("Number and ratios of objects with occlusion levels in splits: ")
    logging.info("=" * 172)

    if len(splits) == 3:
        logging.info(
            "{:<20} | {:<15} {:<15} {:<15} | {:<15} {:<15} {:<15} | {:<15} {:<15} {:<15} |".format(
                "",
                "",
                "Non",
                "",
                "",
                "Partially",
                "",
                "",
                "Mostly",
                "",
            )
        )
    else:
        logging.info(
            "{:<20} | {:<15} {:<15} | {:<15} {:<15} | {:<15} {:<15} |".format(
                "",
                "Non",
                "",
                "Partially",
                "",
                "Mostly",
                "",
            )
        )

    title = f"{'Class':<20} |" + "".join(
        [
            f" {splits[0]:<15} {splits[1]:<15} {splits[2]:<15} |"
            if len(splits) == 3
            else f" {splits[0]:<15} {splits[1]:<15} |"
            for _ in range(len(OCCLUSION_MAP.values()) - 1)
        ]
    )
    logging.info(title)
    logging.info("-" * (20 + (15 * (len(OCCLUSION_MAP.values()) - 1) * len(splits))))
    for j, cls in enumerate(classes):
        if len(splits) == 3:
            # fmt: off
            logging.info(
                "{:<20} | {:<15} {:<15} {:<15} | {:<15} {:<15} {:<15} | {:<15} {:<15} {:<15} |".format(
                    cls,
                    f"{split_occlusion_counts[0][cls]['NOT_OCCLUDED']:<5} ({split_occlusion_ratios[0][cls]['NOT_OCCLUDED']:.3f})",
                    f"{split_occlusion_counts[1][cls]['NOT_OCCLUDED']:<5} ({split_occlusion_ratios[1][cls]['NOT_OCCLUDED']:.3f})",
                    f"{split_occlusion_counts[2][cls]['NOT_OCCLUDED']:<5} ({split_occlusion_ratios[2][cls]['NOT_OCCLUDED']:.3f})",
                    f"{split_occlusion_counts[0][cls]['PARTIALLY_OCCLUDED']:<5} ({split_occlusion_ratios[0][cls]['PARTIALLY_OCCLUDED']:.3f})",
                    f"{split_occlusion_counts[1][cls]['PARTIALLY_OCCLUDED']:<5} ({split_occlusion_ratios[1][cls]['PARTIALLY_OCCLUDED']:.3f})",
                    f"{split_occlusion_counts[2][cls]['PARTIALLY_OCCLUDED']:<5} ({split_occlusion_ratios[2][cls]['PARTIALLY_OCCLUDED']:.3f})",
                    f"{split_occlusion_counts[0][cls]['MOSTLY_OCCLUDED']:<5} ({split_occlusion_ratios[0][cls]['MOSTLY_OCCLUDED']:.3f})",
                    f"{split_occlusion_counts[1][cls]['MOSTLY_OCCLUDED']:<5} ({split_occlusion_ratios[1][cls]['MOSTLY_OCCLUDED']:.3f})",
                    f"{split_occlusion_counts[2][cls]['MOSTLY_OCCLUDED']:<5} ({split_occlusion_ratios[2][cls]['MOSTLY_OCCLUDED']:.3f})",
                )
            )  # fmt: on
        else:
            # fmt: off
            logging.info(
                "{:<20} | {:<15} {:<15} | {:<15} {:<15} | {:<15} {:<15} |".format(
                    cls,
                    f"{split_occlusion_counts[0][cls]['NOT_OCCLUDED']:<5} ({split_occlusion_ratios[0][cls]['NOT_OCCLUDED']:.3f})",
                    f"{split_occlusion_counts[1][cls]['NOT_OCCLUDED']:<5} ({split_occlusion_ratios[1][cls]['NOT_OCCLUDED']:.3f})",
                    f"{split_occlusion_counts[0][cls]['PARTIALLY_OCCLUDED']:<5} ({split_occlusion_ratios[0][cls]['PARTIALLY_OCCLUDED']:.3f})",
                    f"{split_occlusion_counts[1][cls]['PARTIALLY_OCCLUDED']:<5} ({split_occlusion_ratios[1][cls]['PARTIALLY_OCCLUDED']:.3f})",
                    f"{split_occlusion_counts[0][cls]['MOSTLY_OCCLUDED']:<5} ({split_occlusion_ratios[0][cls]['MOSTLY_OCCLUDED']:.3f})",
                    f"{split_occlusion_counts[1][cls]['MOSTLY_OCCLUDED']:<5} ({split_occlusion_ratios[1][cls]['MOSTLY_OCCLUDED']:.3f})",
                )
            )  # fmt: on

    tot = [
        [sum([x[d] for x in split_occlusion_counts[i].values()]) for d in OCCLUSION_MAP.values()]
        for i in range(len(splits))
    ]
    logging.info("-" * (20 + (15 * (len(OCCLUSION_MAP.values()) - 1) * len(splits))))
    if len(splits) == 3:
        logging.info(
            # fmt: off
            "{:<20} | {:<15} {:<15} {:<15} | {:<15} {:<15} {:<15} | {:<15} {:<15} {:<15} |".format(
                "Total",
                f"{sum([x['NOT_OCCLUDED'] for x in split_occlusion_counts[0].values()]):<5} ({tot[0][0]/sum(tot[0]):.3f})",
                f"{sum([x['NOT_OCCLUDED'] for x in split_occlusion_counts[1].values()]):<5} ({tot[1][0]/sum(tot[1]):.3f})",
                f"{sum([x['NOT_OCCLUDED'] for x in split_occlusion_counts[2].values()]):<5} ({tot[2][0]/sum(tot[2]):.3f})",
                f"{sum([x['PARTIALLY_OCCLUDED'] for x in split_occlusion_counts[0].values()]):<5} ({tot[0][1]/sum(tot[0]):.3f})",
                f"{sum([x['PARTIALLY_OCCLUDED'] for x in split_occlusion_counts[1].values()]):<5} ({tot[1][1]/sum(tot[1]):.3f})",
                f"{sum([x['PARTIALLY_OCCLUDED'] for x in split_occlusion_counts[2].values()]):<5} ({tot[2][1]/sum(tot[2]):.3f})",
                f"{sum([x['MOSTLY_OCCLUDED'] for x in split_occlusion_counts[0].values()]):<5} ({tot[0][2]/sum(tot[0]):.3f})",
                f"{sum([x['MOSTLY_OCCLUDED'] for x in split_occlusion_counts[1].values()]):<5} ({tot[1][2]/sum(tot[1]):.3f})",
                f"{sum([x['MOSTLY_OCCLUDED'] for x in split_occlusion_counts[2].values()]):<5} ({tot[2][2]/sum(tot[2]):.3f})",
            )
        )  # fmt: on
    else:
        logging.info(
            # fmt: off
            "{:<20} | {:<15} {:<15} | {:<15} {:<15} | {:<15} {:<15} |".format(
                "Total",
                f"{sum([x['NOT_OCCLUDED'] for x in split_occlusion_counts[0].values()]):<5} ({tot[0][0]/sum(tot[0]):.3f})",
                f"{sum([x['NOT_OCCLUDED'] for x in split_occlusion_counts[1].values()]):<5} ({tot[1][0]/sum(tot[1]):.3f})",
                f"{sum([x['PARTIALLY_OCCLUDED'] for x in split_occlusion_counts[0].values()]):<5} ({tot[0][1]/sum(tot[0]):.3f})",
                f"{sum([x['PARTIALLY_OCCLUDED'] for x in split_occlusion_counts[1].values()]):<5} ({tot[1][1]/sum(tot[1]):.3f})",
                f"{sum([x['MOSTLY_OCCLUDED'] for x in split_occlusion_counts[0].values()]):<5} ({tot[0][2]/sum(tot[0]):.3f})",
                f"{sum([x['MOSTLY_OCCLUDED'] for x in split_occlusion_counts[1].values()]):<5} ({tot[1][2]/sum(tot[1]):.3f})",
            )
        )  # fmt: on

    print()
    if len(splits) == 3:
        logging.info(
            "{:<20} | {:<15} {:<15} {:<15} |".format(
                "",
                "",
                "Unkown",
                "",
            )
        )
    else:
        logging.info(
            "{:<20} | {:<15} {:<15} |".format(
                "",
                "Unkown",
                "",
            )
        )

    title = f"{'Class':<20} |" + "".join(
        [
            f" {splits[0]:<15} {splits[1]:<15} {splits[2]:<15} |"
            if len(splits) == 3
            else f" {splits[0]:<15} {splits[1]:<15} |"
            for _ in range(1)
        ]
    )
    logging.info(title)
    logging.info("-" * (20 + (15 * 1 * len(splits))))
    for j, cls in enumerate(classes):
        if len(splits) == 3:
            logging.info(
                "{:<20} | {:<15} {:<15} {:<15} |".format(
                    cls,
                    f"{split_occlusion_counts[0][cls]['UNKNOWN']:<5} ({split_occlusion_ratios[0][cls]['UNKNOWN']:.3f})",
                    f"{split_occlusion_counts[1][cls]['UNKNOWN']:<5} ({split_occlusion_ratios[1][cls]['UNKNOWN']:.3f})",
                    f"{split_occlusion_counts[2][cls]['UNKNOWN']:<5} ({split_occlusion_ratios[2][cls]['UNKNOWN']:.3f})",
                )
            )
        else:
            logging.info(
                "{:<20} | {:<15} {:<15} |".format(
                    cls,
                    f"{split_occlusion_counts[0][cls]['UNKNOWN']:<5} ({split_occlusion_ratios[0][cls]['UNKNOWN']:.3f})",
                    f"{split_occlusion_counts[1][cls]['UNKNOWN']:<5} ({split_occlusion_ratios[1][cls]['UNKNOWN']:.3f})",
                )
            )
    logging.info("-" * (20 + (15 * 1 * len(splits))))
    if len(splits) == 3:
        logging.info(
            # fmt: off
            "{:<20} | {:<15} {:<15} {:<15} |".format(
                "Total",
                f"{sum([x['UNKNOWN'] for x in split_occlusion_counts[0].values()]):<5} ({tot[0][0]/sum(tot[0]):.3f})",
                f"{sum([x['UNKNOWN'] for x in split_occlusion_counts[1].values()]):<5} ({tot[1][0]/sum(tot[1]):.3f})",
                f"{sum([x['UNKNOWN'] for x in split_occlusion_counts[2].values()]):<5} ({tot[2][0]/sum(tot[2]):.3f})",
            )
        )  # fmt: on
    else:
        logging.info(
            # fmt: off
            "{:<20} | {:<15} {:<15} |".format(
                "Total",
                f"{sum([x['UNKNOWN'] for x in split_occlusion_counts[0].values()]):<5} ({tot[0][0]/sum(tot[0]):.3f})",
                f"{sum([x['UNKNOWN'] for x in split_occlusion_counts[1].values()]):<5} ({tot[1][0]/sum(tot[1]):.3f})",
            )
        )  # fmt: on


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

    bucket_size = {}
    for x, y in data.items():
        bucket_size[x] = y.no_total_frames // segment_size
    segment_list = []
    for x, y in bucket_size.items():
        for z in range(y):
            start_idx = z * segment_size
            end_idx = z * segment_size + segment_size
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

    # multi-processing task function
    def task(position, lock, queue, perms, *args):
        total_iterations = len(perms)
        n_found = 0
        best_loss = None
        best_segment_loss_details = {"loss": None}
        filter_stats = defaultdict(int)
        with lock:
            bar = tqdm(
                desc=f"Process {position}",
                total=total_iterations,
                position=position,
                leave=False,
                mininterval=1,
            )

        for p in perms:
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

            with lock:
                bar.update(1)

        with lock:
            bar.close()

        queue.put((n_found, filter_stats, best_segment_loss_details))

    manager = multiprocessing.Manager()
    lock = manager.Lock()
    queue = manager.Queue()

    def custom_callback(e):
        logging.error(f"{type(e).__name__} {__file__} {e}")

    with multiprocessing.Pool(args.p) as pool:
        start_time = time.perf_counter()
        n = len(perms) // args.p
        perms = [perms[i : i + n] for i in range(0, len(perms), n)]
        for position in range(args.p):
            pool.apply_async(
                task,
                [
                    position + 1,
                    lock,
                    queue,
                    perms[position],
                    data,
                    split_ratios,
                    classes,
                    CLASS_WEIGHTS,
                    DIFFICULTY_WEIGHTS,
                    DISTANCE_WEIGHTS,
                    NUM_POINTS_WEIGHTS,
                    OCCLUSION_WEIGHTS,
                    args.include_all_classes,
                    args.include_all_sequences,
                    args.include_same_classes_in_difficulty,
                    args.difficulty_th,
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

    logging.info("Creating split folders and copying split files...")
    create_and_copy_split(
        target_path, data, best_perm, best_segment_loss_details["split_idx_ranges"], splits
    )

    logging.info("Saving original split details...")
    save_original_split_details(
        target_path,
        segment_size,
        best_perm,
        best_segment_loss_details,
        splits,
        split_ratios,
    )
