import json
import logging
import os
import shutil
import warnings
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from glob import glob
from itertools import permutations
from typing import Dict, List, Optional, Tuple

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

CLASS_WEIGHTS = [1, 15, 15, 5, 5, 15, 15, 20, 5, 5]

DIFFICULTY_MAP = {
    "easy": {"distance": "d<40", "num_points": "n>50", "occlusion": "no_occluded"},
    "moderate": {
        "distance": "d40-50",
        "num_points": "n20-50",
        "occlusion": "partially_occluded",
    },
    "hard": {"distance": "d>50", "num_points": "n<20", "occlusion": "mostly_occluded"},
}
DISTANCE_MAP = {"d<40": [0, 40], "d40-50": [40, 50], "d>50": [50, 9999]}
NUM_POINTS_MAP = {"n<20": [0, 20], "n20-50": [20, 50], "n>50": [50, 9999]}
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


@dataclass()
class TemporalSequenceDetails:
    # fmt: off
    scene_token: str
    no_total_frames: int
    total_difficulty_stats: Dict[str, Dict[str, int]] = field(repr=False, compare=False)
    total_distance_stats: Dict[str, Dict[str, int]] = field(repr=False, compare=False)
    total_num_points_stats: Dict[str, Dict[str, int]] = field(repr=False, compare=False)
    total_occlusion_stats: Dict[str, Dict[str, int]] = field(repr=False, compare=False)
    total_class_stats: Dict[str, int] = field(repr=True, compare=False)
    frame_class_stats: List[Dict[str, int]] = field(default_factory=list, repr=False, compare=False)
    frame_img_s1_paths: List[str] = field(default_factory=list, repr=False, compare=False)
    frame_img_s2_paths: List[str] = field(default_factory=list, repr=False, compare=False)
    frame_img_s1_label_paths: List[str] = field(default_factory=list, repr=False, compare=False)
    frame_img_s2_label_paths: List[str] = field(default_factory=list, repr=False, compare=False)
    frame_pcd_paths: List[str] = field(default_factory=list, repr=False, compare=False)
    frame_pcd_labels_paths: List[str] = field(default_factory=list, repr=False, compare=False)
    # fmt: on

    def get_class_stats_in_range(self, start: int, end: int) -> Dict[str, int]:
        return {cls: sum([x[cls] for x in self.frame_class_stats[start:end]]) for cls in CLASSES}

    def get_path_list_in_range(self, start: int, end: int) -> Dict[str, List[str]]:
        return {
            "frame_img_s1_paths": self.frame_img_s1_paths[start:end],
            "frame_img_s2_paths": self.frame_img_s2_paths[start:end],
            "frame_img_s1_label_paths": self.frame_img_s1_label_paths[start:end],
            "frame_img_s2_label_paths": self.frame_img_s2_label_paths[start:end],
            "frame_pcd_paths": self.frame_pcd_paths[start:end],
            "frame_pcd_labels_paths": self.frame_pcd_labels_paths[start:end],
        }


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
    parser.add_argument("--include-all-classes", default=False, action="store_true")
    parser.add_argument("--include-all-sequences", default=False, action="store_true")
    parser.add_argument(
        "-log",
        "--loglevel",
        default="warning",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )

    return parser.parse_args()


def create_sequence_details(root_path: str) -> Dict[str, TemporalSequenceDetails]:
    """
    Create a dictionary of sequence details.

    Args:
        root_path: root path of the dataset

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
                    no_total_frames=1,
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
            for obj in frame_objects.values():
                obj_type = obj["object_data"]["type"]

                class_stats[obj_type] += 1
                if "cuboid" in obj["object_data"]:
                    # distance from sensor
                    loc = np.asarray(obj["object_data"]["cuboid"]["val"][:3], dtype=np.float32)
                    distance = np.sqrt(np.sum(np.array(loc[:2]) ** 2))
                    for k, v in DISTANCE_MAP.items():
                        if v[0] < distance <= v[1]:
                            data[scene_token].total_distance_stats[obj_type][k] += 1

                    attributes = obj["object_data"]["cuboid"]["attributes"]

                    # occlusion
                    occlusion_level = None
                    for x in attributes["text"]:
                        if x["name"] == "occlusion_level":
                            occlusion_level = x["val"]
                            if occlusion_level != "":
                                data[scene_token].total_occlusion_stats[obj_type][
                                    occlusion_level
                                ] += 1
                            else:
                                data[scene_token].total_occlusion_stats[obj_type]["UNKNOWN"] += 1

                    # number of points
                    num_points = 0
                    for x in attributes["num"]:
                        if x["name"] == "num_points":
                            num_points = x["val"]
                            for k, v in NUM_POINTS_MAP.items():
                                if v[0] < num_points <= v[1]:
                                    data[scene_token].total_num_points_stats[obj_type][k] += 1

                    # difficulty
                    if (distance <= 40 and num_points > 50) or occlusion_level == "NOT_OCCLUDED":
                        data[scene_token].total_difficulty_stats[obj_type]["easy"] += 1
                    elif (
                        (distance > 40 and distance <= 50)
                        and (num_points > 20 and num_points <= 50)
                    ) or occlusion_level == "PARTIALLY_OCCLUDED":
                        data[scene_token].total_difficulty_stats[obj_type]["moderate"] += 1
                    elif (
                        distance > 50 and num_points <= 20
                    ) or occlusion_level == "MOSTLY_OCCLUDED":
                        data[scene_token].total_difficulty_stats[obj_type]["hard"] += 1

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

    return data


def calc_segment_score(
    perm: List[Tuple[str, Tuple[int, int]]],
    data: Dict[str, TemporalSequenceDetails],
    split_ratio: List[float] = [0.8, 0.1, 0.1],
    classes: List[str] = CLASSES,
    class_weights: Optional[List[float]] = None,
    include_all_classes: bool = False,
    include_all_sequences: bool = False,
) -> Tuple[float, List[Dict[str, int]], List[Dict[str, float]], List[Dict[str, int]]]:
    """
    Calculate the score of a given permutation of segments.

    The score is calculated by comparing the class distributions between splits.

    Args:
        segment_size: number of frames in a segment
        perm: permutation of segments
        data: dictionary of sequence details
        split_ratio: ratio of splits
        classes: list of classes
        class_weights: weights of classes
        include_all_classes: whether to force to have each class presence in all splits
        include_all_sequencess: whether to force to have each sequence presence in all splits

    Returns:
        Tuple: score, class counts, class ratios, split index ranges
    """
    score = 0

    split_idx_range_ratios = [
        0,
        (split_ratio[0] * len(perm)),
        ((split_ratio[0] + split_ratio[1]) * len(perm)),
        len(perm),
    ]

    assert all(
        [float(x).is_integer() for x in split_idx_range_ratios]
    ), "segment_size and split_ratio results in non-integer split_idx_ranges"

    split_cls_counts = [{cls: 0 for cls in classes} for _ in range(len(split_ratio))]

    seq_assigned_split = [0] * int(len(perm) * split_ratio[0])
    seq_assigned_split.extend([1] * int(len(perm) * split_ratio[1]))
    seq_assigned_split.extend([2] * int(len(perm) * split_ratio[2]))

    class_counts = []
    for x in perm:
        # get class counts for current segment
        token = x[0]
        start_idx = x[1][0]
        end_idx = x[1][1]
        class_counts.append(data[token].get_class_stats_in_range(start_idx, end_idx))

    for idx, counts in enumerate(class_counts):
        if seq_assigned_split[idx] == 0:  # in train split
            for cls in classes:
                split_cls_counts[0][cls] += counts[cls]
        elif seq_assigned_split[idx] == 1:  # in val split
            for cls in classes:
                split_cls_counts[1][cls] += counts[cls]
        else:  # in test split
            for cls in classes:
                split_cls_counts[2][cls] += counts[cls]

    # calculate the class distributions in splits
    split_cls_ratios = []
    for i in range(len(split_ratio)):
        total_cls_counts = sum(split_cls_counts[i].values())
        split_cls_ratios.append([x / total_cls_counts for x in split_cls_counts[i].values()])

    # force to have each class in all of the splits
    if include_all_classes:
        for x in split_cls_counts:
            for cls, count in x.items():
                if count == 0:
                    return -1e10, split_cls_counts, split_cls_ratios, split_idx_range_ratios

    # force to have each sequence in all of the splits
    if include_all_sequences:
        for i in range(len(split_ratio)):
            for j, seq in enumerate(perm):
                a = (
                    np.asarray(perm)[
                        int(split_idx_range_ratios[i]) : int(split_idx_range_ratios[i + 1])
                    ]
                    == seq
                ).sum()
                if a == 0:
                    return -1e10, split_cls_counts, split_cls_ratios, split_idx_range_ratios

    # compare the class distributions between splits
    for i in range(len(split_cls_ratios) - 1):
        for j in range(len(classes)):
            if class_weights is None:
                score += abs(split_cls_ratios[i][j] - split_cls_ratios[i + 1][j])
            else:
                score += abs(split_cls_ratios[i][j] - split_cls_ratios[i + 1][j]) * class_weights[j]

    return -score, split_cls_counts, split_cls_ratios, split_idx_range_ratios


def create_and_copy_split(
    target_path: str,
    data: Dict[str, TemporalSequenceDetails],
    perm: List[Tuple[str, Tuple[int]]],
    split_idx_ranges: List[float],
) -> None:
    """
    Create and copy the split to the target path.

    Args:
        target_path: target path to copy the split
        data: dictionary of sequence details
        perm: selected permutation of segments
        split_idx_ranges: split index ranges
    """
    split_source_paths = []

    for x in perm:
        # get class counts for current segment
        token = x[0]
        start_idx = x[1][0]
        end_idx = x[1][1]
        split_source_paths.append(data[token].get_path_list_in_range(start_idx, end_idx))

    target_paths = {}
    for split in ["train", "val", "test"]:
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
        if split_idx_ranges[0] <= idx < split_idx_ranges[1]:  # in train split
            split = "train"
        elif split_idx_ranges[1] <= idx < split_idx_ranges[2]:  # in val split
            split = "val"
        else:  # in test split
            split = "test"
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
    split_idx_ranges: List[float],
    best_class_counts: List[Dict[str, int]],
    best_class_ratios: List[Dict[str, float]],
    split_ratio: List[int] = [0.8, 0.1, 0.1],
    classes: List[str] = CLASSES,
) -> None:
    """
    Save the original split details.

    Args:
        target_path: target path to copy the split
        segment_size: number of frames in a segment
        perm: selected permutation of segments
        split_idx_ranges: split index ranges
        best_class_counts: class counts of the best permutation
        best_class_ratios: class ratios of the best permutation
        split_ratio: ratio of splits
        classes: list of classes
    """
    orig_split_details = {}

    for i, split in enumerate(["train", "val", "test"]):
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

        orig_split_details[split]["classes"] = {}
        for j, cls in enumerate(classes):
            orig_split_details[split]["classes"][cls] = {
                "count": best_class_counts[i][cls],
                "ratio": best_class_ratios[i][j],
            }
    seq_assigned_split = ["train"] * int(len(perm) * split_ratio[0])
    seq_assigned_split.extend(["val"] * int(len(perm) * split_ratio[1]))
    seq_assigned_split.extend(["test"] * int(len(perm) * split_ratio[2]))
    orig_split_details["permutation"] = [
        {x[0]: {"start": x[1][0], "end": x[1][1], "split": seq_assigned_split[i]}}
        for i, x in enumerate(perm)
    ]

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


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=args.loglevel.upper())

    segment_size = args.segment_size
    perm_limit = args.perm_limit
    root_path = args.root_path
    target_path = args.out_path

    assert os.path.exists(root_path)

    logging.info("Creating sequence details...")
    data = create_sequence_details(root_path=root_path)

    assert len(data) != 0

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

    logging.info("Creating permutations...")
    perms = []
    while len(perms) < perm_limit:
        perms.append(list(segment_list[np.random.permutation(len(segment_list))]))

    logging.info(f"total number of permutations: {len(perms)}")

    best_perm: List[Tuple[str, Tuple[int]]] = None
    best_score: float = None
    best_class_counts: List[Dict[str, int]] = None
    best_class_ratios: List[Dict[str, float]] = None
    split_idx_ranges: List[float] = None

    for i, p in enumerate(perms):
        score, class_counts, class_ratios, split_idx_ranges = calc_segment_score(
            p,
            data,
            SPLIT_RATIO,
            CLASSES,
            CLASS_WEIGHTS,
            args.include_all_classes,
            args.include_all_sequences,
        )

        if best_score is None or score > best_score:
            best_score = score
            best_perm = p
            best_class_counts = class_counts
            best_class_ratios = class_ratios
            print("Current best score: ", best_score)

    print("Final best score: ", best_score)

    for i, split in enumerate(["train", "val", "test"]):
        print("\n", split, " statistics: ")
        print(
            "\tNumber of frames in the split:",
            segment_size * len(best_perm[int(split_idx_ranges[i]) : int(split_idx_ranges[i + 1])]),
        )
        print("\tNumber of occurences of sequences in the split:")
        for j, seq in enumerate(data.keys()):
            print(
                f"\t\t- {seq}: {(( np.asarray(best_perm)[int(split_idx_ranges[i]):int(split_idx_ranges[i+1])] == seq)).sum()}"
            )
        print("\tNumber of total occurences of classes in the split:")
        for j, cls in enumerate(CLASSES):
            print(f"\t\t- {cls}: {best_class_counts[i][cls]} - {best_class_ratios[i][j]*100:.2f}%")

    # print("best_perm\n", best_perm)

    logging.info("Creating split folders and copying split files...")
    create_and_copy_split(
        target_path,
        data,
        best_perm,
        split_idx_ranges,
    )

    logging.info("Saving original split details...")
    save_original_split_details(
        target_path,
        segment_size,
        best_perm,
        split_idx_ranges,
        best_class_counts,
        best_class_ratios,
        SPLIT_RATIO,
        CLASSES,
    )
