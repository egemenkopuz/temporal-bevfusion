import json
import logging
import os
import uuid
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from glob import glob
from typing import Any, Dict, List, Tuple

from tqdm import tqdm


def get_args() -> Namespace:
    """
    Parse given arguments for tokenize_a9_dataset_labels function.

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


def tokenize_a9_dataset_labels(
    root_path: str,
    out_path: str = None,
    idx_diff_rem: int = 20,
    range_diff_threshold_train: float = 1.0,
    range_diff_threshold_val: float = 2.0,
    range_diff_threshold_test: float = 3.0,
):
    splits = [os.path.basename(x) for x in glob(os.path.join(root_path, "*"))]
    for split in splits:
        if split.lower() in ["images", "labels_images", "labels_point_clouds", "point_clouds"]:
            splits = [""]
            break

    for split in splits:
        if split.endswith(".json"):
            continue
        if split.lower() in ["train", "training"]:
            range_diff_threshold = range_diff_threshold_train
        elif split.lower() in ["val", "validation"]:
            range_diff_threshold = range_diff_threshold_val
        elif split.lower() in ["test", "testing"]:
            range_diff_threshold = range_diff_threshold_test
        else:
            logging.info("Could find any conventional split name; assigning default values")
            range_diff_threshold = range_diff_threshold_train

        # fmt: off
        img_label_s1_folder = os.path.join(root_path, split, "labels_images", "s110_camera_basler_south1_8mm")
        img_label_s2_folder = os.path.join(root_path, split, "labels_images", "s110_camera_basler_south2_8mm")
        pcd_label_folder = os.path.join(root_path, split, "labels_point_clouds", "s110_lidar_ouster_south")
        # fmt: on

        img_label_s1_paths = sorted(glob(os.path.join(img_label_s1_folder, "*")))
        img_label_s2_paths = sorted(glob(os.path.join(img_label_s2_folder, "*")))
        pcd_label_paths = sorted(glob(os.path.join(pcd_label_folder, "*")))

        if not out_path:
            img_label_s1_folder_out = img_label_s1_folder
            img_label_s2_folder_out = img_label_s2_folder
            pcd_label_folder_out = pcd_label_folder
        else:
            # fmt: off
            img_label_s1_folder_out = os.path.join(out_path, split, "labels_images", "s110_camera_basler_south1_8mm")
            img_label_s2_folder_out = os.path.join(out_path, split, "labels_images", "s110_camera_basler_south2_8mm")
            pcd_label_folder_out = os.path.join(out_path, split, "labels_point_clouds", "s110_lidar_ouster_south")
            # fmt: on

            os.makedirs(img_label_s1_folder_out, mode=0o777, exist_ok=True)
            os.makedirs(img_label_s2_folder_out, mode=0o777, exist_ok=True)
            os.makedirs(pcd_label_folder_out, mode=0o777, exist_ok=True)

        assert len(img_label_s1_paths) == len(pcd_label_paths) and len(img_label_s1_paths) == len(
            img_label_s2_paths
        ), "Number of images and point clouds should be same"

        data = []

        for idx, path in enumerate(img_label_s1_paths):
            token = uuid.uuid4().hex

            img_label_s1_name = os.path.basename(path)
            img_label_s2_name = os.path.basename(img_label_s2_paths[idx])
            pcd_label_name = os.path.basename(pcd_label_paths[idx])

            ts = float(img_label_s1_name[:20].replace("_", "."))

            data.append(
                FrameDetails(
                    token=token,
                    ts=ts,
                    img_label_s1_name=img_label_s1_name,
                    img_label_s2_name=img_label_s2_name,
                    pcd_label_name=pcd_label_name,
                    prev=None,
                    next=None,
                    scene_token=None,
                    frame_idx=None,
                )
            )

        logging.info(f"Reading {split} split")

        for idx, frame_details in tqdm(enumerate(data), total=len(data)):
            # prev
            prev = None
            ts_diff = float("inf")

            for i in range(1, idx_diff_rem):
                if idx - i < 0:
                    break

                ts_diff_curr = frame_details.ts - data[idx - i].ts

                if ts_diff_curr > range_diff_threshold:
                    break

                if ts_diff_curr < ts_diff and ts_diff_curr > 0.0:
                    ts_diff = ts_diff_curr
                    prev = data[idx - i].token
                    data[idx].prev = prev

            # next
            next = None
            ts_diff = float("inf")

            for i in range(1, idx_diff_rem):
                if idx + i >= len(data):
                    break

                ts_diff_curr = data[idx + i].ts - frame_details.ts

                if ts_diff_curr > range_diff_threshold:
                    break

                if ts_diff_curr < ts_diff and ts_diff_curr > 0.0:
                    ts_diff = ts_diff_curr
                    next = data[idx + i].token
                    data[idx].next = next

        assert len(set([x.token for x in data])) == len(data)  # assert unique tokens

        logging.info(f"Tokenizing scenes for {split} split")

        frame_scene_dict = {x.token: i for i, x in enumerate(data)}

        def establish_scene_tokens(
            token: str, scene_dict: Dict[str, Any], data: List[FrameDetails]
        ) -> Tuple[str, int]:
            data_idx = scene_dict[token]

            # first frame in the sequence
            if data[data_idx].prev is None and data[data_idx].scene_token is None:
                scene_token = uuid.uuid4().hex
                frame_idx = 0

                data[data_idx].scene_token = scene_token
                data[data_idx].frame_idx = 0

            # first frame in the sequence but not first frame in the scene
            elif data[data_idx].prev is not None and data[data_idx].scene_token is None:
                scene_token, prev_frame_idx = establish_scene_tokens(
                    data[data_idx].prev, scene_dict, data
                )

                frame_idx = prev_frame_idx + 1

                data[data_idx].scene_token = scene_token
                data[data_idx].frame_idx = frame_idx
            else:
                return data[data_idx].scene_token, data[data_idx].frame_idx

            return scene_token, frame_idx

        for token in tqdm(frame_scene_dict.keys(), total=len(data)):
            data_idx = frame_scene_dict[token]
            if data[data_idx].scene_token is None:
                establish_scene_tokens(token, frame_scene_dict, data)

        logging.info(f"Saving {split} split tokens")

        for idx, frame_detail in tqdm(enumerate(data), total=len(data)):
            # img_label_s1
            img_label_s1_json_path = os.path.join(
                img_label_s1_folder, frame_detail.img_label_s1_name
            )
            json_data = None
            with open(img_label_s1_json_path, "r") as f:
                json_data = json.load(f)
                frame_idx = list(json_data["openlabel"]["frames"].keys())[0]
                frame_properties = json_data["openlabel"]["frames"][frame_idx]["frame_properties"]
                frame_properties["token"] = frame_detail.token
                frame_properties["prev"] = frame_detail.prev
                frame_properties["next"] = frame_detail.next
                frame_properties["scene_token"] = frame_detail.scene_token
                frame_properties["frame_idx"] = frame_detail.frame_idx
            if json_data is not None:
                img_label_s1_json_path_out = os.path.join(
                    img_label_s1_folder_out, frame_detail.img_label_s1_name
                )
                with open(img_label_s1_json_path_out, "w") as f:
                    json.dump(json_data, f, indent=4)

            # img_label_s2
            img_label_s2_json_path = os.path.join(
                img_label_s2_folder, frame_detail.img_label_s2_name
            )
            json_data = None
            with open(img_label_s2_json_path, "r") as f:
                json_data = json.load(f)
                frame_idx = list(json_data["openlabel"]["frames"].keys())[0]
                frame_properties = json_data["openlabel"]["frames"][frame_idx]["frame_properties"]
                frame_properties["token"] = frame_detail.token
                frame_properties["prev"] = frame_detail.prev
                frame_properties["next"] = frame_detail.next
                frame_properties["scene_token"] = frame_detail.scene_token
                frame_properties["frame_idx"] = frame_detail.frame_idx
            if json_data is not None:
                img_label_s2_json_path_out = os.path.join(
                    img_label_s2_folder_out, frame_detail.img_label_s2_name
                )
                with open(img_label_s2_json_path_out, "w") as f:
                    json.dump(json_data, f, indent=4)

            # pcd_label
            pcd_label_json_path = os.path.join(pcd_label_folder, frame_detail.pcd_label_name)
            json_data = None
            with open(pcd_label_json_path, "r") as f:
                json_data = json.load(f)
                frame_idx = list(json_data["openlabel"]["frames"].keys())[0]
                frame_properties = json_data["openlabel"]["frames"][frame_idx]["frame_properties"]
                frame_properties["token"] = frame_detail.token
                frame_properties["prev"] = frame_detail.prev
                frame_properties["next"] = frame_detail.next
                frame_properties["scene_token"] = frame_detail.scene_token
                frame_properties["frame_idx"] = frame_detail.frame_idx
            if json_data is not None:
                pcd_label_json_path_out = os.path.join(
                    pcd_label_folder_out, frame_detail.pcd_label_name
                )
                with open(pcd_label_json_path_out, "w") as f:
                    json.dump(json_data, f, indent=4)


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=args.loglevel.upper())
    tokenize_a9_dataset_labels(
        root_path=args.root_path,
        out_path=args.out_path,
        range_diff_threshold_train=args.threshold_train,
        range_diff_threshold_val=args.threshold_val,
        range_diff_threshold_test=args.threshold_test,
    )
