import json
import logging
import os
import uuid
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from glob import glob
from typing import List

from tqdm import tqdm


def get_args() -> Namespace:
    """
    Parse given arguments for tokenize_a9_dataset_labels function.

    Returns:
        Namespace: parsed arguments
    """
    parser = ArgumentParser()

    parser.add_argument("--root-path", type=str, required=True)
    parser.add_argument("--threshold-train", type=float, default=1.0, required=False)
    parser.add_argument("--threshold-val", type=float, default=2.0, required=False)
    parser.add_argument("--threshold-test", type=float, default=3.0, required=False)
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
    prev: str
    next: str


def tokenize_a9_dataset_labels(
    root_path: str,
    splits: List[str] = ["train", "val", "test"],
    idx_diff_rem: int = 20,
    range_diff_threshold_train: float = 1.0,
    range_diff_threshold_val: float = 2.0,
    range_diff_threshold_test: float = 3.0,
):
    for split in splits:
        if split == "train":
            range_diff_threshold = range_diff_threshold_train
        elif split == "val":
            range_diff_threshold = range_diff_threshold_val
        elif split == "test":
            range_diff_threshold = range_diff_threshold_test
        else:
            raise ValueError(f"Invalid split: {split}")

        # fmt: off
        img_label_s1_folder = os.path.join(root_path, split, "labels_images", "s110_camera_basler_south1_8mm")
        img_label_s2_folder = os.path.join(root_path, split, "labels_images", "s110_camera_basler_south2_8mm")
        pcd_label_folder = os.path.join(root_path, split, "labels_point_clouds", "s110_lidar_ouster_south")
        # fmt: on

        img_label_s1_paths = sorted(glob(os.path.join(img_label_s1_folder, "*")))
        img_label_s2_paths = sorted(glob(os.path.join(img_label_s2_folder, "*")))
        pcd_label_paths = sorted(glob(os.path.join(pcd_label_folder, "*")))

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

                if ts_diff_curr < ts_diff and ts_diff_curr < 1.0 and ts_diff_curr > 0.0:
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

                if ts_diff_curr < ts_diff and ts_diff_curr < 1.0 and ts_diff_curr > 0.0:
                    ts_diff = ts_diff_curr
                    next = data[idx + i].token
                    data[idx].next = next

        assert len(set([x.token for x in data])) == len(data)  # assert unique tokens

        logging.info(f"Tokenizing {split} split")

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
            if json_data is not None:
                with open(img_label_s1_json_path, "w") as f:
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
            if json_data is not None:
                with open(img_label_s2_json_path, "w") as f:
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
            if json_data is not None:
                with open(pcd_label_json_path, "w") as f:
                    json.dump(json_data, f, indent=4)


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=args.loglevel.upper())
    tokenize_a9_dataset_labels(
        root_path=args.root_path,
        range_diff_threshold_train=args.threshold_train,
        range_diff_threshold_val=args.threshold_val,
        range_diff_threshold_test=args.threshold_test,
    )
