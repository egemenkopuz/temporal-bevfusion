import copy
import os
import re
from argparse import ArgumentParser, Namespace
from glob import glob
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image

from mmdet3d.core import LiDARInstance3DBoxes

TUMTRAF_OBJECT_PALETTE = {
    "CAR": (0, 204, 246),
    "TRUCK": (63, 233, 185),
    "BUS": (217, 138, 134),
    "TRAILER": (90, 255, 126),
    "VAN": (235, 207, 54),
    "MOTORCYCLE": (185, 164, 84),
    "BICYCLE": (177, 140, 255),
    "PEDESTRIAN": (233, 118, 249),
    "EMERGENCY_VEHICLE": (102, 107, 250),
    "OTHER": (199, 199, 199),
}

OSDAR23_OBJECT_PALETTE = [
    ("lidar__cuboid__person", [0.91372549, 0.462745098, 0.976470588]),
    ("lidar__cuboid__bicycle", [0.694117647, 0.549019608, 1]),
    ("lidar__cuboid__signal", [0, 0.8, 0.964705882]),
    ("lidar__cuboid__catenary_pole", [0.337254902, 1, 0.71372549]),
    ("lidar__cuboid__buffer_stop", [0.352941176, 1, 0.494117647]),
    ("lidar__cuboid__train", [0.921568627, 0.811764706, 0.211764706]),
    ("lidar__cuboid__road_vehicle", [0.4, 0.419607843, 0.980392157]),
    ("lidar__cuboid__signal_pole", [0.725490196, 0.643137255, 0.329411765]),
    ("lidar__cuboid__animal", [0.780392157, 0.780392157, 0.780392157]),
    ("lidar__cuboid__switch", [0.850980392, 0.541176471, 0.525490196]),
    ("lidar__cuboid__crowd", [0.97647059, 0.43529412, 0.36470588]),
    ("lidar__cuboid__wagons", [0.98431373, 0.94901961, 0.75294118]),
    ("lidar__cuboid__signal_bridge", [0.42745098, 0.27058824, 0.29803922]),
]
OSDAR23_OBJECT_PALETTE = {x[0]: np.asarray(x[1]) * 255 for x in OSDAR23_OBJECT_PALETTE}


def get_args() -> Namespace:
    """
    Parse given arguments for create_bundle_video function.

    Returns:
        Namespace: parsed arguments
    """
    parser = ArgumentParser()

    parser.add_argument("dataset", help="name of the dataset")
    parser.add_argument("-s", "--source-pred-folder", type=str, required=True)
    parser.add_argument("-b", "--bboxes-path", type=str, required=True)
    parser.add_argument("-l", "--labels-path", type=str, required=True)
    parser.add_argument("-t", "--target-path", type=str, required=True)
    parser.add_argument("-c", "--classes", nargs="+", required=True)
    parser.add_argument("--image-folder", nargs="+", required=True)

    return parser.parse_args()


def create_bundle_video(
    dataset: str,
    classes: List[str],
    source_pred_folder: str,
    bboxes_path: str,
    labels_path: str,
    target_path: str,
    image_folders: List[str],
    width: int = 1920,
    height: int = 1080,
) -> None:
    aspec_ratio = width / height

    if dataset == "osdar23":
        object_palette = OSDAR23_OBJECT_PALETTE
        class_stats = {x: {"count": 0, "distances": []} for x in classes}
        class_mapping = [x for x in classes]

        original_image_width = 4112
        original_image_height = 2504
        original_image_aspec_ratio = original_image_width / original_image_height

        if original_image_aspec_ratio > aspec_ratio:
            max_w = width
            max_h = int(width / original_image_aspec_ratio)
        else:
            max_w = int(height * original_image_aspec_ratio)
            max_h = height

        max_w = int(width // 4)
        max_h = int(height // 2)

        image_paths = {
            os.path.basename(x): sorted(glob(os.path.join(x, "*.jpg")), key=natural_key)
            for x in image_folders
        }
    elif dataset == "tumtraf-i":
        object_palette = TUMTRAF_OBJECT_PALETTE
        class_stats = {x: {"count": 0, "distances": []} for x in classes}
        class_mapping = [x for x in classes]

        original_image_width = 1920
        original_image_height = 1200
        original_image_aspec_ratio = original_image_width / original_image_height

        if original_image_aspec_ratio > aspec_ratio:
            max_w = width
            max_h = int(width / original_image_aspec_ratio)
        else:
            max_w = int(height * original_image_aspec_ratio)
            max_h = height

        max_w = int(width // 3)
        max_h = int(height // 2)

        image_paths = {
            os.path.basename(x): sorted(glob(os.path.join(x, "*.jpg"))) for x in image_folders
        }

    else:
        raise ValueError(f"Dataset {dataset} not supported.")

    bev_preds = sorted(glob(os.path.join(source_pred_folder, "*.jpg")), key=natural_key)
    bev_bboxes = sorted(glob(os.path.join(bboxes_path, "*.npy")), key=natural_key)
    bev_labels = sorted(glob(os.path.join(labels_path, "*.npy")), key=natural_key)

    if len(image_paths) == 0 or len(bev_preds) == 0 or len(bev_bboxes) == 0 or len(bev_labels) == 0:
        raise ValueError("No files found.")

    os.makedirs(os.path.dirname(target_path), exist_ok=True, mode=0o777)

    image_names = list(image_paths.keys())
    frame_n = len(image_paths[list(image_paths.keys())[0]])

    video_name = os.path.join(target_path)
    video = cv2.VideoWriter(video_name, 0x7634706D, 10, (width, height))

    for i in range(frame_n):
        frame = np.zeros((height, width, 3), np.uint8)
        frame_class_stats = copy.deepcopy(class_stats)

        pred_path = bev_preds[i]
        pred = Image.open(pred_path).convert("RGB")
        if dataset == "osdar23":
            pred = pred.rotate(90, Image.NEAREST, expand=1)
            # crop left and right sides of pred
            crop_w = pred.width // 4
            pred = pred.crop((crop_w, 0, pred.width - crop_w, pred.height))
            # add cropped size to height
            pred = pred.resize((max_w, crop_w * 2), Image.BILINEAR)
            pred = np.array(pred)
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            frame[: crop_w * 2, :max_w] = pred
        elif dataset == "tumtraf-i":
            pred = pred.rotate(90, Image.NEAREST, expand=1)
            pred = pred.resize((max_w, max_h), Image.BILINEAR)
            pred = np.array(pred)
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            frame[:max_h, :max_w] = pred

        # read bboxes and labels
        bboxes = np.load(bev_bboxes[i])
        bboxes = LiDARInstance3DBoxes(bboxes, box_dim=7)
        bboxes_center = bboxes.gravity_center
        labels = np.load(bev_labels[i])

        assert bboxes is not None and labels is not None

        for k in range(len(labels)):
            class_idx = labels[k]
            class_name = class_mapping[class_idx]
            frame_class_stats[class_name]["count"] += 1

            distance = np.linalg.norm(bboxes_center[k])
            frame_class_stats[class_name]["distances"].append(distance)

        for j, image_name in enumerate(image_names):
            image_path = image_paths[image_name][i]

            image = Image.open(image_path).convert("RGB")
            image = image.resize((max_w, max_h), Image.BILINEAR)
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if j == 0:
                frame[:max_h, max_w : max_w * 2] = image
            elif j == 1:
                frame[:max_h, max_w * 2 : max_w * 3] = image
            elif j == 2:
                frame[:max_h, max_w * 3 : max_w * 4] = image

        # put text on frame
        data = []
        for k, cls in enumerate(classes):
            count = frame_class_stats[cls]["count"]
            distances = frame_class_stats[cls]["distances"]

            if len(distances) > 0:
                avg_distance = f"{np.mean(distances):.1f} m"
                min_distance = f"{np.min(distances):.1f} m"
                max_distance = f"{np.max(distances):.1f} m"
                data.append([cls, count, avg_distance, min_distance, max_distance])
            else:
                data.append([cls])

        create_cv2_table(dataset, frame, data, max_w, max_h + 50, object_palette=object_palette)
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()


def create_cv2_table(
    dataset,
    frame,
    data,
    x,
    y,
    headers: List[str] = ["class", "count", "avg", "min", "max"],
    color=(255, 255, 255),
    absent_color=(55, 55, 55),
    col_start_coords=[0, 400, 550, 800, 1050],
    row_height=50,
    object_palette=None,
):
    # create a header
    for i, header in enumerate(headers):
        cv2.putText(
            frame,
            header,
            (x + col_start_coords[i], y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )
    for i, row in enumerate(data):
        for j, col in enumerate(row):
            if j == 0:
                if len(row) > 1:
                    col_color = (
                        rgb_to_bgr(object_palette[col]) if object_palette is not None else color
                    )
                else:
                    col_color = absent_color

                if dataset == "osdar23":
                    col = col[15:].replace("_", " ")
                else:
                    col = col.replace("_", " ").lower()

                cv2.putText(
                    frame,
                    col,
                    (x + col_start_coords[j], y + row_height * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    col_color,
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    str(col),
                    (x + col_start_coords[j], y + row_height * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    1,
                )


def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_) if s]


def rgb_to_bgr(color):
    return (color[2], color[1], color[0])


if __name__ == "__main__":
    args = get_args()
    create_bundle_video(
        args.dataset,
        args.classes,
        args.source_pred_folder,
        args.bboxes_path,
        args.labels_path,
        args.target_path,
        args.image_folder,
    )
