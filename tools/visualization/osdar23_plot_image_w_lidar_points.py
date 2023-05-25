import json
import os
from argparse import ArgumentParser, Namespace
from glob import glob
from typing import List, Literal, Tuple, Union

import cv2
import numpy as np
import open3d as o3d

from .utils import OSDAR23Meta


def get_args() -> Namespace:
    """
    Parse given arguments for osdar23_plot_image_w_lidar_points function.

    Returns:
        Namespace: parsed arguments
    """
    parser = ArgumentParser()

    parser.add_argument("-f", "--images_folder_path", type=str, required=True)
    parser.add_argument("-p", "--point_clouds_folder_path", type=str, required=False, default=None)
    parser.add_argument("-d", "--detections_folder_path", type=str, required=True)
    parser.add_argument("-i", "--index", type=int, required=False, default=0)
    parser.add_argument("--only_camera", type=bool, required=False, default=False)

    return parser.parse_args()


def osdar23_plot_image_w_lidar_points(
    input_folder_path_images: str,
    input_folder_path_point_clouds: str,
    input_folder_path_detections: str,
    index: int = 0,
    camera_location: Union[
        Literal["rgb_center"],
        Literal["rgb_left"],
        Literal["rgb_right"],
        Literal["rgb_highres_center"],
        Literal["rgb_highres_left"],
        Literal["rgb_highres_right"],
    ] = "rgb_center",
    only_camera_label: bool = False,
):
    file_paths_images = glob(os.path.join(input_folder_path_images, "*"))
    file_paths_point_clouds = glob(os.path.join(input_folder_path_point_clouds, "*"))

    file_path_image = file_paths_images[index]
    if not only_camera_label:
        file_path_point_cloud = file_paths_point_clouds[index]
        pcd = o3d.io.read_point_cloud(file_path_point_cloud)

    img = cv2.imread(file_path_image, cv2.IMREAD_UNCHANGED)
    labels = json.load(open(input_folder_path_detections))

    camera_bbox, lidar_bbox = process_labels(img, labels, camera_location, index, only_camera_label)

    cv2.imshow("image", img)
    cv2.waitKey()


def process_labels(
    img, label_data, camera_location: str, index: int = 0, only_camera_label: bool = False
) -> Tuple[List, List]:
    camera_bbox = []
    lidar_bbox = []
    relative_index = list(label_data["openlabel"]["frames"].keys())[index]
    frame_obj = label_data["openlabel"]["frames"][relative_index]
    for id, label in frame_obj["objects"].items():
        if not only_camera_label and "cuboid" in label["object_data"]:
            for x in label["object_data"]["cuboid"]:
                color = OSDAR23Meta.class_colors[x["name"]]
                # swap channels because opencv uses bgr
                color_bgr = (color[2], color[1], color[0])
                # TODO lidar to image projection

        if "bbox" in label["object_data"]:
            for x in label["object_data"]["bbox"]:
                if x["coordinate_system"] == camera_location:
                    color = OSDAR23Meta.class_colors[x["name"]]
                    # swap channels because opencv uses bgr
                    color_bgr = (color[2], color[1], color[0])
                    bbox = np.asarray(x["val"], dtype=np.int32)
                    cv2.rectangle(
                        img,
                        (bbox[0] - bbox[2] // 2, bbox[1] - bbox[3] // 2),
                        (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2),
                        [int(c * 255) for c in color_bgr],
                        1,
                    )
                    camera_bbox.append(bbox)

    return camera_bbox, lidar_bbox


if __name__ == "__main__":
    args = get_args()
    osdar23_plot_image_w_lidar_points(
        input_folder_path_images=args.images_folder_path,
        input_folder_path_point_clouds=args.input_folder_path_point_clouds,
        input_folder_path_detections=args.input_folder_path_detections,
        index=args.index,
        only_camera_label=args.only_camera,
    )
