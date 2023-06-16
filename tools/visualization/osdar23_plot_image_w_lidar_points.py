import json
import os
from argparse import ArgumentParser, Namespace
from glob import glob
from typing import List, Literal, Union

import cv2
import matplotlib as mpl
import numpy as np
import open3d as o3d
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

from .utils import OSDAR23Meta
from .utils.geometry import draw_line


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
    parser.add_argument("--point_size", type=int, required=False, default=2)
    parser.add_argument("--include_camera_label", type=bool, required=False, default=False)

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
    point_size: int = 2,
    include_camera_label: bool = False,
):
    file_paths_images = glob(os.path.join(input_folder_path_images, "*"))
    file_paths_point_clouds = glob(os.path.join(input_folder_path_point_clouds, "*"))

    file_path_image = file_paths_images[index]
    file_path_point_cloud = file_paths_point_clouds[index]

    pcd = o3d.io.read_point_cloud(file_path_point_cloud)
    img = cv2.imread(file_path_image, cv2.IMREAD_UNCHANGED)
    labels = json.load(open(input_folder_path_detections))

    if include_camera_label:
        process_image_labels(img, labels, camera_location, index)

    process_lidar_points(img, pcd, camera_location, point_size)

    # img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
    cv2.imshow("image", img)
    cv2.waitKey()


def process_lidar_points(
    img,
    point_cloud,
    camera_location: Union[
        Literal["rgb_center"],
        Literal["rgb_left"],
        Literal["rgb_right"],
        Literal["rgb_highres_center"],
        Literal["rgb_highres_left"],
        Literal["rgb_highres_right"],
    ] = "rgb_center",
    point_size: int = 2,
) -> List:
    points_3d = np.asarray(point_cloud.points)

    # remove rows having all zeros
    points_3d = points_3d[~np.all(points_3d == 0, axis=1)]

    # crop point cloud to 300 m range
    distances = np.array(
        [np.sqrt(point[0] ** 2 + point[1] ** 2 + point[2] ** 2) for point in points_3d]
    )
    points_3d = points_3d[distances < 300.0]

    points_3d = np.transpose(points_3d)
    points_3d = np.append(points_3d, np.ones((1, points_3d.shape[1])), axis=0)
    distances = []
    indices_to_keep = []
    for i in range(len(points_3d[0, :])):
        point = points_3d[:, i]
        distance = np.sqrt((point[0] ** 2) + (point[1] ** 2) + (point[2] ** 2))
        if distance > 2:
            distances.append(distance)
            indices_to_keep.append(i)

    points_3d = points_3d[:, indices_to_keep]
    print("Raw: ", points_3d.shape[1])

    # project points to 2D
    points = np.matmul(
        OSDAR23Meta.get_projection_matrix_to_image(camera_location), points_3d[:4, :]
    )

    if camera_location[:11] == "rgb_highres":
        max_width = OSDAR23Meta.rgb_highres_width
        max_height = OSDAR23Meta.rgb_highres_height
    else:
        max_width = OSDAR23Meta.rgb_width
        max_height = OSDAR23Meta.rgb_height

    distances_numpy = np.asarray(distances)
    max_distance = max(distances_numpy)
    norm = mpl.colors.Normalize(vmin=70, vmax=250)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    num_points_within_image = 0

    for i in range(len(points[0, :])):
        if points[2, i] > 0:
            pos_x = int(points[0, i] / points[2, i])
            pos_y = int(points[1, i] / points[2, i])
            if pos_x >= 0 and pos_x < max_width and pos_y >= 0 and pos_y < max_height:
                num_points_within_image += 1
                distance_idx = 255 - (int(distances_numpy[i] / max_distance * 255))
                color_rgba = m.to_rgba(distance_idx)
                color_rgb = (
                    color_rgba[0] * 255,
                    color_rgba[1] * 255,
                    color_rgba[2] * 255,
                )
                cv2.circle(img, (pos_x, pos_y), point_size, color_rgb, thickness=-1)
    print("Filtered: ", num_points_within_image)


def process_image_labels(img, label_data, camera_location: str, index: int = 0) -> List:
    camera_bbox = []
    relative_index = list(label_data["openlabel"]["frames"].keys())[index]
    frame_obj = label_data["openlabel"]["frames"][relative_index]
    for id, label in frame_obj["objects"].items():
        if "bbox" in label["object_data"]:
            for x in label["object_data"]["bbox"]:
                if x["coordinate_system"] == camera_location:
                    color = OSDAR23Meta.class_id_colors[x["name"]]
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
    return camera_bbox


if __name__ == "__main__":
    args = get_args()
    osdar23_plot_image_w_lidar_points(
        input_folder_path_images=args.images_folder_path,
        input_folder_path_point_clouds=args.input_folder_path_point_clouds,
        input_folder_path_detections=args.input_folder_path_detections,
        index=args.index,
        point_size=args.point_size,
        include_camera_label=args.include_camera_label,
    )
