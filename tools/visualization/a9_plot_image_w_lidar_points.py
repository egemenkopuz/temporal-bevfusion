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

from .utils import A9Meta
from .utils.geometry import draw_line


def get_args() -> Namespace:
    """
    Parse given arguments for a9_plot_image_w_lidar_points function.

    Returns:
        Namespace: parsed arguments
    """
    parser = ArgumentParser()

    parser.add_argument("--images_south1_folder_path", type=str, required=True)
    parser.add_argument("--images_south2_folder_path", type=str, required=True)
    parser.add_argument("--labels_south1_folder_path", type=str, required=True)
    parser.add_argument("--labels_south2_folder_path", type=str, required=True)
    parser.add_argument("-p", "--point_clouds_folder_path", type=str, required=False, default=None)
    parser.add_argument("-d", "--detections_folder_path", type=str, required=False, default=None)
    parser.add_argument("-i", "--index", type=int, required=False, default=0)
    parser.add_argument("--point_size", type=int, required=False, default=2)
    parser.add_argument("--include_camera_label", type=bool, required=False, default=False)

    return parser.parse_args()


def a9_plot_image_w_lidar_points(
    input_folder_path_images_south1: str,
    input_folder_path_images_south2: str,
    input_folder_path_labels_south1: str,
    input_folder_path_labels_south2: str,
    input_folder_path_point_clouds: str,
    index: int = 0,
    camera_location: Union[Literal["south1"], Literal["south2"]] = "south1",
    point_size: int = 2,
    include_camera_label: bool = False,
):
    file_paths_images_south1 = glob(os.path.join(input_folder_path_images_south1, "*"))
    file_paths_images_south2 = glob(os.path.join(input_folder_path_images_south2, "*"))
    file_paths_labels_south1 = glob(os.path.join(input_folder_path_labels_south1, "*"))
    file_paths_labels_south2 = glob(os.path.join(input_folder_path_labels_south2, "*"))
    file_paths_point_clouds = glob(os.path.join(input_folder_path_point_clouds, "*"))

    assert (
        index < len(file_paths_point_clouds)
        and index < len(file_paths_images_south1)
        and index < len(file_paths_images_south2)
        and index < len(file_paths_labels_south1)
        and index < len(file_paths_labels_south2)
    )

    file_path_point_cloud = file_paths_point_clouds[index]
    file_path_image_south1 = file_paths_images_south1[index]
    file_path_image_south2 = file_paths_images_south2[index]
    file_path_labels_south1 = file_paths_labels_south1[index]
    file_path_labels_south2 = file_paths_labels_south2[index]

    pcd = o3d.io.read_point_cloud(file_path_point_cloud)

    if camera_location == "south2":
        img = cv2.imread(file_path_image_south2, cv2.IMREAD_UNCHANGED)
        img_labels = json.load(open(file_path_labels_south2))
    else:
        img = cv2.imread(file_path_image_south1, cv2.IMREAD_UNCHANGED)
        img_labels = json.load(open(file_path_labels_south1))

    if include_camera_label:
        process_image_labels(img, img_labels)

    process_lidar_points(img, pcd, camera_location, point_size)

    cv2.imshow("image", img)
    cv2.waitKey()


def process_lidar_points(
    img,
    point_cloud,
    camera_location: Union[Literal["south1"], Literal["south2"]],
    point_size: int = 2,
) -> List:
    points_3d = np.asarray(point_cloud.points)

    # remove rows having all zeros
    points_3d = points_3d[~np.all(points_3d == 0, axis=1)]

    # crop point cloud to 120 m range
    distances = np.array(
        [np.sqrt(point[0] ** 2 + point[1] ** 2 + point[2] ** 2) for point in points_3d]
    )
    points_3d = points_3d[distances < 120.0]

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
    if camera_location == "south1":
        points = np.matmul(A9Meta.lidar2s1image, points_3d[:4, :])
    else:
        points = np.matmul(A9Meta.lidar2s2image, points_3d[:4, :])

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
            if pos_x >= 0 and pos_x < 1920 and pos_y >= 0 and pos_y < 1200:
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


def process_image_labels(img, label_data) -> List:
    camera_bbox = []
    for frame_id, frame_obj in label_data["openlabel"]["frames"].items():
        for id, label in frame_obj["objects"].items():
            category = label["object_data"]["type"].upper()
            color = A9Meta.class_colors[category]
            # swap channels because opencv uses bgr
            color_bgr = (color[2], color[1], color[0])
            color_bgr = [int(c * 255) for c in color_bgr]

            x = draw_3d_box_camera_labels(img, label, color_bgr)
            camera_bbox.append(x)

    return camera_bbox


def draw_3d_box_camera_labels(img, label, color):
    points2d_val = label["object_data"]["keypoints_2d"]["attributes"]["points2d"]["val"]
    points_2d = []

    for point2d in points2d_val:
        points_2d.append((int(point2d["point2d"]["val"][0]), int(point2d["point2d"]["val"][1])))

    if len(points_2d) == 8:
        draw_line(img, points_2d[0], points_2d[1], color)
        draw_line(img, points_2d[1], points_2d[2], color)
        draw_line(img, points_2d[2], points_2d[3], color)
        draw_line(img, points_2d[3], points_2d[0], color)
        draw_line(img, points_2d[4], points_2d[5], color)
        draw_line(img, points_2d[5], points_2d[6], color)
        draw_line(img, points_2d[6], points_2d[7], color)
        draw_line(img, points_2d[7], points_2d[4], color)
        draw_line(img, points_2d[0], points_2d[4], color)
        draw_line(img, points_2d[1], points_2d[5], color)
        draw_line(img, points_2d[2], points_2d[6], color)
        draw_line(img, points_2d[3], points_2d[7], color)

    return points_2d


if __name__ == "__main__":
    args = get_args()
    a9_plot_image_w_lidar_points(
        input_folder_path_images_south1=args.images_south1_folder_path,
        input_folder_path_images_south2=args.images_south2_folder_path,
        input_folder_path_labels_south1=args.labels_south1_folder_path,
        input_folder_path_labels_south2=args.labels_south2_folder_path,
        input_folder_path_point_clouds=args.input_folder_path_point_clouds,
        input_folder_path_detections=args.input_folder_path_detections,
        index=args.index,
        point_size=args.point_size,
        only_camera_label=args.include_camera_label,
    )
