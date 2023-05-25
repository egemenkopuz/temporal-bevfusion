import json
import os
from argparse import ArgumentParser, Namespace
from glob import glob
from typing import List, Literal, Union

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from .utils import A9Meta
from .utils.geometry import get_corners


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
    parser.add_argument("--only_camera", type=bool, required=False, default=False)

    return parser.parse_args()


def a9_plot_image_w_lidar_points(
    input_folder_path_images_south1: str,
    input_folder_path_images_south2: str,
    input_folder_path_labels_south1: str,
    input_folder_path_labels_south2: str,
    input_folder_path_point_clouds: str,
    input_folder_path_detections: str,
    index: int = 0,
    lidar_location: Union[Literal["south1"], Literal["south2"]] = "south1",
    only_camera_label: bool = False,
):
    file_paths_images_south1 = glob(os.path.join(input_folder_path_images_south1, "*"))
    file_paths_images_south2 = glob(os.path.join(input_folder_path_images_south2, "*"))
    file_paths_labels_south1 = glob(os.path.join(input_folder_path_labels_south1, "*"))
    file_paths_labels_south2 = glob(os.path.join(input_folder_path_labels_south2, "*"))
    file_paths_point_clouds = glob(os.path.join(input_folder_path_point_clouds, "*"))
    file_paths_detections = glob(os.path.join(input_folder_path_detections, "*"))

    assert (
        index < len(file_paths_images_south1)
        and index < len(file_paths_images_south2)
        and index < len(file_paths_labels_south1)
        and index < len(file_paths_labels_south2)
    )

    if not only_camera_label:
        assert index < len(file_paths_point_clouds) and index < len(file_paths_detections)
        file_path_point_cloud = file_paths_point_clouds[index]
        file_path_detections = file_paths_detections[index]

        pcd = o3d.io.read_point_cloud(file_path_point_cloud)
        pcd_labels = json.load(open(file_path_detections))

    file_path_image_south1 = file_paths_images_south1[index]
    file_path_image_south2 = file_paths_images_south2[index]
    file_path_labels_south1 = file_paths_labels_south1[index]
    file_path_labels_south2 = file_paths_labels_south2[index]

    if lidar_location == "south2":
        img = cv2.imread(file_path_image_south2, cv2.IMREAD_UNCHANGED)
        img_labels = json.load(open(file_path_labels_south2))
    else:
        img = cv2.imread(file_path_image_south1, cv2.IMREAD_UNCHANGED)
        img_labels = json.load(open(file_path_labels_south1))

    if not only_camera_label:
        process_lidar_labels(img, pcd_labels, lidar_location)

    process_image_labels(img, img_labels)

    cv2.imshow("image", img)
    cv2.waitKey()


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


def draw_line(img, start_point, end_point, color, thickness=1):
    cv2.line(img, start_point, end_point, color, thickness)


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


def process_lidar_labels(
    img, label_data, lidar_location: Union[Literal["south1"], Literal["south2"]]
) -> List:
    lidar_bbox = []
    for frame_id, frame_obj in label_data["openlabel"]["frames"].items():
        for id, label in frame_obj["objects"].items():
            cuboid = label["object_data"]["cuboid"]["val"]
            l = float(cuboid[7])
            w = float(cuboid[8])
            h = float(cuboid[9])
            quat_x = float(cuboid[3])
            quat_y = float(cuboid[4])
            quat_z = float(cuboid[5])
            quat_w = float(cuboid[6])
            rotation_yaw = R.from_quat([quat_x, quat_y, quat_z, quat_w]).as_euler(
                "zyx", degrees=False
            )[0]

            position_3d = [
                float(cuboid[0]),
                float(cuboid[1]),
                float(cuboid[2] - h / 2),
            ]

            category = label["object_data"]["type"].upper()
            color = A9Meta.class_colors[category]
            color = (color[2], color[1], color[0])

            points_3d = get_corners(
                [
                    position_3d[0],
                    position_3d[1],
                    position_3d[2],
                    quat_x,
                    quat_y,
                    quat_z,
                    quat_w,
                    l,
                    w,
                    h,
                ]
            )
            points_2d = project_3d_to_2d(points_3d, lidar_location)
            lidar_bbox.append(points_2d)

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

    return lidar_bbox


def project_3d_to_2d(points_3d, lidar_location: Union[Literal["south1"], Literal["south2"]]):
    points_3d = np.transpose(points_3d)
    points_3d = np.append(points_3d, np.ones((1, points_3d.shape[1])), axis=0)

    # project points to 2D
    if lidar_location == "south1":
        points = np.matmul(A9Meta.lidar2s1image, points_3d[:4, :])
    else:
        points = np.matmul(A9Meta.lidar2s2image, points_3d[:4, :])

    edge_points = []
    for i in range(len(points[0, :])):
        if points[2, i] > 0:
            pos_x = int((points[0, i] / points[2, i]))
            pos_y = int((points[1, i] / points[2, i]))
            # if pos_x >= 0 and pos_x < 1920 and pos_y >= 0 and pos_y < 1200:
            if pos_x < A9Meta.image_width and pos_y < A9Meta.image_height:
                edge_points.append((pos_x, pos_y))

    return edge_points


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
        only_camera_label=args.only_camera,
    )
