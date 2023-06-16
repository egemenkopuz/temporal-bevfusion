from typing import Literal, Union

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from . import A9Meta, OSDAR23Meta


def draw_line(img, start_point, end_point, color, thickness=1):
    cv2.line(img, start_point, end_point, color, thickness)


def add_open3d_axis(vis):
    """Add a small 3D axis on Open3D Visualizer"""
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )
    axis.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [0, 2], [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )
    vis.add_geometry(axis)


def get_corners(cuboid):
    l = cuboid[7]
    w = cuboid[8]
    h = cuboid[9]
    # Create a bounding box outline
    bounding_box = np.array(
        [
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2],
        ]
    )

    translation = cuboid[:3]
    # Repeat the [x, y, z] eight times
    eight_points = np.tile(translation, (8, 1))

    rotation_quaternion = cuboid[3:7]
    rotation_matrix = R.from_quat(rotation_quaternion).as_matrix()
    # Translate the rotated bounding box by the
    # original center position to obtain the final box
    corner_box = np.dot(rotation_matrix, bounding_box) + eight_points.transpose()
    return corner_box.transpose()


def visualize_bounding_box(
    l,
    w,
    h,
    rotation_yaw,
    position_3d,
    category,
    vis,
    use_two_colors,
    input_type,
    dataset_name: str = Union[Literal["osdar23"], Literal["a9"]],
):
    quaternion = R.from_euler("xyz", [0, 0, rotation_yaw], degrees=False).as_quat()
    corner_box = get_corners(
        [
            position_3d[0],
            position_3d[1],
            position_3d[2],
            quaternion[0],
            quaternion[1],
            quaternion[2],
            quaternion[3],
            l,
            w,
            h,
        ]
    )
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [0, 3],
        [4, 5],
        [5, 6],
        [6, 7],
        [4, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    if use_two_colors and input_type == "detections":
        color_red = (245, 44, 71)
        color_red_normalized = (color_red[0] / 255, color_red[1] / 255, color_red[2] / 255)
        colors = [color_red_normalized for _ in range(len(lines))]
    elif use_two_colors and input_type == "labels":
        color_green = (27, 250, 27)
        color_green_normalized = (color_green[0] / 255, color_green[1] / 255, color_green[2] / 255)
        colors = [color_green_normalized for _ in range(len(lines))]
    else:
        if dataset_name.lower() == "a9":
            colors = [A9Meta.class_id_colors[category] for _ in range(len(lines))]
        elif dataset_name.lower() == "osdar23":
            colors = [OSDAR23Meta.class_id_colors[category] for _ in range(len(lines))]
        else:
            raise ValueError(f"Dataset name {dataset_name} not supported.")

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corner_box)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    # Display the bounding boxes:
    vis.add_geometry(line_set)
