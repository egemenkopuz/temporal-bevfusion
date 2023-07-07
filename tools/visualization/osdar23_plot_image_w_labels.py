import json
import os
from argparse import ArgumentParser, Namespace
from glob import glob
from typing import List, Literal, Tuple, Union

import cv2
import numpy as np
import open3d as o3d

from .utils import OSDAR23Meta
from .utils.geometry import draw_line, get_corners


def get_args() -> Namespace:
    """
    Parse given arguments for osdar23_plot_image_w_labels function.

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


def osdar23_plot_image_w_labels(
    input_folder_path_images: str,
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
    file_paths_images = sorted(glob(os.path.join(input_folder_path_images, "*")))
    file_path_image = file_paths_images[index]

    img = cv2.imread(file_path_image, cv2.IMREAD_UNCHANGED)
    labels = json.load(open(input_folder_path_detections))

    camera_bbox, lidar_bbox = process_labels(img, labels, camera_location, index, only_camera_label)

    # img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
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
                val = x["val"]
                l = float(val[7])
                w = float(val[8])
                h = float(val[9])
                quat_x = float(val[3])
                quat_y = float(val[4])
                quat_z = float(val[5])
                quat_w = float(val[6])

                position_3d = [
                    float(val[0]),
                    float(val[1]),
                    float(val[2] - h / 2),
                ]

                color = OSDAR23Meta.class_id_colors[x["name"]]
                # swap channels because opencv uses bgr
                color = (255, 255, 255)
                # color = [int(c * 255) for c in color_bgr]

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

                points_2d = project_3d_to_2d(points_3d, camera_location)
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

    return camera_bbox, lidar_bbox


def project_3d_to_2d(
    points_3d,
    camera_location: Union[
        Literal["rgb_center"],
        Literal["rgb_left"],
        Literal["rgb_right"],
        Literal["rgb_highres_center"],
        Literal["rgb_highres_left"],
        Literal["rgb_highres_right"],
    ] = "rgb_center",
):
    points_3d = np.transpose(points_3d)
    points_3d = np.append(points_3d, np.ones((1, points_3d.shape[1])), axis=0)

    # project points to 2D
    points = np.matmul(
        OSDAR23Meta.get_projection_matrix_to_image(camera_location), points_3d[:4, :]
    )

    edge_points = []
    for i in range(len(points[0, :])):
        if points[2, i] > 0:
            pos_x = int((points[0, i] / points[2, i]))
            pos_y = int((points[1, i] / points[2, i]))
            if camera_location[:11] == "rgb_highres":
                if pos_x < OSDAR23Meta.rgb_highres_width and pos_y < OSDAR23Meta.rgb_highres_height:
                    edge_points.append((pos_x, pos_y))
            else:
                if pos_x < OSDAR23Meta.rgb_width and pos_y < OSDAR23Meta.rgb_height:
                    edge_points.append((pos_x, pos_y))

    return edge_points


if __name__ == "__main__":
    args = get_args()
    osdar23_plot_image_w_labels(
        input_folder_path_images=args.images_folder_path,
        input_folder_path_point_clouds=args.input_folder_path_point_clouds,
        input_folder_path_detections=args.input_folder_path_detections,
        index=args.index,
        only_camera_label=args.only_camera,
    )
