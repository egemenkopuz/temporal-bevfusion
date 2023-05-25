import json
import os
from argparse import ArgumentParser, Namespace
from glob import glob
from typing import List

import matplotlib as mpl
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from .utils.geometry import add_open3d_axis, visualize_bounding_box


def get_args() -> Namespace:
    """
    Parse given arguments for osdar23_plot_3d_boxes function.

    Returns:
        Namespace: parsed arguments
    """
    parser = ArgumentParser()

    parser.add_argument("-p", "--point_clouds_folder_path", type=str, required=True)
    parser.add_argument("-d", "--detections_folder_path", type=str, required=True)
    parser.add_argument("-i", "--index", type=int, required=False, default=0)
    parser.add_argument("--point_size", type=int, required=False, default=2)
    parser.add_argument("--enable_coord_frame", type=bool, required=False, default=False)
    parser.add_argument("--enable_color_distance", type=bool, required=False, default=False)
    parser.add_argument("--enable_color_detection", type=bool, required=False, default=False)

    return parser.parse_args()


def osdar23_plot_3d_boxes(
    input_folder_path_point_clouds: str,
    input_folder_path_detections: str,
    index: int = 0,
    point_size: int = 2,
    show_coordinate_frame: bool = False,
    color_distance: bool = False,
    color_detection: bool = False,
):
    file_paths_point_clouds = glob(os.path.join(input_folder_path_point_clouds, "*"))

    file_path_point_cloud = file_paths_point_clouds[index]

    pcd = o3d.io.read_point_cloud(file_path_point_cloud)
    points = np.array(pcd.points)

    print("Raw: ", points.shape)

    # remove rows having all zeroes
    points_filtered = points[~np.all(points == 0, axis=1)]

    # remove rows having z<-10
    points_filtered = points_filtered[points_filtered[:, 2] > -5.0]

    # remove rows having z>-10
    points_filtered = points_filtered[points_filtered[:, 2] < 10.0]

    # remove points with distance>300
    distances = np.array(
        [np.sqrt(row[0] * row[0] + row[1] * row[1] + row[2] * row[2]) for row in points_filtered]
    )
    distances_bev = np.array(
        [np.sqrt(row[0] * row[0] + row[2] * row[2]) for row in points_filtered]
    )

    points_filtered = points_filtered[distances < 300.0]
    distances_bev = distances_bev[distances < 300.0]
    distances = distances[distances < 300.0]

    # remove points with distance<8
    points_filtered = points_filtered[distances > 8.0]
    distances_bev = distances_bev[distances > 8.0]
    distances = distances[distances > 8.0]

    corner_point_min = np.array([-50, -50, 0])
    corner_point_max = np.array([400, 100, 10])

    print("Filtered: ", points_filtered.shape)

    points = np.vstack((points_filtered, corner_point_min, corner_point_max))
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points[:, :3]))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Visualizer", width=1024, height=720)
    vis.get_render_option().background_color = [0.1, 0.1, 0.1]
    vis.get_render_option().point_size = point_size

    if show_coordinate_frame:
        add_open3d_axis(vis)

    try:
        label_data = json.load(open(input_folder_path_detections))

        if color_distance:
            colors = np.zeros((points.shape[0], 1))

            # distance coloring counter for indexing
            global dic
            dic = 0

            def cd_func(x):
                global dic
                if dic >= len(distances_bev):
                    return [0.4, 0.4, 0.4]

                sdistance = np.interp(distances_bev[dic], (0.0, 100.0), (0, 1))
                dic += 1

                c1 = np.array(mpl.colors.to_rgb([0, 98 / 255, 72 / 255]))
                c2 = np.array(mpl.colors.to_rgb([219 / 255, 219 / 255, 72 / 255]))
                return (1 - sdistance) * c1 + sdistance * c2

            colors = np.apply_along_axis(cd_func, 1, colors).astype(np.float64)[:, 0:3]
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            pcd.paint_uniform_color([0.4, 0.4, 0.4])

        process_detections(label_data, pcd, vis, index, color_detection)

        vis.add_geometry(pcd)

        vis.get_view_control().set_zoom(0.12)
        vis.get_view_control().set_front([0.22, 0.141, 0.965])
        vis.get_view_control().set_lookat([22.964, -2.772, -7.230])
        vis.get_view_control().set_up([0.969, -0.148, -0.200])

        vis.run()
    except Exception as e:
        print(e)
        vis.destroy_window()


def process_detections(label_data, pcd, vis, index: int, color_detection: bool = False):
    relative_index = list(label_data["openlabel"]["frames"].keys())[index]
    frame_obj = label_data["openlabel"]["frames"][relative_index]
    for object_id, label in frame_obj["objects"].items():
        if "cuboid" in label["object_data"]:
            cuboid = label["object_data"]["cuboid"][0]["val"]
            l = float(cuboid[7])
            w = float(cuboid[8])
            h = float(cuboid[9])
            quat_x = float(cuboid[3])
            quat_y = float(cuboid[4])
            quat_z = float(cuboid[5])
            quat_w = float(cuboid[6])
            rotation_yaw = R.from_quat([quat_x, quat_y, quat_z, quat_w]).as_euler(
                "xyz", degrees=False
            )[0]
            position_3d = [
                float(cuboid[0]),
                float(cuboid[1]),
                float(cuboid[2]),
            ]
            category = label["object_data"]["cuboid"][0]["name"]

            obb = o3d.geometry.OrientedBoundingBox(
                position_3d,
                np.array(
                    [
                        [np.cos(rotation_yaw), -np.sin(rotation_yaw), 0],
                        [np.sin(rotation_yaw), np.cos(rotation_yaw), 0],
                        [0, 0, 1],
                    ]
                ),
                np.array([l, w, h]),
            )

            if color_detection:
                colors = np.array(pcd.colors)
                indices = obb.get_point_indices_within_bounding_box(pcd.points)
                base_color = (250, 27, 27)
                colors[indices, :] = np.array(
                    [base_color[0] / 255, base_color[1] / 255, base_color[2] / 255]
                )
                pcd.colors = o3d.utility.Vector3dVector(colors)

            visualize_bounding_box(
                l, w, h, rotation_yaw, position_3d, category, vis, None, None, "osdar23"
            )


if __name__ == "__main__":
    args = get_args()
    osdar23_plot_3d_boxes(
        input_folder_path_point_clouds=args.input_folder_path_point_clouds,
        input_folder_path_detections=args.input_folder_path_detections,
        index=args.index,
        point_size=args.point_size,
        show_coordinate_frame=args.show_coordinate_frame,
        color_distance=args.color_distance,
        color_detection=args.color_detection,
    )
